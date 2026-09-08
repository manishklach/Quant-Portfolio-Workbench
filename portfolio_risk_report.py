"""Generate a consolidated portfolio summary with approximate option Greeks and stress scenarios."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_core import (
    active_option_positions,
    default_csv_path,
    estimate_equity_equivalent_exposure,
    load_schwab_holdings,
)


LIVE_QUOTE_WORKERS = 8
LIVE_EQUITY_ASSET_TYPES = {"Equity", "ETFs & Closed End Funds", "Mutual Funds"}


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(__import__("math").erf)(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def bs_d1(spot, strike, t, r, sigma):
    spot = np.maximum(np.asarray(spot, dtype=float), 1e-9)
    strike = np.maximum(np.asarray(strike, dtype=float), 1e-9)
    t = np.maximum(np.asarray(t, dtype=float), 1e-9)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    return (np.log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def bs_delta_vec(spot, strike, t, r, sigma, opt_type):
    d1 = bs_d1(spot, strike, t, r, sigma)
    call_delta = norm_cdf(d1)
    put_delta = call_delta - 1.0
    return np.where(np.asarray(opt_type) == "C", call_delta, put_delta)


def bs_gamma_vec(spot, strike, t, r, sigma):
    d1 = bs_d1(spot, strike, t, r, sigma)
    pdf_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1**2)
    return pdf_d1 / (np.maximum(np.asarray(spot, dtype=float), 1e-9) * np.maximum(np.asarray(sigma, dtype=float), 1e-6) * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9)))


def bs_vega_vec(spot, strike, t, r, sigma):
    d1 = bs_d1(spot, strike, t, r, sigma)
    pdf_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1**2)
    return np.asarray(spot, dtype=float) * pdf_d1 * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9))


def bs_theta_vec(spot, strike, t, r, sigma, opt_type):
    d1 = bs_d1(spot, strike, t, r, sigma)
    d2 = d1 - np.maximum(np.asarray(sigma, dtype=float), 1e-6) * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9))
    pdf_d1 = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1**2)
    carry = r * np.asarray(strike, dtype=float) * np.exp(-r * np.maximum(np.asarray(t, dtype=float), 1e-9))
    call_theta = (
        -np.asarray(spot, dtype=float) * pdf_d1 * np.asarray(sigma, dtype=float) / (2.0 * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9)))
        - carry * norm_cdf(d2)
    )
    put_theta = (
        -np.asarray(spot, dtype=float) * pdf_d1 * np.asarray(sigma, dtype=float) / (2.0 * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9)))
        + carry * norm_cdf(-d2)
    )
    return np.where(np.asarray(opt_type) == "C", call_theta, put_theta)


def build_option_risk_frame(
    df: pd.DataFrame,
    risk_free_rate: float,
    underlying_prices: dict[str, float] | None = None,
) -> pd.DataFrame:
    options = active_option_positions(df)
    if options.empty:
        return options

    options = options.copy()
    spot_fallback = np.where(
        options["Strike Price"].fillna(0.0) > 0,
        options["Strike Price"].astype(float),
        np.maximum(options["Price Numeric"].astype(float) * 100.0, 1.0),
    )
    options["Spot Proxy"] = options["Underlying"].map(underlying_prices or {}).fillna(
        pd.Series(spot_fallback, index=options.index)
    )
    options["T"] = np.maximum(options["Days To Expiry"].fillna(1).astype(float), 1.0) / 365.0
    price_ratio = options["Price Numeric"].astype(float) / np.maximum(options["Spot Proxy"].astype(float), 1.0)
    options["IV Proxy"] = np.clip(
        np.where(options["Price Numeric"].astype(float) > 0, price_ratio * 4.0, 0.55),
        0.10,
        2.0,
    )

    spot = options["Spot Proxy"].to_numpy(dtype=float)
    strike = options["Strike Price"].to_numpy(dtype=float)
    t = options["T"].to_numpy(dtype=float)
    sigma = options["IV Proxy"].to_numpy(dtype=float)
    opt_type = options["Opt Type"].to_numpy(dtype=str)

    options["Delta"] = bs_delta_vec(spot, strike, t, risk_free_rate, sigma, opt_type)
    options["Gamma"] = bs_gamma_vec(spot, strike, t, risk_free_rate, sigma)
    options["Vega"] = bs_vega_vec(spot, strike, t, risk_free_rate, sigma)
    options["Theta"] = bs_theta_vec(spot, strike, t, risk_free_rate, sigma, opt_type)
    options["Delta Dollars"] = options["Qty"] * options["Multiplier"] * options["Delta"] * options["Spot Proxy"]
    options["Gamma Dollars 1%"] = options["Qty"] * options["Multiplier"] * options["Gamma"] * (options["Spot Proxy"] * 0.01) ** 2
    options["Vega Dollars 1 Vol"] = options["Qty"] * options["Multiplier"] * options["Vega"] * 0.01
    options["Theta Dollars 1 Day"] = options["Qty"] * options["Multiplier"] * options["Theta"] / 365.0
    return options


def fetch_live_equity_quote(ticker: str) -> float | None:
    """Fetch the latest usable Yahoo quote for a listed equity or ETF."""
    try:
        instrument = yf.Ticker(ticker)
        fast = dict(instrument.fast_info)
        for key in ("last_price", "lastPrice"):
            value = fast.get(key)
            if value is not None and not pd.isna(value) and float(value) > 0:
                return float(value)

        history = instrument.history(period="1d", interval="1m", auto_adjust=False)
        if not history.empty and "Close" in history:
            closes = history["Close"].dropna()
            if not closes.empty and float(closes.iloc[-1]) > 0:
                return float(closes.iloc[-1])
    except Exception:
        pass
    return None


def fetch_option_chain_marks(
    underlying: str,
    expiration: str,
    option_rows: pd.DataFrame,
) -> dict[tuple[str, str, str, float], tuple[float, str]]:
    """Fetch midpoint/last marks for all held strikes in a single option chain."""
    try:
        expiry_iso = datetime.strptime(expiration, "%m/%d/%Y").strftime("%Y-%m-%d")
        instrument = yf.Ticker(underlying)
        expirations = list(instrument.options or [])
        if expiry_iso not in expirations:
            return {}

        chain = instrument.option_chain(expiry_iso)
    except Exception:
        return {}

    marks: dict[tuple[str, str, str, float], tuple[float, str]] = {}
    for option_type, chain_rows in (("C", chain.calls), ("P", chain.puts)):
        if chain_rows is None or chain_rows.empty:
            continue
        held_strikes = option_rows.loc[
            option_rows["Opt Type"] == option_type, "Strike Price"
        ].dropna().astype(float).unique()
        for strike in held_strikes:
            candidates = chain_rows[np.isclose(chain_rows["strike"].astype(float), strike, atol=0.01)]
            if candidates.empty:
                continue
            quote = candidates.iloc[0]
            bid = pd.to_numeric(quote.get("bid"), errors="coerce")
            ask = pd.to_numeric(quote.get("ask"), errors="coerce")
            last = pd.to_numeric(quote.get("lastPrice"), errors="coerce")
            if pd.notna(bid) and pd.notna(ask) and bid >= 0 and ask >= bid and ask > 0:
                marks[(underlying, expiration, option_type, float(strike))] = (
                    float((bid + ask) / 2.0),
                    "option_chain_mid",
                )
            elif pd.notna(last) and last > 0:
                marks[(underlying, expiration, option_type, float(strike))] = (
                    float(last),
                    "option_chain_last",
                )
    return marks


def refresh_live_market_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a valuation copy with Yahoo equity and option-chain marks where available."""
    out = df.copy()
    out["Live Price"] = out["Price Numeric"]
    out["Live Price Source"] = "csv_snapshot"
    out["Live Marked"] = False

    active_options = active_option_positions(out)

    equity_rows = out[
        (~out["Is Option"])
        & out["Asset Type Normalized"].isin(LIVE_EQUITY_ASSET_TYPES)
        & out["Symbol"].astype(str).str.upper().str.match(r"^[A-Z][A-Z0-9.\-]{0,9}$", na=False)
        & (out["Qty"] != 0)
    ]
    equity_tickers = sorted(equity_rows["Symbol"].astype(str).str.upper().unique())
    quote_tickers = sorted(
        set(equity_tickers)
        | set(active_options["Underlying"].astype(str).str.upper().unique())
    )
    live_quotes: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=LIVE_QUOTE_WORKERS) as executor:
        pending = {executor.submit(fetch_live_equity_quote, ticker): ticker for ticker in quote_tickers}
        for future in as_completed(pending):
            try:
                price = future.result()
            except Exception:
                price = None
            if price is not None:
                live_quotes[pending[future]] = price

    for index, row in equity_rows.iterrows():
        ticker = str(row["Symbol"]).upper()
        price = live_quotes.get(ticker)
        if price is None:
            continue
        out.at[index, "Live Price"] = price
        out.at[index, "Price Numeric"] = price
        out.at[index, "Market Value Numeric"] = float(row["Qty"]) * price
        out.at[index, "Live Price Source"] = "yahoo_live"
        out.at[index, "Live Marked"] = True

    option_marks: dict[tuple[str, str, str, float], tuple[float, str]] = {}
    option_groups = list(active_options.groupby(["Underlying", "Expiration"], sort=False))
    with ThreadPoolExecutor(max_workers=LIVE_QUOTE_WORKERS) as executor:
        pending = {
            executor.submit(fetch_option_chain_marks, str(underlying), str(expiration), group): (underlying, expiration)
            for (underlying, expiration), group in option_groups
        }
        for future in as_completed(pending):
            try:
                option_marks.update(future.result())
            except Exception:
                continue

    for index, row in active_options.iterrows():
        key = (
            str(row["Underlying"]),
            str(row["Expiration"]),
            str(row["Opt Type"]),
            float(row["Strike Price"]),
        )
        mark = option_marks.get(key)
        if mark is None:
            continue
        price, source = mark
        out.at[index, "Live Price"] = price
        out.at[index, "Price Numeric"] = price
        out.at[index, "Market Value Numeric"] = float(row["Qty"]) * float(row["Multiplier"]) * price
        out.at[index, "Live Price Source"] = source
        out.at[index, "Live Marked"] = True

    metadata = {
        "equity_quotes": sum(ticker in live_quotes for ticker in equity_tickers),
        "equity_total": len(equity_tickers),
        "option_marks": int(out.loc[out["Is Option"], "Live Marked"].sum()),
        "option_total": len(active_options),
        "underlying_prices": live_quotes,
    }
    return out, metadata


def summarize_call_debit_spreads(options: pd.DataFrame) -> pd.DataFrame:
    spread_rows = []
    calls = options[options["Opt Type"] == "C"].copy()

    for (underlying, expiry), grp in calls.groupby(["Underlying", "Expiration"]):
        longs = []
        shorts = []
        for _, row in grp.sort_values("Strike Price").iterrows():
            qty = float(row["Qty"])
            leg = {
                "strike": float(row["Strike Price"]),
                "qty": abs(qty),
                "price": float(row["Price Numeric"]),
                "delta": float(row["Delta Dollars"]),
            }
            if qty > 0:
                longs.append(leg)
            elif qty < 0:
                shorts.append(leg)

        for short_leg in sorted(shorts, key=lambda leg: leg["strike"]):
            remaining = short_leg["qty"]
            candidates = [leg for leg in longs if leg["qty"] > 0 and leg["strike"] < short_leg["strike"]]
            candidates.sort(key=lambda leg: leg["strike"], reverse=True)

            for long_leg in candidates:
                if remaining <= 0:
                    break
                matched = min(remaining, long_leg["qty"])
                width = short_leg["strike"] - long_leg["strike"]
                spread_rows.append(
                    {
                        "Underlying": underlying,
                        "Expiration": expiry,
                        "Contracts": matched,
                        "Long Strike": long_leg["strike"],
                        "Short Strike": short_leg["strike"],
                        "Width": width,
                        "Max Spread Value": matched * 100.0 * width,
                        "Net Delta Dollars": matched
                        * (
                            long_leg["delta"] / max(long_leg["qty"], 1.0)
                            + short_leg["delta"] / max(short_leg["qty"], 1.0)
                        ),
                    }
                )
                remaining -= matched
                long_leg["qty"] -= matched

    if not spread_rows:
        return pd.DataFrame(
            columns=[
                "Underlying",
                "Contracts",
                "Max Spread Value",
                "Net Delta Dollars",
            ]
        )

    return (
        pd.DataFrame(spread_rows)
        .groupby("Underlying", as_index=False)[["Contracts", "Max Spread Value", "Net Delta Dollars"]]
        .sum()
        .sort_values("Max Spread Value", ascending=False)
    )


def summarize_uncovered_short_puts(options: pd.DataFrame) -> pd.DataFrame:
    puts = options[options["Opt Type"] == "P"].copy()
    uncovered_rows = []

    for (underlying, expiry), grp in puts.groupby(["Underlying", "Expiration"]):
        longs = []
        shorts = []
        for _, row in grp.sort_values("Strike Price", ascending=False).iterrows():
            qty = float(row["Qty"])
            if qty > 0:
                longs.append({"strike": float(row["Strike Price"]), "qty": qty})
            elif qty < 0:
                shorts.append(row)

        for short_row in shorts:
            remaining = abs(float(short_row["Qty"]))
            short_strike = float(short_row["Strike Price"])
            candidates = [leg for leg in longs if leg["qty"] > 0 and leg["strike"] < short_strike]
            candidates.sort(key=lambda leg: leg["strike"], reverse=True)

            for long_leg in candidates:
                if remaining <= 0:
                    break
                matched = min(remaining, long_leg["qty"])
                remaining -= matched
                long_leg["qty"] -= matched

            if remaining > 0:
                uncovered_rows.append(
                    {
                        "Underlying": underlying,
                        "Contracts": remaining,
                        "Assignment Notional": remaining * 100.0 * short_strike,
                    }
                )

    if not uncovered_rows:
        return pd.DataFrame(columns=["Underlying", "Contracts", "Assignment Notional"])

    return (
        pd.DataFrame(uncovered_rows)
        .groupby("Underlying", as_index=False)[["Contracts", "Assignment Notional"]]
        .sum()
        .sort_values("Assignment Notional", ascending=False)
    )


def print_section(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def print_scenario_line(label: str, pnl: float, nav: float):
    pct_nav = (pnl / nav * 100.0) if nav else 0.0
    print(f"{label}: ${pnl:,.2f} ({pct_nav:+.2f}% NAV)")


def load_vix_context(as_of: datetime.date) -> dict | None:
    end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    start = pd.Timestamp(as_of) - pd.Timedelta(days=370)

    try:
        history = yf.download("^VIX", start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
    except Exception:
        return None

    if history is None or history.empty or "Close" not in history.columns:
        return None

    close = history["Close"].dropna()
    if isinstance(close, pd.DataFrame):
        if close.empty:
            return None
        close = close.iloc[:, 0].dropna()
    if close.empty:
        return None

    latest_date = close.index[-1].date()
    latest_value = float(close.iloc[-1])
    trailing_year = close.tail(min(len(close), 252))
    trailing_avg = float(trailing_year.mean())
    abs_diff = latest_value - trailing_avg
    pct_diff = (abs_diff / trailing_avg * 100.0) if trailing_avg else 0.0

    return {
        "latest_date": latest_date,
        "latest_value": latest_value,
        "trailing_avg": trailing_avg,
        "abs_diff": abs_diff,
        "pct_diff": pct_diff,
        "points_above_avg": abs_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Portfolio summary plus approximate option risk and stress report.")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv next to script)")
    parser.add_argument("--as-of", default=None, help="Valuation date YYYY-MM-DD (default: today in local system date)")
    parser.add_argument("--r", type=float, default=0.04, help="Risk-free rate as decimal (default: 0.04)")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Refresh listed securities and active option marks from Yahoo without modifying the CSV.",
    )
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)
    df = load_schwab_holdings(csv_path, as_of=as_of)
    csv_snapshot_nav = df["Market Value Numeric"].sum()
    live_metadata: dict[str, object] | None = None
    if args.live:
        df, live_metadata = refresh_live_market_values(df)
    df["Equity Equivalent Exposure"] = estimate_equity_equivalent_exposure(df)
    options = build_option_risk_frame(
        df,
        args.r,
        live_metadata["underlying_prices"] if live_metadata else None,
    )
    call_spreads = summarize_call_debit_spreads(options) if not options.empty else pd.DataFrame()
    uncovered_puts = summarize_uncovered_short_puts(options) if not options.empty else pd.DataFrame()
    vix_context = load_vix_context(as_of)

    nav = df["Market Value Numeric"].sum()
    long_mv = df.loc[df["Market Value Numeric"] > 0, "Market Value Numeric"].sum()
    short_mv = df.loc[df["Market Value Numeric"] < 0, "Market Value Numeric"].sum()
    day_change = df["Day Change Numeric"].sum()
    realized_unknown = int((df["Market Value Numeric"] == 0).sum())
    expired_count = int(df["Is Expired"].sum())

    print("Portfolio Risk Report")
    print(f"File: {csv_path}")
    print(f"As-Of Date: {as_of}")
    if live_metadata:
        print("Valuation: estimated live Yahoo marks; the holdings CSV is unchanged")

    print_section("Topline")
    print(f"NAV / Market Value: ${nav:,.2f}")
    if live_metadata:
        print(f"Estimated Change Since CSV Mark: ${nav - csv_snapshot_nav:+,.2f}")
        print(
            "Live Quote Coverage: "
            f"{live_metadata['equity_quotes']}/{live_metadata['equity_total']} listed securities | "
            f"{live_metadata['option_marks']}/{live_metadata['option_total']} active option legs"
        )
    print(f"Gross Long Market Value: ${long_mv:,.2f}")
    print(f"Gross Short Market Value: ${abs(short_mv):,.2f}")
    print(f"Net Day Change: ${day_change:,.2f}")
    print(f"Expired Positions Excluded From Option Risk: {expired_count}")
    print(f"Zero-Market-Value Rows: {realized_unknown}")

    print_section("Asset Mix")
    asset_mix = (
        df.groupby("Asset Type Normalized", dropna=False)["Market Value Numeric"]
        .sum()
        .sort_values(ascending=False)
    )
    for asset_type, value in asset_mix.items():
        label = asset_type if str(asset_type).strip() else "Unknown"
        print(f"{label}: ${value:,.2f}")

    print_section("VIX Context")
    if vix_context is None:
        print("VIX history unavailable.")
    else:
        print(f"Latest available VIX close: {vix_context['latest_value']:.2f} on {vix_context['latest_date']}")
        print(f"Trailing 1-year average VIX close: {vix_context['trailing_avg']:.2f}")
        print(
            f"Above/below 1-year average: {vix_context['abs_diff']:+.2f} pts "
            f"({vix_context['pct_diff']:+.2f}%)"
        )

    print_section("Net Delta Exposure By Underlying")
    if options.empty:
        print("No active option positions found.")
    else:
        delta_exposure = (
            options.groupby("Underlying")["Delta Dollars"]
            .sum()
            .sort_values(key=lambda s: s.abs(), ascending=False)
            .head(15)
        )
        for underlying, value in delta_exposure.items():
            print(f"{underlying}: ${value:,.2f}")

    print_section("Bullish Call Debit Spreads")
    if call_spreads.empty:
        print("No call debit spreads recognized.")
    else:
        for _, row in call_spreads.head(15).iterrows():
            print(
                f"{row['Underlying']}: {row['Contracts']:g} contracts | "
                f"max spread value ${row['Max Spread Value']:,.2f} | "
                f"net delta ${row['Net Delta Dollars']:,.2f}"
            )

    print_section("Uncovered Short Puts")
    if uncovered_puts.empty:
        print("No uncovered short puts recognized.")
    else:
        for _, row in uncovered_puts.head(15).iterrows():
            print(
                f"{row['Underlying']}: {row['Contracts']:g} contracts | "
                f"assignment notional ${row['Assignment Notional']:,.2f}"
            )

    print_section("Option Risk")
    if options.empty:
        print("No active option positions found.")
    else:
        print(f"Net Delta Dollars: ${options['Delta Dollars'].sum():,.2f}")
        print(f"Net Gamma Dollars (1% move): ${options['Gamma Dollars 1%'].sum():,.2f}")
        print(f"Net Vega Dollars (per 1 vol point): ${options['Vega Dollars 1 Vol'].sum():,.2f}")
        print(f"Net Theta Dollars (per day): ${options['Theta Dollars 1 Day'].sum():,.2f}")

        by_underlying = (
            options.groupby("Underlying")[["Delta Dollars", "Gamma Dollars 1%","Vega Dollars 1 Vol", "Theta Dollars 1 Day"]]
            .sum()
            .sort_values(by="Delta Dollars", key=lambda s: s.abs(), ascending=False)
            .head(15)
        )
        for underlying, row in by_underlying.iterrows():
            print(
                f"{underlying}: delta ${row['Delta Dollars']:,.2f} | "
                f"gamma1% ${row['Gamma Dollars 1%']:,.2f} | "
                f"vega ${row['Vega Dollars 1 Vol']:,.2f} | "
                f"theta/day ${row['Theta Dollars 1 Day']:,.2f}"
            )

    print_section("IV-Only Shock Ladder")
    if options.empty:
        print("No active option positions found.")
    else:
        for vol_points in (-20, -10, -5, 5, 10, 20):
            pnl = options["Vega Dollars 1 Vol"].sum() * vol_points
            print_scenario_line(f"IV {vol_points:+} pts", pnl, nav)

    print_section("Objective Attribution")
    if options.empty:
        print("No active option positions found.")
    else:
        spot_only_1pct = (options["Delta Dollars"] * 0.01 + options["Gamma Dollars 1%"]).sum()
        iv_only_1pt = options["Vega Dollars 1 Vol"].sum()
        theta_1day = options["Theta Dollars 1 Day"].sum()
        explained_1day = spot_only_1pct + iv_only_1pt + theta_1day
        print_scenario_line("Spot only (+1%)", spot_only_1pct, nav)
        print_scenario_line("IV only (+1 vol pt)", iv_only_1pt, nav)
        print_scenario_line("Theta only (1 day)", theta_1day, nav)
        print_scenario_line("Explained P/L for +1% spot, +1 vol pt, +1 day", explained_1day, nav)

    print_section("Combined Stress Scenarios")
    if options.empty:
        print("No active option positions found.")
    else:
        stress_rows = [
            ("Underlying -5%, IV +10pts", -0.05, 10),
            ("Underlying -10%, IV +20pts", -0.10, 20),
            ("Underlying +5%, IV -10pts", 0.05, -10),
            ("Underlying +10%, IV -20pts", 0.10, -20),
        ]
        for label, spot_move, vol_points in stress_rows:
            pnl = (
                options["Delta Dollars"] * spot_move
                + options["Gamma Dollars 1%"] * ((spot_move / 0.01) ** 2)
                + options["Vega Dollars 1 Vol"] * vol_points
            ).sum()
            print_scenario_line(label, pnl, nav)


if __name__ == "__main__":
    main()
