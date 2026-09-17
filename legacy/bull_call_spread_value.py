"""For each bull call spread, fetch stock price and sum max value if stock is above both strikes."""

import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_core import active_option_positions, default_csv_path, load_schwab_holdings


def norm_cdf(x):
    x_arr = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x_arr / np.sqrt(2.0)))


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


def identify_bull_call_spreads(df_calls):
    spread_rows = []
    for (underlying, expiry), grp in df_calls.groupby(["Underlying", "Expiration"]):
        longs = []
        shorts = []
        for _, row in grp.sort_values("Strike Price").iterrows():
            qty = float(row["Qty"])
            leg = {
                "strike": float(row["Strike Price"]),
                "qty": abs(qty),
            }
            if qty > 0:
                leg["price"] = float(row["Price Numeric"])
                leg["mv"] = float(row["Market Value Numeric"])
                leg["dc"] = float(row["Day Change Numeric"])
                leg["dte"] = float(row["Days To Expiry"]) if pd.notna(row["Days To Expiry"]) else 1.0
                longs.append(leg)
            elif qty < 0:
                leg["price"] = float(row["Price Numeric"])
                leg["mv"] = float(row["Market Value Numeric"])
                leg["dc"] = float(row["Day Change Numeric"])
                leg["dte"] = float(row["Days To Expiry"]) if pd.notna(row["Days To Expiry"]) else 1.0
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
                long_mv_pro = long_leg["mv"] * matched / long_leg["qty"]
                short_mv_pro = short_leg["mv"] * matched / short_leg["qty"]
                long_dc_pro = long_leg["dc"] * matched / long_leg["qty"]
                short_dc_pro = short_leg["dc"] * matched / short_leg["qty"]
                spread_rows.append({
                    "Underlying": underlying,
                    "Expiration": expiry,
                    "Long Ctr": matched,
                    "Short Ctr": -matched,
                    "Long Strike": long_leg["strike"],
                    "Short Strike": short_leg["strike"],
                    "Width": width,
                    "Long Price": long_leg["price"],
                    "Short Price": short_leg["price"],
                    "Long DTE": long_leg["dte"],
                    "Short DTE": short_leg["dte"],
                    "Long MV": long_mv_pro,
                    "Short MV": short_mv_pro,
                    "Current MV": long_mv_pro + short_mv_pro,
                    "Long DC": long_dc_pro,
                    "Short DC": short_dc_pro,
                    "Net DC": long_dc_pro + short_dc_pro,
                })
                remaining -= matched
                long_leg["qty"] -= matched
    return spread_rows


def fetch_stock_prices(tickers):
    unique = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    prices = {}
    for ticker in unique:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                prices[ticker] = float(hist["Close"].iloc[-1])
        except Exception:
            pass
    return prices


def fetch_stock_day_change(tickers):
    unique = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    changes = {}
    for ticker in unique:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if hist is not None and len(hist) >= 2:
                prev_close = float(hist["Close"].iloc[-2])
                curr_close = float(hist["Close"].iloc[-1])
                changes[ticker] = (curr_close - prev_close) / prev_close
        except Exception:
            pass
    return changes


def main():
    parser = argparse.ArgumentParser(description="Sum bull call spread max values when stock is above both strikes.")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv)")
    parser.add_argument("--as-of", default=None, help="Valuation date YYYY-MM-DD (default: today)")
    parser.add_argument("--ticker", default=None, help="Filter to a specific ticker (e.g. MU)")
    parser.add_argument(
        "--max",
        action="store_true",
        help="Show theoretical max spread value total, optionally filtered by --ticker.",
    )
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)

    df = load_schwab_holdings(csv_path, as_of=as_of)
    options = active_option_positions(df)

    calls = options[options["Opt Type"] == "C"].copy()
    if calls.empty:
        print("No call option positions found.")
        return

    spreads = identify_bull_call_spreads(calls)
    if not spreads:
        print("No bull call spreads identified.")
        return

    df_spreads = pd.DataFrame(spreads)
    df_spreads["Max Spread Value"] = df_spreads["Width"] * df_spreads["Long Ctr"] * 100.0

    if args.ticker:
        args.ticker = args.ticker.strip().upper()
        df_spreads = df_spreads[df_spreads["Underlying"] == args.ticker]
        if df_spreads.empty:
            print(f"No bull call spreads identified for {args.ticker}.")
            return

    if args.max:
        total_max_value = df_spreads["Max Spread Value"].sum()
        label = f" for {args.ticker}" if args.ticker else ""
        print("Bull Call Spread Max Values:")
        print(f"{'Ticker':<8} {'Expiry':<12} {'Spread':<12} {'Ctr':<10} {'Max Value':<12}")
        print("-" * 60)
        for _, row in df_spreads.iterrows():
            spread = f"{row['Long Strike']:.0f}/{row['Short Strike']:.0f}"
            ctr = f"{row['Long Ctr']:.0f}/{abs(row['Short Ctr']):.0f}"
            print(
                f"{row['Underlying']:<8} {row['Expiration']:<12} {spread:<12} {ctr:<10} "
                f"${row['Max Spread Value']:<11,.2f}"
            )
        print()
        print(f"Total Max Value{label}: ${total_max_value:,.2f}")
        return

    tickers = df_spreads["Underlying"].unique()
    stock_prices = fetch_stock_prices(tickers)
    df_spreads["Stock Price"] = df_spreads["Underlying"].map(stock_prices)
    stock_changes = fetch_stock_day_change(tickers)
    df_spreads["Stock Chg %"] = df_spreads["Underlying"].map(stock_changes)

    qualified = df_spreads[df_spreads["Stock Price"].notna() & (df_spreads["Stock Price"] > df_spreads["Long Strike"]) & (df_spreads["Stock Price"] > df_spreads["Short Strike"])]

    if qualified.empty:
        print("No spreads qualify (stock price must be above both strikes).")
        return

    leg_rows = []
    for _, row in qualified.iterrows():
        leg_rows.append({
            "Ticker": row["Underlying"],
            "Expiry": row["Expiration"],
            "Leg": "Long Call",
            "Strike": row["Long Strike"],
            "Ctr": row["Long Ctr"],
            "Stock Price": row["Stock Price"],
            "Market Value": row["Long MV"],
            "Day Change": row["Long DC"],
            "Net DC": row["Net DC"],
        })
        leg_rows.append({
            "Ticker": row["Underlying"],
            "Expiry": row["Expiration"],
            "Leg": "Short Call",
            "Strike": row["Short Strike"],
            "Ctr": row["Short Ctr"],
            "Stock Price": row["Stock Price"],
            "Market Value": row["Short MV"],
            "Day Change": row["Short DC"],
            "Net DC": row["Net DC"],
        })

    df_legs = pd.DataFrame(leg_rows)
    total_mv = df_legs["Market Value"].sum()
    total_dc = df_legs["Day Change"].sum()

    print("\nBull Call Spread Legs (stock > both strikes):")
    print(f"{'Ticker':<8} {'Expiry':<12} {'Leg':<10} {'Strike':<8} {'Ctr':<6} {'Stock':<8} {'Mkt Value':<12} {'Day Chg':<10}")
    print("-" * 76)
    for _, row in df_legs.iterrows():
        sp = row["Stock Price"]
        print(f"{row['Ticker']:<8} {row['Expiry']:<12} {row['Leg']:<10} {row['Strike']:<8.2f} {row['Ctr']:<6.0f} ${sp:<6.2f} ${row['Market Value']:<9,.2f} ${row['Day Change']:<7,.2f}")

    print(f"\nTotal Market Value: ${total_mv:,.2f}")
    print(f"Total Day Change:   ${total_dc:,.2f}")

    anomalies = qualified[qualified["Stock Chg %"].notna() & (qualified["Stock Chg %"] > 0) & (qualified["Net DC"] < 0)]

    print(f"\n--- Day Change Anomaly Detection ---")
    print(f"(Stock went UP today, but short call lost more than long call gained)")
    if anomalies.empty:
        print("No anomalies found.")
    else:
        r = 0.04
        greek_rows = []
        for _, row in anomalies.iterrows():
            sp = row["Stock Price"]
            for leg, strike, ctr, dte in [
                ("Long", row["Long Strike"], row["Long Ctr"], row["Long DTE"]),
                ("Short", row["Short Strike"], abs(row["Short Ctr"]), row["Short DTE"]),
            ]:
                t = max(dte, 1.0) / 365.0
                px = row["Long Price"] if leg == "Long" else row["Short Price"]
                price_ratio = max(px, 0.01) / max(sp, 0.01)
                sigma = np.clip(price_ratio * 4.0, 0.10, 2.0)
                delta = bs_delta_vec(sp, strike, t, r, sigma, np.array(["C"]))
                gamma = bs_gamma_vec(sp, strike, t, r, sigma)
                vega = bs_vega_vec(sp, strike, t, r, sigma)
                d_val = float(delta.flat[0])
                g_val = float(gamma.flat[0])
                v_val = float(vega.flat[0])
                mult = -1.0 if leg == "Short" else 1.0
                greek_rows.append({
                    "Ticker": row["Underlying"],
                    "Expiry": row["Expiration"],
                    "Leg": leg,
                    "Delta $": mult * d_val * ctr * 100.0,
                    "Gamma $/1%": mult * g_val * ctr * 100.0 * (sp * 0.01) ** 2,
                    "Vega $/1vol": mult * v_val * ctr * 100.0 * 0.01,
                })

        df_greeks = pd.DataFrame(greek_rows)
        net = df_greeks.groupby(["Ticker", "Expiry"])[["Delta $", "Gamma $/1%", "Vega $/1vol"]].sum().reset_index()

        print(f"\nAnomalous spreads: {len(anomalies)}")
        print(f"\n{'Ticker':<10} {'Expiry':<12} {'Stock Chg':<10} {'Delta $':<14} {'Gamma $/1%':<14} {'Vega $/1vol':<14} {'Net DC':<12}")
        print("-" * 80)
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        for _, row in anomalies.iterrows():
            g = net[(net["Ticker"] == row["Underlying"]) & (net["Expiry"] == row["Expiration"])]
            if not g.empty:
                d = float(g["Delta $"].iloc[0])
                gm = float(g["Gamma $/1%"].iloc[0])
                v = float(g["Vega $/1vol"].iloc[0])
            else:
                d = gm = v = 0.0
            chg = row["Stock Chg %"]
            total_delta += d
            total_gamma += gm
            total_vega += v
            print(f"{row['Underlying']:<10} {row['Expiration']:<12} {chg*100:>+7.2f}%  ${d:<10,.2f} ${gm:<10,.2f} ${v:<10,.2f} ${row['Net DC']:<9,.2f}")

        print(f"\nTotals:")
        print(f"  Net Delta $:              ${total_delta:>12,.2f}")
        print(f"  Net Gamma $/1%:           ${total_gamma:>12,.2f}")
        print(f"  Net Vega $/1vol:          ${total_vega:>12,.2f}")

        total_delta_pl = 0.0
        total_gamma_pl = 0.0
        for _, row in anomalies.iterrows():
            sp_chg = row["Stock Chg %"]
            g = net[(net["Ticker"] == row["Underlying"]) & (net["Expiry"] == row["Expiration"])]
            if not g.empty:
                d = float(g["Delta $"].iloc[0])
                gm = float(g["Gamma $/1%"].iloc[0])
                total_delta_pl += d * sp_chg
                total_gamma_pl += gm * (sp_chg / 0.01) ** 2

        total_net_dc = anomalies["Net DC"].sum()
        residual = total_net_dc - total_delta_pl - total_gamma_pl

        print(f"\n  Delta P&L (actual move):  ${total_delta_pl:>12,.2f}")
        print(f"  Gamma P&L (actual move):  ${total_gamma_pl:>12,.2f}")
        print(f"  Delta+Gamma P&L:          ${total_delta_pl + total_gamma_pl:>12,.2f}")
        print(f"  Actual Net DC Total:      ${total_net_dc:>12,.2f}")
        print(f"  Residual (vega+theta+):   ${residual:>12,.2f}")


    if args.ticker:
        r = 0.04
        print(f"\n--- Vega Decomposition for {args.ticker} ---")
        print(f"(Shows vega sensitivity per spread; apply your own IV change estimate)")
        print(f"(Deep ITM options: ~99% intrinsic, IV changes cannot be reliably estimated)")
        dc_rows = []
        for _, row in df_spreads.iterrows():
            sp = row["Stock Price"]
            if sp is None or pd.isna(sp):
                continue
            net_vega = 0.0
            net_dc = row["Net DC"]
            for leg, strike, ctr, dte, price, dc in [
                ("Long", row["Long Strike"], row["Long Ctr"], row["Long DTE"], row["Long Price"], row["Long DC"]),
                ("Short", row["Short Strike"], abs(row["Short Ctr"]), row["Short DTE"], row["Short Price"], row["Short DC"]),
            ]:
                t = max(dte, 1.0) / 365.0
                px = float(price)
                mult = -1.0 if leg == "Short" else 1.0
                sigma = np.clip(max(px, 0.01) / max(sp, 0.01) * 4.0, 0.10, 2.0)
                v_val = float(bs_vega_vec(sp, strike, t, r, sigma).flat[0])
                leg_vega = mult * v_val * ctr * 100.0 * 0.01
                net_vega += leg_vega

            vega_1vol = net_vega
            vega_2vol = net_vega * 2.0
            dc_rows.append({
                "Expiry": row["Expiration"],
                "Spread": f"{row['Long Strike']:.0f}/{row['Short Strike']:.0f}",
                "Ctr": f"{row['Long Ctr']:.0f}/{abs(row['Short Ctr']):.0f}",
                "Long DC": row["Long DC"],
                "Short DC": row["Short DC"],
                "Net DC": net_dc,
                "Vega $/1vol": net_vega,
                "Vega @ +1vol": vega_1vol,
                "Vega @ +2vol": vega_2vol,
            })

        df_dc = pd.DataFrame(dc_rows)
        print(f"\n{'Expiry':<12} {'Spread':<10} {'Ctr':<8} {'Long DC':<12} {'Short DC':<13} {'Net DC':<10} {'Vega$/vol':<11} {'P&L@+1vol':<12} {'P&L@+2vol':<12}")
        print("-" * 102)
        total_net = 0.0
        total_vega = 0.0
        for _, rw in df_dc.iterrows():
            total_net += rw["Net DC"]
            total_vega += rw["Vega $/1vol"]
            print(f"{rw['Expiry']:<12} {rw['Spread']:<10} {rw['Ctr']:<8} ${rw['Long DC']:<8,.2f} ${rw['Short DC']:<9,.2f} ${rw['Net DC']:<6,.2f} ${rw['Vega $/1vol']:<7,.2f} ${rw['Vega @ +1vol']:<8,.2f} ${rw['Vega @ +2vol']:<8,.2f}")

        print("-" * 102)
        print(f"{'TOTAL':<12} {'':<10} {'':<8} ${'':<8} ${'':<9} ${total_net:<6,.2f} ${total_vega:<7,.2f}")


if __name__ == "__main__":
    main()
