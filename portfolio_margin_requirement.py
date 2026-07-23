#!/usr/bin/env python3
"""
Estimate portfolio margin/collateral requirement from a Schwab holdings CSV.

This is a transparent approximation tool, not a broker-exact house margin engine.

Default behavior:
  - long cash / money market / fixed-income positions: 0 incremental requirement
  - long equities / ETFs: 0 incremental requirement unless --include-long-regt is used
  - long options: 0 incremental requirement
  - covered calls: 0 incremental requirement
  - naked short puts/calls: Reg-T style estimate using current underlying price
  - defined-risk short verticals: width * contracts * 100 (conservative max-loss style)
  - short stock: 150% of short market value

Examples:
  python portfolio_margin_requirement.py
  python portfolio_margin_requirement.py --file my_holdings.csv
  python portfolio_margin_requirement.py --ticker SOXL
  python portfolio_margin_requirement.py --include-long-regt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from portfolio_core import active_option_positions, clean_numeric, default_csv_path, load_schwab_holdings

try:
    import yfinance as yf
except ImportError:
    yf = None


CASH_LIKE_ASSET_TYPES = {"Cash and Money Market"}
FIXED_INCOME_ASSET_TYPES = {"Fixed Income"}
EQUITY_LIKE_ASSET_TYPES = {"Equity", "ETFs & Closed End Funds", "Mutual Funds"}


def fetch_underlying_prices(tickers):
    unique = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    if not unique:
        return {}
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    prices = {}
    for ticker in unique:
        try:
            tk = yf.Ticker(ticker)
            fast = {}
            try:
                fast = dict(tk.fast_info)
            except Exception:
                pass

            last = fast.get("last_price")
            if last is None or pd.isna(last):
                last = fast.get("lastPrice")
            if last is not None and not pd.isna(last):
                prices[ticker] = float(last)
                continue

            hist = tk.history(period="5d", interval="1d", auto_adjust=False)
            closes = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
            if not closes.empty:
                prices[ticker] = float(closes.iloc[-1])
        except Exception:
            continue
    return prices


def regt_short_put_requirement(spot, strike, option_price, contracts):
    premium = option_price * 100.0
    otm_amount = max(spot - strike, 0.0) * 100.0
    per_contract = premium + max(0.20 * spot * 100.0 - otm_amount, 0.10 * strike * 100.0)
    return max(per_contract, premium) * contracts


def regt_short_call_requirement(spot, strike, option_price, contracts):
    premium = option_price * 100.0
    otm_amount = max(strike - spot, 0.0) * 100.0
    per_contract = premium + max(0.20 * spot * 100.0 - otm_amount, 0.10 * spot * 100.0)
    return max(per_contract, premium) * contracts


def classify_long_security_requirement(row, include_long_regt):
    asset_type = str(row.get("Asset Type Normalized", "")).strip()
    symbol = str(row.get("Symbol", "")).strip().upper()
    qty = float(row.get("Qty", 0.0))
    market_value = abs(clean_numeric(row.get("Mkt Val (Market Value)", 0.0)))

    if qty <= 0:
        return None
    if asset_type in CASH_LIKE_ASSET_TYPES:
        return 0.0, "Long cash / money market"
    if asset_type in FIXED_INCOME_ASSET_TYPES:
        return 0.0, "Long fixed income"
    if symbol == "SNAXX":
        return 0.0, "Long cash-equivalent fund"
    if include_long_regt and asset_type in EQUITY_LIKE_ASSET_TYPES:
        return 0.50 * market_value, "Long marginable security (Reg-T 50%)"
    return 0.0, "Long fully-paid security"


def build_share_coverage_map(df_non_options):
    share_map = {}
    for _, row in df_non_options.iterrows():
        symbol = str(row.get("Symbol", "")).strip().upper()
        qty = float(row.get("Qty", 0.0))
        if not symbol or qty <= 0:
            continue
        share_map[symbol] = share_map.get(symbol, 0.0) + qty
    return share_map


def split_options(df_options):
    grouped = {}
    for _, row in df_options.iterrows():
        key = (str(row["Underlying"]).strip().upper(), str(row["Expiration"]), str(row["Opt Type"]).strip().upper())
        grouped.setdefault(key, []).append(row.copy())
    return grouped


def match_verticals(group_rows):
    longs = []
    shorts = []
    for row in sorted(group_rows, key=lambda r: float(r["Strike Price"])):
        qty = float(row["Qty"])
        leg = {
            "row": row,
            "strike": float(row["Strike Price"]),
            "remaining": abs(qty),
            "qty": qty,
        }
        if qty > 0:
            longs.append(leg)
        elif qty < 0:
            shorts.append(leg)

    matched = []
    leftovers = []
    option_type = str(group_rows[0]["Opt Type"]).strip().upper() if group_rows else ""
    short_iter = sorted(shorts, key=lambda leg: leg["strike"])
    if option_type == "P":
        short_iter = sorted(shorts, key=lambda leg: leg["strike"], reverse=True)

    for short_leg in short_iter:
        remaining_short = short_leg["remaining"]
        candidates = [leg for leg in longs if leg["remaining"] > 0]
        candidates.sort(key=lambda leg: (abs(leg["strike"] - short_leg["strike"]), leg["strike"]))

        for long_leg in candidates:
            if remaining_short <= 0:
                break
            matched_contracts = min(remaining_short, long_leg["remaining"])
            if matched_contracts <= 0:
                continue
            matched.append(
                {
                    "short_row": short_leg["row"].copy(),
                    "long_row": long_leg["row"].copy(),
                    "contracts": matched_contracts,
                    "short_strike": short_leg["strike"],
                    "long_strike": long_leg["strike"],
                    "option_type": option_type,
                }
            )
            remaining_short -= matched_contracts
            long_leg["remaining"] -= matched_contracts
        short_leg["remaining"] = remaining_short

    for short_leg in short_iter:
        if short_leg["remaining"] > 0:
            leftovers.append(
                {
                    "row": short_leg["row"].copy(),
                    "contracts": short_leg["remaining"],
                    "option_type": option_type,
                }
            )
    return matched, leftovers


def estimate_margin(df, include_long_regt=False, ticker_filter=None):
    df = df.copy()
    if ticker_filter:
        ticker_filter = ticker_filter.strip().upper()
        mask = (
            df["Underlying"].astype(str).str.upper().eq(ticker_filter)
            | df["Symbol"].astype(str).str.upper().eq(ticker_filter)
        )
        df = df[mask].copy()

    df_non_options = df[~df["Is Option"] & (df["Qty"] != 0)].copy()
    df_options = active_option_positions(df)
    if ticker_filter:
        df_options = df_options[df_options["Underlying"].astype(str).str.upper() == ticker_filter].copy()

    underlyings = sorted(df_options["Underlying"].astype(str).str.upper().unique())
    prices = fetch_underlying_prices(underlyings) if underlyings else {}
    share_coverage = build_share_coverage_map(df_non_options)

    rows = []

    for _, row in df_non_options.iterrows():
        symbol = str(row.get("Symbol", "")).strip().upper()
        qty = float(row.get("Qty", 0.0))
        market_value = clean_numeric(row.get("Mkt Val (Market Value)", 0.0))
        asset_type = str(row.get("Asset Type Normalized", "")).strip()

        if qty > 0:
            requirement, note = classify_long_security_requirement(row, include_long_regt)
            if requirement is None:
                continue
            rows.append(
                {
                    "ticker": symbol,
                    "category": "long_security",
                    "detail": asset_type or "Long security",
                    "quantity": qty,
                    "requirement": float(requirement),
                    "note": note,
                }
            )
        elif qty < 0:
            requirement = 1.50 * abs(market_value)
            rows.append(
                {
                    "ticker": symbol,
                    "category": "short_security",
                    "detail": asset_type or "Short security",
                    "quantity": qty,
                    "requirement": requirement,
                    "note": "Estimated short stock margin = 150% of short market value",
                }
            )

    grouped = split_options(df_options)
    for (underlying, expiration, opt_type), group_rows in grouped.items():
        matched, leftovers = match_verticals(group_rows)

        for pair in matched:
            short_strike = float(pair["short_strike"])
            long_strike = float(pair["long_strike"])
            contracts = float(pair["contracts"])
            width = abs(short_strike - long_strike)
            short_row = pair["short_row"]
            long_row = pair["long_row"]

            if opt_type == "P" and short_strike > long_strike:
                requirement = width * contracts * 100.0
                note = "Defined-risk short put spread; conservative width-based requirement"
                category = "short_put_spread"
                detail = f"{short_strike:g}/{long_strike:g} P {expiration}"
            elif opt_type == "C" and short_strike < long_strike:
                requirement = width * contracts * 100.0
                note = "Defined-risk short call spread; conservative width-based requirement"
                category = "short_call_spread"
                detail = f"{short_strike:g}/{long_strike:g} C {expiration}"
            else:
                requirement = 0.0
                note = "Long debit spread / covered short by long option; no incremental margin added"
                category = "long_defined_risk_spread"
                if opt_type == "P":
                    detail = f"{long_strike:g}/{short_strike:g} P {expiration}"
                else:
                    detail = f"{long_strike:g}/{short_strike:g} C {expiration}"

            rows.append(
                {
                    "ticker": underlying,
                    "category": category,
                    "detail": detail,
                    "quantity": contracts,
                    "requirement": requirement,
                    "note": note,
                }
            )

        for leftover in leftovers:
            row = leftover["row"]
            contracts = float(leftover["contracts"])
            strike = float(row["Strike Price"])
            option_price = float(row.get("Price Numeric", clean_numeric(row.get("Price", 0.0))))
            spot = prices.get(underlying)
            if spot is None:
                requirement = 0.0
                note = "Could not fetch underlying price; requirement unavailable"
                category = "unpriced_short_option"
            elif opt_type == "P":
                requirement = regt_short_put_requirement(spot, strike, option_price, contracts)
                note = "Estimated naked short put Reg-T requirement"
                category = "naked_short_put"
            else:
                covered_shares = share_coverage.get(underlying, 0.0)
                covered_contracts = min(contracts, covered_shares // 100)
                uncovered_contracts = contracts - covered_contracts
                share_coverage[underlying] = max(covered_shares - covered_contracts * 100, 0.0)
                if uncovered_contracts <= 0:
                    requirement = 0.0
                    note = "Covered call; shares cover short calls"
                    category = "covered_call"
                else:
                    requirement = regt_short_call_requirement(spot, strike, option_price, uncovered_contracts)
                    note = f"{covered_contracts:g} covered, {uncovered_contracts:g} uncovered; estimated naked short call Reg-T requirement"
                    category = "naked_short_call"

            rows.append(
                {
                    "ticker": underlying,
                    "category": category,
                    "detail": f"{strike:g} {opt_type} {expiration}",
                    "quantity": contracts,
                    "requirement": requirement,
                    "note": note,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out, pd.DataFrame(columns=["category", "count", "requirement"])

    out["requirement"] = pd.to_numeric(out["requirement"], errors="coerce").fillna(0.0)
    out = out.sort_values(["requirement", "ticker", "category"], ascending=[False, True, True]).reset_index(drop=True)

    summary = (
        out.groupby("category", dropna=False)
        .agg(count=("ticker", "size"), requirement=("requirement", "sum"))
        .reset_index()
        .sort_values("requirement", ascending=False)
    )
    total = pd.DataFrame([{"category": "TOTAL", "count": len(out), "requirement": out["requirement"].sum()}])
    summary = pd.concat([summary, total], ignore_index=True)
    return out, summary


def main():
    parser = argparse.ArgumentParser(description="Estimate portfolio margin requirement from a Schwab holdings CSV")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv next to script)")
    parser.add_argument("--ticker", default=None, help="Optional underlying ticker filter (e.g. SOXL)")
    parser.add_argument("--include-long-regt", action="store_true", help="Include 50% Reg-T initial margin for long equity/ETF positions")
    parser.add_argument("--output-csv", default=None, help="Optional path to write detailed rows as CSV")
    args = parser.parse_args()

    csv_path = default_csv_path(args.file, __file__)
    df = load_schwab_holdings(csv_path)
    details, summary = estimate_margin(df, include_long_regt=args.include_long_regt, ticker_filter=args.ticker)

    print("\nESTIMATED PORTFOLIO MARGIN REQUIREMENT")
    print("=" * 72)
    print(f"Source CSV: {csv_path}")
    if args.ticker:
        print(f"Ticker filter: {args.ticker.strip().upper()}")

    print("\nAssumptions:")
    print("  Long cash, money market, fixed income, and long options are treated as 0 incremental margin by default.")
    print("  Short vertical spreads use conservative width-based max-loss requirement.")
    print("  Naked short options use a Reg-T style estimate based on the current underlying price.")
    print("  Covered calls are treated as 0 incremental margin when shares cover the short contracts.")
    if args.include_long_regt:
        print("  Long equities and ETFs include a 50% Reg-T initial margin estimate.")

    if details.empty:
        print("\nNo positions matched the requested filter.")
    else:
        display_cols = ["ticker", "category", "detail", "quantity", "requirement", "note"]
        print("\nDETAIL")
        print(details[display_cols].to_string(index=False))

        print("\nSUMMARY")
        print(summary.to_string(index=False))
        total_req = float(summary.loc[summary["category"] == "TOTAL", "requirement"].iloc[0])
        print(f"\nEstimated total requirement: ${total_req:,.2f}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        details.to_csv(output_path, index=False)
        print(f"\nWrote detail CSV: {output_path}")


if __name__ == "__main__":
    main()
