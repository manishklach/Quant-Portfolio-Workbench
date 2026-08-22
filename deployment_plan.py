"""Generate a concrete deployment plan: closes, CSP strikes, CC strikes, and timeline."""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_core import (
    active_option_positions,
    clean_numeric,
    default_csv_path,
    load_schwab_holdings,
)

SECTION_WIDTH = 100
EQUITY = 20_000_000.0
MARGIN_AVAILABLE = 27_000_000.0

CSP_TARGETS = ["QQQ", "XLK", "MSFT", "NVDA", "AVGO"]
CSP_DELTA_TARGET = 0.12
CSP_DTE = 21
CSP_NOTIONAL_TOTAL = 8_000_000.0
CC_DELTA_TARGET = 0.15
CC_DTE = 21


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def bs_put_price(spot, strike, t, r, sigma):
    if t <= 0 or sigma <= 0 or spot <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return strike * math.exp(-r * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def bs_call_price(spot, strike, t, r, sigma):
    if t <= 0 or sigma <= 0 or spot <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    return spot * norm_cdf(d1) - strike * math.exp(-r * t) * norm_cdf(d2)


def bs_call_delta(spot, strike, t, r, sigma):
    if t <= 0 or sigma <= 0 or spot <= 0:
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
    return norm_cdf(d1)


def bs_put_delta(spot, strike, t, r, sigma):
    return bs_call_delta(spot, strike, t, r, sigma) - 1.0


def find_strike_for_delta(target_delta, spot, t, r, sigma, opt_type, lower_bound=0.3, upper_bound=3.0):
    lo = spot * lower_bound
    hi = spot * upper_bound
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if opt_type == "C":
            d = bs_call_delta(spot, mid, t, r, sigma)
        else:
            d = bs_put_delta(spot, mid, t, r, sigma)
        if abs(d - target_delta) < 0.001:
            break
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return round(mid / 2.5) * 2.5


def get_iv(ticker: str) -> float | None:
    try:
        tk = yf.Ticker(ticker)
        exp = tk.options
        if not exp:
            return None
        chain = tk.option_chain(exp[0])
        if chain.calls.empty:
            return None
        atm_idx = (chain.calls["strike"] - tk.history(period="2d")["Close"].iloc[-1]).abs().idxmin()
        mid = (chain.calls.loc[atm_idx, "bid"] + chain.calls.loc[atm_idx, "ask"]) / 2.0
        spot = float(tk.history(period="2d")["Close"].iloc[-1])
        if mid <= 0 or spot <= 0:
            return None
        strike = chain.calls.loc[atm_idx, "strike"]
        t = (datetime.strptime(exp[0], "%Y-%m-%d") - datetime.now()).days / 365.0
        if t <= 0:
            return None
        sigma = 0.3
        for _ in range(100):
            price = bs_call_price(spot, strike, t, 0.04, sigma)
            diff = price - mid
            if abs(diff) < 0.001:
                break
            vega = spot * norm_cdf(
                (math.log(spot / strike) + (0.04 + 0.5 * sigma ** 2) * t) / (sigma * math.sqrt(t))
            ) * math.sqrt(t)
            if vega < 1e-6:
                break
            sigma -= diff / vega
            sigma = max(0.05, min(2.0, sigma))
        return sigma
    except Exception:
        return None


def fmt(v):
    return f"${v:,.2f}"


def fmt_signed(v):
    return f"${v:+,.2f}"


def load_holdings(csv_path):
    df = load_schwab_holdings(csv_path)
    return df


def get_closes(options: pd.DataFrame, as_of: date) -> list[dict]:
    closes = []
    calls = options[(options["Opt Type"] == "C")].copy()

    for (und, expiry), grp in calls.groupby(["Underlying", "Expiration"]):
        dte = int(grp["Days To Expiry"].iloc[0]) if not grp["Days To Expiry"].isna().all() else 0
        longs = []
        shorts = []
        for _, row in grp.iterrows():
            qty = float(row["Qty"])
            leg = {
                "strike": float(row["Strike Price"]),
                "qty": abs(qty),
                "cost": float(row["Cost Basis Numeric"]),
                "mkt": float(row["Market Value Numeric"]),
                "price": float(row["Price Numeric"]),
                "side": "LONG" if qty > 0 else "SHORT",
                "symbol": row["Symbol"],
            }
            if qty > 0:
                longs.append(leg)
            else:
                shorts.append(leg)

        for short_leg in sorted(shorts, key=lambda x: x["strike"]):
            remaining = short_leg["qty"]
            for long_leg in sorted(longs, key=lambda x: x["strike"]):
                if remaining <= 0:
                    break
                avail = long_leg["qty"]
                if avail <= 0:
                    continue
                matched = min(remaining, avail)
                cost_b = long_leg["cost"] * (matched / avail)
                mkt_b = long_leg["mkt"] * (matched / avail)
                pl = (mkt_b - abs(short_leg["mkt"]) * (matched / short_leg["qty"])) - (cost_b - abs(short_leg["cost"]) * (matched / short_leg["qty"]))
                pl_pct = pl / abs(cost_b) * 100 if abs(cost_b) > 0 else 0

                if dte <= 14 and pl_pct < -50:
                    closes.append({
                        "underlying": und,
                        "expiry": expiry,
                        "dte": dte,
                        "long_strike": long_leg["strike"],
                        "short_strike": short_leg["strike"],
                        "qty": matched,
                        "pl": pl,
                        "pl_pct": pl_pct,
                        "reason": f"Expiring DTE={dte}, underwater {pl_pct:.0f}%",
                        "priority": "HIGH",
                    })
                remaining -= matched
                long_leg["qty"] -= matched

    standalone = []
    opts = calls.copy()
    for _, row in opts.iterrows():
        qty = float(row["Qty"])
        dte = int(row["Days To Expiry"]) if pd.notna(row["Days To Expiry"]) else 0
        mkt = float(row["Market Value Numeric"])
        cost = float(row["Cost Basis Numeric"])
        pl = mkt - cost
        pl_pct = pl / abs(cost) * 100 if abs(cost) > 0 else 0
        is_long = qty > 0
        if dte <= 14 and not is_long:
            standalone.append({
                "underlying": row["Underlying"],
                "expiry": row["Expiration"],
                "dte": dte,
                "strike": float(row["Strike Price"]),
                "qty": abs(qty),
                "side": "SHORT",
                "pl": pl,
                "pl_pct": pl_pct,
                "reason": f"Short call expiring DTE={dte}",
                "priority": "HIGH",
            })

    return sorted(closes, key=lambda x: x["dte"]) + standalone


def get_csp_strikes(as_of: date) -> list[dict]:
    results = []
    expiry = as_of + timedelta(days=CSP_DTE)
    t = CSP_DTE / 365.0
    r = 0.04
    per_name_notional = CSP_NOTIONAL_TOTAL / len(CSP_TARGETS)

    for ticker in CSP_TARGETS:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d")
            if hist.empty:
                continue
            spot = float(hist["Close"].iloc[-1])
            sigma = get_iv(ticker) or 0.35
            strike = find_strike_for_delta(
                -CSP_DELTA_TARGET, spot, t, r, sigma, "P"
            )
            if strike >= spot:
                strike = round(spot * 0.85 / 2.5) * 2.5
            premium = bs_put_price(spot, strike, t, r, sigma)
            notional = per_name_notional
            contracts = int(notional / (strike * 100))
            if contracts < 1:
                contracts = 1
            collateral = contracts * strike * 100
            annualized = premium / strike * (365.0 / CSP_DTE) * 100

            results.append({
                "ticker": ticker,
                "spot": spot,
                "strike": strike,
                "delta": -CSP_DELTA_TARGET,
                "premium": premium,
                "premium_total": premium * contracts * 100,
                "contracts": contracts,
                "collateral": collateral,
                "notional": contracts * strike * 100,
                "annualized": annualized,
                "sigma": sigma,
            })
        except Exception:
            continue

    return results


def get_cc_strikes(df: pd.DataFrame, as_of: date) -> list[dict]:
    results = []
    expiry = as_of + timedelta(days=CC_DTE)
    t = CC_DTE / 365.0
    r = 0.04

    # Only QQQI has shares + options chain (XQQI has no options)
    # QQQ/XLK are held via call spreads only, not shares, so no CCs
    cc_targets = [
        ("QQQI", 0.12),
    ]

    for ticker, delta_target in cc_targets:
        try:
            rows = df[df["Symbol"] == ticker]
            if rows.empty:
                continue
            qty = float(rows["Qty"].sum())
            if qty <= 0:
                continue
            cost_basis = float(rows["Cost Basis Numeric"].sum())
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d")
            if hist.empty:
                continue
            spot = float(hist["Close"].iloc[-1])
            sigma = get_iv(ticker) or 0.35
            strike = find_strike_for_delta(delta_target, spot, t, r, sigma, "C")
            premium = bs_call_price(spot, strike, t, r, sigma)
            annualized = premium / spot * (365.0 / CC_DTE) * 100

            target_pct = 0.25
            contracts = max(1, int(qty * target_pct / 100.0))
            premium_total = premium * contracts * 100

            results.append({
                "ticker": ticker,
                "spot": spot,
                "strike": strike,
                "delta": delta_target,
                "premium": premium,
                "premium_total": premium_total,
                "contracts": contracts,
                "annualized": annualized,
                "sigma": sigma,
            })
        except Exception:
            continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate a concrete deployment plan.")
    parser.add_argument("--current-nav", type=float, default=EQUITY, help="Current NAV (default: 20M)")
    parser.add_argument("--target-nav", type=float, default=25_000_000, help="Target NAV (default: 25M)")
    parser.add_argument("--end-date", default="2026-12-31", help="Target date (default: 2026-12-31)")
    parser.add_argument("--file", default=None, help="Holdings CSV path")
    parser.add_argument("--r", type=float, default=0.04, help="Risk-free rate (default: 0.04)")
    parser.add_argument("--as-of", default=None, help="As-of date (default: today)")
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    csv_path = default_csv_path(args.file, __file__)
    current_nav = args.current_nav
    target_nav = args.target_nav
    gap = target_nav - current_nav
    days_total = (end_date - as_of).days

    df = load_holdings(csv_path)
    options = active_option_positions(df)
    cash_row = df[df["Symbol"] == "SNAXX"]
    cash = float(cash_row["Market Value Numeric"].iloc[0]) if not cash_row.empty else 0.0
    nav = df["Market Value Numeric"].sum()

    qqqi_shares = int(df[df["Symbol"] == "QQQI"]["Qty"].sum()) if "QQQI" in df["Symbol"].values else 0
    xqqi_shares = int(df[df["Symbol"] == "XQQI"]["Qty"].sum()) if "XQQI" in df["Symbol"].values else 0
    qqqi_mv = float(df[df["Symbol"] == "QQQI"]["Market Value Numeric"].sum()) if "QQQI" in df["Symbol"].values else 0
    xqqi_mv = float(df[df["Symbol"] == "XQQI"]["Market Value Numeric"].sum()) if "XQQI" in df["Symbol"].values else 0

    closes = get_closes(options, as_of)
    csp_strikes = get_csp_strikes(as_of)
    cc_strikes = get_cc_strikes(df, as_of)

    total_csp_premium = sum(s["premium_total"] for s in csp_strikes)
    total_csp_collateral = sum(s["collateral"] for s in csp_strikes)
    total_csp_notional = sum(s["notional"] for s in csp_strikes)
    total_cc_premium = sum(s["premium_total"] for s in cc_strikes)
    days_remaining = days_total

    print("=" * SECTION_WIDTH)
    print("  DEPLOYMENT PLAN")
    print(f"  {as_of} to {end_date} ({days_remaining} days)")
    print("=" * SECTION_WIDTH)

    print(f"\n{'=' * SECTION_WIDTH}")
    print("  SITUATION SUMMARY")
    print(f"{'=' * SECTION_WIDTH}")
    print(f"  Current NAV:           {fmt(current_nav):>12s}  (Market Value: {fmt(nav):>12s})")
    print(f"  Target NAV:            {fmt(target_nav):>12s}")
    print(f"  Gap to close:          {fmt(gap):>12s}")
    print(f"  Cash (SNAXX):          {fmt(cash):>12s}")
    print(f"  Margin Available:      {fmt(MARGIN_AVAILABLE):>12s}")
    print(f"  QQQI/XQQI:             {fmt(qqqi_mv + xqqi_mv):>12s}  ({qqqi_shares} QQQI + {xqqi_shares} XQQI)")
    print(f"  Required Return:       {gap / current_nav * 100:.1f}% in {days_remaining}d")

    if closes:
        print(f"\n{'=' * SECTION_WIDTH}")
        print("  PHASE 1: IMMEDIATE CLOSES (Priority: HIGH)")
        print(f"{'=' * SECTION_WIDTH}")
        print(f"  {'Underlying':<12s} {'Expiry':<12s} {'Spread':<14s} {'Qty':>5s} {'P&L':>12s} {'P&L%':>7s}  {'Reason'}")
        print(f"  {'-' * (SECTION_WIDTH - 2)}")
        for c in closes:
            long_s = c.get("long_strike", c.get("strike", 0))
            short_s = c.get("short_strike", 0)
            spread = f"${long_s:.0f}/{short_s:.0f}"
            print(
                f"  {c['underlying']:<12s} {c['expiry']:<12s} {spread:<14s} "
                f"{c['qty']:>5.0f} {fmt_signed(c['pl']):>12s} {c['pl_pct']:>+6.0f}%  {c['reason']}"
            )
        cqty = sum(c["qty"] for c in closes)
        print(f"\n  Total: {cqty:.0f} contracts to close")
    else:
        print(f"\n  No urgent closes identified.")

    if csp_strikes:
        print(f"\n{'=' * SECTION_WIDTH}")
        print("  PHASE 2: CSP LADDER (Sell OTM Puts)")
        print(f"  Target: {fmt(CSP_NOTIONAL_TOTAL)} notional, delta={CSP_DELTA_TARGET}, DTE~{CSP_DTE}")
        print(f"{'=' * SECTION_WIDTH}")
        print(f"  {'Underlying':<10s} {'Spot':>8s} {'Strike':>8s} {'Delta':>6s} {'IV':>5s} "
              f"{'Prem':>7s} {'Contracts':>9s} {'Notional':>12s} {'Collateral':>12s} {'Ann.%':>6s}")
        print(f"  {'-' * (SECTION_WIDTH - 2)}")
        for s in csp_strikes:
            print(
                f"  {s['ticker']:<10s} {fmt(s['spot']):>8s} {fmt(s['strike']):>8s} "
                f"{s['delta']:>6.2f} {s['sigma']:>4.0%} {fmt(s['premium']):>7s} "
                f"{s['contracts']:>4d}x    {fmt(s['notional']):>12s} {fmt(s['collateral']):>12s} "
                f"{s['annualized']:>5.1f}%"
            )
        print(f"\n  Total Premium Collected: {fmt(total_csp_premium)}")
        print(f"  Total Notional:          {fmt(total_csp_notional)}")
        print(f"  Total Collateral:        {fmt(total_csp_collateral)}")
        print(f"  Remaining Cash:          {fmt(cash - total_csp_collateral)}")
        weekly_premium = total_csp_premium * (5 / CSP_DTE)
        print(f"  Est. Weekly Theta:       {fmt(weekly_premium)}")
        print(f"  Est. Monthly Theta:      {fmt(total_csp_premium * (21 / CSP_DTE))}")

    if cc_strikes:
        print(f"\n{'=' * SECTION_WIDTH}")
        print("  PHASE 3: COVERED CALLS (On QQQI Shares Only — XQQI Has No Options)")
        print(f"  Target: delta={CC_DELTA_TARGET}, DTE~{CC_DTE}")
        print(f"{'=' * SECTION_WIDTH}")
        print(f"  {'Underlying':<10s} {'Spot':>8s} {'Strike':>8s} {'Delta':>6s} "
              f"{'Prem':>7s} {'Contracts':>9s} {'Premium Total':>13s} {'Ann.%':>6s}")
        print(f"  {'-' * (SECTION_WIDTH - 2)}")
        for s in cc_strikes:
            print(
                f"  {s['ticker']:<10s} {fmt(s['spot']):>8s} {fmt(s['strike']):>8s} "
                f"{s['delta']:>6.2f} {fmt(s['premium']):>7s} "
                f"{s['contracts']:>4d}x    {fmt(s['premium_total']):>13s} "
                f"{s['annualized']:>5.1f}%"
            )
        print(f"\n  Total CC Premium: {fmt(total_cc_premium)}")

    print(f"\n{'=' * SECTION_WIDTH}")
    print("  PHASE 4: DIRECTIONAL ALPHA")
    print(f"{'=' * SECTION_WIDTH}")
    print(f"  {'Setup':<30s} {'Entry':>10s} {'Target':>10s} {'Stop':>10s} {'Size':>10s}")
    print(f"  {'-' * (SECTION_WIDTH - 2)}")
    print(f"  {'LITE (photonics pullback)':<30s} {'-15-20%':>10s} {'Spot':>10s} {'-25%':>10s} {'$500K':>10s}")
    print(f"  {'MSFT (SaaS AI entry)':<30s} {'$380':>10s} {'$420':>10s} {'$370':>10s} {'$500K':>10s}")
    print(f"  {'TQQQ (QQQ correction)':<30s} {'-8% QQQ':>10s} {'Spot':>10s} {'-15%':>10s} {'$500K':>10s}")

    print(f"\n{'=' * SECTION_WIDTH}")
    print("  PHASE 5: CAPITAL ALLOCATION SUMMARY")
    print(f"{'=' * SECTION_WIDTH}")
    print(f"  {'Layer':<30s} {'Capital':>12s} {'Est. Return':>12s} {'% of Goal':>10s}")
    print(f"  {'-' * (SECTION_WIDTH - 2)}")

    from_cc = total_cc_premium * (days_remaining / CC_DTE) * 0.6
    from_csp = total_csp_premium * (days_remaining / CSP_DTE) * 0.5
    from_dividends = (qqqi_mv + xqqi_mv) * 0.085
    from_restructure_mid = 400_000.0
    from_alpha_mid = 1_000_000.0
    from_cash_rest = (cash - total_csp_collateral) * 0.04 * days_remaining / 365.0

    total_est = from_csp + from_cc + from_dividends + from_alpha_mid + from_restructure_mid + from_cash_rest

    print(f"  {'CSP Ladder (6 mo)':<30s} {fmt(total_csp_collateral):>12s} {fmt(from_csp):>12s} {from_csp/gap*100:>9.1f}%")
    print(f"  {'Covered Calls (6 mo)':<30s} {'On QQQI only':>12s} {fmt(from_cc):>12s} {from_cc/gap*100:>9.1f}%")
    print(f"  {'QQQI/XQQI Dividends (6 mo)':<30s} {fmt(qqqi_mv + xqqi_mv):>12s} {fmt(from_dividends):>12s} {from_dividends/gap*100:>9.1f}%")
    print(f"  {'Directional Alpha':<30s} {'$1-2M':>12s} {'$500K-1.5M':>12s} {'25%':>9s}")
    print(f"  {'Call Spread Restructure':<30s} {'$0':>12s} {'$300-500K':>12s} {'10%':>9s}")
    print(f"  {'Cash / Idle':<30s} {fmt(cash - total_csp_collateral):>12s} {fmt(from_cash_rest):>12s} {from_cash_rest/gap*100:>9.1f}%")
    print(f"  {'-' * (SECTION_WIDTH - 2)}")
    print(f"  {'TOTAL ESTIMATED':<30s} {'':>12s} {fmt(total_est):>12s} {total_est/gap*100:>9.1f}%")
    print(f"  {'GAP REMAINING':<30s} {'':>12s} {fmt(gap - total_est):>12s} {(gap-total_est)/gap*100:>9.1f}%")
    print()
    print("  To close the gap, consider:")
    print("   - Add $2-3M more to QQQI/XQQI (+$170-255K dividends)")
    print("   - Increase CSP notional to $10-12M (+$50K)")
    print("   - More aggressive directional sizing (+$200-500K)")

    print(f"\n{'=' * SECTION_WIDTH}")
    print("  ACTION CHECKLIST")
    print(f"{'=' * SECTION_WIDTH}")
    print("  [ ] MONDAY: Execute Phase 1 closes (high-priority underwater spreads)")
    print(f"  [ ] MONDAY: Deploy CSP ladder ({len(csp_strikes)} strikes)")
    if cc_strikes:
        cc_label = f"Sell {len(cc_strikes)} CCs on QQQI (only ETF with options + shares)"
        print(f"  [ ] MONDAY: {cc_label}")
    print("  [ ] WEEKLY: Monitor CSP positions, roll at 50% profit")
    print("  [ ] WEEKLY: Run `python call_spread_review.py` to re-evaluate spreads")
    print("  [ ] WEEKLY: Run `python portfolio_dashboard.py` to track progress")
    print("  [ ] OPPORTUNISTIC: Add QQQI/XQQI on pullbacks (waiting)")
    print("  [ ] OPPORTUNISTIC: Deploy directional alpha on dips")
    print(f"\n  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Data: {csv_path}")


if __name__ == "__main__":
    main()
