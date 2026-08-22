"""Daily portfolio monitoring dashboard with goal tracking toward $25M."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_core import (
    active_option_positions,
    default_csv_path,
    load_schwab_holdings,
)

SECTION_WIDTH = 90

STATE_FILE = Path(__file__).with_name("dashboard_state.json")
TARGET_NAV = 25_000_000.0
EQUITY = 20_000_000.0
MARGIN_AVAILABLE = 27_000_000.0
START_DATE = date(2026, 7, 8)
END_DATE = date(2026, 12, 31)


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def bs_d1(spot, strike, t, r, sigma):
    spot = np.maximum(np.asarray(spot, dtype=float), 1e-9)
    strike = np.maximum(np.asarray(strike, dtype=float), 1e-9)
    t = np.maximum(np.asarray(t, dtype=float), 1e-9)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    return (np.log(spot / strike) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))


def bs_theta_vec(spot, strike, t, r, sigma, opt_type):
    d1 = bs_d1(spot, strike, t, r, sigma)
    d2 = d1 - np.maximum(np.asarray(sigma, dtype=float), 1e-6) * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9))
    pdf_d1 = (1.0 / math.sqrt(2.0 * math.pi)) * np.exp(-0.5 * d1**2)
    carry = r * np.asarray(strike, dtype=float) * np.exp(-r * np.maximum(np.asarray(t, dtype=float), 1e-9))
    call_theta = (
        -np.asarray(spot, dtype=float) * pdf_d1 * np.asarray(sigma, dtype=float)
        / (2.0 * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9)))
        - carry * norm_cdf(d2)
    )
    put_theta = (
        -np.asarray(spot, dtype=float) * pdf_d1 * np.asarray(sigma, dtype=float)
        / (2.0 * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9)))
        + carry * norm_cdf(-d2)
    )
    return np.where(np.asarray(opt_type) == "C", call_theta, put_theta)


def bs_delta_vec(spot, strike, t, r, sigma, opt_type):
    d1 = bs_d1(spot, strike, t, r, sigma)
    call_delta = norm_cdf(d1)
    put_delta = call_delta - 1.0
    return np.where(np.asarray(opt_type) == "C", call_delta, put_delta)


def fmt(v: float) -> str:
    return f"${v:,.2f}"

def fmt_signed(v: float) -> str:
    return f"${v:+,.2f}"


def fmt_pct(v: float) -> str:
    return f"{v:+.2f}%"


def fmt_int(v: float) -> str:
    return f"{v:,.0f}"


def load_state() -> list[dict]:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            pass
    return []


def save_state(history: list[dict]):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(history, f, indent=2)


def compute_option_greeks(options: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    if options.empty:
        return options
    opts = options.copy()
    underlying_list = opts["Underlying"].unique()

    spot_dict = {}
    for und in underlying_list:
        try:
            tk = yf.Ticker(und)
            hist = tk.history(period="2d")
            if not hist.empty:
                spot_dict[und] = float(hist["Close"].iloc[-1])
        except Exception:
            pass

    opts["Spot"] = opts["Underlying"].map(spot_dict).fillna(
        opts["Strike Price"].fillna(opts["Price Numeric"] * 100.0)
    )
    opts["T"] = np.maximum(opts["Days To Expiry"].fillna(1).astype(float), 1.0) / 365.0
    price_ratio = opts["Price Numeric"].astype(float) / np.maximum(opts["Spot"].astype(float), 1.0)
    opts["IV"] = np.clip(np.where(opts["Price Numeric"] > 0, price_ratio * 4.0, 0.55), 0.10, 2.0)

    spot = opts["Spot"].to_numpy(dtype=float)
    strike = opts["Strike Price"].to_numpy(dtype=float)
    t = opts["T"].to_numpy(dtype=float)
    sigma = opts["IV"].to_numpy(dtype=float)
    opt_type = opts["Opt Type"].to_numpy(dtype=str)

    opts["Delta"] = bs_delta_vec(spot, strike, t, risk_free_rate, sigma, opt_type)
    opts["Theta"] = bs_theta_vec(spot, strike, t, risk_free_rate, sigma, opt_type)
    opts["Delta Dollars"] = opts["Qty"] * opts["Multiplier"] * opts["Delta"] * opts["Spot"]
    opts["Theta Dollars 1 Day"] = opts["Qty"] * opts["Multiplier"] * opts["Theta"] / 365.0
    return opts


def print_section(title: str):
    print(f"\n{'=' * SECTION_WIDTH}")
    print(f"  {title}")
    print(f"{'=' * SECTION_WIDTH}")


def print_progress_bar(current: float, target: float, width: int = 40):
    pct = min(max(current / target, 0.0), 1.0)
    filled = int(pct * width)
    bar = "=" * filled + "-" * (width - filled)
    print(f"  [{bar}] {pct * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Portfolio monitoring dashboard.")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv)")
    parser.add_argument("--as-of", default=None, help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--r", type=float, default=0.04, help="Risk-free rate (default: 0.04)")
    parser.add_argument("--reset", action="store_true", help="Reset saved dashboard history")
    args = parser.parse_args()

    today = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)

    if args.reset:
        save_state([])
        print("Dashboard history reset.")

    df = load_schwab_holdings(csv_path, as_of=today)
    nav = df["Market Value Numeric"].sum()
    cash_row = df[df["Symbol"] == "SNAXX"]
    cash = float(cash_row["Market Value Numeric"].iloc[0]) if not cash_row.empty else 0.0

    options = active_option_positions(df)
    risk_free = args.r

    greeks = compute_option_greeks(options, risk_free)
    total_theta = float(greeks["Theta Dollars 1 Day"].sum()) if not greeks.empty else 0.0
    total_delta = float(greeks["Delta Dollars"].sum()) if not greeks.empty else 0.0

    days_total = (END_DATE - START_DATE).days
    days_elapsed = (today - START_DATE).days
    days_remaining = max(0, days_total - days_elapsed)
    pct_time_elapsed = days_elapsed / days_total * 100 if days_total > 0 else 0

    gap = TARGET_NAV - EQUITY
    required_total = TARGET_NAV - nav
    daily_required = required_total / days_remaining if days_remaining > 0 else 0
    progress_toward_goal = max(0.0, nav - EQUITY)  # nav below equity = no progress yet

    margin_used = max(0.0, 2.0 * EQUITY - MARGIN_AVAILABLE)
    margin_remaining = MARGIN_AVAILABLE - margin_used
    margin_pct = margin_used / (2.0 * EQUITY) * 100 if EQUITY > 0 else 0

    qty_col = "Qty (Quantity)"
    short_put_notional = 0.0
    if not greeks.empty:
        short_puts = greeks[(greeks["Opt Type"] == "P") & (greeks["Qty"] < 0)]
        if not short_puts.empty:
            short_put_notional = (short_puts["Strike Price"] * short_puts["Qty"].abs() * 100).sum()

    state = load_state()
    today_entry = {
        "date": today.isoformat(),
        "nav": nav,
        "cash": cash,
        "theta": total_theta,
    }
    if state and state[-1]["date"] == today.isoformat():
        state[-1] = today_entry
    else:
        state.append(today_entry)
    save_state(state)

    print("=" * SECTION_WIDTH)
    print("  PORTFOLIO DASHBOARD")
    print(f"  {today}")
    print(f"  Data: {csv_path}")
    print(f"  State: {STATE_FILE}")
    print("=" * SECTION_WIDTH)

    print_section("NAV & GOAL PROGRESS")
    print(f"  Market Value:              {fmt(nav):>14s}  (={fmt_signed(nav - EQUITY):>14s} from $20M base)")
    print(f"  Equity Base:               {fmt(EQUITY):>14s}")
    print(f"  Target (Dec 31, 2026):     {fmt(TARGET_NAV):>14s}")
    print(f"  Gap to Target:             {fmt(required_total):>14s}")
    print(f"  Days Remaining:            {days_remaining:>14d}  ({pct_time_elapsed:.0f}% of H2 elapsed)")
    print(f"  Required Daily Gain:       {fmt(daily_required):>14s}")
    print()
    print("  Progress:  $20M -> $25M")
    print_progress_bar(progress_toward_goal, gap)
    print(f"  {fmt(progress_toward_goal):>12s} / {fmt(gap)}")

    print_section("CASH & MARGIN")
    print(f"  Cash (SNAXX):                {fmt(cash):>14s}")
    print(f"  Equity Base:                {fmt(EQUITY):>14s}")
    print(f"  Total Buying Power (2x eq): {fmt(2.0 * EQUITY):>14s}")
    print(f"  Margin Used (est.):         {fmt(margin_used):>14s} ({margin_pct:.1f}%)")
    print(f"  Margin Available:           {fmt(MARGIN_AVAILABLE):>14s}")
    print(f"  Short Put Notional (est.):  {fmt(short_put_notional):>14s}")

    print_section("OPTION GREEKS (Est.)")
    if not greeks.empty:
        print(f"  Net Delta Dollars:          {fmt(total_delta):>14s}")
        print(f"  Theta (per day):            {fmt(total_theta):>14s}")
        print(f"  Monthly theta (21d est.):   {fmt(total_theta * 21):>14s}")
        print()
        print("  Top 5 by Delta Exposure:")
        delta_by_und = greeks.groupby("Underlying")["Delta Dollars"].sum().sort_values(
            key=abs, ascending=False
        ).head(5)
        for und, val in delta_by_und.items():
            print(f"    {und:<8s} {fmt(val):>14s}")
        print()
        print("  Top 5 by Theta Decay:")
        theta_by_und = greeks.groupby("Underlying")["Theta Dollars 1 Day"].sum().sort_values(
            ascending=False
        ).head(5)
        for und, val in theta_by_und.items():
            print(f"    {und:<8s} {fmt(val):>14s}")
    else:
        print("  No active option positions")

    print_section("ASSET MIX")
    asset_col = "Asset Type Normalized"
    if asset_col in df.columns:
        mix = df.groupby(asset_col)["Market Value Numeric"].sum().sort_values(ascending=False)
        for atype, aval in mix.items():
            label = atype if str(atype).strip() else "Unknown"
            pct_nav = aval / nav * 100 if nav != 0 else 0
            print(f"  {label:<28s} {fmt(aval):>14s}  ({pct_nav:.1f}%)")

    if state and len(state) > 1:
        print_section("NAV HISTORY (Last 10 Entries)")
        print(f"  {'Date':<14s} {'NAV':>12s} {'Cash':>12s} {'Theta/Day':>12s}")
        for entry in state[-10:]:
            print(
                f"  {entry['date']:<14s} {fmt(entry['nav']):>12s} "
                f"{fmt(entry.get('cash', 0)):>12s} {fmt(entry.get('theta', 0)):>12s}"
            )

        first_nav = state[0]["nav"]
        last_nav = state[-1]["nav"]
        total_change = last_nav - first_nav
        total_days = (datetime.fromisoformat(state[-1]["date"]).date()
                      - datetime.fromisoformat(state[0]["date"]).date()).days
        avg_daily = total_change / total_days if total_days > 0 else 0
        print(f"\n  Tracking period: {state[0]['date']} to {state[-1]['date']} ({total_days} days)")
        print(f"  Total change: {fmt(total_change)} (avg {fmt(avg_daily)}/day)")

    print_section("CSP LADDER STATUS")
    print("  (Not yet deployed - use csp_calc.py to size)")
    print(f"  Target CSP notional: $5-8M  ->  target theta: $8-10K/day")

    print_section("NEXT ACTIONS")
    print("  1. Run `python call_spread_review.py` to review verticals")
    print("  2. Run `python portfolio_growth_plan.py` for a structured plan")
    print("  3. Deploy CSP ladder once spread review is complete")
    print("  4. Consider adding to QQQI/XQQI on pullbacks")
    print(f"\n  Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
