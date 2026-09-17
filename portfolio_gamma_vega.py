"""Compute gamma and vega exposure for the entire option portfolio (calls and puts)."""

import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd

from portfolio_core import active_option_positions, default_csv_path, load_schwab_holdings
from option_math import bs_delta_vec, bs_gamma_vec, bs_vega_vec


def compute_greeks(options, r):
    df = options.copy()
    df["Spot Proxy"] = np.where(
        df["Strike Price"].fillna(0.0) > 0,
        df["Strike Price"].astype(float),
        np.maximum(df["Price Numeric"].astype(float) * 100.0, 1.0),
    )
    df["T"] = np.maximum(df["Days To Expiry"].fillna(1).astype(float), 1.0) / 365.0
    price_ratio = df["Price Numeric"].astype(float) / np.maximum(df["Spot Proxy"].astype(float), 1.0)
    df["IV Proxy"] = np.clip(
        np.where(df["Price Numeric"].astype(float) > 0, price_ratio * 4.0, 0.55),
        0.10,
        2.0,
    )

    spot = df["Spot Proxy"].to_numpy(dtype=float)
    strike = df["Strike Price"].to_numpy(dtype=float)
    t = df["T"].to_numpy(dtype=float)
    sigma = df["IV Proxy"].to_numpy(dtype=float)
    opt_type = df["Opt Type"].to_numpy(dtype=str)
    qty = df["Qty"].to_numpy(dtype=float)

    delta = bs_delta_vec(spot, strike, t, r, sigma, opt_type)
    gamma = bs_gamma_vec(spot, strike, t, r, sigma)
    vega = bs_vega_vec(spot, strike, t, r, sigma)

    df["Delta $"] = qty * 100.0 * delta * spot
    df["Gamma $/1%"] = qty * 100.0 * gamma * (spot * 0.01) ** 2
    df["Vega $/1vol"] = qty * 100.0 * vega * 0.01
    return df


def main():
    parser = argparse.ArgumentParser(description="Portfolio gamma and vega exposure for all options.")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv)")
    parser.add_argument("--as-of", default=None, help="Valuation date YYYY-MM-DD (default: today)")
    parser.add_argument("--r", type=float, default=0.04, help="Risk-free rate (default: 0.04)")
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)

    df = load_schwab_holdings(csv_path, as_of=as_of)
    options = active_option_positions(df)

    if options.empty:
        print("No active option positions found.")
        return

    greeks = compute_greeks(options, args.r)

    total_delta = greeks["Delta $"].sum()
    total_gamma = greeks["Gamma $/1%"].sum()
    total_vega = greeks["Vega $/1vol"].sum()

    by_underlying = greeks.groupby("Underlying")[["Delta $", "Gamma $/1%", "Vega $/1vol"]].sum()
    by_underlying = by_underlying.reindex(by_underlying["Gamma $/1%"].abs().sort_values(ascending=False).index)

    by_type = greeks.groupby("Opt Type")[["Delta $", "Gamma $/1%", "Vega $/1vol"]].sum()

    print("\nPortfolio Greeks — Gamma & Vega Exposure")
    print(f"File: {csv_path}")
    print(f"As-Of: {as_of}")
    print(f"Active option positions: {len(greeks)}")

    print("\n--- Greeks by Option Type ---")
    print(f"{'Type':<10} {'Delta $':<15} {'Gamma $/1%':<15} {'Vega $/1vol':<15}")
    print("-" * 58)
    for opt_type, row in by_type.iterrows():
        print(f"{opt_type:<10} ${row['Delta $']:<11,.2f} ${row['Gamma $/1%']:<11,.2f} ${row['Vega $/1vol']:<11,.2f}")
    print(f"{'TOTAL':<10} ${total_delta:<11,.2f} ${total_gamma:<11,.2f} ${total_vega:<11,.2f}")

    print("\n--- Greeks by Underlying (sorted by |Gamma|) ---")
    print(f"{'Underlying':<10} {'Delta $':<15} {'Gamma $/1%':<15} {'Vega $/1vol':<15}")
    print("-" * 58)
    for underlying, row in by_underlying.head(20).iterrows():
        print(f"{underlying:<10} ${row['Delta $']:<11,.2f} ${row['Gamma $/1%']:<11,.2f} ${row['Vega $/1vol']:<11,.2f}")

    print("\n--- Gamma & Vega Stress Scenarios ---")
    scenarios = []
    for label, spot_move, vol_move in [
        ("Spot +5%, IV +10pts", 0.05, 10),
        ("Spot +5%, IV -10pts", 0.05, -10),
        ("Spot -5%, IV +10pts", -0.05, 10),
        ("Spot -5%, IV -10pts", -0.05, -10),
        ("Spot -10%, IV +20pts", -0.10, 20),
        ("Spot +10%, IV -20pts", 0.10, -20),
        ("IV +5pts only", 0.0, 5),
        ("IV -5pts only", 0.0, -5),
        ("IV +10pts only", 0.0, 10),
        ("IV -10pts only", 0.0, -10),
    ]:
        pnl = (
            greeks["Delta $"] * spot_move
            + greeks["Gamma $/1%"] * (spot_move / 0.01) ** 2
            + greeks["Vega $/1vol"] * vol_move
        ).sum()
        scenarios.append((label, pnl))

    for label, pnl in scenarios:
        pct = pnl / greeks["Market Value Numeric"].sum() * 100 if greeks["Market Value Numeric"].sum() else 0
        print(f"  {label:<30} ${pnl:>12,.2f}  ({pct:+.2f}%)")

    print(f"\n--- Worst 5 Gamma/Vega Positions ---")
    worst = greeks.reindex(greeks["Gamma $/1%"].abs().sort_values(ascending=False).head(5).index)
    print(f"{'Symbol':<30} {'Type':<6} {'Qty':<10} {'Strike':<10} {'Expiry':<12} {'Gamma $/1%':<15} {'Vega $/1vol':<15}")
    print("-" * 100)
    for _, row in worst.iterrows():
        print(f"{row['Symbol']:<30} {row['Opt Type']:<6} {row['Qty']:<10.0f} {row['Strike Price']:<10.2f} {row['Expiration']:<12} ${row['Gamma $/1%']:<11,.2f} ${row['Vega $/1vol']:<11,.2f}")

    nav = df["Market Value Numeric"].sum()
    day_change = df["Day Change Numeric"].sum()
    print(f"\n--- Sizing ---")
    print(f"NAV: ${nav:,.2f}")
    print(f"Day Change: ${day_change:,.2f}")
    print(f"Net Delta $: ${total_delta:,.2f} ({total_delta/nav*100:.2f}% NAV)")
    print(f"Net Gamma $/1%: ${total_gamma:,.2f} ({total_gamma/nav*100:.4f}% NAV)")
    print(f"Net Vega $/1vol: ${total_vega:,.2f} ({total_vega/nav*100:.2f}% NAV)")


if __name__ == "__main__":
    main()
