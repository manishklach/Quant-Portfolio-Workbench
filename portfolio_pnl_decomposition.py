"""Decompose portfolio day change into delta P&L, gamma P&L, vega P&L, theta, and residual noise."""

import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_core import active_option_positions, default_csv_path, load_schwab_holdings
from option_math import (
    bs_delta_vec,
    bs_gamma_vec,
    bs_theta_vec,
    bs_vega_vec,
    norm_cdf,
    norm_pdf,
)


def _theta_vec_per_day(spot, strike, t, r, sigma, opt_type):
    """Daily theta (canonical bs_theta_vec is per-year)."""
    return bs_theta_vec(spot, strike, t, r, sigma, opt_type) / 365.0


def compute_greeks(options, r):
    df = options.copy()
    spot_proxy = np.maximum(df["Strike Price"].fillna(0.0).astype(float), 1.0)
    df["T"] = np.maximum(df["Days To Expiry"].fillna(1).astype(float), 1.0) / 365.0
    price_ratio = np.maximum(df["Price Numeric"].astype(float), 0.01) / np.maximum(spot_proxy, 1.0)
    df["IV Proxy"] = np.clip(price_ratio * 4.0, 0.10, 2.0)

    spot = spot_proxy.to_numpy(dtype=float)
    strike = df["Strike Price"].to_numpy(dtype=float)
    t = df["T"].to_numpy(dtype=float)
    sigma = df["IV Proxy"].to_numpy(dtype=float)
    opt_type = df["Opt Type"].to_numpy(dtype=str)
    qty = df["Qty"].to_numpy(dtype=float)

    delta = bs_delta_vec(spot, strike, t, r, sigma, opt_type)
    gamma = bs_gamma_vec(spot, strike, t, r, sigma)
    vega = bs_vega_vec(spot, strike, t, r, sigma)
    theta = _theta_vec_per_day(spot, strike, t, r, sigma, opt_type)

    df["Delta $"] = qty * 100.0 * delta * spot
    df["Gamma $/1%"] = qty * 100.0 * gamma * spot * 0.01
    df["Vega $/1vol"] = qty * 100.0 * vega * 0.01
    df["Theta $/day"] = qty * 100.0 * theta
    return df


def fetch_stock_prices_and_changes(tickers):
    unique = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    prices = {}
    changes = {}
    import logging
    logger = logging.getLogger("yfinance")
    logger.setLevel(logging.CRITICAL)
    for ticker in unique:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if hist is not None and len(hist) >= 2:
                prices[ticker] = float(hist["Close"].iloc[-1])
                prev_close = float(hist["Close"].iloc[-2])
                curr_close = float(hist["Close"].iloc[-1])
                changes[ticker] = (curr_close - prev_close) / prev_close
            elif len(hist) == 1:
                prices[ticker] = float(hist["Close"].iloc[-1])
                changes[ticker] = 0.0
        except Exception:
            pass
    return prices, changes


def main():
    parser = argparse.ArgumentParser(description="Decompose portfolio day change into Greeks + residual.")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv)")
    parser.add_argument("--as-of", default=None, help="Valuation date YYYY-MM-DD (default: today)")
    parser.add_argument("--r", type=float, default=0.04, help="Risk-free rate (default: 0.04)")
    parser.add_argument("--vol-change", type=float, default=-0.42, help="Assumed vol change in points (default: -0.42 from VIX)")
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)

    df_all = load_schwab_holdings(csv_path, as_of=as_of)
    options = active_option_positions(df_all)

    if options.empty:
        print("No active option positions found.")
        return

    r = args.r
    greeks = compute_greeks(options, r)

    total_dc = df_all["Day Change Numeric"].sum()
    total_nav = df_all["Market Value Numeric"].sum()

    all_underlyings = list(greeks["Underlying"].unique())
    non_opts = df_all[~df_all["Is Option"]]
    for _, row in non_opts.iterrows():
        u = str(row["Underlying"]).strip().upper()
        if u and u not in all_underlyings:
            all_underlyings.append(u)
    stock_prices, stock_changes = fetch_stock_prices_and_changes(all_underlyings)

    greeks["Stock Price"] = greeks["Underlying"].map(stock_prices)
    greeks["Stock Chg %"] = greeks["Underlying"].map(stock_changes)

    # Recompute Greeks with actual stock prices where available
    has_price = greeks["Stock Price"].notna()
    if has_price.any():
        idx = has_price[has_price].index
        spot = greeks.loc[idx, "Stock Price"].to_numpy(dtype=float)
        strike = greeks.loc[idx, "Strike Price"].to_numpy(dtype=float)
        t = greeks.loc[idx, "T"].to_numpy(dtype=float)
        sigma = greeks.loc[idx, "IV Proxy"].to_numpy(dtype=float)
        opt_t = greeks.loc[idx, "Opt Type"].to_numpy(dtype=str)
        qty = greeks.loc[idx, "Qty"].to_numpy(dtype=float)

        greeks.loc[idx, "Delta $"] = qty * 100.0 * bs_delta_vec(spot, strike, t, r, sigma, opt_t) * spot
        greeks.loc[idx, "Gamma $/1%"] = qty * 100.0 * bs_gamma_vec(spot, strike, t, r, sigma) * spot * 0.01
        greeks.loc[idx, "Vega $/1vol"] = qty * 100.0 * bs_vega_vec(spot, strike, t, r, sigma) * 0.01
        greeks.loc[idx, "Theta $/day"] = qty * 100.0 * _theta_vec_per_day(spot, strike, t, r, sigma, opt_t)

    # Add stock/ETF positions as delta-only positions
    stock_rows = []
    for _, row in non_opts.iterrows():
        u = str(row["Underlying"]).strip().upper()
        sp = stock_prices.get(u, np.nan)
        sc = stock_changes.get(u, 0.0)
        qty = float(row["Qty"])
        if pd.notna(sp) and abs(qty) > 0:
            stock_rows.append({
                "Delta $": qty * sp,
                "Gamma $/1%": 0.0,
                "Vega $/1vol": 0.0,
                "Theta $/day": 0.0,
                "Stock Chg %": sc,
                "Symbol": row["Symbol"],
            })
    df_stocks = pd.DataFrame(stock_rows) if stock_rows else pd.DataFrame(columns=["Delta $", "Gamma $/1%", "Vega $/1vol", "Theta $/day", "Stock Chg %", "Symbol"])

    delta_pl = (greeks["Delta $"] * greeks["Stock Chg %"].fillna(0.0)).sum()
    if not df_stocks.empty:
        delta_pl += (df_stocks["Delta $"] * df_stocks["Stock Chg %"].fillna(0.0)).sum()

    gamma_pl = (greeks["Gamma $/1%"] * (greeks["Stock Chg %"].fillna(0.0) / 0.01) ** 2).sum()
    vega_pl = greeks["Vega $/1vol"].sum() * args.vol_change
    theta_pl = greeks["Theta $/day"].sum()

    predicted = delta_pl + gamma_pl + vega_pl + theta_pl
    residual = total_dc - predicted

    # Breakdown by options vs non-options
    non_opt_dc_total = non_opts["Day Change Numeric"].sum() if not df_stocks.empty else 0
    opt_dc = total_dc - non_opt_dc_total
    opt_predicted = (greeks["Delta $"] * greeks["Stock Chg %"].fillna(0.0)).sum() + gamma_pl + vega_pl + theta_pl
    opt_residual = opt_dc - opt_predicted
    non_opt_residual = residual - opt_residual

    print(f"\n{'='*60}")
    print(f"  PORTFOLIO P&L DECOMPOSITION")
    print(f"{'='*60}")
    print(f"  File: {csv_path}")
    print(f"  As-Of: {as_of}")
    print(f"  NAV: ${total_nav:,.2f}")
    print(f"  Assumed vol change: {args.vol_change:+.2f} vol pts (from VIX)")
    print(f"  Active option positions: {len(greeks)}")
    print(f"\n  {'Component':<30} {'Amount':<15} {'% NAV':<10}")
    print(f"  {'-'*30} {'-'*15} {'-'*10}")
    print(f"  {'Actual Day Change':<30} ${total_dc:<11,.2f} {total_dc/total_nav*100:>+7.2f}%")
    print(f"  {'-'*55}")
    print(f"  {'Delta P&L (stock moves)':<30} ${delta_pl:<11,.2f} {delta_pl/total_nav*100:>+7.2f}%")
    print(f"  {'Gamma P&L (acceleration)':<30} ${gamma_pl:<11,.2f} {gamma_pl/total_nav*100:>+7.2f}%")
    print(f"  {'Vega P&L (vol change)':<30} ${vega_pl:<11,.2f} {vega_pl/total_nav*100:>+7.2f}%")
    print(f"  {'Theta (time decay)':<30} ${theta_pl:<11,.2f} {theta_pl/total_nav*100:>+7.2f}%")
    print(f"  {'Expected P&L (sum)':<30} ${predicted:<11,.2f} {predicted/total_nav*100:>+7.2f}%")
    print(f"  {'-'*55}")
    print(f"  {'RESIDUAL (noise)':<30} ${residual:<11,.2f} {residual/total_nav*100:>+7.2f}%")
    print(f"{'='*60}")

    print(f"\n  Noise Breakdown:")
    print(f"    Options:      ${opt_residual:>10,.2f}  (theta=${theta_pl:<8,.2f} + other=${opt_residual-theta_pl:<8,.2f})")
    print(f"    Non-options:  ${non_opt_residual:>10,.2f}  (stock price mismatches, bonds, etc.)")
    print(f"    Total noise:  ${residual:>10,.2f}")
    print(f"\n  Theta: ${theta_pl:,.2f}/day")
    print(f"  Theta + Noise: ${residual + theta_pl:,.2f}")

    print(f"\n  Theta Detail by Option Type:")
    by_type = greeks.groupby("Opt Type")[["Theta $/day"]].sum()
    by_type["Count"] = greeks.groupby("Opt Type").size()
    for opt_type, row in by_type.iterrows():
        label = "Calls" if opt_type == "C" else "Puts"
        print(f"    {label:<10} ${row['Theta $/day']:<10,.2f}/day  ({int(row['Count'])} positions)")

    top_theta = greeks.reindex(greeks["Theta $/day"].abs().sort_values(ascending=False).head(10).index)
    print(f"\n  Top 10 Theta Positions:")
    print(f"  {'Symbol':<35} {'Type':<6} {'Qty':<8} {'Strike':<8} {'Expiry':<12} {'Theta $/day':<12}")
    print(f"  {'-'*80}")
    for _, row in top_theta.iterrows():
        print(f"  {row['Symbol']:<35} {row['Opt Type']:<6} {row['Qty']:<8.0f} {row['Strike Price']:<8.2f} {row['Expiration']:<12} ${row['Theta $/day']:<8,.2f}")

    # === Short put breakdown ===
    short_puts = greeks[(greeks["Opt Type"] == "P") & (greeks["Qty"] < 0)]
    if not short_puts.empty:
        sp_actual = short_puts["Day Change Numeric"].sum()
        sp_delta = (short_puts["Delta $"] * short_puts["Stock Chg %"].fillna(0.0)).sum()
        sp_theta = short_puts["Theta $/day"].sum()
        sp_gamma = (short_puts["Gamma $/1%"] * (short_puts["Stock Chg %"].fillna(0.0) / 0.01) ** 2).sum()
        sp_vega = short_puts["Vega $/1vol"].sum() * args.vol_change
        sp_expected = sp_delta + sp_gamma + sp_vega + sp_theta
        sp_unexplained = sp_actual - sp_expected

        print(f"\n  Short Put Breakdown:")
        print(f"    Actual DC:        ${sp_actual:>10,.2f}")
        print(f"    Delta P&L:        ${sp_delta:>10,.2f}  (directional, already in total)")
        print(f"    Gamma:            ${sp_gamma:>10,.2f}")
        print(f"    Vega:             ${sp_vega:>10,.2f}  (at {args.vol_change:+.2f} vol pts)")
        print(f"    Theta:            ${sp_theta:>10,.2f}/day  (positive = you earn)")
        print(f"    Expected (sum):   ${sp_expected:>10,.2f}")
        print(f"    Residual (noise): ${sp_unexplained:>10,.2f}")
        print(f"    => All already included in portfolio totals above.")
        print(f"    => No separate add-back needed.")

    print(f"\n{'='*60}")
    print(f"  BOTTOM LINE")
    print(f"{'='*60}")
    print(f"  Schwab reported:           ${total_dc:>10,.2f}")
    print(f"  Noise (bid-ask bounce):    ${residual:>10,.2f}")
    print(f"  Real P&L (signal):         ${predicted:>10,.2f}")
    if residual < 0:
        print(f"\n  => Noise was AGAINST you today. ADD back ${abs(residual):,.2f} to reported")
        print(f"    to get real fair-value P&L of ${predicted:,.2f}.")
    else:
        print(f"\n  => Noise was IN YOUR FAVOR today. SUBTRACT ${abs(residual):,.2f} from reported")
        print(f"    to get real fair-value P&L of ${predicted:,.2f}.")
    print(f"  => Noise reverses over time; expect this to mean-revert to ~$0.")


if __name__ == "__main__":
    main()
