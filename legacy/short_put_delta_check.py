"""Check short put delta accuracy: expected delta P&L vs actual day change."""

import argparse
import math
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_core import active_option_positions, default_csv_path, load_schwab_holdings


def norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def bs_d1(spot, strike, t, r, sigma):
    spot = np.maximum(np.asarray(spot, dtype=float), 1e-9)
    strike = np.maximum(np.asarray(strike, dtype=float), 1e-9)
    t = np.maximum(np.asarray(t, dtype=float), 1e-9)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-6)
    return (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))


def bs_delta_vec(spot, strike, t, r, sigma, opt_type):
    d1 = bs_d1(spot, strike, t, r, sigma)
    call_delta = norm_cdf(d1)
    put_delta = call_delta - 1.0
    return np.where(np.asarray(opt_type) == "C", call_delta, put_delta)


def bs_gamma_vec(spot, strike, t, r, sigma):
    d1 = bs_d1(spot, strike, t, r, sigma)
    pdf = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * d1 ** 2)
    return pdf / (np.maximum(np.asarray(spot, dtype=float), 1e-9) * np.maximum(np.asarray(sigma, dtype=float), 1e-6) * np.sqrt(np.maximum(np.asarray(t, dtype=float), 1e-9)))


def fetch_stock_data(tickers):
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    unique = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    prices, changes = {}, {}
    for t in unique:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="5d")
            if hist is not None and len(hist) >= 2:
                prices[t] = float(hist["Close"].iloc[-1])
                changes[t] = (float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[-2])) / float(hist["Close"].iloc[-2])
            elif len(hist) == 1:
                prices[t] = float(hist["Close"].iloc[-1])
                changes[t] = 0.0
        except Exception:
            pass
    return prices, changes


def main():
    parser = argparse.ArgumentParser(description="Check short put delta accuracy.")
    parser.add_argument("--file", default=None, help="Path to holdings CSV")
    parser.add_argument("--as-of", default=None, help="Date YYYY-MM-DD")
    parser.add_argument("--r", type=float, default=0.04, help="Risk-free rate")
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)
    df = load_schwab_holdings(csv_path, as_of=as_of)
    opts = active_option_positions(df)

    short_puts = opts[(opts["Opt Type"] == "P") & (opts["Qty"] < 0)].copy()
    if short_puts.empty:
        print("No short put positions found.")
        return

    tickers = short_puts["Underlying"].unique()
    stock_prices, stock_changes = fetch_stock_data(tickers)
    short_puts["Stock Price"] = short_puts["Underlying"].map(stock_prices)
    short_puts["Stock Chg %"] = short_puts["Underlying"].map(stock_changes)

    r = args.r
    rows = []
    for _, row in short_puts.iterrows():
        sp = row["Stock Price"]
        if pd.isna(sp):
            continue
        strike = float(row["Strike Price"])
        dte = float(row["Days To Expiry"]) if pd.notna(row["Days To Expiry"]) else 1.0
        t = max(dte, 1.0) / 365.0
        px = float(row["Price Numeric"])
        sigma = np.clip(max(px, 0.01) / max(sp, 0.01) * 4.0, 0.10, 2.0)
        qty = float(row["Qty"])
        sc = float(row["Stock Chg %"]) if pd.notna(row["Stock Chg %"]) else 0.0

        delta = float(bs_delta_vec(sp, strike, t, r, sigma, np.array(["P"])).flat[0])
        gamma = float(bs_gamma_vec(sp, strike, t, r, sigma).flat[0])

        # For a SHORT put: qty is negative, delta_bs is negative (put delta), so dollar delta = qty * 100 * delta_bs
        # A short put has positive delta (you want stock to go up)
        dollar_delta = qty * 100.0 * delta * sp

        delta_pnl = dollar_delta * sc
        gamma_pnl = qty * 100.0 * gamma * (sp * 0.01) * (sc / 0.01) ** 2
        actual_dc = float(row["Day Change Numeric"])
        unexplained = actual_dc - delta_pnl

        rows.append({
            "Symbol": row["Symbol"],
            "Underlying": row["Underlying"],
            "Expiry": row["Expiration"],
            "Strike": strike,
            "Qty": qty,
            "Stock Price": sp,
            "Stock Chg": sc * 100,
            "Put Delta": round(delta, 4),
            "Dollar Delta": dollar_delta,
            "Delta P&L": delta_pnl,
            "Gamma P&L": gamma_pnl,
            "Actual DC": actual_dc,
            "Unexplained": unexplained,
        })

    df_res = pd.DataFrame(rows)

    print(f"\n{'='*80}")
    print(f"  SHORT PUT DELTA CHECK")
    print(f"{'='*80}")
    print(f"\n{'Symbol':<35} {'Expiry':<12} {'Strike':<8} {'Qty':<8} {'Stock':<8} {'Chg%':<7} {'PutDelta':<9} {'DeltaPnL':<11} {'ActualDC':<11} {'Unxplnd':<11}")
    print(f"{'-'*125}")
    total_delta_pnl = 0.0
    total_actual = 0.0
    total_unexplained = 0.0
    for _, rw in df_res.iterrows():
        total_delta_pnl += rw["Delta P&L"]
        total_actual += rw["Actual DC"]
        total_unexplained += rw["Unexplained"]
        print(f"{rw['Symbol']:<35} {rw['Expiry']:<12} {rw['Strike']:<8.0f} {rw['Qty']:<8.0f} ${rw['Stock Price']:<5.0f} {rw['Stock Chg']:>+6.2f}% {rw['Put Delta']:<9.4f} ${rw['Delta P&L']:<7,.2f} ${rw['Actual DC']:<7,.2f} ${rw['Unexplained']:<7,.2f}")

    print(f"{'-'*125}")
    print(f"{'TOTAL':<35} {'':<12} {'':<8} {'':<8} {'':<8} {'':<7} {'':<9} ${total_delta_pnl:<7,.2f} ${total_actual:<7,.2f} ${total_unexplained:<7,.2f}")

    print(f"\n  => Delta explains ${total_delta_pnl:,.2f} of ${total_actual:,.2f} actual day change")
    pct = abs(total_delta_pnl / total_actual * 100) if total_actual != 0 else 0
    print(f"  => Unexplained (vega+theta+noise): ${total_unexplained:,.2f} ({pct:.1f}% of actual)")


if __name__ == "__main__":
    main()
