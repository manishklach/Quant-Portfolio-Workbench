#!/usr/bin/env python3
"""Estimate individual-stock and option P/L from Frankfurt versus US close.

Run: python frankfurt_portfolio_pnl.py [my_holdings.csv] [--ticker MU,NVDA]
Uses frankfurt_vs_nasdaq_compare for listings/FX and the after-hours option model.
Options retain CSV implied volatility and time to expiry; this is a spot-only
scenario, not executable option quotes. CSV option marks should be from the
US closing session being compared. Unquoted holdings are unpriced.
"""

import argparse
import math
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from after_hours_portfolio_pnl import estimate_option_after_hours_price
from frankfurt_vs_nasdaq_compare import ETF_TICKERS, build_comparison
from portfolio_core import default_csv_path, load_schwab_holdings


def positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (ValueError, TypeError):
        return False


def completed_us_close(ticker):
    """Exclude today's unfinished daily candle, including during US trading."""
    history = yf.Ticker(ticker).history(period="1mo", auto_adjust=False, timeout=15)
    now = datetime.now(ZoneInfo("America/New_York"))
    closes = history["Close"].dropna()
    if now.time() < time(16):
        closes = closes[[stamp.date() < now.date() for stamp in closes.index]]
    if closes.empty or not positive(closes.iloc[-1]):
        raise ValueError("No completed US close")
    return float(closes.iloc[-1]), str(closes.index[-1].date())


def prepare_quotes(comparison):
    rows = []
    for _, original in comparison.iterrows():
        row = original.to_dict()
        if pd.isna(row.get("quote_venue")):
            row["quote_venue"] = "Frankfurt"
        if pd.isna(row.get("quote_basis")):
            row["quote_basis"] = "last_trade"
        if row["status"] == "ok":
            try:
                if row.get("frankfurt_currency") != "EUR" or row.get("us_currency") != "USD":
                    raise ValueError("Unverified EUR/USD quote currencies")
                close, close_date = completed_us_close(row["ticker"])
                price = row["frankfurt_price_usd"]
                if not positive(price) or not positive(row["fx_eurusd"]):
                    raise ValueError("Missing or invalid price/FX")
                row.update(nasdaq_close_usd=close, us_close_date=close_date,
                           diff_usd=price-close, diff_pct=(price/close-1)*100)
                timestamp = row.get("frankfurt_quote_time")
                if not positive(timestamp):
                    raise ValueError("Missing Frankfurt quote timestamp")
                stamp = pd.Timestamp(timestamp, unit="s", tz="UTC")
                row["frankfurt_time_utc"] = stamp.isoformat()
                # A Frankfurt print before the baseline cannot measure a subsequent move.
                baseline = pd.Timestamp(close_date + " 16:00", tz="America/New_York")
                if stamp < baseline:
                    raise ValueError("Frankfurt print predates US close")
                if abs(row["diff_pct"]) > 15:
                    raise ValueError("Gap exceeds 15%; verify listing ratio/session before using")
            except Exception as exc:
                row["status"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def calculate_positions(holdings, quotes, ticker_filter=None, exclude_etfs=True):
    qmap = {row["ticker"]: row for _, row in quotes.iterrows()}
    wanted = {t.strip().upper() for t in ticker_filter.split(",")} if ticker_filter else None
    rows = []
    direct_etfs = set(holdings.loc[
        holdings["Asset Type Normalized"].str.contains("ETF|exchange traded", case=False, regex=True),
        "Underlying"])
    for _, holding in holdings.iterrows():
        ticker = holding["Underlying"]
        if wanted and ticker not in wanted:
            continue
        qty = holding["Qty"]
        if not math.isfinite(qty) or qty == 0:
            continue
        option = bool(holding["Is Option"])
        row = dict(ticker=ticker, symbol=holding["Symbol"], qty=qty,
                   is_option=option, csv_market_value=holding["Market Value Numeric"],
                   estimated_pnl_usd=float("nan"), method="unpriced")
        q = qmap.get(ticker, {})
        reason = None
        if exclude_etfs and (ticker in ETF_TICKERS or ticker in direct_etfs):
            reason = "ETF excluded"
        elif option and (not holding["Has Valid Expiry"] or holding["Is Expired"]):
            reason = "Expired or invalid option expiry"
        elif q.get("status") != "ok":
            reason = q.get("status", "No supported stock quote")
        if reason:
            row["status"] = reason
            rows.append(row)
            continue
        close, frankfurt = q["nasdaq_close_usd"], q["frankfurt_price_usd"]
        row.update(us_close_usd=close, frankfurt_usd=frankfurt,
                   quote_venue=q.get("quote_venue"), quote_basis=q.get("quote_basis"),
                   underlying_diff_usd=frankfurt-close, status="ok")
        try:
            if option:
                mark = holding["Price Numeric"]
                if not positive(mark):
                    raise ValueError("Invalid CSV option mark")
                estimated, method = estimate_option_after_hours_price(holding, close, frankfurt)
                if not math.isfinite(estimated):
                    raise ValueError("Invalid estimated option price")
                row.update(csv_option_mark=mark, estimated_option_mark=estimated,
                           estimated_pnl_usd=qty*100*(estimated-mark), method=method)
            else:
                row.update(estimated_pnl_usd=qty*(frankfurt-close), method="stock")
        except Exception as exc:
            row["status"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?")
    parser.add_argument("--file", help="Holdings CSV; default is my_holdings.csv beside script")
    parser.add_argument("--ticker", help="Comma-separated stock tickers")
    parser.add_argument("--output", help="Optional position CSV; also writes _quotes and _totals CSVs")
    parser.add_argument("--exclude-etfs", action="store_true", help="Use only individual stocks and their options")
    args = parser.parse_args()
    path = default_csv_path(args.file or args.csv, __file__)
    holdings = load_schwab_holdings(path)
    print(f"Holdings: {path}", flush=True)
    print("Fetching Frankfurt quotes, EUR/USD and completed US closes...", flush=True)
    comparison = build_comparison(path, allow_xetra=False, ticker_filter=args.ticker)
    quotes = prepare_quotes(comparison)
    positions = calculate_positions(holdings, quotes, args.ticker)
    if positions.empty:
        print("No matching positions.")
        return 0
    valid_quotes = quotes[quotes.status == "ok"] if not quotes.empty else quotes
    if not valid_quotes.empty:
        print("\nFrankfurt price comparison (USD per EUR shown in fx_eurusd)")
        print(valid_quotes[["ticker", "quote_venue", "quote_basis", "frankfurt_symbol", "frankfurt_price_eur", "fx_eurusd",
                            "frankfurt_price_usd", "nasdaq_close_usd", "diff_usd", "diff_pct",
                            "us_close_date", "frankfurt_time_utc"]].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    priced = positions[positions.status == "ok"]
    totals = []
    for ticker, group in priced.groupby("ticker"):
        stock = group.loc[~group.is_option, "estimated_pnl_usd"].sum()
        options = group.loc[group.is_option, "estimated_pnl_usd"].sum()
        totals.append(dict(ticker=ticker, stock_pnl_usd=stock, option_pnl_usd=options,
                           total_pnl_usd=stock+options))
    totals = pd.DataFrame(totals, columns=["ticker", "stock_pnl_usd", "option_pnl_usd", "total_pnl_usd"])
    if not totals.empty:
        print("\nFrankfurt portfolio impact (individual stocks and their options)")
        print(totals.to_string(index=False, float_format=lambda x: f"{x:+,.2f}"))
        print(f"\nStocks: ${totals.stock_pnl_usd.sum():+,.2f}")
        print(f"Options (estimated): ${totals.option_pnl_usd.sum():+,.2f}")
        print(f"TOTAL COVERED PORTFOLIO IMPACT: ${totals.total_pnl_usd.sum():+,.2f}")
    else:
        print("No priced positions; portfolio impact is unavailable.")
    print(f"Coverage: {len(priced)}/{len(positions)} nonzero positions.")
    skipped = positions[positions.status != "ok"]
    if not skipped.empty:
        print("\nExcluded/unpriced (not assumed to have zero P/L):")
        print(skipped[["ticker", "status"]].drop_duplicates().to_string(index=False))
    print("\nOptions: constant-IV spot scenario using CSV marks; includes signed long/short legs.")
    print("Use a CSV from the displayed US close. Cash, notes and missing quotes are outside this total.")
    print("ETFs and ETF options are excluded.")
    if args.output:
        output = Path(args.output)
        positions.to_csv(output, index=False)
        quotes.to_csv(output.with_name(output.stem + "_quotes.csv"), index=False)
        totals.to_csv(output.with_name(output.stem + "_totals.csv"), index=False)
        print(f"Saved positions, quotes and totals beside {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
