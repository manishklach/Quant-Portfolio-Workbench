#!/usr/bin/env python3
"""Estimate after-hours portfolio P/L from a Schwab holdings export.

For stocks/ETFs:
  P/L = shares * (after_hours_price - regular_close_price)

For options:
  - infer implied volatility from the regular-session option mark
  - reprice the option using the after-hours underlying price
  - if IV inference fails, fall back to intrinsic-change approximation
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from portfolio_core import clean_numeric, default_csv_path, load_schwab_holdings


RISK_FREE_RATE = 0.045
DIVIDEND_YIELD = 0.0
UTC = timezone.utc
EASTERN_OFFSET = timezone(timedelta(hours=-4))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot: float, strike: float, time_to_expiry: float, rate: float, sigma: float, option_type: str, dividend_yield: float = 0.0) -> float:
    if min(spot, strike, time_to_expiry, sigma) <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "C" else max(strike - spot, 0.0)
        return intrinsic

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * sigma * sigma) * time_to_expiry
    ) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    discounted_spot = spot * math.exp(-dividend_yield * time_to_expiry)
    discounted_strike = strike * math.exp(-rate * time_to_expiry)

    if option_type == "C":
        return discounted_spot * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    return discounted_strike * norm_cdf(-d2) - discounted_spot * norm_cdf(-d1)


def implied_volatility(target_price: float, spot: float, strike: float, time_to_expiry: float, option_type: str) -> float | None:
    if min(target_price, spot, strike, time_to_expiry) <= 0:
        return None

    low = 1e-4
    high = 5.0
    low_price = bs_price(spot, strike, time_to_expiry, RISK_FREE_RATE, low, option_type, DIVIDEND_YIELD)
    high_price = bs_price(spot, strike, time_to_expiry, RISK_FREE_RATE, high, option_type, DIVIDEND_YIELD)

    if target_price < low_price - 1e-6 or target_price > high_price + 1e-6:
        return None

    for _ in range(80):
        mid = (low + high) / 2.0
        mid_price = bs_price(spot, strike, time_to_expiry, RISK_FREE_RATE, mid, option_type, DIVIDEND_YIELD)
        if abs(mid_price - target_price) < 1e-5:
            return mid
        if mid_price < target_price:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def year_fraction_to_expiry(expiration_text: str) -> float:
    try:
        expiry_date = datetime.strptime(str(expiration_text), "%m/%d/%Y").date()
    except Exception:
        return float("nan")

    expiry_dt = datetime.combine(expiry_date, time(16, 0), tzinfo=EASTERN_OFFSET).astimezone(UTC)
    now_dt = datetime.now(UTC)
    return max((expiry_dt - now_dt).total_seconds() / (365.0 * 24.0 * 3600.0), 1.0 / 365.0)


def fetch_quote_snapshot(ticker: str) -> dict[str, float | str | None]:
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    tk = yf.Ticker(ticker)
    info = {}
    fast = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    try:
        fast = dict(tk.fast_info)
    except Exception:
        fast = {}

    regular = info.get("regularMarketPrice")
    if regular is None or pd.isna(regular):
        regular = fast.get("lastPrice")
    previous_close = info.get("regularMarketPreviousClose")
    if previous_close is None or pd.isna(previous_close):
        previous_close = fast.get("regularMarketPreviousClose")
    post = info.get("postMarketPrice")
    exchange = info.get("exchange") or fast.get("exchange")

    return {
        "regular": float(regular) if regular is not None and not pd.isna(regular) else None,
        "previous_close": float(previous_close) if previous_close is not None and not pd.isna(previous_close) else None,
        "post": float(post) if post is not None and not pd.isna(post) else None,
        "exchange": exchange,
    }


def estimate_option_after_hours_price(row: pd.Series, underlying_regular: float, underlying_post: float) -> tuple[float, str]:
    option_type = str(row["Opt Type"]).strip().upper()
    strike = float(row["Strike Price"])
    option_mark = clean_numeric(row.get("Price"))
    time_to_expiry = year_fraction_to_expiry(row["Expiration"])

    if option_mark <= 0 or pd.isna(time_to_expiry):
        return option_mark, "mark"

    iv = implied_volatility(option_mark, underlying_regular, strike, time_to_expiry, option_type)
    if iv is not None:
        return bs_price(underlying_post, strike, time_to_expiry, RISK_FREE_RATE, iv, option_type, DIVIDEND_YIELD), "bs_iv_hold"

    regular_intrinsic = max(underlying_regular - strike, 0.0) if option_type == "C" else max(strike - underlying_regular, 0.0)
    post_intrinsic = max(underlying_post - strike, 0.0) if option_type == "C" else max(strike - underlying_post, 0.0)
    fallback_price = max(0.0, option_mark + (post_intrinsic - regular_intrinsic))
    return fallback_price, "intrinsic_fallback"


def build_after_hours_report(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdings = load_schwab_holdings(csv_path)
    quote_cache: dict[str, dict[str, float | str | None]] = {}
    position_rows: list[dict[str, object]] = []

    for _, row in holdings.iterrows():
        ticker = str(row["Underlying"]).strip().upper()
        market_value = clean_numeric(row.get("Mkt Val (Market Value)"))
        if market_value == 0.0 and ticker not in {"SNAXX"}:
            continue

        if ticker not in quote_cache:
            try:
                quote_cache[ticker] = fetch_quote_snapshot(ticker)
            except Exception:
                quote_cache[ticker] = {"regular": None, "previous_close": None, "post": None, "exchange": None}
        quote = quote_cache[ticker]

        asset_type = str(row.get("Asset Type Normalized", "")).strip()
        is_option = bool(row.get("Is Option"))
        qty = float(row.get("Qty", 0.0))
        multiplier = 100.0 if is_option else 1.0

        regular_price = quote["regular"]
        after_hours_price = quote["post"]
        pricing_method = "unchanged"

        if is_option:
            if regular_price is not None and after_hours_price is not None:
                estimated_option_price, pricing_method = estimate_option_after_hours_price(row, regular_price, after_hours_price)
                price_change = estimated_option_price - clean_numeric(row.get("Price"))
                ah_pl = qty * multiplier * price_change
                ah_mark = estimated_option_price
            else:
                ah_pl = 0.0
                ah_mark = clean_numeric(row.get("Price"))
        elif after_hours_price is not None and regular_price is not None:
            price_change = after_hours_price - regular_price
            ah_pl = qty * price_change
            ah_mark = after_hours_price
            pricing_method = "post_market"
        else:
            ah_pl = 0.0
            ah_mark = clean_numeric(row.get("Price")) if not is_option else clean_numeric(row.get("Price"))

        position_rows.append(
            {
                "ticker": ticker,
                "symbol": row.get("Symbol"),
                "asset_type": asset_type,
                "is_option": is_option,
                "qty": qty,
                "regular_underlying": regular_price,
                "after_hours_underlying": after_hours_price,
                "current_market_value": market_value,
                "estimated_ah_price": ah_mark,
                "estimated_ah_pl": ah_pl,
                "pricing_method": pricing_method,
            }
        )

    positions = pd.DataFrame(position_rows)
    by_ticker = (
        positions.groupby("ticker", as_index=False)
        .agg(
            current_market_value=("current_market_value", "sum"),
            estimated_ah_pl=("estimated_ah_pl", "sum"),
        )
        .sort_values("estimated_ah_pl", ascending=True)
    )
    return positions, by_ticker


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate after-hours portfolio P/L from my_holdings.csv")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv next to script)")
    parser.add_argument("--output", default=None, help="Optional CSV path for position-level output")
    args = parser.parse_args()

    csv_path = default_csv_path(args.file, __file__)
    positions, by_ticker = build_after_hours_report(csv_path)

    total_market_value = positions["current_market_value"].sum()
    total_ah_pl = positions["estimated_ah_pl"].sum()

    print("\nAfter-Hours Portfolio Estimate")
    print("=" * 88)
    print("\nBy Ticker")
    print(by_ticker.to_string(index=False))
    print("\nTotals")
    print(f"Current Market Value: ${total_market_value:,.2f}")
    print(f"Estimated After-Hours P/L: ${total_ah_pl:,.2f}")
    if total_market_value:
        print(f"Estimated After-Hours Return: {100.0 * total_ah_pl / total_market_value:.4f}%")

    if args.output:
        output_path = Path(args.output)
        positions.to_csv(output_path, index=False)
        print(f"\nPosition-level output written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
