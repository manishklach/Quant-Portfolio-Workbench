#!/usr/bin/env python3
"""Fetch Frankfurt-listed quotes for each unique ticker in a Schwab holdings export."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    requests = None

try:
    import yfinance as yf
except ImportError:
    yf = None

from portfolio_core import default_csv_path, load_schwab_holdings


SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
HEADERS = {"User-Agent": "Mozilla/5.0"}
LEGAL_SUFFIX_RE = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|holdings|holding|plc|sa|nv|ag|se|class\s+[a-z])\b",
    flags=re.I,
)
PUNCT_RE = re.compile(r"[^A-Za-z0-9]+")
FRANKFURT_EXCHANGES = {"frankfurt"}
XETRA_EXCHANGES = {"xetra"}


def normalize_name(text: str) -> str:
    cleaned = PUNCT_RE.sub(" ", str(text)).strip()
    cleaned = LEGAL_SUFFIX_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_probable_security_ticker(ticker: str) -> bool:
    ticker = str(ticker).strip().upper()
    if not ticker:
        return False
    if ticker == "SNAXX":
        return False
    return bool(re.match(r"^[A-Z][A-Z0-9.\-]{0,9}$", ticker))


def search_yahoo(query: str, quotes_count: int = 20) -> list[dict]:
    if requests is None:
        raise RuntimeError("requests not installed. Run: pip install requests")

    response = requests.get(
        SEARCH_URL,
        params={"q": query, "quotesCount": quotes_count, "newsCount": 0},
        headers=HEADERS,
        timeout=20,
    )
    response.raise_for_status()
    return response.json().get("quotes", [])


def get_primary_security_metadata(base_ticker: str) -> dict:
    results = search_yahoo(base_ticker, quotes_count=12)
    for item in results:
        if str(item.get("symbol", "")).upper() == base_ticker.upper():
            return item
    return results[0] if results else {}


def build_search_terms(base_ticker: str, metadata: dict) -> list[str]:
    candidates = [
        base_ticker,
        metadata.get("longname") or "",
        metadata.get("shortname") or "",
    ]

    expanded: list[str] = []
    for candidate in candidates:
        candidate = str(candidate).strip()
        if not candidate:
            continue
        expanded.append(candidate)
        normalized = normalize_name(candidate)
        if normalized and normalized != candidate:
            expanded.append(normalized)
        words = normalized.split()
        if len(words) >= 2:
            expanded.append(" ".join(words[:2]))
        if len(words) >= 3:
            expanded.append(" ".join(words[:3]))

    deduped: list[str] = []
    seen: set[str] = set()
    for term in expanded:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped


def score_candidate(candidate: dict, base_metadata: dict, *, allow_xetra: bool) -> int:
    exchange = str(candidate.get("exchDisp") or candidate.get("exchange") or "").strip().lower()
    symbol = str(candidate.get("symbol") or "")
    longname = str(candidate.get("longname") or candidate.get("shortname") or "")
    base_long = str(base_metadata.get("longname") or base_metadata.get("shortname") or "")

    score = 0
    if exchange in FRANKFURT_EXCHANGES:
        score += 100
    elif allow_xetra and exchange in XETRA_EXCHANGES:
        score += 70
    else:
        return -1

    if symbol.endswith(".F"):
        score += 20
    elif symbol.endswith(".DE"):
        score += 10

    if normalize_name(longname) and normalize_name(longname) == normalize_name(base_long):
        score += 40
    elif normalize_name(base_long) and normalize_name(base_long) in normalize_name(longname):
        score += 25

    quote_type = str(candidate.get("quoteType") or "")
    if quote_type in {"EQUITY", "ETF"}:
        score += 10
    return score


def find_frankfurt_listing(base_ticker: str, *, allow_xetra: bool = False) -> dict:
    base_metadata = get_primary_security_metadata(base_ticker)
    if not base_metadata:
        return {}

    best_candidate: dict = {}
    best_score = -1
    best_query = None
    for query in build_search_terms(base_ticker, base_metadata):
        try:
            results = search_yahoo(query, quotes_count=20)
        except Exception:
            continue
        for candidate in results:
            score = score_candidate(candidate, base_metadata, allow_xetra=allow_xetra)
            if score > best_score:
                best_candidate = candidate
                best_score = score
                best_query = query

    if best_score < 0:
        return {}

    listing = dict(best_candidate)
    listing["_matched_query"] = best_query
    listing["_base_name"] = base_metadata.get("longname") or base_metadata.get("shortname") or base_ticker
    return listing


def fetch_quote_snapshot(symbol: str) -> dict[str, float | str | None]:
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    ticker = yf.Ticker(symbol)
    info = {}
    fast = {}
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    try:
        fast = dict(ticker.fast_info)
    except Exception:
        fast = {}

    regular = info.get("regularMarketPrice")
    if regular is None or pd.isna(regular):
        regular = fast.get("lastPrice")

    previous_close = info.get("regularMarketPreviousClose")
    if previous_close is None or pd.isna(previous_close):
        previous_close = fast.get("regularMarketPreviousClose")

    if (regular is None or pd.isna(regular)) or (previous_close is None or pd.isna(previous_close)):
        try:
            hist = ticker.history(period="5d", interval="1d", auto_adjust=False)
        except Exception:
            hist = pd.DataFrame()
        closes = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
        if regular is None or pd.isna(regular):
            if len(closes) >= 1:
                regular = float(closes.iloc[-1])
        if previous_close is None or pd.isna(previous_close):
            if len(closes) >= 2:
                previous_close = float(closes.iloc[-2])

    currency = info.get("currency") or fast.get("currency")
    exchange = info.get("exchange") or fast.get("exchange")
    return {
        "regular": float(regular) if regular is not None and not pd.isna(regular) else None,
        "previous_close": float(previous_close) if previous_close is not None and not pd.isna(previous_close) else None,
        "currency": currency,
        "exchange": exchange,
    }


def fetch_fx_rate_to_usd(currency: str) -> float | None:
    currency = str(currency or "").strip().upper()
    if not currency:
        return None
    if currency == "USD":
        return 1.0
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    pair = f"{currency}USD=X"
    quote = fetch_quote_snapshot(pair)
    rate = quote.get("regular")
    return float(rate) if rate is not None and not pd.isna(rate) else None


def build_report(
    csv_path: Path,
    *,
    allow_xetra: bool = False,
    ticker_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdings = load_schwab_holdings(csv_path)
    tickers = sorted(set(holdings["Underlying"].dropna().astype(str).str.upper()))
    if ticker_filter:
        wanted = {item.strip().upper() for item in ticker_filter.split(",") if item.strip()}
        tickers = [ticker for ticker in tickers if ticker in wanted]

    direct_positions = (
        holdings[~holdings["Is Option"]]
        .groupby("Underlying", as_index=False)
        .agg(
            qty=("Qty", "sum"),
            current_market_value=("Market Value Numeric", "sum"),
        )
        .rename(columns={"Underlying": "ticker"})
    )
    direct_positions["ticker"] = direct_positions["ticker"].astype(str).str.upper()
    if ticker_filter:
        direct_positions = direct_positions[direct_positions["ticker"].isin(wanted)].copy()

    rows: list[dict[str, object]] = []
    fx_cache: dict[str, float | None] = {}
    for base_ticker in tickers:
        if not is_probable_security_ticker(base_ticker):
            continue

        try:
            listing = find_frankfurt_listing(base_ticker, allow_xetra=allow_xetra)
        except Exception as exc:
            continue

        if not listing:
            continue

        yahoo_symbol = str(listing.get("symbol") or "")
        try:
            quote = fetch_quote_snapshot(yahoo_symbol)
        except Exception as exc:
            continue

        regular = quote.get("regular")
        previous_close = quote.get("previous_close")
        if regular is None or pd.isna(regular):
            continue

        change = regular - previous_close if previous_close is not None and not pd.isna(previous_close) else None
        currency = str(quote.get("currency") or "").upper()
        if currency not in fx_cache:
            try:
                fx_cache[currency] = fetch_fx_rate_to_usd(currency)
            except Exception:
                fx_cache[currency] = None
        usd_fx = fx_cache.get(currency)
        price_usd = regular * usd_fx if usd_fx is not None else None
        previous_close_usd = previous_close * usd_fx if previous_close is not None and usd_fx is not None else None
        change_usd = change * usd_fx if change is not None and usd_fx is not None else None

        rows.append(
            {
                "ticker": base_ticker,
                "matched_name": listing.get("longname") or listing.get("shortname"),
                "frankfurt_symbol": yahoo_symbol,
                "frankfurt_exchange": listing.get("exchDisp") or listing.get("exchange"),
                "quote_exchange": quote.get("exchange"),
                "currency": currency,
                "price": regular,
                "previous_close": previous_close,
                "change": change,
                "usd_fx": usd_fx,
                "price_usd": price_usd,
                "previous_close_usd": previous_close_usd,
                "change_usd": change_usd,
                "matched_query": listing.get("_matched_query"),
                "base_name": listing.get("_base_name"),
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return report, pd.DataFrame()

    portfolio = direct_positions.merge(report, on="ticker", how="inner")
    if not portfolio.empty:
        portfolio["approx_position_change_usd"] = portfolio["qty"] * portfolio["change_usd"]
    return report, portfolio


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Frankfurt quotes for each unique portfolio ticker")
    parser.add_argument("csv", nargs="?", help="Optional positional path to holdings CSV")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv next to script)")
    parser.add_argument("--ticker", default=None, help="Optional comma-separated ticker filter, e.g. MU,NVDA,AAPL")
    parser.add_argument("--allow-xetra", action="store_true", help="Allow XETRA listings when no Frankfurt listing is found")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    csv_path = default_csv_path(args.file or args.csv, __file__)
    report, portfolio = build_report(csv_path, allow_xetra=args.allow_xetra, ticker_filter=args.ticker)

    if report.empty:
        print("No rows returned.")
        return 0

    display_columns = [
        col
        for col in [
            "ticker",
            "frankfurt_symbol",
            "frankfurt_exchange",
            "currency",
            "price",
            "previous_close",
            "change",
            "price_usd",
            "previous_close_usd",
            "change_usd",
            "matched_name",
        ]
        if col in report.columns
    ]
    print(report[display_columns].to_string(index=False))

    print("\nApprox Portfolio Change From Frankfurt Quotes")
    if not portfolio.empty:
        total_change = float(portfolio["approx_position_change_usd"].sum())
        print(
            portfolio[
                [
                    "ticker",
                    "qty",
                    "change_usd",
                    "approx_position_change_usd",
                ]
            ]
            .sort_values("approx_position_change_usd", ascending=False)
            .to_string(index=False)
        )
        print(f"\nApprox direct stock/ETF change: ${total_change:,.2f}")
    else:
        print("No overlapping direct stock/ETF holdings were found among the Frankfurt-quoted tickers.")
        print("Approx direct stock/ETF change: $0.00")
    print("Note: this Frankfurt portfolio total includes direct stock/ETF holdings only, not options.")

    if args.output:
        output_path = Path(args.output)
        report.to_csv(output_path, index=False)
        print(f"\nOutput written to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
