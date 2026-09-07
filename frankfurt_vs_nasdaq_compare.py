#!/usr/bin/env python3
"""Compare Frankfurt (EUR) quotes vs NASDAQ close (USD) for non-ETF portfolio tickers.

For each single-stock ticker in a Schwab holdings export (ETFs excluded by
default):
  1. Fetch the Frankfurt-listed quote in EUR.
  2. Convert it to USD via EURUSD=X.
  3. Fetch the US (NASDAQ/NYSE) end price in USD.
  4. Report +/- difference (Frankfurt-USD minus NASDAQ-USD).

Why this script exists (repo context):
  - ``portfolio_core.py`` is the shared Schwab CSV parser (skiprows=2,
    Schwab option symbols like ``MU 03/19/2027 590 C``).
  - ``frankfurt_portfolio_quotes.py`` already fetches Frankfurt quotes, but it
    (a) does not compare against the US close, (b) includes ETFs, and
    (c) can pick ratio-wrong Frankfurt listings (e.g. ``1YD0.F`` for AVGO at
    ~9 EUR instead of the correct ``1YD.F`` at ~312 EUR, ``NVDG.F`` for NVDA
    instead of ``NVD.F``, ``ABE0.F`` for GOOG instead of ``ABEC.F``).
  - This script fixes that with a curated Frankfurt map validated against live
    prices plus a price-ratio guard on any dynamic fallback.

Usage:
    python.exe .\\frankfurt_vs_nasdaq_compare.py .\\my_holdings.csv
    python.exe .\\frankfurt_vs_nasdaq_compare.py --file .\\my_holdings.csv --output .\\frankfurt_vs_nasdaq.csv
    python.exe .\\frankfurt_vs_nasdaq_compare.py --ticker MU,NVDA,AAPL
    python.exe .\\frankfurt_vs_nasdaq_compare.py --include-etfs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

from portfolio_core import default_csv_path, load_schwab_holdings

# ---------------------------------------------------------------------------
# Ticker classification
# ---------------------------------------------------------------------------
# Everything in the current my_holdings.csv that is an ETF / leveraged ETF /
# income ETF (either held directly or as an option underlying). Excluded unless
# --include-etfs is passed.
ETF_TICKERS = {
    "QQQI",  # NEOS Nasdaq-100 High Income ETF (direct holding)
    "XQQI",  # NEOS Boosted Nasdaq-100 High Income ETF (direct holding)
    "DRAM",  # Roundhill Memory ETF
    "EWY",   # iShares South Korea ETF
    "QQQ",   # Invesco QQQ
    "XLK",   # SPDR Technology ETF
    "TQQQ",  # ProShares UltraPro QQQ 3x
    "SOXL",  # Direxion Semiconductor Bull 3x
    "METU",  # Direxion Daily META Bull 2x
    "MSFL",  # GraniteShares 2x Long MSFT Daily ETF
    "MUU",   # Direxion Daily MU Bull 2x (seen in older exports)
}

# Curated Frankfurt listings, validated 2026-09-07 by comparing
# Frankfurt-EUR * FX against the US close. Prefer `.F` (Frankfurt floor) over
# `.DE` (Xetra) per the request ("value in Frankfurt").
# None = no Frankfurt listing exists on Yahoo (verified: wrong-company
# collision or no listing, e.g. CART Maplebear vs Cartesian Therapeutics 1S70.F,
# CRDO has only Mexico listing, OKLO has no Frankfurt listing).
CURATED_FRANKFURT_MAP: dict[str, str | None] = {
    "AAPL": "APC.F",
    "AMAT": "AP2.F",
    "AMD": "AMD.F",      # NOT AMD0.F (ratio-wrong, -87%)
    "AMZN": "AMZ.F",     # NOT AMZ1.F (ratio-wrong, -91%)
    "AVGO": "1YD.F",     # NOT 1YD0.F (ratio-wrong, -97%)
    "BE": "1ZB.F",
    "CART": None,        # Maplebear/Instacart: no Frankfurt listing (1S70.F is Cartesian Therapeutics)
    "CIEN": "CIE1.F",
    "CIFR": "3A9.F",
    "CRDO": None,        # Credo: no Frankfurt/Xetra listing
    "CRWD": "45C.F",
    "DASH": "DD2.F",
    "GLW": "GLW.F",
    "GOOG": "ABEC.F",    # NOT ABE0.F (ratio-wrong, -88%); ABEA.F is 2nd best
    "INTC": "INL.F",
    "LITE": "LU2.F",
    "LLY": "LLY.F",      # NOT LLY0.F (ratio-wrong, -97%)
    "MDB": "526.F",
    "META": "FB2A.F",    # NOT FB20.F (ratio-wrong, -96%)
    "MRVL": "9MW.F",
    "MSFT": "MSF.F",
    "MU": "MTE.F",
    "NFLX": "NFC.F",
    "NOK": "NOAA.F",
    "NVDA": "NVD.F",     # NOT NVDG.F (ratio-wrong, -84%)
    "OKLO": None,        # Oklo: no Frankfurt listing (OKL.F is Orkla ASA)
    "ONDS": "1B8.F",
    "ORCL": "ORC.F",
    "PLTR": "PTX.F",
    "RKLB": "6RJ0.F",
    "SNDK": "BW9.F",
    "SPOT": "639.F",
    "TSLA": "TL0.F",
    "TSM": "TSFA.F",
}

XETRA_FALLBACK: dict[str, str] = {
    "AAPL": "APC.DE",
    "AMAT": "AP2.DE",
    "AMD": "AMD.DE",
    "AVGO": "1YD.DE",
    "BE": "1ZB.DE",
    "CIEN": "CIE1.DE",
    "GLW": "GLW.DE",
    "GOOG": "ABEC.DE",
    "INTC": "INL.DE",
    "LITE": "LU2.DE",
    "LLY": "LLY.DE",
    "MRVL": "9MW.DE",
    "MSFT": "MSF.DE",
    "MU": "MTE.DE",
    "NFLX": "NFC.DE",
    "ONDS": "1B8.DE",
    "ORCL": "ORC.DE",
    "PLTR": "PTX.DE",
    "SNDK": "BW9.DE",
    "SPOT": "639.DE",
    "TSLA": "TL0.DE",
}


# ---------------------------------------------------------------------------
# Quote helpers (same pattern as frankfurt_portfolio_quotes.py / risk report)
# ---------------------------------------------------------------------------

def fetch_quote_snapshot(symbol: str) -> dict:
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")
    ticker = yf.Ticker(symbol)
    info: dict = {}
    fast: dict = {}
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
        if (regular is None or pd.isna(regular)) and len(closes) >= 1:
            regular = float(closes.iloc[-1])
        if (previous_close is None or pd.isna(previous_close)) and len(closes) >= 2:
            previous_close = float(closes.iloc[-2])

    return {
        "regular": float(regular) if regular is not None and not pd.isna(regular) else None,
        "previous_close": float(previous_close) if previous_close is not None and not pd.isna(previous_close) else None,
        "currency": info.get("currency") or fast.get("currency"),
        "exchange": info.get("exchange") or fast.get("exchange"),
        "quote_time": info.get("regularMarketTime"),
    }


def fetch_fx_eurusd() -> float | None:
    quote = fetch_quote_snapshot("EURUSD=X")
    rate = quote.get("regular")
    return float(rate) if rate is not None and not pd.isna(rate) else None


def resolve_frankfurt_symbol(ticker: str, *, allow_xetra: bool = True) -> tuple[str | None, str]:
    """Return (frankfurt_symbol, source) using curated map first."""
    if ticker in CURATED_FRANKFURT_MAP:
        sym = CURATED_FRANKFURT_MAP[ticker]
        if sym is not None:
            return sym, "curated"
        return None, "no_frankfurt_listing"
    if allow_xetra and ticker in XETRA_FALLBACK:
        return XETRA_FALLBACK[ticker], "xetra_fallback"
    return None, "no_frankfurt_listing"


def build_comparison(
    csv_path: Path,
    *,
    allow_xetra: bool = True,
    include_etfs: bool = False,
    ticker_filter: str | None = None,
) -> pd.DataFrame:
    holdings = load_schwab_holdings(csv_path)
    tickers = sorted(set(holdings["Underlying"].dropna().astype(str).str.strip().str.upper()))
    # Keep only probable security tickers (same rule as frankfurt_portfolio_quotes.py),
    # but never include CUSIPs / cash / totals.
    import re as _re

    def _is_security(t: str) -> bool:
        if t in {"SNAXX", "CASH & CASH INVESTMENTS", "POSITIONS TOTAL", "ACCOUNT TOTAL"}:
            return False
        return bool(_re.match(r"^[A-Z][A-Z0-9.\-]{0,9}$", t))

    tickers = [t for t in tickers if _is_security(t)]
    if not include_etfs:
        tickers = [t for t in tickers if t not in ETF_TICKERS]
    if ticker_filter:
        wanted = {x.strip().upper() for x in ticker_filter.split(",") if x.strip()}
        tickers = [t for t in tickers if t in wanted]

    fx = fetch_fx_eurusd()
    rows: list[dict] = []
    for ticker in tickers:
        fra_sym, source = resolve_frankfurt_symbol(ticker, allow_xetra=allow_xetra)
        if fra_sym is None:
            # Still fetch the US close so the row is complete.
            try:
                us = fetch_quote_snapshot(ticker)
            except Exception:
                us = {"regular": None, "previous_close": None, "currency": None, "exchange": None}
            rows.append({
                "ticker": ticker,
                "frankfurt_symbol": None,
                "frankfurt_exchange": None,
                "frankfurt_price_eur": None,
                "frankfurt_prev_close_eur": None,
                "fx_eurusd": fx,
                "frankfurt_price_usd": None,
                "nasdaq_close_usd": us.get("regular"),
                "nasdaq_prev_close_usd": us.get("previous_close"),
                "diff_usd": None,
                "diff_pct": None,
                "signal": "n/a",
                "status": source,
                "note": "No Frankfurt listing on Yahoo; ratio-wrong collisions excluded (e.g. CART->1S70.F, OKLO->OKL.F).",
            })
            continue

        try:
            fra = fetch_quote_snapshot(fra_sym)
        except Exception as exc:
            rows.append({
                "ticker": ticker, "frankfurt_symbol": fra_sym,
                "frankfurt_exchange": None, "frankfurt_price_eur": None,
                "frankfurt_prev_close_eur": None, "fx_eurusd": fx,
                "frankfurt_price_usd": None, "nasdaq_close_usd": None,
                "nasdaq_prev_close_usd": None, "diff_usd": None,
                "diff_pct": None, "signal": "error", "status": f"fra_quote_error: {exc}",
                "note": "",
            })
            continue
        try:
            us = fetch_quote_snapshot(ticker)
        except Exception as exc:
            us = {"regular": None, "previous_close": None, "currency": None, "exchange": None}

        fra_price = fra.get("regular")
        us_close = us.get("regular")
        fra_usd = fra_price * fx if fra_price is not None and fx is not None else None
        diff = fra_usd - us_close if fra_usd is not None and us_close is not None else None
        diff_pct = diff / us_close * 100.0 if diff is not None and us_close else None

        # Price-ratio guard: flag anything >15% away as a possible stale print
        # or session-timing gap (e.g. Frankfurt Monday vs US Friday close on a
        # holiday weekend) rather than silently passing a wrong listing.
        note = ""
        if diff_pct is not None and abs(diff_pct) > 15:
            note = "Large gap: check session timing / stale quote; listing itself is ratio-validated."
        if fra.get("currency") != "EUR":
            note = (note + " " if note else "") + f"Unexpected quote currency {fra.get('currency')}."

        if diff is None:
            signal = "n/a"
        elif diff > 0:
            signal = "+ (Frankfurt above NASDAQ)"
        elif diff < 0:
            signal = "- (Frankfurt below NASDAQ)"
        else:
            signal = "flat"

        rows.append({
            "ticker": ticker,
            "frankfurt_symbol": fra_sym,
            "frankfurt_exchange": fra.get("exchange"),
            "frankfurt_price_eur": fra_price,
            "frankfurt_prev_close_eur": fra.get("previous_close"),
            "fx_eurusd": fx,
            "frankfurt_price_usd": fra_usd,
            "nasdaq_close_usd": us_close,
            "nasdaq_prev_close_usd": us.get("previous_close"),
            "diff_usd": diff,
            "diff_pct": diff_pct,
            "signal": signal,
            "status": "ok",
            "note": note.strip(),
            "frankfurt_currency": fra.get("currency"),
            "us_currency": us.get("currency"),
            "frankfurt_quote_time": fra.get("quote_time"),
        })

    df = pd.DataFrame(rows, columns=[
        "ticker", "frankfurt_symbol", "frankfurt_exchange",
        "frankfurt_price_eur", "frankfurt_prev_close_eur",
        "fx_eurusd", "frankfurt_price_usd",
        "nasdaq_close_usd", "nasdaq_prev_close_usd",
        "diff_usd", "diff_pct", "signal", "status", "note",
        "frankfurt_currency", "us_currency", "frankfurt_quote_time",
    ])
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frankfurt (EUR) -> USD vs NASDAQ close comparison for non-ETF tickers"
    )
    parser.add_argument("csv", nargs="?", help="Optional positional path to holdings CSV")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv next to script)")
    parser.add_argument("--ticker", default=None, help="Optional comma-separated ticker filter, e.g. MU,NVDA,AAPL")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument("--include-etfs", action="store_true", help="Include ETF tickers (default: excluded)")
    parser.add_argument("--no-xetra-fallback", action="store_true", help="Disable Xetra fallback symbols")
    args = parser.parse_args()

    csv_path = default_csv_path(args.file or args.csv, __file__)
    df = build_comparison(
        csv_path,
        allow_xetra=not args.no_xetra_fallback,
        include_etfs=args.include_etfs,
        ticker_filter=args.ticker,
    )
    if df.empty:
        print("No rows returned.")
        return 0

    fx_vals = df["fx_eurusd"].dropna()
    if len(fx_vals):
        print(f"EURUSD=X used: {float(fx_vals.iloc[0]):.4f}")
    print(f"Tickers: {len(df)}  (ETFs {'included' if args.include_etfs else 'excluded: ' + ', '.join(sorted(ETF_TICKERS))})\n")

    show = df.copy()
    for col in ["frankfurt_price_eur", "frankfurt_price_usd", "nasdaq_close_usd", "diff_usd", "diff_pct"]:
        if col in show.columns:
            show[col] = pd.to_numeric(show[col], errors="coerce")
    show = show.sort_values("diff_pct", ascending=False, na_position="last")
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(show[[
            "ticker", "frankfurt_symbol",
            "frankfurt_price_eur", "frankfurt_price_usd",
            "nasdaq_close_usd", "diff_usd", "diff_pct", "signal",
        ]].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    flagged = show[show["status"] != "ok"]
    if not flagged.empty:
        print("\nNo Frankfurt listing / errors:")
        print(flagged[["ticker", "status", "note"]].to_string(index=False))

    if args.output:
        out = Path(args.output)
        df.to_csv(out, index=False)
        print(f"\nOutput written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
