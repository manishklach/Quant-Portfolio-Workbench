#!/usr/bin/env python3
"""Estimate after-hours / overnight portfolio P/L from a Schwab holdings export.

For stocks/ETFs:
  P/L = shares * (after_hours_price - regular_close_price)

For options:
  - infer implied volatility from the regular-session option mark
  - reprice the option using the after-hours / overnight underlying price
  - if IV inference fails, fall back to intrinsic-change approximation
"""

from __future__ import annotations

import argparse
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from datetime import datetime, time, timedelta, timezone
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

from portfolio_core import clean_numeric, default_csv_path, load_schwab_holdings


RISK_FREE_RATE = 0.045
DIVIDEND_YIELD = 0.0
UTC = timezone.utc
EASTERN_OFFSET = timezone(timedelta(hours=-4))
COINBASE_PRODUCTS_URL = "https://api.coinbase.com/api/v3/brokerage/market/products"
HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
DRAM_HOLDINGS_URL = "https://stockanalysis.com/etf/dram/holdings/"
ROBINHOOD_QUOTE_URLS = (
    "https://robinhood.com/us/en/stocks/{ticker}/",
    "https://robinhood.com/us/en/etfs/{ticker}/",
)
DIRECT_PERP_CONFIG = {
    "DRAM": {"source": "hyperliquid", "symbol": "DRAM", "dex": "xyz"},
}
ETF_PROXY_CONFIG = {
    "DRAM": {
        "holdings_url": DRAM_HOLDINGS_URL,
        "source_label": "etf_holdings_proxy_partial",
    },
}
DRAM_PROXY_MAP = {
    "MICRON": "MU",
    "SANDISK": "SNDK",
    "SK HYNIX": "000660.KS",
    "SAMSUNG ELECTRONICS": "005930.KS",
    "KIOXIA": "285A.T",
    "SEAGATE": "STX",
    "WESTERN DIGITAL": "WDC",
    "NANYA": "2408.TW",
    "WINBOND": "2344.TW",
    "GIGADEVICE": "603986.SS",
}
LEVERAGED_PROXY_MAP = {
    "MUU": {"underlying": "MU", "leverage": 2.0, "label": "Direxion Daily MU Bull 2X ETF"},
    "QQQI": {"underlying": "QQQ", "leverage": 1.0, "label": "QQQ Income ETF proxy", "max_direct_ah_proxy_gap": 0.0075},
    "XQQI": {"underlying": "QQQ", "leverage": 1.0, "label": "QQQ Income ETF proxy", "max_direct_ah_proxy_gap": 0.0075},
}

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
POST_MARKET_CLOSE = time(20, 0)
OVERNIGHT_END = time(4, 0)


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


def classify_extended_session(timestamp_text: str | None) -> str | None:
    if not timestamp_text:
        return None
    try:
        dt = datetime.fromisoformat(str(timestamp_text).replace("Z", "+00:00")).astimezone(EASTERN_OFFSET)
    except Exception:
        return None

    ts_time = dt.timetz().replace(tzinfo=None)
    if REGULAR_CLOSE <= ts_time < POST_MARKET_CLOSE:
        return "post_market"
    if ts_time >= POST_MARKET_CLOSE or ts_time < OVERNIGHT_END:
        return "overnight"
    if OVERNIGHT_END <= ts_time < REGULAR_OPEN:
        return "pre_market"
    return None


def find_robinhood_quote_payload(node: object, ticker: str) -> dict[str, object] | None:
    if isinstance(node, dict):
        symbol = str(node.get("symbol") or "").strip().upper()
        if (
            symbol == ticker
            and (
                "last_non_reg_trade_price" in node
                or "last_extended_hours_trade_price" in node
                or "previous_close" in node
            )
        ):
            return node
        for value in node.values():
            found = find_robinhood_quote_payload(value, ticker)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = find_robinhood_quote_payload(item, ticker)
            if found is not None:
                return found
    return None


def fetch_robinhood_quote_snapshot(ticker: str) -> dict[str, float | str | None]:
    if requests is None:
        return {"regular": None, "previous_close": None, "non_regular": None, "session": None, "source": None}

    quote = None
    for url_template in ROBINHOOD_QUOTE_URLS:
        try:
            response = requests.get(
                url_template.format(ticker=ticker),
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=20,
            )
            response.raise_for_status()
        except Exception:
            continue

        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text, re.S)
        if not match:
            continue

        try:
            payload = json.loads(match.group(1))
        except Exception:
            continue

        quote = find_robinhood_quote_payload(payload, ticker)
        if quote:
            break

    if not quote:
        return {"regular": None, "previous_close": None, "non_regular": None, "session": None, "source": None}

    non_regular = pd.to_numeric(
        quote.get("last_non_reg_trade_price", quote.get("last_extended_hours_trade_price")),
        errors="coerce",
    )
    regular = pd.to_numeric(quote.get("last_trade_price"), errors="coerce")
    previous_close = pd.to_numeric(
        quote.get("adjusted_previous_close", quote.get("previous_close")),
        errors="coerce",
    )
    timestamp_text = str(
        quote.get("venue_last_non_reg_trade_time")
        or quote.get("updated_at")
        or ""
    ).strip()
    session = classify_extended_session(timestamp_text)

    return {
        "regular": float(regular) if not pd.isna(regular) else None,
        "previous_close": float(previous_close) if not pd.isna(previous_close) else None,
        "non_regular": float(non_regular) if not pd.isna(non_regular) else None,
        "session": session,
        "source": f"robinhood_{session}" if session else None,
        "timestamp": timestamp_text or None,
    }


_BATCH_QUOTES: dict[str, dict[str, float | str | None]] = {}
_ETF_HOLDINGS_CACHE: dict[str, pd.DataFrame] = {}
_USE_ROBINHOOD = True


def set_use_robinhood(val: bool):
    global _USE_ROBINHOOD
    _USE_ROBINHOOD = val


def set_batch_quotes(quotes: dict[str, dict[str, float | str | None]]):
    _BATCH_QUOTES.clear()
    _BATCH_QUOTES.update(quotes)


def build_batch_quote_cache(tickers: list[str]) -> dict[str, dict[str, float | str | None]]:
    if yf is None:
        return {}

    valid_tickers = [ticker for ticker in tickers if re.match(r"^[A-Z][A-Z0-9.\-]*$", ticker)]
    if not valid_tickers:
        return {}

    try:
        hist = yf.download(
            tickers=valid_tickers,
            period="2d",
            interval="5m",
            prepost=True,
            progress=False,
            auto_adjust=False,
            threads=True,
            group_by="column",
        )
    except Exception:
        return {}

    if hist.empty or "Close" not in hist.columns:
        return {}

    close_df = hist["Close"]
    if isinstance(close_df, pd.Series):
        close_df = close_df.to_frame(name=valid_tickers[0])

    out: dict[str, dict[str, float | str | None]] = {}
    for ticker in close_df.columns:
        series = close_df[ticker].dropna()
        if series.empty:
            continue

        try:
            eastern_index = series.index.tz_convert(EASTERN_OFFSET)
        except Exception:
            eastern_index = series.index

        frame = pd.DataFrame({"price": series.to_numpy()}, index=eastern_index)
        regular_mask = frame.index.indexer_between_time("09:30", "16:00")
        regular_frame = frame.iloc[regular_mask] if len(regular_mask) else frame.iloc[0:0]
        extended_frame = frame.drop(regular_frame.index, errors="ignore")

        regular = float(regular_frame["price"].iloc[-1]) if not regular_frame.empty else None
        previous_close = None
        if regular is not None:
            prior_regular_frame = regular_frame[regular_frame.index.date < regular_frame.index[-1].date()]
            if not prior_regular_frame.empty:
                previous_close = float(prior_regular_frame["price"].iloc[-1])

        post = None
        post_source = None
        extended_hours_session = None
        if not extended_frame.empty:
            latest_extended_ts = extended_frame.index[-1]
            latest_extended_price = float(extended_frame["price"].iloc[-1])
            ts_time = latest_extended_ts.timetz().replace(tzinfo=None)
            post = latest_extended_price
            if REGULAR_CLOSE <= ts_time < POST_MARKET_CLOSE or ts_time >= POST_MARKET_CLOSE:
                post_source = "yahoo_post"
                extended_hours_session = "post_market"
            elif OVERNIGHT_END <= ts_time < REGULAR_OPEN:
                post_source = "yahoo_pre"
                extended_hours_session = "overnight_pre_market"
            else:
                post_source = "yahoo_post"
                extended_hours_session = "post_market"

        out[str(ticker).upper()] = {
            "regular": regular,
            "previous_close": previous_close,
            "post": post,
            "post_source": post_source,
            "extended_hours_session": extended_hours_session,
            "exchange": None,
        }
    return out


def fetch_quote_snapshot(ticker: str, *, allow_numeric_symbol: bool = False) -> dict[str, float | str | None]:
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")
    if allow_numeric_symbol:
        is_valid = bool(re.match(r"^[A-Z0-9.\-]+$", ticker))
    else:
        is_valid = bool(re.match(r"^[A-Z][A-Z0-9.\-]*$", ticker))
    if not is_valid:
        return {"regular": None, "previous_close": None, "post": None, "post_source": None, "extended_hours_session": None, "exchange": None}

    batch_quote = _BATCH_QUOTES.get(ticker)
    if batch_quote is not None:
        regular = batch_quote.get("regular")
        previous_close = batch_quote.get("previous_close")
        post = batch_quote.get("post")
        post_source = batch_quote.get("post_source")
        extended_hours_session = batch_quote.get("extended_hours_session")
        exchange = batch_quote.get("exchange")
        tk_info_needed = False
    else:
        regular = None
        previous_close = None
        post = None
        post_source = None
        extended_hours_session = None
        exchange = None
        tk_info_needed = True

    if tk_info_needed:
        tk = yf.Ticker(ticker)
        try:
            info = tk.info or {}
        except Exception:
            info = {}
        try:
            fast = dict(tk.fast_info)
        except Exception:
            fast = {}

        if regular is None:
            regular = info.get("regularMarketPrice")
            if regular is None or pd.isna(regular):
                regular = fast.get("lastPrice")
        if previous_close is None:
            previous_close = info.get("regularMarketPreviousClose")
            if previous_close is None or pd.isna(previous_close):
                previous_close = fast.get("regularMarketPreviousClose")
        post = info.get("postMarketPrice")
        post_source = "yahoo_post"
        if post is None or pd.isna(post):
            post = fast.get("postMarketPrice")
            post_source = "yahoo_post"
        if post is None or pd.isna(post):
            post = info.get("preMarketPrice")
            post_source = "yahoo_pre"
        if post is None or pd.isna(post):
            post = fast.get("preMarketPrice")
            post_source = "yahoo_pre"
        if post is None or pd.isna(post):
            post_source = None
        if post_source == "yahoo_post":
            extended_hours_session = "post_market"
        elif post_source == "yahoo_pre":
            extended_hours_session = "overnight_pre_market"
        exchange = info.get("exchange") or fast.get("exchange")
    else:
        post = batch_quote.get("post") if batch_quote else None
        post_source = batch_quote.get("post_source") if batch_quote else None
        extended_hours_session = batch_quote.get("extended_hours_session") if batch_quote else None
        if post is None or pd.isna(post) or post_source is None:
            tk = yf.Ticker(ticker)
            try:
                fast = dict(tk.fast_info)
            except Exception:
                fast = {}
            post = fast.get("postMarketPrice")
            post_source = "yahoo_post"
            if post is None or pd.isna(post):
                post = fast.get("preMarketPrice")
                post_source = "yahoo_pre"
            if post is None or pd.isna(post):
                post_source = None
            if post_source == "yahoo_post":
                extended_hours_session = "post_market"
            elif post_source == "yahoo_pre":
                extended_hours_session = "overnight_pre_market"

    robinhood_quote = fetch_robinhood_quote_snapshot(ticker) if _USE_ROBINHOOD else {}

    return {
        "regular": float(regular) if regular is not None and not pd.isna(regular) else None,
        "previous_close": float(previous_close) if previous_close is not None and not pd.isna(previous_close) else None,
        "post": float(post) if post is not None and not pd.isna(post) else None,
        "post_source": post_source,
        "extended_hours_session": extended_hours_session,
        "exchange": exchange,
        "overnight": robinhood_quote.get("non_regular"),
        "overnight_source": robinhood_quote.get("source"),
        "overnight_timestamp": robinhood_quote.get("timestamp"),
    }


def choose_after_hours_price(
    ticker: str,
    quote: dict[str, float | str | None],
    perp_quote: dict[str, float | str | None],
    *,
    prefer_perp: bool = False,
    overnight: bool = False,
) -> tuple[float | None, str | None]:
    if prefer_perp and perp_quote.get("synthetic_price") is not None:
        return perp_quote["synthetic_price"], perp_quote.get("price_source")

    if overnight:
        overnight_price = quote.get("overnight")
        overnight_source = quote.get("overnight_source")
        if overnight_price is not None:
            return overnight_price, overnight_source

    after_hours_price = quote.get("post")
    after_hours_source = quote.get("post_source")
    if overnight and not prefer_perp:
        return after_hours_price, after_hours_source
    if after_hours_price is None and perp_quote.get("synthetic_price") is not None:
        return perp_quote["synthetic_price"], perp_quote.get("price_source")
    return after_hours_price, after_hours_source


def fetch_hyperliquid_perp_snapshots(dex: str = "") -> dict[str, dict[str, float | str | None]]:
    if requests is None:
        return {}

    try:
        response = requests.post(
            HYPERLIQUID_INFO_URL,
            json={"type": "allMids", "dex": dex},
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return {}

    out: dict[str, dict[str, float | str | None]] = {}
    for symbol, mid in payload.items():
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            continue

        mid_price = pd.to_numeric(mid, errors="coerce")
        synthetic_price = mid_price
        price_source = "hyperliquid_mid"

        out[symbol] = {
            "ticker": symbol,
            "symbol": symbol,
            "synthetic_price": float(synthetic_price) if not pd.isna(synthetic_price) else None,
            "mid_price": float(mid_price) if not pd.isna(mid_price) else None,
            "price_source": price_source if not pd.isna(synthetic_price) else None,
            "dex": dex,
        }
    return out


def choose_direct_perp_price(
    ticker: str,
    direct_perp_cache: dict[str, dict[str, float | str | None]],
) -> tuple[float | None, str | None]:
    config = DIRECT_PERP_CONFIG.get(ticker, {})
    source = str(config.get("source") or "").strip().lower()
    symbol = str(config.get("symbol") or ticker).strip().upper()
    dex = str(config.get("dex") or "").strip().lower()
    if source != "hyperliquid":
        return None, None

    cache_key = f"{dex}:{symbol}".upper() if dex else symbol
    perp_quote = direct_perp_cache.get(cache_key, {})
    price = perp_quote.get("synthetic_price")
    source_label = perp_quote.get("price_source")
    return (price, source_label) if price is not None else (None, None)


def fetch_etf_holdings(holdings_url: str) -> pd.DataFrame:
    if holdings_url in _ETF_HOLDINGS_CACHE:
        return _ETF_HOLDINGS_CACHE[holdings_url].copy()
    if requests is None:
        return pd.DataFrame()
    try:
        html = requests.get(holdings_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}).text
        tables = pd.read_html(StringIO(html))
    except Exception:
        return pd.DataFrame()

    if not tables:
        return pd.DataFrame()
    holdings = tables[0].copy()
    expected = {"Symbol", "Name", "% Weight"}
    if not expected.issubset(set(holdings.columns)):
        return pd.DataFrame()

    holdings["weight_pct"] = pd.to_numeric(
        holdings["% Weight"].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )
    holdings["name_upper"] = holdings["Name"].astype(str).str.upper()
    _ETF_HOLDINGS_CACHE[holdings_url] = holdings.copy()
    return holdings


def map_dram_component_ticker(symbol: str, name: str) -> str | None:
    symbol_text = str(symbol).strip().upper()
    name_text = str(name).strip().upper()

    if symbol_text in {"MU", "SNDK", "STX", "WDC"}:
        return symbol_text

    for key, proxy in DRAM_PROXY_MAP.items():
        if key in name_text:
            return proxy
    return None


def build_etf_proxy_snapshot(
    ticker: str,
    perp_cache: dict[str, dict[str, float | str | None]],
    prefer_perp: bool,
    quote_cache: dict[str, dict[str, float | str | None]],
    overnight: bool = False,
) -> dict[str, float | str | None]:
    config = ETF_PROXY_CONFIG.get(ticker, {})
    holdings_url = str(config.get("holdings_url") or "").strip()
    if not holdings_url:
        return {}

    holdings = fetch_etf_holdings(holdings_url)
    if holdings.empty:
        return {}

    rows: list[dict[str, object]] = []
    weighted_return = 0.0
    covered_weight = 0.0

    for _, row in holdings.iterrows():
        if ticker == "DRAM":
            proxy_ticker = map_dram_component_ticker(row.get("Symbol"), row.get("Name"))
        else:
            proxy_ticker = None
        weight_pct = pd.to_numeric(row.get("weight_pct"), errors="coerce")
        if proxy_ticker is None or pd.isna(weight_pct) or weight_pct <= 0:
            continue

        if proxy_ticker not in quote_cache:
            quote_cache[proxy_ticker] = fetch_quote_snapshot(
                proxy_ticker,
                allow_numeric_symbol=("." in proxy_ticker and proxy_ticker[0].isdigit()),
            )
        quote = quote_cache[proxy_ticker]
        perp_quote = perp_cache.get(proxy_ticker, {})
        regular = quote.get("regular")
        after_hours_price, after_hours_source = choose_after_hours_price(
            proxy_ticker,
            quote,
            perp_quote,
            prefer_perp=prefer_perp,
            overnight=overnight,
        )
        if regular is None or after_hours_price is None or regular == 0:
            continue

        component_return = (after_hours_price - regular) / regular
        weight_frac = float(weight_pct) / 100.0
        weighted_return += weight_frac * component_return
        covered_weight += weight_frac
        rows.append(
            {
                "component": proxy_ticker,
                "weight_pct": float(weight_pct),
                "regular": regular,
                "after_hours": after_hours_price,
                "after_hours_source": after_hours_source,
                "component_return_pct": component_return * 100.0,
            }
        )

    if not rows:
        return {}

    return {
        "weighted_return": weighted_return,
        "covered_weight": covered_weight,
        "components_used": rows,
        "source": str(config.get("source_label") or "etf_holdings_proxy_partial"),
    }


def build_leveraged_proxy_snapshot(
    ticker: str,
    perp_cache: dict[str, dict[str, float | str | None]],
    prefer_perp: bool,
    quote_cache: dict[str, dict[str, float | str | None]],
    overnight: bool = False,
) -> dict[str, float | str | None]:
    proxy_meta = LEVERAGED_PROXY_MAP.get(ticker)
    if not proxy_meta:
        return {}

    underlying = str(proxy_meta["underlying"]).upper()
    leverage = float(proxy_meta["leverage"])
    if underlying not in quote_cache:
        quote_cache[underlying] = fetch_quote_snapshot(underlying)
    quote = quote_cache[underlying]
    perp_quote = perp_cache.get(underlying, {})
    regular = quote.get("regular")
    after_hours_price, after_hours_source = choose_after_hours_price(
        underlying,
        quote,
        perp_quote,
        prefer_perp=prefer_perp,
        overnight=overnight,
    )
    if regular is None or after_hours_price is None or regular == 0:
        return {}

    underlying_return = (after_hours_price - regular) / regular
    leveraged_return = leverage * underlying_return
    return {
        "ticker": ticker,
        "underlying": underlying,
        "leverage": leverage,
        "regular_underlying": regular,
        "after_hours_underlying": after_hours_price,
        "underlying_return": underlying_return,
        "leveraged_return": leveraged_return,
        "source": f"leveraged_proxy_{underlying}",
        "underlying_source": after_hours_source,
        "label": proxy_meta.get("label"),
    }


def fetch_coinbase_equity_perp_snapshots() -> dict[str, dict[str, float | str | None]]:
    if requests is None:
        return {}

    params = {
        "product_type": "FUTURE",
        "contract_expiry_type": "PERPETUAL",
        "futures_underlying_type": "FUTURES_UNDERLYING_TYPE_EQUITY",
        "limit": 500,
    }
    try:
        response = requests.get(COINBASE_PRODUCTS_URL, params=params, timeout=20)
        response.raise_for_status()
        products = response.json().get("products", [])
    except Exception:
        return {}

    out: dict[str, dict[str, float | str | None]] = {}
    for product in products:
        details = product.get("future_product_details") or {}
        ticker = (details.get("contract_code") or "").strip().upper()
        if not ticker:
            continue

        index_price = pd.to_numeric(details.get("index_price"), errors="coerce")
        last_price = pd.to_numeric(product.get("price"), errors="coerce")
        mid_price = pd.to_numeric(product.get("mid_market_price"), errors="coerce")

        synthetic_price = index_price
        price_source = "coinbase_perp_index"
        if pd.isna(synthetic_price):
            synthetic_price = last_price
            price_source = "coinbase_perp_last"
        if pd.isna(synthetic_price):
            synthetic_price = mid_price
            price_source = "coinbase_perp_mid"

        out[ticker] = {
            "ticker": ticker,
            "product_id": product.get("product_id"),
            "display_name": product.get("display_name"),
            "synthetic_price": float(synthetic_price) if not pd.isna(synthetic_price) else None,
            "index_price": float(index_price) if not pd.isna(index_price) else None,
            "last_price": float(last_price) if not pd.isna(last_price) else None,
            "mid_price": float(mid_price) if not pd.isna(mid_price) else None,
            "price_source": price_source if not pd.isna(synthetic_price) else None,
        }
    return out


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


def build_after_hours_report(
    csv_path: Path,
    prefer_perp: bool = False,
    prefer_etf_proxy: bool = False,
    overnight: bool = False,
    use_robinhood: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdings = load_schwab_holdings(csv_path)
    set_use_robinhood(use_robinhood)

    unique_tickers = {str(ticker).strip().upper() for ticker in holdings["Underlying"].dropna().astype(str)}
    unique_tickers.update(
        str(meta.get("underlying") or "").strip().upper()
        for meta in LEVERAGED_PROXY_MAP.values()
        if str(meta.get("underlying") or "").strip()
    )
    if "DRAM" in unique_tickers:
        unique_tickers.update({"MU", "SNDK", "STX", "WDC", "000660.KS", "005930.KS", "285A.T", "2408.TW", "2344.TW", "603986.SS"})
    batch_quotes = build_batch_quote_cache(sorted(unique_tickers))
    if batch_quotes:
        set_batch_quotes(batch_quotes)

    quote_cache: dict[str, dict[str, float | str | None]] = {}
    perp_cache = fetch_coinbase_equity_perp_snapshots()
    direct_perp_cache: dict[str, dict[str, float | str | None]] = {}
    for config in DIRECT_PERP_CONFIG.values():
        if str(config.get("source") or "").strip().lower() == "hyperliquid":
            dex = str(config.get("dex") or "").strip().lower()
            if dex:
                direct_perp_cache.update(fetch_hyperliquid_perp_snapshots(dex=dex))
    etf_proxy_cache: dict[str, dict[str, float | str | None]] = {}
    leveraged_proxy_cache: dict[str, dict[str, float | str | None]] = {}
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
                quote_cache[ticker] = {"regular": None, "previous_close": None, "post": None, "post_source": None, "extended_hours_session": None, "exchange": None}
        quote = quote_cache[ticker]
        perp_quote = perp_cache.get(ticker, {})

        asset_type = str(row.get("Asset Type Normalized", "")).strip()
        is_option = bool(row.get("Is Option"))
        qty = float(row.get("Qty", 0.0))
        multiplier = 100.0 if is_option else 1.0

        regular_price = quote["regular"]
        after_hours_price, after_hours_source = choose_after_hours_price(
            ticker,
            quote,
            perp_quote,
            prefer_perp=prefer_perp,
            overnight=overnight,
        )
        if ticker in DIRECT_PERP_CONFIG and regular_price is not None:
            direct_perp_price, direct_perp_source = choose_direct_perp_price(ticker, direct_perp_cache)
            if direct_perp_price is not None and (prefer_perp or after_hours_price is None):
                after_hours_price = direct_perp_price
                after_hours_source = direct_perp_source
        if ticker in LEVERAGED_PROXY_MAP and regular_price is not None:
            if ticker not in leveraged_proxy_cache:
                leveraged_proxy_cache[ticker] = build_leveraged_proxy_snapshot(
                    ticker,
                    perp_cache,
                    prefer_perp,
                    quote_cache,
                    overnight=overnight,
                )
            leveraged_proxy = leveraged_proxy_cache.get(ticker, {})
            if leveraged_proxy:
                proxy_return = float(leveraged_proxy["leveraged_return"])
                direct_return = None
                if after_hours_price is not None and regular_price not in (None, 0):
                    direct_return = (after_hours_price - regular_price) / regular_price

                proxy_meta = LEVERAGED_PROXY_MAP.get(ticker, {})
                max_gap = proxy_meta.get("max_direct_ah_proxy_gap")
                use_proxy = prefer_perp or after_hours_price is None
                if (
                    not use_proxy
                    and direct_return is not None
                    and max_gap is not None
                    and abs(float(direct_return) - proxy_return) > float(max_gap)
                ):
                    use_proxy = True

                if use_proxy:
                    after_hours_price = regular_price * (1.0 + proxy_return)
                    after_hours_source = str(leveraged_proxy["source"])
        if ticker in ETF_PROXY_CONFIG and regular_price is not None:
            if ticker not in etf_proxy_cache:
                etf_proxy_cache[ticker] = build_etf_proxy_snapshot(
                    ticker,
                    perp_cache,
                    prefer_perp,
                    quote_cache,
                    overnight=overnight,
                )
            etf_proxy = etf_proxy_cache.get(ticker, {})
            if etf_proxy and (prefer_etf_proxy or after_hours_price is None):
                after_hours_price = regular_price * (1.0 + float(etf_proxy["weighted_return"]))
                after_hours_source = str(etf_proxy["source"])
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
            pricing_method = "extended_hours"
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
                "after_hours_source": after_hours_source,
                "extended_hours_session": (
                    "perp"
                    if isinstance(after_hours_source, str) and after_hours_source.startswith("coinbase_perp")
                    else "overnight"
                    if isinstance(after_hours_source, str) and after_hours_source == "robinhood_overnight"
                    else "post_market"
                    if isinstance(after_hours_source, str) and after_hours_source == "robinhood_post_market"
                    else "pre_market"
                    if isinstance(after_hours_source, str) and after_hours_source == "robinhood_pre_market"
                    else quote.get("extended_hours_session")
                ),
                "perp_symbol": perp_quote.get("product_id"),
                "perp_index_price": perp_quote.get("index_price"),
                "perp_last_price": perp_quote.get("last_price"),
                "leveraged_proxy_underlying": leveraged_proxy_cache.get(ticker, {}).get("underlying") if ticker in LEVERAGED_PROXY_MAP else None,
                "leveraged_proxy_leverage": leveraged_proxy_cache.get(ticker, {}).get("leverage") if ticker in LEVERAGED_PROXY_MAP else None,
                "etf_proxy_coverage_pct": float(etf_proxy_cache.get(ticker, {}).get("covered_weight", 0.0)) * 100.0 if ticker in ETF_PROXY_CONFIG and etf_proxy_cache.get(ticker) else None,
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
    parser = argparse.ArgumentParser(description="Estimate after-hours / overnight portfolio P/L from my_holdings.csv")
    parser.add_argument("csv", nargs="?", help="Optional positional path to holdings CSV")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv next to script)")
    parser.add_argument("--output", default=None, help="Optional CSV path for position-level output")
    parser.add_argument("--list-perps", action="store_true", help="Print held tickers that have supported perp-based pricing sources")
    parser.add_argument("--prefer-perp", action="store_true", help="Prefer Coinbase equity perpetual prices over Yahoo post/pre-market when available")
    parser.add_argument("--prefer-etf-proxy", action="store_true", help="Prefer ETF basket proxy pricing for supported ETFs like DRAM")
    parser.add_argument("--overnight", action="store_true", help="Use overnight/pre-market quotes only from Yahoo when available; separate from the default post-market path.")
    parser.add_argument("--robinhood", action="store_true", help="Enable Robinhood 24h quote scraping (slow, off by default)")
    args = parser.parse_args()

    csv_path = default_csv_path(args.file or args.csv, __file__)
    if args.list_perps:
        holdings = load_schwab_holdings(csv_path)
        held = sorted(set(holdings["Underlying"].dropna().astype(str).str.upper()))
        coinbase_perps = fetch_coinbase_equity_perp_snapshots()
        hyperliquid_caches: dict[str, dict[str, dict[str, float | str | None]]] = {}
        quote_cache: dict[str, dict[str, float | str | None]] = {}
        proxy_underlyings = {
            str(meta.get("underlying") or "").strip().upper()
            for meta in LEVERAGED_PROXY_MAP.values()
            if str(meta.get("underlying") or "").strip()
        }
        rows = []
        for ticker in held:
            perp = coinbase_perps.get(ticker)
            if perp:
                rows.append(
                    {
                        "ticker": ticker,
                        "perp_source": "coinbase",
                        "perp_symbol": perp.get("product_id"),
                        "price_source": perp.get("price_source"),
                        "perp_price_used": perp.get("synthetic_price"),
                    }
                )
                continue

            direct_cfg = DIRECT_PERP_CONFIG.get(ticker, {})
            if str(direct_cfg.get("source") or "").strip().lower() == "hyperliquid":
                dex = str(direct_cfg.get("dex") or "").strip().lower()
                symbol = str(direct_cfg.get("symbol") or ticker).strip().upper()
                if dex and dex not in hyperliquid_caches:
                    hyperliquid_caches[dex] = fetch_hyperliquid_perp_snapshots(dex=dex)
                cache_key = f"{dex}:{symbol}".upper() if dex else symbol
                perp = hyperliquid_caches.get(dex, {}).get(cache_key)
                if perp:
                    rows.append(
                        {
                            "ticker": ticker,
                            "perp_source": "hyperliquid",
                            "perp_symbol": cache_key,
                            "price_source": perp.get("price_source"),
                            "perp_price_used": perp.get("synthetic_price"),
                        }
                    )
                    continue

            if ticker in proxy_underlyings:
                if ticker not in quote_cache:
                    try:
                        quote_cache[ticker] = fetch_quote_snapshot(ticker)
                    except Exception:
                        quote_cache[ticker] = {"regular": None, "previous_close": None, "post": None, "exchange": None}

                anchor_quote = quote_cache.get(ticker, {})
                anchor_after_hours_price, anchor_price_source = choose_after_hours_price(
                    ticker,
                    anchor_quote,
                    coinbase_perps.get(ticker, {}),
                    prefer_perp=True,
                )
                if anchor_after_hours_price is not None:
                    rows.append(
                        {
                            "ticker": ticker,
                            "perp_source": "proxy_anchor",
                            "perp_symbol": ticker,
                            "price_source": anchor_price_source or "after_hours_quote",
                            "perp_price_used": anchor_after_hours_price,
                        }
                    )
                    continue

            proxy_meta = LEVERAGED_PROXY_MAP.get(ticker)
            if proxy_meta:
                underlying = str(proxy_meta.get("underlying") or "").strip().upper()
                if underlying:
                    if underlying not in quote_cache:
                        try:
                            quote_cache[underlying] = fetch_quote_snapshot(underlying)
                        except Exception:
                            quote_cache[underlying] = {"regular": None, "previous_close": None, "post": None, "post_source": None, "extended_hours_session": None, "exchange": None}

                    underlying_quote = quote_cache.get(underlying, {})
                    underlying_perp = coinbase_perps.get(underlying, {})
                    proxy_after_hours_price, proxy_price_source = choose_after_hours_price(
                        underlying,
                        underlying_quote,
                        underlying_perp,
                        prefer_perp=True,
                    )
                    if proxy_after_hours_price is not None:
                        rows.append(
                            {
                                "ticker": ticker,
                                "perp_source": "proxy",
                                "perp_symbol": underlying,
                                "price_source": proxy_price_source or f"proxy_{underlying}",
                                "perp_price_used": proxy_after_hours_price,
                            }
                        )
        out = pd.DataFrame(rows)
        if out.empty:
            print("No held tickers currently match the supported perp-based pricing sources.")
        else:
            print(out.sort_values("ticker").to_string(index=False))
        return 0

    positions, by_ticker = build_after_hours_report(
        csv_path,
        prefer_perp=args.prefer_perp,
        prefer_etf_proxy=args.prefer_etf_proxy,
        overnight=args.overnight,
        use_robinhood=args.robinhood,
    )

    if args.overnight:
        non_option_positions = positions[~positions["is_option"]].copy()
        if not non_option_positions.empty:
            overnight_count = int((non_option_positions["extended_hours_session"] == "overnight").sum())
            premarket_count = int(
                non_option_positions["extended_hours_session"].isin(["pre_market", "overnight_pre_market"]).sum()
            )
            post_market_count = int((non_option_positions["extended_hours_session"] == "post_market").sum())
            perp_count = int((non_option_positions["extended_hours_session"] == "perp").sum())
            missing_count = int(non_option_positions["after_hours_underlying"].isna().sum())
            print("\nOvernight Session Check")
            print(f"Robinhood/Yahoo quotes tagged as overnight: {overnight_count}")
            print(f"Robinhood/Yahoo quotes tagged as pre-market: {premarket_count}")
            print(f"Robinhood/Yahoo quotes tagged as post-market: {post_market_count}")
            print(f"Perp prices used via explicit --prefer-perp: {perp_count}")
            print(f"Positions with no usable overnight quote: {missing_count}")
            if overnight_count == 0 and post_market_count > 0 and not args.prefer_perp and premarket_count == 0:
                print(
                    "No true overnight prints were available yet for the current run. "
                    "The script used the latest Robinhood 24-hour market quote path, which is currently still tagged as post-market."
                )
            elif overnight_count == 0 and premarket_count == 0 and missing_count > 0:
                print(
                    "No usable Robinhood/Yahoo overnight quotes were available for the current run. "
                    "If you want fallback pricing, rerun with --overnight --prefer-perp where supported."
                )

    total_market_value = positions["current_market_value"].sum()
    total_ah_pl = positions["estimated_ah_pl"].sum()

    print("\nAfter-Hours / Overnight Portfolio Estimate")
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
