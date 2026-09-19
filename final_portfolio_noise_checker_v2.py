#!/usr/bin/env python3
"""
final_portfolio_noise_checker_v2.py

Fix vs prior version:
- Does NOT treat Schwab "Price" as "Strike Price".
- Strike and expiration are parsed from the option symbol/description directly.
- This fixes bogus spreads like 590.495/647.361 C, which were actually option prices,
  not strikes.

Final rules:

CALL SPREADS:
  Flag only:
    - long lower-strike call + short higher-strike call
    - same ticker + same expiration
    - spread width (short - long) <= $10
    - stock price > short call strike
    - stock day change > 0 OR previous close > short call strike
    - net spread day P/L < 0
  Addback is an intrinsic-based scenario adjustment, not verified pricing error.

PUTS:
  NAKED short puts:
    - uncovered short puts only
    - stock price > put strike
  Expected P/L = abs(put_delta) * stock_change * contracts * 100

PUT SPREADS:
  Flag only:
    - short higher-strike put + long lower-strike put
    - same ticker + same expiration
    - stock price > short put strike
    - stock day change > 0
    - net spread day P/L < net-delta expectation

OPTION QUOTE-BASELINE AUDIT:
  - checks every call and put, including standalone and spread legs
  - compares Schwab day P/L with Cboe midpoint-vs-prior-close P/L
  - corroborates the discrepancy with aggregate Cboe delta exposure
  - flags a ticker only when Schwab is materially more negative than both checks

Delta:
  - uses CSV/broker Delta column if present
  - otherwise falls back to yfinance option-chain IV + Black-Scholes put delta

Run:
  pip install pandas yfinance numpy openpyxl
  python final_portfolio_noise_checker_v2.py my_holdings.csv
"""

import argparse
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from option_math import bs_put_delta, implied_volatility, normalize_put_delta
from portfolio_core import config_value

try:
    import yfinance as yf
except ImportError:
    yf = None


_RISK_FREE_RATE = float(config_value("market.risk_free_rate", 0.045))
_DIVIDEND_YIELD = float(config_value("market.dividend_yield", 0.0))


def read_broker_csv(path):
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(lines[:80]):
        low = line.lower()
        if "symbol" in low and "description" in low and ("qty" in low or "quantity" in low):
            header_idx = i
            break
    return pd.read_csv(path, skiprows=header_idx) if header_idx is not None else pd.read_csv(path)


def normalize_col(c):
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def find_col_exact_or_contains(df, candidates, required=True):
    """
    Safe-ish column finder for non-strike fields.
    Exact normalized match first, then candidate-in-column only.
    It intentionally avoids column-in-candidate because that caused Price to match Strike Price.
    """
    norm_map = {normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_col(cand)
        if key in norm_map:
            return norm_map[key]
    for cand in candidates:
        key = normalize_col(cand)
        for k, original in norm_map.items():
            if key and key in k:
                # Never cross-match strike and non-strike columns in the
                # substring fallback (e.g. "Price" must not resolve to
                # "Strike Price" when no plain Price column exists).
                if ("strike" in k) != ("strike" in key):
                    continue
                return original
    if required:
        raise ValueError(f"Could not find required column. Tried {candidates}\nAvailable: {list(df.columns)}")
    return None


def num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s in {"", "--", "N/A", "nan", "None"}:
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "").strip()
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return np.nan


def parse_option_fields(symbol, description):
    """
    Schwab examples:
      Symbol:      MU 03/19/2027 590.00 C
      Description: CALL MICRON TECHNOLOGY I$590 EXP 03/19/27

    Returns ticker, expiration yyyy-mm-dd, strike, option_type.
    """
    sym = str(symbol or "").strip()
    desc = str(description or "").strip()

    # Best source: Schwab Symbol text.
    m = re.match(
        r"^\s*([A-Z]{1,6})\s+(\d{1,2})/(\d{1,2})/(\d{2,4})\s+(\d+(?:\.\d+)?)\s+([CP])\s*$",
        sym,
        flags=re.I,
    )
    if m:
        ticker = m.group(1).upper()
        mo, da, yr = int(m.group(2)), int(m.group(3)), int(m.group(4))
        if yr < 100:
            yr += 2000
        strike = float(m.group(5))
        typ = "CALL" if m.group(6).upper() == "C" else "PUT"
        return ticker, f"{yr:04d}-{mo:02d}-{da:02d}", strike, typ

    # OCC style fallback.
    compact = (sym + " " + desc).replace(" ", "").upper()
    m = re.search(r"\b([A-Z]{1,6})(\d{6})([CP])(\d{8})\b", compact)
    if m:
        ticker = m.group(1)
        yymmdd = m.group(2)
        yr = 2000 + int(yymmdd[:2])
        mo = int(yymmdd[2:4])
        da = int(yymmdd[4:6])
        strike = int(m.group(4)) / 1000.0
        typ = "CALL" if m.group(3) == "C" else "PUT"
        return ticker, f"{yr:04d}-{mo:02d}-{da:02d}", strike, typ

    # Description fallback.
    typ = None
    if re.search(r"\bCALL\b", desc, flags=re.I):
        typ = "CALL"
    elif re.search(r"\bPUT\b", desc, flags=re.I):
        typ = "PUT"

    strike = np.nan
    # CALL ... $590 EXP or PUT ... $600 EXP
    m = re.search(r"\$(\d+(?:\.\d+)?)\s+EXP\b", desc, flags=re.I)
    if m:
        strike = float(m.group(1))

    expiration = None
    m = re.search(r"\bEXP\s+(\d{1,2})/(\d{1,2})/(\d{2,4})\b", desc, flags=re.I)
    if not m:
        m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", sym + " " + desc)
    if m:
        mo, da, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yr < 100:
            yr += 2000
        expiration = f"{yr:04d}-{mo:02d}-{da:02d}"

    ticker = None
    m = re.match(r"^\s*([A-Z]{1,6})\b", sym)
    if m:
        ticker = m.group(1).upper()

    return ticker, expiration, strike, typ


def standardize(df):
    symbol_col = find_col_exact_or_contains(df, ["Symbol"], required=True)
    desc_col = find_col_exact_or_contains(df, ["Description"], required=False)
    qty_col = find_col_exact_or_contains(df, ["Qty (Quantity)", "Quantity", "Qty"], required=True)
    day_col = find_col_exact_or_contains(df, ["Day Chng $ (Day Change $)", "Day Change", "Day P/L", "Day PL"], required=True)

    delta_col = find_col_exact_or_contains(df, ["Delta", "Option Delta"], required=False)
    price_col = find_col_exact_or_contains(df, ["Price"], required=False)

    parsed = []
    for _, row in df.iterrows():
        parsed.append(parse_option_fields(row.get(symbol_col, ""), row.get(desc_col, "") if desc_col else ""))
    p = pd.DataFrame(parsed, columns=["ticker", "expiration", "strike", "option_type"])

    out = df.copy()
    out["ticker"] = p["ticker"]
    out["expiration"] = p["expiration"]
    out["strike"] = p["strike"]
    out["option_type"] = p["option_type"]
    out["quantity"] = out[qty_col].apply(num)
    out["day_pl"] = out[day_col].apply(num)
    out["csv_delta"] = out[delta_col].apply(num) if delta_col else np.nan
    out["price"] = out[price_col].apply(num) if price_col else np.nan

    out = out.dropna(subset=["ticker", "expiration", "strike", "option_type", "quantity", "day_pl"]).copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["option_type"] = out["option_type"].astype(str).str.upper().str.strip()
    out = out[out["option_type"].isin(["CALL", "PUT"])].copy()
    return out


def get_quotes(tickers):
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    rows = []
    for ticker in sorted(set(tickers)):
        try:
            tk = yf.Ticker(ticker)
            fast = {}
            try:
                fast = dict(tk.fast_info)
            except Exception:
                pass
            last = fast.get("last_price", np.nan)
            if pd.isna(last):
                last = fast.get("lastPrice", np.nan)
            prev = fast.get("regularMarketPreviousClose", np.nan)
            if pd.isna(prev):
                prev = fast.get("previous_close", np.nan)
            if pd.isna(prev):
                prev = fast.get("previousClose", np.nan)

            if pd.isna(last) or pd.isna(prev):
                hist = tk.history(period="5d", interval="1d", auto_adjust=False)
                closes = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
                if len(closes) >= 2:
                    prev = float(closes.iloc[-2])
                    last = float(closes.iloc[-1])
                elif len(closes) == 1:
                    last = float(closes.iloc[-1])

            chg = last - prev if not pd.isna(last) and not pd.isna(prev) else np.nan
            rows.append({"ticker": ticker, "last": last, "prev_close": prev, "stock_change": chg})
        except Exception as e:
            rows.append({"ticker": ticker, "last": np.nan, "prev_close": np.nan, "stock_change": np.nan, "quote_error": str(e)})
    return pd.DataFrame(rows)


def call_spread_intrinsic(S, lower, upper, contracts):
    v = max(S - lower, 0) - max(S - upper, 0)
    v = max(0, min(v, upper - lower))
    return v * contracts * 100


def long_put_spread_intrinsic(S, lower, upper, contracts):
    v = max(upper - S, 0) - max(lower - S, 0)
    v = max(0, min(v, upper - lower))
    return v * contracts * 100


def cboe_occ_symbol(ticker, expiration, strike, option_type):
    try:
        exp = pd.Timestamp(expiration)
        cp = "C" if str(option_type).upper() == "CALL" else "P"
        strike_code = int(round(float(strike) * 1000))
        return f"{ticker}{exp:%y%m%d}{cp}{strike_code:08d}"
    except Exception:
        return None


def fetch_cboe_chain(ticker, timeout=15.0):
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload.get("data", {})


def audit_option_day_pl_baselines(options, min_excess=10000.0, timeout=15.0):
    """Find option books whose reported loss is unsupported by Cboe marks and delta."""
    detail_rows = []
    errors = []

    groups = {
        (ticker, option_type): group
        for (ticker, option_type), group in options.groupby(["ticker", "option_type"])
        if float(group["day_pl"].sum()) < 0
    }
    tickers = sorted({ticker for ticker, _ in groups})
    chains = {}
    with ThreadPoolExecutor(max_workers=min(8, max(len(tickers), 1))) as pool:
        futures = {pool.submit(fetch_cboe_chain, ticker, timeout): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                chains[ticker] = future.result()
            except Exception as exc:
                errors.append({"ticker": ticker, "cboe_error": str(exc)})

    for (ticker, option_type), group in groups.items():
        if ticker not in chains:
            continue
        try:
            chain = chains[ticker]
            option_map = {row.get("option"): row for row in chain.get("options", [])}
            stock_change = num(chain.get("price_change"))
        except Exception as exc:
            errors.append({"ticker": ticker, "cboe_error": str(exc)})
            continue

        for _, row in group.iterrows():
            occ = cboe_occ_symbol(ticker, row["expiration"], row["strike"], row["option_type"])
            quote = option_map.get(occ)
            if not quote:
                continue

            bid = num(quote.get("bid"))
            ask = num(quote.get("ask"))
            prior = num(quote.get("prev_day_close"))
            delta = num(quote.get("delta"))
            if any(pd.isna(value) for value in [bid, ask, prior, delta, stock_change]):
                continue

            quantity = float(row["quantity"])
            multiplier = quantity * 100.0
            midpoint = (bid + ask) / 2.0
            actual = float(row["day_pl"])
            cboe_day_pl = (midpoint - prior) * multiplier
            delta_day_pl = delta * stock_change * multiplier
            broker_mark = num(row.get("price"))
            implied_broker_prior = (
                broker_mark - actual / multiplier
                if not pd.isna(broker_mark) and multiplier != 0
                else np.nan
            )
            detail_rows.append({
                "ticker": ticker,
                "option_type": option_type,
                "expiration": row["expiration"],
                "contract": f"{float(row['strike']):g} {'C' if option_type == 'CALL' else 'P'}",
                "quantity": quantity,
                "schwab_mark": broker_mark,
                "cboe_bid": bid,
                "cboe_ask": ask,
                "cboe_mid": midpoint,
                "cboe_prev_close": prior,
                "implied_schwab_prev": implied_broker_prior,
                "cboe_delta": delta,
                "schwab_day_pl": actual,
                "cboe_mid_day_pl": cboe_day_pl,
                "delta_expected_day_pl": delta_day_pl,
                "midpoint_gap": cboe_day_pl - actual,
                "delta_gap": delta_day_pl - actual,
                "volume": num(quote.get("volume")),
                "last_trade_time": quote.get("last_trade_time"),
            })

    details = pd.DataFrame(detail_rows)
    if details.empty:
        return details, pd.DataFrame(), pd.DataFrame(errors)

    ticker_summary = details.groupby(["ticker", "option_type"], as_index=False).agg(
        positions=("contract", "count"),
        schwab_day_pl=("schwab_day_pl", "sum"),
        cboe_mid_day_pl=("cboe_mid_day_pl", "sum"),
        delta_expected_day_pl=("delta_expected_day_pl", "sum"),
        midpoint_gap=("midpoint_gap", "sum"),
        delta_gap=("delta_gap", "sum"),
    )
    ticker_summary["candidate_addback"] = ticker_summary[["midpoint_gap", "delta_gap"]].min(axis=1).clip(lower=0)
    ticker_summary["flagged"] = (
        (ticker_summary["schwab_day_pl"] < 0)
        & (ticker_summary["midpoint_gap"] >= min_excess)
        & (ticker_summary["delta_gap"] >= min_excess)
    )
    ticker_summary.loc[~ticker_summary["flagged"], "candidate_addback"] = 0.0
    ticker_summary = ticker_summary.sort_values(["flagged", "candidate_addback"], ascending=[False, False])

    flagged_books = set(
        ticker_summary.loc[ticker_summary["flagged"], ["ticker", "option_type"]].itertuples(index=False, name=None)
    )
    details["book_flagged"] = [
        (ticker, option_type) in flagged_books
        for ticker, option_type in zip(details["ticker"], details["option_type"])
    ]
    details = details.sort_values(["book_flagged", "midpoint_gap"], ascending=[False, False])
    return details, ticker_summary, pd.DataFrame(errors)


def find_bad_itm_upday_call_spreads(options, quotes, max_call_width=10.0):
    calls = options[options["option_type"] == "CALL"].copy()
    qmap = quotes.set_index("ticker").to_dict("index")
    rows = []

    for (ticker, exp), g in calls.groupby(["ticker", "expiration"], dropna=False):
        q = qmap.get(ticker)
        if not q:
            continue
        last, prev, stock_chg = q.get("last"), q.get("prev_close"), q.get("stock_change")
        if pd.isna(last) or pd.isna(prev) or pd.isna(stock_chg):
            continue

        longs = [
            {
                "strike": float(row["strike"]),
                "remaining_qty": int(abs(row["quantity"])),
                "original_qty": int(abs(row["quantity"])),
                "day_pl": float(row["day_pl"]),
                "row": row.copy(),
            }
            for _, row in g[g["quantity"] > 0].sort_values("strike").iterrows()
        ]
        shorts = [
            {
                "strike": float(row["strike"]),
                "remaining_qty": int(abs(row["quantity"])),
                "original_qty": int(abs(row["quantity"])),
                "day_pl": float(row["day_pl"]),
            }
            for _, row in g[g["quantity"] < 0].sort_values("strike").iterrows()
        ]

        # Pair each short with the nearest lower long, consuming quantities as they are matched.
        for short_leg in shorts:
            upper = short_leg["strike"]
            remaining_short = short_leg["remaining_qty"]
            candidates = [leg for leg in longs if leg["remaining_qty"] > 0 and leg["strike"] < upper]
            candidates.sort(key=lambda leg: leg["strike"], reverse=True)

            for long_leg in candidates:
                if remaining_short <= 0:
                    break

                contracts = min(long_leg["remaining_qty"], remaining_short)
                if contracts <= 0:
                    continue

                lower = long_leg["strike"]
                width = upper - lower
                if width <= 0 or width > max_call_width:
                    # Width-excluded: do not consume; try next (wider) long
                    # which will also be excluded, leaving short unmatched.
                    continue
                long_day_pl = long_leg["day_pl"] * contracts / long_leg["original_qty"]
                short_day_pl = short_leg["day_pl"] * contracts / short_leg["original_qty"]
                actual = long_day_pl + short_day_pl

                # Consume every matched pair, including those excluded from the report.
                eligible = last > upper and (stock_chg > 0 or prev > upper)
                if eligible and actual < 0:
                    expected = call_spread_intrinsic(last, lower, upper, contracts) - call_spread_intrinsic(prev, lower, upper, contracts)
                    diff = expected - actual

                    rows.append({
                        "ticker": ticker,
                        "expiration": exp,
                        "spread": f"{lower:g}/{upper:g} C",
                        "width": width,
                        "contracts": contracts,
                        "stock_change": stock_chg,
                        "long_day_pl": long_day_pl,
                        "short_day_pl": short_day_pl,
                        "schwab_net_day_pl": actual,
                        "intrinsic_expected_day_pl": expected,
                        "diff_to_add_back": diff,
                        "adjustment_basis": "intrinsic_scenario_not_verified_pricing_error",
                    })

                long_leg["remaining_qty"] -= contracts
                remaining_short -= contracts

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("diff_to_add_back", ascending=False)
    return out


def split_short_puts_and_spreads(options):
    puts = options[options["option_type"] == "PUT"].copy()
    naked_rows = []
    spread_rows = []

    for (ticker, exp), g in puts.groupby(["ticker", "expiration"], dropna=False):
        longs = [
            {
                "strike": float(row["strike"]),
                "remaining_qty": int(abs(row["quantity"])),
                "original_qty": int(abs(row["quantity"])),
                "day_pl": float(row["day_pl"]),
                "row": row.copy(),
            }
            for _, row in g[g["quantity"] > 0].sort_values("strike").iterrows()
        ]
        shorts = [
            {
                "row": row.copy(),
                "strike": float(row["strike"]),
                "remaining_qty": int(abs(row["quantity"])),
                "original_qty": int(abs(row["quantity"])),
                "day_pl": float(row["day_pl"]),
            }
            for _, row in g[g["quantity"] < 0].sort_values("strike", ascending=False).iterrows()
        ]

        for short_leg in shorts:
            short_strike = short_leg["strike"]
            remaining_short = short_leg["remaining_qty"]
            candidates = [leg for leg in longs if leg["remaining_qty"] > 0 and leg["strike"] < short_strike]
            candidates.sort(key=lambda leg: leg["strike"], reverse=True)

            for long_leg in candidates:
                if remaining_short <= 0:
                    break

                contracts = min(long_leg["remaining_qty"], remaining_short)
                if contracts <= 0:
                    continue

                matched_row = short_leg["row"].copy()
                matched_row["contracts"] = contracts
                matched_row["long_strike"] = float(long_leg["strike"])
                matched_row["long_day_pl"] = long_leg["day_pl"] * contracts / max(long_leg["original_qty"], 1)
                matched_row["short_day_pl"] = short_leg["day_pl"] * contracts / max(short_leg["original_qty"], 1)
                matched_row["long_csv_delta"] = long_leg["row"].get("csv_delta", np.nan)
                matched_row["long_price"] = long_leg["row"].get("price", np.nan)
                spread_rows.append(matched_row)

                long_leg["remaining_qty"] -= contracts
                remaining_short -= contracts

            if remaining_short > 0:
                naked_row = short_leg["row"].copy()
                ratio = remaining_short / max(short_leg["original_qty"], 1)
                naked_row["quantity"] = -remaining_short
                naked_row["day_pl"] = short_leg["day_pl"] * ratio
                naked_rows.append(naked_row)

    return pd.DataFrame(naked_rows), pd.DataFrame(spread_rows)


def year_frac(exp):
    try:
        e = pd.Timestamp(exp).to_pydatetime().replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max((e - now).total_seconds() / 86400.0 / 365.0, 1 / 365)
    except Exception:
        return np.nan


def nearest_expiration(tk, target):
    try:
        exps = list(tk.options or [])
    except Exception:
        return None
    if not exps:
        return None
    if target in exps:
        return target
    try:
        target_ts = pd.Timestamp(target)
        return sorted(exps, key=lambda e: abs((pd.Timestamp(e) - target_ts).days))[0]
    except Exception:
        return exps[0]


def fetch_yf_iv_for_put(ticker, expiration, strike):
    try:
        tk = yf.Ticker(ticker)
        exp = nearest_expiration(tk, expiration)
        if not exp:
            return np.nan, None, "no_yf_expiration"
        chain = tk.option_chain(exp).puts.copy()
        if chain.empty:
            return np.nan, exp, "empty_put_chain"
        chain["strike_diff"] = (chain["strike"].astype(float) - float(strike)).abs()
        row = chain.sort_values("strike_diff").iloc[0]
        return float(row.get("impliedVolatility", np.nan)), exp, f"yf_matched_strike_{float(row.get('strike', np.nan)):g}"
    except Exception as e:
        return np.nan, None, f"yf_error_{e}"


def check_otm_short_puts(puts, quotes, risk_free_rate=_RISK_FREE_RATE, dividend_yield=_DIVIDEND_YIELD, prefer_csv_delta=True):
    puts = puts.copy()
    qmap = quotes.set_index("ticker").to_dict("index")
    rows = []
    iv_cache = {}

    for _, row in puts.iterrows():
        ticker = row["ticker"]
        q = qmap.get(ticker)
        if not q:
            continue

        last, stock_chg = q.get("last"), q.get("stock_change")
        if pd.isna(last) or pd.isna(stock_chg):
            continue

        strike = float(row["strike"])
        contracts = int(abs(row["quantity"]))
        if last <= strike:
            continue

        delta = np.nan
        delta_source = None
        yf_iv = np.nan
        matched_exp = None
        yf_note = None

        if prefer_csv_delta and not pd.isna(row.get("csv_delta", np.nan)):
            delta = normalize_put_delta(row["csv_delta"])
            delta_source = "csv/broker_delta"

        if pd.isna(delta):
            option_mark = float(row.get("price", np.nan))
            T = year_frac(row["expiration"])
            mark_iv = implied_volatility(
                option_mark,
                last,
                strike,
                T,
                risk_free_rate,
                "P",
                dividend_yield,
            )
            if not pd.isna(mark_iv):
                delta = bs_put_delta(last, strike, T, risk_free_rate, mark_iv, dividend_yield)
                delta_source = "mark_iv_black_scholes"

        if pd.isna(delta):
            key = (ticker, row["expiration"], strike)
            if key not in iv_cache:
                iv_cache[key] = fetch_yf_iv_for_put(ticker, row["expiration"], strike)
            yf_iv, matched_exp, yf_note = iv_cache[key]
            T = year_frac(matched_exp or row["expiration"])
            delta = bs_put_delta(last, strike, T, risk_free_rate, yf_iv, dividend_yield)
            delta_source = "yf_iv_black_scholes"

        expected = np.nan if pd.isna(delta) else abs(delta) * float(stock_chg) * contracts * 100
        actual = float(row["day_pl"])
        diff = np.nan if pd.isna(expected) else expected - actual

        rows.append({
            "ticker": ticker,
            "expiration": row["expiration"],
            "short_put": f"{strike:g} P",
            "contracts": contracts,
            "stock_change": stock_chg,
            "delta_used_abs": abs(delta) if not pd.isna(delta) else np.nan,
            "delta_source": delta_source,
            "schwab_day_pl": actual,
            "delta_expected_day_pl": expected,
            "diff_to_add_back": diff,
            "csv_delta": row.get("csv_delta", np.nan),
            "yf_iv": yf_iv,
            "yf_matched_expiration": matched_exp,
            "yf_note": yf_note,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("diff_to_add_back", ascending=False)
    return out


def resolve_put_delta(ticker, expiration, strike, spot, csv_delta=np.nan, option_mark=np.nan, risk_free_rate=_RISK_FREE_RATE, dividend_yield=_DIVIDEND_YIELD, prefer_csv_delta=True, iv_cache=None):
    delta = np.nan
    delta_source = None
    yf_iv = np.nan
    matched_exp = None
    yf_note = None

    if prefer_csv_delta and not pd.isna(csv_delta):
        delta = normalize_put_delta(csv_delta)
        delta_source = "csv/broker_delta"

    if pd.isna(delta):
        T = year_frac(expiration)
        mark_iv = implied_volatility(
            float(option_mark) if not pd.isna(option_mark) else np.nan,
            spot,
            strike,
            T,
            risk_free_rate,
            "P",
            dividend_yield,
        )
        if not pd.isna(mark_iv):
            delta = bs_put_delta(spot, strike, T, risk_free_rate, mark_iv, dividend_yield)
            delta_source = "mark_iv_black_scholes"

    if pd.isna(delta):
        key = (ticker, expiration, strike)
        if iv_cache is not None and key in iv_cache:
            yf_iv, matched_exp, yf_note = iv_cache[key]
        else:
            yf_iv, matched_exp, yf_note = fetch_yf_iv_for_put(ticker, expiration, strike)
            if iv_cache is not None:
                iv_cache[key] = (yf_iv, matched_exp, yf_note)
        T = year_frac(matched_exp or expiration)
        delta = bs_put_delta(spot, strike, T, risk_free_rate, yf_iv, dividend_yield)
        delta_source = "yf_iv_black_scholes"

    return {
        "delta": delta,
        "delta_source": delta_source,
        "yf_iv": yf_iv,
        "yf_matched_expiration": matched_exp,
        "yf_note": yf_note,
    }


def check_otm_put_spreads(spreads, quotes, risk_free_rate=_RISK_FREE_RATE, dividend_yield=_DIVIDEND_YIELD, prefer_csv_delta=True):
    spreads = spreads.copy()
    qmap = quotes.set_index("ticker").to_dict("index")
    rows = []
    iv_cache = {}

    for _, row in spreads.iterrows():
        ticker = row["ticker"]
        q = qmap.get(ticker)
        if not q:
            continue

        last, prev, stock_chg = q.get("last"), q.get("prev_close"), q.get("stock_change")
        if pd.isna(last) or pd.isna(prev) or pd.isna(stock_chg) or stock_chg <= 0:
            continue

        short_strike = float(row["strike"])
        if last <= short_strike:
            continue

        long_strike = float(row["long_strike"])
        contracts = int(abs(row["contracts"]))
        long_day_pl = float(row["long_day_pl"])
        short_day_pl = float(row["short_day_pl"])
        actual = long_day_pl + short_day_pl
        short_delta_info = resolve_put_delta(
            ticker,
            row["expiration"],
            short_strike,
            float(last),
            csv_delta=row.get("csv_delta", np.nan),
            option_mark=row.get("price", np.nan),
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            prefer_csv_delta=prefer_csv_delta,
            iv_cache=iv_cache,
        )
        long_delta_info = resolve_put_delta(
            ticker,
            row["expiration"],
            long_strike,
            float(last),
            csv_delta=row.get("long_csv_delta", np.nan),
            option_mark=row.get("long_price", np.nan),
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            prefer_csv_delta=prefer_csv_delta,
            iv_cache=iv_cache,
        )
        short_abs = abs(short_delta_info["delta"]) if not pd.isna(short_delta_info["delta"]) else np.nan
        long_abs = abs(long_delta_info["delta"]) if not pd.isna(long_delta_info["delta"]) else np.nan
        net_delta = np.nan if pd.isna(short_abs) or pd.isna(long_abs) else short_abs - long_abs
        expected = np.nan if pd.isna(net_delta) else net_delta * stock_chg * contracts * 100
        diff = expected - actual

        rows.append({
            "ticker": ticker,
            "expiration": row["expiration"],
            "spread": f"{short_strike:g}/{long_strike:g} P",
            "contracts": contracts,
            "stock_change": stock_chg,
            "short_delta_abs": short_abs,
            "long_delta_abs": long_abs,
            "net_spread_delta": net_delta,
            "short_delta_source": short_delta_info["delta_source"],
            "long_delta_source": long_delta_info["delta_source"],
            "long_day_pl": long_day_pl,
            "short_day_pl": short_day_pl,
            "schwab_net_day_pl": actual,
            "delta_expected_day_pl": expected,
            "diff_to_add_back": diff,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("diff_to_add_back", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--risk-free-rate", type=float, default=_RISK_FREE_RATE)
    ap.add_argument("--dividend-yield", type=float, default=_DIVIDEND_YIELD)
    ap.add_argument("--use-yf-delta-only", action="store_true")
    ap.add_argument("--max-call-width", type=float, default=10.0)
    ap.add_argument("--baseline-min-excess", type=float, default=10000.0)
    ap.add_argument("--cboe-timeout", type=float, default=15.0)
    ap.add_argument("--no-cboe-audit", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = read_broker_csv(args.csv)
    options = standardize(raw)
    quotes = get_quotes(sorted(options["ticker"].unique()))
    naked_puts, put_spread_legs = split_short_puts_and_spreads(options)

    bad_calls = find_bad_itm_upday_call_spreads(options, quotes, max_call_width=args.max_call_width)
    if args.no_cboe_audit:
        option_quote_details = pd.DataFrame()
        option_quote_summary = pd.DataFrame()
        option_quote_errors = pd.DataFrame()
    else:
        option_quote_details, option_quote_summary, option_quote_errors = audit_option_day_pl_baselines(
            options,
            min_excess=args.baseline_min_excess,
            timeout=args.cboe_timeout,
        )
    puts = check_otm_short_puts(
        naked_puts,
        quotes,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
        prefer_csv_delta=(not args.use_yf_delta_only),
    )
    put_spreads = check_otm_put_spreads(
        put_spread_legs,
        quotes,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
        prefer_csv_delta=(not args.use_yf_delta_only),
    )

    intrinsic_call_addback = 0.0 if bad_calls.empty else float(pd.to_numeric(bad_calls["diff_to_add_back"], errors="coerce").fillna(0).sum())
    if option_quote_summary.empty:
        baseline_call_by_ticker = pd.Series(dtype=float)
        baseline_put_by_ticker = pd.Series(dtype=float)
    else:
        baseline_call_by_ticker = option_quote_summary.loc[
            option_quote_summary["option_type"] == "CALL"
        ].set_index("ticker")["candidate_addback"]
        baseline_put_by_ticker = option_quote_summary.loc[
            option_quote_summary["option_type"] == "PUT"
        ].set_index("ticker")["candidate_addback"]
    baseline_call_addback = float(baseline_call_by_ticker.sum())
    baseline_put_addback = float(baseline_put_by_ticker.sum())

    intrinsic_by_ticker = (
        pd.Series(dtype=float)
        if bad_calls.empty
        else bad_calls.groupby("ticker")["diff_to_add_back"].sum()
    )
    call_adjustment_by_ticker = pd.concat(
        [intrinsic_by_ticker.rename("intrinsic"), baseline_call_by_ticker.rename("baseline")],
        axis=1,
    ).fillna(0.0)
    call_adjustment_by_ticker["deduplicated_call_adjustment"] = call_adjustment_by_ticker[["intrinsic", "baseline"]].max(axis=1)
    call_addback = float(call_adjustment_by_ticker["deduplicated_call_adjustment"].sum())
    if puts.empty:
        put_addback = 0.0
    else:
        puts["positive_addback"] = pd.to_numeric(puts["diff_to_add_back"], errors="coerce").clip(lower=0)
        put_addback = float(puts["positive_addback"].fillna(0).sum())
    if put_spreads.empty:
        put_spread_addback = 0.0
    else:
        put_spreads["positive_addback"] = pd.to_numeric(put_spreads["diff_to_add_back"], errors="coerce").clip(lower=0)
        put_spread_addback = float(put_spreads["positive_addback"].fillna(0).sum())

    naked_put_by_ticker = (
        pd.Series(dtype=float)
        if puts.empty
        else puts.groupby("ticker")["positive_addback"].sum()
    )
    put_spread_by_ticker = (
        pd.Series(dtype=float)
        if put_spreads.empty
        else put_spreads.groupby("ticker")["positive_addback"].sum()
    )
    put_model_by_ticker = pd.concat(
        [naked_put_by_ticker.rename("naked"), put_spread_by_ticker.rename("spread")],
        axis=1,
    ).fillna(0.0).sum(axis=1)
    put_adjustment_by_ticker = pd.concat(
        [put_model_by_ticker.rename("model"), baseline_put_by_ticker.rename("baseline")],
        axis=1,
    ).fillna(0.0)
    put_adjustment_by_ticker["deduplicated_put_adjustment"] = put_adjustment_by_ticker[["model", "baseline"]].max(axis=1)
    put_adjustment = float(put_adjustment_by_ticker["deduplicated_put_adjustment"].sum())

    baseline_call_count = 0 if option_quote_summary.empty else int(
        ((option_quote_summary["option_type"] == "CALL") & option_quote_summary["flagged"]).sum()
    )
    baseline_put_count = 0 if option_quote_summary.empty else int(
        ((option_quote_summary["option_type"] == "PUT") & option_quote_summary["flagged"]).sum()
    )
    total = call_addback + put_adjustment
    summary = pd.DataFrame([
        {"bucket": "ITM call spreads intrinsic adjustment (diagnostic)", "count": len(bad_calls), "addback": intrinsic_call_addback},
        {"bucket": "Cboe call baseline anomaly (diagnostic)", "count": baseline_call_count, "addback": baseline_call_addback},
        {"bucket": "CALL ADJUSTMENT (deduplicated)", "count": len(call_adjustment_by_ticker), "addback": call_addback},
        {"bucket": "OTM naked short puts delta check", "count": len(puts), "addback": put_addback},
        {"bucket": "OTM up-day put spreads", "count": len(put_spreads), "addback": put_spread_addback},
        {"bucket": "Cboe put baseline anomaly (diagnostic)", "count": baseline_put_count, "addback": baseline_put_addback},
        {"bucket": "PUT ADJUSTMENT (deduplicated)", "count": len(put_adjustment_by_ticker), "addback": put_adjustment},
        {"bucket": "TOTAL", "count": len(call_adjustment_by_ticker) + len(put_adjustment_by_ticker), "addback": total},
    ])

    bad_calls_path = outdir / "bad_itm_upday_call_spreads.csv"
    puts_path = outdir / "otm_short_put_delta_check.csv"
    put_spreads_path = outdir / "otm_upday_put_spreads.csv"
    option_quote_details_path = outdir / "option_quote_baseline_details.csv"
    option_quote_summary_path = outdir / "option_quote_baseline_summary.csv"
    option_quote_errors_path = outdir / "option_quote_baseline_errors.csv"
    summary_path = outdir / "final_noise_summary.csv"
    xlsx_path = outdir / "final_portfolio_noise_report.xlsx"

    bad_calls.to_csv(bad_calls_path, index=False)
    puts.to_csv(puts_path, index=False)
    put_spreads.to_csv(put_spreads_path, index=False)
    option_quote_details.to_csv(option_quote_details_path, index=False)
    option_quote_summary.to_csv(option_quote_summary_path, index=False)
    option_quote_errors.to_csv(option_quote_errors_path, index=False)
    summary.to_csv(summary_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="Summary")
        bad_calls.to_excel(w, index=False, sheet_name="ITM Call Adjustments")
        puts.to_excel(w, index=False, sheet_name="OTM Naked Short Puts")
        put_spreads.to_excel(w, index=False, sheet_name="OTM Up-Day Put Spreads")
        option_quote_summary.to_excel(w, index=False, sheet_name="Option Baseline Summary")
        option_quote_details.to_excel(w, index=False, sheet_name="Option Baseline Detail")
        option_quote_errors.to_excel(w, index=False, sheet_name="Cboe Errors")
        quotes.to_excel(w, index=False, sheet_name="Quotes")
        options.to_excel(w, index=False, sheet_name="Parsed Options")

    print("\nFINAL PORTFOLIO NOISE CHECK v2")
    print("=" * 72)

    print("\nCALL SPREAD RULE:")
    print(f"  Negative day P/L with stock above the short strike; width <= ${args.max_call_width:g}; stock up OR both closes above the short strike.")
    print("  Intrinsic-based scenario adjustment; does not establish that broker marks are wrong.")
    if bad_calls.empty:
        print("  No qualifying ITM call-spread losses found.")
    else:
        cols = ["ticker", "expiration", "spread", "width", "contracts", "stock_change", "schwab_net_day_pl", "intrinsic_expected_day_pl", "diff_to_add_back"]
        print(bad_calls[cols].to_string(index=False))
        print(f"\n  Intrinsic call-spread adjustment: ${intrinsic_call_addback:,.2f}")

    print("\nOPTION QUOTE-BASELINE AUDIT:")
    print("  Flags call or put books only when Schwab P/L is materially worse than both Cboe midpoint P/L and Cboe delta P/L.")
    if args.no_cboe_audit:
        print("  Skipped by --no-cboe-audit.")
    elif option_quote_summary.empty:
        print("  No option books could be audited against Cboe.")
    else:
        flagged = option_quote_summary[option_quote_summary["flagged"]]
        if flagged.empty:
            print("  No corroborated option baseline anomalies found.")
        else:
            cols = ["ticker", "option_type", "positions", "schwab_day_pl", "cboe_mid_day_pl", "delta_expected_day_pl", "candidate_addback"]
            print(flagged[cols].to_string(index=False))
            print(f"\n  Cboe call baseline candidate adjustment: ${baseline_call_addback:,.2f}")
            print(f"  Cboe put baseline candidate adjustment:  ${baseline_put_addback:,.2f}")
        if not option_quote_errors.empty:
            print(f"  Cboe audit unavailable for {len(option_quote_errors)} ticker(s); see {option_quote_errors_path}.")

    print(f"\n  Deduplicated call adjustment: ${call_addback:,.2f}")

    print("\nSHORT PUT RULE:")
    print("  OTM naked short puts only, expected P/L = abs(delta) x stock_change x contracts x 100.")
    if puts.empty:
        print("  No OTM naked short puts found.")
    else:
        cols = ["ticker", "expiration", "short_put", "contracts", "stock_change", "delta_used_abs", "delta_source", "schwab_day_pl", "delta_expected_day_pl", "diff_to_add_back"]
        print(puts[cols].to_string(index=False))
        print(f"\n  Short-put positive add-back: ${put_addback:,.2f}")

    print("\nPUT SPREAD RULE:")
    print("  OTM short put spreads only, stock up, expected P/L = net spread delta x stock_change x contracts x 100.")
    if put_spreads.empty:
        print("  No OTM up-day put spreads found.")
    else:
        cols = ["ticker", "expiration", "spread", "contracts", "stock_change", "short_delta_abs", "long_delta_abs", "net_spread_delta", "schwab_net_day_pl", "delta_expected_day_pl", "diff_to_add_back"]
        print(put_spreads[cols].to_string(index=False))
        print(f"\n  Put-spread positive add-back: ${put_spread_addback:,.2f}")

    print(f"\n  Deduplicated put adjustment: ${put_adjustment:,.2f}")

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nTOTAL MODEL ADD-BACK: ${total:,.2f}")

    print("\nFiles written:")
    print(f"  {bad_calls_path}")
    print(f"  {puts_path}")
    print(f"  {put_spreads_path}")
    print(f"  {option_quote_details_path}")
    print(f"  {option_quote_summary_path}")
    print(f"  {option_quote_errors_path}")
    print(f"  {summary_path}")
    print(f"  {xlsx_path}")


if __name__ == "__main__":
    main()
