#!/usr/bin/env python3
"""
portfolio_noise_checker_v2_fetch_put_deltas.py

Checks option mark noise from a broker holdings CSV:

1) ITM call spreads:
   - Pairs long lower-strike calls with short higher-strike calls
   - Same ticker + expiration
   - Only treats spread as ITM/capped if current underlying price > short call strike
   - Compares broker-reported day P/L vs intrinsic-reference day P/L

2) OTM short puts:
   - Finds short put positions where current underlying price > put strike
   - Fetches the option chain for those short puts only
   - Uses option-chain implied volatility to compute Black-Scholes put delta
   - Compares broker-reported day P/L vs delta-expected day P/L

Outputs:
   - call_spread_noise.csv
   - otm_short_put_noise.csv
   - noise_summary.csv
   - portfolio_noise_report.xlsx

Install:
   pip install pandas yfinance openpyxl numpy

Run:
   python portfolio_noise_checker_v2_fetch_put_deltas.py my_holdings.csv

Optional:
   python portfolio_noise_checker_v2_fetch_put_deltas.py my_holdings.csv --risk-free-rate 0.045 --dividend-yield 0.00

Important:
   - yfinance data is usually delayed and is not guaranteed true real-time.
   - yfinance usually does NOT provide option Greeks directly.
   - This script fetches option-chain implied volatility for short puts and computes delta.
   - For institutional accuracy, replace the quote/option-chain functions with Schwab, IBKR,
     Polygon, ORATS, OptionMetrics, Bloomberg, or another live Greeks source.
"""

import argparse
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

def read_broker_csv(path):
    """
    Robust reader for Schwab-style CSVs that start with a title line, e.g.
    "Positions for account ... as of ..."
    before the real header row.

    It scans for the row containing Symbol + Description + Qty/Quantity.
    """
    path = Path(path)

    # Read raw lines first to find the real header row.
    lines = path.read_text(errors="replace").splitlines()
    header_idx = None

    for i, line in enumerate(lines[:50]):
        low = line.lower()
        if (
            "symbol" in low
            and "description" in low
            and ("qty" in low or "quantity" in low)
            and ("asset type" in low or "day chng" in low or "gain/loss" in low)
        ):
            header_idx = i
            break

    if header_idx is None:
        # fallback: normal pandas behavior
        return pd.read_csv(path)

    return pd.read_csv(path, skiprows=header_idx)



try:
    import yfinance as yf
except ImportError:
    yf = None


# -----------------------------
# Flexible column detection
# -----------------------------

def normalize_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def find_col(df: pd.DataFrame, candidates, required=True):
    norm_map = {normalize_col(c): c for c in df.columns}

    for cand in candidates:
        nc = normalize_col(cand)
        if nc in norm_map:
            return norm_map[nc]

    for cand in candidates:
        nc = normalize_col(cand)
        for k, original in norm_map.items():
            if nc in k or k in nc:
                return original

    if required:
        raise ValueError(
            f"Could not find required column. Tried candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def money_to_float(x):
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
        val = float(s)
        return -val if neg else val
    except ValueError:
        return np.nan


def parse_option_symbol_or_description(row_text: str):
    """
    Tries to parse option fields from either OCC symbols or broker descriptions.

    Returns:
      dict with ticker, expiration, strike, option_type
    """
    text = str(row_text)

    # OCC style:
    # MU  270319C00590000
    # MU270319C00590000
    compact = text.replace(" ", "").upper()
    occ = re.search(r"\b([A-Z]{1,6})(\d{6})([CP])(\d{8})\b", compact)
    if occ:
        ticker = occ.group(1)
        yymmdd = occ.group(2)
        option_type = "CALL" if occ.group(3) == "C" else "PUT"
        strike = int(occ.group(4)) / 1000.0
        yy = int(yymmdd[:2])
        year = 2000 + yy
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        expiration = f"{year:04d}-{month:02d}-{day:02d}"
        return {
            "parsed_ticker": ticker,
            "parsed_expiration": expiration,
            "parsed_strike": strike,
            "parsed_option_type": option_type,
        }

    option_type = None
    if re.search(r"\b(call|calls|c)\b", text, flags=re.I):
        option_type = "CALL"
    if re.search(r"\b(put|puts|p)\b", text, flags=re.I):
        option_type = "PUT"

    strike = np.nan
    m = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(?:Call|Calls|Put|Puts)\b", text, flags=re.I)
    if m:
        strike = float(m.group(1))
    else:
        nums = re.findall(r"\$?\b(\d{1,5}(?:\.\d+)?)\b", text)
        possible = [float(n) for n in nums if 1 <= float(n) <= 10000]
        if possible:
            strike = possible[-1]

    expiration = None
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", text)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        expiration = f"{year:04d}-{month:02d}-{day:02d}"

    ticker = None
    m = re.search(r"\b([A-Z]{1,6})\b", text)
    if m:
        ticker = m.group(1)

    return {
        "parsed_ticker": ticker,
        "parsed_expiration": expiration,
        "parsed_strike": strike,
        "parsed_option_type": option_type,
    }


def standardize_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates standardized columns:
      ticker, expiration, strike, option_type, quantity, day_pl, mark, csv_delta
    """
    out = df.copy()

    symbol_col = find_col(out, ["Symbol", "Underlying Symbol", "Ticker"], required=False)
    desc_col = find_col(out, ["Description", "Instrument", "Security", "Option Symbol", "Symbol"], required=False)
    qty_col = find_col(out, ["Quantity", "Qty", "Qty (Quantity)", "Position Quantity"], required=True)
    day_col = find_col(out, ["Day Change", "Day Chng $", "Day Chng $ (Day Change $)", "Today's Gain/Loss", "Day Gain/Loss", "Gain/Loss Day", "Day P/L", "Day PL"], required=True)

    mark_col = find_col(out, ["Mark", "Last Price", "Price", "Market Price", "Current Price"], required=False)
    delta_col = find_col(out, ["Delta", "Option Delta"], required=False)

    exp_col = find_col(out, ["Expiration", "Expiration Date", "Exp Date"], required=False)
    strike_col = find_col(out, ["Strike", "Strike Price"], required=False)
    type_col = find_col(out, ["Type", "Option Type", "Call/Put", "Put/Call"], required=False)

    parsed_rows = []
    for _, row in out.iterrows():
        parse_text = ""
        if desc_col:
            parse_text += " " + str(row.get(desc_col, ""))
        if symbol_col:
            parse_text += " " + str(row.get(symbol_col, ""))
        parsed_rows.append(parse_option_symbol_or_description(parse_text))
    parsed = pd.DataFrame(parsed_rows)

    if symbol_col:
        out["ticker"] = out[symbol_col].astype(str).str.extract(r"^([A-Z]{1,6})", expand=False)
    else:
        out["ticker"] = np.nan
    out["ticker"] = out["ticker"].fillna(parsed["parsed_ticker"])

    if exp_col:
        out["expiration"] = pd.to_datetime(out[exp_col], errors="coerce").dt.strftime("%Y-%m-%d")
        out["expiration"] = out["expiration"].replace("NaT", np.nan)
    else:
        out["expiration"] = np.nan
    out["expiration"] = out["expiration"].fillna(parsed["parsed_expiration"])

    if strike_col:
        out["strike"] = out[strike_col].apply(money_to_float)
    else:
        out["strike"] = np.nan
    out["strike"] = out["strike"].fillna(parsed["parsed_strike"])

    if type_col:
        t = out[type_col].astype(str).str.upper()
        # Avoid NumPy 2.x dtype-promotion issue from mixing strings with np.nan.
        out["option_type"] = pd.Series(pd.NA, index=out.index, dtype="object")
        out.loc[t.str.contains("CALL|\\bC\\b", na=False), "option_type"] = "CALL"
        out.loc[t.str.contains("PUT|\\bP\\b", na=False), "option_type"] = "PUT"
    else:
        out["option_type"] = pd.Series(pd.NA, index=out.index, dtype="object")
    out["option_type"] = out["option_type"].fillna(parsed["parsed_option_type"])

    out["quantity"] = out[qty_col].apply(money_to_float)
    out["day_pl"] = out[day_col].apply(money_to_float)

    if mark_col:
        out["mark"] = out[mark_col].apply(money_to_float)
    else:
        out["mark"] = np.nan

    # Kept only for audit/fallback display; v2 fetches short-put delta from option-chain IV.
    if delta_col:
        out["csv_delta"] = out[delta_col].apply(money_to_float)
    else:
        out["csv_delta"] = np.nan

    out = out.dropna(subset=["ticker", "strike", "quantity", "day_pl", "option_type"])
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["option_type"] = out["option_type"].astype(str).str.upper().str.strip()
    out = out[out["option_type"].isin(["CALL", "PUT"])].copy()
    return out


# -----------------------------
# Quotes and option chain
# -----------------------------

def get_quotes_yfinance(tickers):
    """
    Returns DataFrame with:
      ticker, last, prev_close, stock_change, stock_change_pct
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    rows = []
    for ticker in sorted(set(tickers)):
        try:
            t = yf.Ticker(ticker)
            fi = {}
            try:
                fi = dict(t.fast_info)
            except Exception:
                fi = {}

            last = fi.get("last_price", np.nan)
            prev = fi.get("previous_close", np.nan)

            if pd.isna(last) or pd.isna(prev):
                hist = t.history(period="5d", interval="1d", auto_adjust=False)
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                    last = float(hist["Close"].iloc[-1])
                elif len(hist) == 1:
                    last = float(hist["Close"].iloc[-1])

            change = last - prev if not pd.isna(last) and not pd.isna(prev) else np.nan
            pct = change / prev if not pd.isna(change) and prev else np.nan

            rows.append({
                "ticker": ticker,
                "last": last,
                "prev_close": prev,
                "stock_change": change,
                "stock_change_pct": pct,
            })
        except Exception as e:
            rows.append({
                "ticker": ticker,
                "last": np.nan,
                "prev_close": np.nan,
                "stock_change": np.nan,
                "stock_change_pct": np.nan,
                "quote_error": str(e),
            })

    return pd.DataFrame(rows)


def nearest_available_expiration(ticker_obj, target_expiration):
    """
    yfinance option expiry list may not exactly match parsed export date.
    First try exact; otherwise return nearest date.
    """
    expirations = list(ticker_obj.options or [])
    if not expirations or not target_expiration:
        return None, "no expirations available"

    if target_expiration in expirations:
        return target_expiration, "exact"

    try:
        target = pd.Timestamp(target_expiration)
        exp_dates = pd.to_datetime(expirations, errors="coerce")
        valid = [(str(e.date()), abs((e - target).days)) for e in exp_dates if not pd.isna(e)]
        if not valid:
            return None, "no valid expirations parsed"
        best, day_diff = sorted(valid, key=lambda x: x[1])[0]
        return best, f"nearest, {day_diff} days away"
    except Exception as e:
        return None, f"expiration matching error: {e}"


def fetch_short_put_chain_data(short_put_rows):
    """
    Fetches option chain data only for the short puts we need.
    Returns rows with chain_iv, chain_bid, chain_ask, chain_last_price, matched_expiration.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: pip install yfinance")

    needed = (
        short_put_rows[["ticker", "expiration", "strike"]]
        .drop_duplicates()
        .sort_values(["ticker", "expiration", "strike"])
    )

    output = []
    cache = {}

    for _, r in needed.iterrows():
        ticker = str(r["ticker"])
        expiration = r["expiration"]
        strike = float(r["strike"])

        try:
            tk = yf.Ticker(ticker)
            matched_exp, match_note = nearest_available_expiration(tk, expiration)

            if matched_exp is None:
                output.append({
                    "ticker": ticker,
                    "expiration": expiration,
                    "strike": strike,
                    "matched_expiration": None,
                    "chain_match_note": match_note,
                    "chain_iv": np.nan,
                    "chain_bid": np.nan,
                    "chain_ask": np.nan,
                    "chain_last_price": np.nan,
                    "chain_delta_source": "no chain",
                })
                continue

            key = (ticker, matched_exp)
            if key not in cache:
                cache[key] = tk.option_chain(matched_exp).puts.copy()

            puts = cache[key]
            if puts.empty:
                raise RuntimeError("empty puts chain")

            puts["strike_diff"] = (puts["strike"].astype(float) - strike).abs()
            match = puts.sort_values("strike_diff").iloc[0]

            output.append({
                "ticker": ticker,
                "expiration": expiration,
                "strike": strike,
                "matched_expiration": matched_exp,
                "chain_match_note": match_note,
                "matched_strike": float(match.get("strike", np.nan)),
                "strike_match_diff": float(match.get("strike_diff", np.nan)),
                "chain_iv": float(match.get("impliedVolatility", np.nan)),
                "chain_bid": float(match.get("bid", np.nan)),
                "chain_ask": float(match.get("ask", np.nan)),
                "chain_last_price": float(match.get("lastPrice", np.nan)),
                "chain_open_interest": match.get("openInterest", np.nan),
                "chain_volume": match.get("volume", np.nan),
                "chain_delta_source": "Black-Scholes from option-chain IV",
            })
        except Exception as e:
            output.append({
                "ticker": ticker,
                "expiration": expiration,
                "strike": strike,
                "matched_expiration": None,
                "chain_match_note": f"error: {e}",
                "chain_iv": np.nan,
                "chain_bid": np.nan,
                "chain_ask": np.nan,
                "chain_last_price": np.nan,
                "chain_delta_source": "chain fetch error",
            })

    return pd.DataFrame(output)


# -----------------------------
# Black-Scholes delta
# -----------------------------

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def year_fraction_to_expiry(expiration):
    """
    Calendar-day year fraction to option expiration.
    Uses UTC current date. Good enough for delta noise checking.
    """
    try:
        exp_dt = pd.Timestamp(expiration).to_pydatetime().replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max((exp_dt - now).total_seconds() / 86400.0, 0.0)
        return max(days / 365.0, 1.0 / 365.0)
    except Exception:
        return np.nan


def bs_put_delta(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes European long-put delta:
      delta_put = -exp(-qT) * N(-d1)

    For American equity puts this is an approximation, but usually good enough
    for a daily delta sanity check.
    """
    if any(pd.isna(x) for x in [S, K, T, r, sigma, q]):
        return np.nan
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return -math.exp(-q * T) * norm_cdf(-d1)


# -----------------------------
# ITM call spread check
# -----------------------------

def intrinsic_call_spread_value(stock_price, lower_strike, upper_strike, contracts):
    long_intrinsic = max(stock_price - lower_strike, 0.0)
    short_intrinsic = max(stock_price - upper_strike, 0.0)
    spread_intrinsic = long_intrinsic - short_intrinsic
    spread_intrinsic = max(0.0, min(spread_intrinsic, upper_strike - lower_strike))
    return spread_intrinsic * contracts * 100.0


def analyze_itm_call_spreads(options, quotes, min_abs_gap=1000.0):
    calls = options[options["option_type"] == "CALL"].copy()
    qmap = quotes.set_index("ticker").to_dict("index")
    rows = []

    for (ticker, exp), g in calls.groupby(["ticker", "expiration"], dropna=False):
        if ticker not in qmap:
            continue
        last = qmap[ticker].get("last", np.nan)
        prev = qmap[ticker].get("prev_close", np.nan)
        stock_change = qmap[ticker].get("stock_change", np.nan)

        if pd.isna(last) or pd.isna(prev):
            continue

        longs = g[g["quantity"] > 0].copy()
        shorts = g[g["quantity"] < 0].copy()

        for _, long_row in longs.iterrows():
            long_qty = int(abs(long_row["quantity"]))
            lower = float(long_row["strike"])

            candidate_shorts = shorts[shorts["strike"] > lower].copy()
            if candidate_shorts.empty:
                continue

            candidate_shorts["qty_diff"] = (candidate_shorts["quantity"].abs() - long_qty).abs()
            candidate_shorts["strike_gap"] = candidate_shorts["strike"] - lower
            candidate_shorts = candidate_shorts.sort_values(["qty_diff", "strike_gap"])

            short_row = candidate_shorts.iloc[0]
            upper = float(short_row["strike"])
            short_qty = int(abs(short_row["quantity"]))
            contracts = min(long_qty, short_qty)
            width = upper - lower

            if contracts <= 0 or width <= 0:
                continue

            # Constraint: only fully ITM/capped call spreads.
            if not (last > upper):
                continue

            actual_day_pl = float(long_row["day_pl"]) + float(short_row["day_pl"])
            current_intrinsic = intrinsic_call_spread_value(last, lower, upper, contracts)
            previous_intrinsic = intrinsic_call_spread_value(prev, lower, upper, contracts)
            expected_day_pl = current_intrinsic - previous_intrinsic

            gap = actual_day_pl - expected_day_pl
            gross_gap = abs(gap)
            max_value = width * contracts * 100.0

            rows.append({
                "ticker": ticker,
                "expiration": exp,
                "spread": f"{lower:g}/{upper:g} CALL",
                "contracts": contracts,
                "width": width,
                "max_value": max_value,
                "stock_last": last,
                "stock_prev_close": prev,
                "stock_change": stock_change,
                "long_strike": lower,
                "short_strike": upper,
                "long_day_pl": float(long_row["day_pl"]),
                "short_day_pl": float(short_row["day_pl"]),
                "actual_net_day_pl": actual_day_pl,
                "intrinsic_expected_day_pl": expected_day_pl,
                "net_mark_noise": gap,
                "gross_mark_noise": gross_gap,
                "flag": gross_gap >= min_abs_gap,
                "interpretation": "Broker worse than intrinsic" if gap < 0 else "Broker better than intrinsic",
            })

    return pd.DataFrame(rows)


# -----------------------------
# OTM short put check using fetched IV -> BS delta
# -----------------------------

def analyze_otm_short_puts(options, quotes, risk_free_rate=0.045, dividend_yield=0.0, min_abs_gap=1000.0):
    puts = options[(options["option_type"] == "PUT") & (options["quantity"] < 0)].copy()
    qmap = quotes.set_index("ticker").to_dict("index")

    # Keep only OTM short puts first, then fetch only those option-chain rows.
    prelim = []
    for _, row in puts.iterrows():
        ticker = row["ticker"]
        if ticker not in qmap:
            continue

        last = qmap[ticker].get("last", np.nan)
        stock_change = qmap[ticker].get("stock_change", np.nan)
        if pd.isna(last) or pd.isna(stock_change):
            continue

        strike = float(row["strike"])
        if last > strike:
            prelim.append(row)

    if not prelim:
        return pd.DataFrame()

    otm_puts = pd.DataFrame(prelim)
    chain = fetch_short_put_chain_data(otm_puts)

    # Merge chain data back into positions.
    merged = otm_puts.merge(
        chain,
        on=["ticker", "expiration", "strike"],
        how="left",
        suffixes=("", "_chain")
    )

    rows = []
    for _, row in merged.iterrows():
        ticker = row["ticker"]
        last = qmap[ticker].get("last", np.nan)
        prev = qmap[ticker].get("prev_close", np.nan)
        stock_change = qmap[ticker].get("stock_change", np.nan)

        strike = float(row["strike"])
        contracts = int(abs(row["quantity"]))
        T = year_fraction_to_expiry(row.get("matched_expiration") or row.get("expiration"))
        iv = float(row.get("chain_iv", np.nan)) if not pd.isna(row.get("chain_iv", np.nan)) else np.nan

        put_delta_long = bs_put_delta(
            S=float(last),
            K=float(strike),
            T=float(T) if not pd.isna(T) else np.nan,
            r=float(risk_free_rate),
            sigma=iv,
            q=float(dividend_yield),
        )

        if pd.isna(put_delta_long):
            expected_day_pl = np.nan
            gap = np.nan
            gross_gap = np.nan
            flag = False
            note = "Missing/invalid fetched IV; could not compute delta"
        else:
            # Long put P/L approx = delta * dS * contracts * 100
            # Short put P/L is opposite.
            expected_day_pl = -put_delta_long * float(stock_change) * contracts * 100.0
            gap = float(row["day_pl"]) - expected_day_pl
            gross_gap = abs(gap)
            flag = gross_gap >= min_abs_gap
            note = "Broker worse than fetched-delta math" if gap < 0 else "Broker better than fetched-delta math"

        rows.append({
            "ticker": ticker,
            "expiration": row.get("expiration", np.nan),
            "matched_expiration": row.get("matched_expiration", np.nan),
            "chain_match_note": row.get("chain_match_note", np.nan),
            "short_put": f"{strike:g} PUT",
            "contracts": contracts,
            "stock_last": last,
            "stock_prev_close": prev,
            "stock_change": stock_change,
            "strike": strike,
            "csv_delta_for_audit_only": row.get("csv_delta", np.nan),
            "chain_iv_used": iv,
            "time_to_expiry_years": T,
            "put_delta_used_long_option": put_delta_long,
            "delta_source": row.get("chain_delta_source", "Black-Scholes from option-chain IV"),
            "chain_bid": row.get("chain_bid", np.nan),
            "chain_ask": row.get("chain_ask", np.nan),
            "chain_last_price": row.get("chain_last_price", np.nan),
            "chain_open_interest": row.get("chain_open_interest", np.nan),
            "chain_volume": row.get("chain_volume", np.nan),
            "actual_day_pl": float(row["day_pl"]),
            "delta_expected_day_pl": expected_day_pl,
            "net_delta_noise": gap,
            "gross_delta_noise": gross_gap,
            "flag": flag,
            "interpretation": note,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Summary/output
# -----------------------------

def make_summary(call_df, put_df):
    """
    Reports:
      - net_noise_all: broker actual P/L minus model expected P/L, summed across rows.
      - negative_row_addback_all: sum of only rows where broker is worse than model.
        This is the "mental add-back" for bad marks hurting today's portfolio.
      - positive_row_giveback_all: sum of rows where broker looks better than model.
      - gross_noise_all: absolute noise across rows.

    Important:
      If net_noise_all is positive, a bucket-level add-back would be $0 even if many
      individual spreads are hurting you. That is why the main add-back uses row-level
      negative noise, not bucket-level netting.
    """
    rows = []

    def safe_sum(df, col):
        if df is None or df.empty or col not in df.columns:
            return 0.0
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())

    def add_row(name, df, actual_col, expected_col, net_col, gross_col):
        empty_row = {
            "bucket": name,
            "count_all_valid": 0,
            "count_flagged": 0,
            "actual_day_pl_all": 0.0,
            "expected_day_pl_all": 0.0,
            "net_noise_all": 0.0,
            "negative_row_addback_all": 0.0,
            "positive_row_giveback_all": 0.0,
            "gross_noise_all": 0.0,
            "net_noise_flagged": 0.0,
            "negative_row_addback_flagged": 0.0,
            "positive_row_giveback_flagged": 0.0,
            "gross_noise_flagged": 0.0,
        }

        if df is None or df.empty:
            rows.append(empty_row)
            return

        valid = df.dropna(subset=[net_col]).copy()
        if valid.empty:
            rows.append(empty_row)
            return

        valid[net_col] = pd.to_numeric(valid[net_col], errors="coerce")
        valid[gross_col] = pd.to_numeric(valid[gross_col], errors="coerce").fillna(valid[net_col].abs())

        flagged = valid[valid["flag"] == True] if "flag" in valid.columns else valid.iloc[0:0]

        def negative_addback(xdf):
            return float((-xdf.loc[xdf[net_col] < 0, net_col]).sum())

        def positive_giveback(xdf):
            return float((xdf.loc[xdf[net_col] > 0, net_col]).sum())

        rows.append({
            "bucket": name,
            "count_all_valid": len(valid),
            "count_flagged": int(flagged["flag"].sum()) if "flag" in flagged.columns and not flagged.empty else 0,
            "actual_day_pl_all": safe_sum(valid, actual_col),
            "expected_day_pl_all": safe_sum(valid, expected_col),

            # net of positive and negative mark noise
            "net_noise_all": safe_sum(valid, net_col),

            # THIS is the add-back you asked for: only Schwab-worse rows.
            "negative_row_addback_all": negative_addback(valid),

            # Rows where Schwab is flattering the P/L vs model.
            "positive_row_giveback_all": positive_giveback(valid),

            # Gross absolute sloshing.
            "gross_noise_all": safe_sum(valid, gross_col),

            "net_noise_flagged": safe_sum(flagged, net_col),
            "negative_row_addback_flagged": negative_addback(flagged),
            "positive_row_giveback_flagged": positive_giveback(flagged),
            "gross_noise_flagged": safe_sum(flagged, gross_col),
        })

    add_row(
        "ITM/capped call spreads, intrinsic only — no delta",
        call_df,
        "actual_net_day_pl",
        "intrinsic_expected_day_pl",
        "net_mark_noise",
        "gross_mark_noise",
    )
    add_row(
        "OTM short puts, fetched delta only",
        put_df,
        "actual_day_pl",
        "delta_expected_day_pl",
        "net_delta_noise",
        "gross_delta_noise",
    )

    summary = pd.DataFrame(rows)
    total = {
        "bucket": "TOTAL",
        "count_all_valid": summary["count_all_valid"].sum(),
        "count_flagged": summary["count_flagged"].sum(),
        "actual_day_pl_all": summary["actual_day_pl_all"].sum(),
        "expected_day_pl_all": summary["expected_day_pl_all"].sum(),
        "net_noise_all": summary["net_noise_all"].sum(),
        "negative_row_addback_all": summary["negative_row_addback_all"].sum(),
        "positive_row_giveback_all": summary["positive_row_giveback_all"].sum(),
        "gross_noise_all": summary["gross_noise_all"].sum(),
        "net_noise_flagged": summary["net_noise_flagged"].sum(),
        "negative_row_addback_flagged": summary["negative_row_addback_flagged"].sum(),
        "positive_row_giveback_flagged": summary["positive_row_giveback_flagged"].sum(),
        "gross_noise_flagged": summary["gross_noise_flagged"].sum(),
    }

    return pd.concat([summary, pd.DataFrame([total])], ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Broker holdings CSV file")
    parser.add_argument("--outdir", default=".", help="Output directory")
    parser.add_argument("--min-gap", type=float, default=1000.0, help="Minimum absolute gap to flag")
    parser.add_argument("--risk-free-rate", type=float, default=0.045, help="Annual risk-free rate for Black-Scholes delta, e.g. 0.045")
    parser.add_argument("--dividend-yield", type=float, default=0.0, help="Annual dividend yield assumption, e.g. 0.01")
    args = parser.parse_args()

    in_path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = read_broker_csv(in_path)
    options = standardize_holdings(raw)

    if options.empty:
        raise RuntimeError("No option rows found after parsing. Check CSV column names/format.")

    tickers = sorted(options["ticker"].dropna().unique())
    quotes = get_quotes_yfinance(tickers)

    call_noise = analyze_itm_call_spreads(options, quotes, min_abs_gap=args.min_gap)
    put_noise = analyze_otm_short_puts(
        options,
        quotes,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
        min_abs_gap=args.min_gap,
    )
    summary = make_summary(call_noise, put_noise)

    call_csv = outdir / "call_spread_noise.csv"
    put_csv = outdir / "otm_short_put_noise.csv"
    summary_csv = outdir / "noise_summary.csv"
    xlsx = outdir / "portfolio_noise_report.xlsx"

    call_noise.to_csv(call_csv, index=False)
    put_noise.to_csv(put_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        call_noise.to_excel(writer, index=False, sheet_name="ITM Call Spreads")
        put_noise.to_excel(writer, index=False, sheet_name="OTM Short Puts")
        quotes.to_excel(writer, index=False, sheet_name="Underlying Quotes")
        options.to_excel(writer, index=False, sheet_name="Parsed Options")

    print("\nDone.\n")
    print(summary.to_string(index=False))
    print("\nFiles written:")
    print(f"  {call_csv}")
    print(f"  {put_csv}")
    print(f"  {summary_csv}")
    print(f"  {xlsx}")

    total_addback_all = float(summary.loc[summary["bucket"] == "TOTAL", "negative_row_addback_all"].iloc[0])
    total_addback_flagged = float(summary.loc[summary["bucket"] == "TOTAL", "negative_row_addback_flagged"].iloc[0])
    total_net_noise = float(summary.loc[summary["bucket"] == "TOTAL", "net_noise_all"].iloc[0])

    call_row = summary[summary["bucket"].astype(str).str.contains("call spreads", case=False, na=False)]
    put_row = summary[summary["bucket"].astype(str).str.contains("short puts", case=False, na=False)]

    call_addback_all = float(call_row["negative_row_addback_all"].iloc[0]) if not call_row.empty else 0.0
    call_addback_flagged = float(call_row["negative_row_addback_flagged"].iloc[0]) if not call_row.empty else 0.0
    call_net_noise = float(call_row["net_noise_all"].iloc[0]) if not call_row.empty else 0.0
    call_positive = float(call_row["positive_row_giveback_all"].iloc[0]) if not call_row.empty else 0.0
    call_count_all = int(call_row["count_all_valid"].iloc[0]) if not call_row.empty else 0
    call_count_flagged = int(call_row["count_flagged"].iloc[0]) if not call_row.empty else 0

    put_addback_all = float(put_row["negative_row_addback_all"].iloc[0]) if not put_row.empty else 0.0
    put_addback_flagged = float(put_row["negative_row_addback_flagged"].iloc[0]) if not put_row.empty else 0.0
    put_net_noise = float(put_row["net_noise_all"].iloc[0]) if not put_row.empty else 0.0
    put_positive = float(put_row["positive_row_giveback_all"].iloc[0]) if not put_row.empty else 0.0
    put_count_all = int(put_row["count_all_valid"].iloc[0]) if not put_row.empty else 0
    put_count_flagged = int(put_row["count_flagged"].iloc[0]) if not put_row.empty else 0

    print(f"\nBREAKDOWN — ROW-LEVEL NEGATIVE ADD-BACK, ALL valid rows:")
    print(f"  ITM call-spread intrinsic add-back: ${call_addback_all:,.2f}  ({call_count_all} spreads)")
    print(f"  OTM short-put delta add-back:       ${put_addback_all:,.2f}  ({put_count_all} puts)")
    print(f"  TOTAL negative add-back:            ${total_addback_all:,.2f}")

    print(f"\nNETTING CHECK:")
    print(f"  Call spread net noise after positive offsets: ${call_net_noise:,.2f}")
    print(f"  Call spread positive give-back rows:          ${call_positive:,.2f}")
    print(f"  Put net noise after positive offsets:         ${put_net_noise:,.2f}")
    print(f"  Put positive give-back rows:                  ${put_positive:,.2f}")
    print(f"  TOTAL net noise after offsets:                ${total_net_noise:,.2f}")

    print(f"\nBREAKDOWN — FLAGGED rows only:")
    print(f"  ITM call-spread intrinsic add-back: ${call_addback_flagged:,.2f}  ({call_count_flagged} spreads)")
    print(f"  OTM short-put delta add-back:       ${put_addback_flagged:,.2f}  ({put_count_flagged} puts)")
    print(f"  TOTAL flagged negative add-back:    ${total_addback_flagged:,.2f}")

    print("\nMethod:")
    print("  Call spreads: intrinsic-value check only; NO call delta is used.")
    print("  Short puts: fetched option-chain IV -> Black-Scholes put delta; only puts use delta.")
    print("\nUse 'negative_row_addback_all' for the mental add-back.")
    print("Use 'net_noise_all' only if you want to net positive and negative mark errors together.")


if __name__ == "__main__":
    main()
