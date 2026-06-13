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
    - stock price > short call strike
    - stock day change > 0
    - net spread day P/L < 0

PUTS:
  Check only:
    - short puts
    - stock price > put strike
  Expected P/L = abs(put_delta) * stock_change * contracts * 100

Delta:
  - uses CSV/broker Delta column if present
  - otherwise falls back to yfinance option-chain IV + Black-Scholes put delta

Run:
  pip install pandas yfinance numpy openpyxl
  python final_portfolio_noise_checker_v2.py my_holdings.csv
"""

import argparse
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


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


def find_bad_itm_upday_call_spreads(options, quotes):
    calls = options[options["option_type"] == "CALL"].copy()
    qmap = quotes.set_index("ticker").to_dict("index")
    rows = []

    for (ticker, exp), g in calls.groupby(["ticker", "expiration"], dropna=False):
        q = qmap.get(ticker)
        if not q:
            continue
        last, prev, stock_chg = q.get("last"), q.get("prev_close"), q.get("stock_change")
        if pd.isna(last) or pd.isna(prev) or pd.isna(stock_chg) or stock_chg <= 0:
            continue

        longs = [
            {
                "strike": float(row["strike"]),
                "remaining_qty": int(abs(row["quantity"])),
                "original_qty": int(abs(row["quantity"])),
                "day_pl": float(row["day_pl"]),
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
                if contracts <= 0 or last <= upper:
                    continue

                lower = long_leg["strike"]
                long_day_pl = long_leg["day_pl"] * contracts / long_leg["original_qty"]
                short_day_pl = short_leg["day_pl"] * contracts / short_leg["original_qty"]
                actual = long_day_pl + short_day_pl

                if actual < 0:
                    expected = call_spread_intrinsic(last, lower, upper, contracts) - call_spread_intrinsic(prev, lower, upper, contracts)
                    diff = expected - actual

                    rows.append({
                        "ticker": ticker,
                        "expiration": exp,
                        "spread": f"{lower:g}/{upper:g} C",
                        "contracts": contracts,
                        "stock_change": stock_chg,
                        "long_day_pl": long_day_pl,
                        "short_day_pl": short_day_pl,
                        "schwab_net_day_pl": actual,
                        "intrinsic_expected_day_pl": expected,
                        "diff_to_add_back": diff,
                    })

                long_leg["remaining_qty"] -= contracts
                remaining_short -= contracts

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("diff_to_add_back", ascending=False)
    return out


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def year_frac(exp):
    try:
        e = pd.Timestamp(exp).to_pydatetime().replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max((e - now).total_seconds() / 86400.0 / 365.0, 1 / 365)
    except Exception:
        return np.nan


def bs_put_delta(S, K, T, r, sigma, q=0.0):
    if any(pd.isna(v) for v in [S, K, T, r, sigma, q]) or S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return -math.exp(-q * T) * norm_cdf(-d1)


def normalize_put_delta(d):
    if pd.isna(d):
        return np.nan
    d = float(d)
    if abs(d) > 1.5 and abs(d) <= 100:
        d /= 100.0
    d = -abs(d)
    return d if abs(d) <= 1.05 else np.nan


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


def check_otm_short_puts(options, quotes, risk_free_rate=0.045, dividend_yield=0.0, prefer_csv_delta=True):
    puts = options[(options["option_type"] == "PUT") & (options["quantity"] < 0)].copy()
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--risk-free-rate", type=float, default=0.045)
    ap.add_argument("--dividend-yield", type=float, default=0.0)
    ap.add_argument("--use-yf-delta-only", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw = read_broker_csv(args.csv)
    options = standardize(raw)
    quotes = get_quotes(sorted(options["ticker"].unique()))

    bad_calls = find_bad_itm_upday_call_spreads(options, quotes)
    puts = check_otm_short_puts(
        options,
        quotes,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
        prefer_csv_delta=(not args.use_yf_delta_only),
    )

    call_addback = 0.0 if bad_calls.empty else float(pd.to_numeric(bad_calls["diff_to_add_back"], errors="coerce").fillna(0).sum())
    if puts.empty:
        put_addback = 0.0
    else:
        puts["positive_addback"] = pd.to_numeric(puts["diff_to_add_back"], errors="coerce").clip(lower=0)
        put_addback = float(puts["positive_addback"].fillna(0).sum())

    total = call_addback + put_addback
    summary = pd.DataFrame([
        {"bucket": "Bad ITM up-day call spreads", "count": len(bad_calls), "addback": call_addback},
        {"bucket": "OTM short puts delta check", "count": len(puts), "addback": put_addback},
        {"bucket": "TOTAL", "count": len(bad_calls) + len(puts), "addback": total},
    ])

    bad_calls_path = outdir / "bad_itm_upday_call_spreads.csv"
    puts_path = outdir / "otm_short_put_delta_check.csv"
    summary_path = outdir / "final_noise_summary.csv"
    xlsx_path = outdir / "final_portfolio_noise_report.xlsx"

    bad_calls.to_csv(bad_calls_path, index=False)
    puts.to_csv(puts_path, index=False)
    summary.to_csv(summary_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summary.to_excel(w, index=False, sheet_name="Summary")
        bad_calls.to_excel(w, index=False, sheet_name="Bad ITM Up-Day Calls")
        puts.to_excel(w, index=False, sheet_name="OTM Short Puts")
        quotes.to_excel(w, index=False, sheet_name="Quotes")
        options.to_excel(w, index=False, sheet_name="Parsed Options")

    print("\nFINAL PORTFOLIO NOISE CHECK v2")
    print("=" * 72)

    print("\nCALL SPREAD RULE:")
    print("  ITM call spreads only, stock up, but spread net day P/L is negative.")
    if bad_calls.empty:
        print("  No bad ITM up-day call spreads found.")
    else:
        cols = ["ticker", "expiration", "spread", "contracts", "stock_change", "schwab_net_day_pl", "intrinsic_expected_day_pl", "diff_to_add_back"]
        print(bad_calls[cols].to_string(index=False))
        print(f"\n  Call-spread add-back: ${call_addback:,.2f}")

    print("\nSHORT PUT RULE:")
    print("  OTM short puts only, expected P/L = abs(delta) × stock_change × contracts × 100.")
    if puts.empty:
        print("  No OTM short puts found.")
    else:
        cols = ["ticker", "expiration", "short_put", "contracts", "stock_change", "delta_used_abs", "delta_source", "schwab_day_pl", "delta_expected_day_pl", "diff_to_add_back"]
        print(puts[cols].to_string(index=False))
        print(f"\n  Short-put positive add-back: ${put_addback:,.2f}")

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nTOTAL CLEAN ADD-BACK: ${total:,.2f}")

    print("\nFiles written:")
    print(f"  {bad_calls_path}")
    print(f"  {puts_path}")
    print(f"  {summary_path}")
    print(f"  {xlsx_path}")


if __name__ == "__main__":
    main()
