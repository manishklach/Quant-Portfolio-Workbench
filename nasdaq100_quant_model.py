#!/usr/bin/env python3
"""Rank Nasdaq-100 stocks with a trend-plus-momentum model and backtest it.

Examples:
  python nasdaq100_quant_model.py scan
  python nasdaq100_quant_model.py scan --top 15 --csv my_holdings.csv
  python nasdaq100_quant_model.py backtest --years 5 --top 12
  python nasdaq100_quant_model.py universe
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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


NASDAQ100_COMPANIES_URL = "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"
USER_AGENT = "Mozilla/5.0"
LOOKBACK_BUFFER_DAYS = 260


@dataclass
class ModelConfig:
    top_n: int = 12
    min_score: float = 60.0
    sell_rank_buffer: int = 8


def fetch_nasdaq100_constituents() -> list[str]:
    if requests is None:
        raise RuntimeError("requests not installed. Run: pip install requests")

    response = requests.get(
        NASDAQ100_COMPANIES_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    html = response.text

    match = re.search(r"<table.*?<tbody>(.*?)</tbody></table>.*?Last updated", html, flags=re.S | re.I)
    if not match:
        raise RuntimeError("Could not parse Nasdaq-100 constituents from Nasdaq's public companies page.")

    block = match.group(1)
    tickers: list[str] = []
    for raw_ticker in re.findall(r"<tr[^>]*>\s*<td[^>]*>\s*([A-Z.\-]{1,10})\s*</td>\s*<td[^>]*>", block, flags=re.I):
        ticker = raw_ticker.strip().upper()
        if ticker != "SYMBOL":
            tickers.append(ticker)

    deduped: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        if ticker not in seen:
            seen.add(ticker)
            deduped.append(ticker)

    if len(deduped) < 95:
        raise RuntimeError(f"Parsed only {len(deduped)} Nasdaq-100 tickers, which looks incomplete.")
    return deduped


def fetch_price_history(tickers: list[str], years: int) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    period_days = max(int(years * 365.25) + LOOKBACK_BUFFER_DAYS, 450)
    tickers_arg = " ".join(tickers)
    data = yf.download(
        tickers=tickers_arg,
        period=f"{period_days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    if data.empty:
        raise RuntimeError("No price history returned from yfinance.")
    return data


def safe_float(value) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fetch_fundamental_snapshots(tickers: list[str]) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not installed. Run: pip install yfinance")

    rows: list[dict[str, object]] = []
    for ticker in tickers:
        info = {}
        fast = {}
        try:
            tk = yf.Ticker(ticker)
            try:
                info = tk.info or {}
            except Exception:
                info = {}
            try:
                fast = dict(tk.fast_info)
            except Exception:
                fast = {}
        except Exception:
            info = {}
            fast = {}

        market_cap = safe_float(info.get("marketCap"))
        if pd.isna(market_cap):
            market_cap = safe_float(fast.get("marketCap"))

        free_cash_flow = safe_float(info.get("freeCashflow"))
        total_revenue = safe_float(info.get("totalRevenue"))

        p_to_fcf = float("nan")
        if not pd.isna(market_cap) and not pd.isna(free_cash_flow) and free_cash_flow > 0:
            p_to_fcf = market_cap / free_cash_flow

        fcf_margin = float("nan")
        if not pd.isna(free_cash_flow) and not pd.isna(total_revenue) and total_revenue > 0:
            fcf_margin = free_cash_flow / total_revenue

        rows.append(
            {
                "ticker": ticker,
                "forward_pe": safe_float(info.get("forwardPE")),
                "price_to_sales_ttm": safe_float(info.get("priceToSalesTrailing12Months")),
                "enterprise_to_revenue": safe_float(info.get("enterpriseToRevenue")),
                "p_to_fcf": p_to_fcf,
                "revenue_growth": safe_float(info.get("revenueGrowth")),
                "earnings_growth": safe_float(info.get("earningsGrowth")),
                "gross_margin": safe_float(info.get("grossMargins")),
                "operating_margin": safe_float(info.get("operatingMargins")),
                "profit_margin": safe_float(info.get("profitMargins")),
                "fcf_margin": fcf_margin,
                "return_on_equity": safe_float(info.get("returnOnEquity")),
                "debt_to_equity": safe_float(info.get("debtToEquity")),
                "market_cap": market_cap,
                "fundamentals_available": bool(info),
            }
        )

    return pd.DataFrame(rows)


def extract_ohlcv(history: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(history.columns, pd.MultiIndex):
        if ticker not in history.columns.get_level_values(0):
            return pd.DataFrame()
        frame = history[ticker].copy()
    else:
        frame = history.copy()

    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected.difference(frame.columns)
    for col in missing:
        frame[col] = np.nan
    frame = frame[list(expected)].sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.dropna(subset=["Close"])


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["Close"].shift(1)
    tr = pd.concat(
        [
            frame["High"] - frame["Low"],
            (frame["High"] - prev_close).abs(),
            (frame["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def build_indicator_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    close = out["Close"]
    out["ema8"] = close.ewm(span=8, adjust=False).mean()
    out["ema21"] = close.ewm(span=21, adjust=False).mean()
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()
    out["rsi14"] = compute_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    out["bb_mid"] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = out["bb_mid"] + 2.0 * bb_std
    out["bb_lower"] = out["bb_mid"] - 2.0 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0.0, np.nan)
    out["bb_width_median_20"] = out["bb_width"].rolling(20).median()

    out["atr14"] = compute_atr(out, 14)
    out["ret_21"] = close.pct_change(21)
    out["ret_63"] = close.pct_change(63)
    out["ret_126"] = close.pct_change(126)
    return out


def percentile_from_series(series: pd.Series) -> pd.Series:
    ranked = series.rank(pct=True, method="average")
    return (ranked * 100.0).fillna(50.0)


def inverse_percentile_from_series(series: pd.Series) -> pd.Series:
    clean = series.copy()
    clean = clean.where(clean > 0)
    return (100.0 - percentile_from_series(clean)).fillna(50.0)


def compute_regime_snapshot(qqq: pd.DataFrame) -> dict[str, object]:
    latest = qqq.iloc[-1]
    risk_on = bool(
        latest["Close"] > latest["sma200"]
        and latest["sma50"] > latest["sma200"]
        and latest["Close"] > latest["ema21"]
    )
    return {
        "date": qqq.index[-1].date().isoformat(),
        "qqq_close": float(latest["Close"]),
        "qqq_ema21": float(latest["ema21"]),
        "qqq_sma50": float(latest["sma50"]),
        "qqq_sma200": float(latest["sma200"]),
        "risk_on": risk_on,
    }


def build_cross_section(indicators: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, frame in indicators.items():
        if as_of not in frame.index:
            continue
        latest = frame.loc[as_of]
        if latest[["Close", "ema21", "sma50", "sma200"]].isna().any():
            continue
        rows.append(
            {
                "ticker": ticker,
                "close": float(latest["Close"]),
                "ema8": float(latest["ema8"]),
                "ema21": float(latest["ema21"]),
                "sma50": float(latest["sma50"]),
                "sma200": float(latest["sma200"]),
                "rsi14": float(latest["rsi14"]),
                "macd_hist": float(latest["macd_hist"]),
                "bb_mid": float(latest["bb_mid"]) if not pd.isna(latest["bb_mid"]) else np.nan,
                "bb_upper": float(latest["bb_upper"]) if not pd.isna(latest["bb_upper"]) else np.nan,
                "bb_width": float(latest["bb_width"]) if not pd.isna(latest["bb_width"]) else np.nan,
                "bb_width_median_20": float(latest["bb_width_median_20"]) if not pd.isna(latest["bb_width_median_20"]) else np.nan,
                "atr14": float(latest["atr14"]) if not pd.isna(latest["atr14"]) else np.nan,
                "ret_21": float(latest["ret_21"]) if not pd.isna(latest["ret_21"]) else np.nan,
                "ret_63": float(latest["ret_63"]) if not pd.isna(latest["ret_63"]) else np.nan,
                "ret_126": float(latest["ret_126"]) if not pd.isna(latest["ret_126"]) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f"No valid cross-section could be built for {as_of.date()}.")
    return out


def score_cross_section(cross_section: pd.DataFrame) -> pd.DataFrame:
    df = cross_section.copy()

    trend_score = pd.Series(0.0, index=df.index)
    trend_score += np.where(df["close"] > df["ema8"], 15.0, 0.0)
    trend_score += np.where(df["close"] > df["ema21"], 20.0, 0.0)
    trend_score += np.where(df["ema8"] > df["ema21"], 15.0, 0.0)
    trend_score += np.where(df["ema21"] > df["sma50"], 20.0, 0.0)
    trend_score += np.where(df["sma50"] > df["sma200"], 20.0, 0.0)
    trend_score += np.where(df["close"] > df["sma200"], 10.0, 0.0)
    df["trend_score"] = trend_score.clip(0.0, 100.0)

    rs_mix = 0.2 * df["ret_21"].fillna(0.0) + 0.5 * df["ret_63"].fillna(0.0) + 0.3 * df["ret_126"].fillna(0.0)
    df["rs_score"] = percentile_from_series(rs_mix)

    confirm_score = pd.Series(0.0, index=df.index)
    confirm_score += np.where(df["macd_hist"] > 0.0, 40.0, 0.0)
    confirm_score += np.where((df["rsi14"] >= 50.0) & (df["rsi14"] <= 70.0), 30.0, 0.0)
    confirm_score += np.where(df["close"] > df["bb_mid"], 15.0, 0.0)
    confirm_score += np.where(df["bb_width"] > df["bb_width_median_20"], 15.0, 0.0)
    df["confirm_score"] = confirm_score.clip(0.0, 100.0)

    penalty = pd.Series(0.0, index=df.index)
    penalty += np.where(df["rsi14"] > 75.0, 35.0, 0.0)
    penalty += np.where((df["bb_upper"].notna()) & (df["close"] > df["bb_upper"] * 1.01), 30.0, 0.0)
    penalty += np.where(
        (df["atr14"].notna()) & (df["ema21"] > 0.0) & ((df["close"] - df["ema21"]) / df["atr14"].replace(0.0, np.nan) > 2.5),
        35.0,
        0.0,
    )
    df["penalty_score"] = penalty.clip(0.0, 100.0)

    df["final_score"] = (
        0.35 * df["trend_score"]
        + 0.30 * df["rs_score"]
        + 0.20 * df["confirm_score"]
        - 0.15 * df["penalty_score"]
    )
    df["technical_score"] = df["final_score"]
    df["rank"] = df["final_score"].rank(ascending=False, method="first")
    return df.sort_values(["final_score", "rs_score"], ascending=[False, False]).reset_index(drop=True)


def attach_fundamental_scores(df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(fundamentals, on="ticker", how="left")

    growth_mix = (
        0.45 * percentile_from_series(out["revenue_growth"])
        + 0.55 * percentile_from_series(out["earnings_growth"])
    )
    out["growth_score"] = growth_mix.fillna(50.0)

    quality_mix = (
        0.30 * percentile_from_series(out["gross_margin"])
        + 0.25 * percentile_from_series(out["operating_margin"])
        + 0.20 * percentile_from_series(out["profit_margin"])
        + 0.15 * percentile_from_series(out["fcf_margin"])
        + 0.10 * percentile_from_series(out["return_on_equity"])
    )
    out["quality_score"] = quality_mix.fillna(50.0)

    valuation_mix = (
        0.40 * inverse_percentile_from_series(out["forward_pe"])
        + 0.30 * inverse_percentile_from_series(out["price_to_sales_ttm"])
        + 0.30 * inverse_percentile_from_series(out["p_to_fcf"])
    )
    out["valuation_score"] = valuation_mix.fillna(50.0)

    debt_score = inverse_percentile_from_series(out["debt_to_equity"]).fillna(50.0)
    out["risk_quality_score"] = (0.65 * out["quality_score"] + 0.35 * debt_score).fillna(50.0)
    out["fundamental_score"] = (
        0.45 * out["growth_score"]
        + 0.30 * out["quality_score"]
        + 0.25 * out["valuation_score"]
    ).fillna(50.0)

    out["composite_score"] = (
        0.50 * out["technical_score"]
        + 0.35 * out["fundamental_score"]
        + 0.15 * out["risk_quality_score"]
    )
    out["final_score"] = out["composite_score"]
    out["rank"] = out["final_score"].rank(ascending=False, method="first")
    return out.sort_values(["final_score", "technical_score"], ascending=[False, False]).reset_index(drop=True)


def classify_actions(df: pd.DataFrame, config: ModelConfig, held_tickers: set[str], risk_on: bool) -> pd.DataFrame:
    out = df.copy()
    out["held"] = out["ticker"].isin(held_tickers)
    out["action"] = "WATCH"
    buy_mask = risk_on & (out["rank"] <= config.top_n) & (out["final_score"] >= config.min_score)
    out.loc[buy_mask & (~out["held"]), "action"] = "BUY"
    out.loc[buy_mask & out["held"], "action"] = "HOLD"

    sell_mask = out["held"] & (
        (out["rank"] > (config.top_n + config.sell_rank_buffer))
        | (out["close"] < out["ema21"])
        | ((out["macd_hist"] < 0.0) & (out["rsi14"] < 45.0))
    )
    out.loc[sell_mask, "action"] = "SELL"

    if not risk_on:
        held_only = out["held"]
        out.loc[held_only, "action"] = np.where(sell_mask.loc[held_only], "SELL", "REDUCE")
        out.loc[~out["held"], "action"] = "WAIT"
    return out


def load_held_tickers(csv_path: str | None, script_file: str) -> set[str]:
    if not csv_path:
        return set()
    holdings = load_schwab_holdings(default_csv_path(csv_path, script_file))
    tickers = {
        str(value).strip().upper()
        for value in holdings["Underlying"].dropna().tolist()
        if re.fullmatch(r"[A-Z.\-]{1,10}", str(value).strip().upper())
    }
    return tickers


def print_scan_report(scored: pd.DataFrame, regime: dict[str, object], top_n: int) -> None:
    print("Nasdaq-100 Quant Model")
    print("=" * 100)
    print(
        f"As of {regime['date']} | QQQ={regime['qqq_close']:.2f} | "
        f"21EMA={regime['qqq_ema21']:.2f} | 50MA={regime['qqq_sma50']:.2f} | "
        f"200MA={regime['qqq_sma200']:.2f} | Regime={'RISK-ON' if regime['risk_on'] else 'RISK-OFF'}"
    )

    display_cols = [
        "ticker",
        "action",
        "held",
        "rank",
        "final_score",
        "technical_score",
        "fundamental_score",
        "risk_quality_score",
        "rsi14",
        "forward_pe",
        "price_to_sales_ttm",
        "p_to_fcf",
        "revenue_growth",
        "earnings_growth",
        "macd_hist",
        "close",
        "ema21",
        "sma50",
        "sma200",
    ]

    print("\nTop Ranked")
    print(scored.head(top_n)[display_cols].to_string(index=False, justify="right", float_format=lambda x: f"{x:0.2f}"))

    buy_like = scored[scored["action"].isin(["BUY", "HOLD"])]
    sell_like = scored[scored["action"].isin(["SELL", "REDUCE"])]
    if not buy_like.empty:
        print("\nBuy / Hold Candidates")
        print(buy_like.head(max(top_n, 10))[display_cols].to_string(index=False, justify="right", float_format=lambda x: f"{x:0.2f}"))
    if not sell_like.empty:
        print("\nSell / Reduce Candidates")
        print(sell_like[display_cols].to_string(index=False, justify="right", float_format=lambda x: f"{x:0.2f}"))

    print("\nWeakest Ranked")
    print(scored.tail(min(top_n, len(scored)))[display_cols].to_string(index=False, justify="right", float_format=lambda x: f"{x:0.2f}"))


def build_indicators_for_universe(history: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    indicators: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = extract_ohlcv(history, ticker)
        if len(frame) < 220:
            continue
        indicators[ticker] = build_indicator_frame(frame)
    if not indicators:
        raise RuntimeError("No indicators could be built for the selected universe.")
    return indicators


def scan_model(years: int, top_n: int, csv_path: str | None, output: str | None) -> int:
    tickers = fetch_nasdaq100_constituents()
    history = fetch_price_history(sorted(set(tickers + ["QQQ"])), years)
    indicators = build_indicators_for_universe(history, tickers)
    qqq = build_indicator_frame(extract_ohlcv(history, "QQQ"))
    regime = compute_regime_snapshot(qqq)
    as_of = qqq.index[-1]
    scored = score_cross_section(build_cross_section(indicators, as_of))
    held_tickers = load_held_tickers(csv_path, __file__)
    candidate_count = max(top_n * 2, 24)
    candidate_tickers = set(scored.head(candidate_count)["ticker"].tolist())
    candidate_tickers.update(ticker for ticker in held_tickers if ticker in indicators)
    fundamentals = fetch_fundamental_snapshots(sorted(candidate_tickers))
    scored = attach_fundamental_scores(scored, fundamentals)
    scored = classify_actions(scored, ModelConfig(top_n=top_n), held_tickers, bool(regime["risk_on"]))
    print(f"Fundamental candidate set size: {len(candidate_tickers)}")
    print_scan_report(scored, regime, top_n)

    if output:
        output_path = Path(output)
        scored.to_csv(output_path, index=False)
        print(f"\nScan output written to: {output_path}")
    return 0


def compute_performance_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().fillna(0.0)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())
    vol = float(returns.std(ddof=0) * math.sqrt(52.0))
    sharpe = float((returns.mean() / returns.std(ddof=0)) * math.sqrt(52.0)) if returns.std(ddof=0) > 0 else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "annual_volatility": vol,
        "sharpe": sharpe,
    }


def backtest_model(years: int, top_n: int, output: str | None) -> int:
    tickers = fetch_nasdaq100_constituents()
    history = fetch_price_history(sorted(set(tickers + ["QQQ"])), years + 1)
    indicators = build_indicators_for_universe(history, tickers)
    qqq = build_indicator_frame(extract_ohlcv(history, "QQQ"))
    common_dates = qqq.index
    for frame in indicators.values():
        common_dates = common_dates.intersection(frame.index)
    common_dates = common_dates.sort_values()
    if len(common_dates) < 260:
        raise RuntimeError("Not enough common history to run the backtest.")

    rebalance_dates = pd.Series(common_dates, index=common_dates).resample("W-FRI").last().dropna()
    rebalance_dates = pd.DatetimeIndex([d for d in rebalance_dates.tolist() if d in common_dates])
    records: list[dict[str, object]] = []
    equity = 1.0
    equity_curve = []

    close_map = {ticker: frame["Close"].reindex(common_dates) for ticker, frame in indicators.items()}
    qqq_close = qqq["Close"].reindex(common_dates)

    for start_date, end_date in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        regime_now = compute_regime_snapshot(qqq.loc[:start_date])
        cross_section = score_cross_section(build_cross_section(indicators, start_date))
        chosen = cross_section.head(top_n)
        chosen = chosen[chosen["final_score"] >= 60.0]
        selected_tickers = chosen["ticker"].tolist() if regime_now["risk_on"] else []

        if selected_tickers:
            next_returns = []
            for ticker in selected_tickers:
                series = close_map[ticker]
                start_px = series.loc[start_date]
                end_px = series.loc[end_date]
                if pd.isna(start_px) or pd.isna(end_px) or start_px == 0:
                    continue
                next_returns.append(float(end_px / start_px - 1.0))
            portfolio_return = float(np.mean(next_returns)) if next_returns else 0.0
        else:
            portfolio_return = 0.0

        qqq_return = float(qqq_close.loc[end_date] / qqq_close.loc[start_date] - 1.0)
        equity *= (1.0 + portfolio_return)
        equity_curve.append({"date": end_date, "equity": equity})
        records.append(
            {
                "rebalance_date": start_date.date().isoformat(),
                "next_date": end_date.date().isoformat(),
                "risk_on": regime_now["risk_on"],
                "selected_count": len(selected_tickers),
                "selected_tickers": ",".join(selected_tickers),
                "portfolio_return": portfolio_return,
                "qqq_return": qqq_return,
                "equity": equity,
            }
        )

    results = pd.DataFrame(records)
    equity_series = pd.Series(
        [1.0] + [row["equity"] for row in equity_curve],
        index=pd.DatetimeIndex([rebalance_dates[0]] + [row["date"] for row in equity_curve]),
        dtype=float,
    )
    benchmark_series = (1.0 + results.set_index(pd.to_datetime(results["next_date"]))["qqq_return"]).cumprod()
    benchmark_series = pd.concat([pd.Series([1.0], index=[rebalance_dates[0]]), benchmark_series])

    metrics = compute_performance_metrics(equity_series)
    benchmark_metrics = compute_performance_metrics(benchmark_series)

    print("Nasdaq-100 Quant Backtest")
    print("=" * 100)
    print(f"Universe source: {NASDAQ100_COMPANIES_URL}")
    print("Warning: this backtest uses the current public constituent list, so it has survivorship bias.")
    print("Warning: the backtest is currently technical-only; live scan mode adds fundamentals separately.")
    print(f"Rebalance: Weekly | Top N: {top_n} | Years requested: {years}")
    print("\nStrategy Metrics")
    print(f"  Total Return: {metrics['total_return'] * 100.0:,.2f}%")
    print(f"  CAGR: {metrics['cagr'] * 100.0:,.2f}%")
    print(f"  Max Drawdown: {metrics['max_drawdown'] * 100.0:,.2f}%")
    print(f"  Annual Volatility: {metrics['annual_volatility'] * 100.0:,.2f}%")
    print(f"  Weekly Sharpe: {metrics['sharpe']:.2f}")
    print("\nQQQ Benchmark")
    print(f"  Total Return: {benchmark_metrics['total_return'] * 100.0:,.2f}%")
    print(f"  CAGR: {benchmark_metrics['cagr'] * 100.0:,.2f}%")
    print(f"  Max Drawdown: {benchmark_metrics['max_drawdown'] * 100.0:,.2f}%")
    print(f"  Annual Volatility: {benchmark_metrics['annual_volatility'] * 100.0:,.2f}%")
    print(f"  Weekly Sharpe: {benchmark_metrics['sharpe']:.2f}")

    print("\nRecent Rebalances")
    print(
        results.tail(12)[
            ["rebalance_date", "risk_on", "selected_count", "portfolio_return", "qqq_return", "selected_tickers"]
        ].to_string(index=False, float_format=lambda x: f"{x * 100.0:0.2f}%" if isinstance(x, float) and abs(x) <= 5 else f"{x:0.4f}")
    )

    if output:
        output_path = Path(output)
        results.to_csv(output_path, index=False)
        print(f"\nBacktest output written to: {output_path}")
    return 0


def print_universe() -> int:
    tickers = fetch_nasdaq100_constituents()
    print("Nasdaq-100 Universe")
    print("=" * 80)
    print(f"Source: {NASDAQ100_COMPANIES_URL}")
    print(f"Constituent count: {len(tickers)}")
    print(", ".join(tickers))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan and backtest a Nasdaq-100 trend/momentum model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Score the current Nasdaq-100 universe.")
    scan_parser.add_argument("--top", type=int, default=12, help="Number of top-ranked names to flag (default: 12).")
    scan_parser.add_argument("--years", type=int, default=2, help="Years of price history to download for the scan (default: 2).")
    scan_parser.add_argument("--csv", default=None, help="Optional Schwab holdings CSV to tag currently held tickers.")
    scan_parser.add_argument("--output", default=None, help="Optional CSV path for the scan results.")

    bt_parser = subparsers.add_parser("backtest", help="Run a simple weekly-rebalance backtest.")
    bt_parser.add_argument("--top", type=int, default=12, help="Number of top-ranked names to hold (default: 12).")
    bt_parser.add_argument("--years", type=int, default=5, help="Years of history to evaluate (default: 5).")
    bt_parser.add_argument("--output", default=None, help="Optional CSV path for backtest results.")

    subparsers.add_parser("universe", help="Print the current Nasdaq-100 constituent list source and tickers.")

    args = parser.parse_args()
    try:
        if args.command == "scan":
            return scan_model(args.years, args.top, args.csv, args.output)
        if args.command == "backtest":
            return backtest_model(args.years, args.top, args.output)
        if args.command == "universe":
            return print_universe()
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
