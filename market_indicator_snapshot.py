#!/usr/bin/env python3
"""Fetch public market-sentiment snapshots for QQQ, Nasdaq, NAMO, and NYMO.

Examples:
  python market_indicator_snapshot.py
  python market_indicator_snapshot.py --days 30
  python market_indicator_snapshot.py --json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from html import unescape


USER_AGENT = "Mozilla/5.0"

QQQ_URL = "https://www.optionsanalysissuite.com/etf/qqq/volume-history"
NASDAQ_EQUITY_PCR_URL = "https://alphalerts.com/live-historical-equity-pcr/"
NDX_PCR_10D_URL = "https://www.alphaquery.com/stock/NDX/volatility-option-statistics/10-day/put-call-ratio-volume"
NAMO_CHART_URL = "https://stockcharts.com/sc3/ui/?s=$namo"
NYMO_CHART_URL = "https://stockcharts.com/sc3/ui/?s=$nymo"
NAMO_ALT_CHART_URL = "https://marketcharts.com/indicators/breadth/$ndx.mccosc"
NYMO_ALT_CHART_URL = "https://marketcharts.com/indicators/breadth/$nya.mccosc"


def fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read().decode("utf-8", errors="replace")


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def parse_qqq_page(html: str) -> dict[str, object]:
    snapshot_date_match = re.search(r"Snapshot as of ([A-Za-z]{3} \d{1,2}, \d{4})", html)
    ratio_match = re.search(r"put/call volume ratio is ([0-9.]+)", html, flags=re.I)
    volume_match = re.search(
        r"traded ([0-9.]+[MK]?) total options contracts, split as ([0-9.]+[MK]?) calls and ([0-9.]+[MK]?) puts",
        html,
        flags=re.I,
    )

    history: list[dict[str, object]] = []
    row_pattern = re.compile(
        r"<tr>\s*<td[^>]*>([A-Za-z]{3} \d{1,2}, \d{4})</td>\s*"
        r"<td[^>]*>([^<]+)</td>\s*"
        r"<td[^>]*>([^<]+)</td>\s*"
        r"<td[^>]*>([^<]+)</td>\s*"
        r"<td[^>]*>([0-9.]+)</td>\s*</tr>"
    )
    for match in row_pattern.finditer(html):
        history.append(
            {
                "date": match.group(1),
                "call_volume": collapse_whitespace(match.group(2)),
                "put_volume": collapse_whitespace(match.group(3)),
                "total_volume": collapse_whitespace(match.group(4)),
                "put_call_ratio": float(match.group(5)),
            }
        )

    return {
        "source_url": QQQ_URL,
        "snapshot_date": snapshot_date_match.group(1) if snapshot_date_match else None,
        "put_call_ratio": float(ratio_match.group(1).rstrip(".")) if ratio_match else None,
        "call_volume": volume_match.group(2) if volume_match else None,
        "put_volume": volume_match.group(3) if volume_match else None,
        "total_volume": volume_match.group(1) if volume_match else None,
        "recent_history": history,
    }


def parse_nasdaq_equity_pcr_page(html: str) -> dict[str, object]:
    live_match = re.search(r"Live Equity Put/Call Ratio:\s*<span[^>]*><strong>([0-9.]+)</strong>", html)
    series_match = re.search(r"name:\s*'Equity P/C Ratio',\s*data:\s*(\[\[.*?\]\])", html, flags=re.S)

    history: list[dict[str, object]] = []
    if series_match:
        raw_points = ast.literal_eval(series_match.group(1))
        for timestamp_ms, ratio in raw_points:
            history.append(
                {
                    "date": datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date().isoformat(),
                    "put_call_ratio": float(ratio),
                }
            )
    latest_embedded_date = history[-1]["date"] if history else None

    return {
        "source_url": NASDAQ_EQUITY_PCR_URL,
        "live_put_call_ratio": float(live_match.group(1)) if live_match else None,
        "embedded_history_latest_date": latest_embedded_date,
        "embedded_history_is_stale": bool(
            latest_embedded_date and datetime.fromisoformat(latest_embedded_date).year < datetime.now(timezone.utc).year - 1
        ),
        "recent_history": history[-90:],
    }


def parse_ndx_page(html: str) -> dict[str, object]:
    ratio_match = re.search(r"had 10-Day Put-Call Ratio \(Volume\) of <strong>([0-9.]+)</strong> for <strong>(\d{4}-\d{2}-\d{2})</strong>", html)
    oi_match = re.search(r"Put-Call Ratio \(Open Interest\)</a>\s*</div>\s*<div[^>]*>\s*([0-9.]+)\s*</div>", html)
    history: list[dict[str, object]] = []
    row_pattern = re.compile(r"<td>(\d{4}-\d{2}-\d{2})</td>\s*<td>([0-9.]+)</td>")
    for match in row_pattern.finditer(html):
        history.append(
            {
                "date": match.group(1),
                "put_call_ratio": float(match.group(2)),
            }
        )

    return {
        "source_url": NDX_PCR_10D_URL,
        "snapshot_date": ratio_match.group(2) if ratio_match else None,
        "put_call_ratio_10d": float(ratio_match.group(1)) if ratio_match else None,
        "put_call_ratio_open_interest": float(oi_match.group(1)) if oi_match else None,
        "recent_history": history,
    }


def build_snapshot() -> dict[str, object]:
    qqq_html = fetch_html(QQQ_URL)
    nasdaq_equity_html = fetch_html(NASDAQ_EQUITY_PCR_URL)
    ndx_html = fetch_html(NDX_PCR_10D_URL)

    return {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "qqq_put_call": parse_qqq_page(qqq_html),
        "nasdaq_equity_put_call": parse_nasdaq_equity_pcr_page(nasdaq_equity_html),
        "nasdaq_100_put_call": parse_ndx_page(ndx_html),
        "put_call_chart_links": {
            "qqq": QQQ_URL,
            "nasdaq_equity": NASDAQ_EQUITY_PCR_URL,
            "nasdaq_100_10d": NDX_PCR_10D_URL,
        },
        "breadth_chart_links": {
            "namo_stockcharts": NAMO_CHART_URL,
            "nymo_stockcharts": NYMO_CHART_URL,
            "namo_marketcharts": NAMO_ALT_CHART_URL,
            "nymo_marketcharts": NYMO_ALT_CHART_URL,
        },
    }


def print_history_block(title: str, rows: list[dict[str, object]], history_limit: int) -> None:
    if not rows:
        print(f"  {title}: no history available.")
        return
    print(f"  {title}:")
    for row in rows[:history_limit]:
        if "call_volume" in row:
            print(
                f"    {row['date']}: ratio={row['put_call_ratio']:.2f}, "
                f"calls={row['call_volume']}, puts={row['put_volume']}, total={row['total_volume']}"
            )
        else:
            print(f"    {row['date']}: ratio={row['put_call_ratio']:.4f}")


def print_text_report(snapshot: dict[str, object], history_limit: int) -> None:
    qqq = snapshot["qqq_put_call"]
    nasdaq_equity = snapshot["nasdaq_equity_put_call"]
    ndx = snapshot["nasdaq_100_put_call"]
    pc_links = snapshot["put_call_chart_links"]
    links = snapshot["breadth_chart_links"]

    print("Market Indicator Snapshot")
    print("=" * 80)
    print(f"Fetched: {snapshot['fetched_at_utc']}")

    print("\nQQQ Put/Call Ratio")
    print(f"  Source: {qqq['source_url']}")
    print(f"  Snapshot Date: {qqq['snapshot_date']}")
    print(f"  Put/Call Ratio: {qqq['put_call_ratio']}")
    print(f"  Call Volume: {qqq['call_volume']}")
    print(f"  Put Volume: {qqq['put_volume']}")
    print(f"  Total Volume: {qqq['total_volume']}")
    print(f"  Chart: {pc_links['qqq']}")
    print_history_block("Last Values", qqq["recent_history"], history_limit)

    print("\nNasdaq Equity Put/Call Ratio")
    print(f"  Source: {nasdaq_equity['source_url']}")
    print(f"  Live Ratio: {nasdaq_equity['live_put_call_ratio']}")
    print(f"  Chart: {pc_links['nasdaq_equity']}")
    if nasdaq_equity["embedded_history_is_stale"]:
        print(
            "  Embedded history in the public page HTML appears stale "
            f"(latest embedded date: {nasdaq_equity['embedded_history_latest_date']})."
        )
    else:
        print_history_block("Last Values", list(reversed(nasdaq_equity["recent_history"][-history_limit:])), history_limit)

    print("\nNasdaq-100 Put/Call Ratio")
    print(f"  Source: {ndx['source_url']}")
    print(f"  Snapshot Date: {ndx['snapshot_date']}")
    print(f"  10-Day Put/Call Ratio (Volume): {ndx['put_call_ratio_10d']}")
    print(f"  Put/Call Ratio (Open Interest): {ndx['put_call_ratio_open_interest']}")
    print(f"  Chart: {pc_links['nasdaq_100_10d']}")
    print_history_block("Last Values", list(reversed(ndx["recent_history"][:history_limit])), history_limit)

    print("\nNAMO / NYMO Chart Links")
    print(f"  NAMO (StockCharts): {links['namo_stockcharts']}")
    print(f"  NYMO (StockCharts): {links['nymo_stockcharts']}")
    print(f"  NAMO (MarketCharts): {links['namo_marketcharts']}")
    print(f"  NYMO (MarketCharts): {links['nymo_marketcharts']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch QQQ/Nasdaq put-call ratios and breadth chart links.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of recent history rows to show in text mode (default: 30).",
    )
    args = parser.parse_args()

    try:
        snapshot = build_snapshot()
    except Exception as exc:
        print(f"Failed to fetch market indicators: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(snapshot, indent=2))
    else:
        print_text_report(snapshot, max(args.days, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
