#!/usr/bin/env python3
"""Aggregate Day Change ($) from a portfolio export file."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from typing import Iterable

from portfolio_core import clean_numeric


DAY_CHANGE_HEADER = "Day Chng $ (Day Change $)"
DAY_CHANGE_HEADER_NORMALIZED = "day chng $ (day change $)"
SYMBOL_HEADER_NORMALIZED = "symbol"
EXCLUDED_TICKERS = {"Cash", "Cash & Cash Investments", "Account Total"}


def normalize_header_name(value: str) -> str:
    return value.strip().lstrip("\ufeff").lower()


def is_day_change_header(value: str) -> bool:
    normalized = normalize_header_name(value)
    return (
        normalized == DAY_CHANGE_HEADER_NORMALIZED
        or "day chng $" in normalized
        or "day change $" in normalized
    )


def normalize_money(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "N/A":
        return None
    parsed = clean_numeric(cleaned)
    if parsed == 0.0 and cleaned not in {"0", "0.0", "$0", "$0.00", "($0.00)", "-0", "-0.0"}:
        return None
    return parsed


def split_row(line: str, delimiter: str) -> list[str]:
    if delimiter == ",":
        return next(csv.reader([line]))
    if delimiter == "\t":
        return re.split(r"\t+", line.strip())
    return re.split(r"\s{2,}", line.strip())


def detect_header(lines: Iterable[str]) -> tuple[list[str], str] | None:
    for line in lines:
        if DAY_CHANGE_HEADER in line or "Day Chng $" in line or "Day Change $" in line:
            if "," in line:
                delimiter = ","
            else:
                delimiter = "\t" if "\t" in line else "  "
            columns = split_row(line, delimiter)
            if any(is_day_change_header(column) for column in columns):
                return columns, delimiter
    return None


def parse_rows(raw_text: str) -> tuple[list[dict[str, str]], str, str]:
    lines = raw_text.splitlines()
    header_result = detect_header(lines)
    if not header_result:
        raise ValueError("Could not find header row with Day Chng $ column.")
    header, delimiter = header_result

    header_index = lines.index(
        next(
            line
            for line in lines
            if DAY_CHANGE_HEADER in line or "Day Chng $" in line or "Day Change $" in line
        )
    )
    data_lines = [line for line in lines[header_index + 1 :] if line.strip()]
    normalized_headers = [normalize_header_name(name) for name in header]
    header_map = {normalized: original for normalized, original in zip(normalized_headers, header)}
    day_change_key = next(
        (header_map[name] for name in normalized_headers if is_day_change_header(name)),
        None,
    )
    if not day_change_key:
        raise ValueError("Could not find Day Chng $ column in header.")
    symbol_key = header_map.get(SYMBOL_HEADER_NORMALIZED)
    if not symbol_key:
        raise ValueError("Could not find Symbol column in header.")
    rows: list[dict[str, str]] = []
    for line in data_lines:
        values = split_row(line, delimiter)
        if len(values) < len(header):
            values.extend([""] * (len(header) - len(values)))
        row = {key.strip(): value.strip() for key, value in zip(header, values)}
        rows.append(row)
    return rows, day_change_key, symbol_key


def normalize_ticker(ticker: str, use_underlying: bool) -> str:
    cleaned = ticker.strip()
    if not use_underlying or not cleaned:
        return cleaned
    return cleaned.split()[0]


def should_exclude_ticker(ticker: str, normalized: str) -> bool:
    excluded_values = {ticker, normalized}
    if excluded_values & EXCLUDED_TICKERS:
        return True

    lowered = {value.lower() for value in excluded_values if value}
    return any(value.startswith("positions") for value in lowered)


def aggregate_day_change(
    rows: list[dict[str, str]],
    symbol_key: str,
    day_change_key: str,
    use_underlying: bool,
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        ticker = row.get(symbol_key, "").strip()
        normalized = normalize_ticker(ticker, use_underlying)
        if not ticker or should_exclude_ticker(ticker, normalized):
            continue
        day_change = normalize_money(row.get(day_change_key, ""))
        if day_change is None:
            continue
        totals[normalized] += day_change
    return dict(totals)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Day Chng $ totals by ticker from a portfolio export."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        help="Path to a portfolio export file. If omitted, input is read from stdin.",
    )
    parser.add_argument(
        "--ticker",
        action="store_true",
        help="Aggregate option symbols by underlying ticker (e.g., AAPL).",
    )
    parser.add_argument(
        "--underlying",
        action="store_true",
        help="Alias for --ticker.",
    )
    if len(sys.argv) == 1:
        parser.print_usage(sys.stderr)
        return 1

    args = parser.parse_args()

    if args.filename:
        try:
            with open(args.filename, "r", encoding="utf-8-sig") as input_file:
                raw_text = input_file.read()
        except OSError as exc:
            print(f"Could not read '{args.filename}': {exc}", file=sys.stderr)
            return 1
    if not raw_text.strip():
        print(f"Input file '{args.filename}' is empty.", file=sys.stderr)
        return 1

    try:
        rows, day_change_key, symbol_key = parse_rows(raw_text)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    group_by_ticker = args.ticker or args.underlying
    totals = aggregate_day_change(rows, symbol_key, day_change_key, group_by_ticker)
    if not totals:
        print("No Day Chng $ values found to aggregate.", file=sys.stderr)
        return 1

    sorted_totals = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    grand_total = sum(value for _, value in sorted_totals)

    for ticker, total in sorted_totals:
        print(f"{ticker}\t{total:,.2f}")
    print(f"TOTAL\t{grand_total:,.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
