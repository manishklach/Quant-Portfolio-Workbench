"""Analyze call spread verticals from current holdings and recommend actions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from portfolio_core import (
    active_option_positions,
    default_csv_path,
    load_schwab_holdings,
)

SECTION_WIDTH = 95


def fetch_spot(ticker: str) -> float | None:
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="2d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def recommend_call_spread(
    long_strike: float,
    short_strike: float,
    spot: float | None,
    dte: int,
    pl_pct: float,
    pl: float,
    width: float,
) -> tuple[str, str]:
    if spot is None:
        return ("HOLD", "No spot price available")

    if dte <= 7:
        if abs(pl) < 0.05 * width * 100:
            return ("LET EXPIRE", f"Expiring in {dte}d, negligible value")
        if pl < 0:
            return ("CLOSE", f"Expiring in {dte}d, underwater -${abs(pl):.0f}")
        return ("LET EXPIRE", f"Expiring in {dte}d, ${pl:+.0f}")

    if dte <= 30:
        mult = 1.0 if spot >= short_strike else 0.0
        max_value = width * 100 * mult
        if spot >= short_strike:
            if pl / max_value > 0.8:
                return ("CLOSE", f"At max value, ${pl:+.0f}, lock in early")
            return ("HOLD", f"In the money, ${pl:+.0f}, let expire")
        if pl < 0 and pl_pct < -60:
            return ("CLOSE", f"Underwater {pl_pct:.0f}%, not enough time left")
        return ("HOLD", f"Still time, DTE={dte}")

    if spot > short_strike:
        return ("HOLD", f"Both ITM, ${pl:+.0f} profit, DTE={dte}")
    if spot >= long_strike:
        if pl > 0:
            return ("HOLD", f"Working, ${pl:+.0f}, spot=${spot:.2f}")
        return ("HOLD", f"Between strikes, ${pl:+.0f}, spot may recover")
    if pl_pct < -70:
        return ("ROLL", f"Deep underwater ({pl_pct:.0f}%), consider roll or put spread")
    if pl_pct < -50:
        return ("HOLD", f"Underwater {pl_pct:.0f}%, monitor for recovery")
    return ("HOLD", f"OTM but manageable, DTE={dte}, pl={pl_pct:.0f}%")


def recommend_standalone_call(
    strike: float,
    qty: int,
    spot: float | None,
    dte: int,
    net_cost: float,
    net_mkt: float,
    is_long: bool,
) -> tuple[str, str]:
    if spot is None:
        return ("HOLD", "No spot")

    pl = net_mkt - net_cost
    pl_pct = pl / abs(net_cost) * 100 if abs(net_cost) > 0 else 0

    if is_long:
        if dte <= 14:
            if spot < strike:
                return ("CLOSE", "Expiring OTM, worthless")
            return ("LET EXPIRE", "Expiring ITM, let expire")
        if spot < strike * 1.1:
            return ("HOLD", f"Long call {strike:.0f}, OTM, DTE={dte}")
        if pl > 0:
            return ("HOLD", f"ITM long call, ${pl:+.0f}")
        return ("HOLD", f"ITM but underwater, DTE={dte}")
    else:
        if dte <= 14:
            if spot >= strike:
                return ("CLOSE", "Short call ITM at expiry, close")
            return ("LET EXPIRE", "Short call OTM at expiry")
        if spot >= strike:
            return ("CLOSE", "Short call ITM, roll or close")
        if pl > 0:
            return ("HOLD", f"Short call decaying, ${pl:+.0f}")
        return ("HOLD", f"Short call, DTE={dte}, premium collected")


def fmt_dol(v: float) -> str:
    return f"${v:>+10,.2f}"


def pct_str(v: float) -> str:
    return f"{v:>+6.1f}%"


def load_holdings_with_fallback_dates(csv_path: Path, as_of: datetime.date):
    return load_schwab_holdings(csv_path, as_of=as_of)


def main():
    parser = argparse.ArgumentParser(description="Call spread vertical review & recommendations.")
    parser.add_argument("--file", default=None, help="Path to holdings CSV (default: my_holdings.csv)")
    parser.add_argument("--as-of", default=None, help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--r", type=float, default=0.04, help="Risk-free rate (default: 0.04)")
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, "%Y-%m-%d").date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)

    df = load_holdings_with_fallback_dates(csv_path, as_of)
    options = active_option_positions(df)

    calls = options[options["Opt Type"] == "C"].copy()
    if calls.empty:
        print("No active call positions found.")
        return

    calls["Days To Expiry"] = calls["Days To Expiry"].fillna(0).astype(int)

    underlying_spots: dict[str, float | None] = {}
    underlying_names = sorted(calls["Underlying"].unique())
    for und in underlying_names:
        underlying_spots[und] = fetch_spot(und)

    print("=" * SECTION_WIDTH)
    print("  CALL SPREAD REVIEW")
    print(f"  {as_of}")
    print("=" * SECTION_WIDTH)

    total_pl = 0.0
    all_rows = []

    for underlying in underlying_names:
        und_calls = calls[calls["Underlying"] == underlying].sort_values(["Expiration", "Strike Price"])
        spot = underlying_spots.get(underlying)

        for expiry, grp in und_calls.groupby("Expiration", sort=False):
            dte = int(grp["Days To Expiry"].iloc[0])
            expiry_label = f"{expiry} (DTE={dte})"

            longs = []
            shorts = []
            for _, row in grp.iterrows():
                qty = float(row["Qty"])
                leg = {
                    "strike": float(row["Strike Price"]),
                    "qty": abs(qty),
                    "cost_basis": float(row["Cost Basis Numeric"]),
                    "mkt_value": float(row["Market Value Numeric"]),
                    "price": float(row["Price Numeric"]),
                    "multiplier": float(row["Multiplier"]),
                }
                if qty > 0:
                    longs.append(leg)
                else:
                    shorts.append(leg)

            longs.sort(key=lambda x: x["strike"])
            shorts.sort(key=lambda x: x["strike"])

            spread_pairs = []
            used_long = [False] * len(longs)
            used_short_qty = [0.0] * len(shorts)

            for si, short_leg in enumerate(shorts):
                remaining = short_leg["qty"]
                for li, long_leg in enumerate(longs):
                    if remaining <= 0:
                        break
                    if used_long[li]:
                        continue
                    if long_leg["strike"] >= short_leg["strike"]:
                        continue
                    avail = long_leg["qty"] - used_long[li]
                    if avail <= 0:
                        continue
                    matched = min(remaining, avail)
                    used_long[li] += matched
                    used_short_qty[si] += matched
                    spread_pairs.append({
                        "long_strike": long_leg["strike"],
                        "short_strike": short_leg["strike"],
                        "qty": matched,
                        "long_cost": long_leg["cost_basis"] * (matched / long_leg["qty"]),
                        "short_cost": short_leg["cost_basis"] * (matched / short_leg["qty"]),
                        "long_mkt": long_leg["mkt_value"] * (matched / long_leg["qty"]),
                        "short_mkt": short_leg["mkt_value"] * (matched / short_leg["qty"]),
                        "width": short_leg["strike"] - long_leg["strike"],
                    })
                    remaining -= matched

                if remaining > 0:
                    standalone_leg = {
                        "strike": short_leg["strike"],
                        "qty": remaining,
                        "cost_basis": short_leg["cost_basis"] * (remaining / short_leg["qty"]),
                        "mkt_value": short_leg["mkt_value"] * (remaining / short_leg["qty"]),
                        "price": short_leg["price"],
                        "is_long": False,
                        "dte": dte,
                        "underlying": underlying,
                        "expiry": expiry,
                    }
                    all_rows.append(standalone_leg)

            for li, long_leg in enumerate(longs):
                remaining = long_leg["qty"] - used_long[li]
                if remaining > 0:
                    standalone_leg = {
                        "strike": long_leg["strike"],
                        "qty": remaining,
                        "cost_basis": long_leg["cost_basis"] * (remaining / long_leg["qty"]),
                        "mkt_value": long_leg["mkt_value"] * (remaining / long_leg["qty"]),
                        "price": long_leg["price"],
                        "is_long": True,
                        "dte": dte,
                        "underlying": underlying,
                        "expiry": expiry,
                    }
                    all_rows.append(standalone_leg)

            if not spread_pairs:
                continue

            print(f"\n{underlying} — {expiry_label}  [Spot: ${spot:,.2f}]" if spot else f"\n{underlying} — {expiry_label}")

            header = (
                f"  {'Long':>7} {'Short':>7} {'Qty':>5} "
                f"{'Cost Basis':>12} {'Mkt Value':>12} {'P&L':>12} {'P&L%':>8} "
                f"{'Status':>8}  {'Recommendation'}"
            )
            print(header)
            print("  " + "-" * (SECTION_WIDTH - 2))

            und_subtotal_cost = 0.0
            und_subtotal_mkt = 0.0

            for sp in spread_pairs:
                net_cost = sp["long_cost"] - abs(sp["short_cost"])
                net_mkt = sp["long_mkt"] - abs(sp["short_mkt"])
                pl = net_mkt - net_cost
                pl_pct = pl / abs(net_cost) * 100 if abs(net_cost) > 0 else 0.0
                und_subtotal_cost += net_cost
                und_subtotal_mkt += net_mkt
                total_pl += pl

                if abs(pl) < 1.0 and abs(net_mkt) < 100:
                    status = "FLAT"
                elif pl > 0:
                    status = "WINNING" if pl_pct > 5 else "FLAT"
                else:
                    status = "LOSING"
                pl_display = pl_pct if abs(pl_pct) < 1000 else 0.0

                rec, reason = recommend_call_spread(
                    sp["long_strike"], sp["short_strike"],
                    spot, dte, pl_pct, pl, sp["width"],
                )
                rec_short = f"{rec}: {reason}"

                print(
                    f"  ${sp['long_strike']:<5.0f} ${sp['short_strike']:<5.0f} "
                    f"{sp['qty']:>5.0f} "
                    f"{fmt_dol(net_cost):>12s} {fmt_dol(net_mkt):>12s} "
                    f"{fmt_dol(pl):>12s} {pct_str(pl_display):>8s} "
                    f"{status:>8s}  {rec_short}"
                )

            if spread_pairs:
                pair_pl = und_subtotal_mkt - und_subtotal_cost
                print(f"  {'-' * (SECTION_WIDTH - 2)}")
                print(
                    f"  {'Group subtotal':>27s} {fmt_dol(und_subtotal_cost):>12s} "
                    f"{fmt_dol(und_subtotal_mkt):>12s} {fmt_dol(pair_pl):>12s}"
                )

    if all_rows:
        print(f"\n{'=' * SECTION_WIDTH}")
        print("  STANDALONE (UNPAIRED) CALLS")
        print(f"{'=' * SECTION_WIDTH}")

        for leg in all_rows:
            pl = leg["mkt_value"] - leg["cost_basis"]
            pl_pct = pl / abs(leg["cost_basis"]) * 100 if abs(leg["cost_basis"]) > 0 else 0.0
            total_pl += pl
            side = "LONG" if leg["is_long"] else "SHORT"
            rec, reason = recommend_standalone_call(
                leg["strike"], leg["qty"], underlying_spots.get(leg["underlying"]),
                leg["dte"], leg["cost_basis"], leg["mkt_value"], leg["is_long"],
            )
            if leg["is_long"]:
                rec_short = f"{rec}: {reason}"
                print(
                    f"  {leg['underlying']:>5s} {leg['expiry']:>12s} "
                    f"${leg['strike']:<6.0f} {side:>5s} {leg['qty']:>5.0f}x "
                    f"{fmt_dol(leg['cost_basis']):>12s} {fmt_dol(leg['mkt_value']):>12s} "
                    f"{fmt_dol(pl):>12s} {pct_str(pl_pct):>8s}  {rec_short}"
                )

    print(f"\n{'=' * SECTION_WIDTH}")
    print(f"  TOTAL NET P&L from call spreads: {fmt_dol(total_pl)}")
    print(f"{'=' * SECTION_WIDTH}")
    print()
    print("  RECOMMENDATION KEY:")
    print("    HOLD       - Keep position as-is")
    print("    CLOSE      - Close position (theta burn or expiry risk)")
    print("    ROLL       - Consider rolling to put credit spread or different structure")
    print("    LET EXPIRE - Allow position to expire worthless/ITM")
    print()
    print("  NOTE: P&L includes open positions. True realized P&L differs.")
    print(f"  As-of date: {as_of}")


if __name__ == "__main__":
    main()
