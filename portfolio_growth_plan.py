#!/usr/bin/env python3
"""Plan a path from current portfolio NAV to a target NAV over a fixed horizon.

Examples:
  python portfolio_growth_plan.py --current-nav 20000000 --target-nav 25000000 --end-date 2026-12-31
  python portfolio_growth_plan.py --current-nav 20000000 --target-nav 25000000 --end-date 2026-12-31 --monthly-net-flow 50000
  python portfolio_growth_plan.py --current-nav 20000000 --target-nav 25000000 --end-date 2026-12-31 ^
      --sleeve "core_growth,12000000,1.0,0.18" ^
      --sleeve "income,5000000,1.0,0.10" ^
      --sleeve "tactical_margin,3000000,2.0,0.22"
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime

from portfolio_core import default_csv_path, load_schwab_holdings


@dataclass
class Sleeve:
    name: str
    net_capital: float
    gross_multiple: float
    annual_return: float


PRESET_TEMPLATES: dict[str, list[tuple[str, float, float, float]]] = {
    "balanced_boost": [
        ("core_growth", 0.55, 1.00, 0.18),
        ("income_csp", 0.20, 1.10, 0.12),
        ("tactical_bcs", 0.15, 1.75, 0.24),
        ("cash_yield", 0.10, 1.00, 0.05),
    ],
    "barbell_boost": [
        ("core_compounders", 0.50, 1.00, 0.17),
        ("defensive_income", 0.25, 1.00, 0.09),
        ("csp_income", 0.15, 1.25, 0.14),
        ("tactical_spreads", 0.10, 2.50, 0.30),
    ],
    "aggressive_margin": [
        ("core_growth", 0.45, 1.00, 0.20),
        ("income", 0.15, 1.00, 0.10),
        ("tactical_margin", 0.25, 2.00, 0.24),
        ("opportunistic_spreads", 0.15, 3.00, 0.35),
    ],
    "income_plus_tactical": [
        ("high_conviction_growth", 0.40, 1.00, 0.19),
        ("covered_income_etf", 0.20, 1.00, 0.09),
        ("cash_secured_put_income", 0.20, 1.20, 0.14),
        ("bull_call_spread_tactical", 0.20, 2.40, 0.32),
    ],
    "semi_offense": [
        ("high_conviction_growth", 0.35, 1.00, 0.18),
        ("semi_momentum_tactical", 0.25, 1.60, 0.26),
        ("cash_secured_put_income", 0.20, 1.15, 0.13),
        ("bull_call_spread_tactical", 0.20, 2.75, 0.36),
    ],
    "goal_seek_25m": [
        ("high_conviction_growth", 0.35, 1.00, 0.20),
        ("covered_income_etf", 0.15, 1.00, 0.08),
        ("cash_secured_put_income", 0.20, 1.25, 0.15),
        ("semi_momentum_tactical", 0.15, 1.75, 0.28),
        ("bull_call_spread_tactical", 0.15, 3.25, 0.40),
    ],
}

SEMI_TICKERS = {
    "AMD",
    "AMAT",
    "ARM",
    "ASML",
    "AVGO",
    "INTC",
    "KLAC",
    "LRCX",
    "MRVL",
    "MU",
    "NVDA",
    "NXPI",
    "QCOM",
    "TSM",
}


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_sleeve(value: str) -> Sleeve:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Sleeve must be 'name,net_capital,gross_multiple,annual_return'."
        )
    name = parts[0]
    try:
        net_capital = float(parts[1])
        gross_multiple = float(parts[2])
        annual_return = float(parts[3])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Sleeve values after the name must be numeric."
        ) from exc
    if net_capital < 0 or gross_multiple < 0:
        raise argparse.ArgumentTypeError("Sleeve net_capital and gross_multiple must be non-negative.")
    return Sleeve(
        name=name,
        net_capital=net_capital,
        gross_multiple=gross_multiple,
        annual_return=annual_return,
    )


def future_value_with_monthly_flows(
    principal: float,
    annual_return: float,
    horizon_days: int,
    monthly_net_flow: float,
) -> float:
    months = max(int(round(horizon_days / 30.4375)), 0)
    if months == 0:
        return principal
    monthly_return = (1.0 + annual_return) ** (1.0 / 12.0) - 1.0
    balance = principal
    for _ in range(months):
        balance *= (1.0 + monthly_return)
        balance += monthly_net_flow
    return balance


def required_annual_return(
    current_nav: float,
    target_nav: float,
    horizon_days: int,
    monthly_net_flow: float,
) -> float | None:
    if horizon_days <= 0:
        return None
    if current_nav <= 0:
        return None

    low = -0.95
    high = 5.0
    target = target_nav

    def f(rate: float) -> float:
        return future_value_with_monthly_flows(current_nav, rate, horizon_days, monthly_net_flow) - target

    f_low = f(low)
    f_high = f(high)
    if f_low > 0:
        return low
    while f_high < 0 and high < 100.0:
        high *= 2.0
        f_high = f(high)
    if f_high < 0:
        return None

    for _ in range(120):
        mid = (low + high) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < 1e-6:
            return mid
        if f_mid < 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def projected_sleeve_return(sleeve: Sleeve, horizon_days: int) -> tuple[float, float]:
    if horizon_days <= 0 or sleeve.net_capital <= 0 or sleeve.gross_multiple <= 0:
        return 0.0, 0.0
    gross_exposure = sleeve.net_capital * sleeve.gross_multiple
    horizon_return = (1.0 + sleeve.annual_return) ** (horizon_days / 365.25) - 1.0
    pnl = gross_exposure * horizon_return
    return gross_exposure, pnl


def build_preset_sleeves(current_nav: float, preset_name: str) -> list[Sleeve]:
    key = str(preset_name).strip().lower()
    if key not in PRESET_TEMPLATES:
        raise SystemExit(
            f"Unknown preset '{preset_name}'. Available presets: {', '.join(sorted(PRESET_TEMPLATES))}"
        )
    return [
        Sleeve(
            name=name,
            net_capital=current_nav * alloc_frac,
            gross_multiple=gross_multiple,
            annual_return=annual_return,
        )
        for name, alloc_frac, gross_multiple, annual_return in PRESET_TEMPLATES[key]
    ]


def annualize_horizon_return(horizon_return: float, horizon_days: int) -> float | None:
    if horizon_days <= 0:
        return None
    if horizon_return <= -1.0:
        return None
    return (1.0 + horizon_return) ** (365.25 / horizon_days) - 1.0


def load_held_tickers(csv_path: str | None, script_file: str) -> set[str]:
    if not csv_path:
        return set()
    holdings = load_schwab_holdings(default_csv_path(csv_path, script_file))
    return {
        str(value).strip().upper()
        for value in holdings["Underlying"].dropna().tolist()
        if str(value).strip()
    }


def build_live_blueprint(
    *,
    current_nav: float,
    preset_name: str,
    csv_path: str | None,
) -> tuple[list[Sleeve], dict[str, object]]:
    from nasdaq100_quant_model import (
        attach_fundamental_scores,
        build_cross_section,
        build_indicator_frame,
        build_indicators_for_universe,
        compute_regime_snapshot,
        extract_ohlcv,
        fetch_fundamental_snapshots,
        fetch_nasdaq100_constituents,
        fetch_price_history,
        score_cross_section,
    )

    tickers = fetch_nasdaq100_constituents()
    history = fetch_price_history(sorted(set(tickers + ["QQQ"])), 2)
    indicators = build_indicators_for_universe(history, tickers)
    qqq = build_indicator_frame(extract_ohlcv(history, "QQQ"))
    regime = compute_regime_snapshot(qqq)
    as_of = qqq.index[-1]

    scored = score_cross_section(build_cross_section(indicators, as_of))
    top_candidates = set(scored.head(30)["ticker"].tolist())
    held_tickers = load_held_tickers(csv_path, __file__)
    top_candidates.update(t for t in held_tickers if t in set(tickers))
    fundamentals = fetch_fundamental_snapshots(sorted(top_candidates))
    scored = attach_fundamental_scores(scored, fundamentals)

    growth_candidates = scored.head(12)["ticker"].tolist()
    semi_candidates = scored[scored["ticker"].isin(SEMI_TICKERS)].head(8)["ticker"].tolist()
    csp_candidates = (
        scored[
            scored["ticker"].isin(list(held_tickers | {"AAPL", "GOOG", "MSFT", "MU", "NVDA", "AVGO", "TSM"}))
            & (scored["rsi14"] >= 40.0)
            & (scored["rsi14"] <= 68.0)
        ]
        .head(8)["ticker"]
        .tolist()
    )
    covered_income_candidates = [ticker for ticker in ["QQQI", "XQQI"] if ticker in held_tickers] or ["QQQI", "XQQI"]

    sleeves = build_preset_sleeves(current_nav, preset_name)
    blueprint = {
        "as_of": regime["date"],
        "risk_on": regime["risk_on"],
        "held_tickers": held_tickers,
        "growth_candidates": growth_candidates,
        "semi_candidates": semi_candidates,
        "csp_candidates": csp_candidates,
        "covered_income_candidates": covered_income_candidates,
    }
    return sleeves, blueprint


def evaluate_plan(
    *,
    current_nav: float,
    target_nav: float,
    horizon_days: int,
    monthly_net_flow: float,
    idle_annual_return: float,
    margin_available: float,
    sleeves: list[Sleeve],
    tactical_return_shift: float = 0.0,
    tactical_name_hints: tuple[str, ...] = ("tactical", "spread", "momentum", "opportunistic"),
) -> dict[str, object]:
    total_sleeve_capital = sum(s.net_capital for s in sleeves)
    idle_capital = current_nav - total_sleeve_capital
    if idle_capital < -1e-6:
        raise SystemExit("Total sleeve net capital exceeds current NAV.")

    modeled_rows: list[dict[str, float | str]] = []
    total_modeled_pnl = 0.0
    total_gross_exposure = 0.0
    for sleeve in sleeves:
        adjusted_annual_return = sleeve.annual_return
        if tactical_return_shift and any(hint in sleeve.name.lower() for hint in tactical_name_hints):
            adjusted_annual_return += tactical_return_shift
        adjusted_sleeve = Sleeve(
            name=sleeve.name,
            net_capital=sleeve.net_capital,
            gross_multiple=sleeve.gross_multiple,
            annual_return=adjusted_annual_return,
        )
        gross_exposure, pnl = projected_sleeve_return(adjusted_sleeve, horizon_days)
        total_modeled_pnl += pnl
        total_gross_exposure += gross_exposure
        modeled_rows.append(
            {
                "name": sleeve.name,
                "net_capital": sleeve.net_capital,
                "gross_multiple": sleeve.gross_multiple,
                "gross_exposure": gross_exposure,
                "annual_return": adjusted_annual_return,
                "projected_pnl": pnl,
            }
        )

    idle_pnl = 0.0
    if idle_capital > 0 and idle_annual_return != 0:
        idle_pnl = idle_capital * ((1.0 + idle_annual_return) ** (horizon_days / 365.25) - 1.0)
    total_modeled_pnl += idle_pnl

    months = horizon_days / 30.4375
    net_flows_total = monthly_net_flow * max(int(round(months)), 0)
    modeled_ending_nav = current_nav + total_modeled_pnl + net_flows_total
    modeled_gap = target_nav - modeled_ending_nav
    total_buying_power = current_nav + margin_available
    gross_headroom_remaining = max(total_buying_power - total_gross_exposure, 0.0)
    required_headroom_horizon_return = (
        modeled_gap / gross_headroom_remaining
        if gross_headroom_remaining > 0 and modeled_gap > 0
        else 0.0
    )
    required_headroom_annual = annualize_horizon_return(required_headroom_horizon_return, horizon_days)

    return {
        "modeled_rows": modeled_rows,
        "total_sleeve_capital": total_sleeve_capital,
        "idle_capital": idle_capital,
        "total_modeled_pnl": total_modeled_pnl,
        "total_gross_exposure": total_gross_exposure,
        "idle_pnl": idle_pnl,
        "net_flows_total": net_flows_total,
        "modeled_ending_nav": modeled_ending_nav,
        "modeled_gap": modeled_gap,
        "total_buying_power": total_buying_power,
        "gross_headroom_remaining": gross_headroom_remaining,
        "required_headroom_horizon_return": required_headroom_horizon_return,
        "required_headroom_annual": required_headroom_annual,
    }


def main() -> int:
    if "--list-presets" in sys.argv:
        print("Available Presets")
        print("=" * 72)
        for name, template in PRESET_TEMPLATES.items():
            print(f"\n{name}")
            for sleeve_name, alloc_frac, gross_multiple, annual_return in template:
                print(
                    f"  {sleeve_name}: alloc={alloc_frac * 100.0:.1f}% "
                    f"gross_x={gross_multiple:.2f} annual_return={annual_return * 100.0:.2f}%"
                )
        return 0

    parser = argparse.ArgumentParser(
        description="Plan a path from a current portfolio NAV to a target NAV over a fixed horizon."
    )
    parser.add_argument("--current-nav", type=float, required=True, help="Current total portfolio NAV.")
    parser.add_argument("--target-nav", type=float, required=True, help="Target portfolio NAV.")
    parser.add_argument(
        "--start-date",
        default=None,
        help="Start date in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Target date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--monthly-net-flow",
        type=float,
        default=0.0,
        help="Expected net monthly contribution (negative for withdrawals).",
    )
    parser.add_argument(
        "--idle-annual-return",
        type=float,
        default=0.0,
        help="Annual return assumption for capital not assigned to a named sleeve.",
    )
    parser.add_argument(
        "--sleeve",
        action="append",
        type=parse_sleeve,
        default=[],
        help="Strategy sleeve: name,net_capital,gross_multiple,annual_return. Repeat for multiple sleeves.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=f"Optional preset sleeve mix. Choices: {', '.join(sorted(PRESET_TEMPLATES))}.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print available preset sleeve mixes and exit.",
    )
    parser.add_argument(
        "--margin-available",
        type=float,
        default=0.0,
        help="Additional gross buying power / margin availability available above current NAV.",
    )
    parser.add_argument(
        "--compare-presets",
        action="store_true",
        help="Compare all built-in presets side by side under the same target and horizon.",
    )
    parser.add_argument(
        "--tactical-return-shift",
        type=float,
        default=0.0,
        help="Add or subtract an annual-return shift for tactical sleeves (e.g. -0.08 or 0.05).",
    )
    parser.add_argument(
        "--scenario-grid",
        action="store_true",
        help="Print a tactical-return / monthly-flow scenario grid for the selected preset.",
    )
    parser.add_argument(
        "--flow-grid",
        default="0,50000,100000,200000",
        help="Comma-separated monthly net-flow values used by --scenario-grid.",
    )
    parser.add_argument(
        "--tactical-shift-grid",
        default="-0.10,-0.05,0.00,0.05",
        help="Comma-separated annual tactical-return shifts used by --scenario-grid.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional Schwab holdings CSV for live blueprint mode.",
    )
    parser.add_argument(
        "--live-blueprint",
        action="store_true",
        help="Build a live strategy blueprint that maps preset sleeves to current quant-ranked candidate tickers.",
    )
    args = parser.parse_args()

    start_date = parse_date(args.start_date) if args.start_date else date.today()
    end_date = parse_date(args.end_date)
    horizon_days = (end_date - start_date).days
    if horizon_days <= 0:
        raise SystemExit("End date must be after start date.")

    current_nav = float(args.current_nav)
    target_nav = float(args.target_nav)
    monthly_net_flow = float(args.monthly_net_flow)
    margin_available = float(args.margin_available)

    sleeves = list(args.sleeve)
    if args.preset:
        if sleeves:
            raise SystemExit("Use either --preset or explicit --sleeve values, not both.")
        sleeves = build_preset_sleeves(current_nav, args.preset)

    required_gain = target_nav - current_nav
    months = horizon_days / 30.4375
    required_total_return = (target_nav / current_nav - 1.0) if current_nav > 0 else float("nan")
    required_annual = required_annual_return(current_nav, target_nav, horizon_days, monthly_net_flow)
    required_monthly = (
        (1.0 + required_annual) ** (1.0 / 12.0) - 1.0
        if required_annual is not None and required_annual > -1.0
        else None
    )

    if args.compare_presets:
        rows = []
        for preset_name in sorted(PRESET_TEMPLATES):
            preset_sleeves = build_preset_sleeves(current_nav, preset_name)
            summary = evaluate_plan(
                current_nav=current_nav,
                target_nav=target_nav,
                horizon_days=horizon_days,
                monthly_net_flow=monthly_net_flow,
                idle_annual_return=args.idle_annual_return,
                margin_available=margin_available,
                sleeves=preset_sleeves,
                tactical_return_shift=args.tactical_return_shift,
            )
            rows.append(
                {
                    "preset": preset_name,
                    "gross_exposure": summary["total_gross_exposure"],
                    "gross_headroom_remaining": summary["gross_headroom_remaining"],
                    "modeled_pnl": summary["total_modeled_pnl"],
                    "modeled_ending_nav": summary["modeled_ending_nav"],
                    "gap_to_target": summary["modeled_gap"],
                    "needed_headroom_return_horizon_pct": summary["required_headroom_horizon_return"] * 100.0,
                    "needed_headroom_return_annual_pct": (
                        summary["required_headroom_annual"] * 100.0
                        if summary["required_headroom_annual"] is not None
                        else float("nan")
                    ),
                }
            )

        out = (
            __import__("pandas").DataFrame(rows)
            .sort_values(["gap_to_target", "needed_headroom_return_annual_pct"], ascending=[True, True])
            .reset_index(drop=True)
        )
        print("\nPreset Strategy Comparison")
        print("=" * 110)
        print(f"Start Date:                 {start_date.isoformat()}")
        print(f"End Date:                   {end_date.isoformat()}")
        print(f"Current NAV:                ${current_nav:,.2f}")
        print(f"Target NAV:                 ${target_nav:,.2f}")
        print(f"Margin Availability:        ${margin_available:,.2f}")
        print(out.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        return 0

    if args.live_blueprint:
        if not args.preset:
            raise SystemExit("--live-blueprint requires --preset.")
        sleeves, blueprint = build_live_blueprint(
            current_nav=current_nav,
            preset_name=args.preset,
            csv_path=args.csv,
        )
        plan = evaluate_plan(
            current_nav=current_nav,
            target_nav=target_nav,
            horizon_days=horizon_days,
            monthly_net_flow=monthly_net_flow,
            idle_annual_return=args.idle_annual_return,
            margin_available=margin_available,
            sleeves=sleeves,
            tactical_return_shift=args.tactical_return_shift,
        )
        held = blueprint["held_tickers"]

        def annotate(tickers: list[str]) -> str:
            if not tickers:
                return "None"
            return ", ".join([f"{ticker}*" if ticker in held else ticker for ticker in tickers])

        print("\nLive Strategy Blueprint")
        print("=" * 110)
        print(f"Preset:                     {args.preset}")
        print(f"Blueprint As-Of:            {blueprint['as_of']}")
        print(f"QQQ Regime Filter:          {'RISK-ON' if blueprint['risk_on'] else 'RISK-OFF'}")
        print(f"Current NAV:                ${current_nav:,.2f}")
        print(f"Target NAV:                 ${target_nav:,.2f}")
        print(f"Margin Availability:        ${margin_available:,.2f}")
        print(f"Modeled Ending NAV:         ${plan['modeled_ending_nav']:,.2f}")
        print(f"Gap To Target:              ${plan['modeled_gap']:,.2f}")
        print("\nCandidate Mapping")
        print(f"High-Conviction Growth:     {annotate(blueprint['growth_candidates'][:8])}")
        print(f"Semi Momentum Tactical:     {annotate(blueprint['semi_candidates'][:6])}")
        print(f"Cash-Secured Put Income:    {annotate(blueprint['csp_candidates'][:6])}")
        print(f"Covered-Income ETF Sleeve:  {annotate(blueprint['covered_income_candidates'])}")
        print("\nSleeve Allocations")
        print(
            f"{'Sleeve':<28} {'Net Capital':>14} {'Gross x':>9} "
            f"{'Gross Exp':>14} {'Ann Return':>12} {'Proj P&L':>14}"
        )
        for row in plan["modeled_rows"]:
            print(
                f"{str(row['name'])[:28]:<28} "
                f"{float(row['net_capital']):>14,.2f} "
                f"{float(row['gross_multiple']):>9.2f} "
                f"{float(row['gross_exposure']):>14,.2f} "
                f"{float(row['annual_return']) * 100.0:>11.2f}% "
                f"{float(row['projected_pnl']):>14,.2f}"
            )
        print("\n* indicates a ticker already present in the current holdings CSV.")
        return 0

    if args.scenario_grid:
        if not args.preset:
            raise SystemExit("--scenario-grid requires --preset.")
        flow_values = [float(item.strip()) for item in str(args.flow_grid).split(",") if item.strip()]
        shift_values = [float(item.strip()) for item in str(args.tactical_shift_grid).split(",") if item.strip()]
        rows = []
        for flow in flow_values:
            for shift in shift_values:
                summary = evaluate_plan(
                    current_nav=current_nav,
                    target_nav=target_nav,
                    horizon_days=horizon_days,
                    monthly_net_flow=flow,
                    idle_annual_return=args.idle_annual_return,
                    margin_available=margin_available,
                    sleeves=sleeves,
                    tactical_return_shift=shift,
                )
                rows.append(
                    {
                        "monthly_net_flow": flow,
                        "tactical_return_shift_pct": shift * 100.0,
                        "modeled_ending_nav": summary["modeled_ending_nav"],
                        "gap_to_target": summary["modeled_gap"],
                        "needed_headroom_return_horizon_pct": summary["required_headroom_horizon_return"] * 100.0,
                        "needed_headroom_return_annual_pct": (
                            summary["required_headroom_annual"] * 100.0
                            if summary["required_headroom_annual"] is not None
                            else float("nan")
                        ),
                    }
                )
        out = __import__("pandas").DataFrame(rows).sort_values(
            ["gap_to_target", "tactical_return_shift_pct", "monthly_net_flow"],
            ascending=[True, False, False],
        )
        print("\nScenario Grid")
        print("=" * 110)
        print(f"Preset:                     {args.preset}")
        print(f"Start Date:                 {start_date.isoformat()}")
        print(f"End Date:                   {end_date.isoformat()}")
        print(f"Current NAV:                ${current_nav:,.2f}")
        print(f"Target NAV:                 ${target_nav:,.2f}")
        print(f"Margin Availability:        ${margin_available:,.2f}")
        print(out.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        return 0

    plan = evaluate_plan(
        current_nav=current_nav,
        target_nav=target_nav,
        horizon_days=horizon_days,
        monthly_net_flow=monthly_net_flow,
        idle_annual_return=args.idle_annual_return,
        margin_available=margin_available,
        sleeves=sleeves,
        tactical_return_shift=args.tactical_return_shift,
    )

    print("\nPortfolio Growth Plan")
    print("=" * 88)
    print(f"Start Date:                 {start_date.isoformat()}")
    print(f"End Date:                   {end_date.isoformat()}")
    print(f"Horizon:                    {horizon_days} days ({months:.2f} months)")
    print(f"Current NAV:                ${current_nav:,.2f}")
    print(f"Target NAV:                 ${target_nav:,.2f}")
    print(f"Required Gain:              ${required_gain:,.2f}")
    print(f"Monthly Net Flow:           ${monthly_net_flow:,.2f}")
    print(f"Margin Availability:        ${margin_available:,.2f}")
    print(f"Total Buying Power:         ${plan['total_buying_power']:,.2f}")
    print(f"Required Total Return:      {required_total_return * 100.0:,.2f}%")
    if required_annual is None:
        print("Required Annual Return:     Not solvable with current inputs")
        print("Required Monthly Return:    Not solvable with current inputs")
    else:
        print(f"Required Annual Return:     {required_annual * 100.0:,.2f}%")
        print(f"Required Monthly Return:    {required_monthly * 100.0:,.2f}%")

    print("\nStrategy Sleeve Plan")
    print(f"Assigned Net Capital:       ${plan['total_sleeve_capital']:,.2f}")
    print(f"Unassigned / Idle Capital:  ${plan['idle_capital']:,.2f}")
    print(f"Total Gross Exposure:       ${plan['total_gross_exposure']:,.2f}")
    print(f"Gross Headroom Remaining:   ${plan['gross_headroom_remaining']:,.2f}")
    if args.preset:
        print(f"Preset Used:                {args.preset}")
    if args.tactical_return_shift:
        print(f"Tactical Return Shift:      {args.tactical_return_shift * 100.0:,.2f}%")
    if plan["modeled_rows"]:
        print(
            f"{'Sleeve':<24} {'Net Capital':>14} {'Gross x':>9} "
            f"{'Gross Exp':>14} {'Ann Return':>12} {'Proj P&L':>14}"
        )
        for row in plan["modeled_rows"]:
            print(
                f"{str(row['name'])[:24]:<24} "
                f"{float(row['net_capital']):>14,.2f} "
                f"{float(row['gross_multiple']):>9.2f} "
                f"{float(row['gross_exposure']):>14,.2f} "
                f"{float(row['annual_return']) * 100.0:>11.2f}% "
                f"{float(row['projected_pnl']):>14,.2f}"
            )
    else:
        print("No sleeves provided.")

    if plan["idle_capital"] > 0:
        print(f"Idle Capital Return Assumption: {args.idle_annual_return * 100.0:,.2f}%")
        print(f"Idle Capital Projected P&L:     ${plan['idle_pnl']:,.2f}")

    print("\nModeled Outcome")
    print(f"Modeled Strategy P&L:       ${plan['total_modeled_pnl']:,.2f}")
    print(f"Modeled Net Flows:          ${plan['net_flows_total']:,.2f}")
    print(f"Modeled Ending NAV:         ${plan['modeled_ending_nav']:,.2f}")
    print(f"Gap To Target:              ${plan['modeled_gap']:,.2f}")
    if target_nav > 0:
        print(f"Gap To Target (% target):   {plan['modeled_gap'] / target_nav * 100.0:,.2f}%")
    if plan["modeled_gap"] > 0 and plan["gross_headroom_remaining"] > 0:
        print(f"Needed Return On Remaining Headroom (horizon): {plan['required_headroom_horizon_return'] * 100.0:,.2f}%")
        if plan["required_headroom_annual"] is not None:
            print(f"Needed Return On Remaining Headroom (annualized): {plan['required_headroom_annual'] * 100.0:,.2f}%")

    print("\nNotes")
    print("1. Annual return inputs are total-return assumptions for each sleeve.")
    print("2. Gross multiple lets you model margin-enhanced sleeves separately from net capital.")
    print("3. Margin availability is treated as extra gross buying power, not as a broker-specific house margin model.")
    print("4. This is a planning calculator, not a broker margin model or risk engine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
