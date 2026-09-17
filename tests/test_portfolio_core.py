"""Tests for portfolio_core parsing, normalization, and shared config."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from portfolio_core import (
    active_option_positions,
    clean_numeric,
    config_value,
    load_config,
    load_schwab_holdings,
    parse_option_symbol,
)

FIXTURE_CSV = """Schwab Positions Export
as of 09/16/2026
Symbol,Security Type,Qty (Quantity),Price,Mkt Val (Market Value),Day Chng $ (Day Change $),Gain $ (Gain/Loss $),Cost Basis
SNAXX,Cash & Cash Investments,1000,1.0,"$1,000.00",$0.00,$0.00,"$1,000.00"
QQQI,ETFs & Closed End Funds,100,50.00,"$5,000.00",$10.00,$200.00,"$4,800.00"
MU 03/19/2027 590 C,Option,-5,12.50,"($6,250.00)",($100.00),($500.00),"($5,750.00)"
"""


@pytest.fixture()
def holdings_csv(tmp_path: Path) -> Path:
    p = tmp_path / "my_holdings.csv"
    p.write_text(FIXTURE_CSV)
    return p


def test_clean_numeric():
    assert clean_numeric("$1,234.56") == pytest.approx(1234.56)
    assert clean_numeric("(1.5)") == pytest.approx(-1.5)
    assert clean_numeric("($6,250.00)") == pytest.approx(-6250.0)
    assert clean_numeric("N/A") == 0.0
    assert clean_numeric("-") == 0.0
    assert clean_numeric("") == 0.0
    assert clean_numeric(None) == 0.0
    assert clean_numeric("12.5%") == pytest.approx(12.5)


def test_parse_option_symbol_valid():
    assert parse_option_symbol("MU 03/19/2027 590 C") == ("MU", "03/19/2027", 590.0, "C")
    assert parse_option_symbol("TSLA 04/17/2026 250 P") == ("TSLA", "04/17/2026", 250.0, "P")
    assert parse_option_symbol("  NVDA 01/15/2027 150.5 C  ") == ("NVDA", "01/15/2027", 150.5, "C")


def test_parse_option_symbol_invalid():
    assert parse_option_symbol("QQQI") is None
    assert parse_option_symbol("MU 590 C") is None  # missing expiry
    assert parse_option_symbol("") is None
    assert parse_option_symbol(None) is None


def test_parse_option_symbol_never_confuses_price_for_strike():
    # Regression test for the historical bug where Schwab's Price column was
    # treated as the strike, producing fake spreads like 590.495/647.361.
    parsed = parse_option_symbol("MU 03/19/2027 590 C")
    assert parsed[2] == pytest.approx(590.0)
    assert parsed[2] != pytest.approx(590.495)


def test_load_schwab_holdings(holdings_csv: Path):
    df = load_schwab_holdings(holdings_csv, as_of=date(2026, 9, 16))
    assert len(df) == 3
    assert {"Qty", "Price Numeric", "Market Value Numeric", "Day Change Numeric"}.issubset(df.columns)

    opt = df[df["Is Option"]].iloc[0]
    assert opt["Underlying"] == "MU"
    assert opt["Strike Price"] == pytest.approx(590.0)
    assert opt["Opt Type"] == "C"
    assert opt["Multiplier"] == 100.0
    assert opt["Price Numeric"] == pytest.approx(12.5)
    assert opt["Market Value Numeric"] == pytest.approx(-6250.0)
    assert not opt["Is Expired"]
    assert opt["Days To Expiry"] == pytest.approx(184, abs=1)

    cash = df[df["Symbol"] == "SNAXX"].iloc[0]
    assert not cash["Is Option"]
    assert cash["Multiplier"] == 1.0


def test_active_option_positions_filters_expired_and_zero_qty(holdings_csv: Path):
    df = load_schwab_holdings(holdings_csv, as_of=date(2026, 9, 16))
    active = active_option_positions(df)
    assert len(active) == 1
    # after expiry, nothing is active
    df_late = load_schwab_holdings(holdings_csv, as_of=date(2028, 1, 1))
    assert active_option_positions(df_late).empty


def test_config_loads_and_has_goal_numbers():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert config_value("market.risk_free_rate") == pytest.approx(0.045)
    assert config_value("portfolio.equity_base") == pytest.approx(16_000_000.0)
    assert config_value("portfolio.target_nav") == pytest.approx(24_000_000.0)
    assert config_value("portfolio.goal_end_date") == "2027-09-13"


def test_config_value_missing_key_returns_default():
    assert config_value("no.such.key", "fallback") == "fallback"
    assert config_value("market.risk_free_rate", 0.01) == pytest.approx(0.045)


def test_config_survives_missing_file(tmp_path: Path):
    # A missing config.yaml must not crash module import paths.
    from portfolio_core import load_config as lc

    assert lc(tmp_path / "does-not-exist.yaml") == {}
