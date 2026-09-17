"""Shared Schwab portfolio parsing and normalization helpers used by the reporting scripts."""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


EXCLUDED_SYMBOLS = {"Account Total", "Cash & Cash Investments", "Positions Total"}
OPTION_SYMBOL_RE = re.compile(
    r"^\s*(\S+)\s+(\d{2}/\d{2}/\d{4})\s+([0-9]+(?:\.[0-9]+)?)\s+([CP])\s*$"
)
CONFIG_FILENAME = "config.yaml"

_config_cache: dict | None = None


def config_path() -> Path:
    """Path to the shared config.yaml next to this module."""
    return Path(__file__).with_name(CONFIG_FILENAME)


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (cached). Missing file or keys -> empty dict; callers apply defaults."""
    global _config_cache
    if path is None and _config_cache is not None:
        return _config_cache
    cfg: dict = {}
    try:
        with open(config_path() if path is None else path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        cfg = {}
    if path is None:
        _config_cache = cfg
    return cfg


def config_value(dotted: str, default=None):
    """Read a dotted config key, e.g. config_value("market.risk_free_rate", 0.045)."""
    node = load_config()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def clean_numeric(val) -> float:
    if pd.isna(val):
        return 0.0

    val_str = str(val).strip()
    if val_str in {"", "-", "N/A"}:
        return 0.0

    val_str = val_str.replace("$", "").replace("%", "").replace(",", "")
    if "(" in val_str and ")" in val_str:
        val_str = "-" + val_str.replace("(", "").replace(")", "")

    try:
        return float(val_str)
    except ValueError:
        return 0.0


def parse_option_symbol(symbol: str):
    match = OPTION_SYMBOL_RE.match(str(symbol))
    if not match:
        return None
    ticker, expiry, strike, opt_type = match.groups()
    return ticker, expiry, float(strike), opt_type


def find_asset_type_column(df: pd.DataFrame) -> str:
    for col in ("Security Type", "Asset Type"):
        if col in df.columns:
            return col
    raise KeyError("Could not find asset type column. Expected 'Security Type' or 'Asset Type'.")


def default_csv_path(cli_value: str | None, script_file: str) -> Path:
    return Path(cli_value) if cli_value else Path(script_file).with_name("my_holdings.csv")


def load_schwab_holdings(csv_path: str | Path, *, as_of: date | None = None) -> pd.DataFrame:
    csv_path = Path(csv_path)
    as_of = as_of or datetime.now().date()
    as_of_ts = pd.Timestamp(as_of)

    df = pd.read_csv(csv_path, skiprows=2)
    df = df[~df["Symbol"].isin(EXCLUDED_SYMBOLS)].copy()

    asset_type_col = find_asset_type_column(df)
    df["Asset Type Normalized"] = df[asset_type_col].fillna("").astype(str).str.strip()

    numeric_columns = {
        "Qty": "Qty (Quantity)",
        "Price Numeric": "Price",
        "Market Value Numeric": "Mkt Val (Market Value)",
        "Day Change Numeric": "Day Chng $ (Day Change $)",
        "Gain/Loss Numeric": "Gain $ (Gain/Loss $)",
        "Cost Basis Numeric": "Cost Basis",
    }
    for output_col, source_col in numeric_columns.items():
        if source_col in df.columns:
            df[output_col] = df[source_col].apply(clean_numeric)
        else:
            df[output_col] = 0.0

    parsed = df["Symbol"].apply(parse_option_symbol)
    df["Is Option"] = parsed.notna()
    df["Underlying"] = df["Symbol"].astype(str).str.strip()
    df["Expiration"] = ""
    df["Strike Price"] = np.nan
    df["Opt Type"] = ""
    df["Expiry Date"] = pd.NaT

    if df["Is Option"].any():
        parsed_df = pd.DataFrame(
            parsed[df["Is Option"]].tolist(),
            columns=["Underlying", "Expiration", "Strike Price", "Opt Type"],
            index=df[df["Is Option"]].index,
        )
        for col in parsed_df.columns:
            df.loc[parsed_df.index, col] = parsed_df[col]

        df.loc[df["Is Option"], "Expiry Date"] = pd.to_datetime(
            df.loc[df["Is Option"], "Expiration"],
            format="%m/%d/%Y",
            errors="coerce",
        )

    df["Multiplier"] = np.where(df["Is Option"], 100.0, 1.0)
    df["Has Valid Expiry"] = df["Expiry Date"].notna()
    df["Is Expired"] = False
    df.loc[df["Has Valid Expiry"], "Is Expired"] = (
        df.loc[df["Has Valid Expiry"], "Expiry Date"].dt.date < as_of
    )
    df["Days To Expiry"] = np.where(
        df["Has Valid Expiry"],
        (df["Expiry Date"] - as_of_ts).dt.days,
        np.nan,
    )
    return df


def active_option_positions(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Is Option"] & (~df["Is Expired"]) & (df["Qty"] != 0)].copy()


def estimate_equity_equivalent_exposure(df: pd.DataFrame) -> pd.Series:
    exposure = pd.Series(0.0, index=df.index, dtype=float)

    non_options = ~df["Is Option"]
    exposure.loc[non_options] = df.loc[non_options, "Market Value Numeric"]

    options = df["Is Option"] & (~df["Is Expired"])
    intrinsic_reference = np.where(
        df.loc[options, "Strike Price"].fillna(0.0) > 0,
        df.loc[options, "Strike Price"].fillna(0.0),
        df.loc[options, "Price Numeric"].fillna(0.0) * 100.0,
    )
    exposure.loc[options] = (
        df.loc[options, "Qty"]
        * df.loc[options, "Multiplier"]
        * intrinsic_reference
    )
    return exposure
