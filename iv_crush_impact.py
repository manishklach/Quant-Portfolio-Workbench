"""Estimate approximate option portfolio sensitivity to implied-volatility crush scenarios."""

import argparse
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from portfolio_core import active_option_positions, default_csv_path, load_schwab_holdings
from option_math import bs_price_vec, bs_vega_vec


@dataclass
class FitParams:
    spot: float
    sigma: float
    used_points: int


def fit_spot_sigma(group_df, r):
    if group_df.empty:
        return None

    strikes = group_df['Strike Price'].to_numpy(dtype=float)
    prices = group_df['Option Price'].to_numpy(dtype=float)
    opt_types = group_df['Opt Type'].to_numpy(dtype=str)
    t = float(group_df['T'].iloc[0])

    valid = (strikes > 0) & (prices > 0) & np.isfinite(strikes) & np.isfinite(prices)
    strikes = strikes[valid]
    prices = prices[valid]
    opt_types = opt_types[valid]

    if len(prices) < 2 or t <= 0:
        return None

    min_k = float(np.min(strikes))
    max_k = float(np.max(strikes))
    if min_k <= 0 or max_k <= 0:
        return None

    s_grid = np.linspace(min_k * 0.7, max_k * 1.3, 90)
    sigma_grid = np.linspace(0.08, 2.0, 90)

    best_err = float('inf')
    best_s = None
    best_sigma = None

    for s in s_grid:
        for sigma in sigma_grid:
            model_prices = bs_price_vec(s, strikes, t, r, sigma, opt_types)
            denom = np.maximum(prices, 0.25)
            err = np.mean(((model_prices - prices) / denom) ** 2)
            if err < best_err:
                best_err = err
                best_s = s
                best_sigma = sigma

    if best_s is None or best_sigma is None:
        return None

    return FitParams(spot=float(best_s), sigma=float(best_sigma), used_points=int(len(prices)))


def match_call_spreads_for_scenario(calls_df, option_delta_per_contract_col):
    total_spread_pnl = 0.0

    for (_, _), grp in calls_df.groupby(['Ticker', 'Expiration']):
        longs = []
        shorts = []
        for _, row in grp.iterrows():
            qty = float(row['Qty'])
            if qty > 0:
                longs.append({
                    'strike': float(row['Strike Price']),
                    'rem': qty,
                    'pnl_per_contract': float(row[option_delta_per_contract_col]),
                })
            elif qty < 0:
                shorts.append({
                    'strike': float(row['Strike Price']),
                    'rem': abs(qty),
                    'pnl_per_contract': -float(row[option_delta_per_contract_col]),
                })

        longs.sort(key=lambda leg: leg['strike'], reverse=True)
        shorts.sort(key=lambda leg: leg['strike'], reverse=True)

        for s_leg in shorts:
            while s_leg['rem'] > 0:
                candidates = [l for l in longs if l['rem'] > 0 and l['strike'] < s_leg['strike']]
                if not candidates:
                    break
                long_leg = max(candidates, key=lambda l: l['strike'])
                matched = min(s_leg['rem'], long_leg['rem'])
                total_spread_pnl += matched * (long_leg['pnl_per_contract'] + s_leg['pnl_per_contract'])
                s_leg['rem'] -= matched
                long_leg['rem'] -= matched

    return total_spread_pnl


def main():
    parser = argparse.ArgumentParser(
        description='Estimate option portfolio P/L from an IV crush tied to VIX drop scenarios.'
    )
    parser.add_argument('--file', default=None, help='Path to holdings CSV (default: my_holdings.csv next to script)')
    parser.add_argument('--r', type=float, default=0.04, help='Risk-free rate as decimal (default: 0.04)')
    parser.add_argument(
        '--as-of',
        default=None,
        help='Valuation date YYYY-MM-DD (default: today in local system date)',
    )
    args = parser.parse_args()

    as_of = datetime.strptime(args.as_of, '%Y-%m-%d').date() if args.as_of else datetime.now().date()
    csv_path = default_csv_path(args.file, __file__)

    df = load_schwab_holdings(csv_path, as_of=as_of)
    df = active_option_positions(df)
    df['Ticker'] = df['Underlying']
    df['Option Price'] = df['Price Numeric']
    df = df[(df['Qty'] != 0) & (df['Option Price'] > 0)].copy()

    df['Expiry Date'] = pd.to_datetime(df['Expiration'], format='%m/%d/%Y', errors='coerce').dt.date
    df = df[df['Expiry Date'].notna()].copy()
    df['Days'] = (df['Expiry Date'] - as_of).apply(lambda d: d.days if pd.notna(d) else -1)
    df['T'] = df['Days'].clip(lower=1) / 365.0

    fit_map = {}
    for (ticker, expiry), grp in df.groupby(['Ticker', 'Expiration']):
        fit = fit_spot_sigma(grp, args.r)
        if fit is None:
            fit = FitParams(
                spot=float(grp['Strike Price'].median()),
                sigma=0.55,
                used_points=int(len(grp)),
            )
        fit_map[(ticker, expiry)] = fit

    df['Est Spot'] = df.apply(lambda r: fit_map[(r['Ticker'], r['Expiration'])].spot, axis=1)
    df['Est IV'] = df.apply(lambda r: fit_map[(r['Ticker'], r['Expiration'])].sigma, axis=1)

    df['Vega'] = bs_vega_vec(
        df['Est Spot'].to_numpy(dtype=float),
        df['Strike Price'].to_numpy(dtype=float),
        df['T'].to_numpy(dtype=float),
        args.r,
        df['Est IV'].to_numpy(dtype=float),
    )

    scenarios = [0.10, 0.20]

    print('\nIV Crush Stress (Approximate Vega Model)')
    print(f'File: {csv_path}')
    print(f'As-Of Date: {as_of}')
    print(f'Groups Fitted: {len(fit_map)} option chains')

    for drop in scenarios:
        shock_label = f'VIX -{int(drop * 100)}%'
        delta_sigma = -drop * df['Est IV']

        df[f'1c_delta_{int(drop * 100)}'] = df['Vega'] * delta_sigma * 100.0

        puts_df = df[df['Opt Type'] == 'P'].copy()
        calls_df = df[df['Opt Type'] == 'C'].copy()

        puts_total = (puts_df['Qty'] * puts_df[f'1c_delta_{int(drop * 100)}']).sum()

        call_spreads_total = match_call_spreads_for_scenario(calls_df, f'1c_delta_{int(drop * 100)}')
        total_calls = (calls_df['Qty'] * calls_df[f'1c_delta_{int(drop * 100)}']).sum()
        unmatched_calls = total_calls - call_spreads_total

        grand_total = puts_total + call_spreads_total

        print(f'\nScenario: {shock_label}')
        print(f'  Call Spreads P/L: ${call_spreads_total:,.2f}')
        print(f'  Puts P/L:         ${puts_total:,.2f}')
        print(f'  Total (Spreads + Puts): ${grand_total:,.2f}')
        print(f'  Unmatched Calls (not in spread pairing): ${unmatched_calls:,.2f}')


if __name__ == '__main__':
    main()
