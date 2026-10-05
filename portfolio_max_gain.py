"""Expiry payoff ceilings for current standard 100-share option positions."""

import argparse
from datetime import date
import math

import pandas as pd

from portfolio_core import active_option_positions, load_schwab_holdings


def payoff_ceiling(legs):
    """Piecewise-linear payoff reaches a finite maximum at zero or a strike."""
    slope = sum(float(r['Qty']) for r in legs if r['Opt Type'] == 'C')
    if slope > 1e-9:
        return math.inf, None
    prices = sorted({0.0, *(float(r['Strike Price']) for r in legs)})
    def payoff(spot):
        return sum(
            float(r['Qty']) * 100 * max(
                spot - float(r['Strike Price']) if r['Opt Type'] == 'C'
                else float(r['Strike Price']) - spot, 0
            ) for r in legs
        )
    best = max(prices, key=payoff)
    return payoff(best), best


def summarize(options, separate_types=False):
    keys = ['Underlying', 'Expiration']
    if separate_types:
        keys.append('Opt Type')
    rows = []
    for key, group in options.groupby(keys, sort=True):
        legs = group.to_dict('records')
        ceiling, spot = payoff_ceiling(legs)
        value = float(group['Market Value Numeric'].sum())
        basis = float(group['Cost Basis Numeric'].sum())
        rows.append({
            'Ticker': key[0], 'Expiry': key[1],
            'Book': key[2] if separate_types else 'Combined',
            'Max payoff': ceiling, 'At stock price': spot,
            'Current value': value, 'Open-leg basis': basis,
            'Max profit vs basis': ceiling - basis,
            'Remaining gain': ceiling - value,
        })
    return pd.DataFrame(rows)


def money(value):
    return 'UNBOUNDED' if math.isinf(value) else f'${value:,.2f}'


def print_table(frame):
    display = frame.copy()
    display['At stock price'] = display['At stock price'].map(
        lambda v: '-' if pd.isna(v) else f'${v:,.2f}'
    )
    for col in ['Max payoff', 'Current value', 'Open-leg basis',
                'Max profit vs basis', 'Remaining gain']:
        display[col] = display[col].map(money)
    print(display.to_string(index=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='?', default=None,
                        help='Holdings CSV; defaults to my_holdings.csv beside this script')
    parser.add_argument('--ticker', help='Filter underlying, e.g. TQQQ')
    parser.add_argument('--as-of', type=date.fromisoformat, default=date.today(),
                        help='Expiry cutoff YYYY-MM-DD')
    from pathlib import Path
    args = parser.parse_args(argv)
    path = Path(args.csv) if args.csv else Path(__file__).with_name('my_holdings.csv')
    options = active_option_positions(load_schwab_holdings(path, as_of=args.as_of))
    if args.ticker:
        options = options[options['Underlying'].str.upper() == args.ticker.strip().upper()]
    if options.empty:
        print('No matching active options.')
        return
    print(f'Source: {path.resolve()}')
    print('Standard 100-share contracts assumed. No live quotes fetched.\n')
    print('CALL AND PUT BOOKS BY EXPIRATION (do not add separate maxima together)')
    print_table(summarize(options, separate_types=True))
    print('\nCOMBINED CALLS + PUTS AT EACH EXPIRATION')
    combined = summarize(options)
    print_table(combined)
    finite = combined[combined['Max payoff'].map(math.isfinite)]
    unlimited = combined[~combined['Max payoff'].map(math.isfinite)]
    print('\nTOTALS')
    print('Bounded-expiration maximum payoff:', money(finite['Max payoff'].sum()))
    print('Bounded-expiration profit versus open-leg basis:', money(finite['Max profit vs basis'].sum()))
    print('Bounded-expiration remaining gain from current marks:', money(finite['Remaining gain'].sum()))
    if not unlimited.empty:
        print('Overall independent-expiry upside ceiling: UNBOUNDED')
        print('Unbounded books:', ', '.join(unlimited['Ticker'] + ' ' + unlimited['Expiry']))
    else:
        print('Overall independent-expiry ceiling equals bounded totals above.')
    multi = options.groupby('Underlying')['Expiration'].nunique()
    if (multi > 1).any():
        print('Multiple expirations:', ', '.join(multi[multi > 1].index))
    print('\nEach maximum assumes its own optimizing expiry stock price; these are not')
    print('a forecast or a single bullish scenario. Cross-expiry diagonals/calendars require')
    print('a price path and exercise/closing plan; the subtotal is not their maximum profit.')
    print('Cost basis covers open legs only, not realized roll results. Remaining gain already')
    print('accounts for current option assets/liabilities; do not add collected premiums again.')


if __name__ == '__main__':
    main()
