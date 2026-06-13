"""Estimate uncovered short put exposure and cash-secured capital required from a Schwab holdings export."""

import argparse
import math
import random
from datetime import datetime, timezone

import pandas as pd
from openpyxl import load_workbook

from portfolio_core import active_option_positions, clean_numeric, default_csv_path, load_schwab_holdings

try:
    import yfinance as yf
except ImportError:
    yf = None

rISK_FREE_RATE = 0.045
DIVIDEND_YIELD = 0.0


FUNNY_STARTUP_MESSAGES = [
    'Warming up the cash-secured put microscope...',
    'Counting naked puts and pretending this is cardio...',
    'Summoning spreadsheet goblins for moneyness inspection...',
    'Peeking under the hood for put-shaped surprises...',
    'Running the premium-powered danger scanner...',
    'Consulting the highly paid committee of suspicious spreadsheets...',
    'Measuring how spicy these short puts really are...',
    'Politely asking the options chain to explain itself...',
]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def year_frac(expiration):
    try:
        exp_dt = pd.Timestamp(expiration).to_pydatetime().replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max((exp_dt - now).total_seconds() / 86400.0 / 365.0, 1 / 365)
    except Exception:
        return float("nan")


def bs_put_delta(spot, strike, time_to_expiry, rate, sigma, dividend_yield=0.0):
    if any(pd.isna(value) for value in [spot, strike, time_to_expiry, rate, sigma, dividend_yield]):
        return float("nan")
    if spot <= 0 or strike <= 0 or time_to_expiry <= 0 or sigma <= 0:
        return float("nan")
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * sigma * sigma) * time_to_expiry
    ) / (sigma * math.sqrt(time_to_expiry))
    return -math.exp(-dividend_yield * time_to_expiry) * norm_cdf(-d1)


def nearest_expiration(ticker_obj, target_expiration):
    try:
        expirations = list(ticker_obj.options or [])
    except Exception:
        return None
    if not expirations:
        return None
    if target_expiration in expirations:
        return target_expiration
    try:
        target_ts = pd.Timestamp(target_expiration)
        return sorted(expirations, key=lambda exp: abs((pd.Timestamp(exp) - target_ts).days))[0]
    except Exception:
        return expirations[0]


def fetch_yf_iv_for_put(ticker, expiration, strike):
    try:
        ticker_obj = yf.Ticker(ticker)
        matched_expiration = nearest_expiration(ticker_obj, expiration)
        if not matched_expiration:
            return float("nan")
        chain = ticker_obj.option_chain(matched_expiration).puts.copy()
        if chain.empty:
            return float("nan")
        chain["strike_diff"] = (chain["strike"].astype(float) - float(strike)).abs()
        row = chain.sort_values("strike_diff").iloc[0]
        return float(row.get("impliedVolatility", float("nan")))
    except Exception:
        return float("nan")


def compute_put_delta(ticker, expiration, strike, stock_price, broker_delta):
    if not pd.isna(broker_delta):
        return broker_delta
    iv = fetch_yf_iv_for_put(ticker, expiration, strike)
    time_to_expiry = year_frac(expiration)
    return bs_put_delta(stock_price, strike, time_to_expiry, rISK_FREE_RATE, iv, DIVIDEND_YIELD)


def fetch_current_stock_prices(tickers):
    unique_tickers = sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})
    if not unique_tickers:
        return {}
    if yf is None:
        raise RuntimeError('yfinance not installed. Run: pip install yfinance')

    prices = {}
    for ticker in unique_tickers:
        try:
            stock = yf.Ticker(ticker)
            fast = {}
            try:
                fast = dict(stock.fast_info)
            except Exception:
                pass

            last_price = fast.get('last_price')
            if last_price is None or pd.isna(last_price):
                last_price = fast.get('lastPrice')
            if last_price is not None and not pd.isna(last_price):
                prices[ticker] = float(last_price)
                continue

            history = stock.history(period='5d', interval='1d', auto_adjust=False)
            if history.empty or 'Close' not in history:
                continue
            closes = history['Close'].dropna()
            if closes.empty:
                continue
            prices[ticker] = float(closes.iloc[-1])
        except Exception:
            continue
    return prices


def autosize_excel_columns(output_path):
    workbook = load_workbook(output_path)
    worksheet = workbook.active

    for column_cells in worksheet.columns:
        column_letter = column_cells[0].column_letter
        max_length = max(len(str(cell.value or '')) for cell in column_cells)
        worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 10), 18)

    workbook.save(output_path)


def format_display_table(df_out):
    display_df = df_out.rename(
        columns={
            'Contracts Sold': 'Qty',
            'Strike Price': 'Strike',
            'Current Stock Price': 'Stock Px',
            'Delta Numeric': 'Delta',
            'Moneyness Status': 'Status',
            'Current Mkt Value': 'Mkt Value',
            'Cash Secured ($)': 'Cash Sec',
        }
    ).copy()

    return display_df.to_string(index=False)


def find_uncovered_short_puts(df_puts, stock_prices):
    uncovered_rows = []
    delta_cache = {}

    for (_, _), group in df_puts.groupby(['Ticker', 'Expiration']):
        longs = []
        shorts = []

        for _, row in group.sort_values(by='Strike Price', ascending=False).iterrows():
            qty = float(row['Qty'])
            if qty > 0:
                longs.append({'strike': float(row['Strike Price']), 'remaining': qty})
            elif qty < 0:
                shorts.append(row.copy())

        for short_row in shorts:
            remaining_short = abs(float(short_row['Qty']))
            short_strike = float(short_row['Strike Price'])

            # Only a lower-strike long put can cap a short put as a vertical spread.
            candidates = [leg for leg in longs if leg['remaining'] > 0 and leg['strike'] < short_strike]
            candidates.sort(key=lambda leg: leg['strike'], reverse=True)

            for long_leg in candidates:
                if remaining_short <= 0:
                    break
                matched = min(remaining_short, long_leg['remaining'])
                remaining_short -= matched
                long_leg['remaining'] -= matched

            if remaining_short > 0:
                uncovered = short_row.copy()
                original_contracts = abs(float(short_row['Qty']))
                ratio = remaining_short / original_contracts if original_contracts else 0.0
                uncovered['Contracts Sold'] = remaining_short
                current_stock_price = stock_prices.get(str(short_row['Ticker']).strip().upper())
                uncovered['Current Stock Price'] = current_stock_price
                strike = float(short_row['Strike Price'])
                expiration = short_row['Expiration']
                delta_key = (str(short_row['Ticker']).strip().upper(), expiration, strike)
                if current_stock_price is None:
                    uncovered['Delta Numeric'] = pd.NA
                else:
                    if delta_key not in delta_cache:
                        delta_cache[delta_key] = compute_put_delta(
                            delta_key[0],
                            expiration,
                            strike,
                            current_stock_price,
                            short_row.get('Delta Numeric', pd.NA),
                        )
                    uncovered['Delta Numeric'] = delta_cache[delta_key]
                if current_stock_price is None:
                    uncovered['Moneyness Status'] = ''
                elif short_strike <= current_stock_price:
                    uncovered['Moneyness Status'] = 'Out of Money'
                else:
                    uncovered['Moneyness Status'] = 'In the Money'
                uncovered['Current Mkt Value'] = clean_numeric(short_row['Mkt Val (Market Value)']) * ratio
                uncovered['Cash Secured ($)'] = remaining_short * 100 * short_strike
                uncovered_rows.append(uncovered)

    if not uncovered_rows:
        columns = list(df_puts.columns) + ['Contracts Sold', 'Current Stock Price', 'Moneyness Status', 'Current Mkt Value', 'Cash Secured ($)']
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(uncovered_rows)


def main():
    parser = argparse.ArgumentParser(description='Find naked short puts and estimate cash-secured requirement')
    parser.add_argument('--file', default=None, help='Path to holdings CSV (default: my_holdings.csv next to script)')
    parser.add_argument('--output', default=None, help='Output Excel path (default: naked_short_puts.xlsx next to script)')
    args = parser.parse_args()

    csv_path = default_csv_path(args.file, __file__)
    output_path = (
        default_csv_path(args.output, __file__).with_name('naked_short_puts.xlsx')
        if args.output is None
        else default_csv_path(args.output, __file__)
    )

    print(f'\n{random.choice(FUNNY_STARTUP_MESSAGES)}\n')

    df = load_schwab_holdings(csv_path)
    df_options = active_option_positions(df)
    df_options['Ticker'] = df_options['Underlying']
    df_puts = df_options[df_options['Opt Type'] == 'P'].copy()
    if 'Delta' in df_puts.columns:
        df_puts['Delta Numeric'] = df_puts['Delta'].apply(clean_numeric)
    else:
        df_puts['Delta Numeric'] = pd.NA
    stock_prices = fetch_current_stock_prices(df_puts['Ticker'])
    df_naked_short_puts = find_uncovered_short_puts(df_puts, stock_prices)

    output_cols = ['Ticker', 'Symbol', 'Contracts Sold', 'Strike Price', 'Current Stock Price', 'Delta Numeric', 'Moneyness Status', 'Current Mkt Value', 'Cash Secured ($)']
    df_out = df_naked_short_puts[output_cols].copy()

    total_row = pd.DataFrame([
        {
            'Ticker': 'GRAND TOTAL',
            'Symbol': '',
            'Contracts Sold': df_out['Contracts Sold'].sum(),
            'Strike Price': 0.0,
            'Current Stock Price': 0.0,
            'Delta Numeric': 0.0,
            'Moneyness Status': '',
            'Current Mkt Value': df_out['Current Mkt Value'].sum(),
            'Cash Secured ($)': df_out['Cash Secured ($)'].sum(),
        }
    ])
    if df_out.empty:
        df_out = total_row.copy()
    else:
        df_out = pd.concat([df_out, total_row], ignore_index=True)

    df_out.to_excel(output_path, index=False)
    autosize_excel_columns(output_path)

    grand_cash = df_naked_short_puts['Cash Secured ($)'].sum()
    grand_mkt = df_naked_short_puts['Current Mkt Value'].sum()

    print(f'\nSuccess! Data exported to Excel file: {output_path}')
    print('\n--- EXCEL CONTENTS ---')
    print(format_display_table(df_out))
    print('\n--- GRAND TOTALS ---')
    print(f'Total Cash Secured: ${grand_cash:,.2f}')
    print(f'Total Current Liability (Mkt Value): ${grand_mkt:,.2f}\n')


if __name__ == '__main__':
    main()
