# Schwab Portfolio Utilities

Local Python utilities for analyzing a Schwab portfolio export on your own machine.

The current primary script in this repo is `final_portfolio_noise_checker_v2.py`. It is a mark-noise and sanity-check tool, not an official P/L calculator, accounting system, or tax report. Its purpose is to help reconcile Schwab's displayed option day P/L against simpler economic checks for a narrow set of call spreads and short puts.

## Quick Start

Install dependencies:

```powershell
python -m pip install pandas yfinance numpy openpyxl requests
```

Run the latest checker against a Schwab holdings CSV:

```powershell
python.exe .\final_portfolio_noise_checker_v2.py .\my_holdings.csv
```

Optional: force yfinance IV plus Black-Scholes delta instead of any CSV delta column:

```powershell
python.exe .\final_portfolio_noise_checker_v2.py .\my_holdings.csv --use-yf-delta-only
```

Optional: override rate and dividend assumptions for the Black-Scholes fallback:

```powershell
python.exe .\final_portfolio_noise_checker_v2.py .\my_holdings.csv --risk-free-rate 0.04 --dividend-yield 0.01
```

Estimate after-hours portfolio P/L from a Schwab holdings export:

```powershell
python.exe .\after_hours_portfolio_pnl.py .\my_holdings.csv
```

Use Coinbase equity perpetuals as the preferred overnight price source for supported names such as `MU`, `BE`, `NVDA`, and `TSM`:

```powershell
python.exe .\after_hours_portfolio_pnl.py .\my_holdings.csv --prefer-perp
```

Force the `DRAM` ETF to use a component basket proxy when you want memory-stock overnight moves reflected through its holdings:

```powershell
python.exe .\after_hours_portfolio_pnl.py .\my_holdings.csv --prefer-etf-proxy
```

## What This Tool Is

`final_portfolio_noise_checker_v2.py` looks for two specific situations:

1. Same-expiration vertical call spreads that should have clean intrinsic behavior but Schwab shows a negative day P/L.
2. Out-of-the-money short puts where a simple delta-based expected P/L can be compared to Schwab's displayed day P/L.

It then estimates a clean add-back for apparent mark noise.

This is not official P/L. It is not a broker replacement, not a tax number, and not a full options valuation engine.

## Privacy and Data Handling

This project is designed for local use only.

- Do not commit or share `my_holdings.csv`.
- Do not commit generated `.csv`, `.xls`, or `.xlsx` outputs.
- Keep the project folder local if it contains real account data.
- If you share the code, share only Python files and `README.md`, not your account exports.

The included `.gitignore` is set up to help prevent accidental commits of raw Schwab exports and generated spreadsheet outputs.

## Supported Input

The scripts expect a Schwab portfolio CSV export with Schwab-style columns such as:

- `Symbol`
- `Description`
- `Qty` or `Quantity`
- `Day Chng $` or similar day P/L column
- `Price`
- `Delta` or `Option Delta` if available

Option symbols are expected in Schwab-style text like:

```text
MU 03/19/2027 590 C
TSLA 04/17/2026 250 P
```

or descriptions like:

```text
CALL MICRON TECHNOLOGY I$590 EXP 03/19/27
PUT TESLA INC $250 EXP 04/17/26
```

## Current Script

Primary script:

- `final_portfolio_noise_checker_v2.py`

Additional utility:

- `after_hours_portfolio_pnl.py`

If other older scripts exist in the repo, treat this file as the intended final noise-checker version unless you are explicitly debugging an older workflow.

## After-Hours Estimator

`after_hours_portfolio_pnl.py` estimates after-hours or overnight mark changes for the holdings in a Schwab export.

For stocks and ETFs:

```text
estimated_ah_pl =
shares × (after_hours_price - regular_price)
```

For options, the script:

- reads the regular-session option mark from the CSV
- infers implied volatility from that mark
- reprices the option using the updated after-hours underlying price
- falls back to an intrinsic-change approximation if IV inference fails

### Price Sources

Default behavior:

- use Yahoo `postMarketPrice` when available
- otherwise use Yahoo `preMarketPrice`
- otherwise fall back to Coinbase equity perpetual prices for supported names

Perpetual-futures support:

- `--list-perps` prints which held tickers currently have Coinbase equity perpetuals
- `--prefer-perp` makes the script prefer the Coinbase perpetual price even if Yahoo post-market data exists

Current supported held names are determined live from Coinbase's public equity perpetual feed and, when last checked, included names such as:

- `AAPL`
- `BE`
- `INTC`
- `META`
- `MSFT`
- `MU`
- `NVDA`
- `SNDK`
- `TSLA`
- `TSM`

For perpetual pricing, the script prefers:

1. Coinbase perpetual `index_price`
2. Coinbase perpetual last `price`
3. Coinbase perpetual `mid_market_price`

This keeps the overnight proxy closer to the underlying reference than to a noisy single trade.

### DRAM ETF Proxy

`DRAM` can also be estimated from its memory-stock basket when you want moves in names such as `MU`, `SNDK`, `SK Hynix`, `Samsung`, or `Kioxia` to flow through the ETF estimate.

Use:

```powershell
python.exe .\after_hours_portfolio_pnl.py .\my_holdings.csv --prefer-etf-proxy
```

Or combine it with perpetual pricing:

```powershell
python.exe .\after_hours_portfolio_pnl.py .\my_holdings.csv --prefer-perp --prefer-etf-proxy
```

Notes:

- by default, if `DRAM` itself has a real Yahoo after-hours quote, the script uses that direct quote
- `--prefer-etf-proxy` overrides that and uses the basket-based estimate instead
- the proxy may be partial if not every component has a usable live quote, so the output includes `etf_proxy_coverage_pct`

### Output Columns

The position-level CSV produced by `after_hours_portfolio_pnl.py --output ...` includes fields such as:

- `regular_underlying`
- `after_hours_underlying`
- `after_hours_source`
- `perp_symbol`
- `perp_index_price`
- `perp_last_price`
- `etf_proxy_coverage_pct`
- `estimated_ah_price`
- `estimated_ah_pl`
- `pricing_method`

## Methodology

### Overview

The script parses option rows from a Schwab holdings CSV, fetches stock quotes, and applies two separate rule sets:

- intrinsic-only logic for qualifying ITM same-expiration vertical call spreads
- delta-based logic for qualifying OTM short puts

It writes both CSV and Excel outputs and prints a terminal summary including `TOTAL CLEAN ADD-BACK`.

### ITM Call-Spread Rule

The script only flags same-expiration vertical call spreads where all of the following are true:

- same ticker
- same expiration
- long lower-strike call
- short higher-strike call
- current stock price is greater than the short call strike, meaning the spread is ITM and capped
- stock day change is positive
- net spread day P/L is negative

Call spreads do not use delta in this script. They use intrinsic value only.

For a spread with lower strike `L`, upper strike `U`, stock price `S`, and `contracts` contracts:

```text
spread_intrinsic =
max(S - L, 0)
-
max(S - U, 0)
```

Then cap it:

```text
spread_intrinsic = min(max(spread_intrinsic, 0), U - L)
```

Then convert intrinsic to position value:

```text
spread_value = spread_intrinsic × contracts × 100
```

Then compute expected day P/L from yesterday to today:

```text
intrinsic_expected_day_pl =
spread_value_today - spread_value_yesterday
```

The script compares that expected intrinsic move to Schwab's displayed spread day P/L:

```text
diff_to_add_back = intrinsic_expected_day_pl - schwab_net_day_pl
```

If a spread was fully capped yesterday and is still fully capped today, `intrinsic_expected_day_pl` is usually zero even if Schwab shows a large negative day P/L.

Example:

- `MU 590/650 C`, 50 contracts
- if MU was already above 650 yesterday and is still above 650 today, the spread intrinsic is `$60` both days
- expected day P/L is `$0`
- if Schwab shows `-$5,716`, the add-back is about `$5,716`

Another example:

- `XLK 175/190 C`, 50 contracts
- width = `$15`
- max value = `$75,000`
- if yesterday XLK was about `186.8`, intrinsic was about `11.8 × 50 × 100 = $59,000`
- if today XLK is above `190`, intrinsic is capped at `15 × 50 × 100 = $75,000`
- expected intrinsic gain is about `$16,000`
- if Schwab showed `-$1,843`, then the add-back is about `$17,843`

### OTM Short-Put Rule

The script only checks short puts where:

- current stock price is greater than the put strike

For those rows, expected day P/L is:

```text
expected_day_pl =
abs(put_delta) × stock_day_change × contracts × 100
```

If `stock_day_change` is negative, expected P/L is negative.

The comparison to Schwab is:

```text
diff_to_add_back =
delta_expected_day_pl - schwab_day_pl
```

Only positive put `diff_to_add_back` values are counted in the final put add-back summary.

Example:

- short 50 `MU 600P`
- put delta = `0.14`
- stock up `$47`
- expected P/L = `0.14 × 47 × 50 × 100 = $32,900`
- if Schwab shows `+$9,994`, the add-back is about `$22,906`

### Delta Source and Black-Scholes Fallback

Preferred source:

- if the CSV contains `Delta` or `Option Delta`, the script uses that broker/platform delta by default

The script normalizes these formats correctly for short-put calculations:

- `-0.14`
- `0.14`
- `-14`
- `14`

All of the above are treated as an absolute put delta of `0.14` for expected short-put P/L.

Fallback behavior:

- if no usable CSV delta exists, or if you run with `--use-yf-delta-only`, the script fetches option-chain implied volatility from `yfinance`
- it then computes a Black-Scholes long-put delta

Formula used:

```text
put_delta = -exp(-qT) × N(-d1)
```

where:

```text
d1 = [ln(S/K) + (r - q + 0.5σ²)T] / [σ√T]
```

Definitions:

- `S` = current stock price
- `K` = strike
- `T` = years to expiration
- `r` = risk-free rate
- `q` = dividend yield
- `σ` = implied volatility
- `N` = standard normal CDF

Defaults:

- risk-free rate = `4.5%`
- dividend yield = `0%`

CLI override example:

```powershell
python.exe .\final_portfolio_noise_checker_v2.py .\my_holdings.csv --risk-free-rate 0.04 --dividend-yield 0.01
```

### Parser Fix and Important Bug Note

The script must parse strikes from the Schwab option symbol or description, not from the Schwab `Price` column.

A prior bug incorrectly treated Schwab's `Price` column as if it were `Strike Price`, which produced fake spreads like:

```text
590.495/647.361 C
```

Those are option prices, not strikes.

Correct spreads should look like:

```text
590/650 C
340/350 C
175/190 C
```

Debugging note:

- if the output shows decimal strike spreads that look like option prices, the parser is wrong or the wrong script version is being run

### Calendars and Diagonals Are Intentionally Excluded

Calendar spreads and diagonal spreads are intentionally not included in the intrinsic-only call-spread add-back rule.

They depend on factors such as:

- front-month delta
- back-month delta
- theta
- vega
- term structure
- dividends
- early assignment risk
- bid/ask marks

Because of that, they should not be judged using the same intrinsic-only rule used for same-expiration vertical call spreads.

## Output Files

The script writes these files:

- `bad_itm_upday_call_spreads.csv`
- `otm_short_put_delta_check.csv`
- `final_noise_summary.csv`
- `final_portfolio_noise_report.xlsx`

Excel workbook sheets:

- `Summary`
- `Bad ITM Up-Day Calls`
- `OTM Short Puts`
- `Quotes`
- `Parsed Options`

### Key Output Columns

For call spreads:

- `ticker`
- `expiration`
- `spread`
- `contracts`
- `stock_change`
- `long_day_pl`
- `short_day_pl`
- `schwab_net_day_pl`
- `intrinsic_expected_day_pl`
- `diff_to_add_back`

For short puts:

- `ticker`
- `expiration`
- `short_put`
- `contracts`
- `stock_change`
- `delta_used_abs`
- `delta_source`
- `schwab_day_pl`
- `delta_expected_day_pl`
- `diff_to_add_back`
- `positive_addback`
- `csv_delta`
- `yf_iv`
- `yf_matched_expiration`
- `yf_note`

Notes on put columns:

- `positive_addback` is created for the final put summary and clips negative values to zero
- `delta_source` is typically `csv/broker_delta` or `yf_iv_black_scholes`
- `yf_note` helps explain expiration or strike matching behavior when `yfinance` fallback is used

## Interpretation

`TOTAL CLEAN ADD-BACK` is the estimated amount by which Schwab's displayed day P/L may be understated under this final rule set.

Interpretation formula:

```text
cleaner estimated day P/L =
broker displayed day P/L + TOTAL CLEAN ADD-BACK
```

Example:

- broker day P/L = `-$300,000`
- `TOTAL CLEAN ADD-BACK = $72,000`
- cleaner estimate = `-$228,000`

Again, this is not official P/L and not an accounting or tax number.

## Troubleshooting

### If You Get Zero Rows

If the script prints:

```text
No bad ITM up-day call spreads found.
No OTM short puts found.
TOTAL CLEAN ADD-BACK: $0.00
```

check the following:

- did the script parse option rows at all
- open `final_portfolio_noise_report.xlsx` and inspect the `Parsed Options` sheet
- are strikes correct, such as `590`, `650`, `175`, `190`
- or are they bogus decimal option prices like `590.495`
- open the `Quotes` sheet
- are `last`, `prev_close`, and `stock_change` populated
- were the stocks actually up on the day being checked
- are the call spreads truly current price greater than short strike
- are the puts truly current price greater than put strike
- does the CSV contain a usable day P/L column and quantity column
- are you running the correct latest script

### If Quotes or yfinance Fallbacks Look Wrong

`yfinance` can be delayed, stale, incomplete, or missing for some expirations and strikes.

Limitations of the fallback:

- `yfinance` option data may be delayed or stale
- `yfinance` usually does not provide Greeks directly
- Black-Scholes here is only a fallback approximation
- Schwab or Thinkorswim delta may differ because broker models may use full volatility surfaces, dividends, rates, skew, early exercise assumptions, and American-option logic
- for best accuracy, export a `Delta` column from the broker if possible

If quote fetching fails and a quote-debug version is present in the repo, use:

- `final_portfolio_noise_checker_v3_quote_debug.py`

It supports:

```powershell
python.exe .\final_portfolio_noise_checker_v3_quote_debug.py .\my_holdings.csv --quote-csv .\quotes.csv
```

Manual quote CSV format can be either:

```csv
ticker,last,prev_close
MU,971,923.52
XLK,191,186.83
```

or:

```csv
ticker,last,stock_change
MU,971,47.48
XLK,191,4.17
```

### Missing Columns

If Schwab changes its export format and you see parsing errors, inspect the CSV header row and verify that the expected symbol, quantity, day P/L, and description fields still exist.

### Excel Output Errors

If Excel export fails, install `openpyxl`:

```powershell
python -m pip install openpyxl
```

## Safe Sharing Checklist

Before sharing the repo or code:

1. Remove `my_holdings.csv`.
2. Remove generated `.csv`, `.xls`, and `.xlsx` files.
3. Share only code and docs, not account data.
