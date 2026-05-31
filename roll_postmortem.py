import csv
import re
from collections import defaultdict

CSV_PATH = r"C:\Users\ManishKL\Downloads\Main_Brokerage_XXX266_Transactions_20260530-132245.csv"

def parse_amount(s):
    if not s or s.strip() == "":
        return 0.0
    return float(s.replace("$", "").replace(",", ""))

def parse_qty(s):
    if not s or s.strip() == "":
        return 0
    return int(float(s.replace(",", "")))

def parse_option_symbol(symbol):
    m = re.match(r"^([A-Z]+)\s+(\d{2}/\d{2}/\d{4})\s+([\d.]+)\s+(C|P)$", symbol)
    if m:
        return m.group(1), m.group(2), float(m.group(3)), m.group(4)
    return None

def main():
    option_trades = []
    seen = set()
    with open(CSV_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 8:
                continue
            date, action, symbol, desc, qty_str, price_str, fees_str, amt_str = row[:8]
            amt = parse_amount(amt_str)
            qty = parse_qty(qty_str)
            price = parse_amount(price_str)
            fees = parse_amount(fees_str)
            opt = parse_option_symbol(symbol)
            if opt:
                underlying, expiry, strike, opt_type = opt
                dedup_key = (date, action, symbol, amt)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                option_trades.append({
                    "date": date, "action": action, "underlying": underlying,
                    "expiry": expiry, "strike": strike, "opt_type": opt_type,
                    "qty": qty, "price": price, "fees": fees, "amount": amt,
                })

    if not option_trades:
        print("No option trades found!")
        return

    targets = ["GOOG", "TSM", "MU", "AAPL", "NVDA", "INTC"]
    call_trades = [t for t in option_trades if t["underlying"] in targets and t["opt_type"] == "C"]

    print("=" * 70)
    print("  CALL SPREAD POST-MORTEM")
    print("=" * 70)
    print(f"\nDeduped option trades: {len(option_trades)}")
    print(f"Call trades for targets: {len(call_trades)}")
    print()

    total_pl = 0
    total_fees = 0

    for underlying in targets:
        und_trades = [t for t in call_trades if t["underlying"] == underlying]
        if not und_trades:
            continue

        sorted_trades = sorted(und_trades, key=lambda t: t["date"])

        print(f"\n{'=' * 70}")
        print(f"  {underlying}")
        print(f"{'=' * 70}")

        # Net P&L by strike
        combined = defaultdict(lambda: {"net_amt": 0.0, "net_qty": 0})
        for t in sorted_trades:
            key = (t["expiry"], t["strike"])
            combined[key]["net_amt"] += t["amount"]
            if "Open" in t["action"]:
                sign = 1 if "Buy" in t["action"] else -1
            else:
                sign = -1 if "Buy" in t["action"] else 1
            combined[key]["net_qty"] += sign * t["qty"]

        print(f"  {'Expiry':<14} {'Strike':<8} {'Net P&L':<14} {'Net Qty':<8} {'Interpretation'}")
        print(f"  {'-' * 70}")

        pl_by_strike = 0.0
        for (expiry, strike), data in sorted(combined.items()):
            qty = data["net_qty"]
            interp = ""
            if abs(qty) > 0:
                if qty > 0:
                    interp = f"Net LONG {abs(qty)}x"
                else:
                    interp = f"Net SHORT {abs(qty)}x"
            else:
                interp = "Closed flat"
            print(f"  {expiry:<14} ${strike:<5.0f}  {data['net_amt']:>+10,.2f}   {qty:>+4}   {interp}")
            pl_by_strike += data["net_amt"]

        fees_total = sum(t["fees"] for t in sorted_trades)
        total_fees += fees_total
        total_pl += pl_by_strike

        print(f"  {'-' * 70}")
        print(f"  TOTAL CALL P&L: ${pl_by_strike:>+10,.2f}  (fees: ${fees_total:,.2f})")

    print(f"\n{'=' * 50}")
    print(f"  FINAL SUMMARY")
    print(f"{'=' * 50}")

    # Recompute for final summary
    cum_pl = 0
    cum_fees = 0
    for underlying in targets:
        und_trades = [t for t in call_trades if t["underlying"] == underlying]
        if not und_trades:
            continue
        pl = sum(t["amount"] for t in und_trades)
        fees = sum(t["fees"] for t in und_trades)
        cum_pl += pl
        cum_fees += fees
        mark = ""
        if pl > 100000:
            mark = " <<< PROFITABLE"
        elif pl < -100000:
            mark = " <<< LOSS"
        print(f"  {underlying:<8} ${pl:>+10,.2f}  (fees: ${fees:>7,.2f}){mark}")
    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':<8} ${cum_pl:>+10,.2f}  (fees: ${cum_fees:>7,.2f})")
    print(f"  {'Net after fees':<8} ${cum_pl - cum_fees:>+10,.2f}")

    print(f"\n  INTERPRETATION:")
    print(f"  These are NET P&L figures from ALL call spread trading")
    print(f"  on each underlying - including original openings, rolls,")
    print(f"  and final closings. Positive = you made money overall on")
    print(f"  call spreads in this name. Negative = total losses from")
    print(f"  rolling, being assigned, and closing at a loss.")
    print(f"\n  NOTE: Data is from 12/01/2025 - 05/29/2026")
    print(f"  (may not capture full trade history - CSV scope ~6 months)")

if __name__ == "__main__":
    main()
