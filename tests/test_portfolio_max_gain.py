import math
import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

import pandas as pd

from portfolio_max_gain import main, payoff_ceiling


def leg(kind, strike, quantity):
    return {'Opt Type': kind, 'Strike Price': strike, 'Qty': quantity}


class FilterTests(unittest.TestCase):
    def run_cli(self, args):
        rows = []
        for ticker in ['AAA', 'BBB']:
            for kind in ['C', 'P']:
                rows.append({**leg(kind, 100, -1), 'Underlying': ticker,
                             'Expiration': '2029-01-19',
                             'Market Value Numeric': -100 if kind == 'C' else -200,
                             'Cost Basis Numeric': -300})
        output = io.StringIO()
        with patch('portfolio_max_gain.load_schwab_holdings'), patch(
                'portfolio_max_gain.active_option_positions', return_value=pd.DataFrame(rows)), redirect_stdout(output):
            main(args)
        return output.getvalue()

    def test_filters_apply_to_totals(self):
        for args, expected in [([], '$600.00'), (['--calls'], '$200.00'),
                               (['--puts'], '$400.00'),
                               (['--puts', '--ticker', 'aaa'], '$200.00'),
                               (['--calls', '--ticker', 'aaa'], '$100.00')]:
            with self.subTest(args=args):
                self.assertIn('Bounded-expiration remaining gain from current marks: ' + expected,
                              self.run_cli(args))

    def test_empty_selection(self):
        self.assertIn('No matching active options.', self.run_cli(['--puts', '--ticker', 'NONE']))

    def test_conflicting_filters(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            main(['--calls', '--puts'])
        self.assertEqual(result.exception.code, 2)


class PayoffTests(unittest.TestCase):
    def test_verticals_and_standalone(self):
        self.assertEqual(payoff_ceiling([leg('C', 100, 2), leg('C', 110, -2)])[0], 2000)
        self.assertEqual(payoff_ceiling([leg('P', 80, -2), leg('P', 70, 2)])[0], 0)
        self.assertEqual(payoff_ceiling([leg('P', 80, 2)])[0], 16000)
        self.assertTrue(math.isinf(payoff_ceiling([leg('C', 100, 1)])[0]))

    def test_combined_maxima_are_not_additive(self):
        legs = [leg('C', 100, 1), leg('C', 110, -1),
                leg('P', 100, 1), leg('P', 90, -1)]
        self.assertEqual(payoff_ceiling(legs)[0], 1000)

    def test_ratio_and_bear_spread(self):
        self.assertEqual(payoff_ceiling([leg('C', 100, 1), leg('C', 110, -2)])[0], 1000)
        self.assertEqual(payoff_ceiling([leg('C', 100, -1), leg('C', 110, 1)])[0], 0)


if __name__ == '__main__':
    unittest.main()
