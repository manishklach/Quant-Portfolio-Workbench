import unittest
from unittest.mock import patch

import pandas as pd

from final_portfolio_noise_checker_v2 import (
    audit_current_call_marks, audit_option_day_pl_baselines,
)


class CurrentMarksTests(unittest.TestCase):
    def options(self, long_price=4.05, short_price=5.05):
        return pd.DataFrame([
            dict(ticker="TEST", expiration="2027-01-15", option_type="CALL",
                 strike=100., quantity=50., price=long_price, day_pl=100.),
            dict(ticker="TEST", expiration="2027-01-15", option_type="CALL",
                 strike=110., quantity=-50., price=short_price, day_pl=100.),
        ])

    def test_bounds_without_network(self):
        result = audit_current_call_marks(self.options(), pd.DataFrame())
        self.assertEqual(result.iloc[0].bounds_adjustment, 5000)
        self.assertTrue(pd.isna(result.iloc[0].signed_valuation_difference))

    def test_overvalued_spread_has_negative_adjustment(self):
        result = audit_current_call_marks(self.options(15, 4), pd.DataFrame())
        self.assertEqual(result.iloc[0].bounds_adjustment, -5000)

    def test_valid_spread_without_quotes_is_not_flagged(self):
        self.assertTrue(audit_current_call_marks(self.options(6, 4), pd.DataFrame()).empty)

    @patch("final_portfolio_noise_checker_v2.fetch_cboe_chain")
    def test_profitable_books_are_fetched_and_theo_retained(self, fetch):
        fetch.return_value = dict(price_change=1, options=[
            dict(option="TEST270115C00100000", bid=4, ask=6, theo=5,
                 prev_day_close=4, delta=.5),
            dict(option="TEST270115C00110000", bid=2, ask=4, theo=3,
                 prev_day_close=2, delta=.3),
        ])
        details, _, errors = audit_option_day_pl_baselines(self.options())
        self.assertEqual(len(details), 2)
        self.assertTrue(errors.empty)
        result = audit_current_call_marks(self.options(), details)
        self.assertEqual(result.iloc[0].signed_valuation_difference, 15000)


if __name__ == "__main__":
    unittest.main()
