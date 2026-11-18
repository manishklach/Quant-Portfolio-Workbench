import unittest
from io import StringIO
from datetime import datetime
from unittest.mock import patch

import pandas as pd

import after_hours_portfolio_pnl as ah


class ExtendedSessionTests(unittest.TestCase):
    def test_summary_uses_only_after_hours_and_handles_missing(self):
        quotes = {
            "XLK": {"post": 166.88, "yahoo_post_asof": "2026-11-17T19:56:00-05:00", "premarket": 170.0},
            "SPY": {"premarket": 700.0},
            "QQQ": {"post": 665.93, "yahoo_post_asof": "2026-11-17T19:59:00-05:00"},
        }
        with patch.object(ah, "_BATCH_QUOTES", quotes), patch("sys.stdout", new_callable=StringIO) as output:
            ah.print_final_ah_quotes()
        self.assertIn("XLK: $166.88", output.getvalue())
        self.assertIn("SPY: N/A", output.getvalue())
        self.assertIn("QQQ: $665.93", output.getvalue())
        self.assertNotIn("170.00", output.getvalue())

    def quotes(self, now, timestamps, prices):
        index = pd.DatetimeIndex(timestamps, tz="America/New_York")
        hist = pd.DataFrame({"Close": prices}, index=index)
        with patch.object(ah.yf, "download", return_value=hist), patch.object(
            ah, "now_eastern", return_value=datetime.fromisoformat(now)
        ):
            return ah.build_batch_quote_cache(["XLK"])["XLK"]

    def test_completed_postmarket_survives_midnight(self):
        quote = self.quotes(
            "2026-11-18T00:30:00-05:00",
            ["2026-11-17 15:59", "2026-11-17 19:59"],
            [163.0, 165.0],
        )
        self.assertEqual(quote["regular"], 163.0)
        self.assertEqual(quote["post"], 165.0)
        self.assertIn("2026-11-17", quote["yahoo_post_asof"])
        self.assertIsNone(quote["overnight"])

    def test_premarket_uses_previous_regular_anchor(self):
        quote = self.quotes(
            "2026-11-18T08:30:00-05:00",
            ["2026-11-17 15:59", "2026-11-17 19:59", "2026-11-18 08:29"],
            [163.0, 165.0, 166.0],
        )
        self.assertEqual(quote["regular"], 163.0)
        self.assertEqual(quote["post"], 165.0)
        self.assertEqual(quote["premarket"], 166.0)
        with patch.object(ah, "current_extended_session_bucket", return_value="pre_market"):
            self.assertEqual(ah.choose_after_hours_price("XLK", quote, {}), (166.0, "yahoo_pre"))

    def test_old_overnight_bar_not_reused(self):
        quote = self.quotes(
            "2026-11-18T00:30:00-05:00",
            ["2026-11-17 00:15", "2026-11-17 15:59", "2026-11-17 19:59"],
            [160.0, 163.0, 165.0],
        )
        self.assertIsNone(quote["overnight"])


if __name__ == "__main__":
    unittest.main()
