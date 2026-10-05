import math
import unittest

from portfolio_max_gain import payoff_ceiling


def leg(kind, strike, quantity):
    return {'Opt Type': kind, 'Strike Price': strike, 'Qty': quantity}


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
