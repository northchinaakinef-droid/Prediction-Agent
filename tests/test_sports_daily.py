from datetime import datetime, timedelta, timezone
import unittest

from prediction_agent.sports_daily import _research_row


class SportsDailyTests(unittest.TestCase):
    def test_started_market_does_not_display_edge_or_ev(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        row = _research_row(
            "cs2", {"id": "e1", "title": "A vs B"},
            {"liquidity": 1000, "spread": .01}, now - timedelta(minutes=1),
            ["A", "B"], [.999, .001], [.6, .4], probability_ok=True,
            money_ok=False, now=now, bankroll=1000, reasons=[],
        )
        self.assertTrue(row["market_started"])
        self.assertFalse(row["market_comparison_valid"])
        self.assertIsNone(row["execution_price"])
        self.assertIsNone(row["edge"])
        self.assertIsNone(row["expected_value"])
        self.assertEqual(row["action"], "NO_BET")


if __name__ == "__main__":
    unittest.main()
