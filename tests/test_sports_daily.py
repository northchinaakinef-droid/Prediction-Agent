from datetime import datetime, timedelta, timezone
import unittest

from prediction_agent.sports_daily import _find_schedule_match, _probability_sanity, _research_row


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


    def test_probability_sanity_flags_extremes_and_divergence(self):
        self.assertTrue(_probability_sanity(0.995, 0.50))
        self.assertTrue(_probability_sanity(0.60, 0.10))
        self.assertFalse(_probability_sanity(0.60, 0.55))

    def test_find_schedule_match_requires_same_sport_and_time(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        schedule = [{
            "sport": "cs2", "team_a": "A", "team_b": "B",
            "start_time": "2026-01-01T12:30:00+00:00",
            "market_mapping_status": "MATCHED",
        }]
        self.assertIsNotNone(_find_schedule_match(schedule, "cs2", "A", "B", now))
        self.assertIsNone(_find_schedule_match(schedule, "cs2", "A", "C", now))
        self.assertIsNone(_find_schedule_match(schedule, "nba", "A", "B", now))

    def test_research_row_marks_market_only_and_suspicious_probability(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        row = _research_row(
            "cs2", {"id": "e1", "title": "A vs B"},
            {"liquidity": 1000, "spread": .01}, now + timedelta(hours=1),
            ["A", "B"], [.999, .001], [.999, .001], probability_ok=True,
            money_ok=False, now=now, bankroll=1000, reasons=[],
        )
        self.assertFalse(row["probability_plausible"])
        self.assertFalse(row["schedule_matched"])
        self.assertEqual(row["market_mapping_status"], "NOT_IN_SCHEDULE")


if __name__ == "__main__":
    unittest.main()
