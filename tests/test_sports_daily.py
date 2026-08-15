from datetime import datetime, timedelta, timezone
import os
import unittest
from unittest.mock import patch

from prediction_agent.sports_daily import _find_schedule_match, _is_major_cs2_event, _is_major_lol_event, _probability_sanity, _research_row


class SportsDailyTests(unittest.TestCase):
    def test_cs2_major_event_filter(self):
        self.assertTrue(_is_major_cs2_event("IEM Katowice 2026"))
        self.assertTrue(_is_major_cs2_event("BLAST Premier World Final"))
        self.assertTrue(_is_major_cs2_event("PGL Major Copenhagen"))
        self.assertTrue(_is_major_cs2_event("Esports World Cup 2026"))
        self.assertFalse(_is_major_cs2_event("CCT Online Finals"))
        self.assertFalse(_is_major_cs2_event("ESEA Advanced"))

    def test_lol_major_event_filter(self):
        self.assertTrue(_is_major_lol_event("LPL Group Ascend"))
        self.assertTrue(_is_major_lol_event("LCK 2026"))
        self.assertTrue(_is_major_lol_event("EWC 2026"))
        self.assertTrue(_is_major_lol_event("Esports World Cup 2026"))
        self.assertTrue(_is_major_lol_event("Worlds 2026"))
        self.assertFalse(_is_major_lol_event("EBL Regular Season"))
        self.assertFalse(_is_major_lol_event("Ultraliga"))
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


    def test_paper_trading_enabled_allows_virtual_bet_without_probability_approval(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        market = {"liquidity": 1000, "spread": .01, "bestBid": .60, "bestAsk": .60}
        with patch.dict(os.environ, {"PAPER_TRADING_ENABLED": "true"}):
            row = _research_row(
                "cs2", {"id": "e1", "title": "A vs B"}, market,
                now + timedelta(hours=1), ["A", "B"], [.60, .40], [.80, .20],
                probability_ok=False, money_ok=False, now=now, bankroll=1000, reasons=[],
            )
        self.assertEqual(row["action"], "BET")
        self.assertFalse(row["real_money_approved"])
        self.assertGreater(float(row["stake"]), 0)

    def test_paper_trading_disabled_keeps_no_bet_without_probability_approval(self):
        now = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        market = {"liquidity": 1000, "spread": .01, "bestBid": .60, "bestAsk": .60}
        with patch.dict(os.environ, {"PAPER_TRADING_ENABLED": "false"}):
            row = _research_row(
                "cs2", {"id": "e1", "title": "A vs B"}, market,
                now + timedelta(hours=1), ["A", "B"], [.60, .40], [.80, .20],
                probability_ok=False, money_ok=False, now=now, bankroll=1000, reasons=[],
            )
        self.assertEqual(row["action"], "NO_BET")

if __name__ == "__main__":
    unittest.main()
