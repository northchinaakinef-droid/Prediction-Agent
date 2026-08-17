from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from prediction_agent.delivery import format_attribution_report, format_daily_report
from prediction_agent.paper_store import (
    attribution, mark_weekly_attribution_sent, record_report, settle_pending,
    weekly_attribution_sent,
)
from prediction_agent.risk import paper_recommend
from prediction_agent.sports_daily import (
    _direction_match, _ev_tier, _lineup_status, _paper_daily_report,
)


class FakeSettledClient:
    def event(self, _event_id):
        return {"closedTime": "2026-01-02T00:00:00Z", "markets": [{
            "gameStartTime": "2026-01-01T01:00:00Z", "sportsMarketType": "moneyline",
            "outcomes": '["A", "B"]', "outcomePrices": '["1", "0"]',
        }]}


class PaperModeTests(unittest.TestCase):
    def test_paper_recommend_uses_fixed_half_percent_stake(self):
        rec = paper_recommend(
            event_id="e", outcome="A", model_probability=0.60,
            decimal_odds=1 / 0.55, bankroll=1000, estimated_cost=0.01,
            direction_match=True,
        )
        self.assertEqual(rec.action, "BET")
        self.assertAlmostEqual(rec.stake, 5.0)
        self.assertAlmostEqual(rec.stake_fraction, 0.005)
        self.assertEqual(rec.decision_probability, 0.60)

    def test_paper_recommend_rejects_opposite_direction_unless_ev_over_10_percent(self):
        rec = paper_recommend(
            event_id="e", outcome="A", model_probability=0.43,
            decimal_odds=2.5, bankroll=1000, estimated_cost=0.0,
            direction_match=False,
        )
        self.assertEqual(rec.action, "NO_BET")
        rec2 = paper_recommend(
            event_id="e", outcome="A", model_probability=0.55,
            decimal_odds=2.5, bankroll=1000, estimated_cost=0.0,
            direction_match=False,
        )
        self.assertEqual(rec2.action, "BET")

    def test_paper_recommend_rejects_low_edge(self):
        rec = paper_recommend(
            event_id="e", outcome="A", model_probability=0.55,
            decimal_odds=1 / 0.54, bankroll=1000, estimated_cost=0.0,
            direction_match=True,
        )
        self.assertEqual(rec.action, "NO_BET")

    def test_attribution_helpers(self):
        self.assertTrue(_direction_match([0.6, 0.4], 0))
        self.assertFalse(_direction_match([0.6, 0.4], 1))
        self.assertEqual(_ev_tier(0.16), "高")
        self.assertEqual(_ev_tier(0.06), "中")
        self.assertEqual(_ev_tier(0.05), "低")
        self.assertEqual(_ev_tier(None), "未知")

    def test_lineup_status(self):
        now = datetime(2026, 1, 10, tzinfo=timezone.utc)
        last = {"A": "2026-01-10T00:00:00Z", "B": "2026-01-09T00:00:00Z"}
        roster = tuple(["p1", "p2", "p3", "p4", "p5"])
        self.assertEqual(_lineup_status(roster, roster, last, ("A", "B"), now, 7), "完整")
        old = {"A": "2026-01-01T00:00:00Z", "B": "2026-01-01T00:00:00Z"}
        self.assertEqual(_lineup_status(roster, roster, old, ("A", "B"), now, 7), "过期")
        self.assertEqual(_lineup_status((), roster, last, ("A", "B"), now, 7), "未知")

    def test_paper_daily_report_counts_bets_and_remaining_limit(self):
        rows = [
            {"action": "BET", "stake": 5, "sport": "cs2"},
            {"action": "NO_BET", "stake": 0, "sport": "lol"},
        ]
        report = _paper_daily_report(rows, 1000, committed_fraction=0.015, max_daily_risk_fraction=0.025)
        self.assertEqual(report["bet_count"], 1)
        self.assertEqual(report["skipped_count"], 1)
        self.assertAlmostEqual(report["remaining_limit"], 10.0)
        self.assertEqual(report["by_sport"]["cs2"]["bets"], 1)
        self.assertEqual(report["by_sport"]["lol"]["skipped"], 1)

    def test_record_report_falls_back_for_missing_attribution_fields(self):
        report = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "report_date": "2026-01-01",
            "recommendations": [{
                "generated_at": "2026-01-01T00:00:00+00:00",
                "sport": "nba", "event_id": "1", "outcome": "A",
                "model_probability": .6, "market_probability": .55,
                "execution_price": .55, "action": "BET", "stake": 5,
                "expected_value": .06,
                "probability_eligible": False, "real_money_approved": False,
                "market_started": False,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            record_report(path, report)
            with closing(sqlite3.connect(path)) as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT lineup_status, ev_tier, direction_match FROM predictions"
                    ).fetchone(),
                    ("未知", "中", 1),
                )

    def test_attribution_separates_legacy_missing_fields(self):
        report = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "report_date": "2026-01-01",
            "recommendations": [{
                "generated_at": "2026-01-01T00:00:00+00:00",
                "sport": "nba", "event_id": "complete", "outcome": "A",
                "model_probability": .6, "market_probability": .55,
                "execution_price": .55, "action": "BET", "stake": 5,
                "expected_value": .16,
                "probability_eligible": False, "real_money_approved": False,
                "market_started": False, "lineup_status": "完整",
                "ev_tier": "高", "direction_match": True,
            }, {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "sport": "nba", "event_id": "legacy", "outcome": "A",
                "model_probability": .6, "market_probability": .55,
                "execution_price": .55, "action": "BET", "stake": 5,
                "expected_value": .06,
                "probability_eligible": False, "real_money_approved": False,
                "market_started": False,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            record_report(path, report)
            with closing(sqlite3.connect(path)) as connection:
                connection.execute(
                    "UPDATE predictions SET lineup_status=NULL, ev_tier=NULL, direction_match=NULL "
                    "WHERE event_id='legacy'"
                )
                connection.commit()
            settle_pending(path, FakeSettledClient())
            attr = attribution(path)
            self.assertEqual(attr["sample_count"], 1)
            self.assertEqual(attr["legacy_sample_count"], 1)
            self.assertEqual(attr["total_sample_count"], 2)
            self.assertEqual(attr["by_ev_tier"]["高"]["bets"], 1)
            self.assertEqual(attr["by_sport"]["nba"]["bets"], 1)
            self.assertEqual(attr["by_data_quality"]["历史数据（字段缺失）"]["bets"], 1)

    def test_attribution_groups_settled_bets(self):
        report = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "report_date": "2026-01-01",
            "recommendations": [{
                "generated_at": "2026-01-01T00:00:00+00:00",
                "sport": "nba", "event_id": "1", "outcome": "A",
                "model_probability": .6, "market_probability": .55,
                "execution_price": .55, "action": "BET", "stake": 5,
                "probability_eligible": False, "real_money_approved": False,
                "market_started": False, "lineup_status": "完整",
                "ev_tier": "高", "direction_match": True,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            record_report(path, report)
            settle_pending(path, FakeSettledClient())
            attr = attribution(path)
            self.assertEqual(attr["sample_count"], 1)
            self.assertEqual(attr["by_lineup_status"]["完整"]["bets"], 1)
            self.assertEqual(attr["by_ev_tier"]["高"]["bets"], 1)
            self.assertEqual(attr["by_direction_match"]["一致"]["bets"], 1)
            self.assertEqual(attr["by_sport"]["nba"]["bets"], 1)
            self.assertGreater(attr["by_lineup_status"]["完整"]["profit"], 0)

    def test_weekly_attribution_send_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            self.assertFalse(weekly_attribution_sent(path, "2026-W02"))
            self.assertTrue(mark_weekly_attribution_sent(path, "2026-W02", "2026-01-12T00:00:00+00:00"))
            self.assertTrue(weekly_attribution_sent(path, "2026-W02"))

    def test_attribution_report_format(self):
        attr = {
            "by_lineup_status": {"完整": {"bets": 1, "win_rate": 1.0, "roi": .05}},
            "by_ev_tier": {"高": {"bets": 1, "win_rate": 1.0, "roi": .05}},
            "by_direction_match": {"一致": {"bets": 1, "win_rate": 1.0, "roi": .05}},
            "by_sport": {"nba": {"bets": 1, "win_rate": 1.0, "roi": .05}},
        }
        message = format_attribution_report(attr, "2026-01-12")
        self.assertIn("【每周归因报告】2026-01-12", message)
        self.assertIn("按阵容状态", message)
        self.assertIn("完整", message)
        self.assertIn("ROI +5.0%", message)


if __name__ == "__main__":
    unittest.main()
