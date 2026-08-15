import tempfile
from pathlib import Path
import unittest

from prediction_agent.paper_store import (current_drawdown, paper_review_detail,
                                          record_post_match_review, record_report,
                                          settle_pending, summary)


class FakeClient:
    def event(self, _event_id):
        return {"closedTime": "2026-01-02T00:00:00Z", "markets": [{
            "gameStartTime": "2026-01-01T01:00:00Z", "sportsMarketType": "moneyline",
            "outcomes": '["A", "B"]', "outcomePrices": '["1", "0"]',
        }]}


class FailingClient:
    def event(self, _event_id):
        raise RuntimeError("boom")


class LosingClient:
    def event(self, _event_id):
        return {"closedTime": "2026-01-02T00:00:00Z", "markets": [{
            "gameStartTime": "2026-01-01T01:00:00Z", "sportsMarketType": "moneyline",
            "outcomes": '["A", "B"]', "outcomePrices": '["0", "1"]',
        }]}


class PaperStoreTests(unittest.TestCase):
    def test_report_recording_is_append_only_and_idempotent(self):
        report = {"generated_at": "2026-01-01T00:00:00+00:00", "report_date": "2026-01-01",
                  "recommendations": [{"generated_at": "2026-01-01T00:00:00+00:00",
                    "sport": "nba", "event_id": "1", "outcome": "A", "model_probability": .6,
                    "market_probability": .55, "execution_price": .56, "action": "NO_BET",
                    "stake": 0, "probability_eligible": True, "real_money_approved": False,
                    "market_started": False}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            first = record_report(path, report)
            second = record_report(path, report)
            self.assertEqual(first["inserted_predictions"], 1)
            self.assertEqual(second["inserted_predictions"], 0)
            self.assertEqual(summary(path)["predictions"], 1)
            settled = settle_pending(path, FakeClient())
            self.assertEqual(settled["settled"], 1)
            stats = summary(path)
            self.assertEqual(stats["settled"], 1)
            self.assertAlmostEqual(stats["by_sport"]["nba"]["model_brier"], .16)

    def test_settle_pending_surfaces_unexpected_provider_errors(self):
        report = {"generated_at": "2026-01-01T00:00:00+00:00", "report_date": "2026-01-01",
                  "recommendations": [{"generated_at": "2026-01-01T00:00:00+00:00",
                    "sport": "nba", "event_id": "1", "outcome": "A", "model_probability": .6,
                    "market_probability": .55, "execution_price": .56, "action": "NO_BET",
                    "stake": 0, "probability_eligible": True, "real_money_approved": False,
                    "market_started": False}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            record_report(path, report)
            result = settle_pending(path, FailingClient())
            self.assertEqual(result["errors"], 1)
            self.assertEqual(result["settled"], 0)

    def test_post_match_review_detail_persists_series_analysis_and_game_samples(self):
        review = {
            "sport": "lol", "event_id": "series-1", "event": "T1 vs Gen.G",
            "generated_at": "2026-01-01T00:00:00+00:00", "actual_winner": "T1",
            "predicted_winner": "T1", "prediction_correct": True,
            "model_probability": .55, "bp_probability": .55,
            "decisive_factors": ["经济领先"],
            "game_samples": [{"game_index": 1, "blue_post_draft_win": .55,
                              "red_post_draft_win": .45, "winner_side": "a"}],
            "series_analysis": "T1 vs Gen.G BO1 系列复盘。\n第1局 BP 后模型胜率：蓝方 55.0%，红方 45.0%。",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            record_post_match_review(path, review)
            details = paper_review_detail(path)
        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["series_analysis"], review["series_analysis"])
        self.assertEqual(details[0]["game_samples"], review["game_samples"])

    def test_current_drawdown_uses_settled_bet_pnl(self):
        report = {"generated_at": "2026-01-01T00:00:00+00:00", "report_date": "2026-01-01",
                  "recommendations": [{"generated_at": "2026-01-01T00:00:00+00:00",
                    "sport": "nba", "event_id": "1", "outcome": "A", "model_probability": .6,
                    "market_probability": .55, "execution_price": .5, "action": "BET",
                    "stake": 100, "probability_eligible": True, "real_money_approved": False,
                    "market_started": False}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            record_report(path, report)
            settle_pending(path, LosingClient())
            self.assertGreater(current_drawdown(path, 1000), 0.10)


if __name__ == "__main__":
    unittest.main()
