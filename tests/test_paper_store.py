import tempfile
from pathlib import Path
import unittest

from prediction_agent.paper_store import record_report, settle_pending, summary


class FakeClient:
    def event(self, _event_id):
        return {"closedTime": "2026-01-02T00:00:00Z", "markets": [{
            "gameStartTime": "2026-01-01T01:00:00Z", "sportsMarketType": "moneyline",
            "outcomes": '["A", "B"]', "outcomePrices": '["1", "0"]',
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


if __name__ == "__main__":
    unittest.main()
