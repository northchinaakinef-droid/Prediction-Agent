from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from prediction_agent.cs2_model import Cs2Model, save_cs2
from prediction_agent.lol_meta_model import LolMetaModel, save_lol_meta
from prediction_agent.nba_model import NbaModel, save_nba
from prediction_agent.lol_model import EloModel, save_model
from prediction_agent.narrative import build_post_match_summary, build_pre_match_summary
from prediction_agent.next_model import (
    AnalystPick, InjuryRecord, analyst_consensus_diff, injury_impact_diff,
)
from prediction_agent.paper_store import (
    paper_review, record_closing_line, record_report, settle_pending, summary,
)
from prediction_agent.sports_daily import flag_rows


class FakeClient:
    def event(self, _event_id):
        return {"closedTime": "2026-01-02T00:00:00Z", "markets": [{
            "gameStartTime": "2026-01-01T01:00:00Z", "sportsMarketType": "moneyline",
            "outcomes": '["A", "B"]', "outcomePrices": '["1", "0"]',
        }]}


def make_report(rows):
    return {"generated_at": "2026-01-01T00:00:00+00:00", "report_date": "2026-01-01",
            "recommendations": rows}


class TimedFeatureTests(unittest.TestCase):
    def test_injury_record_lookahead_is_rejected(self):
        decision_at = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        future = InjuryRecord("Star", "OUT", datetime(2026, 1, 1, 13, tzinfo=timezone.utc), "injury")
        with self.assertRaisesRegex(ValueError, "look-ahead injury"):
            injury_impact_diff([future], [], decision_at)

    def test_injury_impact_diff_uses_status_weights(self):
        decision_at = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        out = InjuryRecord("Star", "OUT", decision_at, "injury")
        available = InjuryRecord("Bench", "AVAILABLE", decision_at, "injury")
        self.assertGreater(injury_impact_diff([out], [available], decision_at), 0)

    def test_analyst_pick_lookahead_is_rejected(self):
        decision_at = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
        future = AnalystPick("Analyst", 0.7, datetime(2026, 1, 1, 13, tzinfo=timezone.utc), "feed")
        with self.assertRaisesRegex(ValueError, "look-ahead analyst"):
            analyst_consensus_diff([future], decision_at, 0.55)


class NarrativeTests(unittest.TestCase):
    def test_narrative_is_read_only(self):
        row = {
            "sport": "nba", "event": "A vs B", "outcome": "A",
            "model_probability": 0.6, "market_probability": 0.55,
            "edge": 0.03, "reasons": ["edge"], "stake": 12.0, "action": "BET",
        }
        before = deepcopy(row)
        text = build_pre_match_summary(row)
        self.assertIsInstance(text, str)
        self.assertEqual(row, before)

    def test_post_match_narrative_is_read_only(self):
        row = {
            "event": "A vs B", "actual_winner": "B", "predicted_winner": "A",
            "prediction_correct": False, "model_probability": 0.8,
        }
        before = deepcopy(row)
        text = build_post_match_summary(row)
        self.assertIsInstance(text, str)
        self.assertEqual(row, before)


class FlagRowTests(unittest.TestCase):
    def test_large_disagreement_lands_in_flagged_bucket(self):
        rows = [{"raw_edge": 0.12}, {"raw_edge": 0.02}]
        flagged = flag_rows(rows, 0.10)
        self.assertTrue(rows[0]["flagged"])
        self.assertFalse(rows[1]["flagged"])
        self.assertEqual(flagged, [rows[0]])


class PaperStoreRoadmapTests(unittest.TestCase):
    def test_clv_is_null_until_closing_line_recorded(self):
        rows = [{"generated_at": "2026-01-01T00:00:00+00:00", "sport": "nba", "event_id": "1",
                 "event": "A vs B", "outcome": "A", "model_probability": 0.6,
                 "market_probability": 0.55, "execution_price": 0.5, "action": "BET",
                 "stake": 100, "probability_eligible": True, "real_money_approved": False,
                 "market_started": False}]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            record_report(path, make_report(rows))
            settle_pending(path, FakeClient())
            self.assertIsNone(summary(path)["by_sport"]["nba"].get("mean_clv"))
            record_closing_line(path, "nba", "1", 0.52)
            self.assertAlmostEqual(summary(path)["by_sport"]["nba"]["mean_clv"], 0.02)

    def test_settle_pending_populates_post_match_reviews_and_review_sorts_biggest_miss_first(self):
        rows = [
            {"generated_at": "2026-01-01T00:00:00+00:00", "sport": "nba", "event_id": "1",
             "event": "A vs B", "outcome": "A", "model_probability": 0.9,
             "market_probability": 0.55, "execution_price": 0.5, "action": "BET",
             "stake": 100, "probability_eligible": True, "real_money_approved": False,
             "market_started": False},
            {"generated_at": "2026-01-01T00:00:00+00:00", "sport": "nba", "event_id": "2",
             "event": "C vs D", "outcome": "B", "model_probability": 0.6,
             "market_probability": 0.45, "execution_price": 0.5, "action": "BET",
             "stake": 100, "probability_eligible": True, "real_money_approved": False,
             "market_started": False},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paper.db"
            record_report(path, make_report(rows))
            settle_pending(path, FakeClient())
            reviews = paper_review(path)
            self.assertEqual(len(reviews), 2)
            self.assertEqual(reviews[0]["predicted_winner"], "B")
            self.assertGreater(reviews[0]["miss"], reviews[1]["miss"])


class SaveGuardTests(unittest.TestCase):
    def _valid_nba_eval(self):
        return {"approved_for_real_money": False, "validation": {"samples": 1}, "retrospective_test": {"samples": 1}}

    def _valid_lol_eval(self):
        return {"approved_for_real_money": False, "validation": {"samples": 1}}

    def _valid_meta_eval(self):
        return {"approved_for_real_money": False, "validation": {"samples": 1}, "final_test": {"samples": 1}}
    def test_save_nba_requires_complete_evaluation(self):
        model = NbaModel({}, {}, {}, "2026-01-01", 1)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "complete evaluation"):
                save_nba(model, {"approved_for_real_money": False}, Path(directory) / "model.json")

    def test_save_nba_accepts_complete_evaluation(self):
        model = NbaModel({}, {}, {}, "2026-01-01", 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_nba(model, self._valid_nba_eval(), path)
            self.assertTrue(path.exists())

    def test_save_model_requires_complete_evaluation(self):
        model = EloModel({}, {}, "2026-01-01", 1)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "complete evaluation"):
                save_model(model, Path(directory) / "model.json", {"approved_for_real_money": False})

    def test_save_model_accepts_complete_evaluation(self):
        model = EloModel({}, {}, "2026-01-01", 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(model, path, self._valid_lol_eval())
            self.assertTrue(path.exists())

    def test_save_cs2_requires_complete_evaluation(self):
        model = Cs2Model({}, {}, {}, {}, {}, {}, "2026-01-01", 0)
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "complete evaluation"):
                save_cs2(model, {"approved_for_real_money": False}, Path(directory) / "model.json")

    def test_save_cs2_accepts_complete_evaluation(self):
        model = Cs2Model({}, {}, {}, {}, {}, {}, "2026-01-01", 0)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_cs2(model, self._valid_meta_eval(), path)
            self.assertTrue(path.exists())

    def test_save_lol_meta_requires_complete_evaluation(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "complete evaluation"):
                save_lol_meta(LolMetaModel(), {"approved_for_real_money": False}, Path(directory) / "model.json")

    def test_save_lol_meta_accepts_complete_evaluation(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_lol_meta(LolMetaModel(), self._valid_meta_eval(), path)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
