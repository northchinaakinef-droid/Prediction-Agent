from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from prediction_agent.feedback import FeedbackStore, PredictionSnapshot


def snapshot(match, sport, key, baseline, final):
    return PredictionSnapshot(match, "2026-08-25T00:00:00+00:00", "q", "p", "m",
        baseline, final-baseline, final, .5, final-.5, {"sport": sport}, [], [], "NO_BET", 0,
        llm_provider="fake", canonical_sample_key=key, quant_action="NO_BET",
        shadow_final_action="BET", quant_ev=.02, shadow_final_ev=.10,
        quant_stake=0, shadow_final_stake=5)


class V23ShadowAttributionTests(unittest.TestCase):
    def test_aggregates_probability_metrics_overall_and_by_sport(self):
        with TemporaryDirectory() as tmp:
            store = FeedbackStore(Path(tmp) / "f.db")
            for item, outcome in ((snapshot("1", "cs2", "k1", .6, .65), 1),
                                  (snapshot("2", "lol", "k2", .4, .35), 0)):
                store.attribute(store.save_snapshot(item), outcome)
            result = store.shadow_attribution_summary()
            self.assertEqual(result["overall"]["samples"], 2)
            self.assertEqual(result["by_sport"]["cs2"]["samples"], 1)
            self.assertIsNotNone(result["overall"]["brier_delta"])

    def test_canonical_key_is_not_double_weighted(self):
        with TemporaryDirectory() as tmp:
            store = FeedbackStore(Path(tmp) / "f.db")
            for match in ("old", "new"):
                item = snapshot(match, "nba", "same", .5, .55)
                store.attribute(store.save_snapshot(item), 1)
            self.assertEqual(store.shadow_attribution_summary()["overall"]["samples"], 1)


if __name__ == "__main__": unittest.main()
