from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from prediction_agent.delivery import format_live_alert
from prediction_agent.live_engine import (
    AlertEngine, DynamicProbabilityEngine, LiveAlert, LiveStore, MarketState,
)
from prediction_agent.providers.live_data import LiveState


class LiveEngineTests(unittest.TestCase):
    def test_lol_probability_updates_from_real_fields_and_remains_research_only(self):
        state = LiveState(
            "test", "1", "lol", datetime.now(timezone.utc), "LIVE", "T1", "Dplus KIA",
            features={"gold_a": 30000, "gold_b": 27000, "kills_a": 10, "kills_b": 6,
                      "towers_a": 5, "towers_b": 2, "dragons_a": 3, "dragons_b": 1,
                      "barons_a": 1, "barons_b": 0, "inhibitors_a": 0, "inhibitors_b": 0},
        )
        result = DynamicProbabilityEngine().update(state, .55)
        self.assertGreater(result.current_probability, .55)
        self.assertEqual(result.calibration_status, "UNVALIDATED_RESEARCH_ONLY")

    def test_probability_is_unavailable_without_independent_prior(self):
        state = LiveState("test", "1", "nba", datetime.now(timezone.utc), "LIVE", "A", "B", 10, 9)
        result = DynamicProbabilityEngine().update(state, None)
        self.assertIsNone(result.current_probability)
        self.assertEqual(result.method, "UNAVAILABLE")

    def test_alert_score_detects_model_market_divergence_and_formats_message(self):
        now = datetime.now(timezone.utc)
        state = LiveState(
            "test", "1", "nba", now, "LIVE", "A", "B", 90, 80,
            features={"period": 4, "game_clock_seconds": 120},
        )
        probability = DynamicProbabilityEngine().update(state, .5)
        market = MarketState("m", "A", .45, .44, .46, .02, 1000, 0, now, True)
        alert = AlertEngine().evaluate(state, probability, market, None)
        self.assertIsNotNone(alert)
        self.assertIn("Alert Score", format_live_alert(alert))

    def test_live_store_persists_snapshots_and_alert_dedupe(self):
        now = datetime.now(timezone.utc)
        state = LiveState("test", "1", "nba", now, "LIVE", "A", "B", 1, 0,
                          features={"period": 1, "game_clock_seconds": 600})
        probability = DynamicProbabilityEngine().update(state, .5)
        market = MarketState(None, None, None, None, None, None, None, None, now, False)
        alert = LiveAlert("nba:a:b", "nba", "OBSERVE", 40, "PROBABILITY_CHANGE", "A vs B", "changed", [], now,
                          "nba:a:b:probability")
        with TemporaryDirectory() as temp:
            store = LiveStore(Path(temp) / "live.db")
            store.save("nba:a:b", state, probability, market)
            self.assertIsNotNone(store.previous("nba:a:b"))
            store.save_alert(alert)
            self.assertTrue(store.alert_recent(alert.dedupe_key, 600))


if __name__ == "__main__":
    unittest.main()
