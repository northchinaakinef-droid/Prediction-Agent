from datetime import datetime, timedelta, timezone
import unittest

from prediction_agent.next_model import ModelRow, TimedFeature, fit_anchored_model, walk_forward_evaluate


def make_rows(count: int = 360, *, execution: bool = True) -> list[ModelRow]:
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        decision = base + timedelta(days=index)
        signal = 1.0 if index % 2 == 0 else -1.0
        rows.append(ModelRow(
            event_id=str(index), league="nba", decision_at=decision,
            start_at=decision + timedelta(hours=2), settled_at=decision + timedelta(hours=5),
            market_probability=0.5, outcome=1 if signal > 0 else 0,
            features={"signal": TimedFeature(signal, decision - timedelta(minutes=5), "test-source")},
            yes_ask=0.5 if execution else None, no_ask=0.5 if execution else None,
            spread=0.01 if execution else None, available_size=1000 if execution else None,
        ))
    return rows


class NextModelTests(unittest.TestCase):
    def test_rejects_feature_observed_after_decision(self):
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(ValueError, "look-ahead feature"):
            ModelRow(
                "x", "nba", now, now + timedelta(hours=2), now + timedelta(hours=5), .5, 1,
                {"injury": TimedFeature(1.0, now + timedelta(minutes=1), "source")},
            )

    def test_anchored_model_learns_incremental_signal(self):
        rows = make_rows(100)
        model = fit_anchored_model(rows[:80], ("signal",))
        self.assertGreater(model.predict(rows[80]), 0.8)
        self.assertLess(model.predict(rows[81]), 0.2)

    def test_walk_forward_can_approve_stable_oos_edge(self):
        report = walk_forward_evaluate(make_rows(), ("signal",), initial_train=120, test_size=50)
        self.assertLess(report.model_brier, report.market_brier)
        self.assertGreaterEqual(report.predictions, 200)
        self.assertTrue(report.approved_for_paper_trading)
        self.assertGreater(report.roi, 0)

    def test_walk_forward_rejects_missing_execution_history(self):
        report = walk_forward_evaluate(make_rows(execution=False), ("signal",), initial_train=120, test_size=50)
        self.assertFalse(report.approved_for_paper_trading)
        self.assertIn("less than 80% execution-price coverage", report.rejection_reasons)


if __name__ == "__main__":
    unittest.main()
