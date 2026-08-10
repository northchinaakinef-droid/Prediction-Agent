from datetime import datetime, timedelta, timezone
import unittest

from prediction_agent.anomaly import detect_market_anomalies
from prediction_agent.backtest import BacktestRow, run_backtest
from prediction_agent.models import MarketSnapshot
from prediction_agent.risk import normalize_two_way, recommend


class CoreTests(unittest.TestCase):
    def test_remove_vig(self):
        a, b = normalize_two_way(0.55, 0.55)
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(b, 0.5)

    def test_no_bet_when_edge_is_weak(self):
        result = recommend(event_id="x", outcome="home", model_probability=0.51,
                           decimal_odds=2.0, bankroll=1000, confidence=0.7)
        self.assertEqual(result.action, "NO_BET")
        self.assertEqual(result.stake, 0)

    def test_stake_is_capped(self):
        result = recommend(event_id="x", outcome="home", model_probability=0.8,
                           decimal_odds=2.0, bankroll=1000, confidence=0.9)
        self.assertEqual(result.action, "BET")
        self.assertLessEqual(result.stake, 10)

    def test_anomaly_detection(self):
        now = datetime.now(timezone.utc)
        history = [MarketSnapshot("x", "yes", now + timedelta(minutes=i), .49, .51, 100, 100) for i in range(5)]
        history.append(MarketSnapshot("x", "yes", now + timedelta(minutes=6), .79, .90, 1000, 1))
        kinds = {a.kind for a in detect_market_anomalies(history)}
        self.assertTrue({"price_jump", "liquidity_drop", "book_imbalance"}.issubset(kinds))

    def test_backtest_rejects_lookahead(self):
        now = datetime.now(timezone.utc)
        row = BacktestRow("x", now, now - timedelta(hours=1), .7, 2.0, True, .9)
        with self.assertRaisesRegex(ValueError, "look-ahead"):
            run_backtest([row])


if __name__ == "__main__":
    unittest.main()
