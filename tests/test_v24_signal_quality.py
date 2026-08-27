import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prediction_agent.delivery import format_daily_report
from prediction_agent.entities import canonical_live_match_id
from prediction_agent.live_engine import AlertEngine, MarketState, ProbabilityUpdate, market_alert_quality
from prediction_agent.live_runtime import LiveSupervisor
from prediction_agent.providers.live_data import LiveState


class V24SignalQualityTests(unittest.TestCase):
    def test_candidate_is_not_actual_virtual_bet(self):
        text = format_daily_report({"recommendations": [{"action": "BET", "event": "A vs B",
                                                          "reasons": ["market quality"]}]})
        self.assertIn("【今日下注候选】1场｜实际虚拟下注 0场", text)
        self.assertIn("最终状态: SKIPPED", text)

    def test_only_canonical_virtual_bet_counts(self):
        rows = [{"action": "BET", "event": "A", "canonical_virtual_bet_id": "v1"},
                {"action": "BET", "event": "B", "final_status": "SKIPPED"}]
        self.assertIn("【今日下注候选】2场｜实际虚拟下注 1场",
                      format_daily_report({"recommendations": rows}))

    def test_delivery_has_no_real_money_language(self):
        old = os.environ.get("REAL_TRADING_DISABLED")
        os.environ["REAL_TRADING_DISABLED"] = "true"
        try:
            text = format_daily_report({"recommendations": [], "virtual_betting": {"count": 101, "roi": .1}})
            for phrase in ("真实建议", "真实下注", "真实执行"):
                self.assertNotIn(phrase, text)
        finally:
            if old is None:
                os.environ.pop("REAL_TRADING_DISABLED", None)
            else:
                os.environ["REAL_TRADING_DISABLED"] = old

    def test_alias_and_reversed_teams_share_identity(self):
        self.assertEqual(canonical_live_match_id("lol", "Hyperion (Portuguese Team)", "GAL"),
                         canonical_live_match_id("lol", "Galions", "Hyperion"))
        self.assertEqual(canonical_live_match_id("lol", "TLNP", "BFX"),
                         canonical_live_match_id("lol", "BNK FEARX", "TLN Pirates"))

    def test_stable_ids_take_priority(self):
        self.assertEqual(canonical_live_match_id("lol", "A", "B", reconciled_event_id="event-7"),
                         canonical_live_match_id("lol", "X", "Y", reconciled_event_id="event-7"))

    def test_team_tiers_remain_isolated(self):
        main = canonical_live_match_id("lol", "T1", "Gen.G")
        for tier in ("T1 Academy", "T1 Youth", "T1 Challengers"):
            self.assertNotEqual(main, canonical_live_match_id("lol", tier, "Gen.G"))

    def test_completed_and_postponed_never_missing(self):
        now = datetime.now(timezone.utc)
        for status in ("COMPLETED", "POSTPONED"):
            report = {"schedule_coverage": {"lol": {"matches": [{
                "team_a": "T1", "team_b": "BFX", "event_status": status,
                "start_time": (now - timedelta(hours=1)).isoformat()}]}}}
            with tempfile.TemporaryDirectory() as temp:
                supervisor = LiveSupervisor(root=Path(temp))
                self.assertEqual(supervisor._watcher_alerts(report, [], now), [])
                self.assertEqual(supervisor._watcher_alerts(report, [], now + timedelta(minutes=5)), [])

    def test_monitor_state_persists_and_recovery_confirms(self):
        now = datetime.now(timezone.utc)
        report = {"schedule_coverage": {"nba": {"matches": [{
            "team_a": "Lakers", "team_b": "Celtics", "event_status": "SCHEDULED",
            "start_time": (now - timedelta(minutes=30)).isoformat()}]}}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(LiveSupervisor(root=root)._watcher_alerts(report, [], now), [])
            supervisor = LiveSupervisor(root=root)
            self.assertEqual(supervisor._watcher_alerts(report, [], now + timedelta(minutes=5))[0].category,
                             "WATCHER_MISSING")
            live = LiveState("nba", "1", "nba", now, "LIVE", "Celtics", "Lakers")
            self.assertEqual(supervisor._watcher_alerts(report, [live], now + timedelta(minutes=10)), [])
            self.assertEqual(supervisor._watcher_alerts(report, [live], now + timedelta(minutes=15))[0].category,
                             "MONITORING_RECOVERY")
            self.assertEqual(supervisor._watcher_alerts(report, [live], now + timedelta(minutes=20)), [])

    @staticmethod
    def _market(spread=.01, depth=1000, volume=1000):
        now = datetime.now(timezone.utc)
        return MarketState("m", "A", .83, .10, .10 + spread, spread, depth, None, now, True,
                           mid_price=.5, last_price=.5, volume=volume)

    def test_bad_market_downgrades_and_classifies(self):
        market = self._market(spread=.64)
        self.assertEqual(market_alert_quality(market)[0], "UNTRADEABLE")
        state = LiveState("x", "1", "lol", datetime.now(timezone.utc), "LIVE", "T1", "BFX")
        alert = AlertEngine().evaluate(state, ProbabilityUpdate(.47, .47, 0, "test", "test", []), market, None)
        self.assertEqual(alert.severity, "OBSERVE")
        self.assertEqual(alert.details["anomaly_classification"], "MARKET_MICROSTRUCTURE_ANOMALY")
        self.assertLess(alert.alert_score, 40)

    def test_low_liquidity_downgrades(self):
        self.assertEqual(market_alert_quality(self._market(depth=20, volume=10))[0], "LOW_LIQUIDITY")

    def test_skip_summary_is_truncated(self):
        rows = [{"action": "NO_BET", "event": f"match-{i}", "blocker": "ROSTER"} for i in range(16)]
        text = format_daily_report({"recommendations": rows})
        self.assertIn("Top 10 / 16场", text)
        self.assertIn("其余跳过 6 场：ROSTER 6", text)
        self.assertNotIn("match-15", text)


if __name__ == "__main__":
    unittest.main()
