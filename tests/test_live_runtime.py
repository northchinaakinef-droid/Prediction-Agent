from __future__ import annotations

import tempfile
import unittest
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from prediction_agent.live_engine import DynamicProbabilityEngine, match_key
from prediction_agent.live_runtime import LiveSupervisor
from prediction_agent.providers.live_data import LiveState


class LiveRuntimeTests(unittest.TestCase):
    def test_stale_daily_report_is_not_used_for_live_alerts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reports").mkdir()
            (root / "reports" / "daily.json").write_text(json.dumps({
                "report_date": "2020-01-01", "recommendations": [{"sport": "lol"}],
            }), encoding="utf-8")
            supervisor = LiveSupervisor(root=root)
            self.assertEqual(supervisor._report(), {})

    def test_lol_post_draft_probability_does_not_require_in_game_fields(self):
        state = LiveState(
            "draft", "g1", "lol", datetime.now(timezone.utc), "LIVE", "T1", "Gen.G",
            features={"post_draft_probability": .64},
        )
        result = DynamicProbabilityEngine().update(state, .55)
        self.assertEqual(result.method, "LOL_POST_DRAFT_MODEL")
        self.assertAlmostEqual(result.current_probability, .64)

    def test_lol_prematch_alert_is_suppressed_without_probability(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            report = {"schedule_coverage": {"lol": {"matches": [{
                "match_id": "m1", "team_a": "T1", "team_b": "Gen.G", "league": "LCK",
                "start_time": (now + timedelta(minutes=20)).isoformat(),
                "missing_reason": "market not found",
            }]}}, "recommendations": []}
            alerts = supervisor._prematch_alerts(report, now)
        self.assertEqual(len(alerts), 0)

    def test_lol_prematch_alert_includes_probability_details(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            report = {
                "schedule_coverage": {"lol": {"matches": [{
                    "match_id": "m1", "team_a": "T1", "team_b": "Gen.G", "league": "LCK",
                    "start_time": (now + timedelta(minutes=20)).isoformat(),
                }]}},
                "recommendations": [{
                    "sport": "lol", "event": "T1 vs Gen.G", "outcome": "T1",
                    "model_probability": .63, "market_probability": .58,
                    "reasons": ["team strength edge"],
                }],
            }
            alerts = supervisor._prematch_alerts(report, now)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].category, "PREMATCH_ANALYSIS")
        self.assertAlmostEqual(alerts[0].details["blue_win_probability"], .63)
        self.assertAlmostEqual(alerts[0].details["red_win_probability"], .37)

    def test_lol_result_review_alerts_compares_predictions(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reports").mkdir()
            (root / "reports" / "daily.json").write_text(json.dumps({
                "report_date": datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(),
                "recommendations": [{
                    "sport": "lol", "event": "T1 vs Gen.G", "outcome": "Gen.G",
                    "model_probability": .58,
                }],
            }), encoding="utf-8")
            supervisor = LiveSupervisor(root=root)
            state = LiveState(
                "leaguepedia", "g1", "lol", now, "FINISHED", "T1", "Gen.G",
                features={"winner_side": "b", "post_draft_probability": .55},
                finished=True,
            )
            alerts = supervisor._result_review_alerts([state], now)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].category, "POSTMATCH_REVIEW")
        self.assertIn("赛前预测", alerts[0].reasons[0])
        self.assertIn("BP后预测", alerts[0].reasons[1])

    def test_postmatch_review_dedupe_key_is_stable_across_scans(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            first = LiveState(
                "leaguepedia", "g1", "lol", now, "FINISHED", "T1", "Gen.G",
                features={"winner_side": "b", "post_draft_probability": .55},
                finished=True,
            )
            second = LiveState(
                "leaguepedia", "g1", "lol", now + timedelta(seconds=30), "FINISHED",
                "T1", "Gen.G", features={"winner_side": "b", "post_draft_probability": .55},
                finished=True,
            )
            first_alert = supervisor._result_review_alerts([first], now)[0]
            second_alert = supervisor._result_review_alerts(
                [second], now + timedelta(seconds=30)
            )[0]
        self.assertEqual(first_alert.dedupe_key, second_alert.dedupe_key)
        self.assertTrue(first_alert.dedupe_key.endswith(":POSTMATCH_REVIEW"))

    def test_legacy_postmatch_review_keys_are_collapsed_to_stable_key(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            legacy_key = "lol:geng:t1:POSTMATCH_REVIEW:2026-01-01T00:00:00+00:00"
            with supervisor.store.connect() as db:
                db.execute(
                    "INSERT INTO live_alerts(dedupe_key, observed_at, alert_json) VALUES (?, ?, ?)",
                    (legacy_key, now.isoformat(), "{}"),
                )
            supervisor._collapse_legacy_postmatch_dedupe()
            self.assertTrue(supervisor.store.alert_exists("lol:geng:t1:POSTMATCH_REVIEW"))

    def test_nba_start_and_finish_lifecycle_alerts(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            live = LiveState("nba", "1", "nba", now, "LIVE", "Lakers", "Celtics", 2, 0,
                             features={"period": 1, "game_clock_seconds": 700})
            supervisor.engine.process([live], {"nba": []}, {match_key(live): .5})
            start = supervisor._lifecycle_alerts([live], {match_key(live): None})
            self.assertTrue(any(alert.category == "MATCH_START" for alert in start))
            previous = supervisor.store.previous(match_key(live))
            finished = LiveState("nba", "1", "nba", now + timedelta(hours=2), "FINISHED",
                                 "Lakers", "Celtics", 101, 99, features={"period": 4}, finished=True)
            supervisor.engine.process([finished], {"nba": []}, {match_key(finished): .5})
            final = supervisor._lifecycle_alerts([finished], {match_key(finished): previous})
        self.assertTrue(any(alert.category == "MATCH_FINISHED" for alert in final))

    def test_missing_watcher_alerts_and_then_recovers(self):
        now = datetime.now(timezone.utc)
        report = {"schedule_coverage": {"nba": {"matches": [{
            "match_id": "n1", "team_a": "Lakers", "team_b": "Celtics", "event_status": "SCHEDULED",
            "start_time": (now - timedelta(minutes=20)).isoformat(),
        }]}}}
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            missing = supervisor._watcher_alerts(report, [], now)
            self.assertEqual(missing[0].category, "WATCHER_MISSING")
            live = LiveState("nba", "1", "nba", now, "LIVE", "Lakers", "Celtics")
            recovered = supervisor._watcher_alerts(report, [live], now)
        self.assertEqual(recovered[0].category, "MONITORING_RECOVERY")


    def test_draft_analysis_not_emitted_for_finished_match(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            live = LiveState("draft", "g1", "lol", now, "FINISHED", "T1", "Gen.G",
                             features={"champions_a": ["A", "B", "C", "D", "E"],
                                       "champions_b": ["F", "G", "H", "I", "J"]}, finished=True)
            alerts = supervisor._lifecycle_alerts([live], {match_key(live): None})
        self.assertFalse(any(alert.category == "DRAFT_ANALYSIS" for alert in alerts))

    def test_draft_analysis_emitted_once_per_new_draft(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            live = LiveState("draft", "g1", "lol", now, "LIVE", "T1", "Gen.G",
                             features={"champions_a": ["A", "B", "C", "D", "E"],
                                       "champions_b": ["F", "G", "H", "I", "J"]})
            first = supervisor._lifecycle_alerts([live], {match_key(live): None})
            self.assertTrue(any(alert.category == "DRAFT_ANALYSIS" for alert in first))
            previous = {"state_json": json.dumps({"features": {
                "champions_a": ["A", "B", "C", "D", "E"],
                "champions_b": ["F", "G", "H", "I", "J"],
            }})}
            again = supervisor._lifecycle_alerts([live], {match_key(live): previous})
        self.assertFalse(any(alert.category == "DRAFT_ANALYSIS" for alert in again))


if __name__ == "__main__":
    unittest.main()
