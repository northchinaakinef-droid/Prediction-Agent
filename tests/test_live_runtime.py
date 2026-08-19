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


def _nba_boxscore():
    return {
        "game_id": "0022400001",
        "game_status": 3,
        "game_status_text": "Final",
        "period": 4,
        "game_time_utc": "2025-06-01T00:00:00Z",
        "duration": "PT02H10M",
        "duration_seconds": 7800.0,
        "home_team": {
            "team_id": 1610612747, "team_name": "Lakers", "team_city": "Los Angeles",
            "team_tricode": "LAL", "score": 110,
            "periods": [{"period": 1, "periodType": "REGULAR", "score": 28},
                        {"period": 2, "periodType": "REGULAR", "score": 30}],
            "statistics": {
                "fieldGoalsMade": 40, "fieldGoalsAttempted": 88, "threePointersMade": 12,
                "freeThrowsMade": 18, "freeThrowsAttempted": 22, "turnoversTotal": 10,
                "turnovers": 10, "reboundsOffensive": 11, "reboundsDefensive": 32,
                "reboundsTotal": 43, "pointsInThePaint": 46, "pointsFastBreak": 12,
                "pointsSecondChance": 10, "pointsFromTurnovers": 14, "benchPoints": 32,
                "assists": 24, "steals": 8, "blocks": 5, "leadChanges": 4,
                "timesTied": 3, "biggestLead": 12, "minutes": "PT240M",
            },
            "players": [{
                "person_id": 1, "name": "LeBron James", "jersey_num": "23", "position": "F",
                "starter": True, "played": True, "minutes": "PT36M00.00S", "minutes_seconds": 2160.0,
                "points": 28, "rebounds_total": 8, "assists": 7, "plus_minus_points": 9,
            }],
        },
        "away_team": {
            "team_id": 1610612738, "team_name": "Celtics", "team_city": "Boston",
            "team_tricode": "BOS", "score": 104,
            "periods": [{"period": 1, "periodType": "REGULAR", "score": 25},
                        {"period": 2, "periodType": "REGULAR", "score": 27}],
            "statistics": {
                "fieldGoalsMade": 38, "fieldGoalsAttempted": 90, "threePointersMade": 10,
                "freeThrowsMade": 18, "freeThrowsAttempted": 24, "turnoversTotal": 14,
                "turnovers": 14, "reboundsOffensive": 12, "reboundsDefensive": 30,
                "reboundsTotal": 42, "pointsInThePaint": 42, "pointsFastBreak": 10,
                "pointsSecondChance": 12, "pointsFromTurnovers": 11, "benchPoints": 25,
                "assists": 22, "steals": 7, "blocks": 4, "leadChanges": 4,
                "timesTied": 3, "biggestLead": 8, "minutes": "PT240M",
            },
            "players": [],
        },
    }


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

    def test_yesterdays_report_can_back_overnight_postmatch_review(self):
        now = datetime.now(timezone.utc)
        zone = ZoneInfo("Asia/Singapore")
        yesterday = (datetime.now(zone).date() - timedelta(days=1)).isoformat()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reports").mkdir()
            (root / "reports" / "daily.json").write_text(json.dumps({
                "report_date": yesterday,
                "generated_at": now.isoformat(),
                "recommendations": [{
                    "sport": "lol", "event": "FlyQuest vs Cloud9", "outcome": "Cloud9",
                    "model_probability": .647,
                }],
            }), encoding="utf-8")
            supervisor = LiveSupervisor(root=root)
            report = supervisor._report()
        self.assertEqual(report["report_date"], yesterday)
        self.assertEqual(report["recommendations"][0]["outcome"], "Cloud9")

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
                    "lineup_status": "完整",
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

    def test_lol_bp_samples_are_recorded_per_small_game(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            state = LiveState("leaguepedia", "series-1", "lol", now, "FINISHED", "T1", "Gen.G",
                              features={"winner_side": "a"}, finished=True)
            game_details = {
                match_key(state): [
                    {"game_id": "g2", "game_time": "2026-01-01T01:00:00", "winner_side": "b",
                     "champions_a": ["A", "B", "C", "D", "E"], "champions_b": ["F", "G", "H", "I", "J"]},
                    {"game_id": "g1", "game_time": "2026-01-01T00:00:00", "winner_side": "a",
                     "champions_a": ["A", "B", "C", "D", "E"], "champions_b": ["F", "G", "H", "I", "J"]},
                ]
            }
            class FakeMeta:
                latest_team_rosters = {
                    "T1": ("p1", "p2", "p3", "p4", "p5"),
                    "Gen.G": ("p6", "p7", "p8", "p9", "p10"),
                }
                def predict_post_draft(self, game):
                    return .61
            samples = supervisor._compute_lol_bp_samples(FakeMeta(), "25.10", state, game_details)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0]["game_id"], "g1")
        self.assertEqual(samples[0]["game_index"], 1)
        self.assertAlmostEqual(samples[0]["blue_post_draft_win"], .61)
        self.assertAlmostEqual(samples[0]["red_post_draft_win"], .39)

    def test_series_review_analysis_is_multi_game_and_structured(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            state = LiveState("leaguepedia", "series-1", "lol", now, "FINISHED", "T1", "Gen.G",
                              features={"winner_side": "a"}, finished=True)
            games = [
                {"game_index": 1, "blue_champions": ["A", "B", "C", "D", "E"],
                 "red_champions": ["F", "G", "H", "I", "J"],
                 "blue_post_draft_win": .55, "red_post_draft_win": .45, "winner_side": "a"},
                {"game_index": 2, "blue_champions": ["K", "L", "M", "N", "O"],
                 "red_champions": ["P", "Q", "R", "S", "T"],
                 "blue_post_draft_win": .48, "red_post_draft_win": .52, "winner_side": "b"},
                {"game_index": 3, "blue_champions": ["U", "V", "W", "X", "Y"],
                 "red_champions": ["Z", "AA", "BB", "CC", "DD"],
                 "blue_post_draft_win": .60, "red_post_draft_win": .40, "winner_side": "a"},
            ]
            text = supervisor._build_series_review_analysis(state, games, {})
        self.assertIn("BO3", text)
        self.assertIn("第1局 BP", text)
        self.assertIn("第2局 BP", text)
        self.assertIn("第3局 BP", text)
        self.assertIn("BP 后模型胜率", text)
        self.assertGreater(len(text.splitlines()), 6)

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


    def test_nba_box_samples_include_analyst_metrics(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            state = LiveState("nba_official", "0022400001", "nba", now, "FINISHED",
                              "Celtics", "Lakers", 104, 110,
                              features={"period": 4}, finished=True)
            game_details = {match_key(state): [_nba_boxscore()]}
            samples = supervisor._compute_nba_box_samples(state, game_details)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["home_team"], "Lakers")
        self.assertEqual(samples[0]["away_team"], "Celtics")
        self.assertIn("home_four_factors", samples[0]["metrics"])
        self.assertIn("pace", samples[0]["metrics"])
        self.assertTrue(samples[0]["decisive_factors"])

    def test_nba_review_analysis_is_analyst_grade(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            state = LiveState("nba_official", "0022400001", "nba", now, "FINISHED",
                              "Celtics", "Lakers", 104, 110,
                              features={"period": 4}, finished=True)
            samples = supervisor._compute_nba_box_samples(
                state, {match_key(state): [_nba_boxscore()]}
            )
            text = supervisor._build_nba_review_analysis(state, samples, {})
        self.assertIn("NBA 赛后复盘", text)
        self.assertIn("【四要素效率】", text)
        self.assertIn("【节奏与效率】", text)
        self.assertIn("【比赛走势】", text)
        self.assertIn("【球员表现】", text)
        self.assertIn("【战术执行与模型解读】", text)
        self.assertGreater(len(text.splitlines()), 10)

    def test_nba_result_review_alert_persists_box_samples(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reports").mkdir()
            (root / "reports" / "daily.json").write_text(json.dumps({
                "report_date": datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(),
                "recommendations": [{
                    "sport": "nba", "event": "Boston Celtics vs Los Angeles Lakers", "outcome": "Lakers",
                    "model_probability": .60,
                }],
            }), encoding="utf-8")
            supervisor = LiveSupervisor(root=root)
            state = LiveState("nba_official", "0022400001", "nba", now, "FINISHED",
                              "Celtics", "Lakers", 104, 110,
                              features={"period": 4}, finished=True)
            alerts = supervisor._result_review_alerts(
                [state], now, {}, {}, {match_key(state): [_nba_boxscore()]}
            )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].category, "POSTMATCH_REVIEW")
        self.assertEqual(alerts[0].details["game_samples"][0]["home_team"], "Lakers")
        self.assertIn("【四要素效率】", alerts[0].details["series_analysis"])
        self.assertTrue(alerts[0].details["decisive_factors"])

    def test_prematch_dedupe_key_is_stable_across_schedule_match_ids(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "reports").mkdir()
            (root / "reports" / "daily.json").write_text(json.dumps({
                "report_date": datetime.now(ZoneInfo("Asia/Singapore")).date().isoformat(),
                "recommendations": [{
                    "sport": "lol", "event": "T1 vs Gen.G", "outcome": "T1",
                    "model_probability": .63,
                    "lineup_status": "完整",
                }],
                "schedule_coverage": {"lol": {"matches": [
                    {"match_id": "m1", "team_a": "T1", "team_b": "Gen.G",
                     "start_time": (now + timedelta(minutes=20)).isoformat()},
                    {"match_id": "m2", "team_a": "T1", "team_b": "Gen.G",
                     "start_time": (now + timedelta(minutes=20)).isoformat()},
                ]}},
            }), encoding="utf-8")
            supervisor = LiveSupervisor(root=root)
            alerts = supervisor._prematch_alerts(json.loads((root / "reports" / "daily.json").read_text(encoding="utf-8")), now)
        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0].dedupe_key, alerts[1].dedupe_key)
        self.assertNotIn("m1", alerts[0].dedupe_key)
        self.assertNotIn("m2", alerts[0].dedupe_key)
        self.assertTrue(alerts[0].dedupe_key.endswith(":PREMATCH_ANALYSIS"))

    def test_legacy_prematch_keys_are_collapsed_to_stable_key(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            legacy_key = "lol:legacy-match-id-123:PREMATCH_ANALYSIS"
            alert_json = json.dumps({"match_key": "lol:geng:t1"})
            with supervisor.store.connect() as db:
                db.execute(
                    "INSERT INTO live_alerts(dedupe_key, observed_at, alert_json) VALUES (?, ?, ?)",
                    (legacy_key, now.isoformat(), alert_json),
                )
            supervisor._collapse_legacy_prematch_dedupe()
            self.assertTrue(supervisor.store.alert_exists("lol:geng:t1:PREMATCH_ANALYSIS"))

if __name__ == "__main__":
    unittest.main()
