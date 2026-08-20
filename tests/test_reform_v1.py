import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prediction_agent.betting_gate import can_place_real_bet, should_place_virtual_bet
from prediction_agent.delivery import format_daily_report, format_live_alert
from prediction_agent.live_runtime import LiveSupervisor
from prediction_agent.paper_store import (
    calc_roi,
    count_settled_virtual_bets,
    count_virtual_bets,
    record_virtual_bet,
    virtual_account_balance,
)


class BettingGateTests(unittest.TestCase):
    def test_virtual_gate_requires_all_four_conditions(self):
        match = {
            "lineup_status": "完整",
            "market_mapping_status": "MATCHED",
            "expected_value": 0.08,
            "direction_match": True,
            "market_started": False,
        }
        ok, reason = should_place_virtual_bet(match)
        self.assertTrue(ok, reason)

        match["lineup_status"] = "未知"
        self.assertFalse(should_place_virtual_bet(match)[0])

        match["lineup_status"] = "完整"
        match["expected_value"] = 0.03
        self.assertFalse(should_place_virtual_bet(match)[0])

    def test_real_advice_shows_virtual_progress_before_100_settled_bets(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "paper.db"
            match = {
                "real_money_approved": False,
                "lineup_status": "完整",
                "market_mapping_status": "MATCHED",
                "expected_value": 0.08,
                "direction_match": True,
                "market_started": False,
            }
            ok, reason = can_place_real_bet(match, db)
            self.assertFalse(ok)
            self.assertIn("虚拟第0场/100场", reason)
            self.assertIn("距真实建议还差100场", reason)

    def test_real_advice_upgrades_at_100_settled_bets_with_nonnegative_roi(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "paper.db"
            for index in range(100):
                record_virtual_bet(db, {
                    "sport": "lol", "event_id": f"e{index}", "event": f"A vs B {index}",
                    "generated_at": f"2026-08-19T00:{index // 60:02d}:{index % 60:02d}+00:00",
                    "bet_side": "A", "model_prob": .6, "market_odds": 1.8,
                    "stake_virtual": 10.0, "result": "A", "pnl_virtual": 0.0,
                })
            match = {
                "real_money_approved": False, "lineup_status": "完整",
                "market_mapping_status": "MATCHED", "expected_value": .08,
                "direction_match": True, "market_started": False,
            }
            ok, reason = can_place_real_bet(match, db)
        self.assertTrue(ok, reason)
        self.assertIn("升级为真实建议", reason)

    def test_real_advice_stays_virtual_when_100_bet_roi_is_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "paper.db"
            for index in range(100):
                record_virtual_bet(db, {
                    "sport": "nba", "event_id": f"e{index}", "event": f"A vs B {index}",
                    "generated_at": f"2026-08-19T01:{index // 60:02d}:{index % 60:02d}+00:00",
                    "bet_side": "A", "model_prob": .6, "market_odds": 1.8,
                    "stake_virtual": 10.0, "result": "B", "pnl_virtual": -10.0,
                })
            ok, reason = can_place_real_bet({}, db)
        self.assertFalse(ok)
        self.assertIn("虚拟ROI=-100.0%", reason)


class PaperStoreVirtualTests(unittest.TestCase):
    def test_record_and_calc_virtual_roi(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "paper.db"
            record_virtual_bet(db, {
                "sport": "lol",
                "event_id": "e1",
                "event": "A vs B",
                "generated_at": "2026-08-19T00:00:00+00:00",
                "bet_side": "A",
                "model_prob": 0.6,
                "market_odds": 1.8,
                "stake_virtual": 100.0,
                "result": "A",
                "pnl_virtual": 10.0,
            })
            self.assertEqual(count_virtual_bets(db), 1)
            self.assertAlmostEqual(calc_roi(db, bet_type="virtual"), 0.1)
            self.assertAlmostEqual(virtual_account_balance(db, 1000.0), 1010.0)

    def test_settlement_table_counts_and_prices_completed_virtual_bet(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "paper.db"
            record_virtual_bet(db, {
                "sport": "lol", "event_id": "e1", "event": "A vs B",
                "generated_at": "2026-08-19T00:00:00+00:00", "bet_side": "A",
                "model_prob": .6, "market_odds": 2.0, "stake_virtual": 10.0,
            })
            connection = sqlite3.connect(db)
            try:
                connection.execute(
                    "INSERT INTO settlements VALUES (?, ?, ?, ?, ?, ?)",
                    ("lol", "e1", "A", "2026-08-19T02:00:00+00:00", "test", "{}"),
                )
                connection.commit()
            finally:
                connection.close()
            self.assertEqual(count_settled_virtual_bets(db), 1)
            self.assertGreater(calc_roi(db, bet_type="virtual"), 0)

    def test_postmatch_virtual_bet_is_excluded_from_roi_and_unlock_count(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "paper.db"
            record_virtual_bet(db, {
                "sport": "lol", "event_id": "prematch-placeholder", "event": "A vs B",
                "generated_at": "2026-08-19T00:00:00+00:00", "bet_side": "A",
                "model_prob": .6, "market_odds": 2.0, "stake_virtual": 10.0,
                "prediction_stage": "POSTMATCH",
            })
            connection = sqlite3.connect(db)
            try:
                stored_stage = connection.execute(
                    "SELECT prediction_stage FROM virtual_bets"
                ).fetchone()[0]
                self.assertEqual(stored_stage, "PREMATCH")
                connection.execute("DELETE FROM virtual_bets")
                connection.execute(
                    """INSERT INTO virtual_bets
                       (match_id, sport, event_id, event, generated_at, bet_side,
                        model_prob, market_odds, stake_virtual, result, pnl_virtual,
                        prediction_stage, payload_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    ("postmatch-1", "lol", "e1", "A vs B",
                     "2026-08-19T02:00:00+00:00", "A", .75, 2.0, 10.0,
                     "A", 10.0, "POSTMATCH", "{}"),
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        """INSERT INTO virtual_bets
                           (match_id, sport, event_id, generated_at, bet_side, model_prob,
                            market_odds, stake_virtual, prediction_stage, payload_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        ("invalid-1", "lol", "e2", "2026-08-19T03:00:00+00:00",
                         "A", .75, 2.0, 10.0, "INVALID", "{}"),
                    )
                connection.commit()
            finally:
                connection.close()
            self.assertEqual(count_settled_virtual_bets(db), 0)
            self.assertIsNone(calc_roi(db, bet_type="virtual"))


class DeliveryReformTests(unittest.TestCase):
    def test_prematch_reference_format(self):
        alert = {
            "category": "PREMATCH_REFERENCE",
            "sport": "lol",
            "title": "T1 vs Gen.G",
            "severity": "IMPORTANT",
            "alert_score": 35,
            "details": {
                "team_a": "T1", "team_b": "Gen.G",
                "blue_win_probability": .62, "red_win_probability": .38,
                "ev": .08,
            },
        }
        text = format_live_alert(alert)
        self.assertIn("赛前参考", text)
        self.assertIn("禁止下注", text)

    def test_prematch_contains_lineup_and_bet_status(self):
        alert = {
            "category": "PREMATCH_ANALYSIS",
            "sport": "lol",
            "title": "T1 vs Gen.G",
            "severity": "IMPORTANT",
            "alert_score": 55,
            "details": {
                "team_a": "T1", "team_b": "Gen.G",
                "blue_win_probability": .62, "red_win_probability": .38,
                "blue_market_probability": .58, "red_market_probability": .42,
                "ev": .08,
                "lineup_a": ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"],
                "lineup_b": ["Kiin", "Canyon", "Chovy", "Peyz", "Lehends"],
                "recent_form_a": {"wins": 7, "losses": 3, "last_n": 10},
                "recent_form_b": {"wins": 6, "losses": 4, "last_n": 10},
                "patch_meta_heroes": ["A", "B", "C", "D", "E"],
                "meta_coverage_a": 80.0,
                "meta_coverage_b": 60.0,
                "bet_status": "虚拟下注",
                "sample_a": 50,
                "sample_b": 48,
            },
        }
        text = format_live_alert(alert)
        self.assertIn("【阵容信息】", text)
        self.assertIn("上路：Zeus", text)
        self.assertIn("【近期状态】", text)
        self.assertIn("【版本关键英雄覆盖】", text)
        self.assertIn("【下注状态】虚拟下注", text)

    def test_daily_report_marks_virtual_bet_status(self):
        report = {
            "report_date": "2026-08-19",
            "bankroll_usdc": 10000,
            "recommendations": [{
                "sport": "lol", "event": "T1 vs Gen.G", "event_id": "e1",
                "outcome": "T1", "action": "BET", "stake": 25.0,
                "model_probability": .63, "market_probability": .58,
                "expected_value": .08, "lineup_status": "完整",
                "bet_status": "虚拟下注",
            }],
            "paper_daily": {},
            "today_scheduled_matches": 3,
            "risk_status": {"current_drawdown": 0.0, "drawdown_level": "normal"},
            "virtual_betting": {"count": 12, "roi": 0.03, "balance": 10030.0},
        }
        text = format_daily_report(report)
        self.assertIn("【虚拟下注】", text)
        self.assertIn("虚拟第12场/100场，距真实建议还差88场", text)
        self.assertIn("【今日赛程】实际场次 3 场", text)


class LiveRuntimeReformTests(unittest.TestCase):
    def test_not_in_schedule_emits_reference_alert(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            report = {
                "schedule_coverage": {"lol": {"matches": []}},
                "recommendations": [{
                    "sport": "lol", "event": "LoL: T1 vs Gen.G", "outcome": "T1",
                    "model_probability": .63, "market_probability": .58,
                    "expected_value": .08, "market_mapping_status": "NOT_IN_SCHEDULE",
                    "lineup_status": "完整", "scheduled_start": (now + timedelta(minutes=20)).isoformat(),
                }],
            }
            alerts = supervisor._prematch_alerts(report, now)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].category, "PREMATCH_REFERENCE")

    def test_missing_lineup_emits_lineup_missing_alert(self):
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            supervisor = LiveSupervisor(root=Path(temp))
            report = {
                "schedule_coverage": {"lol": {"matches": [{
                    "match_id": "m1", "team_a": "T1", "team_b": "Gen.G",
                    "start_time": (now + timedelta(minutes=20)).isoformat(),
                }]}},
                "recommendations": [{
                    "sport": "lol", "event": "T1 vs Gen.G", "outcome": "T1",
                    "model_probability": .63, "lineup_status": "未知",
                }],
            }
            alerts = supervisor._prematch_alerts(report, now)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].category, "LINEUP_MISSING")


if __name__ == "__main__":
    unittest.main()
