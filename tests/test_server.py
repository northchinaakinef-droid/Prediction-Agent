from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch


def load_server_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "server.py"
    spec = importlib.util.spec_from_file_location("prediction_server_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class ServerDeliveryTests(unittest.TestCase):
    def test_app_delivery_uses_post_for_daily_and_text_for_alerts(self):
        server = load_server_module()
        client = MagicMock()
        with patch.dict(os.environ, {
            "FEISHU_APP_ID": "app", "FEISHU_APP_SECRET": "secret", "FEISHU_RECEIVE_ID": "chat",
        }, clear=True), patch.object(server, "FeishuAppClient", return_value=client):
            server._send({"report_date": "2026-08-14", "recommendations": []})
            server._send_message("实时告警")
        client.send_post.assert_called_once()
        client.send_text.assert_called_once_with("实时告警")

    def test_live_push_filters_to_valuable_categories_only(self):
        server = load_server_module()
        with patch.object(server, "_send_message") as send:
            server._send_valuable_alert({"category": "WATCHER_MISSING"})
            send.assert_not_called()
        prematch = {
            "category": "PREMATCH_ANALYSIS", "severity": "IMPORTANT", "alert_score": 55,
            "sport": "lol", "title": "BLG vs WE", "summary": "赛前方向：BLG",
            "reasons": [], "details": {
                "outcome": "BLG", "team_a": "BLG", "team_b": "WE",
                "blue_win_probability": .63, "red_win_probability": .37,
                "blue_market_probability": .58, "red_market_probability": .42,
                "reasons": ["队伍底蕴优势"], "analyst_count": 2,
            },
        }
        with patch.object(server, "_send_message") as send:
            server._send_valuable_alert(prematch)
            send.assert_called_once()

    def test_health_requires_completed_live_scan(self):
        server = load_server_module()
        server.STATE.update(error=None, live_error=None, live=None)
        self.assertFalse(server._health_ready())
        available = {name: {"available": True} for name in (
            "thesportsdb_nba", "pandascore_lol", "leaguepedia_bp", "bo3_cs2",
            "polymarket_nba", "polymarket_lol", "polymarket_cs2",
        )}
        server.STATE["live"] = {"checked_at": "2026-08-14T00:00:00+00:00", "source_status": available}
        self.assertTrue(server._health_ready())
        available["thesportsdb_nba"]["available"] = False
        self.assertFalse(server._health_ready())
        available["thesportsdb_nba"]["available"] = True
        server.STATE["live_error"] = "source loop crashed"
        self.assertFalse(server._health_ready())


    def test_live_push_dedupes_repeated_alert_keys(self):
        server = load_server_module()
        server._SENT_LIVE_ALERT_KEYS.clear()
        alert = {
            "category": "PREMATCH_ANALYSIS", "severity": "IMPORTANT", "alert_score": 55,
            "sport": "lol", "title": "BLG vs WE", "summary": "赛前方向：BLG",
            "reasons": [], "dedupe_key": "lol:we:blg:PREMATCH_ANALYSIS",
            "details": {
                "outcome": "BLG", "team_a": "BLG", "team_b": "WE",
                "blue_win_probability": .63, "red_win_probability": .37,
                "blue_market_probability": .58, "red_market_probability": .42,
                "reasons": ["队伍底蕴优势"], "analyst_count": 2,
            },
        }
        with patch.object(server, "_send_message") as send:
            server._send_valuable_alert(alert)
            server._send_valuable_alert(alert)
        send.assert_called_once()

    def test_next_paper_summary_run_uses_configured_time(self):
        server = load_server_module()
        with patch.dict(os.environ, {"PAPER_SUMMARY_TIME": "23:45"}, clear=False):
            now = datetime(2026, 8, 15, 22, 0, tzinfo=timezone.utc)
            target = server._next_paper_summary_run(now)
            self.assertGreater(target, now)
            self.assertEqual(target.minute, 45)

    def test_run_paper_summary_sends_once_per_report_date(self):
        server = load_server_module()
        with tempfile.TemporaryDirectory() as directory:
            paper_path = Path(directory) / "paper.db"
            with patch.dict(os.environ, {"PAPER_DB_PATH": str(paper_path), "BANKROLL_USDC": "10000"}, clear=False),                  patch.object(server, "settle_pending", return_value={"checked": 0, "settled": 0, "pending": 0}),                  patch.object(server, "_send_message") as send:
                server._run_paper_summary()
                server._run_paper_summary()
        send.assert_called_once()

if __name__ == "__main__":
    unittest.main()
