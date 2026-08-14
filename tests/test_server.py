from __future__ import annotations

import importlib.util
import os
import unittest
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

    def test_health_requires_completed_live_scan(self):
        server = load_server_module()
        server.STATE.update(error=None, live_error=None, live=None)
        self.assertFalse(server._health_ready())
        server.STATE["live"] = {"checked_at": "2026-08-14T00:00:00+00:00"}
        self.assertTrue(server._health_ready())
        server.STATE["live_error"] = "source loop crashed"
        self.assertFalse(server._health_ready())


if __name__ == "__main__":
    unittest.main()
