"""Long-running production scheduler for a Docker/VPS deployment."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

from prediction_agent.delivery import FeishuAppClient, FeishuWebhookClient, format_daily_report
from prediction_agent.sports_daily import run_all
from prediction_agent.paper_store import record_report, settle_pending


ROOT = Path(__file__).resolve().parents[1]
STATE = {"started_at": datetime.now(timezone.utc).isoformat(), "last_run": None,
         "last_ok": None, "last_scan": None, "last_push": None,
         "error": None, "paper_store": None, "paper_settlement": None}
RUN_LOCK = threading.Lock()


def _send(report: dict) -> None:
    message = format_daily_report(report)
    if os.getenv("FEISHU_WEBHOOK_URL"):
        FeishuWebhookClient(os.environ["FEISHU_WEBHOOK_URL"],
                            os.getenv("FEISHU_WEBHOOK_SECRET") or None).send_text(message)
        return
    required = ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_RECEIVE_ID")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("missing Feishu configuration: " + ", ".join(missing))
    FeishuAppClient(os.environ["FEISHU_APP_ID"], os.environ["FEISHU_APP_SECRET"],
                    os.environ["FEISHU_RECEIVE_ID"],
                    os.getenv("FEISHU_RECEIVE_ID_TYPE", "open_id")).send_text(message)


def run_once(*, notify: bool = True) -> None:
    with RUN_LOCK:
        STATE["last_run"] = datetime.now(timezone.utc).isoformat()
        try:
            report = run_all(ROOT / "artifacts", ROOT / "reports" / "daily.json")
            missing = [sport for sport, status in report["sport_status"].items() if not status.get("ready")]
            if missing:
                raise RuntimeError("production models not ready: " + ", ".join(missing))
            paper_path = Path(os.getenv("PAPER_DB_PATH", str(ROOT / "data" / "daily" / "paper.db")))
            STATE["paper_settlement"] = settle_pending(paper_path)
            STATE["paper_store"] = record_report(paper_path, report)
            STATE["last_scan"] = datetime.now(timezone.utc).isoformat()
            if notify:
                _send(report)
                STATE["last_push"] = datetime.now(timezone.utc).isoformat()
            STATE.update(last_ok=datetime.now(timezone.utc).isoformat(), error=None)
        except Exception as error:
            STATE["error"] = repr(error)
            raise


def _next_run(now: datetime) -> datetime:
    zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Shanghai"))
    hour, minute = (int(x) for x in os.getenv("DAILY_RUN_TIME", "06:30").split(":", 1))
    local = now.astimezone(zone)
    target = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= local:
        target += timedelta(days=1)
    return target.astimezone(timezone.utc)


def scheduler() -> None:
    if os.getenv("RUN_ON_START", "false").casefold() == "true":
        try:
            run_once()
        except Exception:
            pass
    while True:
        now = datetime.now(timezone.utc)
        wait = max(1.0, (_next_run(now) - now).total_seconds())
        time.sleep(min(wait, 60))
        if datetime.now(timezone.utc) >= _next_run(now) - timedelta(seconds=1):
            try:
                run_once()
            except Exception:
                pass


def paper_scheduler() -> None:
    minutes = max(5, int(os.getenv("PAPER_SCAN_MINUTES", "30")))
    while True:
        try:
            run_once(notify=False)
        except Exception:
            pass
        time.sleep(minutes * 60)


class Health(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = json.dumps(STATE, ensure_ascii=False).encode("utf-8")
        self.send_response(200 if not STATE["error"] else 503)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        return


def main() -> None:
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=paper_scheduler, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), Health).serve_forever()


if __name__ == "__main__":
    main()
