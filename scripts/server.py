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

from prediction_agent.delivery import FeishuAppClient, FeishuWebhookClient, format_daily_post, format_live_alert
from prediction_agent.live_runtime import LiveSupervisor
from prediction_agent.sports_daily import run_all
from prediction_agent.paper_store import record_report, settle_pending, summary as paper_summary


ROOT = Path(__file__).resolve().parents[1]
STATE = {"started_at": datetime.now(timezone.utc).isoformat(), "last_run": None,
         "last_ok": None, "last_scan": None, "last_push": None,
         "error": None, "paper_store": None, "paper_settlement": None,
         "live": None, "live_error": None}
RUN_LOCK = threading.Lock()
SOURCE_LABELS = {
    "nba_official": "NBA 官方比分", "espn_nba": "ESPN NBA", "thesportsdb_nba": "TheSportsDB NBA",
    "riot_esports": "LoL 官方 BP", "pandascore_lol": "PandaScore LoL",
    "leaguepedia_bp": "Leaguepedia BP", "bo3_cs2": "BO3.gg CS2",
    "grid_cs2": "GRID CS2", "pandascore_cs2": "PandaScore CS2",
    "news_rss": "新闻源", "polymarket_nba": "Polymarket NBA",
    "polymarket_lol": "Polymarket LoL", "polymarket_cs2": "Polymarket CS2",
}


def _send(report: dict) -> None:
    post = format_daily_post(report)
    if os.getenv("FEISHU_WEBHOOK_URL"):
        FeishuWebhookClient(os.environ["FEISHU_WEBHOOK_URL"],
                            os.getenv("FEISHU_WEBHOOK_SECRET") or None).send_post(post)
        return
    required = ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_RECEIVE_ID")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("missing Feishu configuration: " + ", ".join(missing))
    FeishuAppClient(os.environ["FEISHU_APP_ID"], os.environ["FEISHU_APP_SECRET"],
                    os.environ["FEISHU_RECEIVE_ID"],
                    os.getenv("FEISHU_RECEIVE_ID_TYPE", "open_id")).send_post(post)


def _send_message(message: str) -> None:
    if os.getenv("FEISHU_WEBHOOK_URL"):
        FeishuWebhookClient(os.environ["FEISHU_WEBHOOK_URL"],
                            os.getenv("FEISHU_WEBHOOK_SECRET") or None).send_text(message)
        return
    required = ("FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_RECEIVE_ID")
    if any(not os.getenv(name) for name in required):
        return
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
            report["paper_summary"] = paper_summary(paper_path)
            STATE["last_scan"] = datetime.now(timezone.utc).isoformat()
            if notify:
                _send(report)
                STATE["last_push"] = datetime.now(timezone.utc).isoformat()
            STATE.update(last_ok=datetime.now(timezone.utc).isoformat(), error=None)
        except Exception as error:
            STATE["error"] = repr(error)
            raise


def _next_run(now: datetime) -> datetime:
    zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
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


ALLOWED_LIVE_ALERT_CATEGORIES = {"PREMATCH_ANALYSIS", "DRAFT_ANALYSIS", "POSTMATCH_REVIEW"}


def _send_valuable_alert(alert) -> None:
    category = getattr(alert, "category", None)
    if category is None and isinstance(alert, dict):
        category = alert.get("category")
    if category in ALLOWED_LIVE_ALERT_CATEGORIES:
        _send_message(format_live_alert(alert))


def live_scheduler() -> None:
    supervisor = LiveSupervisor(root=ROOT, on_alert=_send_valuable_alert)
    interval = max(10, int(os.getenv("LIVE_SCAN_SECONDS", "30")))
    while True:
        scan_started = time.monotonic()
        try:
            result = supervisor.scan_once()
            STATE["live"] = result
            STATE["live_error"] = None
        except Exception as error:
            STATE["live_error"] = repr(error)
        time.sleep(max(1.0, interval - (time.monotonic() - scan_started)))


def _health_ready() -> bool:
    if STATE["error"] is not None or STATE["live_error"] is not None or STATE["live"] is None:
        return False
    sources = STATE["live"].get("source_status", {})
    required_groups = (
        ("nba_official", "espn_nba", "thesportsdb_nba"),
        ("pandascore_lol",),
        ("riot_esports", "leaguepedia_bp"),
        ("bo3_cs2", "grid_cs2", "pandascore_cs2"),
        ("polymarket_nba",), ("polymarket_lol",), ("polymarket_cs2",),
    )
    return all(any(sources.get(name, {}).get("available") for name in group) for group in required_groups)


class Health(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path not in {"/", "/health"}:
            self.send_error(404)
            return
        payload = json.dumps(STATE, ensure_ascii=False).encode("utf-8")
        self.send_response(200 if _health_ready() else 503)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args):
        return


def main() -> None:
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=paper_scheduler, daemon=True).start()
    threading.Thread(target=live_scheduler, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", int(os.getenv("PORT", "8080"))), Health).serve_forever()


if __name__ == "__main__":
    main()
