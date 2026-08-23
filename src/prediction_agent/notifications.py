from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable
from .telemetry import event_log


class AlertCategory(str, Enum):
    PREMATCH_ANALYSIS = "PREMATCH_ANALYSIS"
    LINEUP_CONFIRMED = "LINEUP_CONFIRMED"
    LINEUP_CHANGED = "LINEUP_CHANGED"
    LINEUP_MISSING = "LINEUP_MISSING"
    DRAFT_ANALYSIS = "DRAFT_ANALYSIS"
    PROBABILITY_CHANGE = "PROBABILITY_CHANGE"
    MARKET_MOVE = "MARKET_MOVE"
    MARKET_ANOMALY = "MARKET_ANOMALY"
    LIVE_STATE_CHANGE = "LIVE_STATE_CHANGE"
    MAJOR_EVENT = "MAJOR_EVENT"
    CLUTCH_TIME = "CLUTCH_TIME"
    DATA_SOURCE_FAILURE = "DATA_SOURCE_FAILURE"
    WATCHER_MISSING = "WATCHER_MISSING"
    MONITORING_RECOVERY = "MONITORING_RECOVERY"
    POSTMATCH_REVIEW = "POSTMATCH_REVIEW"
    DAILY_REPORT = "DAILY_REPORT"


@dataclass(frozen=True)
class OutboxMessage:
    id: int
    event_id: str
    match_id: str | None
    category: str
    payload: dict
    status: str
    retry_count: int
    dedupe_key: str


class NotificationOutbox:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS notification_outbox(
              id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL, match_id TEXT,
              category TEXT NOT NULL, payload_json TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('PENDING','SENDING','SENT','FAILED','DEAD')),
              retry_count INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
              last_attempt_at TEXT, next_retry_at TEXT, sent_at TEXT, error_message TEXT,
              dedupe_key TEXT NOT NULL UNIQUE
            );
            CREATE TABLE IF NOT EXISTS daily_report_runs(
              report_date TEXT PRIMARY KEY, generated_at TEXT NOT NULL, send_status TEXT NOT NULL,
              retry_count INTEGER NOT NULL DEFAULT 0, sent_at TEXT, error TEXT, dedupe_key TEXT NOT NULL
            );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def enqueue(self, *, event_id: str, category: str, payload: dict, dedupe_key: str,
                match_id: str | None = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO notification_outbox
                   (event_id,match_id,category,payload_json,status,retry_count,created_at,next_retry_at,dedupe_key)
                   VALUES(?,?,?,?, 'PENDING',0,?,?,?)""",
                (event_id, match_id, category, json.dumps(payload, ensure_ascii=False), now, now, dedupe_key),
            )
            row = db.execute("SELECT id FROM notification_outbox WHERE dedupe_key=?", (dedupe_key,)).fetchone()
        event_log("alert_created", match_id=match_id or "", run_id=event_id,
                  category=category, dedupe_key=dedupe_key)
        return int(row["id"])

    def enqueue_daily(self, report_date: str, payload: dict) -> int:
        key = f"daily:{report_date}"
        message_id = self.enqueue(
            event_id=report_date, category=AlertCategory.DAILY_REPORT.value,
            payload=payload, dedupe_key=key,
        )
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO daily_report_runs VALUES(?,?, 'PENDING',0,NULL,NULL,?)",
                (report_date, now, key),
            )
        return message_id

    def due(self, limit: int = 50) -> list[OutboxMessage]:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            rows = db.execute(
                """SELECT * FROM notification_outbox
                   WHERE status IN ('PENDING','FAILED') AND (next_retry_at IS NULL OR next_retry_at<=?)
                   ORDER BY id LIMIT ?""", (now, limit),
            ).fetchall()
        return [OutboxMessage(
            int(row["id"]), row["event_id"], row["match_id"], row["category"],
            json.loads(row["payload_json"]), row["status"], int(row["retry_count"]), row["dedupe_key"],
        ) for row in rows]

    def mark_sending(self, message_id: int) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                "UPDATE notification_outbox SET status='SENDING',last_attempt_at=? WHERE id=? AND status IN ('PENDING','FAILED')",
                (datetime.now(timezone.utc).isoformat(), message_id),
            )
        return cursor.rowcount == 1

    def mark_sent(self, message: OutboxMessage) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.connect() as db:
            db.execute("UPDATE notification_outbox SET status='SENT',sent_at=?,error_message=NULL WHERE id=?", (now, message.id))
            if message.category == AlertCategory.DAILY_REPORT.value:
                db.execute(
                    "UPDATE daily_report_runs SET send_status='SENT',sent_at=?,error=NULL WHERE dedupe_key=?",
                    (now, message.dedupe_key),
                )
        event_log("notification_sent", match_id=message.match_id or "", run_id=message.event_id,
                  category=message.category, dedupe_key=message.dedupe_key)

    def mark_failed(self, message: OutboxMessage, error: Exception, max_retries: int = 5) -> None:
        retry = message.retry_count + 1
        status = "DEAD" if retry >= max_retries else "FAILED"
        delay = min(3600, 30 * (2 ** max(0, retry - 1)))
        next_retry = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
        with self.connect() as db:
            db.execute(
                """UPDATE notification_outbox SET status=?,retry_count=?,next_retry_at=?,error_message=?
                   WHERE id=?""", (status, retry, next_retry, repr(error), message.id),
            )
            if message.category == AlertCategory.DAILY_REPORT.value:
                db.execute(
                    "UPDATE daily_report_runs SET send_status=?,retry_count=?,error=? WHERE dedupe_key=?",
                    (status, retry, repr(error), message.dedupe_key),
                )
        event_log("notification_failed", match_id=message.match_id or "", run_id=message.event_id,
                  category=message.category, retry_count=retry, error=repr(error))

    def status_counts(self) -> dict[str, int]:
        with self.connect() as db:
            rows = db.execute("SELECT status,COUNT(*) count FROM notification_outbox GROUP BY status").fetchall()
        result = {name: 0 for name in ("PENDING", "SENDING", "SENT", "FAILED", "DEAD")}
        result.update({row["status"]: int(row["count"]) for row in rows})
        return result


class OutboxWorker:
    def __init__(self, outbox: NotificationOutbox, sender: Callable[[OutboxMessage], None], max_retries: int = 5):
        self.outbox, self.sender, self.max_retries = outbox, sender, max_retries

    def run_once(self) -> dict[str, int]:
        sent = failed = 0
        for message in self.outbox.due():
            if not self.outbox.mark_sending(message.id):
                continue
            try:
                self.sender(message)
            except Exception as error:
                self.outbox.mark_failed(message, error, self.max_retries)
                failed += 1
            else:
                self.outbox.mark_sent(message)
                sent += 1
        return {"sent": sent, "failed": failed}
