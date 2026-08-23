from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

PREMATCH_WINDOWS = (("T24H", timedelta(hours=24)), ("T6H", timedelta(hours=6)),
                    ("T2H", timedelta(hours=2)), ("T30M", timedelta(minutes=30)),
                    ("T10M", timedelta(minutes=10)))


def _utc(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class MatchLifecycleStore:
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS match_lifecycle(
              match_id TEXT PRIMARY KEY, sport TEXT, scheduled_start TEXT,
              first_seen_at TEXT, first_prematch_scan_at TEXT,
              first_prematch_analysis_at TEXT, last_prematch_analysis_at TEXT,
              match_started_at TEXT, lifecycle TEXT, prematch_scan_missed INTEGER DEFAULT 0,
              last_scan_before_start TEXT, seconds_between_last_scan_and_start INTEGER,
              confirmed_roster_seen_at TEXT, confirmed_roster_first_used_at TEXT,
              market_last_updated_at TEXT, evidence_last_refreshed_at TEXT)""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(match_lifecycle)")}
            for name, kind in (("last_scan_before_start", "TEXT"), ("seconds_between_last_scan_and_start", "INTEGER"),
                               ("confirmed_roster_seen_at", "TEXT"), ("confirmed_roster_first_used_at", "TEXT"),
                               ("market_last_updated_at", "TEXT"), ("evidence_last_refreshed_at", "TEXT")):
                if name not in columns: db.execute(f"ALTER TABLE match_lifecycle ADD COLUMN {name} {kind}")
            db.execute("""CREATE TABLE IF NOT EXISTS prematch_scan_windows(
              match_id TEXT, window TEXT, scheduled_at TEXT, status TEXT,
              executed_at TEXT, missed_window TEXT, reason TEXT, PRIMARY KEY(match_id, window))""")
            db.commit()

    def _windows(self, db, match_id: str, start: datetime, first_seen: datetime,
                 now: datetime, eligible: bool, source_failed: bool) -> None:
        schedules = [(name, start - offset) for name, offset in PREMATCH_WINDOWS]
        for index, (name, scheduled) in enumerate(schedules):
            boundary = schedules[index + 1][1] if index + 1 < len(schedules) else start
            old = db.execute("SELECT status FROM prematch_scan_windows WHERE match_id=? AND window=?",
                             (match_id, name)).fetchone()
            if old and old[0] in {"EXECUTED", "MISSED", "SOURCE_FAILED"}: continue
            status, executed, missed, reason = "NOT_DUE", None, None, None
            if now >= scheduled:
                if now < boundary and eligible:
                    status = "SOURCE_FAILED" if source_failed else "EXECUTED"
                    executed = now.isoformat(); reason = "SOURCE_FAILED" if source_failed else None
                elif now >= boundary:
                    status, missed = "MISSED", name
                    reason = "SERVICE_NOT_RUNNING" if first_seen > boundary else "SCHEDULER_MISSED"
            db.execute("""INSERT INTO prematch_scan_windows VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(match_id,window) DO UPDATE SET status=excluded.status,
                executed_at=COALESCE(excluded.executed_at,prematch_scan_windows.executed_at),
                missed_window=excluded.missed_window,reason=excluded.reason""",
                (match_id, name, scheduled.isoformat(), status, executed, missed, reason))

    def observe(self, matches: list[dict], recommendations: list[dict], now: datetime,
                *, source_failed: bool = False) -> list[dict]:
        now = _utc(now); now_text = now.isoformat()
        recs = {str(row.get("schedule_match_id")): row for row in recommendations}
        with closing(sqlite3.connect(self.path)) as db:
            for match in matches:
                match_id = str(match.get("match_id") or ""); start = _utc(match["start_time"])
                prestart = now < start; matched = match.get("market_mapping_status") == "MATCHED"; analyzed = match_id in recs
                old = db.execute("SELECT first_seen_at,first_prematch_analysis_at FROM match_lifecycle WHERE match_id=?",
                                 (match_id,)).fetchone()
                first_seen = _utc(old[0]) if old else now; first_analysis = old[1] if old else None
                if analyzed and prestart and not first_analysis: first_analysis = now_text
                lifecycle = "PREMATCH_ANALYZED" if first_analysis else ("NO_MARKET" if not matched else
                    "DATA_INCOMPLETE" if prestart else "FINISHED" if str(match.get("event_status") or "").upper() == "FINISHED"
                    else "MARKET_STARTED_OR_IN_PLAY")
                missed = int(not prestart and matched and not first_analysis and first_seen < start)
                if missed: lifecycle = "PREMATCH_SCAN_MISSED"
                rec = recs.get(match_id, {}); confirmed = str(rec.get("lineup_status") or rec.get("roster_status") or "").upper() == "CONFIRMED"
                self._windows(db, match_id, start, first_seen, now, prestart and matched, source_failed)
                values = (match_id, match.get("sport"), start.isoformat(), first_seen.isoformat(), now_text if prestart and matched else None,
                          first_analysis, now_text if analyzed and prestart else None, start.isoformat() if not prestart else None,
                          lifecycle, missed, now_text if prestart and matched else None,
                          max(0, int((start-now).total_seconds())) if prestart and matched else None,
                          now_text if confirmed else None, now_text if confirmed and analyzed else None,
                          now_text if matched else None, now_text if analyzed else None)
                db.execute("""INSERT INTO match_lifecycle VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(match_id) DO UPDATE SET
                  first_prematch_scan_at=COALESCE(match_lifecycle.first_prematch_scan_at,excluded.first_prematch_scan_at),
                  first_prematch_analysis_at=COALESCE(match_lifecycle.first_prematch_analysis_at,excluded.first_prematch_analysis_at),
                  last_prematch_analysis_at=COALESCE(excluded.last_prematch_analysis_at,match_lifecycle.last_prematch_analysis_at),
                  match_started_at=COALESCE(excluded.match_started_at,match_lifecycle.match_started_at), lifecycle=excluded.lifecycle,
                  prematch_scan_missed=excluded.prematch_scan_missed,last_scan_before_start=COALESCE(excluded.last_scan_before_start,match_lifecycle.last_scan_before_start),
                  seconds_between_last_scan_and_start=COALESCE(excluded.seconds_between_last_scan_and_start,match_lifecycle.seconds_between_last_scan_and_start),
                  confirmed_roster_seen_at=COALESCE(match_lifecycle.confirmed_roster_seen_at,excluded.confirmed_roster_seen_at),
                  confirmed_roster_first_used_at=COALESCE(match_lifecycle.confirmed_roster_first_used_at,excluded.confirmed_roster_first_used_at),
                  market_last_updated_at=COALESCE(excluded.market_last_updated_at,match_lifecycle.market_last_updated_at),
                  evidence_last_refreshed_at=COALESCE(excluded.evidence_last_refreshed_at,match_lifecycle.evidence_last_refreshed_at)""", values)
            db.commit(); db.row_factory = sqlite3.Row; result = []
            for row in db.execute("SELECT * FROM match_lifecycle ORDER BY scheduled_start"):
                item = dict(row); windows = db.execute("SELECT * FROM prematch_scan_windows WHERE match_id=?", (item["match_id"],)).fetchall()
                item["prematch_scan_windows"] = {window[1]: dict(window) for window in windows}; result.append(item)
            return result

    def daily_metrics(self, report_date: str) -> dict:
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            matches = db.execute("SELECT * FROM match_lifecycle WHERE substr(scheduled_start,1,10)=?", (report_date,)).fetchall()
            eligible = [row for row in matches if row["first_prematch_scan_at"]]; metrics = {"prematch_eligible": len(eligible)}
            for name, _ in PREMATCH_WINDOWS:
                count = db.execute("""SELECT COUNT(*) FROM prematch_scan_windows w JOIN match_lifecycle m ON m.match_id=w.match_id
                  WHERE substr(m.scheduled_start,1,10)=? AND w.window=? AND w.status='EXECUTED'""", (report_date, name)).fetchone()[0]
                key = {"T24H": "t24_scan_rate", "T6H": "t6_scan_rate", "T2H": "t2_scan_rate",
                       "T30M": "t30_scan_rate", "T10M": "t10_scan_rate"}[name]
                metrics[key] = count / len(eligible) if eligible else 0.0
            metrics["matches_with_scan_within_15m_of_start"] = sum(row["seconds_between_last_scan_and_start"] is not None and row["seconds_between_last_scan_and_start"] <= 900 for row in eligible)
            metrics["confirmed_roster_available_before_start"] = sum(bool(row["confirmed_roster_seen_at"]) for row in matches)
            metrics["confirmed_roster_used_before_start"] = sum(bool(row["confirmed_roster_first_used_at"]) for row in matches)
            return metrics
