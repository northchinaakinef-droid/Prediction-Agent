from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from .telemetry import event_log


SKIP_REASONS = {
    "NO_MARKET", "MISSING_ROSTER", "MISSING_ODDS", "ENTITY_MATCH_FAILED",
    "MODEL_UNAVAILABLE", "SOURCE_ERROR", "DATA_INCOMPLETE", "MARKET_STARTED_OR_IN_PLAY",
}


@dataclass(frozen=True)
class CoverageEvent:
    match_id: str
    sport: str
    league: str
    schedule_source: str
    market_status: str
    analysis_status: str
    skip_reason: str | None = None


@dataclass(frozen=True)
class CoverageReport:
    report_date: str
    total_schedule_events: int
    matched_market_events: int
    analyzed_events: int
    skipped_events: int
    data_incomplete_events: int
    no_market_events: int
    events: tuple[CoverageEvent, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def build_coverage_report(report_date: str, schedule_matches: Iterable[dict],
                          recommendations: Iterable[dict], model_ready: dict[str, bool] | None = None) -> CoverageReport:
    recs = list(recommendations)
    model_ready = model_ready or {}
    events: list[CoverageEvent] = []
    for match in schedule_matches:
        match_id = str(match.get("match_id") or "")
        sport = str(match.get("sport") or "")
        market_status = "MATCHED" if match.get("market_mapping_status") == "MATCHED" else "NO_MARKET"
        related = [row for row in recs if row.get("schedule_match_id") == match_id]
        explicit_reason = str(match.get("skip_reason") or "") or None
        if explicit_reason:
            status, reason = "DATA_INCOMPLETE", explicit_reason
        elif not model_ready.get(sport, True):
            status, reason = "DATA_INCOMPLETE", "MODEL_UNAVAILABLE"
        elif market_status == "NO_MARKET":
            status, reason = "NO_MARKET", "NO_MARKET"
        elif not related:
            watcher = str(match.get("watcher_status") or "").upper()
            event_status = str(match.get("event_status") or "").upper()
            if watcher in {"LIVE", "FINISHED"} or event_status in {"LIVE", "CURRENT", "FINISHED"}:
                status, reason = "DATA_INCOMPLETE", "MARKET_STARTED_OR_IN_PLAY"
            else:
                status, reason = "DATA_INCOMPLETE", "MISSING_ODDS"
        else:
            status, reason = "ANALYZED", None
        events.append(CoverageEvent(
            match_id, sport, str(match.get("league") or ""),
            ",".join(str(value) for value in (match.get("sources") or [match.get("source")]) if value),
            market_status, status, reason,
        ))
    analyzed = sum(row.analysis_status == "ANALYZED" for row in events)
    no_market = sum(row.market_status == "NO_MARKET" for row in events)
    incomplete = sum(row.analysis_status == "DATA_INCOMPLETE" for row in events)
    return CoverageReport(
        report_date, len(events), sum(row.market_status == "MATCHED" for row in events),
        analyzed, len(events) - analyzed, incomplete, no_market, tuple(events),
    )


class CoverageStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS coverage_runs(
              report_date TEXT PRIMARY KEY, generated_at TEXT NOT NULL, report_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS coverage_events(
              report_date TEXT NOT NULL, match_id TEXT NOT NULL, sport TEXT NOT NULL, league TEXT NOT NULL,
              market_status TEXT NOT NULL, analysis_status TEXT NOT NULL, skip_reason TEXT,
              schedule_source TEXT,
              PRIMARY KEY(report_date, match_id)
            );
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(coverage_events)")}
            if "schedule_source" not in columns:
                db.execute("ALTER TABLE coverage_events ADD COLUMN schedule_source TEXT")
            db.commit()

    def save(self, report: CoverageReport) -> None:
        payload = json.dumps(report.as_dict(), ensure_ascii=False)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute(
                "INSERT OR REPLACE INTO coverage_runs VALUES(?,?,?)",
                (report.report_date, datetime.now(timezone.utc).isoformat(), payload),
            )
            for row in report.events:
                db.execute(
                    """INSERT OR REPLACE INTO coverage_events(
                       report_date,match_id,sport,league,market_status,analysis_status,skip_reason,schedule_source
                       ) VALUES(?,?,?,?,?,?,?,?)""",
                    (report.report_date, row.match_id, row.sport, row.league,
                     row.market_status, row.analysis_status, row.skip_reason, row.schedule_source),
                )
                event_log("analysis_completed" if row.analysis_status == "ANALYZED" else "analysis_skipped",
                          match_id=row.match_id, sport=row.sport, league=row.league,
                          run_id=report.report_date, skip_reason=row.skip_reason)
            db.commit()

    def event(self, report_date: str, match_id: str) -> dict | None:
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM coverage_events WHERE report_date=? AND match_id=?", (report_date, match_id),
            ).fetchone()
        return dict(row) if row else None
