from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import sqlite3
from contextlib import closing
from pathlib import Path


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    sport: str
    capability: str
    status: str
    last_attempt_at: datetime
    last_success_at: datetime | None
    http_status: int | None
    latency_ms: int | None
    records_received: int
    error_type: str | None
    error_message: str | None
    freshness_seconds: int | None

    def as_dict(self) -> dict:
        row = asdict(self)
        row["last_attempt_at"] = self.last_attempt_at.isoformat()
        row["last_success_at"] = self.last_success_at.isoformat() if self.last_success_at else None
        return row


PROVIDER_CAPABILITIES = {
    "nextmatch": ("SCHEDULE",), "esportagenda": ("SCHEDULE",),
    "leaguepedia_sched": ("SCHEDULE",), "pandascore_lol_sched": ("SCHEDULE",),
    "nba_official": ("SCHEDULE", "RESULTS"), "hupu": ("SCHEDULE",),
    "espn_core": ("SCHEDULE", "RESULTS", "TEAM_STATS"), "espn": ("SCHEDULE", "RESULTS"),
    "thesportsdb": ("SCHEDULE", "RESULTS"), "sportsrc": ("SCHEDULE",),
    "bo3": ("SCHEDULE",), "esportagenda_cs2": ("SCHEDULE",),
    "grid": ("SCHEDULE", "RESULTS", "ROSTER", "PLAYER_STATS", "MAP_STATS"),
    "pandascore": ("SCHEDULE", "RESULTS", "ROSTER", "PLAYER_STATS"),
    "polymarket": ("MARKET", "MARKET_HISTORY"),
}


class ProviderHealthStore:
    """Append-only runtime observations used for full-day provider statistics."""
    def __init__(self, path: str | Path):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS provider_health_observations(
              observed_at TEXT, provider TEXT, sport TEXT, capability TEXT, status TEXT,
              latency_ms INTEGER, records_received INTEGER, last_success_at TEXT,
              error_type TEXT, error_message TEXT)"""); db.commit()

    def record(self, rows: list[dict], observed_at: datetime) -> None:
        with closing(sqlite3.connect(self.path)) as db:
            db.executemany("INSERT INTO provider_health_observations VALUES(?,?,?,?,?,?,?,?,?,?)", [
                (observed_at.isoformat(), row["provider"], row["sport"], row["capability"], row["status"],
                 row.get("latency_ms"), row.get("records_received", 0), row.get("last_success_at"),
                 row.get("error_type"), row.get("error_message")) for row in rows]); db.commit()

    def daily_summary(self, report_date: str) -> dict:
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            groups = db.execute("""SELECT provider,sport,capability,COUNT(*) requests,
              SUM(CASE WHEN status='ACTIVE' THEN 1 ELSE 0 END) success,
              SUM(CASE WHEN status!='ACTIVE' THEN 1 ELSE 0 END) failure,
              MAX(last_success_at) last_success,SUM(records_received) records_received
              FROM provider_health_observations WHERE substr(observed_at,1,10)=?
              GROUP BY provider,sport,capability""", (report_date,)).fetchall()
            result = {}
            for group in groups:
                latencies = [row[0] for row in db.execute("""SELECT latency_ms FROM provider_health_observations
                  WHERE substr(observed_at,1,10)=? AND provider=? AND sport=? AND capability=? AND latency_ms IS NOT NULL ORDER BY latency_ms""",
                  (report_date, group["provider"], group["sport"], group["capability"])).fetchall()]
                p95 = latencies[max(0, int(len(latencies) * .95 + .999) - 1)] if latencies else None
                item = dict(group); item["success_rate"] = item["success"] / item["requests"] if item["requests"] else 0.0
                item["p95_latency_ms"] = p95
                result[f'{group["sport"]}:{group["provider"]}:{group["capability"]}'] = item
            return result
