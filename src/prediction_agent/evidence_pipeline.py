from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field

from .ai.schemas import Evidence


SPORT_EVIDENCE_TYPES = {
    "lol": ("ROSTER", "RECENT_FORM", "TEAM_STRENGTH", "PLAYER_STRENGTH", "PATCH",
            "CHAMPION_POOL", "SIDE", "SCHEDULE", "MARKET", "MARKET_MOVEMENT", "NEWS"),
    "cs2": ("ROSTER", "RECENT_FORM", "TEAM_RATING", "PLAYER_RATING", "MAP_POOL",
            "MAP_WIN_RATE", "MAP_PICK_BAN", "LAN_ONLINE_CONTEXT", "SCHEDULE", "MARKET",
            "MARKET_MOVEMENT", "NEWS"),
    "nba": ("INJURY", "CONFIRMED_STARTERS", "EXPECTED_LINEUP", "PLAYER_AVAILABILITY",
            "EXPECTED_MINUTES", "RECENT_FORM", "REST", "BACK_TO_BACK", "SCHEDULE_DENSITY",
            "TEAM_STRENGTH", "OFFENSIVE_RATING", "DEFENSIVE_RATING", "PACE", "MARKET",
            "MARKET_MOVEMENT", "NEWS", "SCHEDULE"),
}

FRESHNESS_SLA_SECONDS = {
    "SCHEDULE": 24 * 3600, "MARKET": 5 * 60, "MARKET_MOVEMENT": 5 * 60,
    "ROSTER": 24 * 3600, "RECENT_FORM": 72 * 3600, "INJURY": 6 * 3600,
    "CONFIRMED_STARTERS": 2 * 3600, "NEWS": 24 * 3600,
}


def freshness_sla(evidence_type: str) -> int:
    name = f"EVIDENCE_FRESHNESS_{evidence_type}_SECONDS"
    return int(os.getenv(name, str(FRESHNESS_SLA_SECONDS.get(evidence_type, 7 * 24 * 3600))))


class PlayerAvailabilityEvidence(BaseModel):
    player_id: str
    team_id: str
    status: str = "UNKNOWN"
    expected_minutes: float | None = Field(default=None, ge=0)
    source: str
    published_at: datetime | None = None
    observed_at: datetime
    confidence: float = Field(ge=0, le=1)


class EvidenceStore:
    """Immutable, content-addressed facts supplied to the LLM."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS evidence(
                evidence_id TEXT PRIMARY KEY, match_id TEXT NOT NULL, evidence_type TEXT NOT NULL,
                observed_at TEXT NOT NULL, payload_json TEXT NOT NULL)""")
            db.commit()

    def put_many(self, evidence: list[Evidence]) -> None:
        with closing(sqlite3.connect(self.path)) as db:
            db.executemany("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?,?)", [
                (item.evidence_id, item.match_id, item.evidence_type, item.observed_at.isoformat(),
                 json.dumps(item.as_dict(), sort_keys=True, ensure_ascii=False)) for item in evidence
            ])
            db.commit()

    def for_match(self, match_id: str) -> list[dict]:
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT payload_json FROM evidence WHERE match_id=? ORDER BY evidence_type",
                              (match_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]


def build_match_evidence(row: dict[str, Any], sport: str, match_id: str,
                         observed_at: datetime) -> list[Evidence]:
    """Normalize a production analysis row; missing sources remain explicit UNKNOWN facts."""
    payloads: dict[str, Any] = {
        "SCHEDULE": {"start": row.get("scheduled_start"), "mapping": row.get("market_mapping_status")},
        "MARKET": {"probability": row.get("market_probability"), "price": row.get("execution_price"),
                   "volume": row.get("market_volume"), "liquidity": row.get("market_liquidity")},
        "MARKET_MOVEMENT": row.get("market_movement"),
        "ROSTER": {"status": row.get("lineup_status"), "team_a": row.get("lineup_a"),
                   "team_b": row.get("lineup_b")},
        "RECENT_FORM": {"team_a": row.get("recent_form_a"), "team_b": row.get("recent_form_b")},
        "TEAM_STRENGTH": {"quant_probability": row.get("model_probability")},
        "TEAM_RATING": {"quant_probability": row.get("model_probability")},
        "PLAYER_STRENGTH": row.get("player_strength"), "PLAYER_RATING": row.get("player_rating"),
        "PATCH": row.get("patch"),
        "CHAMPION_POOL": row.get("champion_pool") or (
            {"meta_heroes": row.get("patch_meta_heroes"), "coverage_a": row.get("meta_coverage_a"),
             "coverage_b": row.get("meta_coverage_b")}
            if row.get("patch_meta_heroes") and row.get("meta_coverage_a") is not None
            and row.get("meta_coverage_b") is not None else None),
        "SIDE": row.get("side"),
        "MAP_POOL": row.get("map_pool"), "MAP_WIN_RATE": row.get("map_win_rate"),
        "MAP_PICK_BAN": row.get("map_pick_ban"), "LAN_ONLINE_CONTEXT": row.get("lan_online_context"),
        "INJURY": row.get("injury_status"), "CONFIRMED_STARTERS": row.get("confirmed_starters"),
        "EXPECTED_LINEUP": row.get("expected_lineup"), "PLAYER_AVAILABILITY": row.get("player_availability"),
        "EXPECTED_MINUTES": row.get("expected_minutes"), "REST": row.get("rest"),
        "BACK_TO_BACK": row.get("back_to_back"), "SCHEDULE_DENSITY": row.get("schedule_density"),
        "OFFENSIVE_RATING": row.get("offensive_rating"), "DEFENSIVE_RATING": row.get("defensive_rating"),
        "PACE": row.get("pace"), "NEWS": row.get("news_evidence"),
    }
    result = []
    for evidence_type in SPORT_EVIDENCE_TYPES[sport]:
        value = payloads.get(evidence_type)
        if evidence_type == "ROSTER":
            unknown = (not isinstance(value, dict) or not value.get("team_a") or not value.get("team_b")
                       or str(value.get("status") or "UNKNOWN").upper() == "UNKNOWN")
        elif evidence_type == "RECENT_FORM":
            unknown = not isinstance(value, dict) or not value.get("team_a") or not value.get("team_b")
        else:
            unknown = value is None or value == {} or (isinstance(value, dict) and
                      all(item is None or item == [] for item in value.values()))
        published = ({"ROSTER": row.get("roster_published_at"),
                      "RECENT_FORM": row.get("recent_form_artifact_generated_at"),
                      "NEWS": row.get("news_published_at")}.get(evidence_type))
        timestamp = None
        if published:
            try:
                timestamp = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                if timestamp.tzinfo is None: timestamp = timestamp.replace(tzinfo=timezone.utc)
            except ValueError:
                timestamp = None
        age = max(0, int((observed_at - timestamp).total_seconds())) if timestamp else 0
        state = "UNKNOWN" if unknown else ("AVAILABLE_STALE" if age > freshness_sla(evidence_type)
                                             else "AVAILABLE_FRESH")
        payload = {"status": state, "freshness_seconds": age,
                   "freshness_sla_seconds": freshness_sla(evidence_type)}
        if not unknown:
            payload["value"] = value
        semantic_identity = {
            "match_id": match_id, "evidence_type": evidence_type,
            "availability_state": state, "value": payload.get("value"),
            "source": "production_normalizer", "published_at": published,
            "reliability": .85 if not unknown else 0.0,
        }
        digest = hashlib.sha256(json.dumps(semantic_identity, sort_keys=True,
                                           default=str).encode()).hexdigest()[:20]
        result.append(Evidence.validate_payload({
            "evidence_id": f"{evidence_type.lower()}:{digest}", "match_id": match_id,
            "evidence_type": evidence_type, "source": "production_normalizer",
            "observed_at": observed_at.isoformat(), "published_at": published, "payload": payload,
            "reliability_score": .85 if not unknown else 0.0,
            "freshness_score": 1.0 if state == "AVAILABLE_FRESH" else 0.0,
        }))
    return result


def evidence_hash(evidence: list[Evidence]) -> str:
    return hashlib.sha256(json.dumps([item.as_dict() for item in evidence], sort_keys=True,
                                     default=str).encode()).hexdigest()


def evidence_availability(evidence: list[Evidence]) -> dict[str, Any]:
    states = {item.evidence_type: str(item.payload.get("status") or "UNKNOWN").upper()
              for item in evidence}
    available = [item for item in evidence if states[item.evidence_type].startswith("AVAILABLE_")]
    fresh = [item for item in evidence if states[item.evidence_type] == "AVAILABLE_FRESH"]
    stale = [item for item in evidence if states[item.evidence_type] == "AVAILABLE_STALE"]
    available_ids = {item.evidence_id for item in available}
    total = len(evidence)
    return {"evidence_total_count": total, "evidence_available_count": len(available),
            "evidence_fresh_count": len(fresh), "evidence_stale_count": len(stale),
            "evidence_unknown_count": total - len(available),
            "evidence_available_ratio": len(available) / total if total else 0.0,
            "fresh_available_ratio": len(fresh) / total if total else 0.0,
            "evidence_type_status": states,
            "available_evidence_ids": [item.evidence_id for item in available],
            "fresh_evidence_ids": [item.evidence_id for item in fresh],
            "unknown_evidence_ids": [item.evidence_id for item in evidence
                                     if item.evidence_id not in available_ids]}
