from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable

from .schemas import Evidence


ANALYST_VERSION = "v2.2"


def semantic_evidence_fingerprint(evidence: Iterable[Evidence]) -> str:
    """Hash evidence meaning, not scan-clock metadata.

    Fresh/stale/unknown status is semantic and therefore included.  Volatile
    observation timestamps and freshness age counters are deliberately omitted.
    """
    rows = []
    for item in evidence:
        payload = dict(item.payload)
        rows.append({
            "evidence_type": item.evidence_type,
            "status": str(payload.get("status") or "UNKNOWN").upper(),
            "value": payload.get("value"),
            "source": item.source,
            "published_at": item.published_at.isoformat() if item.published_at else None,
            "reliability_score": item.reliability_score,
            "freshness_state": "FRESH" if str(payload.get("status") or "").upper() == "AVAILABLE_FRESH"
                               else "STALE" if str(payload.get("status") or "").upper() == "AVAILABLE_STALE"
                               else "UNKNOWN",
        })
    rows.sort(key=lambda row: (str(row["evidence_type"]), str(row["source"])))
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def cache_key(match_id: str, evidence_hash: str, prompt_version: str,
              baseline_probability: float | None = None, provider: str = "unknown",
              model: str = "unknown", analyst_version: str = ANALYST_VERSION) -> str:
    payload = {"match_id": match_id, "evidence_hash": evidence_hash,
               "baseline_probability": baseline_probability,
               "prompt_version": prompt_version, "provider": provider,
               "model": model, "analyst_version": analyst_version}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class AnalysisCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS ai_cache(cache_key TEXT PRIMARY KEY,payload_json TEXT NOT NULL)")
            db.commit()

    def get(self, key: str) -> dict | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT payload_json FROM ai_cache WHERE cache_key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict) -> None:
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("INSERT OR REPLACE INTO ai_cache VALUES(?,?)", (key, json.dumps(payload, default=str)))
            db.commit()
