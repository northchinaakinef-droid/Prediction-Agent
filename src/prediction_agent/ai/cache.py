from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path


def cache_key(match_id: str, evidence_hash: str, prompt_version: str) -> str:
    return hashlib.sha256(f"{match_id}|{evidence_hash}|{prompt_version}".encode()).hexdigest()


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
