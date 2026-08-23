from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelRecord:
    model_name: str
    version: str
    trained_at: str
    training_end_date: str
    dataset_hash: str
    metrics: dict
    status: str


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def read(self) -> dict[str, ModelRecord]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return {name: ModelRecord(**row) for name, row in payload.items()}

    def register(self, record: ModelRecord) -> None:
        records = self.read()
        records[record.model_name] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: asdict(v) for k, v in records.items()}, indent=2), encoding="utf-8")

    @staticmethod
    def freshness(training_end_date: str, *, today: date | None = None, interval_days: int = 42) -> dict:
        today = today or datetime.now(timezone.utc).date()
        trained = datetime.fromisoformat(training_end_date.replace("Z", "+00:00")).date()
        age = (today - trained).days
        return {"days_since_training": age, "interval_days": interval_days,
                "status": "STALE" if age > interval_days else "OK"}


def dataset_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
