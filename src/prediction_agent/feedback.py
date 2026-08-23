from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class PredictionSnapshot:
    match_id: str
    prediction_time: str
    quant_model_version: str
    ai_prompt_version: str | None
    ai_model_name: str | None
    baseline_probability: float
    ai_adjustment: float
    final_probability: float
    market_fair_probability: float | None
    edge: float | None
    features_json: dict
    evidence_json: list[dict]
    risk_flags_json: list[str]
    recommendation: str
    position_size: float
    llm_provider: str | None = None
    data_quality_score: float = 0.0
    missing_evidence: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    evidence_hash: str = ""
    ai_confidence: float = 0.0
    market_price: float | None = None
    adjusted_edge: float | None = None
    expected_profit_per_share: float | None = None
    expected_roi_on_capital: float | None = None
    execution_price: float | None = None
    decimal_odds: float | None = None
    raw_model_edge: float | None = None
    executable_edge: float | None = None
    risk_inputs: dict | None = None
    risk_output: dict | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class PostmatchAttribution:
    snapshot_id: str
    prediction_correct: bool | None
    error_type: tuple[str, ...]
    probability_error: float
    brier_score: float
    log_loss: float
    market_clv: float | None
    feature_failures: tuple[str, ...]
    ai_evidence_failures: tuple[str, ...]
    unexpected_events: tuple[str, ...]


class FeedbackStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS prediction_snapshots(
              snapshot_id TEXT PRIMARY KEY, match_id TEXT NOT NULL, prediction_time TEXT NOT NULL,
              payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS postmatch_attributions(
              snapshot_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload_json TEXT NOT NULL
            );
            """)
            db.commit()

    def save_snapshot(self, snapshot: PredictionSnapshot) -> str:
        payload = json.dumps(asdict(snapshot), ensure_ascii=False, sort_keys=True, default=str)
        snapshot_id = hashlib.sha256(payload.encode()).hexdigest()
        with closing(sqlite3.connect(self.path)) as db:
            db.execute(
                "INSERT OR IGNORE INTO prediction_snapshots VALUES(?,?,?,?)",
                (snapshot_id, snapshot.match_id, snapshot.prediction_time, payload),
            )
            db.commit()
        return snapshot_id

    def snapshot(self, snapshot_id: str) -> dict | None:
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute("SELECT payload_json FROM prediction_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def attribute(self, snapshot_id: str, outcome: int, *, closing_probability: float | None = None,
                  feature_failures: tuple[str, ...] = (), ai_failures: tuple[str, ...] = (),
                  unexpected_events: tuple[str, ...] = ()) -> PostmatchAttribution:
        snapshot = self.snapshot(snapshot_id)
        if snapshot is None:
            raise KeyError(snapshot_id)
        probability = min(.999999, max(.000001, float(snapshot["final_probability"])))
        probability_error = abs(float(outcome) - probability)
        correct = (probability >= .5) == bool(outcome)
        errors = []
        if probability_error >= .35:
            errors.append("MODEL_OVERCONFIDENCE")
        if "roster" in " ".join(feature_failures).casefold():
            errors.append("ROSTER_MISS")
        attribution = PostmatchAttribution(
            snapshot_id, correct, tuple(errors), probability_error,
            (probability - outcome) ** 2,
            -(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)),
            (closing_probability - float(snapshot.get("market_fair_probability")))
            if closing_probability is not None and snapshot.get("market_fair_probability") is not None else None,
            feature_failures, ai_failures, unexpected_events,
        )
        with closing(sqlite3.connect(self.path)) as db:
            db.execute(
                "INSERT OR IGNORE INTO postmatch_attributions VALUES(?,?,?)",
                (snapshot_id, datetime.now(timezone.utc).isoformat(), json.dumps(asdict(attribution))),
            )
            db.commit()
        return attribution


@dataclass(frozen=True)
class CandidateEvaluation:
    sample_size: int
    brier: float
    calibration_error: float
    log_loss: float
    roi: float
    backtest_completed: bool


def candidate_can_promote(production: CandidateEvaluation, candidate: CandidateEvaluation,
                          minimum_samples: int = 100) -> bool:
    return bool(
        candidate.backtest_completed and candidate.sample_size >= minimum_samples
        and candidate.brier < production.brier
        and candidate.calibration_error < production.calibration_error
        and candidate.log_loss <= production.log_loss
        and candidate.roi >= 0
    )


def collect_feedback_only(production_model: object, _attribution: PostmatchAttribution) -> object:
    """Feedback collection is deliberately side-effect free for the production model."""
    return production_model
