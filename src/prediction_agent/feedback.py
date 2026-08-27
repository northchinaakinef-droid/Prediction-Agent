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
    ai_status: str = "QUANT_FALLBACK"
    semantic_evidence_hash: str = ""
    cache_hit: bool = False
    decision_window: str | None = None
    canonical_sample_key: str | None = None
    final_action: str = "NO_BET"
    final_stake: float = 0.0
    quant_action: str = "NO_BET"
    quant_stake: float = 0.0
    quant_ev: float | None = None
    shadow_final_action: str = "NO_BET"
    shadow_final_stake: float = 0.0
    shadow_final_ev: float | None = None
    llm_decision_mode: str = "ACTIVE"


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
    quant_brier_score: float | None = None
    final_brier_score: float | None = None
    quant_log_loss: float | None = None
    final_log_loss: float | None = None
    calibration_delta: float | None = None
    sport: str | None = None
    canonical_sample_key: str | None = None
    quant_action: str | None = None
    shadow_final_action: str | None = None
    quant_ev: float | None = None
    final_ev: float | None = None
    quant_stake: float = 0.0
    shadow_final_stake: float = 0.0
    llm_adjustment: float = 0.0
    llm_provider: str | None = None
    llm_model: str | None = None


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
        baseline = min(.999999, max(.000001, float(snapshot.get("baseline_probability", probability))))
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
            (baseline - outcome) ** 2, (probability - outcome) ** 2,
            -(outcome * math.log(baseline) + (1 - outcome) * math.log(1 - baseline)),
            -(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)),
            abs(probability - outcome) - abs(baseline - outcome),
            (snapshot.get("features_json") or {}).get("sport"),
            snapshot.get("canonical_sample_key"), snapshot.get("quant_action"),
            snapshot.get("shadow_final_action"), snapshot.get("quant_ev"),
            snapshot.get("shadow_final_ev"), float(snapshot.get("quant_stake") or 0),
            float(snapshot.get("shadow_final_stake") or 0),
            float(snapshot.get("ai_adjustment") or 0), snapshot.get("llm_provider"),
            snapshot.get("ai_model_name"),
        )
        with closing(sqlite3.connect(self.path)) as db:
            db.execute(
                "INSERT OR IGNORE INTO postmatch_attributions VALUES(?,?,?)",
                (snapshot_id, datetime.now(timezone.utc).isoformat(), json.dumps(asdict(attribution))),
            )
            db.commit()
        return attribution

    def shadow_attribution_summary(self) -> dict:
        """Aggregate canonical Quant-vs-Final probability quality overall/by sport."""
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute(
                "SELECT created_at,payload_json FROM postmatch_attributions ORDER BY created_at"
            ).fetchall()
        canonical: dict[str, dict] = {}
        for _created_at, payload_json in rows:
            payload = json.loads(payload_json)
            key = str(payload.get("canonical_sample_key") or payload.get("snapshot_id"))
            canonical[key] = payload

        def aggregate(items: list[dict]) -> dict:
            if not items:
                return {"samples": 0, "quant_brier": None, "final_brier": None,
                        "quant_log_loss": None, "final_log_loss": None,
                        "brier_delta": None, "log_loss_delta": None}
            def mean(field: str) -> float | None:
                values = [float(item[field]) for item in items if item.get(field) is not None]
                return sum(values) / len(values) if values else None
            quant_brier, final_brier = mean("quant_brier_score"), mean("final_brier_score")
            quant_log, final_log = mean("quant_log_loss"), mean("final_log_loss")
            return {"samples": len(items), "quant_brier": quant_brier,
                    "final_brier": final_brier, "quant_log_loss": quant_log,
                    "final_log_loss": final_log,
                    "brier_delta": (final_brier - quant_brier)
                    if final_brier is not None and quant_brier is not None else None,
                    "log_loss_delta": (final_log - quant_log)
                    if final_log is not None and quant_log is not None else None,
                    "quant_bets": sum(item.get("quant_action") == "BET" for item in items),
                    "shadow_final_bets": sum(item.get("shadow_final_action") == "BET" for item in items)}
        values = list(canonical.values())
        by_sport = {sport: aggregate([item for item in values if item.get("sport") == sport])
                    for sport in ("lol", "cs2", "nba")}
        return {"overall": aggregate(values), "by_sport": by_sport}


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
