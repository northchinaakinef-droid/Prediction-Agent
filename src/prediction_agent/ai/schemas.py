from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, Field


class SchemaError(ValueError):
    pass


class Evidence(BaseModel):
    evidence_id: str = Field(validation_alias=AliasChoices("evidence_id", "source_id"))
    match_id: str = "UNKNOWN"
    evidence_type: str = Field(validation_alias=AliasChoices("evidence_type", "source_type"))
    source: str = "unknown"
    observed_at: datetime
    published_at: datetime | None = None
    team: str | None = None
    player: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    reliability_score: float = Field(ge=0, le=1)
    freshness_score: float = Field(default=1.0, ge=0, le=1)

    @property
    def source_id(self) -> str:
        return self.evidence_id

    @property
    def source_type(self) -> str:
        return self.evidence_type

    @classmethod
    def validate_payload(cls, payload: dict) -> "Evidence":
        row = dict(payload)
        evidence_id = row.pop("source_id", None) or row.get("evidence_id")
        evidence_type = row.pop("source_type", None) or row.get("evidence_type")
        content = row.pop("content", None)
        normalized = {
            **row,
            "evidence_id": evidence_id,
            "match_id": row.get("match_id") or "UNKNOWN",
            "evidence_type": evidence_type,
            "source": row.get("source") or str(evidence_type or "unknown").casefold(),
            "payload": row.get("payload") or ({"content": content} if content is not None else {}),
            "freshness_score": row.get("freshness_score", 1.0),
        }
        try:
            return cls.model_validate(normalized)
        except Exception as error:
            raise SchemaError(str(error)) from error

    def as_dict(self) -> dict:
        return self.model_dump(mode="json")


class AnalysisFactor(BaseModel):
    evidence_id: str
    factor_type: str
    impact: float = Field(ge=-0.05, le=0.05)
    reason: str


class AIAnalysis(BaseModel):
    match_id: str
    used_evidence_ids: list[str]
    factors: list[AnalysisFactor]
    adjustment: float = Field(ge=-0.05, le=0.05)
    confidence: float = Field(ge=0, le=1)
    risk_flags: list[str]
    unknowns: list[str]
    summary: str

    @classmethod
    def validate_payload(cls, payload: dict) -> "AIAnalysis":
        row = dict(payload)
        if "used_evidence_ids" not in row:
            findings = row.pop("evidence", [])
            row["used_evidence_ids"] = [item.get("source_id") for item in findings
                                         if isinstance(item, dict) and item.get("source_id")]
        if isinstance(row.get("factors"), dict):
            fallback_id = (row.get("used_evidence_ids") or ["UNKNOWN"])[0]
            row["factors"] = [
                {"evidence_id": fallback_id, "factor_type": key,
                 "impact": max(-.05, min(.05, float(value))), "reason": key}
                for key, value in row["factors"].items()
            ]
        row.setdefault("adjustment", sum(float(item.get("impact", 0)) for item in row.get("factors", [])))
        try:
            return cls.model_validate(row)
        except Exception as error:
            raise SchemaError(str(error)) from error

    def as_dict(self) -> dict:
        return self.model_dump(mode="json")
