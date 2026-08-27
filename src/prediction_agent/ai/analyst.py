from __future__ import annotations

import json
import os
import socket
from dataclasses import dataclass
from urllib.error import HTTPError, URLError

from .cache import ANALYST_VERSION, AnalysisCache, cache_key, semantic_evidence_fingerprint
from .client import LLMClient
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT
from .schemas import AIAnalysis, Evidence, SchemaError
from .validator import capped_adjustment, validate_evidence_references


@dataclass(frozen=True)
class AnalystResult:
    analysis: AIAnalysis | None
    adjustment: float
    final_probability: float
    status: str
    error: str | None = None
    cache_hit: bool = False
    adjustment_status: str = "QUANT_FALLBACK"
    error_category: str | None = None
    attempted: bool = False
    success: bool = False
    provider: str | None = None
    model: str | None = None
    semantic_evidence_hash: str = ""


def _error_category(error: Exception) -> str:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return "TIMEOUT"
    if isinstance(error, HTTPError):
        return "HTTP_ERROR"
    if isinstance(error, URLError):
        return "TIMEOUT" if isinstance(error.reason, (TimeoutError, socket.timeout)) else "HTTP_ERROR"
    if isinstance(error, ValueError) and "evidence" in str(error).casefold():
        return "EVIDENCE_REFERENCE_ERROR"
    if isinstance(error, SchemaError):
        return "SCHEMA_VALIDATION_ERROR"
    return "UNKNOWN_ERROR"


class EvidenceAnalyst:
    def __init__(self, client: LLMClient | None, cache: AnalysisCache | None = None):
        self.client, self.cache = client, cache

    def analyze(self, *, match_id: str, baseline_probability: float,
                evidence: list[Evidence]) -> AnalystResult:
        if self.client is None:
            return AnalystResult(None, 0.0, baseline_probability, "LLM_NOT_ACTIVE", "LLM not configured",
                                 error_category="NOT_CONFIGURED")
        provider = str(getattr(self.client, "provider", self.client.__class__.__name__)).casefold()
        model = str(getattr(self.client, "model", "unknown"))
        gate = minimum_evidence_gate(evidence)
        available = [row for row in evidence if row.evidence_type in gate["available_types"]]
        if gate["structured"] and not gate["eligible"]:
            return AnalystResult(None, 0.0, baseline_probability, "QUANT_FALLBACK",
                                 f"MINIMUM_EVIDENCE_GATE:{gate['gate_name']}:{len(available)}/"
                                 f"{gate['minimum_count']}:missing={gate['missing']}",
                                 error_category="MINIMUM_EVIDENCE_GATE", provider=provider, model=model)
        rows = [row.as_dict() for row in available]
        unknown_types = [row.evidence_type for row in evidence if row not in available]
        evidence_hash = semantic_evidence_fingerprint(evidence)
        key = cache_key(match_id, evidence_hash, PROMPT_VERSION, baseline_probability,
                        provider, model, ANALYST_VERSION)
        attempted = False
        try:
            cached = self.cache.get(key) if self.cache else None
            if cached:
                analysis = AIAnalysis.validate_payload(cached)
            else:
                attempted = True
                last_error = None
                for _attempt in range(2):
                    try:
                        analysis = self.client.analyze(
                            SYSTEM_PROMPT,
                            json.dumps({"match_id": match_id, "quant_probability": baseline_probability,
                                        "available_evidence": rows,
                                        "unknown_evidence_types": unknown_types,
                                        "prompt_version": PROMPT_VERSION}, default=str),
                            AIAnalysis,
                        )
                        validate_evidence_references(analysis, {row.source_id for row in available})
                        break
                    except Exception as error:
                        last_error = error
                else:
                    raise last_error or RuntimeError("LLM validation failed")
            validate_evidence_references(analysis, {row.source_id for row in available})
            adjustment = capped_adjustment(analysis)
            final = max(.001, min(.999, baseline_probability + adjustment))
            if self.cache and not cached:
                self.cache.put(key, analysis.as_dict())
            adjustment_status = "AI_ADJUSTMENT_ZERO_BY_MODEL" if adjustment == 0 else "AI_ADJUSTMENT_APPLIED"
            return AnalystResult(analysis, adjustment, final, "LLM_ACTIVE", cache_hit=bool(cached),
                                 adjustment_status=adjustment_status, attempted=attempted, success=True,
                                 provider=provider, model=model, semantic_evidence_hash=evidence_hash)
        except Exception as error:
            return AnalystResult(None, 0.0, baseline_probability, "QUANT_FALLBACK", repr(error),
                                 error_category=_error_category(error), attempted=attempted,
                                 provider=provider, model=model,
                                 semantic_evidence_hash=evidence_hash)


def minimum_evidence_gate(evidence: list[Evidence]) -> dict:
    """Evaluate the existing minimum gate without contacting an LLM."""
    structured = any("status" in row.payload for row in evidence)
    available = [row for row in evidence if str(row.payload.get("status") or "AVAILABLE_FRESH").upper()
                 in {"AVAILABLE", "AVAILABLE_FRESH"}]
    available_types = {row.evidence_type for row in available}
    all_types = {row.evidence_type for row in evidence}
    alternatives: tuple[str, ...] = ()
    if "INJURY" in all_types:
        required = {"SCHEDULE", "MARKET", "TEAM_STRENGTH", "RECENT_FORM"}
        alternatives = ("INJURY", "PLAYER_AVAILABILITY")
        alternative_ok = bool(available_types & set(alternatives))
        gate_name = "NBA"
    elif "MAP_POOL" in all_types:
        required = {"SCHEDULE", "MARKET", "TEAM_RATING", "RECENT_FORM", "ROSTER"}
        alternative_ok = True; gate_name = "CS2"
    elif "PATCH" in all_types:
        required = {"SCHEDULE", "MARKET", "TEAM_STRENGTH", "RECENT_FORM", "ROSTER"}
        alternative_ok = True; gate_name = "LOL"
    else:
        required = set(); alternative_ok = True; gate_name = "GENERIC"
    minimum = int(os.getenv("LLM_MIN_AVAILABLE_EVIDENCE", "3"))
    missing = sorted(required - available_types)
    if alternatives and not alternative_ok:
        missing.append("INJURY_OR_PLAYER_AVAILABILITY")
    sport_gate = not missing
    eligible = not structured or (len(available) >= minimum and sport_gate)
    return {"gate_name": gate_name, "structured": structured,
            "minimum_count": minimum, "required_types": sorted(required),
            "alternative_types": list(alternatives), "available_types": sorted(available_types),
            "missing": missing, "available_count": len(available), "eligible": eligible}
