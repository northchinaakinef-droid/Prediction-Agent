from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from .cache import AnalysisCache, cache_key
from .client import LLMClient
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT
from .schemas import AIAnalysis, Evidence
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


class EvidenceAnalyst:
    def __init__(self, client: LLMClient | None, cache: AnalysisCache | None = None):
        self.client, self.cache = client, cache

    def analyze(self, *, match_id: str, baseline_probability: float,
                evidence: list[Evidence]) -> AnalystResult:
        if self.client is None:
            return AnalystResult(None, 0.0, baseline_probability, "LLM_NOT_ACTIVE", "LLM not configured")
        structured = any("status" in row.payload for row in evidence)
        available = [row for row in evidence if str(row.payload.get("status") or "AVAILABLE_FRESH").upper()
                     in {"AVAILABLE", "AVAILABLE_FRESH"}]
        available_types = {row.evidence_type for row in available}
        all_types = {row.evidence_type for row in evidence}
        if "INJURY" in all_types:
            required = {"SCHEDULE", "MARKET", "TEAM_STRENGTH", "RECENT_FORM"}
            alternative_ok = bool(available_types & {"INJURY", "PLAYER_AVAILABILITY"})
            sport_gate = required <= available_types and alternative_ok
            gate_name = "NBA"
        elif "MAP_POOL" in all_types:
            required = {"SCHEDULE", "MARKET", "TEAM_RATING", "RECENT_FORM", "ROSTER"}
            sport_gate = required <= available_types; gate_name = "CS2"
        elif "PATCH" in all_types:
            required = {"SCHEDULE", "MARKET", "TEAM_STRENGTH", "RECENT_FORM", "ROSTER"}
            sport_gate = required <= available_types; gate_name = "LOL"
        else:
            required = set(); sport_gate = True; gate_name = "GENERIC"
        minimum = int(os.getenv("LLM_MIN_AVAILABLE_EVIDENCE", "3"))
        if structured and (len(available) < minimum or not sport_gate):
            missing = sorted(required - available_types)
            return AnalystResult(None, 0.0, baseline_probability, "QUANT_FALLBACK",
                                 f"MINIMUM_EVIDENCE_GATE:{gate_name}:{len(available)}/{minimum}:missing={missing}")
        rows = [row.as_dict() for row in available]
        unknown_types = [row.evidence_type for row in evidence if row not in available]
        evidence_hash = hashlib.sha256(json.dumps(
            {"available": rows, "unknown_types": unknown_types}, sort_keys=True, default=str
        ).encode()).hexdigest()
        key = cache_key(match_id, evidence_hash, PROMPT_VERSION)
        try:
            cached = self.cache.get(key) if self.cache else None
            if cached:
                analysis = AIAnalysis.validate_payload(cached)
            else:
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
                                 adjustment_status=adjustment_status)
        except Exception as error:
            return AnalystResult(None, 0.0, baseline_probability, "QUANT_FALLBACK", repr(error))
