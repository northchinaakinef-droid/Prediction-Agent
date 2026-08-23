from __future__ import annotations

from .schemas import AIAnalysis, SchemaError


MAX_AI_ADJUSTMENT = 0.05


def validate_evidence_references(analysis: AIAnalysis, supplied_source_ids: set[str]) -> None:
    referenced = set(analysis.used_evidence_ids)
    referenced.update(factor.evidence_id for factor in analysis.factors)
    unknown = referenced - supplied_source_ids
    if unknown:
        raise SchemaError(f"AI referenced unknown evidence: {sorted(unknown)}")


def capped_adjustment(analysis: AIAnalysis, maximum: float = MAX_AI_ADJUSTMENT) -> float:
    return max(-maximum, min(maximum, float(analysis.adjustment)))
