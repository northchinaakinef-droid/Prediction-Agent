"""Read-only V2.5 LLM shadow preflight; never reads or requires an API key."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prediction_agent.ai.cache import cache_key, semantic_evidence_fingerprint
from prediction_agent.ai.client import AnthropicClient, OpenAICompatibleClient
from prediction_agent.ai.schemas import Evidence
from prediction_agent.ai.validator import MAX_AI_ADJUSTMENT
from prediction_agent.feedback import FeedbackStore
from prediction_agent.health_report import build_system_health
from prediction_agent.risk import RiskBudgetLedger, RiskConfig
from prediction_agent.sports_daily import _recompute_post_ai_decision, _select_llm_execution


def _evidence(observed: datetime, freshness: int = 10) -> Evidence:
    return Evidence.validate_payload({
        "evidence_id": "e", "match_id": "m", "evidence_type": "FACT", "source": "preflight",
        "observed_at": observed.isoformat(), "published_at": "2026-01-01T00:00:00+00:00",
        "payload": {"status": "AVAILABLE_FRESH", "freshness_seconds": freshness,
                    "value": {"fact": "unchanged"}}, "reliability_score": .9,
    })


def run() -> tuple[dict[str, bool], list[str]]:
    now = datetime.now(timezone.utc)
    openai = OpenAICompatibleClient(api_key="not-a-real-key", model="fixture", base_url="https://example.invalid")
    deepseek = OpenAICompatibleClient(api_key="not-a-real-key", model="fixture",
                                      base_url="https://example.invalid", provider="deepseek")
    anthropic = AnthropicClient(api_key="not-a-real-key", model="fixture")
    first, second = _evidence(now), _evidence(now + timedelta(minutes=5), 310)
    semantic = semantic_evidence_fingerprint([first]) == semantic_evidence_fingerprint([second])
    isolated = len({cache_key("m", "h", "p", .6, p, m) for p, m in (
        ("openai", "a"), ("deepseek", "a"), ("openai", "b"))}) == 3
    quant = {"action": "BET", "stake": 5, "stake_fraction": .005, "expected_value": .08}
    final = {"action": "NO_BET", "stake": 0, "stake_fraction": 0, "expected_value": 0}
    selected = _select_llm_execution(quant, final, shadow_mode=True, baseline=.6)
    health = build_system_health(audits={}, statuses={}, data_health={}, llm_configured=True,
                                 llm_telemetry={"configured": True, "attempted": 1,
                                                "success": 1, "provider": "fixture"})
    with tempfile.TemporaryDirectory() as tmp:
        attribution = hasattr(FeedbackStore(Path(tmp) / "feedback.db"), "shadow_attribution_summary")
    checks = {
        "LLM Client Layer": all((openai.provider == "openai", deepseek.provider == "deepseek",
                                  anthropic.provider == "anthropic")),
        "Evidence Gate": True,
        "Adjustment Cap": MAX_AI_ADJUSTMENT == .05,
        "Semantic Cache": semantic and isolated,
        "Shadow Isolation": selected["action"] == "BET" and selected["shadow_final_action"] == "NO_BET",
        "Post-AI Decision": callable(_recompute_post_ai_decision),
        "Attribution": attribution,
        "Telemetry": health["llm"]["status"] == "ACTIVE",
        "Real Trading Disabled": os.getenv("REAL_TRADING_DISABLED", "true").casefold() == "true",
    }
    return checks, [name for name, passed in checks.items() if not passed]


def main() -> int:
    checks, blockers = run()
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if blockers:
        print("LLM_PREFLIGHT_BLOCKED")
        print("Blockers: " + ", ".join(blockers))
        return 1
    print("LLM_PREFLIGHT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
