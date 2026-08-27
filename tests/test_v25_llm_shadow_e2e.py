from datetime import datetime, timezone
from unittest.mock import patch
import unittest

from prediction_agent.ai.analyst import EvidenceAnalyst
from prediction_agent.ai.schemas import Evidence
from prediction_agent.risk import RiskBudgetLedger, RiskConfig
from prediction_agent.sports_daily import _recompute_post_ai_decision, _select_llm_execution


def evidence():
    return Evidence.validate_payload({
        "evidence_id": "e1", "match_id": "m", "evidence_type": "FACT", "source": "fixture",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "payload": {"status": "AVAILABLE_FRESH", "value": {"signal": "stable"}},
        "reliability_score": .9,
    })


class FakeClient:
    provider = "fake"
    model = "fixture-v1"

    def __init__(self, adjustment=.05, error=None, evidence_id="e1"):
        self.adjustment, self.error, self.evidence_id, self.calls = adjustment, error, evidence_id, 0

    def analyze(self, _system, _user, schema):
        self.calls += 1
        if self.error:
            raise self.error
        return schema.validate_payload({
            "match_id": "m", "used_evidence_ids": [self.evidence_id], "factors": [],
            "adjustment": self.adjustment, "confidence": 1, "risk_flags": [],
            "unknowns": [], "summary": "fixture",
        })


def candidate(probability):
    ledger = RiskBudgetLedger(RiskConfig(), 1000)
    row = {
        "event_id": "m", "outcome": "A", "model_probability": .60,
        "execution_price": .55, "market_probability": .55, "market_started": False,
        "estimated_cost_rate": 0, "direction_match": True, "market_quality_qualified": True,
        "confidence": 1, "probability_eligible": True, "reasons": [],
    }
    with patch.dict("os.environ", {"PAPER_TRADING_ENABLED": "true"}):
        _recompute_post_ai_decision(row, final_probability=probability, bankroll=1000,
                                    config=RiskConfig(), ledger=ledger, group_key="g")
    return row


class V25LlmShadowE2ETests(unittest.TestCase):
    def test_plus_five_recomputes_final_but_shadow_executes_quant(self):
        result = EvidenceAnalyst(FakeClient(.05)).analyze(
            match_id="m", baseline_probability=.60, evidence=[evidence()] * 3)
        quant, final = candidate(.60), candidate(result.final_probability)
        row = _select_llm_execution(quant, final, shadow_mode=True, baseline=.60)
        self.assertAlmostEqual(result.final_probability, .65)
        self.assertAlmostEqual(row["shadow_final_ev"], .65 / .55 - 1)
        self.assertEqual((row["action"], row["stake"]), (quant["action"], quant["stake"]))
        self.assertEqual(row["shadow_final_action"], final["action"])

    def test_minus_five_can_make_final_no_bet_without_changing_formal_quant(self):
        result = EvidenceAnalyst(FakeClient(-.05)).analyze(
            match_id="m", baseline_probability=.60, evidence=[evidence()] * 3)
        quant, final = candidate(.60), candidate(result.final_probability)
        row = _select_llm_execution(quant, final, shadow_mode=True, baseline=.60)
        self.assertEqual(quant["action"], "BET")
        self.assertEqual(final["action"], "NO_BET")
        self.assertEqual(row["action"], "BET")
        self.assertEqual(row["shadow_final_action"], "NO_BET")

    def test_request_failure_is_safe_quant_fallback(self):
        result = EvidenceAnalyst(FakeClient(error=RuntimeError("offline"))).analyze(
            match_id="m", baseline_probability=.60, evidence=[evidence()] * 3)
        self.assertEqual(result.status, "QUANT_FALLBACK")
        self.assertEqual(result.final_probability, .60)

    def test_unknown_evidence_reference_is_validation_fallback(self):
        result = EvidenceAnalyst(FakeClient(evidence_id="not-supplied")).analyze(
            match_id="m", baseline_probability=.60, evidence=[evidence()] * 3)
        self.assertEqual(result.status, "QUANT_FALLBACK")
        self.assertEqual(result.final_probability, .60)
        self.assertEqual(result.error_category, "EVIDENCE_REFERENCE_ERROR")


if __name__ == "__main__":
    unittest.main()
