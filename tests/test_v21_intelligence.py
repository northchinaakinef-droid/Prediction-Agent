from datetime import datetime, timezone

import pytest

from prediction_agent.ai.analyst import EvidenceAnalyst
from prediction_agent.ai.schemas import Evidence
from prediction_agent.evidence_pipeline import EvidenceStore, build_match_evidence, evidence_availability
from prediction_agent.risk import RiskConfig, binary_share_math, kelly_fraction, recommend
from prediction_agent.sports_daily import _apply_intelligence_gate, _attach_risk_audit


def test_evidence_store_is_structured_and_traceable(tmp_path):
    evidence = build_match_evidence({"market_probability": .55, "execution_price": .56,
                                     "lineup_status": "HISTORICAL"}, "lol", "m1",
                                    datetime.now(timezone.utc))
    store = EvidenceStore(tmp_path / "evidence.db")
    store.put_many(evidence)
    assert len(evidence) == 11
    assert {item.evidence_type for item in evidence} >= {"ROSTER", "RECENT_FORM", "PATCH", "MARKET"}
    assert len(store.for_match("m1")) == len(evidence)
    availability = evidence_availability(evidence)
    assert availability["evidence_total_count"] == 11
    assert availability["evidence_available_count"] < availability["evidence_total_count"]
    assert availability["evidence_unknown_count"] > 0


def test_critical_missing_evidence_forces_no_bet():
    row = {"action": "BET", "stake": 75, "stake_fraction": .0075, "edge": .15,
           "data_quality_score": .7, "data_quality_missing": ["RECENT_FORM"], "reasons": []}
    _apply_intelligence_gate(row, "cs2")
    assert row["action"] == "NO_BET" and row["stake"] == 0 and row["recommendation_state"] == "WATCH"


@pytest.mark.parametrize("probability,price", [(.01, .99), (.5, .999999), (.999, .999999), (.51, .5)])
def test_risk_math_boundaries_are_finite(probability, price):
    row = {"model_probability": probability, "final_probability": probability,
           "market_probability": price, "execution_price": price, "edge": probability-price,
           "data_quality_score": 1, "ai_adjustment": 0, "ai_status": "LLM_NOT_ACTIVE",
           "stake": 0, "decision_gate": "PASS"}
    _attach_risk_audit(row, 10_000, RiskConfig())
    assert all(abs(float(row["risk_calculation"][key])) < 1e12
               for key in ("adjusted_probability", "expected_profit_per_share",
                           "expected_roi_on_capital", "kelly_fraction"))


def test_edge_and_confidence_change_uncapped_kelly():
    low = recommend(event_id="a", outcome="A", model_probability=.53, decimal_odds=2,
                    bankroll=10_000, confidence=.7, max_bet_fraction=1, trading_enabled=True,
                    spread=.01, available_size=1_000_000)
    high = recommend(event_id="b", outcome="A", model_probability=.65, decimal_odds=2,
                     bankroll=10_000, confidence=.9, max_bet_fraction=1, trading_enabled=True,
                     spread=.01, available_size=1_000_000)
    assert high.stake > low.stake
    assert kelly_fraction(.999, 1.000001) >= 0


def test_every_penalty_changes_probability_edge_ev_and_stake_before_cap():
    base = {"model_probability": .65, "final_probability": .65, "market_probability": .5,
            "execution_price": .5, "edge": .15, "data_quality_score": 1,
            "ai_adjustment": 0, "ai_confidence": 1, "ai_status": "LLM_ACTIVE",
            "confidence": 1, "stake": 1_000, "decision_gate": "PASS"}
    no_penalty = dict(base); _attach_risk_audit(no_penalty, 10_000, RiskConfig(max_bet_fraction=1))
    uncertainty = dict(base, confidence=.5); _attach_risk_audit(uncertainty, 10_000, RiskConfig(max_bet_fraction=1))
    quality = dict(base, data_quality_score=.5); _attach_risk_audit(quality, 10_000, RiskConfig(max_bet_fraction=1))
    for penalized in (uncertainty, quality):
        assert penalized["risk_calculation"]["adjusted_probability"] < no_penalty["risk_calculation"]["adjusted_probability"]
        assert penalized["adjusted_edge"] < no_penalty["adjusted_edge"]
        assert penalized["expected_roi_on_capital"] < no_penalty["expected_roi_on_capital"]
        assert penalized["risk_calculation"]["stake_before_cap"] < no_penalty["risk_calculation"]["stake_before_cap"]
    ai_full = dict(base, final_probability=.67, ai_adjustment=.02, ai_confidence=1, edge=.17)
    ai_penalized = dict(ai_full, ai_confidence=.5)
    _attach_risk_audit(ai_full, 10_000, RiskConfig(max_bet_fraction=1))
    _attach_risk_audit(ai_penalized, 10_000, RiskConfig(max_bet_fraction=1))
    for key in ("adjusted_probability", "expected_profit_per_share", "expected_roi_on_capital"):
        assert ai_penalized["risk_calculation"].get(key, ai_penalized.get(key)) < ai_full["risk_calculation"].get(key, ai_full.get(key))
    assert ai_penalized["risk_calculation"]["stake_before_cap"] < ai_full["risk_calculation"]["stake_before_cap"]


def test_uncertainty_penalty_is_in_adjusted_probability_formula():
    row = {"model_probability": .62, "final_probability": .62, "market_probability": .45,
           "execution_price": .46, "edge": .10, "data_quality_score": .8,
           "ai_adjustment": 0, "ai_confidence": 0, "ai_status": "LLM_NOT_ACTIVE",
           "confidence": .5, "stake": 0, "decision_gate": "PASS"}
    _attach_risk_audit(row, 10_000, RiskConfig())
    risk = row["risk_calculation"]
    expected = .62 - risk["data_quality_penalty"] - risk["uncertainty_penalty"] - risk["ai_confidence_penalty"]
    assert risk["adjusted_probability"] == pytest.approx(expected)


@pytest.mark.parametrize("price,profit,roi", [
    (.50, .05, .10), (.54, .01, .01/.54), (.60, -.05, -.05/.60),
    (.999999, .55-.999999, (.55-.999999)/.999999),
    (.000001, .55-.000001, (.55-.000001)/.000001),
])
def test_binary_market_semantics(price, profit, roi):
    result = binary_share_math(.55, price)
    assert result["decimal_odds"] == pytest.approx(1 / price)
    assert result["executable_edge"] == pytest.approx(.55 - price)
    assert result["expected_profit_per_share"] == pytest.approx(profit)
    assert result["expected_roi_on_capital"] == pytest.approx(roi)


class HallucinatingClient:
    def __init__(self): self.calls = 0
    def analyze(self, _system, _user, schema):
        self.calls += 1
        return schema.validate_payload({"match_id": "m1", "used_evidence_ids": ["invented"],
            "factors": [{"evidence_id": "invented", "factor_type": "NEWS", "impact": .01, "reason": "x"}],
            "adjustment": .01, "confidence": .8, "risk_flags": [], "unknowns": [], "summary": "x"})


def test_hallucinated_evidence_retries_then_quant_fallback():
    client = HallucinatingClient()
    evidence = [Evidence.validate_payload({"evidence_id": "real", "match_id": "m1",
        "evidence_type": "MARKET", "source": "test", "observed_at": datetime.now(timezone.utc).isoformat(),
        "payload": {"price": .5}, "reliability_score": 1, "freshness_score": 1})]
    result = EvidenceAnalyst(client).analyze(match_id="m1", baseline_probability=.6, evidence=evidence)
    assert client.calls == 2
    assert result.status == "QUANT_FALLBACK" and result.adjustment == 0 and result.final_probability == .6
