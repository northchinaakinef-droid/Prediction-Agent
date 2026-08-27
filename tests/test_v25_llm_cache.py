from datetime import datetime, timedelta, timezone
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest
from unittest.mock import patch

from prediction_agent.ai.analyst import EvidenceAnalyst
from prediction_agent.ai.cache import AnalysisCache, cache_key, semantic_evidence_fingerprint
from prediction_agent.ai.schemas import AIAnalysis, Evidence


def evidence(status="AVAILABLE_FRESH", value=None, observed=None):
    return Evidence.validate_payload({
        "evidence_id": "e1", "match_id": "m1", "evidence_type": "FACT",
        "source": "unit", "observed_at": (observed or datetime.now(timezone.utc)).isoformat(),
        "published_at": "2026-01-01T00:00:00+00:00",
        "payload": {"status": status, "freshness_seconds": 10, "value": value or {"lineup": ["a"]}},
        "reliability_score": .9, "freshness_score": 1 if status == "AVAILABLE_FRESH" else 0,
    })


class FakeClient:
    provider = "fake"
    model = "v1"
    def __init__(self): self.calls = 0
    def analyze(self, _system, _user, schema):
        self.calls += 1
        return schema.validate_payload({"match_id": "m1", "used_evidence_ids": ["e1"],
            "factors": [], "adjustment": .01, "confidence": .8,
            "risk_flags": [], "unknowns": [], "summary": "ok"})


class V22CacheTests(unittest.TestCase):
    def test_scan_clock_changes_do_not_change_semantic_fingerprint(self):
        first = evidence(observed=datetime(2026, 1, 1, tzinfo=timezone.utc))
        second = evidence(observed=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=5))
        second.payload["freshness_seconds"] = 310
        self.assertEqual(semantic_evidence_fingerprint([first]), semantic_evidence_fingerprint([second]))

    def test_status_and_value_changes_force_miss(self):
        fresh = evidence()
        stale = evidence("AVAILABLE_STALE")
        changed = evidence(value={"lineup": ["b"]})
        self.assertNotEqual(semantic_evidence_fingerprint([fresh]), semantic_evidence_fingerprint([stale]))
        self.assertNotEqual(semantic_evidence_fingerprint([fresh]), semantic_evidence_fingerprint([changed]))

    def test_cache_key_isolates_baseline_provider_and_model(self):
        base = cache_key("m", "h", "p", .60, "openai", "a")
        self.assertNotEqual(base, cache_key("m", "h", "p", .61, "openai", "a"))
        self.assertNotEqual(base, cache_key("m", "h", "p", .60, "anthropic", "a"))
        self.assertNotEqual(base, cache_key("m", "h", "p", .60, "openai", "b"))

    def test_second_identical_analysis_is_cache_hit_without_request(self):
        with TemporaryDirectory() as tmp:
            client = FakeClient()
            analyst = EvidenceAnalyst(client, AnalysisCache(Path(tmp) / "paper.db"))
            with patch.dict("os.environ", {"LLM_MIN_AVAILABLE_EVIDENCE": "1"}):
                first = analyst.analyze(match_id="m1", baseline_probability=.60, evidence=[evidence()])
                second = analyst.analyze(match_id="m1", baseline_probability=.60, evidence=[evidence()])
            self.assertTrue(first.success)
            self.assertTrue(second.cache_hit)
            self.assertFalse(second.attempted)
            self.assertEqual(client.calls, 1)


if __name__ == "__main__": unittest.main()
