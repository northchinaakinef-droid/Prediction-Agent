from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone
from tempfile import TemporaryDirectory
from pathlib import Path
import unittest

from prediction_agent.ai.analyst import EvidenceAnalyst
from prediction_agent.ai.schemas import AIAnalysis, Evidence, SchemaError
from prediction_agent.ai.validator import capped_adjustment, validate_evidence_references
from prediction_agent.coverage_service import CoverageStore, build_coverage_report
from prediction_agent.feedback import (
    CandidateEvaluation, FeedbackStore, PredictionSnapshot, candidate_can_promote,
    collect_feedback_only,
)
from prediction_agent.model_registry import ModelRegistry
from prediction_agent.notifications import AlertCategory, NotificationOutbox, OutboxWorker
from prediction_agent.roster import RosterConfidence, RosterStatus, historical_roster, lineup_changed


def match(match_id="m1", market="MARKET_NOT_FOUND"):
    return {"match_id": match_id, "sport": "lol", "league": "LCK", "market_mapping_status": market}


class CoverageV2Tests(unittest.TestCase):
    def test_schedule_event_without_market_is_not_dropped(self):
        report = build_coverage_report("2026-08-23", [match()], [])
        self.assertEqual(report.total_schedule_events, 1)
        self.assertEqual(report.no_market_events, 1)
        self.assertEqual(report.events[0].skip_reason, "NO_MARKET")

    def test_every_schedule_event_has_analysis_status(self):
        report = build_coverage_report("2026-08-23", [match("a"), match("b", "MATCHED")], [])
        self.assertTrue(all(row.analysis_status for row in report.events))

    def test_skip_reason_is_persisted(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "coverage.db"
            report = build_coverage_report("2026-08-23", [match()], [])
            store = CoverageStore(path)
            store.save(report)
            self.assertEqual(store.event("2026-08-23", "m1")["skip_reason"], "NO_MARKET")


class NotificationV2Tests(unittest.TestCase):
    def test_failed_feishu_notification_retries(self):
        with TemporaryDirectory() as directory:
            outbox = NotificationOutbox(Path(directory) / "outbox.db")
            outbox.enqueue(event_id="m1", category=AlertCategory.PREMATCH_ANALYSIS.value,
                           payload={"message": "x"}, dedupe_key="m1:pre:1")
            worker = OutboxWorker(outbox, lambda _row: (_ for _ in ()).throw(RuntimeError("Feishu down")))
            self.assertEqual(worker.run_once()["failed"], 1)
            self.assertEqual(outbox.status_counts()["FAILED"], 1)
            with outbox.connect() as db:
                db.execute("UPDATE notification_outbox SET next_retry_at='2000-01-01T00:00:00+00:00'")
            sent = []
            self.assertEqual(OutboxWorker(outbox, sent.append).run_once()["sent"], 1)
            self.assertEqual(len(sent), 1)

    def test_alert_is_not_marked_sent_before_success(self):
        with TemporaryDirectory() as directory:
            outbox = NotificationOutbox(Path(directory) / "outbox.db")
            outbox.enqueue(event_id="m1", category="MAJOR_EVENT", payload={}, dedupe_key="k")
            OutboxWorker(outbox, lambda _row: (_ for _ in ()).throw(RuntimeError("no"))).run_once()
            self.assertEqual(outbox.status_counts()["SENT"], 0)

    def test_daily_report_retry(self):
        with TemporaryDirectory() as directory:
            outbox = NotificationOutbox(Path(directory) / "outbox.db")
            outbox.enqueue_daily("2026-08-23", {"report_date": "2026-08-23"})
            OutboxWorker(outbox, lambda _row: (_ for _ in ()).throw(RuntimeError("no"))).run_once()
            with outbox.connect() as db:
                row = db.execute("SELECT send_status,retry_count FROM daily_report_runs").fetchone()
            self.assertEqual((row[0], row[1]), ("FAILED", 1))

    def test_outbox_dedupe(self):
        with TemporaryDirectory() as directory:
            outbox = NotificationOutbox(Path(directory) / "outbox.db")
            first = outbox.enqueue(event_id="m1", category="MARKET_MOVE", payload={}, dedupe_key="same")
            second = outbox.enqueue(event_id="m1", category="MARKET_MOVE", payload={}, dedupe_key="same")
            self.assertEqual(first, second)
            self.assertEqual(len(outbox.due()), 1)


class RosterV2Tests(unittest.TestCase):
    def test_historical_roster_not_marked_confirmed(self):
        roster = historical_roster(("a", "b", "c", "d", "e"))
        self.assertEqual(roster.status, RosterConfidence.HISTORICAL)
        self.assertNotEqual(roster.status, RosterConfidence.CONFIRMED)

    def test_lineup_change_triggers_reanalysis(self):
        old = RosterStatus(("a", "b"), RosterConfidence.EXPECTED, "x", None)
        new = RosterStatus(("a", "c"), RosterConfidence.CONFIRMED, "official", None)
        self.assertTrue(lineup_changed(old, new))


class FakeClient:
    def __init__(self, payload=None, error=None):
        self.payload, self.error = payload, error

    def analyze(self, _system, _user, schema):
        if self.error:
            raise self.error
        return schema.validate_payload(self.payload)


def evidence(source_id="e1"):
    return Evidence.validate_payload({
        "source_id": source_id, "source_type": "OFFICIAL", "content": "confirmed",
        "observed_at": "2026-08-23T00:00:00+00:00", "published_at": None, "reliability_score": .9,
    })


def analysis_payload(source_id="e1", factor=.02):
    return {"match_id": "m1", "evidence": [{"type": "ROSTER", "team": "A", "description": "confirmed",
            "impact": factor, "confidence": .9, "source_id": source_id}],
            "factors": {"roster_impact": factor}, "risk_flags": [], "confidence": .8,
            "summary": "official roster confirmed", "unknowns": []}


class AIV2Tests(unittest.TestCase):
    def test_ai_output_requires_schema(self):
        with self.assertRaises(SchemaError):
            AIAnalysis.validate_payload({"summary": "missing everything"})

    def test_ai_cannot_reference_unknown_evidence(self):
        analysis = AIAnalysis.validate_payload(analysis_payload("invented"))
        with self.assertRaises(SchemaError):
            validate_evidence_references(analysis, {"e1"})

    def test_ai_adjustment_capped(self):
        analysis = AIAnalysis.validate_payload(analysis_payload(factor=.15))
        self.assertEqual(capped_adjustment(analysis), .05)

    def test_llm_failure_falls_back_to_quant(self):
        result = EvidenceAnalyst(FakeClient(error=RuntimeError("down"))).analyze(
            match_id="m1", baseline_probability=.61, evidence=[evidence()],
        )
        self.assertEqual(result.status, "QUANT_FALLBACK")
        self.assertEqual(result.final_probability, .61)


def snapshot() -> PredictionSnapshot:
    return PredictionSnapshot(
        "m1", "2026-08-23T00:00:00+00:00", "q1", "p1", None, .6, 0, .6, .55, .05,
        {}, [], [], "WATCH", 0,
    )


class FeedbackV2Tests(unittest.TestCase):
    def test_prediction_snapshot_is_immutable(self):
        row = snapshot()
        with self.assertRaises(FrozenInstanceError):
            row.final_probability = .9

    def test_postmatch_attribution_created(self):
        with TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "feedback.db")
            snapshot_id = store.save_snapshot(snapshot())
            result = store.attribute(snapshot_id, 0)
            self.assertGreater(result.brier_score, 0)
            self.assertGreater(result.log_loss, 0)

    def test_feedback_does_not_mutate_production_model(self):
        model = {"rating": 1500}
        self.assertIs(collect_feedback_only(model, None), model)
        self.assertEqual(model["rating"], 1500)

    def test_candidate_model_requires_backtest_before_promotion(self):
        production = CandidateEvaluation(200, .22, .08, .65, .01, True)
        candidate = CandidateEvaluation(200, .20, .06, .64, .02, False)
        self.assertFalse(candidate_can_promote(production, candidate))


class FreshnessV2Tests(unittest.TestCase):
    def test_stale_model_is_reported(self):
        result = ModelRegistry.freshness("2026-01-01", today=date(2026, 8, 23), interval_days=42)
        self.assertEqual(result["status"], "STALE")


if __name__ == "__main__":
    unittest.main()
