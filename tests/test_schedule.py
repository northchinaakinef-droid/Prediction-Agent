from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zoneinfo import ZoneInfo

from prediction_agent.schedule import (
    SourceResult, build_schedule_audit, detect_source_mismatches, make_match, match_markets,
    parse_esportagenda, parse_nextmatch, reconcile_sources,
)


class ScheduleTests(unittest.TestCase):
    def test_nextmatch_parser_uses_singapore_calendar_day(self):
        page = '''<div data-slot="card"><time datetime="2026-08-14T08:00:00.000Z">08:00</time>
        <a href="/lck/schedule/">LCK Round 3-4</a><span> T1 vs DK</span> BO3</div>'''
        rows = parse_nextmatch(page, date(2026, 8, 14), ZoneInfo("Asia/Singapore"))
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].team_a, rows[0].team_b), ("T1", "Dplus KIA"))
        self.assertEqual(rows[0].league, "LCK")

    def test_esportagenda_parser_reads_serialized_json_ld(self):
        page = r'''\"@type\":\"SportsEvent\",\"name\":\"T1 vs Dplus KIA\",
        \"startDate\":\"2026-08-14T08:00:00Z\",\"organizer\":{\"@type\":\"Organization\",\"name\":\"LCK\"'''
        rows = parse_esportagenda(page, date(2026, 8, 14), ZoneInfo("Asia/Singapore"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].team_b, "Dplus KIA")

    def test_reconciliation_and_market_mapping_keep_unanalyzable_match_visible(self):
        start = datetime(2026, 8, 14, 8, tzinfo=timezone.utc)
        a = make_match(source="a", league="LCK", team_a="T1", team_b="DK",
                       start_time=start, event_name="LCK", best_of=3)
        b = make_match(source="b", league="LCK", team_a="T1 Esports", team_b="Dplus KIA",
                       start_time=start, event_name="LCK", best_of=3)
        expected, disagreements = reconcile_sources([SourceResult("a", True, [a]), SourceResult("b", True, [b])])
        self.assertEqual(len(expected), 1)
        self.assertFalse(disagreements)
        events = [{"id": "e1", "markets": [{
            "id": "m1", "sportsMarketType": "moneyline", "gameStartTime": start.isoformat(),
            "outcomes": '["T1", "Dplus KIA"]',
        }]}]
        with TemporaryDirectory() as temp:
            audit = build_schedule_audit(
                [SourceResult("a", True, [a]), SourceResult("b", True, [b])], events,
                report_day=date(2026, 8, 14), now=start, registry_path=Path(temp) / "watchers.json",
            )
        self.assertEqual(audit["coverage"], 1.0)
        self.assertEqual(audit["leagues"]["LCK"]["market_matched"], 1)
        self.assertEqual(audit["matches"][0]["analysis_status"], "UNAVAILABLE")
        self.assertFalse(audit["data_incomplete"])
        self.assertTrue(audit["watcher_health"]["healthy"])

    def test_source_failure_is_data_unavailable_not_zero_matches(self):
        with TemporaryDirectory() as temp:
            audit = build_schedule_audit(
                [SourceResult("a", False, [], "timeout"), SourceResult("b", True, [])], [],
                report_day=date(2026, 8, 14), now=datetime(2026, 8, 14, tzinfo=timezone.utc),
                registry_path=Path(temp) / "watchers.json",
            )
        self.assertTrue(audit["data_incomplete"])
        self.assertEqual(audit["coverage"], 0.0)
        self.assertEqual(audit["leagues"]["LCK"]["source_status"]["a"], "DATA_UNAVAILABLE")

    def test_low_confidence_fuzzy_market_requires_review(self):
        start = datetime(2026, 8, 14, 8, tzinfo=timezone.utc)
        match = make_match(source="a", league="LCK", team_a="Unlisted Alpha", team_b="Unlisted Bravo",
                           start_time=start, event_name="LCK")
        events = [{"id": "e1", "markets": [{
            "id": "m1", "sportsMarketType": "moneyline", "gameStartTime": start.isoformat(),
            "outcomes": '["Unlisted Club", "Unlisted Bravo"]',
        }]}]
        match_markets([match], events)
        self.assertEqual(match.market_mapping_status, "MARKET_MAPPING_REVIEW")
        self.assertIsNone(match.market_id)


    def test_market_matching_handles_reversed_outcomes(self):
        start = datetime(2026, 8, 14, 8, tzinfo=timezone.utc)
        match = make_match(source="a", league="LCK", team_a="T1", team_b="DK",
                           start_time=start, event_name="LCK")
        events = [{"id": "e1", "markets": [{
            "id": "m1", "sportsMarketType": "moneyline", "gameStartTime": start.isoformat(),
            "outcomes": '["Dplus KIA", "T1"]',
        }]}]
        match_markets([match], events)
        self.assertEqual(match.market_mapping_status, "MATCHED")

    def test_market_matching_accepts_case_insensitive_type(self):
        start = datetime(2026, 8, 14, 8, tzinfo=timezone.utc)
        match = make_match(source="a", league="LCK", team_a="T1", team_b="DK",
                           start_time=start, event_name="LCK")
        events = [{"id": "e1", "markets": [{
            "id": "m1", "sportsMarketType": "Moneyline", "gameStartTime": start.isoformat(),
            "outcomes": '["T1", "Dplus KIA"]',
        }]}]
        match_markets([match], events)
        self.assertEqual(match.market_mapping_status, "MATCHED")


    def test_lol_aliases_merge_abbreviation_and_full_name(self):
        start = datetime(2026, 8, 14, 8, tzinfo=timezone.utc)
        a = make_match(source="a", league="LPL", team_a="NIP", team_b="LNG",
                       start_time=start, event_name="LPL")
        b = make_match(source="b", league="LPL", team_a="Ninjas in Pyjamas", team_b="LNG Esports",
                       start_time=start, event_name="LPL")
        expected, disagreements = reconcile_sources([SourceResult("a", True, [a]), SourceResult("b", True, [b])])
        self.assertEqual(len(expected), 1)
        self.assertFalse(disagreements)
        self.assertEqual({expected[0].team_a, expected[0].team_b}, {"Ninjas in Pyjamas", "LNG Esports"})

    def test_source_time_mismatch_is_flagged_as_data_mismatch(self):
        start_a = datetime(2026, 8, 14, 8, tzinfo=timezone.utc)
        start_b = datetime(2026, 8, 14, 12, tzinfo=timezone.utc)
        a = make_match(source="a", league="LCK", team_a="T1", team_b="DK",
                       start_time=start_a, event_name="LCK")
        b = make_match(source="b", league="LCK", team_a="T1", team_b="Dplus KIA",
                       start_time=start_b, event_name="LCK")
        mismatches = detect_source_mismatches([SourceResult("a", True, [a]), SourceResult("b", True, [b])])
        self.assertEqual(len(mismatches), 1)
        self.assertIn("cross-referenced sources disagree on start time", mismatches[0]["reason"])


if __name__ == "__main__":
    unittest.main()
