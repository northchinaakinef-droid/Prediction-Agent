from datetime import date, datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from prediction_agent.schedule import (
    LolScheduleDiscovery, SourceResult, build_schedule_audit, detect_source_mismatches, make_match,
    match_markets, parse_esportagenda, parse_leaguepedia_schedule, parse_nextmatch,
    parse_pandascore_lol_schedule, parse_zhibo8_lol_schedule, reconcile_sources,
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


class LolBackupSourceTests(unittest.TestCase):
    zone = ZoneInfo("Asia/Singapore")
    day = date(2026, 8, 14)

    @patch("prediction_agent.schedule._fetch_json")
    def test_leaguepedia_schedule_filters_day_and_target_league(self, fetch_json):
        fetch_json.return_value = {"cargoquery": [
            {"title": {"Team1": "T1", "Team2": "DK", "DateTime_UTC": "2026-08-14 08:00:00",
                       "BestOf": "3", "OverviewPage": "LCK/2026 Season"}},
            {"title": {"Team1": "Old", "Team2": "Match", "DateTime_UTC": "2026-08-13 08:00:00",
                       "BestOf": "3", "OverviewPage": "LCK/2026 Season"}},
            {"title": {"Team1": "Minor", "Team2": "League", "DateTime_UTC": "2026-08-14 09:00:00",
                       "BestOf": "3", "OverviewPage": "NACL/2026 Season"}},
        ]}
        with patch.dict("os.environ", {"LOL_TARGET_LEAGUES": "LCK,LPL"}):
            rows = parse_leaguepedia_schedule(self.day, self.zone)
        self.assertEqual(len(rows), 1)
        self.assertIn("leaguepedia_sched", rows[0].sources)
        self.assertEqual(rows[0].league, "LCK")
        self.assertEqual(fetch_json.call_args.kwargs["params"]["tables"], "MatchSchedule=MS")

    def test_pandascore_schedule_requires_token(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "PANDASCORE_TOKEN not configured"):
                parse_pandascore_lol_schedule(self.day, self.zone)

    @patch("prediction_agent.schedule._fetch_json")
    def test_pandascore_schedule_parses_upcoming_response(self, fetch_json):
        fetch_json.side_effect = [[], [{
            "id": 42, "begin_at": "2026-08-14T08:00:00Z", "status": "not_started",
            "number_of_games": 3, "league": {"name": "LCK"},
            "tournament": {"name": "LCK Summer"},
            "opponents": [
                {"opponent": {"id": 1, "name": "T1"}},
                {"opponent": {"id": 2, "name": "DK"}},
            ],
        }]]
        with patch.dict("os.environ", {"PANDASCORE_TOKEN": "test-token"}):
            rows = parse_pandascore_lol_schedule(self.day, self.zone)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].best_of, 3)
        self.assertIn("pandascore_lol_sched", rows[0].sources)
        self.assertTrue(fetch_json.call_args.args[0].endswith("/upcoming"))

    def test_zhibo8_parses_json_ld(self):
        page = '''<script type="application/ld+json">{
          "@type":"SportsEvent", "name":"T1 vs DK", "startDate":"2026-08-14T16:00:00+08:00",
          "organizer":{"@type":"Organization", "name":"LCK"}
        }</script>'''
        rows = parse_zhibo8_lol_schedule(page, self.day, self.zone)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].team_b, "Dplus KIA")
        self.assertIn("zhibo8", rows[0].sources)

    def test_zhibo8_parses_table_fallback(self):
        page = "<table><tr><td>16:00</td><td>LPL</td><td>NIP vs LNG</td></tr></table>"
        rows = parse_zhibo8_lol_schedule(page, self.day, self.zone)
        self.assertEqual(len(rows), 1)
        self.assertEqual({rows[0].team_a, rows[0].team_b}, {"Ninjas in Pyjamas", "LNG Esports"})

    @patch("prediction_agent.schedule.parse_zhibo8_lol_schedule", return_value=[])
    @patch("prediction_agent.schedule.parse_pandascore_lol_schedule", return_value=[])
    @patch("prediction_agent.schedule.parse_leaguepedia_schedule", return_value=[])
    def test_discover_returns_five_named_sources(self, _leaguepedia, _pandascore, _zhibo8):
        discovery = LolScheduleDiscovery(fetch=lambda _url: "")
        results = discovery.discover(self.day)
        self.assertEqual(
            [result.name for result in results],
            ["nextmatch", "esportagenda", "leaguepedia_sched", "pandascore_lol_sched"],
        )

    @patch("prediction_agent.schedule.parse_zhibo8_lol_schedule", return_value=[])
    @patch("prediction_agent.schedule.parse_leaguepedia_schedule", return_value=[])
    def test_discover_marks_missing_pandascore_token_unavailable(self, _leaguepedia, _zhibo8):
        with patch.dict("os.environ", {}, clear=True):
            results = LolScheduleDiscovery(fetch=lambda _url: "").discover(self.day)
        pandascore = next(result for result in results if result.name == "pandascore_lol_sched")
        self.assertFalse(pandascore.available)
        self.assertIn("PANDASCORE_TOKEN not configured", pandascore.error)

    @patch("prediction_agent.schedule.parse_pandascore_lol_schedule", return_value=[])
    @patch("prediction_agent.schedule.parse_leaguepedia_schedule", return_value=[])
    def test_zhibo8_failure_does_not_affect_other_sources(self, _leaguepedia, _pandascore):
        def fetch(url):
            if "zhibo8" in url:
                raise TimeoutError("zhibo8 timeout")
            return ""

        results = LolScheduleDiscovery(fetch=fetch).discover(self.day)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(result.available for result in results))


if __name__ == "__main__":
    unittest.main()
