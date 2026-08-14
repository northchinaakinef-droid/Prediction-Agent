from datetime import date
import os
import unittest
from unittest.mock import patch

from prediction_agent.providers.live_data import (
    Bo3Cs2Provider, DataSourceUnavailable, EspnNbaProvider, GridOpenAccessProvider,
    LeaguepediaDraftProvider, PandaScoreProvider, SportSrcNbaProvider, TheSportsDbNbaProvider,
)


class LiveDataProviderTests(unittest.TestCase):
    @patch("prediction_agent.providers.live_data._get_json")
    def test_espn_schedule_parses_canonical_event(self, get_json):
        get_json.return_value = {"events": [{
            "id": "401", "date": "2026-08-14T12:00:00Z",
            "status": {"type": {"name": "STATUS_SCHEDULED"}},
            "competitions": [{"competitors": [
                {"homeAway": "away", "team": {"displayName": "Boston Celtics"}},
                {"homeAway": "home", "team": {"displayName": "Los Angeles Lakers"}},
            ]}],
        }]}
        rows = EspnNbaProvider().schedule(date(2026, 8, 14))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].team_a, "Boston Celtics")

    def test_grid_missing_key_is_explicitly_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(DataSourceUnavailable, "GRID_API_KEY"):
                GridOpenAccessProvider(api_key=None).schedule(date(2026, 8, 14))

    def test_pandascore_missing_token_is_explicitly_unavailable(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(DataSourceUnavailable, "PANDASCORE_TOKEN"):
                PandaScoreProvider(token=None).live("cs2")

    @patch("prediction_agent.providers.live_data._get_json")
    def test_leaguepedia_parses_completed_draft(self, get_json):
        get_json.return_value = {"cargoquery": [{"title": {
            "Team1": "T1", "Team2": "Gen.G", "Tournament": "LCK 2026",
            "GameId": "g1", "Winner": "", "Team1Picks": "A,B,C,D,E", "Team2Picks": "F,G,H,I,J",
        }}, {"title": {
            "Team1": "T1", "Team2": "Gen.G", "Tournament": "LCK 2026",
            "GameId": "older", "Winner": "1", "Team1Picks": "K,L,M,N,O", "Team2Picks": "P,Q,R,S,T",
        }}]}
        rows = LeaguepediaDraftProvider().live()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].features["champions_a"], ["A", "B", "C", "D", "E"])

    @patch("prediction_agent.providers.live_data._get_json")
    def test_thesportsdb_filters_out_non_nba_basketball(self, get_json):
        get_json.return_value = {"events": [
            {"idEvent": "1", "strLeague": "WNBA", "strTimestamp": "2026-08-14T10:00:00Z",
             "strAwayTeam": "A", "strHomeTeam": "B"},
            {"idEvent": "2", "strLeague": "NBA", "strTimestamp": "2026-08-14T12:00:00Z",
             "strAwayTeam": "Celtics", "strHomeTeam": "Lakers"},
        ]}
        rows = TheSportsDbNbaProvider().schedule(date(2026, 8, 14))
        self.assertEqual([row.source_id for row in rows], ["2"])

    @patch("prediction_agent.providers.live_data._get_json")
    def test_thesportsdb_live_parses_score_and_status(self, get_json):
        get_json.return_value = {"events": [{
            "idEvent": "2", "strLeague": "NBA", "strAwayTeam": "Celtics", "strHomeTeam": "Lakers",
            "strStatus": "In Progress - 4th Quarter", "intAwayScore": "101", "intHomeScore": "99",
        }]}
        rows = TheSportsDbNbaProvider().live(date(2026, 8, 14))
        self.assertEqual(rows[0].status, "LIVE")
        self.assertEqual((rows[0].score_a, rows[0].score_b), (101, 99))

    @patch("prediction_agent.providers.live_data._get_json")
    def test_bo3_schedule_filters_to_configured_tiers(self, get_json):
        get_json.return_value = {"results": [
            {"id": 1, "status": "upcoming", "start_date": "2026-08-14T12:00:00+00:00",
             "bo_type": 3, "team1": {"name": "Spirit"}, "team2": {"name": "Vitality"},
             "tournament": {"name": "Major", "tier_rank": 1}},
            {"id": 2, "status": "upcoming", "start_date": "2026-08-14T13:00:00+00:00",
             "bo_type": 3, "team1": {"name": "Academy A"}, "team2": {"name": "Academy B"},
             "tournament": {"name": "Local", "tier_rank": 4}},
        ]}
        with patch.dict(os.environ, {"CS2_MAX_TIER_RANK": "2"}):
            rows = Bo3Cs2Provider().schedule(date(2026, 8, 14))
        self.assertEqual([row.source_id for row in rows], ["1"])

    @patch("prediction_agent.providers.live_data._get_json")
    def test_sportsrc_uses_entity_registry_to_exclude_wnba(self, get_json):
        get_json.return_value = {"data": [
            {"id": "nba", "date": 1786750200000,
             "teams": {"away": {"name": "Lakers"}, "home": {"name": "Celtics"}}},
            {"id": "wnba", "date": 1786750200000,
             "teams": {"away": {"name": "Dallas Wings"}, "home": {"name": "Indiana Fever"}}},
        ]}
        rows = SportSrcNbaProvider().schedule(date(2026, 8, 14))
        self.assertEqual([row.source_id for row in rows], ["nba"])


if __name__ == "__main__":
    unittest.main()
