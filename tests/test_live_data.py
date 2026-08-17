from datetime import date
import os
import unittest
from unittest.mock import patch

from prediction_agent.providers.live_data import (
    Bo3Cs2Provider, DataSourceUnavailable, EspnNbaProvider, GridOpenAccessProvider,
    LeaguepediaDraftProvider, NbaBoxscoreProvider, PandaScoreProvider, RiotEsportsProvider,
    SportSrcNbaProvider, TheSportsDbNbaProvider,
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

    def test_riot_esports_rejects_standard_riot_api_key_without_network(self):
        with patch("prediction_agent.providers.live_data._get_json") as get_json:
            with self.assertRaisesRegex(DataSourceUnavailable, "RGAPI"):
                RiotEsportsProvider(api_key="RGAPI-daa2a710-e02f-4a8f-b070-df970c4f990b").league_ids([])
        get_json.assert_not_called()

    @patch("prediction_agent.providers.live_data._get_json")
    def test_riot_esports_uses_x_api_key_header_for_esports_key(self, get_json):
        get_json.return_value = {"data": {"leagues": [
            {"id": "1", "name": "LCK"},
            {"id": "2", "name": "LPL"},
        ]}}
        provider = RiotEsportsProvider(api_key="esports-style-key")
        self.assertEqual(provider.league_ids(["LCK"]), ["1"])
        get_json.assert_called_once()
        self.assertEqual(get_json.call_args.kwargs["headers"], {"x-api-key": "esports-style-key"})

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
            "strStatus": "In Progress", "strProgress": "Q4 01:42",
            "intAwayScore": "101", "intHomeScore": "99",
        }]}
        rows = TheSportsDbNbaProvider().live(date(2026, 8, 14))
        self.assertEqual(rows[0].status, "LIVE")
        self.assertEqual((rows[0].score_a, rows[0].score_b), (101, 99))
        self.assertEqual(rows[0].features["period"], 4)
        self.assertEqual(rows[0].features["game_clock_seconds"], 102)

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


    @patch("prediction_agent.providers.live_data._get_json")
    def test_nba_boxscore_parses_team_and_player_stats(self, get_json):
        get_json.return_value = {"game": {
            "gameId": "0022400001", "gameStatus": 3, "gameStatusText": "Final",
            "period": 4, "gameTimeUTC": "2025-06-01T00:00:00Z", "duration": "PT02H10M",
            "homeTeam": {
                "teamId": 1610612747, "teamName": "Lakers", "teamCity": "Los Angeles",
                "teamTricode": "LAL", "score": 110,
                "periods": [{"period": 1, "periodType": "REGULAR", "score": 28}],
                "statistics": {
                    "fieldGoalsMade": 40, "fieldGoalsAttempted": 88, "threePointersMade": 12,
                    "freeThrowsMade": 18, "freeThrowsAttempted": 22, "turnoversTotal": 10,
                    "reboundsOffensive": 11, "reboundsDefensive": 32, "reboundsTotal": 43,
                    "pointsInThePaint": 46, "pointsFastBreak": 12, "pointsSecondChance": 10,
                    "pointsFromTurnovers": 14, "benchPoints": 32, "assists": 24,
                    "steals": 8, "blocks": 5, "leadChanges": 4, "timesTied": 3,
                    "biggestLead": 12, "minutes": "PT240M",
                },
                "players": [{
                    "personId": 1, "name": "LeBron James", "jerseyNum": "23", "position": "F",
                    "starter": "1", "played": "1",
                    "statistics": {"points": 28, "reboundsTotal": 8, "assists": 7,
                                   "plusMinusPoints": 9, "minutes": "PT36M00.00S"},
                }],
            },
            "awayTeam": {
                "teamId": 1610612738, "teamName": "Celtics", "teamCity": "Boston",
                "teamTricode": "BOS", "score": 104,
                "periods": [{"period": 1, "periodType": "REGULAR", "score": 25}],
                "statistics": {
                    "fieldGoalsMade": 38, "fieldGoalsAttempted": 90, "threePointersMade": 10,
                    "freeThrowsMade": 18, "freeThrowsAttempted": 24, "turnoversTotal": 14,
                    "reboundsOffensive": 12, "reboundsDefensive": 30, "reboundsTotal": 42,
                    "pointsInThePaint": 42, "pointsFastBreak": 10, "pointsSecondChance": 12,
                    "pointsFromTurnovers": 11, "benchPoints": 25, "assists": 22,
                    "steals": 7, "blocks": 4, "leadChanges": 4, "timesTied": 3,
                    "biggestLead": 8, "minutes": "PT240M",
                },
                "players": [],
            },
        }}
        box = NbaBoxscoreProvider().boxscore("0022400001")
        self.assertIsNotNone(box)
        self.assertEqual(box["game_id"], "0022400001")
        self.assertEqual(box["home_team"]["team_tricode"], "LAL")
        self.assertTrue(box["home_team"]["winner"])
        self.assertEqual(box["away_team"]["players"], [])
        self.assertEqual(box["home_team"]["players"][0]["name"], "LeBron James")
        self.assertAlmostEqual(box["home_team"]["players"][0]["minutes_seconds"], 2160.0)

if __name__ == "__main__":
    unittest.main()
