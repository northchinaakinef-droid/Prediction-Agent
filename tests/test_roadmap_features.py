from datetime import datetime, timezone
import unittest

from prediction_agent.lol_model import (
    LolGame, patch_adaptation_uncertainty, player_level_team_rating,
    recent_form_rating, roster_adjusted_probability, side_advantage,
)
from prediction_agent.nba_model import (
    back_to_back, fit_nba, nba_schedule_fatigue,
)


class NbaRoadmapFeatureTests(unittest.TestCase):
    def test_margin_efficiency_rating_updates_with_margin(self):
        big = fit_nba([LolGame("1", datetime(2024, 1, 1, tzinfo=timezone.utc), "nba", "A", "B", 1, margin=20)])
        small = fit_nba([LolGame("2", datetime(2024, 1, 1, tzinfo=timezone.utc), "nba", "A", "B", 1, margin=2)])
        self.assertGreater(abs(big.efficiency_ratings["A"]), abs(small.efficiency_ratings["A"]))

    def test_back_to_back_and_travel_fatigue_helpers(self):
        played = datetime(2024, 1, 2, tzinfo=timezone.utc)
        self.assertTrue(back_to_back("2024-01-01", played))
        fatigue = nba_schedule_fatigue(
            "Los Angeles Lakers", "2024-01-01", "Los Angeles Lakers",
            "Boston Celtics", played,
        )
        self.assertTrue(fatigue["back_to_back"])
        self.assertGreater(fatigue["travel_distance_miles"], 0)


class LolRoadmapFeatureTests(unittest.TestCase):
    def test_roster_swap_changes_probability_immediately(self):
        ratings = {"A": 1500, "B": 1500}
        base = roster_adjusted_probability(
            1500, 1500,
            player_level_team_rating(ratings, ["P1", "P2", "P3", "P4", "P5"]),
            player_level_team_rating(ratings, ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        )
        swapped = roster_adjusted_probability(
            1500, 1500,
            player_level_team_rating({**ratings, "P1": 1800}, ["P1", "P2", "P3", "P4", "P5"]),
            player_level_team_rating(ratings, ["Q1", "Q2", "Q3", "Q4", "Q5"]),
        )
        self.assertGreater(swapped, base)

    def test_patch_adaptation_uncertainty(self):
        self.assertLess(patch_adaptation_uncertainty(0), 1.0)
        self.assertEqual(patch_adaptation_uncertainty(7), 1.0)

    def test_side_advantage_varies_by_patch_and_region(self):
        table = {("14.1", "LCK"): 0.045, ("14.1", "LPL"): 0.030}
        self.assertNotEqual(side_advantage("14.1", "LCK", table), side_advantage("14.1", "LPL", table))
        self.assertEqual(side_advantage("99.9", "NOWHERE", table), 0.0)

    def test_recent_form_rating_uses_recent_window(self):
        games = [
            LolGame("1", datetime(2024, 1, i + 1, tzinfo=timezone.utc), "lol", "A", "B", 1)
            for i in range(10)
        ]
        self.assertGreater(recent_form_rating(games, "A", datetime(2024, 1, 11, tzinfo=timezone.utc)), 0.5)


if __name__ == "__main__":
    unittest.main()
