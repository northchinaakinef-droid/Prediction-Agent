import unittest

from prediction_agent.nba_analytics import (
    boxscore_metrics, duration_seconds, flatten_nba_sample, four_factors,
)


class NbaAnalyticsTests(unittest.TestCase):
    def test_duration_seconds_parses_iso_8601(self):
        self.assertEqual(duration_seconds("PT14M34.00S"), 874.0)
        self.assertEqual(duration_seconds("PT240M"), 14400.0)
        self.assertEqual(duration_seconds(None), 0.0)

    def test_four_factors_and_flattened_features(self):
        home_stats = {
            "fieldGoalsMade": 40, "fieldGoalsAttempted": 88, "threePointersMade": 12,
            "freeThrowsMade": 18, "freeThrowsAttempted": 22, "turnoversTotal": 10,
            "reboundsOffensive": 11, "reboundsDefensive": 32, "reboundsTotal": 43,
        }
        away_stats = {
            "fieldGoalsMade": 38, "fieldGoalsAttempted": 90, "threePointersMade": 10,
            "freeThrowsMade": 18, "freeThrowsAttempted": 24, "turnoversTotal": 14,
            "reboundsOffensive": 12, "reboundsDefensive": 30, "reboundsTotal": 42,
        }
        factors = four_factors(home_stats, away_stats)
        self.assertAlmostEqual(factors["effective_field_goal_pct"], 46 / 88, places=3)
        self.assertLess(factors["turnover_pct"], 0.12)
        box = {
            "duration": "PT48M",
            "home_team": {"score": 110, "statistics": home_stats, "players": []},
            "away_team": {"score": 104, "statistics": away_stats, "players": []},
        }
        metrics = boxscore_metrics(box)
        self.assertIn("home_four_factors", metrics)
        self.assertIn("pace", metrics)
        sample = {
            "home_score": 110, "away_score": 104, "winner_side": "b",
            "metrics": metrics,
        }
        flat = flatten_nba_sample(sample)
        self.assertAlmostEqual(flat["pace"], metrics["pace"])
        self.assertAlmostEqual(flat["home_efg"], factors["effective_field_goal_pct"])


if __name__ == "__main__":
    unittest.main()
