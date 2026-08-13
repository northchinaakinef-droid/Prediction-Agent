from datetime import datetime, timezone
import unittest

from prediction_agent.lol_model import LolGame
from prediction_agent.nba_model import fit_nba, walk_forward_nba


class NbaModelTests(unittest.TestCase):
    def test_home_advantage_and_rest_are_pregame_features(self):
        games = [LolGame("1", datetime(2024, 1, 1, tzinfo=timezone.utc), "nba", "A", "B", 1)]
        model = fit_nba(games, home_advantage=60, rest_day_value=8)
        probability = model.game_probability("C", "D", datetime(2024, 1, 2, tzinfo=timezone.utc))
        self.assertGreater(probability, .5)

    def test_walk_forward_is_before_result_update(self):
        games = [
            LolGame("1", datetime(2024, 1, 1, tzinfo=timezone.utc), "nba", "A", "B", 1),
            LolGame("2", datetime(2024, 1, 2, tzinfo=timezone.utc), "nba", "A", "B", 1),
        ]
        rows = walk_forward_nba(games, home_advantage=0, rest_day_value=0)
        self.assertEqual(rows[0]["model_probability_a"], .5)
        self.assertGreater(rows[1]["model_probability_a"], .5)


if __name__ == "__main__":
    unittest.main()
