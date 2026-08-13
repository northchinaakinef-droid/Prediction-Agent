from datetime import date
import unittest

from prediction_agent.cs2_model import Cs2Series, evaluate_cs2, fit_cs2


class Cs2ModelTests(unittest.TestCase):
    def test_roster_aware_model_learns_players_and_team(self):
        games = [Cs2Series(str(i), date(2024, 1, 1), "A", "B",
                           ("a1", "a2", "a3", "a4", "a5"),
                           ("b1", "b2", "b3", "b4", "b5"), 1) for i in range(30)]
        model = fit_cs2(games)
        self.assertGreater(model.probability("A", "B", games[0].roster_a, games[0].roster_b), .75)

    def test_strict_year_split(self):
        games = []
        for year in (2024, 2025, 2026):
            for i in range(350):
                games.append(Cs2Series(f"{year}-{i}", date(year, 1, 1), "A", "B",
                                       ("a1", "a2", "a3", "a4", "a5"),
                                       ("b1", "b2", "b3", "b4", "b5"), i % 2))
        _, report = evaluate_cs2(games)
        self.assertEqual(report["data"]["test_samples"], 350)
        self.assertTrue(report["protocol"]["final_test_is_locked"])
        self.assertFalse(report["approved_for_real_money"])


if __name__ == "__main__":
    unittest.main()
