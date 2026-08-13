from datetime import datetime, timezone
import unittest

from prediction_agent.lol_daily import build_lol_report
from prediction_agent.lol_model import EloModel, LolGame, evaluate_periods, evaluate_years, fit_elo, series_probability
from prediction_agent.sports_daily import analyze_sport


class LolModelTests(unittest.TestCase):
    def test_series_probability(self):
        self.assertAlmostEqual(series_probability(.6, 3), .648)
        self.assertAlmostEqual(series_probability(.5, 5), .5)

    def test_independent_elo_learns_without_market_input(self):
        games = [LolGame(str(i), datetime(2023, 1, i + 1, tzinfo=timezone.utc), "LPL", "A", "B", 1)
                 for i in range(20)]
        model = fit_elo(games)
        self.assertGreater(model.game_probability("A", "B"), .75)

    def test_year_splits_are_strict(self):
        games = []
        for year in (2022, 2023, 2024, 2025):
            for index in range(600):
                games.append(LolGame(f"{year}-{index}", datetime(year, 1, 1, tzinfo=timezone.utc),
                                     "LPL", "A", "B", index % 2))
        report = evaluate_years(games)
        self.assertEqual(report["protocol"]["validation"], "2024-2024")
        self.assertEqual(report["protocol"]["final_test"], "2025-2025")
        self.assertTrue(report["protocol"]["final_test_is_locked"])
        self.assertFalse(report["approved_for_real_money"])

    def test_multi_year_validation_and_final_test(self):
        games = []
        for year in range(2020, 2026):
            for index in range(500):
                games.append(LolGame(f"{year}-{index}", datetime(year, 1, 1, tzinfo=timezone.utc),
                                     "nba", "A", "B", index % 2))
        report = evaluate_periods(games, train_end=2021, validation_start=2022,
                                  validation_end=2023, test_start=2024, test_end=2025)
        self.assertEqual(report["validation"]["samples"], 1000)
        self.assertEqual(report["final_test"]["samples"], 1000)
        self.assertEqual(report["protocol"]["final_test"], "2024-2025")

    def test_daily_report_keeps_model_and_market_probabilities_separate(self):
        model = EloModel({"A": 1700, "B": 1500}, {"A": 100, "B": 100}, "2026-08-12", 200)
        event = {
            "id": "1", "title": "LoL: A vs B (BO3) - Test",
            "description": "initially scheduled for August 13 at 1:00AM ET.",
            "markets": [{
                "question": "LoL: A vs B (BO3) - Test", "outcomes": '["A","B"]',
                "outcomePrices": '["0.60","0.40"]', "bestBid": .59, "bestAsk": .61,
                "spread": .02, "liquidity": "5000",
            }],
        }
        report = build_lol_report(model, {"approved_for_probability_use": True}, [event],
                                  now=datetime(2026, 8, 13, 0, tzinfo=timezone.utc))
        row = report["recommendations"][0]
        self.assertNotEqual(row["model_probability"], row["market_probability"])
        self.assertEqual(row["action"], "NO_BET")
        self.assertTrue(any("市场仅用于比较" in reason for reason in row["reasons"]))

    def test_nba_uses_direct_game_probability(self):
        model = EloModel({"Boston Celtics": 1700, "Los Angeles Lakers": 1500},
                         {"Boston Celtics": 100, "Los Angeles Lakers": 100},
                         "2026-06-01", 200)
        event = {"id": "nba-1", "title": "Celtics vs Lakers", "markets": [{
            "question": "Celtics vs Lakers", "sportsMarketType": "moneyline",
            "gameStartTime": "2026-08-13T10:00:00Z", "outcomes": '["Celtics","Lakers"]',
            "outcomePrices": '["0.60","0.40"]', "bestBid": .59, "bestAsk": .61,
            "spread": .02, "liquidity": "5000",
        }]}
        rows = analyze_sport("nba", model, {"approved_for_probability_use": True}, [event],
                             now=datetime(2026, 8, 13, 0, tzinfo=timezone.utc), bankroll=1000)
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["model_probability"], .7)
        self.assertEqual(rows[0]["action"], "NO_BET")

    def test_non_game_market_is_excluded(self):
        model = EloModel({}, {}, "2026-06-01", 10)
        event = {"id": "prop", "title": "Will a player retire?", "markets": [{
            "question": "Will a player retire?", "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.1","0.9"]',
        }]}
        rows = analyze_sport("nba", model, {}, [event],
                             now=datetime(2026, 8, 13, tzinfo=timezone.utc), bankroll=1000)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
