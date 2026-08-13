from datetime import datetime, timezone
import unittest

from prediction_agent.lol_meta_model import LolDraftGame, fit_lol_meta


class LolMetaModelTests(unittest.TestCase):
    def test_separates_pre_and_post_draft_probability(self):
        games = []
        for index in range(40):
            games.append(LolDraftGame(
                str(index), datetime(2024, 1, 1, tzinfo=timezone.utc), "14.01", "LCK", "A", "B",
                ("a1", "a2", "a3", "a4", "a5"), ("b1", "b2", "b3", "b4", "b5"),
                ("c1", "c2", "c3", "c4", "c5"), ("d1", "d2", "d3", "d4", "d5"), 1))
        model = fit_lol_meta(games)
        game = games[-1]
        self.assertGreater(model.predict_pre_draft(game), .5)
        self.assertGreater(model.predict_post_draft(game), .5)
        self.assertNotEqual(model.predict_pre_draft(game), model.predict_post_draft(game))


if __name__ == "__main__":
    unittest.main()
