import unittest

from prediction_agent.providers.analysts import AnalystFeedProvider


class AnalystFeedTests(unittest.TestCase):
    def test_lol_feed_keeps_league_of_legends_articles(self):
        provider = AnalystFeedProvider("lol", urls=[])
        self.assertTrue(provider._matches("League of Legends patch 15.4 changes"))
        self.assertTrue(provider._matches("LPL spring playoff preview"))
        self.assertFalse(provider._matches("NBA finals preview"))

    def test_nba_feed_keeps_basketball_articles(self):
        provider = AnalystFeedProvider("nba", urls=[])
        self.assertTrue(provider._matches("NBA playoff predictions"))
        self.assertTrue(provider._matches("Lakers vs Celtics breakdown"))
        self.assertFalse(provider._matches("League of Legends worlds preview"))


if __name__ == "__main__":
    unittest.main()
