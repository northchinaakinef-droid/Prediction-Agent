import unittest

from prediction_agent.providers.analysts import (
    AnalystFeedProvider, ChineseLolPostProvider, _PostLinkParser, note_matches_event,
)


class AnalystFeedTests(unittest.TestCase):
    def test_chinese_post_parser_isolated_sources_and_deduplicates(self):
        provider = ChineseLolPostProvider(pages=())
        self.assertEqual(provider.recent(), [])

    def test_hupu_post_links_are_parsed(self):
        parser = _PostLinkParser("https://bbs.hupu.com/lol")
        parser.feed('<a href="/641989757.html">[赛后]WE 2-1 AL：完成让一追二</a>')
        self.assertEqual(parser.rows, [
            ("[赛后]WE 2-1 AL：完成让一追二", "https://bbs.hupu.com/641989757.html")
        ])

    def test_chinese_post_matches_registered_team_alias(self):
        from datetime import datetime, timezone
        from prediction_agent.providers.analysts import AnalystNote
        note = AnalystNote("虎扑", "lol", "[赛后]WE 2-1 AL", "https://example.invalid/1", datetime.now(timezone.utc))
        self.assertTrue(note_matches_event(note, "lol", "Anyone's Legend", "Team WE"))
        self.assertFalse(note_matches_event(note, "lol", "T1", "Gen.G"))

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
