import unittest

from prediction_agent.sports_daily import _text


class TextCleaningTests(unittest.TestCase):
    def test_removes_invisible_format_characters(self):
        self.assertEqual(_text("\u2060Hooligans"), "Hooligans")


if __name__ == "__main__":
    unittest.main()
