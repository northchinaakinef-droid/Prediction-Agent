import unittest

from scripts.check_llm_preflight import run


class V25LlmPreflightTests(unittest.TestCase):
    def test_preflight_passes_without_api_key(self):
        checks, blockers = run()
        self.assertFalse(blockers)
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
