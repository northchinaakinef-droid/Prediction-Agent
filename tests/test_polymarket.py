import unittest
from unittest.mock import patch

from prediction_agent.providers.polymarket import PolymarketClient


class PolymarketClientTests(unittest.TestCase):
    @patch("prediction_agent.providers.polymarket.get_json")
    def test_closed_events_include_pagination_parameters(self, get_json):
        get_json.return_value = []
        client = PolymarketClient(timeout=7)
        client.events_by_tag("100780", active=False, closed=True, offset=100)
        params = get_json.call_args.args[1]
        self.assertEqual(params["active"], "false")
        self.assertEqual(params["closed"], "true")
        self.assertEqual(params["offset"], 100)

    @patch("prediction_agent.providers.polymarket.get_json")
    def test_price_history_returns_history_points(self, get_json):
        get_json.return_value = {"history": [{"t": 10, "p": 0.6}]}
        points = PolymarketClient().price_history("token", start_ts=1, end_ts=20)
        self.assertEqual(points, [{"t": 10, "p": 0.6}])

    def test_batch_history_enforces_api_limit(self):
        with self.assertRaises(ValueError):
            PolymarketClient().batch_price_history([])
        with self.assertRaises(ValueError):
            PolymarketClient().batch_price_history([{}] * 21)


if __name__ == "__main__":
    unittest.main()
