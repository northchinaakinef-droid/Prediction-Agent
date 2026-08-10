from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .http import get_json


BASE = "https://api.the-odds-api.com/v4"


@dataclass
class OddsApiClient:
    api_key: str
    regions: str = "us,eu"
    timeout: float = 15

    def sports(self, *, include_inactive: bool = True) -> list[dict[str, Any]]:
        return get_json(f"{BASE}/sports", {"apiKey": self.api_key, "all": str(include_inactive).lower()}, self.timeout)

    def odds(self, sport_key: str, markets: str = "h2h,spreads,totals") -> list[dict[str, Any]]:
        return get_json(
            f"{BASE}/sports/{sport_key}/odds",
            {"apiKey": self.api_key, "regions": self.regions, "markets": markets, "oddsFormat": "decimal"},
            self.timeout,
        )

    def historical_odds(self, sport_key: str, timestamp: str, markets: str = "h2h") -> dict[str, Any]:
        return get_json(
            f"{BASE}/historical/sports/{sport_key}/odds",
            {
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": markets,
                "oddsFormat": "decimal",
                "date": timestamp,
            },
            self.timeout,
        )

