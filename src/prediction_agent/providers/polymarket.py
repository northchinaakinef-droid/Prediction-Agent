from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .http import get_json


GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"


@dataclass
class PolymarketClient:
    timeout: float = 15

    def markets(self, *, limit: int = 100, active: bool = True, closed: bool = False) -> list[dict[str, Any]]:
        return get_json(
            f"{GAMMA}/markets",
            {"limit": limit, "active": str(active).lower(), "closed": str(closed).lower()},
            self.timeout,
        )

    def sports(self) -> list[dict[str, Any]]:
        return get_json(f"{GAMMA}/sports", timeout=self.timeout)

    def events_by_tag(self, tag_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        return get_json(
            f"{GAMMA}/events",
            {
                "tag_id": tag_id,
                "related_tags": "true",
                "limit": limit,
                "active": "true",
                "closed": "false",
            },
            self.timeout,
        )

    def search_sports(self, query: str, *, limit: int = 100) -> list[dict[str, Any]]:
        needle = query.casefold()
        for sport in self.sports():
            if needle == str(sport.get("sport", "")).casefold():
                tags = [tag.strip() for tag in str(sport.get("tags", "")).split(",") if tag.strip()]
                events: list[dict[str, Any]] = []
                for tag in tags:
                    events.extend(self.events_by_tag(tag, limit=limit))
                # The same event may occur under related tags.
                return list({str(e.get("id")): e for e in events}.values())[:limit]
        rows = self.markets(limit=limit)
        return [m for m in rows if needle in f"{m.get('question', '')} {m.get('description', '')}".casefold()]

    def lol_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Active LoL events. Tag 65 is Polymarket's primary LoL tag."""
        return self.events_by_tag("65", limit=limit)

    def order_book(self, token_id: str) -> dict[str, Any]:
        return get_json(f"{CLOB}/book", {"token_id": token_id}, self.timeout)

    def midpoint(self, token_id: str) -> float:
        row = get_json(f"{CLOB}/midpoint", {"token_id": token_id}, self.timeout)
        return float(row["mid"])

    def holders(self, condition_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return get_json(f"{DATA}/holders", {"market": condition_id, "limit": limit}, self.timeout)

    @staticmethod
    def token_ids(market: dict[str, Any]) -> list[str]:
        value = market.get("clobTokenIds", [])
        if isinstance(value, str):
            value = json.loads(value)
        return [str(v) for v in value]
