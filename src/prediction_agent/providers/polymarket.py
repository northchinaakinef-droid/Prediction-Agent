from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .http import get_json, post_json


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

    def event(self, event_id: str) -> dict[str, Any]:
        return get_json(f"{GAMMA}/events/{event_id}", timeout=self.timeout)

    def events_by_tag(
        self,
        tag_id: str,
        *,
        limit: int = 100,
        active: bool = True,
        closed: bool = False,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return get_json(
            f"{GAMMA}/events",
            {
                "tag_id": tag_id,
                "related_tags": "true",
                "limit": limit,
                "active": str(active).lower(),
                "closed": str(closed).lower(),
                "offset": offset,
            },
            self.timeout,
        )

    def all_events_by_tag(
        self,
        tag_id: str,
        *,
        page_size: int = 100,
        active: bool = True,
        closed: bool = False,
        max_pages: int = 20,
    ) -> list[dict[str, Any]]:
        """Return every event for a tag instead of silently truncating page one."""
        if page_size < 1 or page_size > 500:
            raise ValueError("page_size must be between 1 and 500")
        events: dict[str, dict[str, Any]] = {}
        for page in range(max_pages):
            rows = self.events_by_tag(
                tag_id, limit=page_size, active=active, closed=closed,
                offset=page * page_size,
            )
            for row in rows:
                events[str(row.get("id"))] = row
            if len(rows) < page_size:
                return list(events.values())
        raise RuntimeError(f"Polymarket pagination exceeded {max_pages} pages for tag {tag_id}")

    def search_sports(self, query: str, *, limit: int = 100) -> list[dict[str, Any]]:
        needle = query.casefold()
        for sport in self.sports():
            if needle == str(sport.get("sport", "")).casefold():
                tags = [tag.strip() for tag in str(sport.get("tags", "")).split(",") if tag.strip()]
                events: list[dict[str, Any]] = []
                for tag in tags:
                    events.extend(self.all_events_by_tag(tag, page_size=min(limit, 500)))
                # The same event may occur under related tags.
                return list({str(e.get("id")): e for e in events}.values())[:limit]
        rows = self.markets(limit=limit)
        return [m for m in rows if needle in f"{m.get('question', '')} {m.get('description', '')}".casefold()]

    def lol_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        """Active LoL events. Tag 65 is Polymarket's primary LoL tag."""
        return self.all_events_by_tag("65", page_size=limit)

    def order_book(self, token_id: str) -> dict[str, Any]:
        return get_json(f"{CLOB}/book", {"token_id": token_id}, self.timeout)

    def midpoint(self, token_id: str) -> float:
        row = get_json(f"{CLOB}/midpoint", {"token_id": token_id}, self.timeout)
        return float(row["mid"])

    def price_history(
        self, token_id: str, *, start_ts: int, end_ts: int, fidelity: int = 30
    ) -> list[dict[str, Any]]:
        """Public aggregate price history; this is not historical executable depth."""
        row = get_json(
            f"{CLOB}/prices-history",
            {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity},
            self.timeout,
        )
        return list(row.get("history", []))

    def batch_price_history(self, markets: list[dict[str, object]]) -> Any:
        """Fetch up to 20 public price histories using Polymarket's batch endpoint."""
        if not 1 <= len(markets) <= 20:
            raise ValueError("batch price history requires 1 to 20 markets")
        return post_json(
            f"{CLOB}/batch-prices-history",
            {"markets": markets},
            timeout=self.timeout,
        )

    def holders(self, condition_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        return get_json(f"{DATA}/holders", {"market": condition_id, "limit": limit}, self.timeout)

    @staticmethod
    def token_ids(market: dict[str, Any]) -> list[str]:
        value = market.get("clobTokenIds", [])
        if isinstance(value, str):
            value = json.loads(value)
        return [str(v) for v in value]
