from __future__ import annotations

import email.utils
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen


@dataclass
class AnalystNote:
    source: str
    sport: str
    title: str
    link: str
    published_at: datetime


DEFAULT_FEEDS = {
    "lol": ["https://dotesports.com/feed"],
    "nba": ["https://www.espn.com/espn/rss/nba/news"],
}

SPORT_KEYWORDS = {
    "lol": (
        "league of legends", "lol", "lcs", "lec", "lck", "lpl", "msi",
        "worlds", "summoner", "champion", "patch", "lta", "lcp",
    ),
    "nba": (
        "nba", "basketball", "lakers", "celtics", "warriors", "mavericks",
        "nuggets", "thunder", "knicks", "heat", "bucks", "suns",
        "clippers", "76ers", "playoffs", "regular season",
    ),
}


class AnalystFeedProvider:
    """Fetch public analyst/news RSS items for LoL and NBA research context."""

    def __init__(self, sport: str, urls: list[str] | None = None):
        self.sport = sport.casefold()
        configured = [value.strip() for value in os.getenv("ANALYST_RSS_URLS", "").split(",") if value.strip()]
        self.urls = urls or configured or DEFAULT_FEEDS.get(self.sport, [])

    @staticmethod
    def _normalized(value: str) -> str:
        return value.casefold()

    def _matches(self, title: str) -> bool:
        lowered = self._normalized(title)
        return any(keyword in lowered for keyword in SPORT_KEYWORDS.get(self.sport, ()))

    def recent(self, hours: int = 24) -> list[AnalystNote]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        rows: list[AnalystNote] = []
        for url in self.urls:
            request = Request(url, headers={"User-Agent": "PredictionAgent/0.2", "Accept": "application/rss+xml"})
            with urlopen(request, timeout=15) as response:
                root = ET.fromstring(response.read())
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if not title or not self._matches(title):
                    continue
                parsed = email.utils.parsedate_to_datetime(item.findtext("pubDate") or "")
                published = parsed.astimezone(timezone.utc) if parsed else datetime.now(timezone.utc)
                if published < cutoff:
                    continue
                rows.append(AnalystNote(url, self.sport, title, link, published))
        return rows
