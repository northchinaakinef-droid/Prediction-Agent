from __future__ import annotations

import email.utils
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.request import Request, urlopen


@dataclass
class NewsItem:
    source: str
    title: str
    link: str
    published_at: datetime
    importance: int
    category: str


KEYWORDS = {
    "lineup": 35, "starting lineup": 40, "roster": 30, "substitute": 35,
    "out": 30, "injury": 35, "questionable": 25, "doubtful": 30,
    "suspended": 45, "pause": 35, "technical issue": 40, "illness": 35,
    "ejected": 45, "minutes restriction": 35,
}


class RssNewsProvider:
    def __init__(self, urls: list[str] | None = None):
        configured = [value.strip() for value in os.getenv("NEWS_RSS_URLS", "").split(",") if value.strip()]
        self.urls = urls or configured or ["https://www.espn.com/espn/rss/nba/news"]

    def recent(self, hours: int = 24) -> list[NewsItem]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        rows = []
        for url in self.urls:
            request = Request(url, headers={"User-Agent": "PredictionAgent/0.2", "Accept": "application/rss+xml"})
            with urlopen(request, timeout=15) as response:
                root = ET.fromstring(response.read())
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                parsed = email.utils.parsedate_to_datetime(item.findtext("pubDate") or "")
                published = parsed.astimezone(timezone.utc) if parsed else datetime.now(timezone.utc)
                if published < cutoff:
                    continue
                lowered = title.casefold()
                hits = [(keyword, score) for keyword, score in KEYWORDS.items() if keyword in lowered]
                importance = max((score for _, score in hits), default=10)
                category = max(hits, key=lambda value: value[1])[0] if hits else "general"
                rows.append(NewsItem(url, title, link, published, importance, category))
        return rows
