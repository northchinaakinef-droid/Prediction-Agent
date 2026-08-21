from __future__ import annotations

import email.utils
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from ..entities import canonical_team, normalized_name, team_entities


@dataclass
class AnalystNote:
    source: str
    sport: str
    title: str
    link: str
    published_at: datetime
    summary: str = ""


DEFAULT_FEEDS = {
    "lol": ["https://dotesports.com/feed"],
    "nba": ["https://www.espn.com/espn/rss/nba/news"],
}

CHINESE_LOL_PAGES = (
    ("虎扑", "https://bbs.hupu.com/lol"),
    ("直播吧", "https://news.zhibo8.com/game/"),
)


class _PostLinkParser(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.current_href = ""
        self.current_text: list[str] = []
        self.rows: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self.current_href = dict(attrs).get("href") or ""
            self.current_text = []

    def handle_data(self, data: str) -> None:
        if self.current_href:
            self.current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self.current_href:
            return
        title = " ".join("".join(self.current_text).split())
        href = urljoin(self.base_url, self.current_href)
        if title and re.search(r"(?:/\d{6,}(?:-\d+)?\.html|/game/.*\.htm)", href):
            self.rows.append((title, href))
        self.current_href = ""
        self.current_text = []


class ChineseLolPostProvider:
    """Public Chinese LoL posts used as attributed context, never as labels."""

    def __init__(self, pages: tuple[tuple[str, str], ...] | None = None):
        self.pages = CHINESE_LOL_PAGES if pages is None else pages

    def recent(self, hours: int = 24) -> list[AnalystNote]:
        del hours  # Index pages are already ordered by recency; event matching filters them.
        now = datetime.now(timezone.utc)
        notes: list[AnalystNote] = []
        seen: set[str] = set()
        for source, url in self.pages:
            try:
                request = Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                })
                with urlopen(request, timeout=15) as response:
                    raw = response.read()
                    encoding = response.headers.get_content_charset() or "utf-8"
                parser = _PostLinkParser(url)
                parser.feed(raw.decode(encoding, errors="replace"))
            except Exception:
                # Hupu and Zhibo8 fail independently; one blocked source must
                # not erase posts already collected from the other.
                continue
            for title, link in parser.rows:
                if link in seen:
                    continue
                seen.add(link)
                notes.append(AnalystNote(source, "lol", title, link, now))
        return notes[:200]


def note_matches_event(note: AnalystNote, sport: str, team_a: str, team_b: str) -> bool:
    """Match a post using canonical names and registered abbreviations."""
    title = normalized_name(note.title)

    def aliases(team: str) -> set[str]:
        canonical = canonical_team(sport, team)
        values = {team, canonical}
        for entity in team_entities():
            if entity.get("sport") == sport and entity.get("canonical_name") == canonical:
                values.update(entity.get("aliases", []))
                values.add(str(entity.get("team_id") or ""))
        return {normalized_name(value) for value in values if len(normalized_name(value)) >= 2}

    return any(value in title for value in aliases(team_a)) or any(value in title for value in aliases(team_b))

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
