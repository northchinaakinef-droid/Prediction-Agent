from __future__ import annotations

import hashlib
import html
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from .entities import canonical_team, normalized_name


REPORT_ZONE = "Asia/Singapore"
TARGET_LOL_LEAGUES = ("LPL", "LCK", "LEC", "LTA", "MSI", "WORLDS", "FIRST STAND")


def _fetch_text(url: str, timeout: float = 30) -> str:
    request = Request(url, headers={"Accept": "text/html", "User-Agent": "PredictionAgent/0.1 schedule-audit"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def _clean(value: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", value)).replace("\xa0", " ").split())


def canonical_league(value: str) -> str:
    upper = _clean(value).upper()
    aliases = {
        "LCS": "LTA", "LTA N": "LTA", "LTA S": "LTA", "LTA NORTH": "LTA",
        "WORLD CHAMPIONSHIP": "WORLDS", "LOL WORLD CHAMPIONSHIP": "WORLDS",
        "FIRSTSTAND": "FIRST STAND",
    }
    for name in TARGET_LOL_LEAGUES:
        if name in upper:
            return name
    return aliases.get(upper, upper.split()[0] if upper else "UNKNOWN")


@dataclass
class CanonicalMatch:
    match_id: str
    sport: str
    league: str
    team_a: str
    team_b: str
    start_time: datetime
    best_of: int | None
    event_name: str
    sources: list[str] = field(default_factory=list)
    market_id: str | None = None
    market_mapping_status: str = "MARKET_NOT_FOUND"
    watcher_status: str = "DISCOVERED"
    analysis_status: str = "UNAVAILABLE"
    missing_stage: str | None = None
    missing_reason: str | None = None

    def as_dict(self) -> dict:
        row = asdict(self)
        row["start_time"] = self.start_time.astimezone(timezone.utc).isoformat()
        return row


@dataclass
class SourceResult:
    name: str
    available: bool
    matches: list[CanonicalMatch]
    error: str | None = None


def make_match(*, source: str, league: str, team_a: str, team_b: str,
               start_time: datetime, event_name: str, best_of: int | None = None) -> CanonicalMatch:
    a, b = canonical_team("lol", team_a), canonical_team("lol", team_b)
    start = start_time.astimezone(timezone.utc)
    key = "|".join(("lol", canonical_league(league), normalized_name(a), normalized_name(b), start.isoformat()))
    return CanonicalMatch(
        hashlib.sha256(key.encode()).hexdigest()[:20], "lol", canonical_league(league),
        a, b, start, best_of, _clean(event_name), [source],
    )


def parse_nextmatch(html_text: str, report_day: date, zone: ZoneInfo) -> list[CanonicalMatch]:
    matches: list[CanonicalMatch] = []
    cards = re.split(r'<div data-slot="card"', html_text)[1:]
    for card in cards:
        start_match = re.search(r'<time[^>]+datetime="([^"]+)"', card)
        detail = re.search(
            r'<a href="/([^/"]+)/schedule/"[^>]*>(.*?)</a>\s*<span>(.*?)</span>', card, re.S,
        )
        if not start_match or not detail:
            continue
        start = datetime.fromisoformat(start_match.group(1).replace("Z", "+00:00"))
        if start.astimezone(zone).date() != report_day:
            continue
        versus = _clean(detail.group(3))
        split = re.split(r"\s+vs\.?\s+", versus, maxsplit=1, flags=re.I)
        if len(split) != 2:
            continue
        # Some cards prefix a round label (for example "Seeding match 1:") to team A.
        split[0] = split[0].rsplit(":", 1)[-1].strip()
        event_name = _clean(detail.group(2))
        bo = re.search(r"\bBO(\d)\b", card[:5000], re.I)
        matches.append(make_match(
            source="nextmatch", league=detail.group(1), team_a=split[0], team_b=split[1],
            start_time=start, event_name=event_name, best_of=int(bo.group(1)) if bo else None,
        ))
    return matches


def parse_esportagenda(html_text: str, report_day: date, zone: ZoneInfo) -> list[CanonicalMatch]:
    # Nuxt serializes JSON-LD into its payload, so quotes may be escaped.
    text = html.unescape(html_text).replace('\\"', '"')
    pattern = re.compile(
        r'"@type":"SportsEvent","name":"([^"]+?) vs\.? ([^"]+?)".*?'
        r'"startDate":"([^"]+)".*?"organizer":\{"@type":"Organization","name":"([^"]+)"',
        re.S,
    )
    matches: list[CanonicalMatch] = []
    for team_a, team_b, start_text, league in pattern.findall(text):
        start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
        if start.astimezone(zone).date() != report_day:
            continue
        matches.append(make_match(
            source="esportagenda", league=league, team_a=team_a, team_b=team_b,
            start_time=start, event_name=league,
        ))
    return matches


class LolScheduleDiscovery:
    def __init__(self, fetch: Callable[[str], str] = _fetch_text):
        self.fetch = fetch
        self.zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", REPORT_ZONE))

    def discover(self, report_day: date) -> list[SourceResult]:
        results: list[SourceResult] = []
        try:
            text = self.fetch(os.getenv("NEXTMATCH_SCHEDULE_URL", "https://nextmatch.lol/schedule/"))
            results.append(SourceResult("nextmatch", True, parse_nextmatch(text, report_day, self.zone)))
        except Exception as error:
            results.append(SourceResult("nextmatch", False, [], repr(error)))

        urls = os.getenv(
            "ESPORTAGENDA_LOL_URLS",
            "https://www.esportagenda.com/leagues/league-of-legends/lck,"
            "https://www.esportagenda.com/leagues/league-of-legends/lpl,"
            "https://www.esportagenda.com/leagues/league-of-legends/lec,"
            "https://www.esportagenda.com/leagues/league-of-legends/lta",
        ).split(",")
        secondary: list[CanonicalMatch] = []
        errors: list[str] = []
        for url in filter(None, (value.strip() for value in urls)):
            try:
                secondary.extend(parse_esportagenda(self.fetch(url), report_day, self.zone))
            except Exception as error:
                errors.append(f"{url}: {error!r}")
        results.append(SourceResult("esportagenda", not errors, secondary, "; ".join(errors) or None))
        return results


def _same_match(left: CanonicalMatch, right: CanonicalMatch) -> bool:
    left_teams = {normalized_name(left.team_a), normalized_name(left.team_b)}
    right_teams = {normalized_name(right.team_a), normalized_name(right.team_b)}
    return left.league == right.league and left_teams == right_teams and abs(
        (left.start_time - right.start_time).total_seconds()
    ) <= 90 * 60


def reconcile_sources(results: Iterable[SourceResult]) -> tuple[list[CanonicalMatch], list[dict]]:
    expected: list[CanonicalMatch] = []
    for result in results:
        if not result.available:
            continue
        for item in result.matches:
            existing = next((row for row in expected if _same_match(row, item)), None)
            if existing:
                existing.sources = sorted(set(existing.sources + item.sources))
                existing.best_of = existing.best_of or item.best_of
            else:
                expected.append(item)
    disagreements = [
        {"match_id": item.match_id, "league": item.league, "event": f"{item.team_a} vs {item.team_b}",
         "sources": item.sources, "reason": "only one schedule source returned this match"}
        for item in expected if len(item.sources) < 2
    ]
    return sorted(expected, key=lambda row: row.start_time), disagreements


def _market_candidates(events: list[dict]) -> Iterable[tuple[dict, dict, datetime, list[str]]]:
    for event in events:
        for market in event.get("markets", []):
            if market.get("sportsMarketType") != "moneyline" or not market.get("gameStartTime"):
                continue
            outcomes = market.get("outcomes", [])
            outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
            if len(outcomes) != 2:
                continue
            start = datetime.fromisoformat(str(market["gameStartTime"]).replace("Z", "+00:00"))
            yield event, market, start, [canonical_team("lol", str(value)) for value in outcomes]


def match_markets(matches: list[CanonicalMatch], events: list[dict]) -> None:
    candidates = list(_market_candidates(events))
    for item in matches:
        scored = []
        reviews = []
        wanted = {normalized_name(item.team_a), normalized_name(item.team_b)}
        for event, market, start, teams in candidates:
            actual = {normalized_name(value) for value in teams}
            if abs((item.start_time - start).total_seconds()) > 90 * 60:
                continue
            if wanted == actual:
                scored.append((event, market))
                continue
            left, right = list(wanted), list(actual)
            direct = (SequenceMatcher(None, left[0], right[0]).ratio() +
                      SequenceMatcher(None, left[1], right[1]).ratio()) / 2
            crossed = (SequenceMatcher(None, left[0], right[1]).ratio() +
                       SequenceMatcher(None, left[1], right[0]).ratio()) / 2
            similarity = max(direct, crossed)
            if similarity >= 0.92:
                scored.append((event, market))
            elif similarity >= 0.75:
                reviews.append((event, market, similarity))
        if len(scored) == 1:
            event, market = scored[0]
            item.market_id = str(market.get("id") or event.get("id"))
            item.market_mapping_status = "MATCHED"
        elif len(scored) > 1 or reviews:
            item.market_mapping_status = "MARKET_MAPPING_REVIEW"
            item.missing_stage = "market_mapping"
            item.missing_reason = "ambiguous or below-threshold fuzzy market mapping"
        else:
            item.missing_stage = "market_mapping"
            item.missing_reason = "Polymarket moneyline market not found"


def update_watcher_registry(matches: list[CanonicalMatch], *, now: datetime, path: str | Path) -> dict:
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if registry_path.exists():
        existing = {row["match_id"]: row for row in json.loads(registry_path.read_text(encoding="utf-8"))}
    recovery = []
    for item in matches:
        if not item.market_id:
            item.watcher_status = "MISSING_MARKET"
        elif now >= item.start_time:
            item.watcher_status = "LIVE"
            if item.match_id not in existing:
                recovery.append({"match_id": item.match_id, "event": f"{item.team_a} vs {item.team_b}",
                                 "minutes_since_start": max(0, int((now-item.start_time).total_seconds()/60))})
        else:
            item.watcher_status = "WAITING"
        existing[item.match_id] = {
            "match_id": item.match_id, "league": item.league, "start_time": item.start_time.isoformat(),
            "status": item.watcher_status, "market_id": item.market_id,
            "watcher_status": item.watcher_status, "last_game_update": now.isoformat(),
            "last_market_update": now.isoformat() if item.market_id else None, "last_news_update": None,
        }
    registry_path.write_text(json.dumps(list(existing.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    return {"active": sum(row.watcher_status in {"WAITING", "LIVE"} for row in matches), "recovery": recovery}


def build_schedule_audit(results: list[SourceResult], market_events: list[dict], *,
                         report_day: date, now: datetime, registry_path: str | Path) -> dict:
    matches, disagreements = reconcile_sources(results)
    match_markets(matches, market_events)
    watcher = update_watcher_registry(matches, now=now, path=registry_path)
    by_league = {}
    for league in sorted(set(TARGET_LOL_LEAGUES) | {row.league for row in matches}):
        rows = [row for row in matches if row.league == league]
        expected = len(rows)
        market = sum(row.market_mapping_status == "MATCHED" for row in rows)
        watching = sum(row.watcher_status in {"WAITING", "LIVE"} for row in rows)
        by_league[league] = {
            "expected": expected, "discovered": expected, "market_matched": market,
            "watching": watching, "finished": sum(row.watcher_status == "FINISHED" for row in rows),
            "source_status": {result.name: "OK" if result.available else "DATA_UNAVAILABLE" for result in results},
        }
    total = len(matches)
    matched = sum(row.market_mapping_status == "MATCHED" for row in matches)
    watching = sum(row.watcher_status in {"WAITING", "LIVE"} for row in matches)
    unavailable = [asdict(result) | {"matches": len(result.matches)} for result in results if not result.available]
    incomplete = bool(unavailable or matched < total or watching < total)
    expected_live = sum(row.start_time <= now and row.watcher_status != "FINISHED" for row in matches)
    active_live_watchers = sum(row.watcher_status == "LIVE" for row in matches)
    return {
        "report_date": report_day.isoformat(), "timezone": str(ZoneInfo(os.getenv("REPORT_TIMEZONE", REPORT_ZONE))),
        "query_buffer_hours": 6, "minimum_schedule_coverage": 1.0,
        "expected": total, "discovered": total, "market_matched": matched, "watching": watching,
        "finished": 0, "coverage": (watching / total if total else (1.0 if not unavailable else 0.0)),
        "data_incomplete": incomplete,
        "status": "DATA_INCOMPLETE" if incomplete else ("COMPLETE_WITH_SOURCE_WARNINGS" if disagreements else "COMPLETE"),
        "source_warning": bool(disagreements),
        "watcher_health": {"expected_live": expected_live, "active_live": active_live_watchers,
                           "healthy": expected_live == active_live_watchers},
        "leagues": by_league, "source_errors": unavailable, "source_disagreements": disagreements,
        "missing": [row.as_dict() for row in matches if row.watcher_status not in {"WAITING", "LIVE"}],
        "matches": [row.as_dict() for row in matches], "monitoring_recovery": watcher["recovery"],
    }


def append_schedule_audit(audit: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(audit, ensure_ascii=False) + "\n")
