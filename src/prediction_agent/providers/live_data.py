from __future__ import annotations

import json
import os
import html
import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from ..entities import NBA, canonical_team
from .http import post_json


class DataSourceUnavailable(RuntimeError):
    pass


def _get_json(url: str, *, params: dict[str, object] | None = None,
              headers: dict[str, str] | None = None, timeout: float = 20) -> Any:
    if params:
        url = f"{url}?{urlencode(params)}"
    request_headers = {
        "Accept": "application/json", "User-Agent": "Mozilla/5.0 PredictionAgent/0.2",
        "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
    }
    request_headers.update(headers or {})
    request = Request(url, headers=request_headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise DataSourceUnavailable(f"{url}: {error!r}") from error


@dataclass
class DiscoveredEvent:
    source: str
    source_id: str
    sport: str
    league: str
    team_a: str
    team_b: str
    start_time: datetime
    status: str = "SCHEDULED"
    best_of: int | None = None
    event_name: str = ""

    def as_dict(self) -> dict:
        row = asdict(self)
        row["start_time"] = self.start_time.astimezone(timezone.utc).isoformat()
        return row


@dataclass
class LiveState:
    source: str
    source_id: str
    sport: str
    observed_at: datetime
    status: str
    team_a: str
    team_b: str
    score_a: float | None = None
    score_b: float | None = None
    period: str | None = None
    clock_seconds: float | None = None
    features: dict[str, Any] = field(default_factory=dict)
    key_events: list[str] = field(default_factory=list)
    finished: bool = False
    raw_available: bool = True

    def as_dict(self) -> dict:
        row = asdict(self)
        row["observed_at"] = self.observed_at.astimezone(timezone.utc).isoformat()
        return row


class NbaOfficialProvider:
    schedule_url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
    scoreboard_url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    headers = {"Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
               "Accept-Language": "en-US,en;q=0.9"}

    def schedule(self, day: date) -> list[DiscoveredEvent]:
        payload = _get_json(self.schedule_url, headers=self.headers)
        rows = []
        for block in payload.get("leagueSchedule", {}).get("gameDates", []):
            for game in block.get("games", []):
                start = datetime.fromisoformat(str(game["gameDateTimeUTC"]).replace("Z", "+00:00"))
                if start.date() < day - timedelta(days=1) or start.date() > day + timedelta(days=1):
                    continue
                rows.append(DiscoveredEvent(
                    "nba_official", str(game["gameId"]), "nba", "NBA",
                    game["awayTeam"]["teamName"], game["homeTeam"]["teamName"], start,
                    str(game.get("gameStatusText") or "SCHEDULED"), event_name="NBA",
                ))
        return rows

    def live(self) -> list[LiveState]:
        payload = _get_json(self.scoreboard_url, headers=self.headers)
        now = datetime.now(timezone.utc)
        states = []
        for game in payload.get("scoreboard", {}).get("games", []):
            away, home = game["awayTeam"], game["homeTeam"]
            clock = str(game.get("gameClock") or "")
            minutes = seconds = 0
            if "PT" in clock:
                import re
                parsed = re.search(r"PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?", clock)
                if parsed:
                    minutes, seconds = int(parsed.group(1) or 0), float(parsed.group(2) or 0)
            states.append(LiveState(
                "nba_official", str(game["gameId"]), "nba", now,
                "FINISHED" if game.get("gameStatus") == 3 else "LIVE" if game.get("gameStatus") == 2 else "WAITING",
                away["teamName"], home["teamName"], float(away.get("score") or 0), float(home.get("score") or 0),
                str(game.get("period") or ""), minutes * 60 + seconds,
                {"period": int(game.get("period") or 0), "game_clock_seconds": minutes * 60 + seconds},
                finished=game.get("gameStatus") == 3,
            ))
        return states


class EspnNbaProvider:
    base = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    headers = {"Referer": "https://www.espn.com/", "Origin": "https://www.espn.com",
               "Accept-Language": "en-US,en;q=0.9"}

    def schedule(self, day: date) -> list[DiscoveredEvent]:
        rows = []
        # ESPN's date parameter is not expressed in the report timezone. Query adjacent
        # dates, then let the caller select the Asia/Singapore natural day.
        for offset in (-1, 0, 1):
            query_day = day + timedelta(days=offset)
            payload = _get_json(self.base, params={"dates": query_day.strftime("%Y%m%d"), "limit": 100},
                                headers=self.headers)
            for event in payload.get("events", []):
                competition = event["competitions"][0]
                teams = {row["homeAway"]: row for row in competition["competitors"]}
                start = datetime.fromisoformat(str(event["date"]).replace("Z", "+00:00"))
                rows.append(DiscoveredEvent(
                    "espn", str(event["id"]), "nba", "NBA",
                    teams["away"]["team"]["displayName"], teams["home"]["team"]["displayName"], start,
                    str(event.get("status", {}).get("type", {}).get("name") or "SCHEDULED"), event_name="NBA",
                ))
        return list({row.source_id: row for row in rows}.values())

    def live(self, day: date | None = None) -> list[LiveState]:
        day = day or datetime.now(timezone.utc).date()
        payload = _get_json(self.base, params={"dates": day.strftime("%Y%m%d"), "limit": 100}, headers=self.headers)
        now = datetime.now(timezone.utc)
        states = []
        for event in payload.get("events", []):
            competition = event["competitions"][0]
            teams = {row["homeAway"]: row for row in competition["competitors"]}
            status = event.get("status", {})
            state = status.get("type", {}).get("state")
            clock = float(status.get("clock") or 0)
            states.append(LiveState(
                "espn", str(event["id"]), "nba", now,
                "FINISHED" if state == "post" else "LIVE" if state == "in" else "WAITING",
                teams["away"]["team"]["displayName"], teams["home"]["team"]["displayName"],
                float(teams["away"].get("score") or 0), float(teams["home"].get("score") or 0),
                str(status.get("period") or ""), clock,
                {"period": int(status.get("period") or 0), "game_clock_seconds": clock},
                finished=state == "post",
            ))
        return states


class TheSportsDbNbaProvider:
    """Anonymous crowd-sourced schedule fallback (public key documented by TheSportsDB)."""

    base = "https://www.thesportsdb.com/api/v1/json/123/eventsday.php"

    def schedule(self, day: date) -> list[DiscoveredEvent]:
        payload = _get_json(self.base, params={"d": day.isoformat(), "l": "NBA"})
        rows = []
        for event in payload.get("events") or []:
            if str(event.get("strLeague") or "").strip().upper() != "NBA":
                continue
            timestamp = event.get("strTimestamp")
            if not timestamp:
                continue
            start = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            rows.append(DiscoveredEvent(
                "thesportsdb", str(event.get("idEvent") or event.get("strEvent")), "nba", "NBA",
                str(event.get("strAwayTeam") or ""), str(event.get("strHomeTeam") or ""), start,
                str(event.get("strStatus") or "SCHEDULED"), event_name="NBA",
            ))
        return rows

    def live(self, day: date | None = None) -> list[LiveState]:
        day = day or datetime.now(ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))).date()
        payload = _get_json(self.base, params={"d": day.isoformat(), "l": "NBA"},
                            headers={"Referer": "https://www.thesportsdb.com/", "Origin": "https://www.thesportsdb.com"})
        now = datetime.now(timezone.utc)
        rows = []
        for event in payload.get("events") or []:
            if str(event.get("strLeague") or "").strip().upper() != "NBA":
                continue
            display_status = " ".join(str(value) for value in (
                event.get("strStatus"), event.get("strProgress")
            ) if value).strip()
            status_text = display_status.casefold()
            finished = (status_text in {"ft", "finished", "match finished", "final"} or
                        "match finished" in status_text or re.search(r"\bfinal\b", status_text) is not None)
            live = (any(token in status_text for token in ("quarter", "qtr", "halftime", "in progress", "ot")) or
                    re.search(r"\bq[1-4]\b", status_text) is not None)
            period_match = re.search(r"\bq([1-4])\b|\b([1-4])(?:st|nd|rd|th)?\s+(?:quarter|qtr)\b", status_text)
            period = int(next(value for value in period_match.groups() if value)) if period_match else 0
            clock_match = re.search(r"\b(\d{1,2}):(\d{2})\b", status_text)
            clock = int(clock_match.group(1)) * 60 + int(clock_match.group(2)) if clock_match else 0
            away_score = event.get("intAwayScore")
            home_score = event.get("intHomeScore")
            rows.append(LiveState(
                "thesportsdb", str(event.get("idEvent") or event.get("strEvent")), "nba", now,
                "FINISHED" if finished else "LIVE" if live else "WAITING",
                str(event.get("strAwayTeam") or ""), str(event.get("strHomeTeam") or ""),
                float(away_score) if away_score not in (None, "") else None,
                float(home_score) if home_score not in (None, "") else None,
                period=str(period or ""), clock_seconds=float(clock) if clock else None,
                features={"status_text": display_status, "period": period, "game_clock_seconds": clock},
                finished=finished,
            ))
        return rows


class SportSrcNbaProvider:
    """Anonymous upcoming basketball schedule, filtered through the NBA entity registry."""

    url = "https://api.sportsrc.org/"

    def schedule(self, day: date) -> list[DiscoveredEvent]:
        payload = _get_json(self.url, params={"data": "matches", "category": "basketball"})
        canonical_nba = set(NBA.values())
        rows = []
        for event in payload.get("data") or []:
            teams = event.get("teams") or {}
            away = str((teams.get("away") or {}).get("name") or "")
            home = str((teams.get("home") or {}).get("name") or "")
            if canonical_team("nba", away) not in canonical_nba or canonical_team("nba", home) not in canonical_nba:
                continue
            timestamp = float(event.get("date") or 0)
            if timestamp <= 0:
                continue
            rows.append(DiscoveredEvent(
                "sportsrc", str(event.get("id") or event.get("title")), "nba", "NBA",
                away, home, datetime.fromtimestamp(timestamp / 1000, timezone.utc),
                "SCHEDULED", event_name="NBA",
            ))
        return rows


class PandaScoreProvider:
    base = "https://api.pandascore.co"

    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("PANDASCORE_TOKEN")

    def _require(self) -> str:
        if not self.token:
            raise DataSourceUnavailable("PANDASCORE_TOKEN is not configured")
        return self.token

    def matches(self, sport: str, endpoint: str, *, day: date | None = None) -> list[dict]:
        prefix = "csgo" if sport == "cs2" else "lol"
        params: dict[str, object] = {"token": self._require(), "per_page": 100, "page": 1}
        if day:
            zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
            local_start = datetime.combine(day, datetime.min.time(), zone) - timedelta(hours=6)
            local_end = datetime.combine(day + timedelta(days=1), datetime.min.time(), zone) + timedelta(hours=6)
            params["range[begin_at]"] = (
                f"{local_start.astimezone(timezone.utc).isoformat()},"
                f"{local_end.astimezone(timezone.utc).isoformat()}"
            )
        return list(_get_json(f"{self.base}/{prefix}/matches/{endpoint}", params=params))

    def schedule(self, sport: str, day: date) -> list[DiscoveredEvent]:
        rows = []
        for endpoint in ("running", "upcoming"):
            for match in self.matches(sport, endpoint, day=day):
                opponents = [row.get("opponent") for row in match.get("opponents", []) if row.get("opponent")]
                if len(opponents) != 2 or not match.get("begin_at"):
                    continue
                rows.append(DiscoveredEvent(
                    "pandascore", str(match["id"]), sport, str(match.get("league", {}).get("name") or "UNKNOWN"),
                    opponents[0]["name"], opponents[1]["name"],
                    datetime.fromisoformat(match["begin_at"].replace("Z", "+00:00")),
                    str(match.get("status") or "SCHEDULED"), int(match.get("number_of_games") or 0) or None,
                    str(match.get("tournament", {}).get("name") or ""),
                ))
        return list({(row.source_id, row.sport): row for row in rows}.values())

    def live(self, sport: str) -> list[LiveState]:
        now = datetime.now(timezone.utc)
        rows = []
        for match in self.matches(sport, "running"):
            opponents = [row.get("opponent") for row in match.get("opponents", []) if row.get("opponent")]
            if len(opponents) != 2:
                continue
            scores = {str(row.get("team_id")): float(row.get("score") or 0) for row in match.get("results", [])}
            rows.append(LiveState(
                "pandascore", str(match["id"]), sport, now, "LIVE", opponents[0]["name"], opponents[1]["name"],
                scores.get(str(opponents[0]["id"])), scores.get(str(opponents[1]["id"])),
                features={"series_score_only": True},
            ))
        return rows


class LeaguepediaDraftProvider:
    """Public Cargo fallback for completed champion drafts; updates can be editor-delayed."""

    base = "https://lol.fandom.com/api.php"

    @staticmethod
    def _picks(value: object) -> list[str]:
        return [item.strip() for item in str(value or "").split(",") if item.strip()]

    def live(self) -> list[LiveState]:
        now = datetime.now(timezone.utc)
        since = (now - timedelta(hours=10)).strftime("%Y-%m-%d %H:%M:%S")
        payload = _get_json(self.base, params={
            "action": "cargoquery", "format": "json", "limit": 100,
            "tables": "ScoreboardGames=SG",
            "fields": "SG.Team1,SG.Team2,SG.DateTime_UTC,SG.Team1Picks,SG.Team2Picks,SG.GameId,SG.Tournament,SG.Winner",
            "where": f"SG.DateTime_UTC >= '{since}'", "order_by": "SG.DateTime_UTC DESC",
        }, headers={"User-Agent": "PredictionAgent/0.2"})
        targets = tuple(value.strip().casefold() for value in os.getenv(
            "LOL_TARGET_LEAGUES", "LPL,LCK,LEC,LCS,LTA,LCP,MSI,Worlds,First Stand"
        ).split(",") if value.strip())
        rows = []
        for result in payload.get("cargoquery", []):
            item = result.get("title", {})
            tournament = str(item.get("Tournament") or "")
            if targets and not any(target in tournament.casefold() for target in targets):
                continue
            champions_a, champions_b = self._picks(item.get("Team1Picks")), self._picks(item.get("Team2Picks"))
            if len(champions_a) != 5 or len(champions_b) != 5:
                continue
            finished = str(item.get("Winner") or "") in {"1", "2"}
            features = {"champions_a": champions_a, "champions_b": champions_b,
                        "draft_source_delayed": True}
            if finished:
                features["winner_side"] = "a" if str(item.get("Winner")) == "1" else "b"
            rows.append(LiveState(
                "leaguepedia", str(item.get("GameId") or f"{item.get('Team1')}:{item.get('Team2')}"),
                "lol", now, "FINISHED" if finished else "LIVE", str(item.get("Team1") or ""),
                str(item.get("Team2") or ""), features=features, finished=finished,
            ))
        latest = {}
        for row in rows:
            key = tuple(sorted((row.team_a.casefold(), row.team_b.casefold())))
            latest.setdefault(key, row)
        return list(latest.values())


class GridOpenAccessProvider:
    central_url = "https://api-op.grid.gg/central-data/graphql"
    state_url = "https://api-op.grid.gg/live-data-feed/series-state/graphql"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GRID_API_KEY")

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise DataSourceUnavailable("GRID_API_KEY is not configured; Open Access registration is required")
        return {"x-api-key": self.api_key}

    def schedule(self, day: date) -> list[DiscoveredEvent]:
        zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
        start = (datetime.combine(day, datetime.min.time(), zone) - timedelta(hours=6)).astimezone(timezone.utc)
        end = (datetime.combine(day + timedelta(days=1), datetime.min.time(), zone) +
               timedelta(hours=6)).astimezone(timezone.utc)
        query = """query AllSeries { allSeries(filter:{startTimeScheduled:{gte:\"%s\",lte:\"%s\"}},orderBy:StartTimeScheduled){edges{node{id startTimeScheduled teams{baseInfo{id name}} tournament{name id} title{nameShortened id}}} pageInfo{hasNextPage endCursor}}}""" % (start.isoformat(), end.isoformat())
        payload = post_json(self.central_url, {"query": query}, headers=self._headers())
        if payload.get("errors"):
            raise DataSourceUnavailable(str(payload["errors"]))
        rows = []
        for edge in payload.get("data", {}).get("allSeries", {}).get("edges", []):
            node = edge["node"]
            title_name = str(node.get("title", {}).get("nameShortened") or "")
            if not any(token in title_name.casefold() for token in ("counter-strike", "cs2", "cs:go", "csgo")):
                continue
            teams = node.get("teams", [])
            if len(teams) != 2:
                continue
            names = [row["baseInfo"]["name"] for row in teams]
            rows.append(DiscoveredEvent(
                "grid", str(node["id"]), "cs2", str(node.get("tournament", {}).get("name") or "CS2"),
                names[0], names[1], datetime.fromisoformat(node["startTimeScheduled"].replace("Z", "+00:00")),
                event_name=str(node.get("tournament", {}).get("name") or "CS2"),
            ))
        return rows

    def live(self, series_ids: list[str]) -> list[LiveState]:
        now = datetime.now(timezone.utc)
        rows = []
        for series_id in series_ids:
            query = """query SeriesState { seriesState(id:\"%s\"){id started finished teams{id name won score kills deaths} games{sequenceNumber finished teams{id name won score kills deaths}}}}""" % series_id
            payload = post_json(self.state_url, {"query": query}, headers=self._headers())
            if payload.get("errors"):
                raise DataSourceUnavailable(str(payload["errors"]))
            state = payload.get("data", {}).get("seriesState")
            if not state or len(state.get("teams", [])) != 2:
                continue
            teams = state["teams"]
            current_game = next((game for game in reversed(state.get("games", [])) if not game.get("finished")), None)
            game_teams = {str(row["id"]): row for row in (current_game or {}).get("teams", [])}
            features = {"series_score_a": teams[0].get("score"), "series_score_b": teams[1].get("score")}
            if current_game:
                features.update({
                    "map_number": current_game.get("sequenceNumber"),
                    "round_score_a": game_teams.get(str(teams[0]["id"]), {}).get("score"),
                    "round_score_b": game_teams.get(str(teams[1]["id"]), {}).get("score"),
                    "kills_a": game_teams.get(str(teams[0]["id"]), {}).get("kills"),
                    "kills_b": game_teams.get(str(teams[1]["id"]), {}).get("kills"),
                })
            rows.append(LiveState(
                "grid", str(series_id), "cs2", now,
                "FINISHED" if state.get("finished") else "LIVE" if state.get("started") else "WAITING",
                teams[0]["name"], teams[1]["name"], teams[0].get("score"), teams[1].get("score"),
                str(features.get("map_number") or ""), features=features, finished=bool(state.get("finished")),
            ))
        return rows


class Bo3Cs2Provider:
    """Anonymous BO3.gg JSON fallback for CS2 Tier 1/2 schedules and live series state."""

    base = "https://api.bo3.gg/api/v1"
    headers = {"Origin": "https://bo3.gg", "Referer": "https://bo3.gg/"}

    def matches(self, day: date, statuses: str = "upcoming,current,finished") -> list[dict]:
        zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
        start = datetime.combine(day, datetime.min.time(), zone) - timedelta(hours=6)
        end = datetime.combine(day + timedelta(days=1), datetime.min.time(), zone) + timedelta(hours=6)
        rows = []
        offset = 0
        while True:
            payload = _get_json(f"{self.base}/matches", params={
                "scope": "widget-matches", "page[offset]": offset, "page[limit]": 100,
                "sort": "tier_rank,-start_date", "filter[matches.status][in]": statuses,
                "filter[matches.start_date][lt]": end.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "filter[matches.start_date][gt]": start.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "filter[matches.discipline_id][eq]": 1, "with": "teams,tournament,games",
            }, headers=self.headers)
            page = list(payload.get("results") or [])
            rows.extend(page)
            if len(page) < 100:
                break
            offset += len(page)
        return rows

    @staticmethod
    def _in_scope(match: dict) -> bool:
        maximum = int(os.getenv("CS2_MAX_TIER_RANK", "2"))
        rank = match.get("tournament", {}).get("tier_rank")
        return rank is not None and int(rank) <= maximum

    def schedule(self, day: date) -> list[DiscoveredEvent]:
        rows = []
        for match in self.matches(day):
            if not self._in_scope(match) or not match.get("team1") or not match.get("team2"):
                continue
            rows.append(DiscoveredEvent(
                "bo3", str(match["id"]), "cs2", str(match.get("tournament", {}).get("name") or "CS2"),
                str(match["team1"]["name"]), str(match["team2"]["name"]),
                datetime.fromisoformat(str(match["start_date"]).replace("Z", "+00:00")),
                str(match.get("status") or "SCHEDULED"), int(match.get("bo_type") or 0) or None,
                str(match.get("tournament", {}).get("name") or "CS2"),
            ))
        return rows

    def live(self, day: date | None = None) -> list[LiveState]:
        day = day or datetime.now(ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))).date()
        now = datetime.now(timezone.utc)
        rows = []
        for match in self.matches(day, "current"):
            if not self._in_scope(match) or not match.get("team1") or not match.get("team2"):
                continue
            games = list(match.get("games") or [])
            current = next((game for game in reversed(games) if game.get("status") == "current"), None)
            features = {
                "series_score_a": match.get("team1_score"), "series_score_b": match.get("team2_score"),
                "map_number": (current or {}).get("number"),
                "round_score_a": (current or {}).get("team1_score"),
                "round_score_b": (current or {}).get("team2_score"),
            }
            rows.append(LiveState(
                "bo3", str(match["id"]), "cs2", now, "LIVE",
                str(match["team1"]["name"]), str(match["team2"]["name"]),
                float(match.get("team1_score") or 0), float(match.get("team2_score") or 0),
                str(features.get("map_number") or ""), features=features,
            ))
        return rows


class EsportAgendaCs2Provider:
    """Public HTML/JSON-LD cross-check for tournaments selected dynamically by BO3 tier metadata."""

    url = "https://www.esportagenda.com/cs2"

    def schedule(self, day: date, target_tournaments: set[str]) -> list[DiscoveredEvent]:
        request = Request(self.url, headers={"Accept": "text/html", "User-Agent": "PredictionAgent/0.2"})
        try:
            with urlopen(request, timeout=30) as response:
                text = html.unescape(response.read().decode("utf-8")).replace(r'\"', '"')
        except Exception as error:
            raise DataSourceUnavailable(f"{self.url}: {error!r}") from error
        pattern = re.compile(
            r'"@type":"SportsEvent".*?"name":"([^"]+?) vs\.? ([^"]+?)".*?'
            r'"startDate":"([^"]+)".*?"organizer":\{"@type":"Organization","name":"([^"]+)"',
            re.S,
        )
        zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
        wanted = {value.casefold() for value in target_tournaments}
        rows = []
        for team_a, team_b, start_text, tournament in pattern.findall(text):
            tournament_key = tournament.casefold()
            if not any(tournament_key in value or value in tournament_key for value in wanted):
                continue
            start = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            if start.astimezone(zone).date() != day:
                continue
            rows.append(DiscoveredEvent(
                "esportagenda_cs2", f"{team_a}:{team_b}:{start.isoformat()}", "cs2", tournament,
                team_a, team_b, start, event_name=tournament,
            ))
        return list({row.source_id: row for row in rows}.values())


class RiotEsportsProvider:
    api = "https://esports-api.lolesports.com/persisted/gw"
    feed = "https://feed.lolesports.com/livestats/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("LOLESPORTS_API_KEY")

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise DataSourceUnavailable("LOLESPORTS_API_KEY is not configured")
        return {"x-api-key": self.api_key}

    def league_ids(self, target_names: list[str]) -> list[str]:
        payload = _get_json(f"{self.api}/getLeagues", params={"hl": "en-US"}, headers=self._headers())
        targets = {value.casefold() for value in target_names}
        return [str(row["id"]) for row in payload.get("data", {}).get("leagues", [])
                if str(row.get("name") or "").casefold() in targets or
                str(row.get("slug") or "").casefold() in targets]

    def schedule(self, league_ids: list[str]) -> list[DiscoveredEvent]:
        payload = _get_json(f"{self.api}/getSchedule", params={"hl": "en-US", "leagueId": ",".join(league_ids)},
                            headers=self._headers())
        rows = []
        for event in payload.get("data", {}).get("schedule", {}).get("events", []):
            match = event.get("match") or {}
            teams = match.get("teams", [])
            if len(teams) != 2 or not event.get("startTime"):
                continue
            strategy = match.get("strategy", {})
            rows.append(DiscoveredEvent(
                "riot_esports", str(event["id"]), "lol", str(event.get("league", {}).get("name") or "UNKNOWN"),
                teams[0]["name"], teams[1]["name"], datetime.fromisoformat(event["startTime"].replace("Z", "+00:00")),
                str(event.get("state") or "SCHEDULED"), int(strategy.get("count") or 0) or None,
                str(event.get("blockName") or ""),
            ))
        return rows

    def game_ids(self, event_id: str) -> list[str]:
        payload = _get_json(f"{self.api}/getEventDetails", params={"hl": "en-US", "id": event_id},
                            headers=self._headers())
        games = payload.get("data", {}).get("event", {}).get("match", {}).get("games", [])
        return [str(game["id"]) for game in games if game.get("state") in {"inProgress", "completed"}]

    def live_game(self, game_id: str, team_a: str, team_b: str) -> LiveState:
        query_time = (datetime.now(timezone.utc) - timedelta(minutes=2)).replace(microsecond=0)
        payload = _get_json(f"{self.feed}/window/{game_id}", params={"startingTime": query_time.isoformat()})
        frames = payload.get("frames", [])
        if not frames:
            raise DataSourceUnavailable(f"LoL live feed has no frames for game {game_id}")
        frame = frames[-1]
        blue, red = frame["blueTeam"], frame["redTeam"]
        features = {
            "game_time_seconds": frame.get("rfc460Timestamp"),
            "gold_a": blue.get("totalGold"), "gold_b": red.get("totalGold"),
            "kills_a": blue.get("totalKills"), "kills_b": red.get("totalKills"),
            "towers_a": blue.get("towers"), "towers_b": red.get("towers"),
            "dragons_a": len(blue.get("dragons", [])), "dragons_b": len(red.get("dragons", [])),
            "barons_a": blue.get("barons"), "barons_b": red.get("barons"),
            "inhibitors_a": blue.get("inhibitors"), "inhibitors_b": red.get("inhibitors"),
            "champions_a": [row.get("championName") for row in blue.get("participants", [])],
            "champions_b": [row.get("championName") for row in red.get("participants", [])],
        }
        return LiveState(
            "riot_esports", game_id, "lol", datetime.now(timezone.utc), "LIVE",
            team_a, team_b,
            float(blue.get("totalKills") or 0), float(red.get("totalKills") or 0), features=features,
        )
