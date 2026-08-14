from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from .entities import canonical_team, normalized_name
from .live_engine import LiveAlert, LiveAnalysisEngine, LiveStore, match_key
from .providers.live_data import (
    Bo3Cs2Provider, DataSourceUnavailable, EspnNbaProvider, GridOpenAccessProvider,
    NbaOfficialProvider, PandaScoreProvider, RiotEsportsProvider,
)
from .providers.polymarket import PolymarketClient
from .providers.news import RssNewsProvider
from .lol_meta_model import LolDraftGame, load_lol_meta


TAGS = {"nba": "745", "lol": "65", "cs2": "100780"}


class LiveSupervisor:
    def __init__(self, *, root: str | Path, on_alert: Callable[[LiveAlert], None] | None = None):
        self.root = Path(root)
        self.on_alert = on_alert
        self.store = LiveStore(os.getenv("LIVE_DB_PATH", str(self.root / "data" / "daily" / "live.db")))
        self.engine = LiveAnalysisEngine(self.store)
        self.polymarket = PolymarketClient(timeout=15)
        self.source_status: dict[str, dict] = {}

    def _attempt(self, name: str, function):
        try:
            rows = function()
            self.source_status[name] = {"available": True, "rows": len(rows), "error": None,
                                        "checked_at": datetime.now(timezone.utc).isoformat()}
            return rows
        except Exception as error:
            self.source_status[name] = {"available": False, "rows": 0, "error": repr(error),
                                        "checked_at": datetime.now(timezone.utc).isoformat()}
            return []

    def collect_states(self) -> list:
        today = datetime.now(timezone.utc).date()
        # Priority order is official/game-server data first, then fixture-score fallbacks.
        groups = [
            ("nba_official", lambda: NbaOfficialProvider().live()),
            ("espn_nba", lambda: EspnNbaProvider().live(today)),
        ]
        panda = PandaScoreProvider()
        grid = GridOpenAccessProvider()
        riot = RiotEsportsProvider()
        def riot_states():
            configured = [value.strip() for value in os.getenv("LOLESPORTS_LEAGUE_IDS", "").split(",") if value.strip()]
            target_names = [value.strip() for value in os.getenv(
                "LOL_TARGET_LEAGUES", "LPL,LCK,LEC,LTA,MSI,Worlds,First Stand"
            ).split(",") if value.strip()]
            league_ids = configured or riot.league_ids(target_names)
            if not league_ids:
                raise DataSourceUnavailable("Riot Esports returned no configured target leagues")
            states = []
            for event in riot.schedule(league_ids):
                if event.status.casefold() not in {"inprogress", "completed"}:
                    continue
                game_ids = riot.game_ids(event.source_id)
                if game_ids:
                    states.append(riot.live_game(game_ids[-1], event.team_a, event.team_b))
            return states
        groups.append(("riot_esports", riot_states))

        def grid_states():
            events = grid.schedule(today)
            candidates = [row.source_id for row in events if row.start_time <= datetime.now(timezone.utc)]
            return grid.live(candidates)

        groups.extend([
            ("bo3_cs2", lambda: Bo3Cs2Provider().live()),
            ("grid_cs2", grid_states),
            ("pandascore_lol", lambda: panda.live("lol")),
            ("pandascore_cs2", lambda: panda.live("cs2")),
        ])
        states = []
        seen = set()
        for name, function in groups:
            for state in self._attempt(name, function):
                key = match_key(state)
                if key not in seen:
                    states.append(state)
                    seen.add(key)
        return states

    def _priors(self, states: list, market_events: dict[str, list[dict]]) -> dict[str, float]:
        path = self.root / "reports" / "daily.json"
        if not path.exists():
            return {}
        report = json.loads(path.read_text(encoding="utf-8"))
        priors = {}
        recommendations = {(str(row.get("sport")), str(row.get("event_id"))): row
                           for row in report.get("recommendations", [])}
        for state in states:
            key = match_key(state)
            wanted = {normalized_name(canonical_team(state.sport, state.team_a)),
                      normalized_name(canonical_team(state.sport, state.team_b))}
            event_id = None
            for event in market_events.get(state.sport, []):
                for market in event.get("markets", []):
                    outcomes = market.get("outcomes", [])
                    outcomes = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
                    actual = {normalized_name(canonical_team(state.sport, str(value))) for value in outcomes}
                    if market.get("sportsMarketType") == "moneyline" and actual == wanted:
                        event_id = str(event.get("id"))
                        break
                if event_id:
                    break
            row = recommendations.get((state.sport, event_id or ""))
            if not row or row.get("model_probability") is None:
                continue
            outcome = normalized_name(canonical_team(state.sport, str(row.get("outcome") or "")))
            probability = float(row["model_probability"])
            if outcome not in wanted:
                continue
            priors[key] = probability if outcome == normalized_name(
                canonical_team(state.sport, state.team_a)) else 1 - probability
        return priors

    def _add_lol_draft_probabilities(self, states: list) -> None:
        path = self.root / "artifacts" / "lol_meta_model.json"
        if not path.exists():
            return
        model, _ = load_lol_meta(path)
        patch = os.getenv("LOL_CURRENT_PATCH", "unknown")
        for state in states:
            if state.sport != "lol":
                continue
            champions_a = tuple(value for value in state.features.get("champions_a", []) if value)
            champions_b = tuple(value for value in state.features.get("champions_b", []) if value)
            team_a, team_b = canonical_team("lol", state.team_a), canonical_team("lol", state.team_b)
            players_a = tuple(model.latest_team_rosters.get(team_a, ()))
            players_b = tuple(model.latest_team_rosters.get(team_b, ()))
            if len(players_a) != 5 or len(players_b) != 5 or len(champions_a) != 5 or len(champions_b) != 5:
                continue
            game = LolDraftGame(
                f"live-{state.source_id}", state.observed_at, patch, "live", team_a, team_b,
                players_a, players_b, champions_a, champions_b, 0,
            )
            state.features["post_draft_probability"] = model.predict_post_draft(game)

    def scan_once(self) -> dict:
        states = self.collect_states()
        self._add_lol_draft_probabilities(states)
        news = self._attempt("news_rss", lambda: RssNewsProvider().recent())
        for state in states:
            matched = [item for item in news if state.team_a.casefold() in item.title.casefold() or
                       state.team_b.casefold() in item.title.casefold()]
            if matched:
                state.features["news_importance"] = max(item.importance for item in matched)
                state.key_events.extend(f"NEWS: {item.title}" for item in matched[:3])
        market_events = {
            sport: self._attempt(f"polymarket_{sport}", lambda tag=tag: self.polymarket.all_events_by_tag(tag))
            for sport, tag in TAGS.items()
        }
        alerts = self.engine.process(states, market_events, self._priors(states, market_events))
        for alert in alerts:
            if self.on_alert:
                self.on_alert(alert)
        incomplete = [name for name, status in self.source_status.items() if not status["available"]]
        return {
            "checked_at": datetime.now(timezone.utc).isoformat(), "live_matches": len(states),
            "alerts": [row.as_dict() for row in alerts], "source_status": self.source_status,
            "data_incomplete": bool(incomplete), "unavailable_sources": incomplete,
        }
