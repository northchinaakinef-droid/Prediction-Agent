from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .entities import canonical_team, normalized_name
from .live_engine import LiveAlert, LiveAnalysisEngine, LiveStore, match_key
from .providers.live_data import (
    Bo3Cs2Provider, DataSourceUnavailable, EspnNbaProvider, GridOpenAccessProvider,
    LeaguepediaDraftProvider, NbaOfficialProvider, PandaScoreProvider, RiotEsportsProvider,
    TheSportsDbNbaProvider,
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
        self.missing_watchers: set[str] = set()

    def _report(self) -> dict:
        path = self.root / "reports" / "daily.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _emit_once(self, alert: LiveAlert) -> bool:
        if self.store.alert_exists(alert.dedupe_key):
            return False
        self.store.save_alert(alert)
        if self.on_alert:
            self.on_alert(alert)
        return True

    @staticmethod
    def _probability_summary(snapshot: dict | None) -> str:
        if not snapshot or snapshot.get("current_probability") is None:
            return "当前概率暂不可用，比赛仍保持监控。"
        model = f"{float(snapshot['current_probability']):.1%}"
        market = (f"{float(snapshot['market_probability']):.1%}"
                  if snapshot.get("market_probability") is not None else "暂无对应市场")
        return f"当前模型胜率：{model}｜市场胜率：{market}"

    def _prematch_alerts(self, report: dict, now: datetime) -> list[LiveAlert]:
        lead = max(10, int(os.getenv("PREMATCH_PUSH_MINUTES", "30")))
        recommendations = [row for row in report.get("recommendations", []) if row.get("sport") == "lol"]
        audit = report.get("schedule_coverage", {}).get("lol", {})
        alerts = []
        for match in audit.get("matches", []):
            try:
                start = datetime.fromisoformat(str(match["start_time"]).replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if not now - timedelta(minutes=10) <= start <= now + timedelta(minutes=lead):
                continue
            wanted = {normalized_name(canonical_team("lol", str(match.get("team_a") or ""))),
                      normalized_name(canonical_team("lol", str(match.get("team_b") or "")))}
            row = next((candidate for candidate in recommendations if all(
                name and name in normalized_name(str(candidate.get("event") or "")) for name in wanted
            )), None)
            title = f"{match.get('team_a')} vs {match.get('team_b')}"
            if row and row.get("model_probability") is not None:
                summary = (
                    f"研究方向：{row.get('outcome') or '-'}｜赛前模型胜率：{float(row['model_probability']):.1%}｜"
                    f"市场胜率：{float(row['market_probability']):.1%}"
                    if row.get("market_probability") is not None else
                    f"研究方向：{row.get('outcome') or '-'}｜赛前模型胜率：{float(row['model_probability']):.1%}｜市场暂无报价"
                )
                reasons = list(row.get("reasons", []))[:3]
            else:
                summary = "比赛已发现，但当前没有满足条件的赛前概率；不会将比赛隐藏。"
                reasons = [str(match.get("missing_reason") or "市场映射、阵容或模型数据不足")]
            alerts.append(LiveAlert(
                f"lol:{':'.join(sorted(wanted))}", "lol", "OBSERVE", 30, "PREMATCH_ANALYSIS",
                title, summary, reasons, now, f"lol:{match.get('match_id')}:PREMATCH_ANALYSIS",
            ))
        return alerts

    def _lifecycle_alerts(self, states: list, previous: dict[str, dict | None]) -> list[LiveAlert]:
        alerts = []
        for state in states:
            key = match_key(state)
            before = previous.get(key)
            before_state = json.loads(before["state_json"]) if before and before.get("state_json") else {}
            snapshot = self.store.previous(key)
            summary = self._probability_summary(snapshot)
            if state.sport == "nba":
                prior_status = str(before_state.get("status") or "")
                if state.status == "LIVE" and prior_status != "LIVE":
                    alerts.append(LiveAlert(key, "nba", "IMPORTANT", 60, "MATCH_START", state.team_a + " vs " + state.team_b,
                                            summary, ["比赛已开始，实时分析已建立。"], state.observed_at,
                                            key + ":MATCH_START"))
                prior_period = int((before_state.get("features") or {}).get("period") or 0)
                period = int(state.features.get("period") or 0)
                if state.status == "LIVE" and prior_status == "LIVE" and period > prior_period:
                    alerts.append(LiveAlert(key, "nba", "OBSERVE", 35, "PERIOD_UPDATE", state.team_a + " vs " + state.team_b,
                                            summary, [f"第 {period} 节已开始。", f"当前比分：{state.score_a:.0f} - {state.score_b:.0f}"],
                                            state.observed_at, f"{key}:PERIOD:{period}"))
                prior_clock = float((before_state.get("features") or {}).get("game_clock_seconds") or 9999)
                clock = float(state.features.get("game_clock_seconds") or 0)
                if state.status == "LIVE" and period >= 4 and 0 < clock <= 120 < prior_clock:
                    alerts.append(LiveAlert(key, "nba", "IMPORTANT", 65, "CLUTCH_TIME", state.team_a + " vs " + state.team_b,
                                            summary, [f"末节剩余 {clock:.0f} 秒。", f"当前比分：{state.score_a:.0f} - {state.score_b:.0f}"],
                                            state.observed_at, key + ":CLUTCH_TIME"))
                if state.status == "FINISHED" and prior_status != "FINISHED":
                    alerts.append(LiveAlert(key, "nba", "IMPORTANT", 60, "MATCH_FINISHED", state.team_a + " vs " + state.team_b,
                                            f"最终比分：{state.score_a:.0f} - {state.score_b:.0f}", [summary], state.observed_at,
                                            key + ":MATCH_FINISHED"))
            elif state.sport == "lol":
                champions_a = tuple(value for value in state.features.get("champions_a", []) if value)
                champions_b = tuple(value for value in state.features.get("champions_b", []) if value)
                if len(champions_a) == len(champions_b) == 5:
                    draft_id = ":".join(sorted(champions_a + champions_b))
                    alerts.append(LiveAlert(
                        key, "lol", "IMPORTANT", 60, "DRAFT_ANALYSIS", state.team_a + " vs " + state.team_b,
                        summary, ["蓝方：" + "、".join(champions_a), "红方：" + "、".join(champions_b)],
                        state.observed_at, f"{key}:DRAFT:{draft_id}",
                    ))
        return alerts

    def _watcher_alerts(self, report: dict, states: list, now: datetime) -> list[LiveAlert]:
        grace = timedelta(minutes=max(5, int(os.getenv("WATCHER_START_GRACE_MINUTES", "10"))))
        active = {match_key(state) for state in states if state.status == "LIVE"}
        finished = {match_key(state) for state in states if state.status == "FINISHED" or state.finished}
        expected: dict[str, tuple[str, dict]] = {}
        for sport in ("lol", "nba"):
            for match in report.get("schedule_coverage", {}).get(sport, {}).get("matches", []):
                try:
                    start = datetime.fromisoformat(str(match["start_time"]).replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if start + grace <= now <= start + timedelta(hours=8) and str(match.get("event_status", "")).casefold() not in {
                    "finished", "completed", "post", "final"
                }:
                    teams = sorted((normalized_name(canonical_team(sport, str(match.get("team_a") or ""))),
                                    normalized_name(canonical_team(sport, str(match.get("team_b") or "")))))
                    expected[f"{sport}:{teams[0]}:{teams[1]}"] = (sport, match)
        missing = set(expected) - active - finished
        alerts = []
        for key in sorted(missing - self.missing_watchers):
            sport, match = expected[key]
            alerts.append(LiveAlert(
                key, sport, "EMERGENCY", 90, "WATCHER_MISSING",
                f"{match.get('team_a')} vs {match.get('team_b')}",
                "比赛预计已经开始，但没有活跃监控器。", ["已标记为数据不完整，正在持续重试。"], now,
                f"{key}:WATCHER_MISSING:{match.get('start_time')}",
            ))
        for key in sorted(self.missing_watchers - missing):
            alerts.append(LiveAlert(key, key.split(":", 1)[0], "IMPORTANT", 60, "MONITORING_RECOVERY", key,
                                    "实时监控已经自动恢复。", [], now, key + ":MONITORING_RECOVERY"))
        self.missing_watchers = missing
        return alerts

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
            ("thesportsdb_nba", lambda: TheSportsDbNbaProvider().live(today)),
        ]
        panda = PandaScoreProvider()
        grid = GridOpenAccessProvider()
        riot = RiotEsportsProvider()
        def riot_states():
            configured = [value.strip() for value in os.getenv("LOLESPORTS_LEAGUE_IDS", "").split(",") if value.strip()]
            target_names = [value.strip() for value in os.getenv(
                "LOL_TARGET_LEAGUES", "LPL,LCK,LEC,LCS,LTA,LCP,MSI,Worlds,First Stand"
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
            ("leaguepedia_bp", lambda: LeaguepediaDraftProvider().live()),
            ("pandascore_cs2", lambda: panda.live("cs2")),
        ])
        states_by_key = {}
        for name, function in groups:
            for state in self._attempt(name, function):
                key = match_key(state)
                existing = states_by_key.get(key)
                if existing is None:
                    states_by_key[key] = state
                    continue
                for field, value in state.features.items():
                    if value not in (None, [], ""):
                        existing.features[field] = value
                existing.key_events.extend(value for value in state.key_events if value not in existing.key_events)
                if existing.score_a is None:
                    existing.score_a = state.score_a
                if existing.score_b is None:
                    existing.score_b = state.score_b
                if state.status == "LIVE":
                    existing.status = "LIVE"
                    existing.finished = False
        return list(states_by_key.values())

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
        now = datetime.now(timezone.utc)
        report = self._report()
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
        previous = {match_key(state): self.store.previous(match_key(state)) for state in states}
        alerts = self.engine.process(states, market_events, self._priors(states, market_events),
                                     alert_sports={"nba", "cs2"})
        for alert in alerts:
            if self.on_alert:
                self.on_alert(alert)
        lifecycle = self._prematch_alerts(report, now) + self._lifecycle_alerts(states, previous) + self._watcher_alerts(
            report, states, now)
        emitted_lifecycle = [alert for alert in lifecycle if self._emit_once(alert)]
        alerts.extend(emitted_lifecycle)
        incomplete = [name for name, status in self.source_status.items() if not status["available"]]
        return {
            "checked_at": datetime.now(timezone.utc).isoformat(), "live_matches": len(states),
            "alerts": [row.as_dict() for row in alerts], "source_status": self.source_status,
            "watcher_health": {"expected_missing": len(self.missing_watchers),
                               "missing": sorted(self.missing_watchers)},
            "data_incomplete": bool(incomplete), "unavailable_sources": incomplete,
        }
