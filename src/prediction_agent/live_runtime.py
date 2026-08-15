from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from .entities import canonical_team, normalized_name
from .live_engine import LiveAlert, LiveAnalysisEngine, LiveStore, match_key
from .providers.live_data import (
    Bo3Cs2Provider, DataSourceUnavailable, EspnNbaProvider, GridOpenAccessProvider,
    LeaguepediaDraftProvider, NbaOfficialProvider, PandaScoreProvider, RiotEsportsProvider,
    TheSportsDbNbaProvider,
)
from .providers.polymarket import PolymarketClient
from .providers.news import RssNewsProvider
from .providers.analysts import AnalystFeedProvider
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
        report = json.loads(path.read_text(encoding="utf-8"))
        zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
        if report.get("report_date") != datetime.now(zone).date().isoformat():
            return {}
        return report

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

    def _prematch_alerts(self, report: dict, now: datetime,
                         analyst_notes: dict[str, list] | None = None) -> list[LiveAlert]:
        lead = max(10, int(os.getenv("PREMATCH_PUSH_MINUTES", "30")))
        analyst_notes = analyst_notes or {}
        alerts = []
        for sport in ("lol", "nba"):
            recommendations = [row for row in report.get("recommendations", []) if row.get("sport") == sport]
            audit = report.get("schedule_coverage", {}).get(sport, {})
            for match in audit.get("matches", []):
                try:
                    start = datetime.fromisoformat(str(match["start_time"]).replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if not now - timedelta(minutes=10) <= start <= now + timedelta(minutes=lead):
                    continue
                team_a = str(match.get("team_a") or "")
                team_b = str(match.get("team_b") or "")
                wanted = {normalized_name(canonical_team(sport, team_a)),
                          normalized_name(canonical_team(sport, team_b))}
                row = next((candidate for candidate in recommendations if all(
                    name and name in normalized_name(str(candidate.get("event") or "")) for name in wanted
                )), None)
                if not row or row.get("model_probability") is None:
                    continue
                outcome = str(row.get("outcome") or "")
                outcome_norm = normalized_name(canonical_team(sport, outcome))
                team_a_norm = normalized_name(canonical_team(sport, team_a))
                team_b_norm = normalized_name(canonical_team(sport, team_b))
                if outcome_norm not in {team_a_norm, team_b_norm}:
                    continue
                model_probability = float(row["model_probability"])
                blue_win_probability = (model_probability if outcome_norm == team_a_norm
                                        else 1 - model_probability)
                red_win_probability = 1 - blue_win_probability
                market_probability = (float(row["market_probability"])
                                      if row.get("market_probability") is not None else None)
                blue_market_probability = None
                red_market_probability = None
                if market_probability is not None:
                    blue_market_probability = (market_probability if outcome_norm == team_a_norm
                                               else 1 - market_probability)
                    red_market_probability = 1 - blue_market_probability
                reasons = list(row.get("reasons", []))[:3]
                title = f"{team_a} vs {team_b}"
                summary = (
                    f"赛前方向：{outcome}｜模型胜率：蓝方 {blue_win_probability:.1%}｜红方 {red_win_probability:.1%}"
                )
                notes = [note for note in analyst_notes.get(sport, [])
                         if team_a.casefold() in str(note.title).casefold()
                         or team_b.casefold() in str(note.title).casefold()]
                alerts.append(LiveAlert(
                    f"{sport}:{':'.join(sorted(wanted))}", sport, "IMPORTANT", 55, "PREMATCH_ANALYSIS",
                    title, summary, reasons, now, f"{sport}:{match.get('match_id')}:PREMATCH_ANALYSIS",
                    {
                        "outcome": outcome,
                        "team_a": team_a,
                        "team_b": team_b,
                        "blue_win_probability": blue_win_probability,
                        "red_win_probability": red_win_probability,
                        "blue_market_probability": blue_market_probability,
                        "red_market_probability": red_market_probability,
                        "reasons": reasons,
                        "analyst_count": len(notes),
                        "analyst_notes": [{"title": note.title, "link": note.link, "source": note.source}
                                          for note in notes[:3]],
                    },
                ))
        return alerts

    @staticmethod
    def _winner_side(state) -> str | None:
        if state.score_a is not None and state.score_b is not None:
            if state.score_a > state.score_b:
                return "a"
            if state.score_b > state.score_a:
                return "b"
        side = state.features.get("winner_side")
        if side in {"a", "b"}:
            return side
        winner = str(state.features.get("winner") or "")
        if winner:
            if normalized_name(canonical_team(state.sport, winner)) == normalized_name(
                canonical_team(state.sport, state.team_a)):
                return "a"
            if normalized_name(canonical_team(state.sport, winner)) == normalized_name(
                canonical_team(state.sport, state.team_b)):
                return "b"
        return None

    def _prematch_side(self, state) -> str | None:
        report = self._report()
        if not report:
            return None
        recommendations = [row for row in report.get("recommendations", []) if row.get("sport") == state.sport]
        wanted = {normalized_name(canonical_team(state.sport, state.team_a)),
                  normalized_name(canonical_team(state.sport, state.team_b))}
        row = next((candidate for candidate in recommendations if all(
            name and name in normalized_name(str(candidate.get("event") or "")) for name in wanted
        )), None)
        if not row or row.get("model_probability") is None:
            return None
        outcome_norm = normalized_name(canonical_team(state.sport, str(row.get("outcome") or "")))
        team_a_norm = normalized_name(canonical_team(state.sport, state.team_a))
        team_b_norm = normalized_name(canonical_team(state.sport, state.team_b))
        if outcome_norm == team_a_norm:
            return "a"
        if outcome_norm == team_b_norm:
            return "b"
        return None

    def _result_review_alerts(self, states: list, now: datetime,
                               analyst_notes: dict[str, list] | None = None) -> list[LiveAlert]:
        analyst_notes = analyst_notes or {}
        alerts = []
        for state in states:
            if state.sport not in {"lol", "nba"}:
                continue
            if not (state.finished or str(state.status).casefold() in {"finished", "completed", "post", "final"}):
                continue
            actual_side = self._winner_side(state)
            if actual_side is None:
                continue
            prematch_side = self._prematch_side(state)
            bp_probability = state.features.get("post_draft_probability") if state.sport == "lol" else None
            if prematch_side is None and bp_probability is None:
                continue
            actual_team = state.team_a if actual_side == "a" else state.team_b
            bp_side = None
            if bp_probability is not None:
                bp_probability = float(bp_probability)
                bp_side = "a" if bp_probability >= .5 else "b"
            prematch_team = state.team_a if prematch_side == "a" else state.team_b if prematch_side == "b" else None
            bp_team = state.team_a if bp_side == "a" else state.team_b if bp_side == "b" else None

            reasons = []
            if prematch_side is not None:
                result = "正确" if prematch_side == actual_side else "错误"
                reasons.append(f"赛前预测：{prematch_team}；判断{result}。")
            else:
                reasons.append("赛前预测：未生成有效概率。")
            if bp_team is not None:
                result = "正确" if bp_side == actual_side else "错误"
                reasons.append(
                    f"BP后预测：蓝方 {bp_probability:.1%}｜红方 {1-bp_probability:.1%}；判断{result}。"
                )
            if (prematch_side == actual_side) or (bp_side == actual_side):
                reasons.append("复盘：比赛走势与预测方向一致，优势方按预期滚动并转化为胜利。")
            else:
                reasons.append("复盘：比赛走势偏离预测，需重点核查阵容克制、资源节奏或临场发挥。")

            notes = [note for note in analyst_notes.get(state.sport, [])
                     if state.team_a.casefold() in str(note.title).casefold()
                     or state.team_b.casefold() in str(note.title).casefold()]
            key = match_key(state)
            alerts.append(LiveAlert(
                key, state.sport, "IMPORTANT", 70, "POSTMATCH_REVIEW",
                f"{state.team_a} vs {state.team_b}",
                f"比赛已结束，实际胜者：{actual_team}。",
                reasons, now, f"{key}:POSTMATCH_REVIEW:{state.observed_at.isoformat()}",
                {
                    "actual_winner": actual_team,
                    "team_a": state.team_a,
                    "team_b": state.team_b,
                    "score_a": state.score_a,
                    "score_b": state.score_b,
                    "actual_side": actual_side,
                    "prematch_side": prematch_side,
                    "prematch_team": prematch_team,
                    "bp_side": bp_side,
                    "bp_team": bp_team,
                    "bp_probability": bp_probability,
                    "analyst_count": len(notes),
                    "analyst_notes": [{"title": note.title, "link": note.link, "source": note.source}
                                      for note in notes[:3]],
                },
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
                if state.finished or str(state.status).casefold() in {"finished", "completed", "post", "final"}:
                    continue
                champions_a = tuple(value for value in state.features.get("champions_a", []) if value)
                champions_b = tuple(value for value in state.features.get("champions_b", []) if value)
                if len(champions_a) == len(champions_b) == 5:
                    before_champions_a = tuple(value for value in (before_state.get("features") or {}).get("champions_a", []) if value)
                    before_champions_b = tuple(value for value in (before_state.get("features") or {}).get("champions_b", []) if value)
                    if before_champions_a == champions_a and before_champions_b == champions_b:
                        continue
                    draft_id = ":".join(sorted(champions_a + champions_b))
                    post = state.features.get("post_draft_probability")
                    readout = state.features.get("draft_readout")
                    draft_summary = (f"BP 后模型胜率：蓝方 {post:.1%}｜红方 {1-post:.1%}"
                                     if post is not None else summary)
                    details = {
                        "blue_champions": list(champions_a),
                        "red_champions": list(champions_b),
                        "post_draft_probability": post,
                        "readout": readout,
                    }
                    alerts.append(LiveAlert(
                        key, "lol", "IMPORTANT", 60, "DRAFT_ANALYSIS", state.team_a + " vs " + state.team_b,
                        draft_summary, ["蓝方：" + "、".join(champions_a), "红方：" + "、".join(champions_b)],
                        state.observed_at, f"{key}:DRAFT:{draft_id}", details,
                    ))
        return alerts

    def _watcher_alerts(self, report: dict, states: list, now: datetime) -> list[LiveAlert]:
        grace = timedelta(minutes=max(5, int(os.getenv("WATCHER_START_GRACE_MINUTES", "10"))))
        window = timedelta(minutes=max(30, int(os.getenv("WATCHER_MISSING_WINDOW_MINUTES", "240"))))
        zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
        active = {match_key(state) for state in states if state.status == "LIVE"}
        finished = {match_key(state) for state in states if state.status == "FINISHED" or state.finished}
        expected: dict[str, tuple[str, dict]] = {}
        for sport in ("lol", "nba"):
            for match in report.get("schedule_coverage", {}).get(sport, {}).get("matches", []):
                try:
                    start = datetime.fromisoformat(str(match["start_time"]).replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if start + grace <= now <= start + window and str(match.get("event_status", "")).casefold() not in {
                    "finished", "completed", "post", "final"
                }:
                    teams = sorted((normalized_name(canonical_team(sport, str(match.get("team_a") or ""))),
                                    normalized_name(canonical_team(sport, str(match.get("team_b") or "")))))
                    expected[f"{sport}:{teams[0]}:{teams[1]}"] = (sport, match)
        missing = set(expected) - active - finished
        alerts = []
        for key in sorted(missing - self.missing_watchers):
            sport, match = expected[key]
            match_start = datetime.fromisoformat(str(match["start_time"]).replace("Z", "+00:00"))
            start_display = match_start.astimezone(zone).strftime("%H:%M")
            summary = (f"比赛预计 {start_display} 已开始，但当前没有活跃监控器；"
                       "若比赛已结束可忽略本提醒。")
            alerts.append(LiveAlert(
                key, sport, "EMERGENCY", 90, "WATCHER_MISSING",
                f"{match.get('team_a')} vs {match.get('team_b')}",
                summary, ["已标记为数据不完整，正在持续重试。"], now,
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
                "LOL_TARGET_LEAGUES", "LPL,LCK,LEC,LCS,LTA,LCP,MSI,Worlds,First Stand,EWC,Esports World Cup"
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
        report = self._report()
        if not report:
            return {}
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
            state.features["draft_readout"] = model.draft_readout(game)

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
        analyst_notes = {
            sport: self._attempt(f"analyst_{sport}", lambda s=sport: AnalystFeedProvider(s).recent())
            for sport in ("lol", "nba")
        }
        for state in states:
            notes = [note for note in analyst_notes.get(state.sport, [])
                     if state.team_a.casefold() in note.title.casefold()
                     or state.team_b.casefold() in note.title.casefold()]
            if notes:
                state.features["analyst_notes"] = [{"title": note.title, "link": note.link,
                                                    "source": note.source} for note in notes[:5]]
        previous = {match_key(state): self.store.previous(match_key(state)) for state in states}
        priors = self._priors(states, market_events)
        alerts = self.engine.process(states, market_events, priors,
                                     alert_sports={"nba", "cs2"})
        for alert in alerts:
            if self.on_alert:
                self.on_alert(alert)
        lifecycle = (self._prematch_alerts(report, now, analyst_notes) +
                     self._lifecycle_alerts(states, previous) +
                     self._result_review_alerts(states, now, analyst_notes) +
                     self._watcher_alerts(report, states, now))
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
