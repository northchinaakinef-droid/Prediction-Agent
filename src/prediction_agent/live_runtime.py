from __future__ import annotations

import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from .entities import canonical_team, normalized_name
from .live_engine import LiveAlert, LiveAnalysisEngine, LiveStore, match_key
from . import nba_analytics
from .providers.live_data import (
    Bo3Cs2Provider, DataSourceUnavailable, EspnNbaProvider, GridOpenAccessProvider, HupuNbaProvider,
    LeaguepediaDraftProvider, NbaBoxscoreProvider, NbaOfficialProvider, PandaScoreProvider,
    RiotEsportsProvider, TheSportsDbNbaProvider,
)
from .providers.polymarket import PolymarketClient
from .providers.news import RssNewsProvider
from .providers.analysts import AnalystFeedProvider
from .lol_meta_model import LolDraftGame, load_lol_meta
from .paper_store import record_post_match_review


TAGS = {"nba": "745", "lol": "65", "cs2": "100780"}


class LiveSupervisor:
    def __init__(self, *, root: str | Path, on_alert: Callable[[LiveAlert], None] | None = None):
        self.root = Path(root)
        self.on_alert = on_alert
        self.store = LiveStore(os.getenv("LIVE_DB_PATH", str(self.root / "data" / "daily" / "live.db")))
        self._collapse_legacy_postmatch_dedupe()
        self._collapse_legacy_prematch_dedupe()
        self.engine = LiveAnalysisEngine(self.store)
        self.polymarket = PolymarketClient(timeout=15)
        self.source_status: dict[str, dict] = {}
        self.missing_watchers: set[str] = set()

    def _collapse_legacy_postmatch_dedupe(self) -> None:
        """Make pre-fix post-match review alerts visible under the stable key.

        Older versions appended ``state.observed_at`` to the POSTMATCH_REVIEW
        dedupe key, so a restart with the fixed key would otherwise push each
        finished match one more time before settling down. Collapse those
        legacy keys into the stable per-match key once at startup.
        """
        for row in self.store.legacy_postmatch_dedupe_keys():
            dedupe_key = str(row.get("dedupe_key") or "")
            marker = dedupe_key.split(":POSTMATCH_REVIEW:", 1)[0] + ":POSTMATCH_REVIEW"
            if marker != dedupe_key:
                self.store.ensure_alert_marker(marker, str(row.get("observed_at") or ""))

    def _collapse_legacy_prematch_dedupe(self) -> None:
        """Map old match_id-based pre-match keys to the stable team-pair key.

        Pre-match analysis previously used ``sport:{match_id}:PREMATCH_ANALYSIS``,
        so a refreshed schedule could regenerate ``match_id`` and push the same
        LoL/NBA match more than once. The stored alert JSON still contains the
        canonical ``match_key``, which is the stable identity we use now.
        """
        for row in self.store.legacy_prematch_dedupe_rows():
            dedupe_key = str(row.get("dedupe_key") or "")
            try:
                alert = json.loads(str(row.get("alert_json") or "{}"))
            except json.JSONDecodeError:
                continue
            stable_key = str(alert.get("match_key") or "")
            if not stable_key:
                continue
            marker = f"{stable_key}:PREMATCH_ANALYSIS"
            if marker != dedupe_key:
                self.store.ensure_alert_marker(marker, str(row.get("observed_at") or ""))

    def _report(self) -> dict:
        path = self.root / "reports" / "daily.json"
        if not path.exists():
            return {}
        report = json.loads(path.read_text(encoding="utf-8"))
        zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
        local_today = datetime.now(zone).date()
        # A pre-match report is generated on the Asia/Singapore calendar day and
        # can cover matches that start after midnight local time.  Post-match
        # reviews for those matches therefore still need yesterday's report even
        # though today's report has not been generated yet.
        valid_dates = {local_today.isoformat(), (local_today - timedelta(days=1)).isoformat()}
        if report.get("report_date") not in valid_dates:
            return {}
        generated_at = report.get("generated_at")
        if generated_at:
            try:
                generated_dt = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
                if generated_dt.tzinfo is None:
                    generated_dt = generated_dt.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - generated_dt.astimezone(timezone.utc)) > timedelta(hours=48):
                    return {}
            except ValueError:
                pass
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

        def _team_pair(text: str):
            text = str(text or "")
            for prefix in ("LoL: ", "LOL: ", "NBA: ", "CS2: "):
                if text.startswith(prefix):
                    text = text.removeprefix(prefix)
            parts = re.split(r"\s+(?:vs\.?|VS\.?|对)\s+", text.strip(), maxsplit=1, flags=re.I)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
            return text.strip(), ""

        def _build_details(sport, team_a, team_b, row, notes):
            outcome = str(row.get("outcome") or "")
            outcome_norm = normalized_name(canonical_team(sport, outcome))
            team_a_norm = normalized_name(canonical_team(sport, team_a))
            team_b_norm = normalized_name(canonical_team(sport, team_b))
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
            return {
                "outcome": outcome,
                "team_a": team_a,
                "team_b": team_b,
                "blue_win_probability": blue_win_probability,
                "red_win_probability": red_win_probability,
                "blue_market_probability": blue_market_probability,
                "red_market_probability": red_market_probability,
                "ev": row.get("expected_value"),
                "lineup_a": row.get("lineup_a") or [],
                "lineup_b": row.get("lineup_b") or [],
                "recent_form_a": row.get("recent_form_a"),
                "recent_form_b": row.get("recent_form_b"),
                "patch_meta_heroes": row.get("patch_meta_heroes") or [],
                "meta_coverage_a": row.get("meta_coverage_a"),
                "meta_coverage_b": row.get("meta_coverage_b"),
                "format": row.get("format"),
                "map_strengths_a": row.get("map_strengths_a") or [],
                "map_strengths_b": row.get("map_strengths_b") or [],
                "sample_a": row.get("sample_a"),
                "sample_b": row.get("sample_b"),
                "bet_status": row.get("bet_status"),
                "real_bet_reason": row.get("real_bet_reason"),
                "analyst_count": len(notes),
                "analyst_notes": [{"title": note.title, "link": note.link, "source": note.source}
                                  for note in notes[:3]],
                "reasons": list(row.get("reasons", []))[:3],
            }

        for sport in ("lol", "nba"):
            recommendations = [row for row in report.get("recommendations", []) if row.get("sport") == sport]
            audit = report.get("schedule_coverage", {}).get(sport, {})
            emitted = set()
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
                title = f"{team_a} vs {team_b}"
                match_identity = f"{sport}:{':'.join(sorted(wanted))}"
                if str(row.get("lineup_status") or "") != "完整":
                    alerts.append(LiveAlert(
                        match_identity, sport, "OBSERVE", 35, "LINEUP_MISSING",
                        title,
                        f"{team_a} vs {team_b} 赛程已命中，但阵容尚未完整。",
                        [str(row.get("lineup_status") or "阵容未知")], now,
                        f"{match_identity}:LINEUP_MISSING",
                        {"team_a": team_a, "team_b": team_b,
                         "lineup_status": row.get("lineup_status")},
                    ))
                    emitted.add((sport, tuple(sorted(wanted))))
                    continue
                notes = [note for note in analyst_notes.get(sport, [])
                         if team_a.casefold() in str(note.title).casefold()
                         or team_b.casefold() in str(note.title).casefold()]
                details = _build_details(sport, team_a, team_b, row, notes)
                alerts.append(LiveAlert(
                    match_identity, sport, "IMPORTANT", 55, "PREMATCH_ANALYSIS",
                    title, "", list(details.get("reasons", [])), now, f"{match_identity}:PREMATCH_ANALYSIS",
                    details,
                ))
                emitted.add((sport, tuple(sorted(wanted))))

            for row in recommendations:
                if row.get("market_mapping_status") != "NOT_IN_SCHEDULE":
                    continue
                if not row.get("scheduled_start") or row.get("model_probability") is None:
                    continue
                try:
                    start = datetime.fromisoformat(str(row["scheduled_start"]).replace("Z", "+00:00"))
                except (KeyError, ValueError):
                    continue
                if not now - timedelta(minutes=10) <= start <= now + timedelta(minutes=lead):
                    continue
                team_a, team_b = _team_pair(row.get("event") or "")
                if not team_a or not team_b:
                    continue
                wanted = {normalized_name(canonical_team(sport, team_a)),
                          normalized_name(canonical_team(sport, team_b))}
                key = (sport, tuple(sorted(wanted)))
                if key in emitted:
                    continue
                notes = [note for note in analyst_notes.get(sport, [])
                         if team_a.casefold() in str(note.title).casefold()
                         or team_b.casefold() in str(note.title).casefold()]
                title = f"{team_a} vs {team_b}"
                match_identity = f"{sport}:{':'.join(sorted(wanted))}"
                details = _build_details(sport, team_a, team_b, row, notes)
                alerts.append(LiveAlert(
                    match_identity, sport, "IMPORTANT", 35, "PREMATCH_REFERENCE",
                    title, "", list(details.get("reasons", [])), now, f"{match_identity}:PREMATCH_REFERENCE",
                    details,
                ))
                emitted.add(key)
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

    @staticmethod
    def _recommendation_for_state(report: dict, state) -> dict | None:
        recommendations = [row for row in report.get("recommendations", []) if row.get("sport") == state.sport]
        wanted = {normalized_name(canonical_team(state.sport, state.team_a)),
                  normalized_name(canonical_team(state.sport, state.team_b))}
        return next((candidate for candidate in recommendations if all(
            name and name in normalized_name(str(candidate.get("event") or "")) for name in wanted
        )), None)

    def _pre_match_status(self, report: dict, state) -> str:
        row = self._recommendation_for_state(report, state)
        if not row:
            return "赛前未推送（静默跳过）"
        if (row.get("market_mapping_status") == "MATCHED"
                and str(row.get("lineup_status") or "") == "完整"):
            return "赛前已推送完整分析"
        if row.get("market_mapping_status") == "NOT_IN_SCHEDULE":
            return "赛前仅推送降级参考"
        return "赛前未推送（静默跳过）"


    def _prematch_side(self, state) -> str | None:
        report = self._report()
        if not report:
            return None
        row = self._recommendation_for_state(report, state)
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

    @staticmethod
    def _decisive_factors(state) -> list[str]:
        factors: list[str] = []
        features = state.features or {}
        if state.sport == "lol":
            metrics = (
                ("经济差", features.get("gold_a"), features.get("gold_b"), "gold"),
                ("击杀差", features.get("kills_a"), features.get("kills_b"), "kills"),
                ("防御塔差", features.get("towers_a"), features.get("towers_b"), "towers"),
                ("小龙差", features.get("dragons_a"), features.get("dragons_b"), "dragons"),
                ("大龙差", features.get("barons_a"), features.get("barons_b"), "barons"),
                ("高地塔差", features.get("inhibitors_a"), features.get("inhibitors_b"), "inhibitors"),
            )
            rows = []
            for label, a, b, _ in metrics:
                if a is None or b is None:
                    continue
                diff = float(a) - float(b)
                if abs(diff) < 1e-9:
                    continue
                side = "蓝方" if diff > 0 else "红方"
                rows.append((abs(diff), f"{label} {diff:+.0f}（{side}占优）"))
            factors = [text for _, text in sorted(rows, reverse=True)[:3]]
        elif state.sport == "nba":
            explicit = list(features.get("nba_decisive_factors") or [])
            if explicit:
                factors = explicit[:5]
            elif state.score_a is not None and state.score_b is not None:
                diff = float(state.score_a) - float(state.score_b)
                side = "A队" if diff > 0 else "B队"
                factors.append(f"最终分差 {diff:+.0f}（{side}占优）")
        return factors

    @staticmethod
    def _is_finished_state(state) -> bool:
        return state.finished or str(state.status).casefold() in {
            "finished", "completed", "post", "final"
        }

    @staticmethod
    def _nba_game_id_for_state(state, official_by_teams: dict) -> str | None:
        if state.source == "nba_official" and re.fullmatch(r"\d{10}", str(state.source_id or "")):
            return str(state.source_id)
        key = tuple(sorted((
            normalized_name(canonical_team("nba", state.team_a)),
            normalized_name(canonical_team("nba", state.team_b)),
        )))
        official = official_by_teams.get(key)
        if official and re.fullmatch(r"\d{10}", str(official.source_id or "")):
            return str(official.source_id)
        return None

    def _nba_game_details(self, states: list) -> dict[str, list[dict]]:
        """Fetch NBA.com box scores for every finished NBA state."""
        finished_states = [state for state in states
                           if state.sport == "nba" and self._is_finished_state(state)]
        if not finished_states:
            return {}
        official = self._attempt("nba_official_boxscore_lookup", lambda: NbaOfficialProvider().live())
        official_by_teams: dict = {}
        for official_state in official:
            key = tuple(sorted((
                normalized_name(canonical_team("nba", official_state.team_a)),
                normalized_name(canonical_team("nba", official_state.team_b)),
            )))
            official_by_teams.setdefault(key, official_state)
        details: dict[str, list[dict]] = {}
        provider = NbaBoxscoreProvider()
        hupu_provider = HupuNbaProvider()
        for state in finished_states:
            game_id = self._nba_game_id_for_state(state, official_by_teams)
            box = None
            if game_id:
                name = f"nba_boxscore_{game_id}"
                try:
                    box = provider.boxscore(game_id)
                    self.source_status[name] = {
                        "available": box is not None, "rows": 1 if box else 0,
                        "error": None, "checked_at": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception as error:
                    logging.exception("live_runtime nba boxscore %s failed", game_id)
                    self.source_status[name] = {
                        "available": False, "rows": 0, "error": repr(error),
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    }
            hupu_id = state.features.get("hupu_game_id")
            if not hupu_id and state.source == "hupu":
                hupu_id = state.source_id
            if box is None and hupu_id:
                name = f"hupu_nba_boxscore_{hupu_id}"
                try:
                    box = hupu_provider.boxscore(str(hupu_id))
                    self.source_status[name] = {
                        "available": box is not None, "rows": 1 if box else 0,
                        "error": None, "checked_at": datetime.now(timezone.utc).isoformat(),
                    }
                except Exception as error:
                    logging.exception("live_runtime hupu nba boxscore %s failed", hupu_id)
                    self.source_status[name] = {
                        "available": False, "rows": 0, "error": repr(error),
                        "checked_at": datetime.now(timezone.utc).isoformat(),
                    }
            if box:
                details.setdefault(match_key(state), []).append(box)
        return details

    def _compute_nba_box_samples(self, state, game_details: dict[str, list[dict]]) -> list[dict]:
        boxes = (game_details or {}).get(match_key(state), [])
        samples: list[dict] = []
        for index, box in enumerate(boxes, 1):
            metrics = nba_analytics.boxscore_metrics(box)
            samples.append({
                "game_index": index,
                "game_id": box.get("game_id") or state.source_id,
                "game_time": str(box.get("game_time_utc") or state.observed_at.isoformat()),
                "away_team": state.team_a,
                "home_team": state.team_b,
                "away_score": box.get("away_team", {}).get("score"),
                "home_score": box.get("home_team", {}).get("score"),
                "winner_side": self._winner_side(state),
                "period": box.get("period"),
                "game_status_text": box.get("game_status_text"),
                "duration": box.get("duration"),
                "duration_seconds": box.get("duration_seconds"),
                "arena": box.get("arena"),
                "officials": box.get("officials"),
                "away_team_boxscore": box.get("away_team"),
                "home_team_boxscore": box.get("home_team"),
                "metrics": metrics,
                "decisive_factors": nba_analytics.decisive_factors(box),
            })
        return samples

    @staticmethod
    def _nba_num(value) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.0f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _nba_player_lines(team_name: str, box_team: dict, limit: int = 8) -> list[str]:
        players = sorted((box_team.get("players") or []),
                         key=lambda row: float(row.get("points") or 0), reverse=True)[:limit]
        lines = []
        for player in players:
            name = str(player.get("name") or "未知")
            starter = "首发" if player.get("starter") else "替补"
            minutes = player.get("minutes") or ""
            points = LiveSupervisor._nba_num(player.get("points"))
            rebounds = LiveSupervisor._nba_num(player.get("rebounds_total"))
            assists = LiveSupervisor._nba_num(player.get("assists"))
            plus_minus = player.get("plus_minus_points")
            plus_minus_text = f"正负值 {float(plus_minus):+.0f}" if plus_minus is not None else ""
            parts = [f"{name}({starter})", minutes, f"{points}分 {rebounds}板 {assists}助", plus_minus_text]
            lines.append(" ".join(part for part in parts if part))
        return lines

    def _nba_strategy_readout(self, sample: dict) -> str:
        metrics = sample.get("metrics") or {}
        home_box = sample.get("home_team_boxscore") or {}
        away_box = sample.get("away_team_boxscore") or {}
        home_stats = home_box.get("statistics") or {}
        away_stats = away_box.get("statistics") or {}
        pace = float(metrics.get("pace") or 0)
        home_paint = float(home_stats.get("pointsInThePaint") or 0)
        away_paint = float(away_stats.get("pointsInThePaint") or 0)
        home_fast = float(home_stats.get("pointsFastBreak") or 0)
        away_fast = float(away_stats.get("pointsFastBreak") or 0)
        home_bench = float(home_stats.get("benchPoints") or 0)
        away_bench = float(away_stats.get("benchPoints") or 0)
        home_to = float(home_stats.get("turnoversTotal") or home_stats.get("turnovers") or 0)
        away_to = float(away_stats.get("turnoversTotal") or away_stats.get("turnovers") or 0)
        parts = []
        if pace >= 103:
            parts.append("节奏偏快，更多回合和转换进攻")
        elif pace <= 96:
            parts.append("节奏偏慢，半场阵地战权重更高")
        else:
            parts.append("节奏处于中游，攻守转换与半场阵地相对均衡")
        if home_paint > away_paint + 8:
            parts.append(f"{sample.get('home_team')} 内线终结优势明显")
        elif away_paint > home_paint + 8:
            parts.append(f"{sample.get('away_team')} 内线终结优势明显")
        if home_fast > away_fast + 8:
            parts.append(f"{sample.get('home_team')} 快攻转换更高效")
        elif away_fast > home_fast + 8:
            parts.append(f"{sample.get('away_team')} 快攻转换更高效")
        if home_bench > away_bench + 8:
            parts.append(f"{sample.get('home_team')} 替补深度占优")
        elif away_bench > home_bench + 8:
            parts.append(f"{sample.get('away_team')} 替补深度占优")
        if home_to > away_to + 4:
            parts.append(f"{sample.get('home_team')} 失误偏多，球权管理更差")
        elif away_to > home_to + 4:
            parts.append(f"{sample.get('away_team')} 失误偏多，球权管理更差")
        return "；".join(parts) + "。"

    def _build_nba_review_analysis(self, state, game_samples: list[dict],
                                   analyst_notes: dict[str, list]) -> str:
        """Return a structured, analyst-grade NBA post-game review."""
        lines = [f"{state.team_a} vs {state.team_b} NBA 赛后复盘。"]
        actual_side = self._winner_side(state)
        actual_team = state.team_a if actual_side == "a" else state.team_b if actual_side == "b" else None
        if actual_team:
            lines.append(f"实际胜者：{actual_team}。")
        if state.score_a is not None and state.score_b is not None:
            lines.append(f"最终比分：{state.team_a} {state.score_a:.0f} - {state.score_b:.0f} {state.team_b}。")
        prematch_side = self._prematch_side(state)
        if prematch_side is not None:
            prematch_team = state.team_a if prematch_side == "a" else state.team_b
            result = "正确" if prematch_side == actual_side else "错误"
            lines.append(f"赛前预测：{prematch_team}；判断{result}。")
        else:
            lines.append("赛前预测：未生成有效概率，跳过赛前维度。")

        sample = game_samples[0] if game_samples else {}
        if sample:
            home_team = sample.get("home_team")
            away_team = sample.get("away_team")
            home_box = sample.get("home_team_boxscore") or {}
            away_box = sample.get("away_team_boxscore") or {}
            metrics = sample.get("metrics") or {}
            home_factors = metrics.get("home_four_factors") or {}
            away_factors = metrics.get("away_four_factors") or {}
            home_ratings = metrics.get("home_ratings") or {}
            away_ratings = metrics.get("away_ratings") or {}

            lines.append("")
            lines.append("【四要素效率】")
            lines.append(
                f"{away_team}：有效命中率 {float(away_factors.get('effective_field_goal_pct') or 0):.1%}，"
                f"失误率 {float(away_factors.get('turnover_pct') or 0):.1%}，"
                f"进攻篮板率 {float(away_factors.get('offensive_rebound_pct') or 0):.1%}，"
                f"罚球率 {float(away_factors.get('free_throw_rate') or 0):.2f}。"
            )
            lines.append(
                f"{home_team}：有效命中率 {float(home_factors.get('effective_field_goal_pct') or 0):.1%}，"
                f"失误率 {float(home_factors.get('turnover_pct') or 0):.1%}，"
                f"进攻篮板率 {float(home_factors.get('offensive_rebound_pct') or 0):.1%}，"
                f"罚球率 {float(home_factors.get('free_throw_rate') or 0):.2f}。"
            )
            lines.append("")
            lines.append("【节奏与效率】")
            lines.append(f"回合数约 {float(metrics.get('possessions') or 0):.1f}，Pace {float(metrics.get('pace') or 0):.1f}。")
            lines.append(
                f"{away_team} 进攻效率 {float(away_ratings.get('offensive_rating') or 0):.1f}，"
                f"防守效率 {float(away_ratings.get('defensive_rating') or 0):.1f}，"
                f"净效率 {float(away_ratings.get('net_rating') or 0):+.1f}。"
            )
            lines.append(
                f"{home_team} 进攻效率 {float(home_ratings.get('offensive_rating') or 0):.1f}，"
                f"防守效率 {float(home_ratings.get('defensive_rating') or 0):.1f}，"
                f"净效率 {float(home_ratings.get('net_rating') or 0):+.1f}。"
            )
            lines.append("")
            lines.append("【比赛走势】")
            away_periods = away_box.get("periods") or []
            home_periods = home_box.get("periods") or []
            if away_periods or home_periods:
                for i in range(max(len(away_periods), len(home_periods))):
                    a_score = away_periods[i].get("score") if i < len(away_periods) else "—"
                    h_score = home_periods[i].get("score") if i < len(home_periods) else "—"
                    lines.append(f"第{i + 1}节：{away_team} {a_score} - {h_score} {home_team}。")
            home_stats = home_box.get("statistics") or {}
            away_stats = away_box.get("statistics") or {}
            if home_stats.get("leadChanges") is not None:
                lines.append(
                    f"领先交替 {home_stats.get('leadChanges')} 次，平局 {home_stats.get('timesTied')} 次，"
                    f"主队最大领先 {home_stats.get('biggestLead')}，客队最大领先 {away_stats.get('biggestLead')}。"
                )
            lines.append("")
            lines.append("【得分构成与球权细节】")
            for label, key in (
                ("内线得分", "pointsInThePaint"),
                ("快攻得分", "pointsFastBreak"),
                ("二次进攻得分", "pointsSecondChance"),
                ("利用失误得分", "pointsFromTurnovers"),
                ("替补得分", "benchPoints"),
                ("篮板总数", "reboundsTotal"),
                ("助攻", "assists"),
                ("抢断", "steals"),
                ("封盖", "blocks"),
                ("失误", "turnoversTotal"),
            ):
                away_value = away_stats.get(key)
                home_value = home_stats.get(key)
                if away_value is None and home_value is None:
                    continue
                lines.append(f"{label}：{away_team} {self._nba_num(away_value)} - {self._nba_num(home_value)} {home_team}。")
            lines.append("")
            lines.append("【球员表现】")
            away_lines = self._nba_player_lines(away_team, away_box)
            home_lines = self._nba_player_lines(home_team, home_box)
            if away_lines:
                lines.append(f"{away_team}：{'；'.join(away_lines)}。")
            if home_lines:
                lines.append(f"{home_team}：{'；'.join(home_lines)}。")
            factors = sample.get("decisive_factors") or []
            if factors:
                lines.append("")
                lines.append("【胜负手】")
                lines.extend(f"• {factor}" for factor in factors)
            lines.append("")
            lines.append("【战术执行与模型解读】")
            lines.append(self._nba_strategy_readout(sample))
        else:
            factors = self._decisive_factors(state)
            if factors:
                lines.append("全场关键数据：" + "；".join(factors) + "。")

        notes = [note for note in analyst_notes.get(state.sport, [])
                 if state.team_a.casefold() in str(note.title).casefold()
                 or state.team_b.casefold() in str(note.title).casefold()]
        if notes:
            lines.append(f"公开分析师参考：{len(notes)} 篇（仅作为特征留存，不改变本场结算概率）。")
        lines.append("该复盘文本作为赛后样本沉淀，用于后续模型迭代，不参与当前推送数值计算。")
        return "\n".join(lines)

    def _result_review_alerts(self, states: list, now: datetime,
                               analyst_notes: dict[str, list] | None = None,
                               game_details: dict[str, list[dict]] | None = None,
                               nba_game_details: dict[str, list[dict]] | None = None) -> list[LiveAlert]:
        analyst_notes = analyst_notes or {}
        report = self._report()
        alerts = []
        for state in states:
            if state.sport not in {"lol", "nba"}:
                continue
            if not self._is_finished_state(state):
                continue
            actual_side = self._winner_side(state)
            if actual_side is None:
                continue
            recommendation = self._recommendation_for_state(report, state)
            has_prematch_probability = bool(
                recommendation and recommendation.get("model_probability") is not None
            )
            has_bet = bool(recommendation and (
                recommendation.get("action") == "BET"
                or recommendation.get("virtual_bet")
                or float(recommendation.get("stake_virtual") or 0) > 0
            ))
            if not has_prematch_probability and not has_bet:
                continue
            prematch_side = self._prematch_side(state)
            bp_probability = state.features.get("post_draft_probability") if state.sport == "lol" else None
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

            if state.sport == "nba":
                game_samples = self._compute_nba_box_samples(state, nba_game_details or {})
                if game_samples:
                    state.features["nba_game_samples"] = game_samples
                    state.features["nba_decisive_factors"] = list(game_samples[0].get("decisive_factors") or [])
            else:
                game_samples = state.features.get("bp_game_samples", [])

            if state.sport == "nba":
                if prematch_side == actual_side:
                    reasons.append("复盘：比赛走势与赛前方向一致，执行力与节奏控制兑现为胜果。")
                else:
                    reasons.append("复盘：比赛走势偏离赛前方向，需重点核查临场阵容、轮换与战术匹配。")
            else:
                if (prematch_side == actual_side) or (bp_side == actual_side):
                    reasons.append("复盘：比赛走势与预测方向一致，优势方按预期滚动并转化为胜利。")
                else:
                    reasons.append("复盘：比赛走势偏离预测，需重点核查阵容克制、资源节奏或临场发挥。")

            notes = [note for note in analyst_notes.get(state.sport, [])
                     if state.team_a.casefold() in str(note.title).casefold()
                     or state.team_b.casefold() in str(note.title).casefold()]
            decisive_factors = self._decisive_factors(state)
            if state.sport == "lol":
                series_analysis = self._build_series_review_analysis(state, game_samples, analyst_notes)
            else:
                series_analysis = self._build_nba_review_analysis(state, game_samples, analyst_notes)
            key = match_key(state)
            alerts.append(LiveAlert(
                key, state.sport, "IMPORTANT", 70, "POSTMATCH_REVIEW",
                f"{state.team_a} vs {state.team_b}",
                f"比赛已结束，实际胜者：{actual_team}。",
                reasons, now, f"{key}:POSTMATCH_REVIEW",
                {
                    "event_id": state.source_id,
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
                    "pre_match_status": self._pre_match_status(report, state),
                    "analyst_count": len(notes),
                    "analyst_notes": [{"title": note.title, "link": note.link, "source": note.source}
                                      for note in notes[:3]],
                    "decisive_factors": decisive_factors,
                    "game_samples": game_samples,
                    "series_analysis": series_analysis,
                },
            ))
        for alert in alerts:
            if alert.category == "POSTMATCH_REVIEW":
                details = alert.details or {}
                record_post_match_review(
                    os.getenv("PAPER_DB_PATH", str(self.root / "data" / "daily" / "paper.db")),
                    {
                        "sport": alert.sport,
                        "event_id": details.get("event_id") or alert.match_key,
                        "event": alert.title,
                        "generated_at": now.isoformat(),
                        "actual_winner": details.get("actual_winner"),
                        "predicted_winner": details.get("prematch_team") or details.get("bp_team"),
                        "prediction_correct": (details.get("prematch_side") == details.get("actual_side"))
                                              if details.get("prematch_side") is not None
                                              else (details.get("bp_side") == details.get("actual_side")),
                        "model_probability": details.get("bp_probability"),
                        "bp_probability": details.get("bp_probability"),
                        "decisive_factors": details.get("decisive_factors", []),
                        "game_samples": details.get("game_samples", []),
                        "series_analysis": details.get("series_analysis", ""),
                    },
                )
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
                        "team_a": state.team_a,
                        "team_b": state.team_b,
                        "blue_champions": list(champions_a),
                        "red_champions": list(champions_b),
                        "blue_bans": list(state.features.get("bans_a") or []),
                        "red_bans": list(state.features.get("bans_b") or []),
                        "blue_players": list(state.features.get("players_a") or []),
                        "red_players": list(state.features.get("players_b") or []),
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
            logging.exception("live_runtime source %s failed", name)
            self.source_status[name] = {"available": False, "rows": 0, "error": repr(error),
                                        "checked_at": datetime.now(timezone.utc).isoformat()}
            return []

    def collect_states(self) -> list:
        today = datetime.now(timezone.utc).date()
        # Priority order is official/game-server data first, then fixture-score fallbacks.
        groups = [
            ("nba_official", lambda: NbaOfficialProvider().live()),
            ("hupu_nba", lambda: HupuNbaProvider().live()),
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
                if state.finished or state.status == "FINISHED":
                    if state.score_a is not None:
                        existing.score_a = state.score_a
                    if state.score_b is not None:
                        existing.score_b = state.score_b
                    existing.status = "FINISHED"
                    existing.finished = True
                elif state.status == "LIVE" and not existing.finished:
                    if state.score_a is not None:
                        existing.score_a = state.score_a
                    if state.score_b is not None:
                        existing.score_b = state.score_b
                    existing.status = "LIVE"
                    existing.finished = False
                else:
                    if existing.score_a is None:
                        existing.score_a = state.score_a
                    if existing.score_b is None:
                        existing.score_b = state.score_b
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

    def _lol_game_details(self) -> dict[str, list[dict]]:
        """Return per-game draft/result rows for each LoL series.

        Leaguepedia publishes one row per small game (picks + winner), which is
        the data needed to evaluate every individual BP instead of only the
        latest sampled game.
        """
        details: dict[str, list[dict]] = {}
        for state in self._attempt("leaguepedia_bp_all", lambda: LeaguepediaDraftProvider().live_all()):
            key = match_key(state)
            details.setdefault(key, []).append({
                "game_id": state.source_id,
                "game_time": str(state.features.get("game_time") or state.observed_at.isoformat()),
                "champions_a": list(state.features.get("champions_a", [])),
                "champions_b": list(state.features.get("champions_b", [])),
                "winner_side": state.features.get("winner_side"),
                "finished": bool(state.finished),
                "dragons_a": state.features.get("dragons_a"),
                "dragons_b": state.features.get("dragons_b"),
                "barons_a": state.features.get("barons_a"),
                "barons_b": state.features.get("barons_b"),
                "towers_a": state.features.get("towers_a"),
                "towers_b": state.features.get("towers_b"),
                "rift_heralds_a": state.features.get("rift_heralds_a"),
                "rift_heralds_b": state.features.get("rift_heralds_b"),
                "inhibitors_a": state.features.get("inhibitors_a"),
                "inhibitors_b": state.features.get("inhibitors_b"),
                "game_length_seconds": state.features.get("game_length_seconds"),
                "patch": state.features.get("patch"),
                "game_number": state.features.get("game_number"),
                "players_a": list(state.features.get("players_a", [])),
                "players_b": list(state.features.get("players_b", [])),
                "bans_a": list(state.features.get("bans_a", [])),
                "bans_b": list(state.features.get("bans_b", [])),
            })
        player_stats = self._attempt("leaguepedia_bp_players", lambda: LeaguepediaDraftProvider().live_player_stats())
        team_stats = self._attempt("leaguepedia_bp_teams", lambda: LeaguepediaDraftProvider().live_team_stats())
        for games in details.values():
            games.sort(key=lambda row: row.get("game_time") or "")
            for index, row in enumerate(games, 1):
                row["game_index"] = index
                row["players"] = player_stats.get(str(row.get("game_id") or ""), [])
                row["teams"] = team_stats.get(str(row.get("game_id") or ""), [])
        return details

    def _compute_lol_bp_samples(self, model, patch: str, state, game_details: dict[str, list[dict]]) -> list[dict]:
        team_a, team_b = canonical_team("lol", state.team_a), canonical_team("lol", state.team_b)
        players_a = tuple(model.latest_team_rosters.get(team_a, ()))
        players_b = tuple(model.latest_team_rosters.get(team_b, ()))
        if len(players_a) != 5 or len(players_b) != 5:
            return []
        samples = []
        games = sorted((game_details or {}).get(match_key(state), []),
                       key=lambda row: row.get("game_time") or "")
        for offset, game in enumerate(games, 1):
            champions_a = tuple(value for value in game.get("champions_a", []) if value)
            champions_b = tuple(value for value in game.get("champions_b", []) if value)
            if len(champions_a) != 5 or len(champions_b) != 5:
                continue
            game_obj = LolDraftGame(
                f"live-{game.get('game_id') or state.source_id}", state.observed_at, patch, "live",
                team_a, team_b, players_a, players_b, champions_a, champions_b, 0,
            )
            blue_win = model.predict_post_draft(game_obj)
            samples.append({
                "game_index": int(game.get("game_index") or offset),
                "game_id": str(game.get("game_id") or ""),
                "game_time": str(game.get("game_time") or ""),
                "patch": str(game.get("patch") or patch),
                "game_number": game.get("game_number"),
                "game_length_seconds": game.get("game_length_seconds"),
                "blue_team": state.team_a,
                "red_team": state.team_b,
                "blue_champions": list(champions_a),
                "red_champions": list(champions_b),
                "blue_players": list(game.get("players_a") or players_a),
                "red_players": list(game.get("players_b") or players_b),
                "blue_bans": list(game.get("bans_a") or []),
                "red_bans": list(game.get("bans_b") or []),
                "dragons_a": game.get("dragons_a"),
                "dragons_b": game.get("dragons_b"),
                "barons_a": game.get("barons_a"),
                "barons_b": game.get("barons_b"),
                "towers_a": game.get("towers_a"),
                "towers_b": game.get("towers_b"),
                "rift_heralds_a": game.get("rift_heralds_a"),
                "rift_heralds_b": game.get("rift_heralds_b"),
                "inhibitors_a": game.get("inhibitors_a"),
                "inhibitors_b": game.get("inhibitors_b"),
                "blue_post_draft_win": round(blue_win, 4),
                "red_post_draft_win": round(1 - blue_win, 4),
                "winner_side": game.get("winner_side"),
                "players": list(game.get("players") or []),
                "teams": list(game.get("teams") or []),
            })
        return samples

    def _add_lol_draft_probabilities(self, states: list,
                                     game_details: dict[str, list[dict]] | None = None) -> None:
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
            if len(players_a) == 5 and len(players_b) == 5 and len(champions_a) == 5 and len(champions_b) == 5:
                game = LolDraftGame(
                    f"live-{state.source_id}", state.observed_at, patch, "live", team_a, team_b,
                    players_a, players_b, champions_a, champions_b, 0,
                )
                state.features["post_draft_probability"] = model.predict_post_draft(game)
                state.features["draft_readout"] = model.draft_readout(game)
            state.features["bp_game_samples"] = self._compute_lol_bp_samples(model, patch, state, game_details)

    def _build_series_review_analysis(self, state, game_samples: list[dict], analyst_notes: dict[str, list]) -> str:
        """Return a structured, multi-paragraph post-match review.

        The full text is persisted as a training/evaluation sample; the push
        message keeps only the short version in ``format_live_alert``.
        """
        lines: list[str] = []
        if state.sport == "lol":
            games = game_samples or []
            lines.append(f"{state.team_a} vs {state.team_b} BO{max(1, len(games))} 系列复盘。")
            actual_side = self._winner_side(state)
            prematch_side = self._prematch_side(state)
            if prematch_side is not None:
                prematch_team = state.team_a if prematch_side == "a" else state.team_b
                result = "正确" if prematch_side == actual_side else "错误"
                lines.append(f"赛前方向：{prematch_team}；系列结果判断{result}。")
            else:
                lines.append("赛前方向：未生成有效概率，跳过赛前维度。")
            if games:
                for game in games:
                    index = int(game.get("game_index") or 0)
                    blue = game.get("blue_post_draft_win")
                    red = game.get("red_post_draft_win")
                    winner_side = game.get("winner_side")
                    winner_team = (state.team_a if winner_side == "a"
                                   else state.team_b if winner_side == "b" else None)
                    lines.append(
                        f"第{index}局 BP：蓝方 {state.team_a} "
                        f"({'、'.join(game.get('blue_champions') or [])})；红方 {state.team_b} "
                        f"({'、'.join(game.get('red_champions') or [])})。"
                    )
                    blue_bans = game.get("blue_bans") or []
                    red_bans = game.get("red_bans") or []
                    if blue_bans or red_bans:
                        lines.append(
                            f"第{index}局 Ban：蓝方 {'、'.join(blue_bans)}；红方 {'、'.join(red_bans)}。"
                        )
                    if blue is not None and red is not None:
                        lines.append(
                            f"第{index}局 BP 后模型胜率：蓝方 {float(blue):.1%}，红方 {float(red):.1%}。"
                        )
                    objective_parts = []
                    for label, key_a, key_b in (
                        ("龙", "dragons_a", "dragons_b"),
                        ("大龙", "barons_a", "barons_b"),
                        ("塔", "towers_a", "towers_b"),
                        ("先锋", "rift_heralds_a", "rift_heralds_b"),
                        ("高地", "inhibitors_a", "inhibitors_b"),
                    ):
                        a, b = game.get(key_a), game.get(key_b)
                        if a is not None and b is not None:
                            objective_parts.append(f"{label} {a}-{b}")
                    if objective_parts:
                        lines.append(f"第{index}局资源控制：{'；'.join(objective_parts)}。")
                    players = game.get("players") or []
                    blue_player_lines = []
                    red_player_lines = []
                    for player in players:
                        team = str(player.get("team") or "")
                        try:
                            team_norm = normalized_name(canonical_team("lol", team))
                            blue_norm = normalized_name(canonical_team("lol", state.team_a))
                            red_norm = normalized_name(canonical_team("lol", state.team_b))
                        except Exception:
                            continue
                        k = player.get("kills")
                        d = player.get("deaths")
                        a = player.get("assists")
                        gold = player.get("gold")
                        cs = player.get("cs")
                        line = (f"{player.get('player') or '未知'} {player.get('champion') or ''} "
                                f"KDA {k}/{d}/{a} 金币 {gold} CS {cs}")
                        if team_norm == blue_norm:
                            blue_player_lines.append(line)
                        elif team_norm == red_norm:
                            red_player_lines.append(line)
                    if blue_player_lines:
                        lines.append(f"第{index}局蓝方选手表现：{'；'.join(blue_player_lines)}。")
                    if red_player_lines:
                        lines.append(f"第{index}局红方选手表现：{'；'.join(red_player_lines)}。")
                    if winner_team:
                        predicted_side = "a" if float(blue or 0) >= float(red or 0) else "b"
                        correct = winner_side == predicted_side
                        lines.append(
                            f"第{index}局实际胜者：{winner_team}；BP 后模型判断{'正确' if correct else '错误'}。"
                        )
                usable_games = [game for game in games if game.get("winner_side") in {"a", "b"}]
                bp_errors = sum(
                    1 for game in usable_games
                    if (("a" if float(game.get("blue_post_draft_win") or 0) >=
                         float(game.get("red_post_draft_win") or 0) else "b") != game.get("winner_side"))
                )
                lines.append(
                    f"学习标签：系列共 {len(usable_games)} 局可结算，BP 后方向错误 {bp_errors} 局；"
                    "该标签只进入后续按时间顺序的离线评估，不在本次扫描中更新模型。"
                )
            factors = self._decisive_factors(state)
            if factors:
                lines.append("全场关键数据：" + "；".join(factors) + "。")
            lines.append("该复盘文本作为赛后样本沉淀，用于后续模型迭代，不参与当前推送数值计算。")
        else:
            lines.append(f"{state.team_a} vs {state.team_b} 赛后复盘。")
            actual_side = self._winner_side(state)
            prematch_side = self._prematch_side(state)
            actual_team = state.team_a if actual_side == "a" else state.team_b
            lines.append(f"实际胜者：{actual_team}。")
            if prematch_side is not None:
                prematch_team = state.team_a if prematch_side == "a" else state.team_b
                result = "正确" if prematch_side == actual_side else "错误"
                lines.append(f"赛前预测：{prematch_team}；判断{result}。")
            factors = self._decisive_factors(state)
            if factors:
                lines.append("全场关键数据：" + "；".join(factors) + "。")
            lines.append("该复盘文本作为赛后样本沉淀，用于后续模型迭代。")
        return "\n".join(lines)

    def scan_once(self) -> dict:
        now = datetime.now(timezone.utc)
        report = self._report()
        states = self.collect_states()
        lol_game_details = self._lol_game_details()
        self._add_lol_draft_probabilities(states, lol_game_details)
        nba_game_details = self._nba_game_details(states)
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
                     self._result_review_alerts(states, now, analyst_notes, lol_game_details, nba_game_details) +
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
