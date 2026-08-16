from __future__ import annotations

import json
import logging
import math
import os
import unicodedata
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .cs2_model import Cs2Model, load_cs2
from .lol_meta_model import LolDraftGame, LolMetaModel, load_lol_meta
from .lol_model import EloModel, load_model, series_probability
from .nba_model import NbaModel, load_nba
from .providers.polymarket import PolymarketClient
from .providers.live_data import (
    Bo3Cs2Provider, EsportAgendaCs2Provider, EspnNbaProvider, GridOpenAccessProvider, NbaOfficialProvider,
    PandaScoreProvider, SportSrcNbaProvider, TheSportsDbNbaProvider,
)
from .risk import RiskBudgetLedger, RiskConfig, paper_recommend, recommend
from .entities import canonical_team, normalized_name
from .costs import estimate_cost_rate
from .narrative import build_pre_match_summary
from .schedule import (
    LolScheduleDiscovery, SourceResult, append_schedule_audit, build_schedule_audit, make_match,
)


TAGS = {"nba": "745", "lol": "65", "cs2": "100780"}


def _paper_trading_enabled() -> bool:
    """Paper betting can run before real-money ROI acceptance.

    This only controls the append-only paper ledger.  Real-money approval is a
    separate, stricter ``approved_for_real_money`` flag and is never implied.
    """
    return os.getenv("PAPER_TRADING_ENABLED", "true").casefold() == "true"


def _in_horizon(scheduled: datetime | None, now: datetime) -> bool:
    hours = float(os.getenv("MARKET_HORIZON_HOURS", "30"))
    return scheduled is not None and now < scheduled <= now + timedelta(hours=hours)


def _timing(scheduled: datetime, now: datetime) -> tuple[float, str]:
    hours = (scheduled - now).total_seconds() / 3600
    target = min((1, 6, 24), key=lambda value: abs(value - hours))
    return hours, f"T-{target}h"


LOL_MAJOR_EVENT_KEYWORDS = (
    "lpl", "lck", "lec", "lcs", "lta", "lcp", "msi", "worlds",
    "world championship", "first stand", "ewc", "esports world cup",
)


def _is_major_lol_event(title: object) -> bool:
    lowered = str(title or "").casefold()
    return any(keyword in lowered for keyword in LOL_MAJOR_EVENT_KEYWORDS)


CS2_MAJOR_EVENT_KEYWORDS = (
    "major", "iem", "intel extreme masters", "esl pro league",
    "blast premier", "blast open", "blast showdown", "blast.tv",
    "pgl", "esports world cup", "ewc",
)

CS2_MINOR_EVENT_KEYWORDS = (
    "cct", "esea", "esl challenger", "rising", "regional", "national",
    "academy", "open qualifier", "closed qualifier", "qualifier",
)


def _is_major_cs2_event(title: object) -> bool:
    lowered = str(title or "").casefold()
    if any(keyword in lowered for keyword in CS2_MINOR_EVENT_KEYWORDS):
        return False
    return any(keyword in lowered for keyword in CS2_MAJOR_EVENT_KEYWORDS)


def _field(value):
    return json.loads(value) if isinstance(value, str) else list(value or [])


def _text(value: object) -> str:
    return "".join(character for character in str(value)
                   if unicodedata.category(character) not in {"Cc", "Cf"}).strip()


def _probability_sanity(model_probability: float, market_probability: float | None) -> list[str]:
    """Surface implausible probabilities instead of silently publishing them."""
    problems: list[str] = []
    if not math.isfinite(model_probability) or not 0 <= model_probability <= 1:
        problems.append("模型概率超出 [0, 1] 有效范围，已标记为可疑。")
    elif model_probability > 0.98 or model_probability < 0.02:
        problems.append("模型概率处于极端区间（>98% 或 <2%），建议人工复核后再使用。")
    if market_probability is not None and math.isfinite(market_probability):
        divergence = abs(model_probability - market_probability)
        if divergence > 0.35:
            problems.append(f"模型与市场价格分歧过大（{divergence:.1%}），可能为赛程-市场映射异常或数据错误。")
    return problems


def _ev_tier(value: float | None) -> str:
    if value is None:
        return "未知"
    if value > 0.15:
        return "高"
    if value > 0.05:
        return "中"
    return "低"


def _direction_match(prices: list[float], side: int) -> bool:
    """Return True when the model side is the market favorite (lower payout side)."""
    if not prices or side is None:
        return False
    return side == max(range(len(prices)), key=lambda index: prices[index])


def _lineup_status(roster_a: tuple[str, ...], roster_b: tuple[str, ...],
                   last_games: dict[str, str], teams: tuple[str, str],
                   now: datetime, max_age_days: int) -> str:
    """Classify lineup availability for paper-betting attribution."""
    if len(roster_a) != 5 or len(roster_b) != 5:
        return "未知"
    return "完整" if _roster_fresh(last_games, teams, now, max_age_days) else "过期"


def _paper_daily_report(rows: list[dict], bankroll: float,
                        committed_fraction: float, max_daily_risk_fraction: float) -> dict:
    bets = [row for row in rows if row.get("action") == "BET"]
    total_stake = sum(float(row.get("stake") or 0) for row in bets)
    return {
        "bet_count": len(bets),
        "skipped_count": len(rows) - len(bets),
        "total_stake": total_stake,
        "committed_fraction": committed_fraction,
        "remaining_limit": max(0.0, bankroll * (max_daily_risk_fraction - committed_fraction)),
    }

def _find_schedule_match(schedule_matches: list[dict] | None, sport: str,
                         team_a: str, team_b: str, scheduled: datetime | None) -> dict | None:
    if not schedule_matches or scheduled is None:
        return None
    wanted = {normalized_name(team_a), normalized_name(team_b)}
    for row in schedule_matches:
        if row.get("sport") != sport:
            continue
        actual = {normalized_name(row.get("team_a")), normalized_name(row.get("team_b"))}
        if actual != wanted:
            continue
        try:
            start = datetime.fromisoformat(str(row.get("start_time")))
        except (TypeError, ValueError):
            continue
        if abs((scheduled - start).total_seconds()) <= 90 * 60:
            return row
    return None


def _start(market: dict) -> datetime | None:
    value = market.get("gameStartTime")
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def _main_market(event: dict) -> dict | None:
    markets = event.get("markets", [])
    candidates = [m for m in markets if m.get("gameStartTime")]
    return next((m for m in candidates if m.get("sportsMarketType") == "moneyline"), None) or next(
        (m for m in candidates if m.get("question") == event.get("title")), None)


def analyze_sport(sport: str, model: EloModel | NbaModel, evaluation: dict, events: list[dict], *,
                  now: datetime, bankroll: float, estimated_cost: float | None = None,
                  schedule_matches: list[dict] | None = None,
                  risk_config: RiskConfig | None = None,
                  ledger: RiskBudgetLedger | None = None,
                  group_key: str | None = None) -> list[dict]:
    risk_config = risk_config or RiskConfig()
    if ledger is None:
        ledger = RiskBudgetLedger(risk_config, bankroll)
    paper_enabled = _paper_trading_enabled()
    rows = []
    for event in events:
        market = _main_market(event)
        if not market:
            continue
        scheduled = _start(market)
        if not _in_horizon(scheduled, now):
            continue
        outcomes = [_text(x) for x in _field(market.get("outcomes"))]
        prices = [float(x) for x in _field(market.get("outcomePrices"))]
        if len(outcomes) != 2 or len(prices) != 2 or min(prices) <= 0:
            continue
        market_team_a, market_team_b = outcomes
        team_a, team_b = canonical_team(sport, market_team_a), canonical_team(sport, market_team_b)
        game_p = (model.game_probability(team_a, team_b, scheduled)
                  if sport == "nba" and isinstance(model, NbaModel)
                  else model.game_probability(team_a, team_b))
        best_of = 1
        if sport == "lol":
            title = str(event.get("title") or "")
            best_of = 5 if "BO5" in title else 3 if "BO3" in title else 1
        p_a = series_probability(game_p, best_of)
        model_ps = [p_a, 1 - p_a]
        side = max(range(2), key=lambda index: model_ps[index] - prices[index])
        direction_match = _direction_match(prices, side)
        first_bid, first_ask = market.get("bestBid"), market.get("bestAsk")
        ask = float(first_ask) if side == 0 and first_ask is not None else (
            1 - float(first_bid) if side == 1 and first_bid is not None else prices[side])
        known = model.games.get(team_a, 0) >= 10 and model.games.get(team_b, 0) >= 10
        probability_ok = bool(evaluation.get("approved_for_probability_use")) and known
        money_ok = bool(evaluation.get("approved_for_real_money"))
        started = scheduled is None or scheduled <= now
        assert scheduled is not None
        event_key = str(event.get("id"))
        group = group_key or f"{sport}:{scheduled.date().isoformat()}"
        cost = estimate_cost_rate(ask) if estimated_cost is None else estimated_cost
        cap = ledger.cap_for(event_key, group)
        risk_reasons = ledger.exhausted_reasons(event_key, group)
        if paper_enabled and not started:
            rec = paper_recommend(
                event_id=event_key, outcome=outcomes[side], model_probability=model_ps[side],
                decimal_odds=1 / ask, bankroll=bankroll, estimated_cost=cost,
                max_bet_fraction=cap, direction_match=direction_match, risk_reasons=risk_reasons,
            )
        else:
            rec = recommend(
                event_id=event_key, outcome=outcomes[side], model_probability=model_ps[side],
                decimal_odds=1 / ask, bankroll=bankroll,
                confidence=.75 if probability_ok else .25,
                spread=float(market["spread"]) if market.get("spread") is not None else None,
                available_size=float(market.get("liquidity") or 0), estimated_cost=cost,
                trading_enabled=probability_ok and not started,
                kelly_scale=risk_config.kelly_scale, max_bet_fraction=cap,
                min_edge=risk_config.min_edge, min_confidence=risk_config.min_confidence,
                max_spread=risk_config.max_spread, min_available_size=risk_config.min_available_size,
                max_depth_fraction=risk_config.max_depth_fraction, risk_reasons=risk_reasons,
            )
        if rec.action == "BET":
            ledger.commit(event_key, group, rec.stake_fraction)
        reasons = (model.explain(team_a, team_b, scheduled)
                   if sport == "nba" and isinstance(model, NbaModel)
                   else model.explain(team_a, team_b))
        reasons.append(f"{market_team_a} 独立胜率 {p_a:.1%}，{market_team_b} 独立胜率 {1-p_a:.1%}")
        reasons.append("市场价格只用于估值和下注判断，不进入独立胜率模型")
        if not probability_ok:
            reasons.append("概率模型未通过锁箱验收或队伍历史样本不足")
        if not money_ok:
            reasons.append("未通过历史可成交赔率 ROI 验收，禁止真钱建议")
        if started:
            reasons.append("比赛已开始或开赛时间无法核验，禁止赛前下注")
        market_probability = prices[side]
        model_probability = model_ps[side]
        sanity = _probability_sanity(model_probability, market_probability)
        if sanity:
            reasons.extend(sanity)
        schedule_match = _find_schedule_match(schedule_matches, sport, outcomes[0], outcomes[1], scheduled)
        row = asdict(rec)
        row["generated_at"] = rec.generated_at.isoformat()
        hours_to_start, decision_window = _timing(scheduled, now)
        row.update({
            "sport": sport, "event": _text(event.get("title") or ""),
            "scheduled_start": scheduled.isoformat() if scheduled else None,
            "market_probability": prices[side], "execution_price": ask,
            "edge": rec.decision_probability - ask - cost,
            "probability_eligible": probability_ok,
            "real_money_approved": money_ok,
            "market_started": started,
            "market_comparison_valid": not started,
            "hours_to_start": hours_to_start,
            "decision_window": decision_window,
            "probability_plausible": not bool(sanity),
            "schedule_matched": bool(schedule_match and schedule_match.get("market_mapping_status") == "MATCHED"),
            "market_mapping_status": schedule_match.get("market_mapping_status") if schedule_match else "NOT_IN_SCHEDULE",
            "lineup_status": "未知",
            "ev_tier": _ev_tier(rec.expected_value),
            "direction_match": bool(direction_match),
            "reasons": reasons + list(rec.reasons),
        })
        if started:
            row.update({"decision_probability": None, "raw_edge": None, "edge": None,
                        "expected_value": None, "execution_price": None})
        row["narrative_summary"] = build_pre_match_summary(row)
        rows.append(row)
    return rows



def _roster_fresh(last_games: dict[str, str], teams: tuple[str, str], now: datetime,
                  max_age_days: int) -> bool:
    dates = []
    for team in teams:
        value = last_games.get(team)
        if not value:
            return False
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        dates.append(parsed.replace(tzinfo=parsed.tzinfo or timezone.utc))
    return all((now.date() - value.date()).days <= max_age_days for value in dates)


def _market_rows(events: list[dict], now: datetime) -> list[tuple[dict, dict, datetime | None, list[str], list[float]]]:
    result = []
    for event in events:
        market = _main_market(event)
        if not market:
            continue
        scheduled = _start(market)
        if not _in_horizon(scheduled, now):
            continue
        outcomes = [_text(x) for x in _field(market.get("outcomes"))]
        prices = [float(x) for x in _field(market.get("outcomePrices"))]
        if len(outcomes) == 2 and len(prices) == 2 and min(prices) > 0:
            result.append((event, market, scheduled, outcomes, prices))
    return result


def _research_row(sport: str, event: dict, market: dict, scheduled: datetime | None,
                  outcomes: list[str], prices: list[float], probabilities: list[float], *,
                  probability_ok: bool, money_ok: bool, now: datetime, bankroll: float,
                  reasons: list[str], estimated_cost: float | None = None,
                  schedule_matches: list[dict] | None = None,
                  risk_config: RiskConfig | None = None,
                  ledger: RiskBudgetLedger | None = None,
                  group_key: str | None = None,
                  lineup_status: str = "未知") -> dict:
    risk_config = risk_config or RiskConfig()
    if ledger is None:
        ledger = RiskBudgetLedger(risk_config, bankroll)
    paper_enabled = _paper_trading_enabled()
    side = max(range(2), key=lambda index: probabilities[index] - prices[index])
    direction_match = _direction_match(prices, side)
    bid, best_ask = market.get("bestBid"), market.get("bestAsk")
    ask = float(best_ask) if side == 0 and best_ask is not None else (
        1 - float(bid) if side == 1 and bid is not None else prices[side])
    started = scheduled is None or scheduled <= now
    assert scheduled is not None
    event_key = str(event.get("id"))
    group = group_key or f"{sport}:{scheduled.date().isoformat()}"
    cost = estimate_cost_rate(ask) if estimated_cost is None else estimated_cost
    cap = ledger.cap_for(event_key, group)
    risk_reasons = ledger.exhausted_reasons(event_key, group)
    if paper_enabled and not started:
        rec = paper_recommend(
            event_id=event_key, outcome=outcomes[side], model_probability=probabilities[side],
            decimal_odds=1 / ask, bankroll=bankroll, estimated_cost=cost,
            max_bet_fraction=cap, direction_match=direction_match, risk_reasons=risk_reasons,
        )
    else:
        rec = recommend(
            event_id=event_key, outcome=outcomes[side], model_probability=probabilities[side],
            decimal_odds=1 / ask, bankroll=bankroll,
            confidence=.75 if probability_ok else .25,
            spread=float(market["spread"]) if market.get("spread") is not None else None,
            available_size=float(market.get("liquidity") or 0), estimated_cost=cost,
            trading_enabled=probability_ok and not started,
            kelly_scale=risk_config.kelly_scale, max_bet_fraction=cap,
            min_edge=risk_config.min_edge, min_confidence=risk_config.min_confidence,
            max_spread=risk_config.max_spread, min_available_size=risk_config.min_available_size,
            max_depth_fraction=risk_config.max_depth_fraction, risk_reasons=risk_reasons,
        )
    if rec.action == "BET":
        ledger.commit(event_key, group, rec.stake_fraction)
    if not probability_ok:
        reasons.append("阵容未知、阵容过期、样本不足或概率模型未通过验收，因此只展示研究值。")
    if not money_ok:
        reasons.append("尚未通过带历史可成交赔率的样本外 ROI 验收，禁止真钱下注建议。")
    if started:
        reasons.append("比赛已开始或开赛时间无法核验，禁止赛前下注。")
    market_probability = prices[side]
    model_probability = probabilities[side]
    sanity = _probability_sanity(model_probability, market_probability)
    if sanity:
        reasons.extend(sanity)
    schedule_match = _find_schedule_match(schedule_matches, sport, outcomes[0], outcomes[1], scheduled)
    row = asdict(rec)
    row["generated_at"] = rec.generated_at.isoformat()
    hours_to_start, decision_window = _timing(scheduled, now)
    row.update({
        "sport": sport, "event": _text(event.get("title") or ""),
        "scheduled_start": scheduled.isoformat() if scheduled else None,
        "market_probability": prices[side], "execution_price": ask,
        "edge": rec.decision_probability - ask - cost,
        "probability_eligible": probability_ok,
        "real_money_approved": money_ok,
        "market_started": started,
        "market_comparison_valid": not started,
        "hours_to_start": hours_to_start,
        "decision_window": decision_window,
        "probability_plausible": not bool(sanity),
        "schedule_matched": bool(schedule_match and schedule_match.get("market_mapping_status") == "MATCHED"),
        "market_mapping_status": schedule_match.get("market_mapping_status") if schedule_match else "NOT_IN_SCHEDULE",
        "lineup_status": lineup_status,
        "ev_tier": _ev_tier(rec.expected_value),
        "direction_match": bool(direction_match),
        "reasons": reasons + list(rec.reasons),
    })
    if started:
        row.update({"decision_probability": None, "raw_edge": None, "edge": None,
                    "expected_value": None, "execution_price": None})
    row["narrative_summary"] = build_pre_match_summary(row)
    return row



def analyze_cs2(model: Cs2Model, evaluation: dict, events: list[dict], *,
                now: datetime, bankroll: float,
                schedule_matches: list[dict] | None = None,
                risk_config: RiskConfig | None = None,
                ledger: RiskBudgetLedger | None = None,
                group_key: str | None = None) -> list[dict]:
    rows = []
    for event, market, scheduled, outcomes, prices in _market_rows(events, now):
        schedule_match = _find_schedule_match(schedule_matches, "cs2", outcomes[0], outcomes[1], scheduled)
        if schedule_match is None and not _is_major_cs2_event(event.get("title")):
            continue
        a, b = canonical_team("cs2", outcomes[0]), canonical_team("cs2", outcomes[1])
        roster_a = tuple(model.latest_team_rosters.get(a, ()))
        roster_b = tuple(model.latest_team_rosters.get(b, ()))
        probability = model.probability(a, b, roster_a, roster_b)
        known = model.team_games.get(a, 0) >= 10 and model.team_games.get(b, 0) >= 10
        roster_ok = len(roster_a) == len(roster_b) == 5 and _roster_fresh(model.team_last_game, (a, b), now, 60)
        lineup_status = _lineup_status(roster_a, roster_b, model.team_last_game, (a, b), now, 60)
        probability_ok = bool(evaluation.get("approved_for_probability_use")) and known and roster_ok
        reasons = [
            f"阵容感知胜率：{outcomes[0]} {probability:.1%}，{outcomes[1]} {1-probability:.1%}。",
            f"历史样本：{a} {model.team_games.get(a, 0)} 场，{b} {model.team_games.get(b, 0)} 场。",
            "当前基线包含战队和五人阵容强度；地图池、veto 与 LAN/线上层仍在补充。",
        ]
        rows.append(_research_row("cs2", event, market, scheduled, outcomes, prices,
                                  [probability, 1-probability], probability_ok=probability_ok,
                                  money_ok=bool(evaluation.get("approved_for_real_money")),
                                  now=now, bankroll=bankroll, reasons=reasons,
                                  schedule_matches=schedule_matches, risk_config=risk_config, ledger=ledger, group_key=group_key,
                                  lineup_status=lineup_status))
    return rows


def analyze_lol_meta(model: LolMetaModel, evaluation: dict, events: list[dict], *,
                     now: datetime, bankroll: float,
                     schedule_matches: list[dict] | None = None,
                     risk_config: RiskConfig | None = None,
                     ledger: RiskBudgetLedger | None = None,
                     group_key: str | None = None) -> list[dict]:
    rows = []
    for event, market, scheduled, outcomes, prices in _market_rows(events, now):
        if not _is_major_lol_event(event.get("title")):
            continue
        a, b = canonical_team("lol", outcomes[0]), canonical_team("lol", outcomes[1])
        roster_a = tuple(model.latest_team_rosters.get(a, ()))
        roster_b = tuple(model.latest_team_rosters.get(b, ()))
        blank = ("", "", "", "", "")
        game_ab = LolDraftGame("forecast-ab", now, "unknown", "unknown", a, b,
                               roster_a, roster_b, blank, blank, 0)
        game_ba = LolDraftGame("forecast-ba", now, "unknown", "unknown", b, a,
                               roster_b, roster_a, blank, blank, 0)
        neutral_game_p = (model.predict_pre_draft(game_ab) + 1 - model.predict_pre_draft(game_ba)) / 2
        title = str(event.get("title") or "")
        best_of = 5 if "BO5" in title else 3 if "BO3" in title else 1
        probability = series_probability(neutral_game_p, best_of)
        known = model.team_games.get(a, 0) >= 10 and model.team_games.get(b, 0) >= 10
        roster_ok = len(roster_a) == len(roster_b) == 5 and _roster_fresh(model.team_last_game, (a, b), now, 90)
        lineup_status = _lineup_status(roster_a, roster_b, model.team_last_game, (a, b), now, 90)
        probability_ok = bool(evaluation.get("approved_for_probability_use")) and known and roster_ok
        reasons = [
            f"赛前阵容模型：{outcomes[0]} {probability:.1%}，{outcomes[1]} {1-probability:.1%}（BO{best_of}）。",
            f"历史样本：{a} {model.team_games.get(a, 0)} 局，{b} {model.team_games.get(b, 0)} 局。",
            "BP 未开始时不使用英雄选择；BP 完成后必须重新计算版本英雄强度与选手英雄熟练度。",
        ]
        rows.append(_research_row("lol", event, market, scheduled, outcomes, prices,
                                  [probability, 1-probability], probability_ok=probability_ok,
                                  money_ok=bool(evaluation.get("approved_for_real_money")),
                                  now=now, bankroll=bankroll, reasons=reasons,
                                  schedule_matches=schedule_matches, risk_config=risk_config, ledger=ledger, group_key=group_key,
                                  lineup_status=lineup_status))
    return rows


FLAG_DISAGREEMENT_DEFAULT = 0.10


def flag_rows(rows: list[dict], flag_threshold: float = FLAG_DISAGREEMENT_DEFAULT) -> list[dict]:
    """Tag rows needing human review and return them in a separate bucket."""
    flagged = []
    for row in rows:
        raw_edge = row.get("raw_edge")
        disagreement = raw_edge is not None and abs(float(raw_edge)) >= flag_threshold
        data_problem = row.get("market_mapping_status") in {"DATA_UNAVAILABLE", "DATA_MISMATCH"}
        row["flagged"] = bool(disagreement or data_problem)
        if row["flagged"]:
            flagged.append(row)
    return flagged


def run_all(model_dir: str | Path, output: str | Path, *, now: datetime | None = None,
            report_day=None) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Singapore"))
    report_day = report_day or now.astimezone(zone).date()
    bankroll = float(os.getenv("BANKROLL_USDC", "10000"))
    risk_config = RiskConfig.from_env()
    ledger = RiskBudgetLedger(risk_config, bankroll)
    paper_db = os.getenv("PAPER_DB_PATH", "data/daily/paper.db")
    ledger.load_prior(paper_db, report_day.isoformat())
    try:
        from .paper_store import current_drawdown
        drawdown = current_drawdown(paper_db, bankroll) if Path(paper_db).exists() else 0.0
    except Exception:
        logging.exception("run_all: unable to compute drawdown from paper_store")
        drawdown = 0.0
    if drawdown >= risk_config.max_drawdown_fraction:
        ledger.breaker_reason = "account drawdown circuit breaker triggered"
    recommendations, statuses = [], {}
    client = PolymarketClient(timeout=30)
    market_events = {sport: client.all_events_by_tag(tag, page_size=100) for sport, tag in TAGS.items()}
    market_search = lambda item: client.public_search(f"{item.team_a} {item.team_b}")
    lol_sources = LolScheduleDiscovery().discover(report_day)
    lol_audit = build_schedule_audit(
        lol_sources, market_events["lol"], report_day=report_day, now=now,
        registry_path=os.getenv("WATCHER_REGISTRY_PATH", "data/daily/watcher_registry.json"),
        market_search=market_search,
    )

    def external_source(name, sport, call):
        try:
            events = call()
            matches = [make_match(
                source=name, sport=sport, league=row.league, team_a=row.team_a, team_b=row.team_b,
                start_time=row.start_time, event_name=row.event_name, best_of=row.best_of,
                event_status=row.status,
            ) for row in events if row.start_time.astimezone(zone).date() == report_day]
            return SourceResult(name, True, matches)
        except Exception as error:
            logging.exception("external_source %s failed", name)
            return SourceResult(name, False, [], repr(error))

    nba_sources = [
        external_source("nba_official", "nba", lambda: NbaOfficialProvider().schedule(report_day)),
        external_source("espn", "nba", lambda: EspnNbaProvider().schedule(report_day)),
        external_source("thesportsdb", "nba", lambda: TheSportsDbNbaProvider().schedule(report_day)),
        external_source("sportsrc", "nba", lambda: SportSrcNbaProvider().schedule(report_day)),
    ]
    panda = PandaScoreProvider()
    grid = GridOpenAccessProvider()
    bo3_source = external_source("bo3", "cs2", lambda: Bo3Cs2Provider().schedule(report_day))
    target_tournaments = {row.event_name for row in bo3_source.matches}
    cs2_sources = [
        bo3_source,
        external_source("esportagenda_cs2", "cs2", lambda: EsportAgendaCs2Provider().schedule(
            report_day, target_tournaments)),
        external_source("grid", "cs2", lambda: grid.schedule(report_day)),
        external_source("pandascore", "cs2", lambda: panda.schedule("cs2", report_day)),
    ]
    nba_audit = build_schedule_audit(
        nba_sources, market_events["nba"], report_day=report_day, now=now,
        registry_path=os.getenv("WATCHER_REGISTRY_PATH", "data/daily/watcher_registry.json"),
        target_leagues=("NBA",),
        market_search=market_search,
    )
    cs2_audit = build_schedule_audit(
        cs2_sources, market_events["cs2"], report_day=report_day, now=now,
        registry_path=os.getenv("WATCHER_REGISTRY_PATH", "data/daily/watcher_registry.json"),
        target_leagues=("CS2",),
        market_search=market_search,
    )
    audits = {"lol": lol_audit, "nba": nba_audit, "cs2": cs2_audit}
    audit_matches = {sport: audit["matches"] for sport, audit in audits.items()}
    for sport, audit in audits.items():
        append_schedule_audit(
            audit, os.getenv("SCHEDULE_AUDIT_LOG",
                             f"data/daily/schedule_audits/{report_day.isoformat()}-{sport}.jsonl"),
        )
    for sport, tag in TAGS.items():
        filename = "lol_meta_model.json" if sport == "lol" else f"{sport}_model.json"
        path = Path(model_dir) / filename
        if not path.exists():
            statuses[sport] = {"ready": False, "reason": "模型工件不存在"}
            continue
        if sport == "cs2":
            model, evaluation = load_cs2(path)
        elif sport == "lol":
            model, evaluation = load_lol_meta(path)
        elif sport == "nba":
            model, evaluation = load_nba(path)
        else:
            model, evaluation = load_model(path)
        events = market_events[sport]
        if sport == "cs2":
            sport_rows = analyze_cs2(model, evaluation, events, now=now, bankroll=bankroll,
                                     schedule_matches=audit_matches[sport], risk_config=risk_config,
                                     ledger=ledger, group_key=f"{sport}:{report_day.isoformat()}")
        elif sport == "lol":
            sport_rows = analyze_lol_meta(model, evaluation, events, now=now, bankroll=bankroll,
                                          schedule_matches=audit_matches[sport], risk_config=risk_config,
                                          ledger=ledger, group_key=f"{sport}:{report_day.isoformat()}")
        else:
            sport_rows = analyze_sport(sport, model, evaluation, events, now=now, bankroll=bankroll,
                                       schedule_matches=audit_matches[sport], risk_config=risk_config,
                                       ledger=ledger, group_key=f"{sport}:{report_day.isoformat()}")
        recommendations.extend(sport_rows)
        statuses[sport] = {
            "ready": True, "artifact_ready": True,
            "trained_through": model.trained_through, "samples": model.samples,
            "probability_approved": bool(evaluation.get("approved_for_probability_use")),
            "real_money_approved": bool(evaluation.get("approved_for_real_money")),
            "today_markets": len(sport_rows),
            "today_prestart_markets": sum(not row["market_started"] for row in sport_rows),
            "today_probability_eligible": sum(row["probability_eligible"] for row in sport_rows),
            "today_bet_candidates": sum(row["action"] == "BET" for row in sport_rows),
        }
    flag_threshold = float(os.getenv("FLAG_DISAGREEMENT_THRESHOLD", str(FLAG_DISAGREEMENT_DEFAULT)))
    staleness = {}
    for sport, status in statuses.items():
        if status.get("ready") and status.get("trained_through"):
            try:
                trained = datetime.fromisoformat(str(status["trained_through"])).date()
                staleness[sport] = (now.date() - trained).days
            except ValueError:
                pass
    retrain_interval = int(os.getenv("MODEL_RETRAIN_INTERVAL_DAYS", "42"))
    max_staleness = max(staleness.values(), default=0)
    model_staleness = {
        "days_by_sport": staleness,
        "max_days": max_staleness,
        "interval_days": retrain_interval,
        "warning": max_staleness > retrain_interval,
    }
    flagged = flag_rows(recommendations, flag_threshold)
    if ledger.breaker_reason:
        for row in recommendations:
            row["flagged"] = True
        flagged = list(recommendations)
    paper_daily = _paper_daily_report(
        recommendations, bankroll, ledger.daily_committed, risk_config.max_daily_risk_fraction,
    )
    paper_mode = "已开启" if _paper_trading_enabled() else "未开启"
    risk_notes = [
        "NBA、LoL、CS2 分别训练和验收；CBA 已暂停。",
        f"虚拟投注：{paper_mode}；仅写入 paper.db，不涉及真实资金。",
        "真实资金下注仍保持关闭。",
    ]
    if ledger.breaker_reason:
        risk_notes.append(ledger.breaker_reason)
    report = {
        "report_date": report_day.isoformat(), "generated_at": now.isoformat(),
        "bankroll_usdc": bankroll, "recommendations": recommendations, "sport_status": statuses,
        "schedule_coverage": audits,
        "data_incomplete": any(audit["data_incomplete"] for audit in audits.values()),
        "flagged": flagged,
        "flag_threshold": flag_threshold,
        "paper_daily": paper_daily,
        "risk_notes": risk_notes,
        "model_staleness": model_staleness,
        "risk_status": {
            "bankroll_usdc": bankroll,
            "max_bet_fraction": risk_config.max_bet_fraction,
            "max_daily_risk_fraction": risk_config.max_daily_risk_fraction,
            "max_event_risk_fraction": risk_config.max_event_risk_fraction,
            "max_drawdown_fraction": risk_config.max_drawdown_fraction,
            "daily_committed_fraction": ledger.daily_committed,
            "current_drawdown": drawdown,
            "circuit_breaker": ledger.breaker_reason is not None,
            "circuit_breaker_reason": ledger.breaker_reason,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
