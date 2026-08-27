from __future__ import annotations
import json
import copy
import logging
import math
import os
import re
from time import perf_counter
import unicodedata
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from .cs2_model import Cs2Model, load_cs2
from .lol_meta_model import LolDraftGame, LolMetaModel, load_lol_meta
from .lol_model import EloModel, load_model, series_probability
from .nba_model import NbaModel, load_nba
from .providers.polymarket import PolymarketClient
from .providers.live_data import (
    Bo3Cs2Provider, EsportAgendaCs2Provider, EspnCoreNbaProvider, EspnNbaProvider,
    GridOpenAccessProvider, HupuNbaProvider, NbaOfficialProvider,
    PandaScoreProvider, SportSrcNbaProvider, TheSportsDbNbaProvider,
)
from .risk import RiskBudgetLedger, RiskConfig, binary_share_math, kelly_fraction, paper_recommend, recommend
from .entities import canonical_team, normalized_name
from .betting_gate import bet_status, can_place_real_bet, should_place_virtual_bet
from .context import (load_recent_form, lol_roster_health, player_display_names,
                      recent_form_artifact_generated_at, recent_form_artifact_health,
                      recent_form_for)
from .paper_store import (
    calc_roi, count_settled_virtual_bets, count_virtual_bets, current_drawdown,
    record_virtual_bet, virtual_account_balance,
)
from .costs import estimate_cost_rate
from .narrative import build_pre_match_summary
from .schedule import (
    LolScheduleDiscovery, SourceResult, append_schedule_audit, build_schedule_audit, make_match,
)
from .coverage_service import CoverageStore, build_coverage_report
from .roster import historical_roster
from .ai.analyst import EvidenceAnalyst, minimum_evidence_gate
from .ai.cache import AnalysisCache
from .ai.client import client_from_env
from .ai.prompts import PROMPT_VERSION
from .ai.schemas import Evidence
from .feedback import FeedbackStore, PredictionSnapshot
from .health_report import build_system_health
from .intelligence import DataQualityScore, RecommendationState
from .evidence_pipeline import EvidenceStore, build_match_evidence, evidence_availability, evidence_hash
from .provider_health import PROVIDER_CAPABILITIES, ProviderHealth, ProviderHealthStore
from .lifecycle import MatchLifecycleStore
TAGS = {"nba": "745", "lol": "65", "cs2": "100780"}
def _paper_trading_enabled() -> bool:
    """Paper betting can run before real-money ROI acceptance.
    This only controls the append-only paper ledger.  Real-money approval is a
    separate, stricter ``approved_for_real_money`` flag and is never implied.
    """
    return os.getenv("PAPER_TRADING_ENABLED", "true").casefold() == "true"
def _real_trading_enabled() -> bool:
    return os.getenv("REAL_TRADING_DISABLED", "true").casefold() == "false"
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


def _direction_alignment(model_probabilities: list[float], market_probabilities: list[float]) -> bool:
    """Return True when model side and market recommended side agree.

    ``model_side`` uses the model probability for team A > 0.5, while
    ``market_recommended_side`` uses the market probability for team A > 0.5.
    """
    if not model_probabilities or not market_probabilities or len(model_probabilities) < 2 or len(market_probabilities) < 2:
        return False
    model_side = "team_a" if float(model_probabilities[0]) > 0.5 else "team_b"
    market_recommended_side = "team_a" if float(market_probabilities[0]) > 0.5 else "team_b"
    return model_side == market_recommended_side
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
    by_sport: dict[str, dict[str, float | int]] = {}
    for row in rows:
        sport = str(row.get("sport") or "unknown")
        stat = by_sport.setdefault(sport, {"bets": 0, "skipped": 0, "stake": 0.0})
        if row.get("action") == "BET":
            stat["bets"] = int(stat["bets"]) + 1
            stat["stake"] = float(stat["stake"]) + float(row.get("stake") or 0)
        else:
            stat["skipped"] = int(stat["skipped"]) + 1
    return {
        "bet_count": len(bets),
        "skipped_count": len(rows) - len(bets),
        "total_stake": total_stake,
        "committed_fraction": committed_fraction,
        "remaining_limit": max(0.0, bankroll * (max_daily_risk_fraction - committed_fraction)),
        "by_sport": by_sport,
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


def _numeric_market_value(*values: object) -> float:
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(parsed, default=0.0)


def _daily_market_quality(event: dict, market: dict, risk_config: RiskConfig) -> dict:
    """Apply the configurable liquidity/volume gate used by the daily push."""
    volume = _numeric_market_value(
        market.get("volume"), market.get("volumeNum"), market.get("volume24hr"),
        event.get("volume"), event.get("volumeNum"), event.get("volume24hr"),
    )
    liquidity = _numeric_market_value(
        market.get("liquidity"), market.get("liquidityNum"), market.get("liquidityClob"),
        event.get("liquidity"), event.get("liquidityNum"),
    )
    minimum_volume = float(os.getenv("MIN_DAILY_MARKET_VOLUME", "1000"))
    minimum_liquidity = float(os.getenv("MIN_DAILY_MARKET_LIQUIDITY", str(risk_config.min_available_size)))
    return {
        "volume": volume, "liquidity": liquidity,
        "minimum_volume": minimum_volume, "minimum_liquidity": minimum_liquidity,
        "qualified": volume >= minimum_volume and liquidity >= minimum_liquidity,
    }


def _market_movement_from_history(history: list[dict], now: datetime) -> dict | None:
    points = sorted((int(item.get("t")), float(item.get("p"))) for item in history
                    if item.get("t") is not None and item.get("p") is not None)
    if len(points) < 2:
        return None
    latest_t, latest_price = points[-1]
    def movement(minutes: int):
        cutoff = int(now.timestamp()) - minutes * 60
        prior = min(points, key=lambda item: abs(item[0] - cutoff))
        return latest_price - prior[1]
    return {"source": "polymarket_clob_prices_history", "latest_timestamp": latest_t,
            "latest_trade_or_mark": latest_price, "movement_5m": movement(5),
            "movement_15m": movement(15), "movement_1h": movement(60),
            "historical_depth_available": False}


def _attach_data_quality(row: dict, sport: str, model_status: str = "OK") -> None:
    missing: list[str] = []
    lineup = str(row.get("lineup_status") or "UNKNOWN").upper()
    roster_quality = {"CONFIRMED": 1.0, "EXPECTED": .7, "HISTORICAL": .4}.get(lineup, .1)
    if lineup != "CONFIRMED":
        missing.append("CONFIRMED_LINEUP")
    recent_available = bool(row.get("recent_form_a") and row.get("recent_form_b"))
    if not recent_available:
        missing.append("RECENT_FORM")
    if sport == "nba":
        if not row.get("injury_status"):
            missing.append("INJURY_STATUS")
        if not row.get("confirmed_starters"):
            missing.append("CONFIRMED_STARTERS")
    quality = DataQualityScore(
        schedule_quality=1.0 if row.get("schedule_match_id") else .3,
        roster_quality=roster_quality,
        market_quality=1.0 if row.get("market_quality_qualified") else .3,
        stat_quality=.85 if recent_available else .25,
        news_quality=.7 if row.get("analyst_notes") else .3,
        model_freshness=1.0 if model_status == "OK" else .2,
        missing=tuple(sorted(set(missing))),
    )
    if row.get("market_mapping_status") in {"MARKET_NOT_FOUND", "NOT_IN_SCHEDULE"}:
        state = RecommendationState.NO_MARKET
    elif quality.level == "LOW":
        state = RecommendationState.WATCH
    elif row.get("action") == "BET" and not quality.missing and float(row.get("expected_value") or 0) > .15:
        state = RecommendationState.STRONG_VALUE
    elif row.get("action") == "BET":
        state = RecommendationState.VALUE
    else:
        state = RecommendationState.WATCH
    row.update({
        "data_quality_score": quality.score,
        "data_quality_level": quality.level,
        "data_quality_missing": list(quality.missing),
        "recommendation_state": state.value,
    })


def _apply_intelligence_gate(row: dict, sport: str) -> None:
    missing = set(row.get("data_quality_missing") or [])
    critical = {"lol": {"RECENT_FORM"}, "cs2": {"RECENT_FORM", "ROSTER", "MAP_DATA"},
                "nba": {"INJURY_STATUS", "CONFIRMED_STARTERS"}}[sport]
    critical_missing = sorted(value for value in missing if value in critical or
                              (value.startswith("STALE_") and value[6:] in critical))
    score = float(row.get("data_quality_score") or 0)
    adjusted_edge = float(row.get("adjusted_edge") or row.get("edge") or 0)
    gate_reason = None
    if score < .6:
        gate_reason = "LOW_DATA_QUALITY"
    elif critical_missing:
        gate_reason = "CRITICAL_EVIDENCE_MISSING:" + ",".join(critical_missing)
    elif score < .8 and adjusted_edge < .08:
        gate_reason = "MEDIUM_DATA_QUALITY_REQUIRES_8PCT_EDGE"
    if gate_reason:
        row.update({"action": "NO_BET", "stake": 0.0, "stake_fraction": 0.0,
                    "decision": "NO TRADE", "recommendation_state": "WATCH",
                    "decision_gate": gate_reason})
        row["reasons"] = list(row.get("reasons") or []) + [gate_reason]
    else:
        row["decision_gate"] = "PASS"


def _apply_post_ai_quality_gates(row: dict, sport: str, model_status: str) -> None:
    """Apply the unchanged quality gates independently to each candidate."""
    _attach_data_quality(row, sport, model_status)
    stale_critical = sorted({kind for kind, state in row.get("evidence_type_status", {}).items()
                             if state == "AVAILABLE_STALE" and kind in {
                                 "ROSTER", "RECENT_FORM", "INJURY", "CONFIRMED_STARTERS"}})
    if stale_critical:
        row["data_quality_missing"] = sorted(set(row["data_quality_missing"]) |
                                             {f"STALE_{kind}" for kind in stale_critical})
        row["data_quality_score"] = max(0.0, row["data_quality_score"] - .1 * len(stale_critical))
        row["data_quality_level"] = "LOW" if row["data_quality_score"] < .6 else "MEDIUM"
    _apply_intelligence_gate(row, sport)


def _attach_risk_audit(row: dict, bankroll: float, config: RiskConfig) -> None:
    market_price = float(row.get("execution_price") or row.get("market_probability") or 0)
    market_probability = float(row.get("market_probability") or market_price)
    probability = float(row.get("execution_decision_probability") or
                        row.get("final_probability") or row.get("model_probability") or .5)
    raw_edge = probability - market_probability
    quality = float(row.get("data_quality_score") or 0)
    # Signed penalties shrink probability toward the market on either side.
    data_penalty = max(0.0, 1 - quality) * raw_edge
    confidence = float(row.get("ai_confidence") or (1.0 if row.get("ai_status") == "LLM_ACTIVE" else 0.0))
    ai_penalty = (1 - confidence) * float(row.get("ai_adjustment") or 0)
    # Shrink only the Quant contribution toward the market.  Do not derive this
    # from row.edge because that field may already include execution costs.
    quant_confidence = float(row.get("confidence") or 0)
    quant_edge = float(row.get("model_probability") or probability) - market_probability
    uncertainty_penalty = (1 - quant_confidence) * quant_edge
    fee_rate = max(0.0, float(row.get("estimated_cost_rate") or 0))
    slippage = max(0.0, float(row.get("estimated_slippage_per_share") or 0))
    adjusted_probability = max(.001, min(.999, probability - data_penalty
                                         - uncertainty_penalty - ai_penalty))
    economics = binary_share_math(adjusted_probability, market_price,
                                  fee_rate_on_capital=fee_rate,
                                  slippage_per_share=slippage)
    adjusted_edge = economics["expected_profit_per_share"]
    odds = economics["decimal_odds"]
    kelly = kelly_fraction(adjusted_probability, odds) if odds > 1 else 0.0
    fractional = kelly * config.kelly_scale; before_cap = bankroll * fractional
    final_stake = float(row.get("stake") or 0)
    cap_reason = "NONE" if final_stake >= before_cap else ("DECISION_GATE" if row.get("decision_gate") != "PASS" else "RISK_CAP")
    row.update({"market_fair_probability": market_probability, "raw_model_edge": raw_edge,
                "executable_edge": economics["executable_edge"],
                "expected_profit_per_share": economics["expected_profit_per_share"],
                "expected_roi_on_capital": economics["expected_roi_on_capital"],
                "adjusted_edge": adjusted_edge, "risk_calculation": {
        "model_probability": row.get("model_probability"), "market_fair_probability": market_probability,
        "execution_price": market_price, "decimal_odds": odds, "raw_model_edge": raw_edge,
        "data_quality_penalty": data_penalty, "uncertainty_penalty": uncertainty_penalty,
        "quant_confidence": quant_confidence,
        "ai_confidence_penalty": ai_penalty, "fee_rate_on_capital": fee_rate,
        "adjusted_probability": adjusted_probability, **economics,
        "kelly_fraction": kelly, "fractional_kelly": fractional,
        "single_bet_cap": bankroll * config.max_bet_fraction,
        "event_cap": bankroll * config.max_event_risk_fraction,
        "daily_cap": bankroll * config.max_daily_risk_fraction,
        "stake_before_cap": before_cap, "cap_reason": cap_reason, "final_stake": final_stake}})


def _apply_daily_market_gate(rec, quality: dict):
    if rec.action != "BET" or quality["qualified"]:
        return rec
    reasons = list(rec.reasons)
    if quality["volume"] < quality["minimum_volume"]:
        reasons.append(
            f"market volume below daily threshold ({quality['volume']:.0f} < {quality['minimum_volume']:.0f})"
        )
    if quality["liquidity"] < quality["minimum_liquidity"]:
        reasons.append(
            f"market liquidity below daily threshold ({quality['liquidity']:.0f} < {quality['minimum_liquidity']:.0f})"
        )
    return replace(
        rec, action="NO_BET", decision="NO TRADE", stake=0.0,
        stake_fraction=0.0, reasons=tuple(reasons),
    )


def _scheduled_market_events(sport: str, events: list[dict],
                             schedule_matches: list[dict]) -> list[dict]:
    """Keep only market events present in today's discovered schedule."""
    scheduled_events = []
    for event in events:
        market = _main_market(event)
        if not market:
            continue
        scheduled = _start(market)
        outcomes = [_text(value) for value in _field(market.get("outcomes"))]
        if len(outcomes) != 2 or scheduled is None:
            continue
        if _find_schedule_match(schedule_matches, sport, outcomes[0], outcomes[1], scheduled):
            scheduled_events.append(event)
    return scheduled_events
def analyze_sport(sport: str, model: EloModel | NbaModel, evaluation: dict, events: list[dict], *,
                  now: datetime, bankroll: float, estimated_cost: float | None = None,
                  schedule_matches: list[dict] | None = None,
                  risk_config: RiskConfig | None = None,
                  ledger: RiskBudgetLedger | None = None,
                  group_key: str | None = None,
                  paper_db: str | Path | None = None) -> list[dict]:
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
        direction_match = _direction_alignment(model_ps, prices)
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
                trading_enabled=_real_trading_enabled() and probability_ok and not started,
                kelly_scale=risk_config.kelly_scale, max_bet_fraction=cap,
                min_edge=risk_config.min_edge, min_confidence=risk_config.min_confidence,
                max_spread=risk_config.max_spread, min_available_size=risk_config.min_available_size,
                max_depth_fraction=risk_config.max_depth_fraction, risk_reasons=risk_reasons,
            )
        market_quality = _daily_market_quality(event, market, risk_config)
        rec = _apply_daily_market_gate(rec, market_quality)
        reasons = (model.explain(team_a, team_b, scheduled)
                   if sport == "nba" and isinstance(model, NbaModel)
                   else model.explain(team_a, team_b))
        reasons.append(f"{market_team_a} 独立胜率 {p_a:.1%}，{market_team_b} 独立胜率 {1-p_a:.1%}")
        reasons.append("市场价格只用于估值和下注判断，不进入独立胜率模型")
        if not probability_ok:
            reasons.append("概率模型未通过锁箱验收或队伍历史样本不足")
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
            "market_probability": prices[side], "market_fair_probability": prices[side],
            "execution_price": ask, "estimated_cost_rate": cost,
            "market_token_id": (_field(market.get("clobTokenIds")) + [None, None])[side],
            "edge": rec.decision_probability - ask - cost,
            "probability_eligible": probability_ok,
            "real_money_approved": money_ok,
            "market_started": started,
            "market_comparison_valid": not started,
            "hours_to_start": hours_to_start,
            "decision_window": decision_window,
            "probability_plausible": not bool(sanity),
            "schedule_matched": bool(schedule_match and schedule_match.get("market_mapping_status") == "MATCHED"),
            "schedule_match_id": schedule_match.get("match_id") if schedule_match else None,
            "market_mapping_status": schedule_match.get("market_mapping_status") if schedule_match else "NOT_IN_SCHEDULE",
            "lineup_status": "未知",
            "ev_tier": _ev_tier(rec.expected_value),
            "market_volume": market_quality["volume"],
            "market_liquidity": market_quality["liquidity"],
            "minimum_market_volume": market_quality["minimum_volume"],
            "minimum_market_liquidity": market_quality["minimum_liquidity"],
            "market_quality_qualified": market_quality["qualified"],
            "daily_candidate": rec.action == "BET" and rec.expected_value > 0,
            "direction_match": bool(direction_match),
            "reasons": reasons + list(rec.reasons),
        })
        _attach_bet_fields(row, model_probability=model_probability,
                           execution_price=ask, bankroll=bankroll,
                           cap=cap, paper_db=paper_db)
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
def _virtual_stake(model_probability: float | None, execution_price: float | None,
                    bankroll: float, cap: float) -> float:
    """Compute 1/4-Kelly virtual stake, capped by the active risk budget."""
    if (model_probability is None or execution_price is None or execution_price <= 0
            or bankroll <= 0 or cap <= 0):
        return 0.0
    decimal_odds = 1.0 / float(execution_price)
    fraction = kelly_fraction(float(model_probability), decimal_odds) * 0.25
    return round(float(bankroll) * min(max(0.0, cap), max(0.0, fraction)), 2)
def _attach_bet_fields(row: dict, *, model_probability: float | None,
                       execution_price: float | None, bankroll: float,
                       cap: float, paper_db: str | Path | None) -> dict:
    """Attach cold-start virtual-betting metadata to a recommendation row."""
    virtual_ok, virtual_reason = should_place_virtual_bet(row)
    row["virtual_bet"] = virtual_ok
    row["virtual_bet_reason"] = virtual_reason
    row["stake_virtual"] = (_virtual_stake(model_probability, execution_price, bankroll, cap)
                            if virtual_ok else 0.0)
    row["bet_status"] = bet_status(row, paper_db)
    row["real_bet_reason"] = can_place_real_bet(row, paper_db)[1] if virtual_ok else ""
    if virtual_ok and row["real_bet_reason"]:
        row.setdefault("reasons", []).append(row["real_bet_reason"])
    return row


def _recompute_post_ai_decision(row: dict, *, final_probability: float,
                                bankroll: float, config: RiskConfig,
                                ledger: RiskBudgetLedger, group_key: str) -> float:
    """Rebuild all executable economics from one candidate probability."""
    row["final_probability"] = final_probability
    row["decision_probability"] = final_probability
    if row.get("market_started") or row.get("execution_price") is None:
        row.update({"action": "NO_BET", "decision": "NO TRADE", "stake": 0.0,
                    "stake_fraction": 0.0, "daily_candidate": False})
        return 0.0
    event_id = str(row.get("event_id") or row.get("schedule_match_id") or "")
    price = float(row["execution_price"])
    cost = max(0.0, float(row.get("estimated_cost_rate") or 0))
    cap = ledger.cap_for(event_id, group_key)
    risk_reasons = ledger.exhausted_reasons(event_id, group_key)
    if _paper_trading_enabled():
        rec = paper_recommend(
            event_id=event_id, outcome=str(row.get("outcome") or ""),
            model_probability=final_probability, decimal_odds=1 / price,
            bankroll=bankroll, estimated_cost=cost, max_bet_fraction=cap,
            direction_match=bool(row.get("direction_match")), risk_reasons=risk_reasons,
        )
    else:
        rec = recommend(
            event_id=event_id, outcome=str(row.get("outcome") or ""),
            model_probability=final_probability, decimal_odds=1 / price,
            bankroll=bankroll, confidence=float(row.get("confidence") or 0),
            spread=row.get("market_spread"), available_size=float(row.get("market_liquidity") or 0),
            estimated_cost=cost,
            trading_enabled=_real_trading_enabled() and bool(row.get("probability_eligible")),
            kelly_scale=config.kelly_scale, max_bet_fraction=cap,
            min_edge=config.min_edge, min_confidence=config.min_confidence,
            max_spread=config.max_spread, min_available_size=config.min_available_size,
            max_depth_fraction=config.max_depth_fraction, risk_reasons=risk_reasons,
        )
    if not bool(row.get("market_quality_qualified")):
        rec = _apply_daily_market_gate(rec, {
            "qualified": False, "volume": float(row.get("market_volume") or 0),
            "liquidity": float(row.get("market_liquidity") or 0),
            "minimum_volume": float(row.get("minimum_market_volume") or 0),
            "minimum_liquidity": float(row.get("minimum_market_liquidity") or 0),
        })
    ev = final_probability / price - 1 - cost
    row.update({
        "decision_probability": final_probability,
        "raw_edge": final_probability - price, "edge": final_probability - price - cost,
        "expected_value": ev, "action": rec.action, "decision": rec.decision,
        "stake": rec.stake, "stake_fraction": rec.stake_fraction,
        "confidence": rec.confidence, "daily_candidate": rec.action == "BET" and ev > 0,
        "ev_tier": _ev_tier(ev), "post_ai_risk_cap": cap,
    })
    row["reasons"] = list(row.get("reasons") or []) + list(rec.reasons)
    return cap


def _select_llm_execution(quant_candidate: dict, final_candidate: dict,
                          *, shadow_mode: bool, baseline: float) -> dict:
    """Select one executable candidate and retain the other for attribution."""
    execution = copy.deepcopy(quant_candidate if shadow_mode else final_candidate)
    execution.update({
        "llm_decision_mode": "SHADOW" if shadow_mode else "ACTIVE",
        "llm_shadow_mode": shadow_mode, "quant_probability": baseline,
        "quant_action": quant_candidate.get("action"),
        "quant_stake": float(quant_candidate.get("stake") or 0),
        "quant_ev": quant_candidate.get("expected_value"),
        "shadow_final_action": final_candidate.get("action"),
        "shadow_final_stake": float(final_candidate.get("stake") or 0),
        "shadow_final_ev": final_candidate.get("expected_value"),
        "shadow_final_risk_calculation": final_candidate.get("risk_calculation"),
    })
    return execution


def _patch_meta_context(model, sport: str, roster_a, roster_b) -> tuple[list[str], float | None, float | None]:
    """Derive top patch heroes and each roster's coverage from the frozen model."""
    if sport != "lol" or not hasattr(model, "patch_champion_ratings"):
        return [], None, None
    keys = list(getattr(model, "patch_champion_ratings", {}).keys())
    patches = sorted({str(key).split("|", 1)[0] for key in keys if "|" in str(key)})
    patch = os.getenv("LOL_CURRENT_PATCH") or (patches[-1] if patches else "")
    if not patch:
        return [], None, None
    heroes = sorted(
        ((str(key).split("|", 1)[1], float(rating))
         for key, rating in model.patch_champion_ratings.items()
         if str(key).startswith(patch + "|") and "|" in str(key)),
        key=lambda item: item[1], reverse=True,
    )[:5]
    if not heroes:
        return [], None, None
    hero_names = [hero for hero, _ in heroes]
    def coverage(roster):
        if not roster or len(hero_names) != 5:
            return None
        covered = 0
        for hero in hero_names:
            if any(f"{player}|{hero}" in model.player_champion_ratings for player in roster):
                covered += 1
        return covered / 5 * 100.0
    return hero_names, coverage(roster_a), coverage(roster_b)
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
                  lineup_status: str = "未知",
                  paper_db: str | Path | None = None) -> dict:
    risk_config = risk_config or RiskConfig()
    if ledger is None:
        ledger = RiskBudgetLedger(risk_config, bankroll)
    paper_enabled = _paper_trading_enabled()
    side = max(range(2), key=lambda index: probabilities[index] - prices[index])
    direction_match = _direction_alignment(probabilities, prices)
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
            trading_enabled=_real_trading_enabled() and probability_ok and not started,
            kelly_scale=risk_config.kelly_scale, max_bet_fraction=cap,
            min_edge=risk_config.min_edge, min_confidence=risk_config.min_confidence,
            max_spread=risk_config.max_spread, min_available_size=risk_config.min_available_size,
            max_depth_fraction=risk_config.max_depth_fraction, risk_reasons=risk_reasons,
        )
    market_quality = _daily_market_quality(event, market, risk_config)
    rec = _apply_daily_market_gate(rec, market_quality)
    if not probability_ok:
        reasons.append("阵容未知、阵容过期、样本不足或概率模型未通过验收，因此只展示研究值。")
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
        "market_probability": prices[side], "market_fair_probability": prices[side],
        "execution_price": ask, "estimated_cost_rate": cost,
        "market_token_id": (_field(market.get("clobTokenIds")) + [None, None])[side],
        "edge": rec.decision_probability - ask - cost,
        "probability_eligible": probability_ok,
        "real_money_approved": money_ok,
        "market_started": started,
        "market_comparison_valid": not started,
        "hours_to_start": hours_to_start,
        "decision_window": decision_window,
        "probability_plausible": not bool(sanity),
        "schedule_matched": bool(schedule_match and schedule_match.get("market_mapping_status") == "MATCHED"),
        "schedule_match_id": schedule_match.get("match_id") if schedule_match else None,
        "market_mapping_status": schedule_match.get("market_mapping_status") if schedule_match else "NOT_IN_SCHEDULE",
        "lineup_status": lineup_status,
        "ev_tier": _ev_tier(rec.expected_value),
        "market_volume": market_quality["volume"],
        "market_liquidity": market_quality["liquidity"],
        "minimum_market_volume": market_quality["minimum_volume"],
        "minimum_market_liquidity": market_quality["minimum_liquidity"],
        "market_quality_qualified": market_quality["qualified"],
        "daily_candidate": rec.action == "BET" and rec.expected_value > 0,
        "direction_match": bool(direction_match),
        "reasons": reasons + list(rec.reasons),
    })
    _attach_bet_fields(row, model_probability=model_probability,
                       execution_price=ask, bankroll=bankroll,
                       cap=cap, paper_db=paper_db)
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
                group_key: str | None = None,
                paper_db: str | Path | None = None) -> list[dict]:
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
        roster_state_a, roster_state_b = historical_roster(roster_a), historical_roster(roster_b)
        lineup_status = "HISTORICAL" if roster_a and roster_b else "UNKNOWN"
        probability_ok = bool(evaluation.get("approved_for_probability_use")) and known and roster_ok
        reasons = [
            f"阵容感知胜率：{outcomes[0]} {probability:.1%}，{outcomes[1]} {1-probability:.1%}。",
            f"历史样本：{a} {model.team_games.get(a, 0)} 场，{b} {model.team_games.get(b, 0)} 场。",
            "当前基线包含战队和五人阵容强度；地图池、veto 与 LAN/线上层仍在补充。",
            roster_state_a.explanatory_note,
        ]
        row = _research_row("cs2", event, market, scheduled, outcomes, prices,
                             [probability, 1-probability], probability_ok=probability_ok,
                             money_ok=bool(evaluation.get("approved_for_real_money")),
                             now=now, bankroll=bankroll, reasons=reasons,
                             schedule_matches=schedule_matches, risk_config=risk_config, ledger=ledger, group_key=group_key,
                             lineup_status=lineup_status, paper_db=paper_db)
        row.update({
            "lineup_a": player_display_names("cs2", roster_a),
            "lineup_b": player_display_names("cs2", roster_b),
            "recent_form_a": recent_form_for("cs2", a),
            "recent_form_b": recent_form_for("cs2", b),
            "recent_form_artifact_generated_at": recent_form_artifact_generated_at(),
            "roster_published_at": min((model.team_last_game.get(a), model.team_last_game.get(b)),
                                       key=lambda value: value or "") or None,
            "best_of": int(schedule_match.get("best_of") or 0) if schedule_match else 1,
            "format": f"BO{int(schedule_match.get('best_of') or 1)}" if schedule_match else "BO1",
            "map_strengths_a": [],
            "map_strengths_b": [],
            "sample_a": model.team_games.get(a, 0),
            "sample_b": model.team_games.get(b, 0),
            "roster_status_a": roster_state_a.status.value,
            "roster_status_b": roster_state_b.status.value,
            "entity_debug": {
                "team_a": {"canonical_team_id": a, "provider_team_name": outcomes[0],
                           "entity_match_status": "MATCHED" if a in model.team_ratings else "TRAINING_DATA_MISSING",
                           "roster_rows": len(roster_a), "recent_form_rows": (recent_form_for("cs2", a) or {}).get("last_n", 0),
                           "player_rows": len(roster_a)},
                "team_b": {"canonical_team_id": b, "provider_team_name": outcomes[1],
                           "entity_match_status": "MATCHED" if b in model.team_ratings else "TRAINING_DATA_MISSING",
                           "roster_rows": len(roster_b), "recent_form_rows": (recent_form_for("cs2", b) or {}).get("last_n", 0),
                           "player_rows": len(roster_b)},
            },
        })
        rows.append(row)
    return rows
def analyze_lol_meta(model: LolMetaModel, evaluation: dict, events: list[dict], *,
                     now: datetime, bankroll: float,
                     schedule_matches: list[dict] | None = None,
                     risk_config: RiskConfig | None = None,
                     ledger: RiskBudgetLedger | None = None,
                     group_key: str | None = None,
                     paper_db: str | Path | None = None) -> list[dict]:
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
        roster_state_a, roster_state_b = historical_roster(roster_a), historical_roster(roster_b)
        lineup_status = "HISTORICAL" if roster_a and roster_b else "UNKNOWN"
        probability_ok = bool(evaluation.get("approved_for_probability_use")) and known and roster_ok
        reasons = [
            f"赛前阵容模型：{outcomes[0]} {probability:.1%}，{outcomes[1]} {1-probability:.1%}（BO{best_of}）。",
            f"历史样本：{a} {model.team_games.get(a, 0)} 局，{b} {model.team_games.get(b, 0)} 局。",
            "BP 未开始时不使用英雄选择；BP 完成后必须重新计算版本英雄强度与选手英雄熟练度。",
            roster_state_a.explanatory_note,
        ]
        heroes, coverage_a, coverage_b = _patch_meta_context(model, "lol", roster_a, roster_b)
        recent_a = recent_form_for("lol", a)
        recent_b = recent_form_for("lol", b)
        if recent_a and recent_b:
            reasons.append(
                f"近期状态：{a} 最近{recent_a['last_n']}场 {recent_a['wins']}胜{recent_a['losses']}负；"
                f"{b} 最近{recent_b['last_n']}场 {recent_b['wins']}胜{recent_b['losses']}负。"
            )
        if heroes and coverage_a is not None and coverage_b is not None:
            reasons.append(
                f"版本池覆盖：{a} {coverage_a:.0f}%，{b} {coverage_b:.0f}%（基于冻结训练样本，不含未开始的 BP）。"
            )
        elif heroes:
            reasons.append("版本英雄池存在，但至少一队的历史阵容覆盖率缺失；未使用默认值替代。")
        row = _research_row("lol", event, market, scheduled, outcomes, prices,
                            [probability, 1-probability], probability_ok=probability_ok,
                            money_ok=bool(evaluation.get("approved_for_real_money")),
                            now=now, bankroll=bankroll, reasons=reasons,
                            schedule_matches=schedule_matches, risk_config=risk_config, ledger=ledger, group_key=group_key,
                            lineup_status=lineup_status, paper_db=paper_db)
        row.update({
            "lineup_a": player_display_names("lol", roster_a),
            "lineup_b": player_display_names("lol", roster_b),
            "recent_form_a": recent_a,
            "recent_form_b": recent_b,
            "recent_form_artifact_generated_at": recent_form_artifact_generated_at(),
            "roster_published_at": min((model.team_last_game.get(a), model.team_last_game.get(b)),
                                       key=lambda value: value or "") or None,
            "best_of": best_of,
            "format": f"BO{best_of}",
            "patch_meta_heroes": heroes,
            "meta_coverage_a": coverage_a,
            "meta_coverage_b": coverage_b,
            "sample_a": model.team_games.get(a, 0),
            "sample_b": model.team_games.get(b, 0),
            "roster_status_a": roster_state_a.status.value,
            "roster_status_b": roster_state_b.status.value,
            "entity_debug": {
                "team_a": {"canonical_team_id": a, "provider_team_name": outcomes[0],
                           "entity_match_status": "MATCHED" if a in model.team_ratings else "TRAINING_DATA_MISSING",
                           "roster_rows": len(roster_a), "recent_form_rows": (recent_a or {}).get("last_n", 0),
                           "player_rows": len(roster_a)},
                "team_b": {"canonical_team_id": b, "provider_team_name": outcomes[1],
                           "entity_match_status": "MATCHED" if b in model.team_ratings else "TRAINING_DATA_MISSING",
                           "roster_rows": len(roster_b), "recent_form_rows": (recent_b or {}).get("last_n", 0),
                           "player_rows": len(roster_b)},
            },
        })
        rows.append(row)
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
        drawdown = current_drawdown(paper_db, bankroll) if Path(paper_db).exists() else 0.0
    except Exception:
        logging.exception("run_all: unable to compute drawdown from paper_store")
        drawdown = 0.0
    ledger.paper_mode = _paper_trading_enabled()
    llm_client = client_from_env()
    ai_analyst = EvidenceAnalyst(llm_client, AnalysisCache(paper_db))
    llm_shadow_mode = os.getenv("LLM_SHADOW_MODE", "true").casefold() == "true"
    llm_telemetry = {
        "provider": getattr(llm_client, "provider", None), "model": getattr(llm_client, "model", None),
        "configured": llm_client is not None, "eligible": 0, "attempted": 0, "success": 0,
        "cache_hits": 0, "fallback": 0, "evidence_gate_fallback": 0,
        "validation_failures": 0, "request_failures": 0, "last_success": None,
        "last_error_category": None, "adjustment_sum": 0.0,
        "positive_adjustments": 0, "negative_adjustments": 0, "zero_adjustments": 0,
    }
    feedback_store = FeedbackStore(paper_db)
    evidence_store = EvidenceStore(paper_db)
    if drawdown >= risk_config.drawdown_circuit_fraction:
        if ledger.paper_mode:
            ledger.drawdown_level = "warn"
            ledger.warn_reason = ("virtual account drawdown reached circuit threshold; "
                                  "paper mode halves exposure instead of pausing")
        else:
            ledger.drawdown_level = "circuit"
            ledger.breaker_reason = "account drawdown circuit breaker triggered"
    elif drawdown >= risk_config.drawdown_warn_fraction:
        ledger.drawdown_level = "warn"
        ledger.warn_reason = "account drawdown warn threshold triggered"
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
        attempted = datetime.now(timezone.utc); started = perf_counter()
        try:
            events = call()
            matches = [make_match(
                source=name, sport=sport, league=row.league, team_a=row.team_a, team_b=row.team_b,
                start_time=row.start_time, event_name=row.event_name, best_of=row.best_of,
                event_status=row.status,
            ) for row in events if row.start_time.astimezone(zone).date() == report_day]
            return SourceResult(name, True, matches, latency_ms=int((perf_counter() - started) * 1000),
                                last_attempt_at=attempted, last_success_at=datetime.now(timezone.utc))
        except Exception as error:
            logging.exception("external_source %s failed", name)
            return SourceResult(name, False, [], repr(error), int((perf_counter() - started) * 1000),
                                attempted, None)
    nba_sources = [
        external_source("nba_official", "nba", lambda: NbaOfficialProvider().schedule(report_day)),
        external_source("hupu", "nba", lambda: HupuNbaProvider().schedule(report_day)),
        external_source("espn_core", "nba", lambda: EspnCoreNbaProvider().schedule(report_day)),
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
        events = _scheduled_market_events(sport, market_events[sport], audit_matches[sport])
        if sport == "cs2":
            sport_rows = analyze_cs2(model, evaluation, events, now=now, bankroll=bankroll,
                                     schedule_matches=audit_matches[sport], risk_config=risk_config,
                                     ledger=ledger, group_key=f"{sport}:{report_day.isoformat()}",
                                     paper_db=paper_db)
        elif sport == "lol":
            sport_rows = analyze_lol_meta(model, evaluation, events, now=now, bankroll=bankroll,
                                          schedule_matches=audit_matches[sport], risk_config=risk_config,
                                          ledger=ledger, group_key=f"{sport}:{report_day.isoformat()}",
                                          paper_db=paper_db)
        else:
            sport_rows = analyze_sport(sport, model, evaluation, events, now=now, bankroll=bankroll,
                                       schedule_matches=audit_matches[sport], risk_config=risk_config,
                                       ledger=ledger, group_key=f"{sport}:{report_day.isoformat()}",
                                       paper_db=paper_db)
        try:
            model_age = (report_day - datetime.fromisoformat(str(model.trained_through)).date()).days
        except ValueError:
            model_age = 10**9
        row_model_status = "STALE" if model_age > int(os.getenv("MODEL_RETRAIN_INTERVAL_DAYS", "42")) else "OK"
        for row in sport_rows:
            match_id = str(row.get("schedule_match_id") or row.get("event_id") or "")
            token_id = row.get("market_token_id")
            if token_id:
                try:
                    history = client.price_history(str(token_id), start_ts=int(now.timestamp()) - 4 * 3600,
                                                   end_ts=int(now.timestamp()), fidelity=5)
                    row["market_movement"] = _market_movement_from_history(history, now)
                except Exception as error:
                    row["market_movement"] = None
                    row["market_movement_error"] = repr(error)
            evidence = build_match_evidence(row, sport, match_id, now)
            evidence_store.put_many(evidence)
            availability = evidence_availability(evidence)
            evidence_gate = minimum_evidence_gate(evidence)
            llm_telemetry["eligible"] += int(evidence_gate["eligible"])
            baseline = float(row.get("model_probability") or .5)
            ai_result = ai_analyst.analyze(match_id=match_id, baseline_probability=baseline, evidence=evidence)
            llm_telemetry["attempted"] += int(ai_result.attempted)
            llm_telemetry["success"] += int(ai_result.success)
            llm_telemetry["cache_hits"] += int(ai_result.cache_hit)
            llm_telemetry["fallback"] += int(ai_result.status != "LLM_ACTIVE")
            if ai_result.success:
                llm_telemetry["last_success"] = now.isoformat()
            if ai_result.error_category in {"SCHEMA_VALIDATION_ERROR", "EVIDENCE_REFERENCE_ERROR"}:
                llm_telemetry["validation_failures"] += 1
            elif ai_result.error_category not in {None, "NOT_CONFIGURED", "MINIMUM_EVIDENCE_GATE"}:
                llm_telemetry["request_failures"] += 1
            if ai_result.error_category == "MINIMUM_EVIDENCE_GATE":
                llm_telemetry["evidence_gate_fallback"] += 1
            if ai_result.error_category not in {None, "NOT_CONFIGURED", "MINIMUM_EVIDENCE_GATE"}:
                llm_telemetry["last_error_category"] = ai_result.error_category
            if ai_result.status == "LLM_ACTIVE":
                llm_telemetry["adjustment_sum"] += ai_result.adjustment
                key = "positive_adjustments" if ai_result.adjustment > 0 else (
                    "negative_adjustments" if ai_result.adjustment < 0 else "zero_adjustments")
                llm_telemetry[key] += 1
            row.update({
                "quant_baseline_probability": baseline,
                "quant_model_version": str(model.trained_through),
                "ai_prompt_version": PROMPT_VERSION,
                "ai_model_name": ai_result.model,
                "llm_provider": ai_result.provider,
                "ai_status": ai_result.status,
                "ai_adjustment": ai_result.adjustment,
                "ai_adjustment_status": ai_result.adjustment_status,
                "ai_confidence": ai_result.analysis.confidence if ai_result.analysis else 0.0,
                "ai_used_evidence_ids": ai_result.analysis.used_evidence_ids if ai_result.analysis else [],
                "ai_risk_flags": ai_result.analysis.risk_flags if ai_result.analysis else [],
                "ai_unknowns": ai_result.analysis.unknowns if ai_result.analysis else [],
                "ai_cache_hit": ai_result.cache_hit,
                "final_probability": ai_result.final_probability,
                "ai_analysis": ai_result.analysis.as_dict() if ai_result.analysis else None,
                "evidence": [item.as_dict() for item in evidence],
                "evidence_count": len(evidence),
                "evidence_ids": [item.evidence_id for item in evidence],
                "evidence_types": [item.evidence_type for item in evidence],
                "evidence_hash": evidence_hash(evidence),
                "semantic_evidence_hash": ai_result.semantic_evidence_hash,
                "ai_error_category": ai_result.error_category,
                "evidence_gate_eligible": evidence_gate["eligible"],
                "evidence_gate_required": evidence_gate["required_types"],
                "evidence_gate_alternatives": evidence_gate["alternative_types"],
                "evidence_gate_blockers": evidence_gate["missing"],
                **availability,
            })
            group = f"{sport}:{report_day.isoformat()}"
            quant_candidate = copy.deepcopy(row)
            quant_cap = _recompute_post_ai_decision(
                quant_candidate, final_probability=baseline, bankroll=bankroll,
                config=risk_config, ledger=ledger, group_key=group)
            quant_candidate["final_probability"] = ai_result.final_probability
            quant_candidate["execution_decision_probability"] = baseline
            _apply_post_ai_quality_gates(quant_candidate, sport, row_model_status)
            _attach_risk_audit(quant_candidate, bankroll, risk_config)

            final_candidate = copy.deepcopy(row)
            final_cap = _recompute_post_ai_decision(
                final_candidate, final_probability=ai_result.final_probability, bankroll=bankroll,
                config=risk_config, ledger=ledger, group_key=group)
            final_candidate["execution_decision_probability"] = ai_result.final_probability
            _apply_post_ai_quality_gates(final_candidate, sport, row_model_status)
            _attach_risk_audit(final_candidate, bankroll, risk_config)

            execution = _select_llm_execution(
                quant_candidate, final_candidate, shadow_mode=llm_shadow_mode, baseline=baseline)
            row.clear(); row.update(execution)
            cap = quant_cap if llm_shadow_mode else final_cap
            if row.get("action") == "BET":
                ledger.commit(str(row.get("event_id") or match_id), group,
                              float(row.get("stake_fraction") or 0))
            _attach_bet_fields(
                row, model_probability=(baseline if llm_shadow_mode else ai_result.final_probability),
                execution_price=row.get("execution_price"), bankroll=bankroll,
                cap=cap, paper_db=paper_db)
            row["narrative_summary"] = build_pre_match_summary(row)
            snapshot = PredictionSnapshot(
                match_id=match_id, prediction_time=now.isoformat(),
                quant_model_version=str(model.trained_through), ai_prompt_version=PROMPT_VERSION,
                ai_model_name=row["ai_model_name"], baseline_probability=baseline,
                ai_adjustment=ai_result.adjustment, final_probability=ai_result.final_probability,
                market_fair_probability=row.get("market_fair_probability"),
                edge=row.get("edge"),
                features_json={"sport": sport, "recent_form_a": row.get("recent_form_a"),
                               "recent_form_b": row.get("recent_form_b"),
                               "lineup_status": row.get("lineup_status"),
                               "data_quality_level": row.get("data_quality_level"),
                               "data_quality_missing": row.get("data_quality_missing")},
                evidence_json=row["evidence"],
                risk_flags_json=(ai_result.analysis.risk_flags if ai_result.analysis else [ai_result.error or "LLM unavailable"]),
                recommendation=str(row.get("action") or "NO_BET"), position_size=float(row.get("stake") or 0),
                llm_provider=ai_result.provider,
                data_quality_score=float(row.get("data_quality_score") or 0),
                missing_evidence=tuple(row.get("data_quality_missing") or ()),
                evidence_ids=tuple(row.get("evidence_ids") or ()), evidence_hash=row.get("evidence_hash") or "",
                ai_confidence=float(row.get("ai_confidence") or 0), market_price=row.get("execution_price"),
                adjusted_edge=row.get("adjusted_edge"),
                expected_profit_per_share=row.get("expected_profit_per_share"),
                expected_roi_on_capital=row.get("expected_roi_on_capital"),
                execution_price=row.get("execution_price"),
                decimal_odds=(row.get("risk_calculation") or {}).get("decimal_odds"),
                raw_model_edge=row.get("raw_model_edge"), executable_edge=row.get("executable_edge"),
                risk_inputs={"bankroll": bankroll, "config": asdict(risk_config)},
                risk_output=row.get("risk_calculation"), created_at=now.isoformat(),
                ai_status=ai_result.status,
                semantic_evidence_hash=row.get("semantic_evidence_hash") or "",
                cache_hit=bool(ai_result.cache_hit),
                canonical_sample_key=str(row.get("canonical_sample_key") or match_id) or None,
                final_action=str(row.get("action") or "NO_BET"),
                final_stake=float(row.get("stake") or 0),
                quant_action=str(row.get("quant_action") or "NO_BET"),
                quant_stake=float(row.get("quant_stake") or 0), quant_ev=row.get("quant_ev"),
                shadow_final_action=str(row.get("shadow_final_action") or "NO_BET"),
                shadow_final_stake=float(row.get("shadow_final_stake") or 0),
                shadow_final_ev=row.get("shadow_final_ev"),
                llm_decision_mode=str(row.get("llm_decision_mode") or "SHADOW"),
            )
            row["prediction_snapshot_id"] = feedback_store.save_snapshot(snapshot)
        recommendations.extend(sport_rows)
        statuses[sport] = {
            "ready": True, "artifact_ready": True,
            "trained_through": model.trained_through, "samples": model.samples,
            "probability_approved": bool(evaluation.get("approved_for_probability_use")),
            "real_money_approved": bool(evaluation.get("approved_for_real_money")),
            "model_status": row_model_status,
            "days_since_training": model_age,
            "today_scheduled_matches": len(audit_matches[sport]),
            "today_markets": len(sport_rows),
            "today_prestart_markets": sum(not row["market_started"] for row in sport_rows),
            "today_probability_eligible": sum(row["probability_eligible"] for row in sport_rows),
            "today_bet_candidates": sum(row["action"] == "BET" for row in sport_rows),
            "model_team_count": len(getattr(model, "team_ratings", {}) or {}),
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
    for sport, age in staleness.items():
        if sport in statuses:
            statuses[sport]["model_status"] = "STALE" if age > retrain_interval else "OK"
            statuses[sport]["days_since_training"] = age
    recent_form_health = load_recent_form()
    cs2_form = recent_form_health.get("cs2") or {}
    cs2_health = recent_form_artifact_health()
    cs2_model_teams = int(statuses.get("cs2", {}).get("model_team_count") or 0)
    roster_health = lol_roster_health()
    data_health = {
        "lol_recent_form": "OK" if recent_form_health.get("lol") else "ERROR",
        "cs2_recent_form": cs2_health["artifact_status"],
        "cs2_recent_form_detail": {
            "status": cs2_health["artifact_status"], "artifact_status": cs2_health["artifact_status"],
            "refresh_status": cs2_health["refresh_status"], "team_count": len(cs2_form),
            "record_count": cs2_health["record_count"], "coverage": cs2_health["coverage"],
            "last_attempt": cs2_health["last_attempt"], "last_success": cs2_health["last_success"],
            "last_error_category": cs2_health["last_error_category"], "provider": cs2_health["provider"],
            "latest_match_at": max((str(value.get("latest_match_at")) for value in cs2_form.values()
                                    if value.get("latest_match_at")), default=None),
            "coverage_ratio": (len(cs2_form) / cs2_model_teams) if cs2_model_teams else 0.0,
            "artifact_generated_at": cs2_health["generated_at"],
            "artifact_age_seconds": cs2_health["age_seconds"],
            "alert": None if cs2_form else "DATA_SOURCE_FAILURE",
        },
        "lol_roster": roster_health["status"],
        "lol_roster_detail": roster_health,
    }
    flagged = flag_rows(recommendations, flag_threshold)
    if ledger.breaker_reason:
        for row in recommendations:
            row["flagged"] = True
        flagged = list(recommendations)
    for row in recommendations:
        if row.get("virtual_bet"):
            record_virtual_bet(paper_db, {
                "sport": row.get("sport"),
                "event_id": row.get("event_id"),
                "event": row.get("event"),
                "generated_at": row.get("generated_at"),
                "bet_side": row.get("outcome"),
                "model_prob": row.get("execution_decision_probability") or row.get("model_probability"),
                "market_odds": (1.0 / float(row["execution_price"])
                                if row.get("execution_price") else 0.0),
                "stake_virtual": row.get("stake_virtual") or 0.0,
            })
    virtual_betting = {
        "count": count_settled_virtual_bets(paper_db),
        "recorded_count": count_virtual_bets(paper_db),
        "roi": calc_roi(paper_db, bet_type="virtual"),
        "balance": virtual_account_balance(paper_db, bankroll),
    }
    paper_daily = _paper_daily_report(
        recommendations, bankroll, ledger.daily_committed, risk_config.max_daily_risk_fraction,
    )
    paper_mode = "已开启" if _paper_trading_enabled() else "未开启"
    all_schedule_matches = [match for rows in audit_matches.values() for match in rows]
    coverage = build_coverage_report(
        report_day.isoformat(), all_schedule_matches, recommendations,
        {sport: bool(status.get("ready")) for sport, status in statuses.items()},
    )
    CoverageStore(os.getenv("COVERAGE_DB_PATH", paper_db)).save(coverage)
    lifecycle_store = MatchLifecycleStore(os.getenv("LIFECYCLE_DB_PATH", paper_db))
    lifecycle_rows = lifecycle_store.observe(all_schedule_matches, recommendations, now)
    risk_notes = [
        "NBA、LoL、CS2 分别训练和验收；CBA 已暂停。",
        f"虚拟投注：{paper_mode}；仅写入 paper.db，不涉及真实资金。",
        "系统仅输出研究建议，不执行任何真实资金交易。",
    ]
    if ledger.breaker_reason:
        risk_notes.append(ledger.breaker_reason)
    if ledger.warn_reason:
        risk_notes.append(ledger.warn_reason)
    llm_telemetry["average_adjustment"] = (
        llm_telemetry["adjustment_sum"] / llm_telemetry["success"]
        if llm_telemetry["success"] else 0.0)
    llm_telemetry["mode"] = "SHADOW" if llm_shadow_mode else "ACTIVE"
    system_health = build_system_health(
        audits=audits, statuses=statuses, data_health=data_health,
        llm_provider=getattr(llm_client, "provider", None), llm_configured=llm_client is not None,
        llm_telemetry=llm_telemetry,
    )
    provider_health = {}
    for sport, source_rows in {"lol": lol_sources, "nba": nba_sources, "cs2": cs2_sources}.items():
        fallback_used = any(item.available for item in source_rows) and any(not item.available for item in source_rows)
        for item in source_rows:
            http = re.search(r"HTTP(?: Error)?\s+(\d{3})", item.error or "", re.IGNORECASE)
            attempted = item.last_attempt_at or now
            for capability in PROVIDER_CAPABILITIES.get(item.name, ("SCHEDULE",)):
                health = ProviderHealth(item.name, sport, capability,
                    "ACTIVE" if item.available else ("CREDENTIAL_REQUIRED" if "not configured" in (item.error or "") else "ERROR"),
                    attempted, item.last_success_at or (now if item.available else None),
                    int(http.group(1)) if http else None, item.latency_ms, len(item.matches),
                    None if item.available else type(item.error).__name__, item.error,
                    0 if item.available else None)
                row = health.as_dict(); row["fallback_used"] = bool(fallback_used and not item.available)
                provider_health[f"{sport}:{item.name}:{capability}"] = row
    provider_store = ProviderHealthStore(os.getenv("PROVIDER_HEALTH_DB_PATH", paper_db))
    provider_store.record(list(provider_health.values()), now)
    report = {
        "report_date": report_day.isoformat(), "generated_at": now.isoformat(),
        "bankroll_usdc": bankroll, "recommendations": recommendations, "sport_status": statuses,
        "llm_attribution": feedback_store.shadow_attribution_summary(),
        "schedule_coverage": audits,
        "coverage_report": coverage.as_dict(),
        "match_lifecycle": lifecycle_rows,
        "prematch_scan_missed": sum(int(row["prematch_scan_missed"]) for row in lifecycle_rows),
        "today_scheduled_matches": sum(len(rows) for rows in audit_matches.values()),
        "data_incomplete": any(audit["data_incomplete"] for audit in audits.values()),
        "flagged": flagged,
        "flag_threshold": flag_threshold,
        "paper_daily": paper_daily,
        "virtual_betting": virtual_betting,
        "risk_notes": risk_notes,
        "model_staleness": model_staleness,
        "data_health": data_health,
        "provider_health": provider_health,
        "provider_health_daily": provider_store.daily_summary(report_day.isoformat()),
        "prematch_scan_daily": lifecycle_store.daily_metrics(report_day.isoformat()),
        "system_health": system_health,
        "risk_status": {
            "bankroll_usdc": bankroll,
            "max_bet_fraction": risk_config.max_bet_fraction,
            "max_daily_risk_fraction": risk_config.max_daily_risk_fraction,
            "max_event_risk_fraction": risk_config.max_event_risk_fraction,
            "max_drawdown_fraction": risk_config.max_drawdown_fraction,
            "drawdown_warn_fraction": risk_config.drawdown_warn_fraction,
            "drawdown_circuit_fraction": risk_config.drawdown_circuit_fraction,
            "drawdown_level": ledger.drawdown_level,
            "paper_mode_exception": bool(ledger.paper_mode and ledger.drawdown_level == "warn"
                                         and drawdown >= risk_config.drawdown_circuit_fraction),
            "warn_reason": ledger.warn_reason,
            "daily_committed_fraction": ledger.daily_committed,
            "current_drawdown": drawdown,
            "virtual_betting": virtual_betting,
            "circuit_breaker": ledger.breaker_reason is not None,
            "circuit_breaker_reason": ledger.breaker_reason,
        },
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
