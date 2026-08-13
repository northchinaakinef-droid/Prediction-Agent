from __future__ import annotations

import json
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
from .risk import recommend
from .entities import canonical_team


TAGS = {"nba": "745", "lol": "65", "cs2": "100780"}


def _in_horizon(scheduled: datetime | None, now: datetime) -> bool:
    hours = float(os.getenv("MARKET_HORIZON_HOURS", "30"))
    return scheduled is not None and now < scheduled <= now + timedelta(hours=hours)


def _timing(scheduled: datetime, now: datetime) -> tuple[float, str]:
    hours = (scheduled - now).total_seconds() / 3600
    target = min((1, 6, 24), key=lambda value: abs(value - hours))
    return hours, f"T-{target}h"


def _field(value):
    return json.loads(value) if isinstance(value, str) else list(value or [])


def _text(value: object) -> str:
    return "".join(character for character in str(value)
                   if unicodedata.category(character) not in {"Cc", "Cf"}).strip()


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
                  now: datetime, bankroll: float, estimated_cost: float = .01) -> list[dict]:
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
        first_bid, first_ask = market.get("bestBid"), market.get("bestAsk")
        ask = float(first_ask) if side == 0 and first_ask is not None else (
            1 - float(first_bid) if side == 1 and first_bid is not None else prices[side])
        known = model.games.get(team_a, 0) >= 10 and model.games.get(team_b, 0) >= 10
        probability_ok = bool(evaluation.get("approved_for_probability_use")) and known
        money_ok = bool(evaluation.get("approved_for_real_money"))
        started = scheduled is None or scheduled <= now
        rec = recommend(
            event_id=str(event.get("id")), outcome=outcomes[side], model_probability=model_ps[side],
            decimal_odds=1 / ask, bankroll=bankroll, confidence=.75 if probability_ok else .25,
            spread=float(market["spread"]) if market.get("spread") is not None else None,
            available_size=float(market.get("liquidity") or 0), estimated_cost=estimated_cost,
            trading_enabled=money_ok and not started,
        )
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
        row = asdict(rec)
        row["generated_at"] = rec.generated_at.isoformat()
        hours_to_start, decision_window = _timing(scheduled, now)
        row.update({
            "sport": sport, "event": _text(event.get("title") or ""),
            "scheduled_start": scheduled.isoformat() if scheduled else None,
            "market_probability": prices[side], "execution_price": ask,
            "edge": rec.decision_probability - ask - estimated_cost,
            "probability_eligible": probability_ok,
            "real_money_approved": money_ok,
            "market_started": started,
            "market_comparison_valid": not started,
            "hours_to_start": hours_to_start,
            "decision_window": decision_window,
            "reasons": reasons + list(rec.reasons),
        })
        if started:
            row.update({"decision_probability": None, "raw_edge": None, "edge": None,
                        "expected_value": None, "execution_price": None})
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
                  reasons: list[str], estimated_cost: float = .01) -> dict:
    side = max(range(2), key=lambda index: probabilities[index] - prices[index])
    bid, best_ask = market.get("bestBid"), market.get("bestAsk")
    ask = float(best_ask) if side == 0 and best_ask is not None else (
        1 - float(bid) if side == 1 and bid is not None else prices[side])
    started = scheduled is None or scheduled <= now
    rec = recommend(
        event_id=str(event.get("id")), outcome=outcomes[side], model_probability=probabilities[side],
        decimal_odds=1 / ask, bankroll=bankroll, confidence=.75 if probability_ok else .25,
        spread=float(market["spread"]) if market.get("spread") is not None else None,
        available_size=float(market.get("liquidity") or 0), estimated_cost=estimated_cost,
        trading_enabled=money_ok and probability_ok and not started,
    )
    if not probability_ok:
        reasons.append("阵容未知、阵容过期、样本不足或概率模型未通过验收，因此只展示研究值。")
    if not money_ok:
        reasons.append("尚未通过带历史可成交赔率的样本外 ROI 验收，禁止真钱下注建议。")
    if started:
        reasons.append("比赛已开始或开赛时间无法核验，禁止赛前下注。")
    row = asdict(rec)
    row["generated_at"] = rec.generated_at.isoformat()
    assert scheduled is not None
    hours_to_start, decision_window = _timing(scheduled, now)
    row.update({
        "sport": sport, "event": _text(event.get("title") or ""),
        "scheduled_start": scheduled.isoformat() if scheduled else None,
        "market_probability": prices[side], "execution_price": ask,
        "edge": rec.decision_probability - ask - estimated_cost,
        "probability_eligible": probability_ok,
        "real_money_approved": money_ok,
        "market_started": started,
        "market_comparison_valid": not started,
        "hours_to_start": hours_to_start,
        "decision_window": decision_window,
        "reasons": reasons + list(rec.reasons),
    })
    if started:
        row.update({"decision_probability": None, "raw_edge": None, "edge": None,
                    "expected_value": None, "execution_price": None})
    return row


def analyze_cs2(model: Cs2Model, evaluation: dict, events: list[dict], *,
                now: datetime, bankroll: float) -> list[dict]:
    rows = []
    for event, market, scheduled, outcomes, prices in _market_rows(events, now):
        a, b = canonical_team("cs2", outcomes[0]), canonical_team("cs2", outcomes[1])
        roster_a = tuple(model.latest_team_rosters.get(a, ()))
        roster_b = tuple(model.latest_team_rosters.get(b, ()))
        probability = model.probability(a, b, roster_a, roster_b)
        known = model.team_games.get(a, 0) >= 10 and model.team_games.get(b, 0) >= 10
        roster_ok = len(roster_a) == len(roster_b) == 5 and _roster_fresh(model.team_last_game, (a, b), now, 60)
        probability_ok = bool(evaluation.get("approved_for_probability_use")) and known and roster_ok
        reasons = [
            f"阵容感知胜率：{outcomes[0]} {probability:.1%}，{outcomes[1]} {1-probability:.1%}。",
            f"历史样本：{a} {model.team_games.get(a, 0)} 场，{b} {model.team_games.get(b, 0)} 场。",
            "当前基线包含战队和五人阵容强度；地图池、veto 与 LAN/线上层仍在补充。",
        ]
        rows.append(_research_row("cs2", event, market, scheduled, outcomes, prices,
                                  [probability, 1-probability], probability_ok=probability_ok,
                                  money_ok=bool(evaluation.get("approved_for_real_money")),
                                  now=now, bankroll=bankroll, reasons=reasons))
    return rows


def analyze_lol_meta(model: LolMetaModel, evaluation: dict, events: list[dict], *,
                     now: datetime, bankroll: float) -> list[dict]:
    rows = []
    for event, market, scheduled, outcomes, prices in _market_rows(events, now):
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
        probability_ok = bool(evaluation.get("approved_for_probability_use")) and known and roster_ok
        reasons = [
            f"赛前阵容模型：{outcomes[0]} {probability:.1%}，{outcomes[1]} {1-probability:.1%}（BO{best_of}）。",
            f"历史样本：{a} {model.team_games.get(a, 0)} 局，{b} {model.team_games.get(b, 0)} 局。",
            "BP 未开始时不使用英雄选择；BP 完成后必须重新计算版本英雄强度与选手英雄熟练度。",
        ]
        rows.append(_research_row("lol", event, market, scheduled, outcomes, prices,
                                  [probability, 1-probability], probability_ok=probability_ok,
                                  money_ok=bool(evaluation.get("approved_for_real_money")),
                                  now=now, bankroll=bankroll, reasons=reasons))
    return rows


def run_all(model_dir: str | Path, output: str | Path, *, now: datetime | None = None) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    bankroll = float(os.getenv("BANKROLL_USDC", "1000"))
    recommendations, statuses = [], {}
    client = PolymarketClient(timeout=30)
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
        events = client.events_by_tag(tag, limit=100)
        if sport == "cs2":
            sport_rows = analyze_cs2(model, evaluation, events, now=now, bankroll=bankroll)
        elif sport == "lol":
            sport_rows = analyze_lol_meta(model, evaluation, events, now=now, bankroll=bankroll)
        else:
            sport_rows = analyze_sport(sport, model, evaluation, events, now=now, bankroll=bankroll)
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
    zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Shanghai"))
    report = {
        "report_date": now.astimezone(zone).date().isoformat(), "generated_at": now.isoformat(),
        "bankroll_usdc": bankroll, "recommendations": recommendations, "sport_status": statuses,
        "risk_notes": ["NBA、LoL、CS2 分别训练和验收；CBA 已暂停。历史可成交赔率 ROI 验收前均保持 NO_BET。"],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
