from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .lol_model import EloModel, load_model, series_probability
from .providers.polymarket import PolymarketClient
from .risk import recommend


TITLE = re.compile(r"^LoL: (.+?) vs (.+?) \(BO(\d)\)")
SCHEDULE = re.compile(r"scheduled for ([A-Z][a-z]+ \d{1,2}) at (\d{1,2}:\d{2})(AM|PM) ET", re.I)


def _json_field(value):
    return json.loads(value) if isinstance(value, str) else value


def _scheduled_at(event: dict, year: int) -> datetime | None:
    match = SCHEDULE.search(str(event.get("description") or ""))
    if not match:
        return None
    parsed = datetime.strptime(f"{match.group(1)} {year} {match.group(2)}{match.group(3).upper()}",
                               "%B %d %Y %I:%M%p")
    return parsed.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(timezone.utc)


def build_lol_report(model: EloModel, evaluation: dict, events: list[dict], *,
                     now: datetime | None = None, bankroll: float = 1000.0,
                     estimated_cost: float = 0.01) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    report_zone = ZoneInfo(os.getenv("REPORT_TIMEZONE", "Asia/Shanghai"))
    report_day = now.astimezone(report_zone).date()
    rows = []
    probability_approved = bool(evaluation.get("approved_for_probability_use"))
    money_approved = bool(evaluation.get("approved_for_real_money"))
    for event in events:
        title = str(event.get("title") or "")
        parsed = TITLE.match(title)
        if not parsed:
            continue
        team_a, team_b, best_of_text = parsed.groups()
        scheduled = _scheduled_at(event, now.year)
        if scheduled and scheduled.astimezone(report_zone).date() != report_day:
            continue
        main_market = next((m for m in event.get("markets", []) if m.get("question") == title), None)
        if not main_market:
            continue
        outcomes = list(_json_field(main_market.get("outcomes") or []))
        prices = [float(x) for x in _json_field(main_market.get("outcomePrices") or [])]
        if len(outcomes) != 2 or len(prices) != 2 or min(prices) <= 0:
            continue
        game_p = model.game_probability(team_a, team_b)
        model_a = series_probability(game_p, int(best_of_text))
        probabilities = [model_a, 1 - model_a]
        best_index = max(range(2), key=lambda i: probabilities[i] - prices[i])
        first_bid = main_market.get("bestBid")
        first_ask = main_market.get("bestAsk")
        ask = float(first_ask) if best_index == 0 and first_ask is not None else (
            1 - float(first_bid) if best_index == 1 and first_bid is not None else prices[best_index])
        spread = float(main_market["spread"]) if main_market.get("spread") is not None else None
        liquidity = float(main_market.get("liquidity") or 0)
        started = scheduled is None or scheduled <= now
        known_teams = model.games.get(team_a, 0) >= 10 and model.games.get(team_b, 0) >= 10
        rec = recommend(
            event_id=str(event.get("id")), outcome=str(outcomes[best_index]),
            model_probability=probabilities[best_index], decimal_odds=1 / ask,
            bankroll=bankroll, confidence=0.75 if probability_approved and known_teams else 0.25,
            spread=spread, available_size=liquidity, estimated_cost=estimated_cost,
            trading_enabled=money_approved and not started,
        )
        reasons = model.explain(team_a, team_b)
        reasons.extend([
            f"由单局胜率换算 BO{best_of_text}：{team_a} {model_a:.1%}，{team_b} {1-model_a:.1%}",
            "市场仅用于比较价格，没有作为独立胜率模型的输入",
        ])
        if not known_teams:
            reasons.append("至少一支队伍历史样本不足 10 局，降低置信度并禁止下注")
        if started:
            reasons.append("比赛已开始或无法核验开赛时间，禁止赛前下注")
        if not money_approved:
            reasons.append("尚未通过带历史可成交赔率的锁箱 ROI 验收，保持 NO BET")
        row = asdict(rec)
        row["generated_at"] = rec.generated_at.isoformat()
        row.update({
            "sport": "lol", "event": title, "scheduled_start": scheduled.isoformat() if scheduled else None,
            "market_probability": prices[best_index], "execution_price": ask,
            "edge": rec.decision_probability - ask - estimated_cost,
            "reasons": reasons + list(rec.reasons),
        })
        rows.append(row)
    return {
        "report_date": report_day.isoformat(), "generated_at": now.isoformat(),
        "bankroll_usdc": bankroll, "recommendations": rows,
        "model_status": {
            "independent_probability": True, "probability_approved": probability_approved,
            "real_money_approved": money_approved, "trained_through": model.trained_through,
            "samples": model.samples,
        },
        "risk_notes": [
            "没有通过可成交赔率锁箱 ROI 验收前，系统不会给出真实下注金额",
            "临场阵容、首发或突发换人缺失时应降低置信度或 NO BET",
        ],
    }


def run_daily(model_path: str | Path, output: str | Path, *, dry_run: bool = False) -> dict:
    model, evaluation = load_model(model_path)
    events = PolymarketClient().lol_events(limit=100)
    report = build_lol_report(model, evaluation, events, bankroll=float(os.getenv("BANKROLL_USDC", "1000")))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
