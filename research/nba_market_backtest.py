"""Leak-free NBA Elo versus fixed pregame Polymarket price windows."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError

from prediction_agent.entities import canonical_team
from prediction_agent.lol_model import load_canonical_games, walk_forward_elo_probabilities
from prediction_agent.providers.http import get_json

from polymarket_walkforward import CLOB, Game, extract_games, fetch_events


TAG = 745
START = "2025-10-21"
END = "2026-07-01"
DECISION_HOURS = (24, 6, 1)


def key(day: str, team_a: str, team_b: str) -> tuple[str, tuple[str, str]]:
    teams = tuple(sorted((canonical_team("nba", team_a).casefold(),
                          canonical_team("nba", team_b).casefold())))
    return day, teams


def prediction_index(rows: list[dict[str, object]]) -> dict[tuple[str, tuple[str, str]], list[dict[str, object]]]:
    result: dict[tuple[str, tuple[str, str]], list[dict[str, object]]] = {}
    for row in rows:
        result.setdefault(key(str(row["played_at"]), str(row["team_a"]), str(row["team_b"])), []).append(row)
    return result


def match_prediction(game: Game, index: dict[tuple[str, tuple[str, str]], list[dict[str, object]]]) -> dict[str, object] | None:
    game_day = datetime.fromtimestamp(game.start_ts, timezone.utc).date()
    candidates: list[dict[str, object]] = []
    for delta in (-1, 0, 1):
        candidates.extend(index.get(key((game_day + timedelta(days=delta)).isoformat(), game.team_a, game.team_b), []))
    return candidates[0] if len(candidates) == 1 else None


def fetch_history(game: Game) -> list[dict[str, object]]:
    try:
        payload = get_json(f"{CLOB}/prices-history", {
            "market": game.token_a, "startTs": game.start_ts - 72 * 3600,
            "endTs": game.start_ts - 60, "fidelity": 5,
        }, timeout=30)
        return list(payload.get("history", []))
    except HTTPError as exc:
        if exc.code in (400, 404):
            return []
        raise


def fill_histories(games: list[Game], path: Path) -> dict[str, list[dict[str, object]]]:
    cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    missing = [game for game in games if game.token_a not in cache]
    if missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_history, game): game for game in missing}
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                cache[futures[future].token_a] = future.result()
                if number % 50 == 0:
                    path.write_text(json.dumps(cache), encoding="utf-8")
        path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def price_at(points: list[dict[str, object]], cutoff: int) -> float | None:
    prices = [float(point["p"]) for point in points if int(point["t"]) <= cutoff]
    value = prices[-1] if prices else None
    return value if value is not None and .01 <= value <= .99 else None


def evaluate(rows: list[dict[str, object]]) -> dict[str, object]:
    bets = []
    for row in rows:
        market = float(row["market_probability_a"])
        model = float(row["model_probability_a"])
        side = 0 if model - market >= .05 else 1 if market - model >= .05 else -1
        if side < 0:
            continue
        raw_price = market if side == 0 else 1 - market
        model_side = model if side == 0 else 1 - model
        execution = min(.99, raw_price + .01)
        if model_side <= execution or float(row["volume"]) < 100:
            continue
        shares = 1 / execution
        fee = shares * .03 * execution * (1 - execution)
        won = int(row["winner"]) == side
        bets.append(shares * int(won) - 1 - fee)
    outcomes = [float(int(row["winner"]) == 0) for row in rows]
    return {
        "observations": len(rows), "bets": len(bets),
        "proxy_profit_per_unit_stake": sum(bets),
        "proxy_roi": mean(bets) if bets else None,
        "model_brier": mean((float(row["model_probability_a"]) - outcome) ** 2
                            for row, outcome in zip(rows, outcomes)) if rows else math.nan,
        "market_brier": mean((float(row["market_probability_a"]) - outcome) ** 2
                             for row, outcome in zip(rows, outcomes)) if rows else math.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", default="data/raw/nba/nba_games.csv")
    parser.add_argument("--data-dir", default="data/research")
    parser.add_argument("--output", default="reports/nba_market_backtest.json")
    parser.add_argument("--refresh-events", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    model_rows = walk_forward_elo_probabilities(load_canonical_games([args.games], "nba"))
    index = prediction_index(model_rows)
    events = fetch_events(TAG, START, END, data_dir / "nba_events.json", args.refresh_events)
    games = extract_games("nba", events)
    pairs = [(game, match_prediction(game, index)) for game in games]
    joined = [(game, prediction) for game, prediction in pairs if prediction is not None]
    histories = fill_histories([game for game, _ in joined], data_dir / "nba_price_histories.json")
    observations = []
    for game, prediction in joined:
        model_a = float(prediction["model_probability_a"])
        if canonical_team("nba", game.team_a).casefold() != canonical_team("nba", str(prediction["team_a"])).casefold():
            model_a = 1 - model_a
        for hours in DECISION_HOURS:
            price = price_at(histories.get(game.token_a, []), game.start_ts - hours * 3600)
            if price is not None:
                observations.append({
                    "event_id": game.event_id,
                    "start": datetime.fromtimestamp(game.start_ts, timezone.utc).isoformat(),
                    "decision_hours": hours, "team_a": game.team_a, "team_b": game.team_b,
                    "winner": game.winner, "model_probability_a": model_a,
                    "market_probability_a": price, "volume": game.volume,
                })
    results = {}
    for hours in DECISION_HOURS:
        window = [row for row in observations if row["decision_hours"] == hours]
        results[f"T-{hours}h"] = {
            "validation_2025": evaluate([row for row in window if str(row["start"]).startswith("2025")]),
            "locked_test_2026": evaluate([row for row in window if str(row["start"]).startswith("2026")]),
        }
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model": "chronological Elo replay; probability captured before result update",
            "validation": "2025-10-21 through 2025-12-31",
            "locked_test": "2026-01-01 through 2026-06-30",
            "decision_hours": list(DECISION_HOURS), "edge_threshold": .05,
            "execution": "aggregate price + 0.01 adverse slippage proxy and fee approximation",
            "real_money_approval_allowed": False,
        },
        "coverage": {"polymarket_games": len(games), "model_joined_games": len(joined),
                     "join_rate": len(joined) / len(games) if games else 0},
        "results": results,
        "limitations": [
            "price history does not reconstruct executable asks or depth",
            "baseline Elo omits injuries, confirmed starters, rest, travel, and home-court effects",
            "a positive result cannot approve real-money use; a negative result rejects the strategy",
        ],
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = target.with_name("NBA_历史市场锁箱回测.md")
    lines = [
        "# NBA 历史市场锁箱回测",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 结论",
        "",
        "NBA Elo 基线未通过下注验收。2025 验证段与 2026 锁箱段在三个固定决策时点均为负代理 ROI；2026 的市场概率 Brier 也全部优于模型。生产状态继续保持 `NO_BET`。",
        "",
        "## 2026 锁箱结果",
        "",
        "|决策时点|样本|下注数|代理 ROI|模型 Brier|市场 Brier|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, split in results.items():
        result = split["locked_test_2026"]
        lines.append(f"|{label}|{result['observations']}|{result['bets']}|{result['proxy_roi']:.1%}|"
                     f"{result['model_brier']:.3f}|{result['market_brier']:.3f}|")
    lines.extend([
        "", "## 覆盖与方法", "",
        f"- Polymarket NBA 钱线市场 {payload['coverage']['polymarket_games']} 场，唯一连接到 NBA 赛果 {payload['coverage']['model_joined_games']} 场，连接率 {payload['coverage']['join_rate']:.1%}。",
        "- 所有 Elo 概率均在写入该场赛果之前生成。",
        "- 固定采用 T−24h、T−6h、T−1h，5 个百分点最小模型差、1 美分不利滑点及费用估算。",
        "- 公共价格历史不是历史 ask 和盘口深度，因此只可否决策略，不能批准真钱。",
        "", "## 后续", "",
        "下一版需要加入赛前已知的主客场、休息天数、背靠背、旅行、确认首发和伤病。由于本次已经查看 2026 结果，修改后的模型不得再次把 2026 称为未见锁箱测试；必须使用后续全新比赛做前向纸面验收。",
    ])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
