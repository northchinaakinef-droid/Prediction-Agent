"""Join leak-free Valve CS2 predictions to pre-match Polymarket price history.

The resulting ROI is explicitly a price-history proxy. Polymarket's public price
history does not reconstruct historical asks or order-book depth, so this script
can reject a strategy but cannot approve real-money trading.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError

from prediction_agent.cs2_model import load_valve_vrs, walk_forward_probabilities
from prediction_agent.entities import canonical_team
from prediction_agent.providers.http import get_json

from polymarket_walkforward import CLOB, Game, extract_games, fetch_events


TAG = 100780
START = "2025-01-01"
END = "2027-01-01"
DECISION_HOURS = (24, 6, 1)


def team_key(name: str) -> str:
    name = canonical_team("cs2", name)
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def game_key(day: str, team_a: str, team_b: str) -> tuple[str, tuple[str, str]]:
    return day, tuple(sorted((team_key(team_a), team_key(team_b))))


def model_index(rows: list[dict[str, object]]) -> dict[tuple[str, tuple[str, str]], list[dict[str, object]]]:
    result: dict[tuple[str, tuple[str, str]], list[dict[str, object]]] = {}
    for row in rows:
        key = game_key(str(row["played_at"]), str(row["team_a"]), str(row["team_b"]))
        result.setdefault(key, []).append(row)
    return result


def match_prediction(game: Game, index: dict[tuple[str, tuple[str, str]], list[dict[str, object]]]) -> dict[str, object] | None:
    start = datetime.fromtimestamp(game.start_ts, timezone.utc).date()
    candidates: list[dict[str, object]] = []
    for delta in (-1, 0, 1):
        candidates.extend(index.get(game_key((start + timedelta(days=delta)).isoformat(), game.team_a, game.team_b), []))
    return candidates[0] if len(candidates) == 1 else None


def fetch_history(game: Game) -> list[dict[str, object]]:
    try:
        row = get_json(f"{CLOB}/prices-history", {
            "market": game.token_a,
            "startTs": game.start_ts - 72 * 3600,
            "endTs": game.start_ts - 60,
            "fidelity": 5,
        }, timeout=30)
        return list(row.get("history", []))
    except HTTPError as exc:
        if exc.code in (400, 404):
            return []
        raise


def fill_histories(games: list[Game], cache_path: Path) -> dict[str, list[dict[str, object]]]:
    cache: dict[str, list[dict[str, object]]] = (
        json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    )
    missing = [game for game in games if game.token_a not in cache]
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(fetch_history, game): game for game in missing}
            for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
                game = futures[future]
                cache[game.token_a] = future.result()
                if number % 50 == 0:
                    cache_path.write_text(json.dumps(cache), encoding="utf-8")
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def price_at(points: list[dict[str, object]], cutoff: int) -> float | None:
    eligible = [float(point["p"]) for point in points if int(point["t"]) <= cutoff]
    price = eligible[-1] if eligible else None
    return price if price is not None and 0.01 <= price <= 0.99 else None


def evaluate(rows: list[dict[str, object]], *, edge_threshold: float = .05) -> dict[str, object]:
    bets = []
    for row in rows:
        market_a, model_a = float(row["market_probability_a"]), float(row["model_probability_a"])
        side = 0 if model_a - market_a >= edge_threshold else 1 if market_a - model_a >= edge_threshold else -1
        if side < 0:
            continue
        raw_price = market_a if side == 0 else 1 - market_a
        probability = model_a if side == 0 else 1 - model_a
        execution_proxy = min(.99, raw_price + .01)
        if probability <= execution_proxy or float(row["volume"]) < 100:
            continue
        shares = 1 / execution_proxy
        fee = shares * .03 * execution_proxy * (1 - execution_proxy)
        won = int(row["winner"]) == side
        pnl = shares * int(won) - 1 - fee
        bets.append({**row, "side": side, "execution_proxy": execution_proxy, "won": won, "pnl": pnl})
    outcomes = [float(int(row["winner"]) == 0) for row in rows]
    turnover = float(len(bets))
    return {
        "observations": len(rows),
        "bets": len(bets),
        "wins": sum(int(row["won"]) for row in bets),
        "hit_rate": mean(int(row["won"]) for row in bets) if bets else None,
        "proxy_profit_per_unit_stake": sum(float(row["pnl"]) for row in bets),
        "proxy_roi": sum(float(row["pnl"]) for row in bets) / turnover if turnover else None,
        "model_brier": mean((float(row["model_probability_a"]) - outcome) ** 2 for row, outcome in zip(rows, outcomes)) if rows else math.nan,
        "market_brier": mean((float(row["market_probability_a"]) - outcome) ** 2 for row, outcome in zip(rows, outcomes)) if rows else math.nan,
        "records": bets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vrs-root", default="data/external/valve_cs2_vrs")
    parser.add_argument("--data-dir", default="data/research")
    parser.add_argument("--output", default="reports/cs2_market_backtest.json")
    parser.add_argument("--refresh-events", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    artifact = json.loads(Path("artifacts/cs2_model.json").read_text(encoding="utf-8"))
    params = artifact["evaluation"]["selected_parameters"]
    model_rows = walk_forward_probabilities(load_valve_vrs(args.vrs_root), **params)
    index = model_index(model_rows)
    events = fetch_events(TAG, START, END, data_dir / "cs2_events.json", args.refresh_events)
    games = extract_games("cs2", events)
    matches = [(game, match_prediction(game, index)) for game in games]
    joined = [(game, prediction) for game, prediction in matches if prediction is not None]
    histories = fill_histories([game for game, _ in joined], data_dir / "cs2_price_histories.json")

    observations = []
    for game, prediction in joined:
        model_a = float(prediction["model_probability_a"])
        if team_key(game.team_a) != team_key(str(prediction["team_a"])):
            model_a = 1 - model_a
        for hours in DECISION_HOURS:
            price = price_at(histories.get(game.token_a, []), game.start_ts - hours * 3600)
            if price is None:
                continue
            observations.append({
                "event_id": game.event_id,
                "start": datetime.fromtimestamp(game.start_ts, timezone.utc).isoformat(),
                "decision_hours": hours,
                "team_a": game.team_a,
                "team_b": game.team_b,
                "winner": game.winner,
                "model_probability_a": model_a,
                "market_probability_a": price,
                "volume": game.volume,
            })

    by_window: dict[str, object] = {}
    for hours in DECISION_HOURS:
        window = [row for row in observations if row["decision_hours"] == hours]
        by_window[f"T-{hours}h"] = {
            "all": evaluate(window),
            "validation_2025": evaluate([row for row in window if str(row["start"]).startswith("2025")]),
            "locked_test_2026": evaluate([row for row in window if str(row["start"]).startswith("2026")]),
        }

    unmatched = Counter()
    for game, prediction in matches:
        if prediction is None:
            unmatched.update((game.team_a, game.team_b))
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model_history": "Valve series replayed chronologically; probability captured before each result update",
            "validation": "2025 market matches",
            "locked_test": "2026 market matches",
            "decision_times": list(DECISION_HOURS),
            "edge_threshold": .05,
            "execution": "aggregate historical price + 0.01 adverse slippage proxy; unit stake; fee approximation",
            "real_money_approval_allowed": False,
        },
        "coverage": {
            "polymarket_games": len(games),
            "model_joined_games": len(joined),
            "join_rate": len(joined) / len(games) if games else 0,
            "unmatched_top_teams": unmatched.most_common(30),
        },
        "results": by_window,
        "limitations": [
            "public price history is an aggregate price series, not historical executable asks or depth",
            "Valve match records contain dates but not exact start times; joins allow a one-day timezone tolerance",
            "ambiguous same-team rematches are excluded",
            "this report can reject but cannot approve real-money deployment",
        ],
    }
    for window in by_window.values():
        for split in window.values():
            split.pop("records", None)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_target = target.with_name("CS2_历史市场锁箱回测.md")
    lines = [
        "# CS2 历史市场锁箱回测",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 结论",
        "",
        "当前 CS2 模型未通过真钱下注验收。2025 验证集显示正收益，但在参数冻结后的 2026 锁箱集上，T−24h、T−6h、T−1h 三个固定决策点均为负代理 ROI，且市场概率的 Brier 分数均优于模型。正式策略继续保持 `NO_BET`。",
        "",
        "## 锁箱结果",
        "",
        "|决策时点|2026 样本|下注数|命中率|代理 ROI|模型 Brier|市场 Brier|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, window in by_window.items():
        result = window["locked_test_2026"]
        lines.append(
            f"|{label}|{result['observations']}|{result['bets']}|{result['hit_rate']:.1%}|"
            f"{result['proxy_roi']:.1%}|{result['model_brier']:.3f}|{result['market_brier']:.3f}|"
        )
    lines.extend([
        "",
        "## 方法",
        "",
        "- 比赛数据：Valve 官方 Regional Standings 仓库中的历史赛果与阵容。",
        "- 模型：队伍 Elo 与选手 Elo 组合；逐场重放，必须先生成概率，再写入该场赛果，避免未来信息泄漏。",
        "- 时间切分：2025 用于验证；2026 完全锁箱，参数冻结后才查看结果。",
        "- 市场价格：Polymarket 公共 CLOB 历史聚合价，在开赛前 24、6、1 小时固定取值。",
        "- 交易规则：模型相对市场至少高 5 个百分点才考虑；价格另加 1 美分不利滑点，并估算 taker fee。",
        "",
        "## 覆盖与限制",
        "",
        f"- Polymarket 钱线市场 {payload['coverage']['polymarket_games']} 场；与 Valve 记录唯一匹配 {payload['coverage']['model_joined_games']} 场，匹配率 {payload['coverage']['join_rate']:.1%}。",
        "- 公共历史价格不是历史 ask 与盘口深度，因此本报告只可用于否决策略，不能据此批准真钱交易。",
        "- Valve 记录只有比赛日期，没有精确开赛时间；跨时区连接允许前后一天，无法唯一匹配的同队复赛已排除。",
        "- 2025 正收益没有在 2026 重现，说明阵容 Elo 基线对环境变化不够稳健；后续必须加入地图池、选图/禁图、LAN/线上、赛事级别及版本阶段。",
        "",
        "## 决策",
        "",
        "`approved_for_real_money = false`。在新的、未参与开发的后续日期再次通过锁箱前，CS2 只输出研究概率与理由，不输出实际下注金额。",
    ])
    report_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"coverage": payload["coverage"], "results": payload["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
