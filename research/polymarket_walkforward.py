"""Reproducible, leakage-aware walk-forward backtest on public Polymarket sports data.

The strategy is intentionally fixed across leagues: online Elo, 50% shrinkage toward
the market, five percentage-point minimum edge, 0.75% bankroll cap. It does not tune
parameters on the test period.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError

from prediction_agent.providers.http import get_json


GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

LEAGUES = {
    "nba": {"tag": 745, "start": "2025-10-21", "end": "2026-07-01"},
    "cba": {"tag": 103097, "start": "2025-12-01", "end": "2026-07-01"},
    # Include the previous official Worlds as historical context, plus 2026 H1.
    "lol": {"tag": 65, "start": "2025-10-15", "end": "2026-07-01"},
}


@dataclass
class Game:
    league: str
    event_id: str
    start_ts: int
    team_a: str
    team_b: str
    winner: int
    token_a: str
    condition_id: str
    volume: float
    price_a: float | None = None


def parsed(value: Any) -> list[Any]:
    return json.loads(value) if isinstance(value, str) else list(value or [])


def iso_ts(value: str) -> int:
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def fetch_events(tag: int, start: str, end: str, cache: Path, refresh: bool = False) -> list[dict[str, Any]]:
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while True:
        params: dict[str, object] = {
            "tag_id": tag, "closed": "true", "limit": 500,
            "start_time_min": start + "T00:00:00Z", "start_time_max": end + "T00:00:00Z",
            "order": "startTime", "ascending": "true",
        }
        if cursor:
            params["after_cursor"] = cursor
        response = get_json(f"{GAMMA}/events/keyset", params, timeout=30)
        page = response.get("events", [])
        if not page:
            break
        rows.extend(page)
        next_cursor = response.get("next_cursor")
        if not next_cursor or next_cursor == cursor or len(rows) >= 10_000:
            break
        cursor = next_cursor
        time.sleep(0.15)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def extract_games(league: str, events: list[dict[str, Any]]) -> list[Game]:
    cfg = LEAGUES[league]
    low, high = iso_ts(cfg["start"] + "T00:00:00Z"), iso_ts(cfg["end"] + "T00:00:00Z")
    games: list[Game] = []
    for event in events:
        for market in event.get("markets", []):
            outcomes, prices, tokens = parsed(market.get("outcomes")), parsed(market.get("outcomePrices")), parsed(market.get("clobTokenIds"))
            start = market.get("gameStartTime")
            if not start or len(outcomes) != 2 or len(tokens) != 2 or len(prices) != 2:
                continue
            start_ts = iso_ts(start)
            if not low <= start_ts < high:
                continue
            resolved = [float(x) for x in prices]
            if sorted(resolved) != [0.0, 1.0]:
                continue
            market_type = market.get("sportsMarketType")
            if market_type not in (None, "moneyline"):
                continue
            games.append(Game(
                league, str(event.get("id")), start_ts, str(outcomes[0]), str(outcomes[1]),
                0 if resolved[0] == 1 else 1, str(tokens[0]), str(market.get("conditionId") or ""),
                float(market.get("volume") or 0),
            ))
    # De-duplicate event/teams; Gamma tags can expose the same event more than once.
    unique = {(g.event_id, g.team_a, g.team_b, g.start_ts): g for g in games}
    return sorted(unique.values(), key=lambda g: g.start_ts)


def fetch_pregame_price(game: Game, cutoff_seconds: int = 3600) -> float | None:
    cutoff = game.start_ts - cutoff_seconds
    for retry in range(3):
        try:
            row = get_json(f"{CLOB}/prices-history", {
                "market": game.token_a, "startTs": game.start_ts - 172800, "endTs": cutoff, "fidelity": 30,
            }, timeout=30)
            points = [p for p in row.get("history", []) if int(p["t"]) <= cutoff]
            value = float(points[-1]["p"]) if points else None
            return value if value is None or 0.01 <= value <= 0.99 else None
        except HTTPError as exc:
            if exc.code in (400, 404):
                return None
            if retry == 2:
                raise
            time.sleep(1 + retry)
        except Exception:
            if retry == 2:
                raise
            time.sleep(1 + retry)
    return None


def fill_prices(games: list[Game], price_cache: Path) -> None:
    cached: dict[str, Any] = json.loads(price_cache.read_text(encoding="utf-8")) if price_cache.exists() else {}
    missing = [game for game in games if game.token_a not in cached]
    if missing:
        price_cache.parent.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(fetch_pregame_price, game): game for game in missing}
            for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
                game = futures[future]
                cached[game.token_a] = future.result()
                if index % 50 == 0:
                    price_cache.write_text(json.dumps(cached), encoding="utf-8")
        price_cache.write_text(json.dumps(cached), encoding="utf-8")
    for game in games:
        game.price_a = cached.get(game.token_a)


def elo_probability(rating_a: float, rating_b: float) -> float:
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def taker_fee(shares: float, price: float) -> float:
    return shares * 0.03 * price * (1 - price)


def max_drawdown(values: list[float]) -> float:
    peak, result = values[0], 0.0
    for value in values:
        peak = max(peak, value)
        result = max(result, (peak - value) / peak)
    return result


def test_league(league: str, games: list[Game]) -> dict[str, Any]:
    ratings: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    split = int(len(games) * 0.60)
    bankroll = 1000.0
    curve = [bankroll]
    turnover = profit = 0.0
    bets = wins = 0
    for index, game in enumerate(games):
        ra, rb = ratings.get(game.team_a, 1500.0), ratings.get(game.team_b, 1500.0)
        elo_a = elo_probability(ra, rb)
        actual_a = float(game.winner == 0)
        if index >= split and game.price_a is not None:
            market_a = game.price_a
            # Fixed ex-ante rule: shrink the independent signal halfway to the market.
            estimate_a = market_a + 0.50 * (elo_a - market_a)
            side = 0 if estimate_a - market_a >= 0.05 else 1 if market_a - estimate_a >= 0.05 else -1
            pnl = stake = 0.0
            won = None
            execution_price = None
            if side >= 0:
                raw_price = market_a if side == 0 else 1 - market_a
                model_p = estimate_a if side == 0 else 1 - estimate_a
                # One-cent adverse slippage approximation; reject extreme/illiquid quotes.
                execution_price = min(0.99, raw_price + 0.01)
                edge = model_p - execution_price
                if edge > 0 and game.volume >= 100:
                    net_odds = (1 / execution_price) - 1
                    kelly = max(0.0, (model_p * (1 / execution_price) - 1) / net_odds)
                    stake = min(bankroll * 0.0075, bankroll * 0.25 * kelly)
                    shares = stake / execution_price
                    fee = taker_fee(shares, execution_price)
                    won = game.winner == side
                    pnl = shares * (1.0 if won else 0.0) - stake - fee
                    bankroll += pnl
                    turnover += stake
                    profit += pnl
                    bets += 1
                    wins += int(won)
                    curve.append(bankroll)
            records.append({
                "event_id": game.event_id, "start": datetime.fromtimestamp(game.start_ts, timezone.utc).isoformat(),
                "team_a": game.team_a, "team_b": game.team_b, "winner": game.winner,
                "market_p_a": market_a, "elo_p_a": elo_a, "estimate_p_a": estimate_a,
                "side": side, "execution_price": execution_price, "stake": stake, "pnl": pnl, "won": won,
            })
        expected = elo_a
        score = actual_a
        ratings[game.team_a] = ra + 20 * (score - expected)
        ratings[game.team_b] = rb + 20 * ((1 - score) - (1 - expected))
    test_records = [r for r in records]
    market_brier = mean((r["market_p_a"] - float(r["winner"] == 0)) ** 2 for r in test_records) if test_records else math.nan
    elo_brier = mean((r["elo_p_a"] - float(r["winner"] == 0)) ** 2 for r in test_records) if test_records else math.nan
    return {
        "league": league, "all_games": len(games), "train_games": split, "test_games_with_price": len(test_records),
        "price_coverage": len(test_records) / max(1, len(games) - split), "bets": bets, "wins": wins,
        "hit_rate": wins / bets if bets else None, "turnover": turnover, "profit": profit,
        "roi": profit / turnover if turnover else None, "bankroll_return": bankroll / 1000 - 1,
        "max_drawdown": max_drawdown(curve), "market_brier": market_brier, "elo_brier": elo_brier,
        "records": test_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/research")
    parser.add_argument("--output", default="reports/polymarket_walkforward.json")
    parser.add_argument("--refresh-events", action="store_true")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    reports = []
    for league, cfg in LEAGUES.items():
        events = fetch_events(int(cfg["tag"]), str(cfg["start"]), str(cfg["end"]), data_dir / f"{league}_events.json", args.refresh_events)
        games = extract_games(league, events)
        price_cache = data_dir / f"{league}_prices.json"
        fill_prices(games, price_cache)
        report = test_league(league, games)
        reports.append(report)
        print(league, {k: v for k, v in report.items() if k != "records"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
