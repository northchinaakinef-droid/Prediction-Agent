"""Leakage-aware calibrated sports model with historical smart-money flow.

Workflow:
  collect -> downloads public large-trade history
  fit     -> trains and selects on train/validation only, writes frozen config
  finalize -> evaluates the untouched final slice once and writes a lock marker
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prediction_agent.providers.http import get_json
from research.polymarket_walkforward import Game, LEAGUES, extract_games, taker_fee


DATA = Path("data/research")
REPORTS = Path("reports/smart_money")
FROZEN = REPORTS / "frozen_config.json"
FINAL_LOCK = REPORTS / "FINAL_TEST_LOCK.json"
DATA_API = "https://data-api.polymarket.com"
TRADE_MIN_CASH = 500.0

# 2026 Tier-1 team names observed in Polymarket/Riot schedules. Academy/Challenger
# names are deliberately absent. Aliases are normalized before matching.
MAJOR_TEAMS = {
    # LCK
    "t1", "gen.g", "geng", "dplus kia", "hanwha life esports", "kt rolster",
    "nongshim red force", "bnk fearx", "kiwoom drx", "hanjin brion", "dn soopers",
    # LPL
    "bilibili gaming", "jd gaming", "beijing jdg esports", "invictus gaming",
    "weibo gaming", "top esports", "team we", "xian team we", "anyones legend",
    "ninjas in pyjamas", "thundertalk gaming", "oh my god", "lng esports",
    "suzhou lng esports", "edward gaming", "lgd gaming", "ultra prime", "funplus phoenix",
    # LEC
    "g2 esports", "karmine corp", "movistar koi", "giantx", "natus vincere",
    "team vitality", "fnatic", "team heretics", "sk gaming", "los ratones",
    # LCS
    "cloud9", "cloud9 kia", "team liquid", "team liquid alienware", "flyquest",
    "lyon", "sentinels", "shopify rebellion", "dignitas", "disguised",
}


def norm_team(name: str) -> str:
    return " ".join(name.strip().lower().replace("'", "").replace(".", ".").split())


def is_official_lol(game: Game) -> bool:
    dt = datetime.fromtimestamp(game.start_ts, timezone.utc)
    # Riot Worlds 2025, including qualified teams from outside the four leagues.
    if datetime(2025, 10, 15, tzinfo=timezone.utc) <= dt < datetime(2025, 11, 15, tzinfo=timezone.utc):
        return True
    # Riot 2026 First Stand and MSI are official international events.
    international = (
        datetime(2026, 3, 16, tzinfo=timezone.utc) <= dt < datetime(2026, 3, 23, tzinfo=timezone.utc)
        or datetime(2026, 6, 28, tzinfo=timezone.utc) <= dt < datetime(2026, 7, 13, tzinfo=timezone.utc)
    )
    return international or (norm_team(game.team_a) in MAJOR_TEAMS and norm_team(game.team_b) in MAJOR_TEAMS)


def load_games() -> dict[str, list[Game]]:
    result: dict[str, list[Game]] = {}
    for league in LEAGUES:
        events = json.loads((DATA / f"{league}_events.json").read_text(encoding="utf-8"))
        games = extract_games(league, events)
        prices = json.loads((DATA / f"{league}_prices.json").read_text(encoding="utf-8"))
        for game in games:
            game.price_a = prices.get(game.token_a)
        games = [g for g in games if g.price_a is not None]
        if league == "lol":
            games = [g for g in games if is_official_lol(g)]
        result[league] = games
    return result


def _trade_batch(event_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = None
        for retry in range(4):
            try:
                page = get_json(f"{DATA_API}/trades", {
                    "eventId": ",".join(event_ids), "limit": 10000, "offset": offset,
                    "takerOnly": "false", "filterType": "CASH", "filterAmount": TRADE_MIN_CASH,
                }, timeout=60)
                break
            except (HTTPError, TimeoutError):
                if retry == 3:
                    if len(event_ids) > 1:
                        middle = len(event_ids) // 2
                        return _trade_batch(event_ids[:middle]) + _trade_batch(event_ids[middle:])
                    raise
                time.sleep(2 ** retry)
        assert page is not None
        rows.extend(page)
        if len(page) < 10000 or offset >= 10000:
            break
        offset += len(page)
    return rows


def _cached_trade_batch(args: tuple[str, list[str]]) -> list[dict[str, Any]]:
    league, event_ids = args
    key = hashlib.sha1(",".join(event_ids).encode()).hexdigest()[:16]
    folder = DATA / "trade_batches" / league
    path = folder / f"{key}.json"
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    else:
        rows = _trade_batch(event_ids)
    keys = ("proxyWallet", "side", "conditionId", "size", "price", "timestamp", "outcomeIndex")
    rows = [
        {key: row.get(key) for key in keys}
        for row in rows
        if float(row.get("size") or 0) * float(row.get("price") or 0) >= TRADE_MIN_CASH
    ]
    if not path.exists():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def collect_trades(games_by_league: dict[str, list[Game]], refresh: bool = False) -> None:
    for league, games in games_by_league.items():
        path = DATA / f"{league}_large_trades.json"
        if path.exists() and not refresh:
            print(league, "trade cache exists")
            continue
        event_ids = sorted({g.event_id for g in games}, key=int)
        batches = [event_ids[i:i + 10] for i in range(0, len(event_ids), 10)]
        rows: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for page in pool.map(_cached_trade_batch, [(league, batch) for batch in batches]):
                rows.extend(page)
        conditions = {g.condition_id for g in games}
        rows = [r for r in rows if r.get("conditionId") in conditions]
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(league, "games", len(games), "trades", len(rows), "wallets", len({r.get('proxyWallet') for r in rows}))


def logit(p: float) -> float:
    p = min(0.995, max(0.005, p))
    return math.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1 / (1 + np.exp(-x))


def build_features(games: list[Game], trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        by_condition[str(trade["conditionId"])].append(trade)
    ratings: dict[str, float] = {}
    wallet_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "invested": 0.0, "games": 0.0})
    rows: list[dict[str, Any]] = []
    for game in sorted(games, key=lambda g: g.start_ts):
        cutoff = game.start_ts - 3600
        ra, rb = ratings.get(game.team_a, 1500.0), ratings.get(game.team_b, 1500.0)
        elo_p = 1 / (1 + 10 ** ((rb - ra) / 400))
        flow = denom = 0.0
        prior_smart_wallets = 0
        market_trades = sorted(by_condition.get(game.condition_id, []), key=lambda x: int(x["timestamp"]))
        for trade in market_trades:
            if int(trade["timestamp"]) > cutoff:
                continue
            wallet = str(trade["proxyWallet"])
            stat = wallet_stats[wallet]
            if stat["games"] < 5 or stat["invested"] <= 0:
                continue
            historical_roi = max(0.0, min(0.5, stat["pnl"] / stat["invested"]))
            quality = historical_roi * math.sqrt(stat["games"] / (stat["games"] + 10))
            if quality <= 0:
                continue
            cash = float(trade["size"]) * float(trade["price"])
            direction = 1 if int(trade["outcomeIndex"]) == 0 else -1
            if str(trade["side"]).upper() == "SELL":
                direction *= -1
            flow += quality * direction * cash
            denom += quality * cash
            prior_smart_wallets += 1
        smart_flow = flow / denom if denom else 0.0
        rows.append({
            "league": game.league, "event_id": game.event_id, "condition_id": game.condition_id,
            "start_ts": game.start_ts, "team_a": game.team_a, "team_b": game.team_b,
            "y": int(game.winner == 0), "market_p": float(game.price_a), "elo_p": elo_p,
            "smart_flow": smart_flow, "smart_trade_count": prior_smart_wallets,
        })
        actual = float(game.winner == 0)
        ratings[game.team_a] = ra + 20 * (actual - elo_p)
        ratings[game.team_b] = rb + 20 * ((1 - actual) - (1 - elo_p))
        # Update wallet quality only after this game's result becomes available.
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for trade in market_trades:
            if int(trade["timestamp"]) <= game.start_ts:
                grouped[str(trade["proxyWallet"])].append(trade)
        for wallet, wallet_trades in grouped.items():
            cashflow = 0.0
            positions = [0.0, 0.0]
            invested = 0.0
            for trade in wallet_trades:
                size, price, outcome = float(trade["size"]), float(trade["price"]), int(trade["outcomeIndex"])
                if str(trade["side"]).upper() == "BUY":
                    cashflow -= size * price
                    positions[outcome] += size
                    invested += size * price
                else:
                    cashflow += size * price
                    positions[outcome] -= size
            pnl = cashflow + positions[game.winner]
            if invested > 0:
                wallet_stats[wallet]["pnl"] += pnl
                wallet_stats[wallet]["invested"] += invested
                wallet_stats[wallet]["games"] += 1
    return rows


FEATURES = {
    "market": ["market_logit"],
    "market_elo": ["market_logit", "elo_logit"],
    "market_smart": ["market_logit", "smart_flow"],
    "all": ["market_logit", "elo_logit", "smart_flow"],
}


def matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    data = []
    for row in rows:
        values = {"market_logit": logit(row["market_p"]), "elo_logit": logit(row["elo_p"]), "smart_flow": row["smart_flow"]}
        data.append([values[name] for name in feature_names])
    return np.asarray(data, dtype=float), np.asarray([r["y"] for r in rows], dtype=float)


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mu) / sd])
    w = np.zeros(z.shape[1])
    penalty = np.eye(len(w)) * ridge
    penalty[0, 0] = 0
    for _ in range(100):
        p = sigmoid(z @ w)
        weight = np.maximum(p * (1 - p), 1e-6)
        hessian = z.T @ (z * weight[:, None]) + penalty
        gradient = z.T @ (p - y) + penalty @ w
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w, mu, sd


def predict(rows: list[dict[str, Any]], config: dict[str, Any]) -> np.ndarray:
    x, _ = matrix(rows, config["features"])
    z = np.column_stack([np.ones(len(x)), (x - np.asarray(config["mu"])) / np.asarray(config["sd"])])
    return sigmoid(z @ np.asarray(config["weights"]))


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"brier": float(np.mean((p - y) ** 2)), "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}


def trade_metrics(rows: list[dict[str, Any]], probs: np.ndarray, edge_threshold: float) -> dict[str, Any]:
    bankroll = 1000.0
    peak = bankroll
    max_dd = 0.0
    turnover = profit = 0.0
    bets = wins = 0
    curve = []
    for row, p_a in zip(rows, probs):
        market_a = float(row["market_p"])
        side = 0 if p_a - market_a >= edge_threshold else 1 if market_a - p_a >= edge_threshold else -1
        if side < 0:
            continue
        raw = market_a if side == 0 else 1 - market_a
        model_p = float(p_a) if side == 0 else 1 - float(p_a)
        price = min(0.99, raw + 0.01)
        if model_p <= price:
            continue
        decimal_odds = 1 / price
        kelly = max(0.0, (model_p * decimal_odds - 1) / (decimal_odds - 1))
        stake = min(bankroll * 0.0075, bankroll * 0.25 * kelly)
        if stake <= 0:
            continue
        shares = stake / price
        fee = taker_fee(shares, price)
        won = int(row["y"]) == (1 if side == 0 else 0)
        pnl = shares * float(won) - stake - fee
        bankroll += pnl
        turnover += stake
        profit += pnl
        bets += 1
        wins += int(won)
        peak = max(peak, bankroll)
        max_dd = max(max_dd, (peak - bankroll) / peak)
        curve.append({"ts": row["start_ts"], "bankroll": bankroll})
    audit_records = []
    for row, p_a in zip(rows, probs):
        market_a = float(row["market_p"])
        side = 0 if p_a - market_a >= edge_threshold else 1 if market_a - p_a >= edge_threshold else -1
        raw = market_a if side == 0 else 1 - market_a if side == 1 else None
        model_p = float(p_a) if side == 0 else 1 - float(p_a) if side == 1 else None
        entry = min(0.99, raw + 0.01) if raw is not None else None
        audit_records.append({
            "event_id": row["event_id"], "start_ts": row["start_ts"], "outcome": int(row["y"]),
            "model_probability_a": float(p_a), "market_probability_a": market_a, "side": side,
            "decision": "NO TRADE" if side < 0 or model_p <= entry else ("BUY YES" if side == 0 else "BUY NO"),
            "entry_ask": entry, "edge": model_p - entry if entry is not None else None,
            "spread": None, "available_size": None,
        })
    days = max(1.0, (rows[-1]["start_ts"] - rows[0]["start_ts"]) / 86400) if rows else 1
    annualized = (bankroll / 1000) ** (365 / days) - 1 if bankroll > 0 else -1.0
    return {
        "bets": bets, "wins": wins, "hit_rate": wins / bets if bets else None,
        "turnover": turnover, "profit": profit, "roi": profit / turnover if turnover else None,
        "bankroll_return": bankroll / 1000 - 1, "max_drawdown": max_dd,
        "annualized_return": annualized, "calmar": annualized / max_dd if max_dd else None,
        "curve": curve, "audit_records": audit_records,
    }


def calibration(y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for low in np.arange(0, 1, 0.1):
        mask = (p >= low) & (p < low + 0.1 if low < 0.9 else p <= 1)
        if mask.any():
            result.append({"low": float(low), "high": float(low + .1), "count": int(mask.sum()), "predicted": float(p[mask].mean()), "actual": float(y[mask].mean())})
    return result


def confidence_strata(y: np.ndarray, p: np.ndarray) -> list[dict[str, Any]]:
    confidence = np.maximum(p, 1 - p)
    correct = ((p >= .5).astype(int) == y).astype(float)
    result = []
    for low, high in [(0.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
        mask = (confidence >= low) & (confidence < high)
        if mask.any():
            result.append({"confidence": f"{low:.1f}-{min(high,1):.1f}", "count": int(mask.sum()), "mean_confidence": float(confidence[mask].mean()), "actual_accuracy": float(correct[mask].mean())})
    return result


def deviation_stats(rows: list[dict[str, Any]], p: np.ndarray) -> dict[str, float]:
    d = p - np.asarray([r["market_p"] for r in rows])
    return {"mean": float(d.mean()), "std": float(d.std()), "p05": float(np.quantile(d, .05)), "p25": float(np.quantile(d, .25)), "median": float(np.median(d)), "p75": float(np.quantile(d, .75)), "p95": float(np.quantile(d, .95))}


def prepare() -> dict[str, list[dict[str, Any]]]:
    games = load_games()
    print("scope", {k: len(v) for k, v in games.items()})
    result = {}
    for league, league_games in games.items():
        trades = json.loads((DATA / f"{league}_large_trades.json").read_text(encoding="utf-8"))
        result[league] = build_features(league_games, trades)
    return result


def fit_and_freeze() -> None:
    datasets = prepare()
    configs = {}
    validation = {}
    for league, rows in datasets.items():
        n = len(rows)
        train, val = rows[:int(n * .55)], rows[int(n * .55):int(n * .80)]
        candidates = []
        for name, names in FEATURES.items():
            x, y = matrix(train, names)
            for ridge in (0.1, 1.0, 10.0):
                w, mu, sd = fit_logistic(x, y, ridge)
                config = {"name": name, "features": names, "ridge": ridge, "weights": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist()}
                probs = predict(val, config)
                pm = probability_metrics(np.asarray([r["y"] for r in val]), probs)
                candidates.append((pm["brier"] + .25 * pm["log_loss"], config, pm, probs))
        candidates.sort(key=lambda x: x[0])
        _, best, pm, probs = candidates[0]
        threshold_results = [(edge, trade_metrics(val, probs, edge)) for edge in (.03, .05, .08)]
        eligible = [(e, m) for e, m in threshold_results if m["bets"] >= 30 and (m["roi"] or -1) > 0]
        edge, tm = max(eligible, key=lambda x: x[1]["roi"]) if eligible else max(threshold_results, key=lambda x: x[1]["roi"] if x[1]["roi"] is not None else -999)
        best["edge_threshold"] = edge
        best["train_end_ts"] = train[-1]["start_ts"]
        best["validation_end_ts"] = val[-1]["start_ts"]
        best["validation_positive"] = bool(eligible)
        configs[league] = best
        validation[league] = {"probability": pm, "trading": {k: v for k, v in tm.items() if k != "curve"}, "candidates": len(candidates)}
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(), "configs": configs, "validation": validation,
        "data_hash": hashlib.sha256(json.dumps(datasets, sort_keys=True).encode()).hexdigest(),
        "test_policy": "final 20%, execute exactly once with finalize",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


def write_svg(results: dict[str, Any]) -> None:
    colors = {"nba": "#2563eb", "cba": "#dc2626", "lol": "#16a34a"}
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="560" viewBox="0 0 720 560">', '<rect width="100%" height="100%" fill="white"/>', '<text x="360" y="32" text-anchor="middle" font-size="20">Final test calibration</text>', '<line x1="80" y1="480" x2="640" y2="80" stroke="#999" stroke-dasharray="5 5"/>']
    for i in range(6):
        x, y = 80 + i * 112, 480 - i * 80
        parts += [f'<line x1="{x}" y1="480" x2="{x}" y2="486" stroke="black"/>', f'<text x="{x}" y="505" text-anchor="middle" font-size="12">{i/5:.1f}</text>', f'<line x1="74" y1="{y}" x2="80" y2="{y}" stroke="black"/>', f'<text x="64" y="{y+4}" text-anchor="end" font-size="12">{i/5:.1f}</text>']
    for league, result in results.items():
        points = " ".join(f'{80+b["predicted"]*560:.1f},{480-b["actual"]*400:.1f}' for b in result["calibration"])
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[league]}" stroke-width="3"/>')
        parts.append(f'<text x="650" y="{110+25*list(results).index(league)}" fill="{colors[league]}" font-size="14">{league.upper()}</text>')
    parts += ['<text x="360" y="540" text-anchor="middle" font-size="14">Mean predicted probability</text>', '<text transform="translate(20,280) rotate(-90)" text-anchor="middle" font-size="14">Observed frequency</text>', '</svg>']
    (REPORTS / "calibration.svg").write_text("\n".join(parts), encoding="utf-8")


def finalize_once() -> None:
    if FINAL_LOCK.exists():
        raise SystemExit("FINAL TEST IS LOCKED: refusing to run twice")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    datasets = prepare()
    if hashlib.sha256(json.dumps(datasets, sort_keys=True).encode()).hexdigest() != frozen["data_hash"]:
        raise SystemExit("dataset changed after freeze")
    results = {}
    for league, rows in datasets.items():
        test = rows[int(len(rows) * .80):]
        config = frozen["configs"][league]
        probs = predict(test, config)
        y = np.asarray([r["y"] for r in test])
        tm = trade_metrics(test, probs, config["edge_threshold"])
        market_probs = np.asarray([r["market_p"] for r in test])
        results[league] = {
            "samples": len(test), "probability": probability_metrics(y, probs),
            "market_probability": probability_metrics(y, market_probs),
            "trading": tm, "calibration": calibration(y, probs),
            "confidence_strata": confidence_strata(y, probs), "deviation": deviation_stats(test, probs),
            "model": {k: config[k] for k in ["name", "features", "ridge", "edge_threshold", "validation_positive"]},
        }
    lock = {"ran_at": datetime.now(timezone.utc).isoformat(), "results": results, "frozen_hash": hashlib.sha256(FROZEN.read_bytes()).hexdigest()}
    FINAL_LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    write_svg(results)
    print(json.dumps({k: {"samples": v["samples"], "probability": v["probability"], "trading": {x:y for x,y in v["trading"].items() if x != "curve"}, "model": v["model"]} for k,v in results.items()}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect", "fit", "finalize"])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.mode == "collect":
        collect_trades(load_games(), args.refresh)
    elif args.mode == "fit":
        fit_and_freeze()
    else:
        finalize_once()


if __name__ == "__main__":
    main()
