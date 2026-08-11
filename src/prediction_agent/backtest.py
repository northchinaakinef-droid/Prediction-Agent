from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from .risk import decimal_implied_probability, recommend


@dataclass(frozen=True)
class BacktestRow:
    event_id: str
    decision_at: datetime
    start_at: datetime
    model_probability: float
    decimal_odds: float
    won: bool
    confidence: float
    league: str = "unknown"
    entry_price: float | None = None
    spread: float | None = None
    available_size: float | None = None
    estimated_cost: float = 0.0


@dataclass(frozen=True)
class Stratum:
    label: str
    samples: int
    bets: int
    wins: int
    win_rate: float | None
    average_entry_price: float | None
    roi: float | None
    max_drawdown: float
    profit_factor: float | None


@dataclass(frozen=True)
class BacktestReport:
    samples: int
    bets: int
    opportunities: int
    wins: int
    win_rate: float | None
    direction_accuracy: float
    average_entry_price: float | None
    average_model_probability: float
    average_market_probability: float
    average_edge: float
    turnover: float
    profit: float
    roi: float | None
    max_drawdown: float
    profit_factor: float | None
    brier_score: float
    market_brier_score: float
    brier_skill_score: float | None
    log_loss: float
    market_log_loss: float
    maximum_losing_streak: int
    roi_ci95: tuple[float, float] | None
    calibration: tuple[dict[str, float | int], ...]
    edge_strata: tuple[Stratum, ...]
    entry_price_strata: tuple[Stratum, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


EDGE_BINS = ((-math.inf, 0.0, "<0%"), (0.0, .02, "0%-2%"), (.02, .05, "2%-5%"),
             (.05, .10, "5%-10%"), (.10, math.inf, ">=10%"))
PRICE_BINS = ((.50, .60, "0.50-0.60"), (.60, .70, "0.60-0.70"), (.70, .80, "0.70-0.80"),
              (.80, .90, "0.80-0.90"), (.90, .95, "0.90-0.95"), (.95, .97, "0.95-0.97"),
              (.97, math.inf, "0.97+"))


def _optional_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name)
    return None if value is None or not value.strip() else float(value)


def load_csv(path: str | Path) -> list[BacktestRow]:
    rows: list[BacktestRow] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            rows.append(BacktestRow(
                event_id=r["event_id"],
                decision_at=datetime.fromisoformat(r["decision_at"].replace("Z", "+00:00")),
                start_at=datetime.fromisoformat(r["start_at"].replace("Z", "+00:00")),
                model_probability=float(r["model_probability"]),
                decimal_odds=float(r["decimal_odds"]),
                won=r["won"].strip().lower() in {"1", "true", "yes"},
                confidence=float(r.get("confidence") or 1.0),
                league=(r.get("league") or "unknown").lower(),
                entry_price=_optional_float(r, "entry_price"),
                spread=_optional_float(r, "spread"),
                available_size=_optional_float(r, "available_size"),
                estimated_cost=float(r.get("estimated_cost") or 0.0),
            ))
    return rows


def _log_loss(y: float, p: float) -> float:
    p = min(1 - 1e-12, max(1e-12, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _max_drawdown(pnls: list[float], initial_bankroll: float) -> float:
    bankroll = peak = initial_bankroll
    result = 0.0
    for pnl in pnls:
        bankroll += pnl
        peak = max(peak, bankroll)
        result = max(result, 0.0 if peak <= 0 else (peak - bankroll) / peak)
    return result


def _profit_factor(pnls: list[float]) -> float | None:
    gains, losses = sum(x for x in pnls if x > 0), -sum(x for x in pnls if x < 0)
    if losses == 0:
        return None
    return gains / losses


def _max_losing_streak(pnls: list[float]) -> int:
    longest = current = 0
    for pnl in pnls:
        current = current + 1 if pnl < 0 else 0
        longest = max(longest, current)
    return longest


def _roi_sample(trades: list[dict[str, float | bool]], rng: random.Random) -> float:
    sample = [trades[rng.randrange(len(trades))] for _ in trades]
    turnover = sum(float(x["stake"]) for x in sample)
    return sum(float(x["pnl"]) for x in sample) / turnover


def _strata(
    evaluated: list[dict[str, Any]], bins: tuple[tuple[float, float, str], ...],
    value: Callable[[dict[str, Any]], float], initial_bankroll: float,
) -> tuple[Stratum, ...]:
    result = []
    for low, high, label in bins:
        members = [x for x in evaluated if low <= value(x) < high]
        trades = [x for x in members if x["traded"]]
        pnls = [float(x["pnl"]) for x in trades]
        stakes = [float(x["stake"]) for x in trades]
        wins = sum(bool(x["won"]) for x in trades)
        result.append(Stratum(
            label, len(members), len(trades), wins, wins / len(trades) if trades else None,
            mean(float(x["entry_price"]) for x in trades) if trades else None,
            sum(pnls) / sum(stakes) if stakes and sum(stakes) else None,
            _max_drawdown(pnls, initial_bankroll), _profit_factor(pnls),
        ))
    return tuple(result)


def _calibration(rows: list[BacktestRow]) -> tuple[dict[str, float | int], ...]:
    result = []
    for index in range(10):
        low, high = index / 10, (index + 1) / 10
        members = [r for r in rows if low <= r.model_probability < high or index == 9 and r.model_probability == 1]
        if members:
            result.append({"low": low, "high": high, "samples": len(members),
                           "mean_prediction": mean(r.model_probability for r in members),
                           "observed_rate": mean(float(r.won) for r in members)})
    return tuple(result)


def run_backtest(rows: list[BacktestRow], *, initial_bankroll: float = 1000, seed: int = 7) -> BacktestReport:
    if not rows:
        raise ValueError("backtest requires rows")
    ordered = sorted(rows, key=lambda r: r.decision_at)
    evaluated: list[dict[str, Any]] = []
    bankroll = initial_bankroll
    for row in ordered:
        if row.decision_at >= row.start_at:
            raise ValueError(f"look-ahead leakage for event {row.event_id}")
        if not 0 <= row.model_probability <= 1:
            raise ValueError(f"invalid model probability for event {row.event_id}")
        market_probability = row.entry_price if row.entry_price is not None else decimal_implied_probability(row.decimal_odds)
        rec = recommend(
            event_id=row.event_id, outcome="selection", model_probability=row.model_probability,
            decimal_odds=1 / market_probability, bankroll=bankroll, confidence=row.confidence,
            spread=row.spread, available_size=row.available_size, estimated_cost=row.estimated_cost,
            require_market_quality=False, trading_enabled=True,
        )
        traded = rec.action == "BET"
        stake = rec.stake if traded else 0.0
        pnl = stake * (1 / market_probability - 1) if traded and row.won else -stake if traded else 0.0
        bankroll += pnl
        evaluated.append({"row": row, "traded": traded, "stake": stake, "pnl": pnl, "won": row.won,
                          "entry_price": market_probability, "edge": row.model_probability - market_probability})

    trades = [x for x in evaluated if x["traded"]]
    pnls = [float(x["pnl"]) for x in trades]
    stakes = [float(x["stake"]) for x in trades]
    y = [float(r.won) for r in ordered]
    model_brier = mean((r.model_probability - actual) ** 2 for r, actual in zip(ordered, y))
    market_probs = [r.entry_price if r.entry_price is not None else decimal_implied_probability(r.decimal_odds) for r in ordered]
    market_brier = mean((p - actual) ** 2 for p, actual in zip(market_probs, y))
    turnover, profit = sum(stakes), sum(pnls)
    if trades:
        rng = random.Random(seed)
        samples = sorted(_roi_sample(trades, rng) for _ in range(2000))
        ci: tuple[float, float] | None = (samples[49], samples[1949])
    else:
        ci = None
    wins = sum(bool(x["won"]) for x in trades)
    return BacktestReport(
        samples=len(ordered), bets=len(trades), opportunities=len(ordered), wins=wins,
        win_rate=wins / len(trades) if trades else None,
        direction_accuracy=mean(float((r.model_probability >= .5) == r.won) for r in ordered),
        average_entry_price=mean(float(x["entry_price"]) for x in trades) if trades else None,
        average_model_probability=mean(r.model_probability for r in ordered),
        average_market_probability=mean(market_probs),
        average_edge=mean(r.model_probability - p for r, p in zip(ordered, market_probs)),
        turnover=turnover, profit=profit, roi=profit / turnover if turnover else None,
        max_drawdown=_max_drawdown(pnls, initial_bankroll), profit_factor=_profit_factor(pnls),
        brier_score=model_brier, market_brier_score=market_brier,
        brier_skill_score=1 - model_brier / market_brier if market_brier else None,
        log_loss=mean(_log_loss(actual, r.model_probability) for r, actual in zip(ordered, y)),
        market_log_loss=mean(_log_loss(actual, p) for p, actual in zip(market_probs, y)),
        maximum_losing_streak=_max_losing_streak(pnls), roi_ci95=ci, calibration=_calibration(ordered),
        edge_strata=_strata(evaluated, EDGE_BINS, lambda x: float(x["edge"]), initial_bankroll),
        entry_price_strata=_strata(evaluated, PRICE_BINS, lambda x: float(x["entry_price"]), initial_bankroll),
    )
