from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from .risk import recommend


@dataclass(frozen=True)
class BacktestRow:
    event_id: str
    decision_at: datetime
    start_at: datetime
    model_probability: float
    decimal_odds: float
    won: bool
    confidence: float


@dataclass(frozen=True)
class BacktestReport:
    bets: int
    opportunities: int
    turnover: float
    profit: float
    roi: float
    max_drawdown: float
    brier_score: float
    roi_ci95: tuple[float, float]


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
            ))
    return rows


def _roi_sample(profits: list[float], stakes: list[float], rng: random.Random) -> float:
    indices = [rng.randrange(len(profits)) for _ in profits]
    turnover = sum(stakes[i] for i in indices)
    return 0.0 if turnover == 0 else sum(profits[i] for i in indices) / turnover


def run_backtest(rows: list[BacktestRow], *, initial_bankroll: float = 1000, seed: int = 7) -> BacktestReport:
    if not rows:
        raise ValueError("backtest requires rows")
    ordered = sorted(rows, key=lambda r: r.decision_at)
    bankroll = peak = initial_bankroll
    max_drawdown = 0.0
    profits: list[float] = []
    stakes: list[float] = []
    brier: list[float] = []
    for row in ordered:
        if row.decision_at >= row.start_at:
            raise ValueError(f"look-ahead leakage for event {row.event_id}")
        rec = recommend(
            event_id=row.event_id, outcome="selection", model_probability=row.model_probability,
            decimal_odds=row.decimal_odds, bankroll=bankroll, confidence=row.confidence,
        )
        brier.append((rec.model_probability - float(row.won)) ** 2)
        if rec.action == "NO_BET":
            continue
        pnl = rec.stake * (row.decimal_odds - 1) if row.won else -rec.stake
        bankroll += pnl
        profits.append(pnl)
        stakes.append(rec.stake)
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, (peak - bankroll) / peak)
    turnover, profit = sum(stakes), sum(profits)
    roi = 0.0 if turnover == 0 else profit / turnover
    if profits:
        rng = random.Random(seed)
        samples = sorted(_roi_sample(profits, stakes, rng) for _ in range(2000))
        ci = (samples[49], samples[1949])
    else:
        ci = (0.0, 0.0)
    return BacktestReport(len(profits), len(rows), turnover, profit, roi, max_drawdown, mean(brier), ci)

