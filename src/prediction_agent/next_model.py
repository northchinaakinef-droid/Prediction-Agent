"""Market-anchored, leakage-aware models for the next research cycle."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_FEATURES: dict[str, tuple[str, ...]] = {
    "nba": ("consensus_probability", "injury_impact_diff", "rest_days_diff", "lineup_strength_diff"),
    "cba": ("consensus_probability", "injury_impact_diff", "rest_days_diff", "foreign_player_strength_diff"),
    "lol": ("consensus_probability", "roster_stability_diff", "patch_form_diff", "side_advantage"),
    "cs2": ("consensus_probability", "rating_diff", "roster_stability_diff", "map_pool_diff", "lan_advantage"),
}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed


def _clip_probability(value: float) -> float:
    if not 0 < value < 1:
        raise ValueError("probabilities must be strictly between zero and one")
    return min(1 - 1e-6, max(1e-6, value))


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 40))
        return 1 / (1 + z)
    z = math.exp(max(value, -40))
    return z / (1 + z)


@dataclass(frozen=True)
class TimedFeature:
    value: float
    observed_at: datetime
    source: str

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("feature observed_at must be timezone-aware")
        if not self.source.strip() or not math.isfinite(self.value):
            raise ValueError("feature requires a source and finite value")


@dataclass(frozen=True)
class ModelRow:
    event_id: str
    league: str
    decision_at: datetime
    start_at: datetime
    settled_at: datetime
    market_probability: float
    outcome: int
    features: dict[str, TimedFeature]
    yes_ask: float | None = None
    no_ask: float | None = None
    spread: float | None = None
    available_size: float | None = None

    def __post_init__(self) -> None:
        if self.league not in DEFAULT_FEATURES:
            raise ValueError(f"unsupported league: {self.league}")
        if any(t.tzinfo is None for t in (self.decision_at, self.start_at, self.settled_at)):
            raise ValueError("row timestamps must be timezone-aware")
        if self.decision_at >= self.start_at or self.start_at > self.settled_at:
            raise ValueError("invalid decision/start/settlement chronology")
        _clip_probability(self.market_probability)
        if self.outcome not in (0, 1):
            raise ValueError("outcome must be 0 or 1")
        for name, feature in self.features.items():
            if feature.observed_at > self.decision_at:
                raise ValueError(f"look-ahead feature {name} for event {self.event_id}")
        for ask in (self.yes_ask, self.no_ask):
            if ask is not None and not 0 < ask < 1:
                raise ValueError("ask prices must be between zero and one")


def row_from_dict(value: dict[str, Any]) -> ModelRow:
    features = {
        name: TimedFeature(float(item["value"]), _parse_time(item["observed_at"]), str(item["source"]))
        for name, item in value.get("features", {}).items()
    }
    return ModelRow(
        event_id=str(value["event_id"]), league=str(value["league"]).lower(),
        decision_at=_parse_time(value["decision_at"]), start_at=_parse_time(value["start_at"]),
        settled_at=_parse_time(value["settled_at"]), market_probability=float(value["market_probability"]),
        outcome=int(value["outcome"]), features=features,
        yes_ask=float(value["yes_ask"]) if value.get("yes_ask") is not None else None,
        no_ask=float(value["no_ask"]) if value.get("no_ask") is not None else None,
        spread=float(value["spread"]) if value.get("spread") is not None else None,
        available_size=float(value["available_size"]) if value.get("available_size") is not None else None,
    )


def load_jsonl(path: str | Path) -> list[ModelRow]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(row_from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid row {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("dataset is empty")
    return rows


@dataclass(frozen=True)
class AnchoredLogitModel:
    league: str
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    trained_through: datetime
    samples: int

    def predict(self, row: ModelRow) -> float:
        if row.league != self.league:
            raise ValueError("model and row league differ")
        missing = [name for name in self.feature_names if name not in row.features]
        if missing:
            raise ValueError("missing features: " + ", ".join(missing))
        adjustment = self.weights[0]
        for index, name in enumerate(self.feature_names):
            z = (row.features[name].value - self.means[index]) / self.scales[index]
            adjustment += self.weights[index + 1] * z
        return _sigmoid(_logit(row.market_probability) + adjustment)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["trained_through"] = self.trained_through.isoformat()
        return result


def fit_anchored_model(
    rows: Iterable[ModelRow], feature_names: Iterable[str], *, ridge: float = 1.0,
    learning_rate: float = 0.15, iterations: int = 1200,
) -> AnchoredLogitModel:
    data = sorted(rows, key=lambda row: row.decision_at)
    if len(data) < 30:
        raise ValueError("at least 30 settled training rows are required")
    leagues = {row.league for row in data}
    if len(leagues) != 1:
        raise ValueError("each model must contain exactly one league")
    names = tuple(feature_names)
    if not names:
        raise ValueError("at least one independent feature is required")
    missing = [(row.event_id, name) for row in data for name in names if name not in row.features]
    if missing:
        raise ValueError(f"missing feature {missing[0][1]} for event {missing[0][0]}")
    columns = [[row.features[name].value for row in data] for name in names]
    means = tuple(mean(column) for column in columns)
    scales = tuple(max(1e-8, math.sqrt(mean((value - center) ** 2 for value in column)))
                   for column, center in zip(columns, means))
    matrix = [[(row.features[name].value - means[index]) / scales[index]
               for index, name in enumerate(names)] for row in data]
    weights = [0.0] * (len(names) + 1)
    for iteration in range(iterations):
        gradients = [0.0] * len(weights)
        for row, values in zip(data, matrix):
            prediction = _sigmoid(_logit(row.market_probability) + weights[0]
                                  + sum(weight * value for weight, value in zip(weights[1:], values)))
            error = prediction - row.outcome
            gradients[0] += error
            for index, value in enumerate(values, 1):
                gradients[index] += error * value
        gradients[0] /= len(data)
        for index in range(1, len(gradients)):
            gradients[index] = gradients[index] / len(data) + ridge * weights[index] / len(data)
        step = learning_rate / math.sqrt(1 + iteration / 100)
        for index in range(len(weights)):
            weights[index] -= step * gradients[index]
        if max(abs(value) for value in gradients) < 1e-7:
            break
    return AnchoredLogitModel(
        league=next(iter(leagues)), feature_names=names, means=means, scales=scales,
        weights=tuple(weights), trained_through=max(row.settled_at for row in data), samples=len(data),
    )


def _log_loss(outcome: int, probability: float) -> float:
    probability = _clip_probability(probability)
    return -(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability))


def _max_drawdown(pnls: list[float], initial: float = 1000.0) -> float:
    bankroll = peak = initial
    worst = 0.0
    for pnl in pnls:
        bankroll += pnl
        peak = max(peak, bankroll)
        worst = max(worst, (peak - bankroll) / peak)
    return worst

def _mean_ci95(values: list[float]) -> tuple[float, float] | None:
    if len(values) < 2:
        return None
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return center - margin, center + margin


@dataclass(frozen=True)
class FoldResult:
    train_samples: int
    test_samples: int
    test_start: str
    test_end: str
    model_brier: float
    market_brier: float
    model_log_loss: float
    market_log_loss: float
    trades: int
    roi: float | None


@dataclass(frozen=True)
class WalkForwardReport:
    league: str
    feature_names: tuple[str, ...]
    samples: int
    predictions: int
    folds: tuple[FoldResult, ...]
    model_brier: float
    market_brier: float
    brier_skill_vs_market: float
    brier_improvement_ci95: tuple[float, float] | None
    model_log_loss: float
    market_log_loss: float
    execution_coverage: float
    trades: int
    roi: float | None
    roi_ci95: tuple[float, float] | None
    max_drawdown: float
    profit_factor: float | None
    positive_fold_ratio: float
    approved_for_paper_trading: bool
    rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def walk_forward_evaluate(
    rows: Iterable[ModelRow], feature_names: Iterable[str] | None = None, *,
    initial_train: int = 120, test_size: int = 50, min_edge: float = 0.03,
    max_spread: float = 0.03, min_liquidity: float = 100.0,
) -> WalkForwardReport:
    data = sorted(rows, key=lambda row: row.decision_at)
    if not data:
        raise ValueError("dataset is empty")
    leagues = {row.league for row in data}
    if len(leagues) != 1:
        raise ValueError("evaluate each league separately")
    league = next(iter(leagues))
    names = tuple(feature_names or DEFAULT_FEATURES[league])
    if initial_train < 30 or test_size < 10 or len(data) < initial_train + test_size:
        raise ValueError("insufficient rows for requested walk-forward windows")
    predicted: list[tuple[ModelRow, float]] = []
    folds: list[FoldResult] = []
    all_pnls: list[float] = []
    executable = 0
    for start in range(initial_train, len(data), test_size):
        test = data[start:min(len(data), start + test_size)]
        if len(test) < 10:
            break
        cutoff = test[0].decision_at
        train = [row for row in data[:start] if row.settled_at <= cutoff]
        if len(train) < 30:
            continue
        model = fit_anchored_model(train, names)
        fold_predictions = [(row, model.predict(row)) for row in test]
        predicted.extend(fold_predictions)
        fold_pnls, fold_turnover = [], 0.0
        for row, probability in fold_predictions:
            if row.yes_ask is None or row.no_ask is None:
                continue
            executable += 1
            if row.spread is None or row.available_size is None or row.spread > max_spread or row.available_size < min_liquidity:
                continue
            side = 1 if probability - row.yes_ask >= min_edge else 0 if (1 - probability) - row.no_ask >= min_edge else -1
            if side < 0:
                continue
            price = row.yes_ask if side == 1 else row.no_ask
            won = row.outcome == side
            fee = 0.03 * (1 - price)
            pnl = ((1 / price) if won else 0.0) - 1.0 - fee
            fold_pnls.append(pnl)
            all_pnls.append(pnl)
            fold_turnover += 1
        outcomes = [row.outcome for row, _ in fold_predictions]
        model_probs = [probability for _, probability in fold_predictions]
        market_probs = [row.market_probability for row, _ in fold_predictions]
        folds.append(FoldResult(
            train_samples=len(train), test_samples=len(test), test_start=test[0].decision_at.isoformat(),
            test_end=test[-1].decision_at.isoformat(),
            model_brier=mean((p - y) ** 2 for p, y in zip(model_probs, outcomes)),
            market_brier=mean((p - y) ** 2 for p, y in zip(market_probs, outcomes)),
            model_log_loss=mean(_log_loss(y, p) for p, y in zip(model_probs, outcomes)),
            market_log_loss=mean(_log_loss(y, p) for p, y in zip(market_probs, outcomes)),
            trades=len(fold_pnls), roi=sum(fold_pnls) / fold_turnover if fold_turnover else None,
        ))
    if not predicted:
        raise ValueError("no valid walk-forward fold; check settlement timestamps")
    outcomes = [row.outcome for row, _ in predicted]
    model_probs = [probability for _, probability in predicted]
    market_probs = [row.market_probability for row, _ in predicted]
    model_brier = mean((p - y) ** 2 for p, y in zip(model_probs, outcomes))
    market_brier = mean((p - y) ** 2 for p, y in zip(market_probs, outcomes))
    model_ll = mean(_log_loss(y, p) for p, y in zip(model_probs, outcomes))
    market_ll = mean(_log_loss(y, p) for p, y in zip(market_probs, outcomes))
    brier_improvements = [
        (market - outcome) ** 2 - (model - outcome) ** 2
        for market, model, outcome in zip(market_probs, model_probs, outcomes)
    ]
    brier_ci = _mean_ci95(brier_improvements)
    gains, losses = sum(x for x in all_pnls if x > 0), -sum(x for x in all_pnls if x < 0)
    positive_folds = sum(fold.model_brier < fold.market_brier for fold in folds)
    execution_coverage = executable / len(predicted)
    roi = sum(all_pnls) / len(all_pnls) if all_pnls else None
    roi_ci = _mean_ci95(all_pnls)
    reasons = []
    if len(predicted) < 200:
        reasons.append("fewer than 200 out-of-sample predictions")
    if len(folds) < 3:
        reasons.append("fewer than 3 walk-forward folds")
    if model_brier >= market_brier:
        reasons.append("Brier score does not beat market")
    if brier_ci is None or brier_ci[0] <= 0:
        reasons.append("Brier improvement 95% CI is not above zero")
    if model_ll >= market_ll:
        reasons.append("Log Loss does not beat market")
    if positive_folds / len(folds) < 2 / 3:
        reasons.append("probability edge is not stable across folds")
    if execution_coverage < 0.8:
        reasons.append("less than 80% execution-price coverage")
    if len(all_pnls) < 100:
        reasons.append("fewer than 100 costed OOS trades")
    if roi is None or roi <= 0:
        reasons.append("costed OOS ROI is not positive")
    if roi_ci is None or roi_ci[0] <= 0:
        reasons.append("costed OOS ROI 95% CI is not above zero")
    if losses and gains / losses <= 1.05:
        reasons.append("profit factor is not above 1.05")
    if _max_drawdown(all_pnls, initial=100.0) > 0.20:
        reasons.append("maximum drawdown exceeds 20%")
    return WalkForwardReport(
        league=league, feature_names=names, samples=len(data), predictions=len(predicted), folds=tuple(folds),
        model_brier=model_brier, market_brier=market_brier,
        brier_skill_vs_market=1 - model_brier / market_brier if market_brier else 0.0,
        brier_improvement_ci95=brier_ci,
        model_log_loss=model_ll, market_log_loss=market_ll, execution_coverage=execution_coverage,
        trades=len(all_pnls), roi=roi, roi_ci95=roi_ci,
        max_drawdown=_max_drawdown(all_pnls, initial=100.0),
        profit_factor=gains / losses if losses else None,
        positive_fold_ratio=positive_folds / len(folds), approved_for_paper_trading=not reasons,
        rejection_reasons=tuple(reasons),
    )
