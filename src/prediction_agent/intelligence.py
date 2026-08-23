from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class RecommendationState(str, Enum):
    NO_BET = "NO_BET"
    WATCH = "WATCH"
    VALUE = "VALUE"
    STRONG_VALUE = "STRONG_VALUE"
    DATA_INCOMPLETE = "DATA_INCOMPLETE"
    NO_MARKET = "NO_MARKET"


@dataclass(frozen=True)
class QuantPrediction:
    baseline_probability: float
    calibrated_probability: float
    model_version: str
    features: dict


@dataclass(frozen=True)
class DataQualityScore:
    schedule_quality: float
    roster_quality: float
    market_quality: float
    stat_quality: float
    news_quality: float
    model_freshness: float
    missing: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return sum((self.schedule_quality, self.roster_quality, self.market_quality,
                    self.stat_quality, self.news_quality, self.model_freshness)) / 6

    @property
    def level(self) -> str:
        return "HIGH" if self.score >= .8 else "MEDIUM" if self.score >= .55 else "LOW"


@dataclass(frozen=True)
class DecisionResult:
    final_probability: float
    market_probability: float | None
    edge: float | None
    recommendation: RecommendationState
    max_position: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
