from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum


class Sport(StrEnum):
    NBA = "nba"
    CBA = "cba"
    LOL = "lol"
    CS2 = "cs2"


@dataclass(frozen=True)
class Quote:
    source: str
    event_id: str
    outcome: str
    decimal_odds: float
    observed_at: datetime
    available_size: float | None = None

    def __post_init__(self) -> None:
        if self.decimal_odds <= 1.0:
            raise ValueError("decimal_odds must be > 1")
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")


@dataclass(frozen=True)
class MarketSnapshot:
    event_id: str
    outcome: str
    observed_at: datetime
    best_bid: float
    best_ask: float
    bid_size: float
    ask_size: float
    volume: float | None = None

    @property
    def midpoint(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def imbalance(self) -> float:
        total = self.bid_size + self.ask_size
        return 0.0 if total == 0 else (self.bid_size - self.ask_size) / total


@dataclass(frozen=True)
class Evidence:
    source: str
    published_at: datetime
    title: str
    url: str
    reliability: float = 0.5


@dataclass(frozen=True)
class Recommendation:
    event_id: str
    outcome: str
    action: str
    model_probability: float
    market_probability: float
    edge: float
    expected_value: float
    stake: float
    stake_fraction: float
    confidence: float
    reasons: tuple[str, ...] = field(default_factory=tuple)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

