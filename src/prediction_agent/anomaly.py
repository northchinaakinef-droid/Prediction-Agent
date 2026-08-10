from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev

from .models import MarketSnapshot


@dataclass(frozen=True)
class Anomaly:
    kind: str
    severity: float
    message: str


def detect_market_anomalies(history: list[MarketSnapshot], z_threshold: float = 3.0) -> list[Anomaly]:
    """Detect jumps, spread deterioration and one-sided books without claiming causality."""
    if len(history) < 5:
        return []
    current = history[-1]
    prior = history[:-1]
    mids = [x.midpoint for x in prior]
    sd = stdev(mids)
    delta = current.midpoint - mean(mids)
    z = (0.0 if delta == 0 else float("inf") if delta > 0 else float("-inf")) if sd == 0 else delta / sd
    median_spread = sorted(x.spread for x in prior)[len(prior) // 2]
    anomalies: list[Anomaly] = []
    if abs(z) >= z_threshold:
        anomalies.append(Anomaly("price_jump", abs(z), f"midpoint z-score {z:.2f}"))
    if median_spread > 0 and current.spread >= median_spread * 2.5:
        anomalies.append(Anomaly("liquidity_drop", current.spread / median_spread, "spread widened sharply"))
    if abs(current.imbalance) >= 0.80:
        anomalies.append(Anomaly("book_imbalance", abs(current.imbalance), f"order-book imbalance {current.imbalance:.2f}"))
    return anomalies
