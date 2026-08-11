from __future__ import annotations

import math

from .models import Recommendation


def normalize_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Remove a two-way market's proportional overround."""
    if prob_a <= 0 or prob_b <= 0:
        raise ValueError("probabilities must be positive")
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def decimal_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("decimal odds must be > 1")
    return 1.0 / decimal_odds


def kelly_fraction(probability: float, decimal_odds: float) -> float:
    if not 0 <= probability <= 1 or decimal_odds <= 1:
        raise ValueError("invalid probability or odds")
    net = decimal_odds - 1
    return max(0.0, (probability * decimal_odds - 1) / net)


def recommend(
    *,
    event_id: str,
    outcome: str,
    model_probability: float,
    decimal_odds: float,
    bankroll: float,
    confidence: float,
    kelly_scale: float = 0.25,
    max_bet_fraction: float = 0.0075,
    min_edge: float = 0.025,
    min_confidence: float = 0.60,
    spread: float | None = None,
    available_size: float | None = None,
    max_spread: float = 0.03,
    min_available_size: float = 100.0,
    estimated_cost: float = 0.0,
    require_market_quality: bool = True,
    trading_enabled: bool = False,
) -> Recommendation:
    """Risk-capped recommendation with uncertainty shrinkage toward the market."""
    if bankroll < 0 or not 0 <= confidence <= 1:
        raise ValueError("invalid bankroll or confidence")
    market_probability = decimal_implied_probability(decimal_odds)
    # Low-confidence models should not manufacture large edges.
    shrunk_probability = market_probability + confidence * (model_probability - market_probability)
    edge = shrunk_probability - market_probability
    net_edge = edge - max(0.0, estimated_cost)
    ev = shrunk_probability * decimal_odds - 1 - max(0.0, estimated_cost)
    raw_fraction = kelly_fraction(shrunk_probability, decimal_odds) * kelly_scale
    stake_fraction = min(max_bet_fraction, raw_fraction)
    reasons: list[str] = [
        f"uncertainty-adjusted edge={edge:.2%}",
        f"cost-adjusted edge={net_edge:.2%}",
        f"net EV={ev:.2%}",
    ]
    if net_edge < min_edge:
        reasons.append("cost-adjusted edge below threshold")
    if confidence < min_confidence:
        reasons.append("confidence below threshold")
    if not trading_enabled:
        reasons.append("strategy not approved for trading")
    market_quality_missing = require_market_quality and (spread is None or available_size is None)
    spread_too_wide = spread is not None and spread > max_spread
    liquidity_too_low = available_size is not None and available_size < min_available_size
    if market_quality_missing:
        reasons.append("spread/liquidity unavailable")
    if spread_too_wide:
        reasons.append("spread too wide")
    if liquidity_too_low:
        reasons.append("insufficient available size")
    if (not trading_enabled or net_edge < min_edge or confidence < min_confidence or not math.isfinite(ev) or ev <= 0
            or market_quality_missing or spread_too_wide or liquidity_too_low):
        action, stake_fraction = "NO_BET", 0.0
    else:
        action = "BET"
    return Recommendation(
        event_id=event_id,
        outcome=outcome,
        action=action,
        model_probability=shrunk_probability,
        market_probability=market_probability,
        edge=edge,
        expected_value=ev,
        stake=round(bankroll * stake_fraction, 2),
        stake_fraction=stake_fraction,
        confidence=confidence,
        reasons=tuple(reasons),
        decision=f"BUY {outcome.upper()}" if action == "BET" else "NO TRADE",
    )
