from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class RiskConfig:
    """Environment-wired risk limits with the legacy hardcoded values as defaults.

    The values in ``.env.example`` are no longer phantom config: ``RiskConfig.from_env()``
    reads them once and the daily/live paths pass the resulting object into every
    ``recommend()`` call site.
    """
    kelly_scale: float = 0.25
    max_bet_fraction: float = 0.0075
    max_daily_risk_fraction: float = 0.025
    max_event_risk_fraction: float = 0.01
    max_drawdown_fraction: float = 0.10
    drawdown_warn_fraction: float = 0.10
    drawdown_circuit_fraction: float = 0.15
    max_correlation_group_fraction: float = 0.05
    max_depth_fraction: float = 0.10
    min_edge: float = 0.025
    min_confidence: float = 0.60
    max_spread: float = 0.03
    min_available_size: float = 100.0

    @classmethod
    def from_env(cls) -> "RiskConfig":
        def read(name: str, default: float) -> float:
            raw = os.getenv(name)
            if raw is None or not str(raw).strip():
                return default
            value = float(raw)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            return value

        return cls(
            kelly_scale=read("KELLY_FRACTION", 0.25),
            max_bet_fraction=read("MAX_BET_FRACTION", 0.0075),
            max_daily_risk_fraction=read("MAX_DAILY_RISK_FRACTION", 0.025),
            max_event_risk_fraction=read("MAX_EVENT_RISK_FRACTION", 0.01),
            max_drawdown_fraction=read("MAX_DRAWDOWN_FRACTION", 0.10),
            drawdown_warn_fraction=read("DRAWDOWN_WARN_FRACTION", 0.10),
            drawdown_circuit_fraction=read("DRAWDOWN_CIRCUIT_FRACTION", 0.15),
            max_correlation_group_fraction=read("MAX_CORRELATION_GROUP_FRACTION", 0.05),
            max_depth_fraction=read("MAX_DEPTH_FRACTION", 0.10),
            min_edge=read("MIN_EDGE", 0.025),
            min_confidence=read("MIN_CONFIDENCE", 0.60),
            max_spread=read("MAX_SPREAD", 0.03),
            min_available_size=read("MIN_AVAILABLE_SIZE", 100.0),
        )


@dataclass
class RiskBudgetLedger:
    """Accumulates committed stake fractions within one daily run.

    The daily cap, same-event cap, and same correlation-group cap are all enforced
    through this object by computing an effective per-bet cap before calling
    ``recommend()`` and then committing the actual returned ``stake_fraction``.
    """
    config: RiskConfig
    bankroll: float
    daily_committed: float = 0.0
    event_committed: dict[str, float] = field(default_factory=dict)
    group_committed: dict[str, float] = field(default_factory=dict)
    breaker_reason: str | None = None
    drawdown_level: str = "normal"
    paper_mode: bool = True
    warn_reason: str | None = None

    def load_prior(self, path: str | None, report_date: str) -> None:
        if not path:
            return
        try:
            from .paper_store import committed_exposure
            daily, events = committed_exposure(path, report_date, self.bankroll)
        except Exception:
            # A corrupt/unreadable ledger should not silently blow up the daily run,
            # but it also must not make the system think it has zero exposure.
            self.breaker_reason = "paper ledger unavailable; risk budget could not be restored"
            return
        self.daily_committed = max(self.daily_committed, daily)
        for event_id, fraction in events.items():
            self.event_committed[event_id] = max(self.event_committed.get(event_id, 0.0), fraction)

    def remaining_daily(self) -> float:
        return max(0.0, self.config.max_daily_risk_fraction - self.daily_committed)

    def remaining_event(self, event_id: str) -> float:
        used = self.event_committed.get(event_id, 0.0)
        return max(0.0, self.config.max_event_risk_fraction - used)

    def remaining_group(self, group_key: str) -> float:
        used = self.group_committed.get(group_key, 0.0)
        return max(0.0, self.config.max_correlation_group_fraction - used)

    def cap_for(self, event_id: str, group_key: str | None = None) -> float:
        if self.breaker_reason:
            return 0.0
        cap = min(self.config.max_bet_fraction, self.remaining_daily(), self.remaining_event(event_id))
        if group_key:
            cap = min(cap, self.remaining_group(group_key))
        cap = max(0.0, cap)
        if self.drawdown_level == "warn":
            cap *= 0.5
        elif self.drawdown_level == "circuit":
            cap = 0.0 if not self.paper_mode else cap * 0.5
        return max(0.0, cap)

    def exhausted_reasons(self, event_id: str, group_key: str | None = None) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.breaker_reason:
            reasons.append(self.breaker_reason)
        if self.warn_reason:
            reasons.append(self.warn_reason)
        elif self.drawdown_level == "warn":
            reasons.append("account drawdown warn threshold triggered")
        elif self.drawdown_level == "circuit" and self.paper_mode:
            reasons.append("virtual account drawdown reached circuit threshold; paper mode halves exposure")
        if self.remaining_daily() <= 0:
            reasons.append("daily risk budget exhausted")
        if self.remaining_event(event_id) <= 0:
            reasons.append("event risk budget exhausted")
        if group_key and self.remaining_group(group_key) <= 0:
            reasons.append("correlation-group risk budget exhausted")
        return tuple(reasons)

    def commit(self, event_id: str, group_key: str | None, stake_fraction: float) -> None:
        fraction = max(0.0, float(stake_fraction))
        self.daily_committed += fraction
        self.event_committed[event_id] = self.event_committed.get(event_id, 0.0) + fraction
        if group_key:
            self.group_committed[group_key] = self.group_committed.get(group_key, 0.0) + fraction


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
    max_depth_fraction: float = 0.10,
    risk_reasons: tuple[str, ...] = (),
) -> Recommendation:
    """Risk-capped recommendation with uncertainty shrinkage toward the market.

    ``max_bet_fraction`` is the caller-computed effective per-bet cap (already
    reduced for daily/event/group budget constraints).  ``risk_reasons`` lets the
    caller inject the budget/breaker reasons that caused that cap.
    """
    if bankroll < 0 or not 0 <= confidence <= 1:
        raise ValueError("invalid bankroll or confidence")
    market_probability = decimal_implied_probability(decimal_odds)
    # Low-confidence models should not manufacture large edges.
    shrunk_probability = market_probability + confidence * (model_probability - market_probability)
    raw_edge = model_probability - market_probability
    edge = shrunk_probability - market_probability
    net_edge = edge - max(0.0, estimated_cost)
    ev = shrunk_probability * decimal_odds - 1 - max(0.0, estimated_cost)
    raw_fraction = kelly_fraction(shrunk_probability, decimal_odds) * kelly_scale
    stake_fraction = min(max(0.0, max_bet_fraction), max(0.0, raw_fraction))
    reasons: list[str] = [
        f"uncertainty-adjusted edge={edge:.2%}",
        f"cost-adjusted edge={net_edge:.2%}",
        f"net EV={ev:.2%}",
    ]
    reasons.extend(risk_reasons)
    if max_bet_fraction <= 0:
        reasons.append("risk cap allows zero stake")

    # Capacity check: never consume more than a configurable fraction of visible depth.
    if (stake_fraction > 0 and available_size is not None and max_depth_fraction > 0
            and bankroll > 0 and available_size >= 0):
        depth_fraction = (available_size * max_depth_fraction) / bankroll
        if stake_fraction > depth_fraction:
            reasons.append(f"stake capped to {max_depth_fraction:.0%} of available depth")
            stake_fraction = max(0.0, depth_fraction)

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
            or market_quality_missing or spread_too_wide or liquidity_too_low or stake_fraction <= 0):
        action, stake_fraction = "NO_BET", 0.0
    else:
        action = "BET"
    return Recommendation(
        event_id=event_id,
        outcome=outcome,
        action=action,
        # Keep the independently-produced probability visible.  Confidence
        # shrinkage is a risk decision, not a model prediction.
        model_probability=model_probability,
        market_probability=market_probability,
        decision_probability=shrunk_probability,
        raw_edge=raw_edge,
        edge=edge,
        expected_value=ev,
        stake=round(bankroll * stake_fraction, 2),
        stake_fraction=stake_fraction,
        confidence=confidence,
        reasons=tuple(reasons),
        decision=f"BUY {outcome.upper()}" if action == "BET" else "NO TRADE",
    )

PAPER_STAKE_FRACTION = 0.005
PAPER_MIN_EDGE = 0.015
PAPER_MIN_EV = 0.05
PAPER_OPPOSITE_DIRECTION_MIN_EV = 0.10


def paper_recommend(
    *,
    event_id: str,
    outcome: str,
    model_probability: float,
    decimal_odds: float,
    bankroll: float,
    estimated_cost: float = 0.0,
    max_bet_fraction: float = PAPER_STAKE_FRACTION,
    direction_match: bool | None = None,
    risk_reasons: tuple[str, ...] = (),
) -> Recommendation:
    """Recommend a paper bet under the cold-start paper-trading rules.

    Paper mode is intentionally looser than real-money mode.  It does not use
    Kelly sizing and does not require a passed ROI acceptance or a complete
    lineup; instead it records a fixed 0.5% stake whenever the edge and EV
    thresholds pass.  The daily risk cap is still enforced by the caller
    through ``max_bet_fraction``.
    """
    if bankroll < 0 or not 0 <= model_probability <= 1 or decimal_odds <= 1:
        raise ValueError("invalid probability or odds")
    market_probability = decimal_implied_probability(decimal_odds)
    # Paper mode does not shrink toward the market: lineup uncertainty is
    # recorded as a field, not used to discard the sample.
    edge = model_probability - market_probability
    net_edge = edge - max(0.0, estimated_cost)
    ev = model_probability * decimal_odds - 1 - max(0.0, estimated_cost)
    if direction_match is None:
        direction_match = market_probability >= 0.5
    if max_bet_fraction <= 0:
        stake_fraction = 0.0
    else:
        stake_fraction = min(PAPER_STAKE_FRACTION, max_bet_fraction)
    reasons: list[str] = [
        f"paper edge={edge:.2%}",
        f"paper cost-adjusted edge={net_edge:.2%}",
        f"paper net EV={ev:.2%}",
        "paper mode fixed 0.5% stake",
    ]
    reasons.extend(risk_reasons)
    if 0 < stake_fraction < PAPER_STAKE_FRACTION:
        reasons.append("paper stake reduced to available risk budget")
    if not direction_match:
        reasons.append("prediction direction opposes market favorite")
    if net_edge <= PAPER_MIN_EDGE:
        reasons.append("paper cost-adjusted edge below 1.5%")
    if ev <= PAPER_MIN_EV:
        reasons.append("paper net EV below 5%")
    if not direction_match and ev <= PAPER_OPPOSITE_DIRECTION_MIN_EV:
        reasons.append("opposite-direction paper bet requires EV>10%")
    if (not math.isfinite(ev) or net_edge <= PAPER_MIN_EDGE or ev <= PAPER_MIN_EV
            or stake_fraction <= 0 or (not direction_match and ev <= PAPER_OPPOSITE_DIRECTION_MIN_EV)):
        action, stake_fraction = "NO_BET", 0.0
    else:
        action = "BET"
    return Recommendation(
        event_id=event_id,
        outcome=outcome,
        action=action,
        model_probability=model_probability,
        market_probability=market_probability,
        decision_probability=model_probability,
        raw_edge=edge,
        edge=edge,
        expected_value=ev,
        stake=round(bankroll * stake_fraction, 2),
        stake_fraction=stake_fraction,
        confidence=1.0,
        reasons=tuple(reasons),
        decision=f"BUY {outcome.upper()}" if action == "BET" else "NO TRADE",
    )
