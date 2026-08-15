"""Canonical Polymarket transaction-cost model.

All three costed surfaces (live recommendations, walk-forward validation, and
paper-ledger reconciliation) must import these functions so that a strategy is
validated and then executed under the same cost assumptions.
"""
from __future__ import annotations

import os


def _env_fee_rate() -> float:
    value = os.getenv("POLYMARKET_FEE_RATE", "0.03")
    rate = float(value)
    if rate < 0:
        raise ValueError("POLYMARKET_FEE_RATE must be non-negative")
    return rate


def estimate_cost_rate(price: float, fee_rate: float | None = None) -> float:
    """Return the cost of a trade as a fraction of the notional stake.

    This mirrors Polymarket's documented fee schedule for a binary outcome:
    ``fee_rate * (1 - price)``.  ``price`` is the execution price for the side
    being bought (best ask), in ``(0, 1)``.
    """
    if not 0 < price < 1:
        raise ValueError("price must be strictly between zero and one")
    rate = _env_fee_rate() if fee_rate is None else fee_rate
    if rate < 0:
        raise ValueError("fee_rate must be non-negative")
    return max(0.0, rate * (1 - price))


def estimate_cost(price: float, size: float, fee_rate: float | None = None) -> float:
    """Return the cost of a trade in absolute USDC for a given ``size`` stake."""
    if size < 0:
        raise ValueError("size must be non-negative")
    return size * estimate_cost_rate(price, fee_rate)
