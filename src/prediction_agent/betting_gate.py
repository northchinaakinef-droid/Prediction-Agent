"""Cold-start betting gate for virtual accumulation and real-money staging.

This module intentionally only reads the paper ledger and row dictionaries.
It never touches a wallet, exchange SDK, or private key, and real-money
suggestions remain gated by ``real_money_approved``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


VIRTUAL_ACCUMULATION_TARGET = 100
VIRTUAL_ROI_FLOOR = -0.30
REAL_RECENT_ROI_FLOOR = -0.15
REAL_RECENT_WINDOW = 30
VIRTUAL_EV_THRESHOLD = 0.05


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def should_place_virtual_bet(match: dict[str, Any]) -> tuple[bool, str]:
    """Return whether a row qualifies for the cold-start virtual ledger.

    The document requires all four gates to be met simultaneously:
    complete lineup, MATCHED schedule, EV > 5%, and model/market direction
    alignment.
    """
    if str(match.get("lineup_status") or "") != "完整":
        return False, "阵容不完整"
    if str(match.get("market_mapping_status") or "").upper() != "MATCHED":
        return False, "赛程未命中"
    if bool(match.get("market_started")):
        return False, "比赛已开始或开赛时间无法核验"
    ev = _as_float(match.get("expected_value"))
    if ev is None or ev <= VIRTUAL_EV_THRESHOLD:
        return False, f"EV={ev if ev is None else ev:.1%} 未超过阈值"
    if not bool(match.get("direction_match")):
        return False, "模型方向与市场方向不一致"
    return True, "符合虚拟下注条件"


def can_place_real_bet(match: dict[str, Any], db_path: str | Path | None = None) -> tuple[bool, str]:
    """Return whether real-money advice is allowed under the staged gate.

    ``real_money_approved`` is the final safety lockbox.  The ROI acceptance
    hard-gate is replaced by the virtual-accumulation count and a rolling real
    ROI circuit breaker below.
    """
    from .paper_store import calc_roi, count_virtual_bets

    if not bool(match.get("real_money_approved")):
        return False, "真实资金锁箱未通过，仍为虚拟下注"

    if not db_path:
        return False, "缺少虚拟投注数据库路径"

    virtual_count = count_virtual_bets(db_path)
    if virtual_count < VIRTUAL_ACCUMULATION_TARGET:
        return False, f"虚拟积累期（{virtual_count}/{VIRTUAL_ACCUMULATION_TARGET}），本场记录为虚拟下注"

    virtual_roi = calc_roi(db_path, bet_type="virtual")
    if virtual_roi is not None and virtual_roi < VIRTUAL_ROI_FLOOR:
        return False, f"虚拟期 ROI={virtual_roi:.1%}，未达解锁阈值，继续虚拟积累"

    recent_roi = calc_roi(db_path, bet_type="real", last_n=REAL_RECENT_WINDOW)
    if recent_roi is not None and recent_roi < REAL_RECENT_ROI_FLOOR:
        return False, f"最近{REAL_RECENT_WINDOW}场真实ROI={recent_roi:.1%}，触发熔断，暂停真实下注"

    return True, "通过验收"


def bet_status(match: dict[str, Any], db_path: str | Path | None = None) -> str:
    """Return the user-facing betting status for one recommendation row."""
    virtual_ok, _ = should_place_virtual_bet(match)
    if virtual_ok:
        real_ok, _ = can_place_real_bet(match, db_path)
        return "真实建议" if real_ok else "虚拟下注"
    return "跳过"
