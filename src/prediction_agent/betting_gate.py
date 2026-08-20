"""Cold-start betting gate for virtual accumulation and real-money staging.

This module intentionally only reads the paper ledger and row dictionaries.
It never touches a wallet, exchange SDK, or private key. ``真实建议`` is a
user-facing research status, not permission for this repository to trade.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


VIRTUAL_ACCUMULATION_TARGET = 100
VIRTUAL_ROI_FLOOR = 0.0
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
    """Return whether the research output may be labelled ``真实建议``.

    Promotion requires 100 settled virtual bets and non-negative virtual ROI.
    It does not mutate ``real_money_approved`` or authorize trade execution.
    """
    from .paper_store import calc_roi, count_settled_virtual_bets

    if not db_path:
        return False, "缺少虚拟投注数据库路径"

    virtual_count = count_settled_virtual_bets(db_path)
    if virtual_count < VIRTUAL_ACCUMULATION_TARGET:
        remaining = VIRTUAL_ACCUMULATION_TARGET - virtual_count
        return False, (f"虚拟第{virtual_count}场/{VIRTUAL_ACCUMULATION_TARGET}场，"
                       f"距真实建议还差{remaining}场")

    virtual_roi = calc_roi(db_path, bet_type="virtual")
    if virtual_roi is None:
        return False, "虚拟已完成100场，ROI暂不可用，当前继续显示虚拟下注"
    if virtual_roi < VIRTUAL_ROI_FLOOR:
        return False, f"虚拟已完成{virtual_count}场，虚拟ROI={virtual_roi:.1%}，当前继续显示虚拟下注"

    return True, f"虚拟已完成{virtual_count}场且ROI={virtual_roi:.1%}，升级为真实建议"


def bet_status(match: dict[str, Any], db_path: str | Path | None = None) -> str:
    """Return the user-facing betting status for one recommendation row."""
    virtual_ok, _ = should_place_virtual_bet(match)
    if virtual_ok:
        real_ok, _ = can_place_real_bet(match, db_path)
        return "真实建议" if real_ok else "虚拟下注"
    return "跳过"
