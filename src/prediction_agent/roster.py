from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class RosterConfidence(str, Enum):
    CONFIRMED = "CONFIRMED"
    EXPECTED = "EXPECTED"
    HISTORICAL = "HISTORICAL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RosterStatus:
    players: tuple[str, ...]
    status: RosterConfidence
    source: str
    updated_at: datetime | None

    @property
    def explanatory_note(self) -> str:
        if self.status == RosterConfidence.HISTORICAL:
            return "首发未确认，当前采用历史/预期阵容。"
        if self.status == RosterConfidence.UNKNOWN:
            return "当前阵容未知。"
        return f"阵容状态：{self.status.value}（来源：{self.source}）。"


def historical_roster(players: tuple[str, ...], updated_at: datetime | None = None) -> RosterStatus:
    return RosterStatus(players, RosterConfidence.HISTORICAL if players else RosterConfidence.UNKNOWN,
                        "latest_completed_match", updated_at)


def lineup_changed(previous: RosterStatus, current: RosterStatus) -> bool:
    return bool(previous.players and current.players and previous.players != current.players)
