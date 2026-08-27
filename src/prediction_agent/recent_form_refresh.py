"""Independent CS2 recent-form refresh; never called by inference."""
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from .entities import canonical_team
from .providers.live_data import Bo3Cs2Provider


@dataclass(frozen=True)
class Cs2MatchResult:
    match_id: str
    played_at: datetime
    team_a: str
    team_b: str
    winner: str
    tier_rank: int | None = None


class Cs2RecentFormProvider(Protocol):
    name: str
    def fetch_finished(self, start: date, end: date) -> Iterable[Cs2MatchResult]: ...


class Bo3RecentFormProvider:
    """BO3 result adapter. Historical coverage deliberately includes every tier.

    ``Bo3Cs2Provider._in_scope`` is a schedule policy (Tier 1/2 by default), not
    a data-validity rule. Applying it here used to discard genuine lower-tier
    completed results even when PandaScore discovered those teams upcoming.
    """
    name = "bo3"

    def __init__(self, client: Bo3Cs2Provider | None = None):
        self.client = client or Bo3Cs2Provider()

    def fetch_finished(self, start: date, end: date) -> Iterable[Cs2MatchResult]:
        seen: set[str] = set()
        day = start
        while day <= end:
            for row in self.client.matches(day, "finished"):
                match_id = str(row.get("id") or "")
                if not match_id or match_id in seen:
                    continue
                team_a = str((row.get("team1") or {}).get("name") or "").strip()
                team_b = str((row.get("team2") or {}).get("name") or "").strip()
                try:
                    score_a = int(row.get("team1_score"))
                    score_b = int(row.get("team2_score"))
                    played = datetime.fromisoformat(str(row.get("start_date")).replace("Z", "+00:00"))
                except (TypeError, ValueError):
                    continue
                if not team_a or not team_b or score_a == score_b:
                    continue
                if played.tzinfo is None:
                    played = played.replace(tzinfo=timezone.utc)
                seen.add(match_id)
                rank = (row.get("tournament") or {}).get("tier_rank")
                try: rank = int(rank) if rank is not None else None
                except (TypeError, ValueError): rank = None
                yield Cs2MatchResult(match_id, played.astimezone(timezone.utc), team_a, team_b,
                                     team_a if score_a > score_b else team_b, rank)
            day += timedelta(days=1)

    def fetch_scheduled(self, start: date, end: date) -> list[dict[str, Any]]:
        """Return identities only; this is refresh-time coverage telemetry."""
        seen: set[str] = set(); result = []
        day = start
        while day <= end:
            for row in self.client.matches(day, "upcoming,current"):
                match_id = str(row.get("id") or "")
                if not match_id or match_id in seen:
                    continue
                a = str((row.get("team1") or {}).get("name") or "").strip()
                b = str((row.get("team2") or {}).get("name") or "").strip()
                if not a or not b: continue
                rank = (row.get("tournament") or {}).get("tier_rank")
                try: rank = int(rank) if rank is not None else None
                except (TypeError, ValueError): rank = None
                seen.add(match_id); result.append({"match_id": match_id, "team_a": a, "team_b": b,
                                                   "tier_rank": rank})
            day += timedelta(days=1)
        return result


def build_cs2_recent_form(matches: Iterable[Cs2MatchResult], window: int = 10,
                          active_days: int = 30, now: datetime | None = None) -> tuple[dict, dict]:
    if window <= 0:
        raise ValueError("window must be positive")
    by_team: dict[str, list[tuple[datetime, bool, str]]] = defaultdict(list)
    match_ids: set[str] = set()
    latest: datetime | None = None
    source_teams: set[str] = set()
    lower_tier_teams: set[str] = set()
    latest_by_team: dict[str, datetime] = {}
    for match in matches:
        if match.match_id in match_ids or match.team_a == match.team_b:
            continue
        winner = canonical_team("cs2", match.winner)
        team_a = canonical_team("cs2", match.team_a)
        team_b = canonical_team("cs2", match.team_b)
        if winner not in {team_a, team_b}:
            continue
        played = match.played_at.astimezone(timezone.utc)
        match_ids.add(match.match_id)
        source_teams.update((team_a, team_b))
        if match.tier_rank is not None and match.tier_rank > 2:
            lower_tier_teams.update((team_a, team_b))
        latest = max(latest, played) if latest else played
        latest_by_team[team_a] = max(latest_by_team.get(team_a, played), played)
        latest_by_team[team_b] = max(latest_by_team.get(team_b, played), played)
        by_team[team_a].append((played, winner == team_a, match.match_id))
        by_team[team_b].append((played, winner == team_b, match.match_id))
    recent: dict[str, dict] = {}
    for team, rows in by_team.items():
        selected = sorted(rows, key=lambda item: (item[0], item[2]))[-window:]
        if not selected:
            continue
        recent[team] = {"wins": sum(won for _, won, _ in selected),
                        "losses": sum(not won for _, won, _ in selected),
                        "last_n": len(selected),
                        "latest_match_at": selected[-1][0].isoformat()}
    reference = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    active = {team for team, played in latest_by_team.items()
              if played >= reference - timedelta(days=active_days)}
    metadata = {"record_count": len(match_ids), "team_count": len(recent),
                "source_team_count": len(source_teams),
                "active_team_count": len(active), "active_days": active_days,
                "lower_tier_covered_team_count": len(lower_tier_teams),
                "latest_match_time": latest.isoformat() if latest else None}
    return recent, metadata


def _atomic_json_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False)
    temp_path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        # The refresh runs outside the container (normally as a systemd
        # oneshot), while inference reads the bind-mounted frozen artifact as
        # uid 10001. NamedTemporaryFile defaults to 0600, so normalize the
        # completed file before the atomic replace.
        os.chmod(temp_path, 0o644)
        os.replace(temp_path, path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise


def refresh_cs2_recent_form(*, provider: Cs2RecentFormProvider, artifact_path: str | Path,
                            status_path: str | Path | None = None, now: datetime | None = None,
                            lookback_days: int = 120, expected_team_count: int | None = None) -> dict:
    """Refresh atomically; any failure preserves the last usable artifact."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    target = Path(artifact_path)
    status_target = Path(status_path) if status_path else target.with_name("recent_form_refresh_status.json")
    previous_status: dict[str, Any] = {}
    try:
        loaded_status = json.loads(status_target.read_text(encoding="utf-8"))
        previous_status = loaded_status if isinstance(loaded_status, dict) else {}
    except (OSError, json.JSONDecodeError):
        pass
    report = {"attempted_at": now.isoformat(), "provider": provider.name,
              "artifact_path": str(target), "replaced": False,
              "last_success": previous_status.get("last_success")}
    try:
        matches = list(provider.fetch_finished((now - timedelta(days=lookback_days)).date(), now.date()))
        cs2, metadata = build_cs2_recent_form(matches, now=now)
        if (not cs2 or not metadata["record_count"] or not metadata["team_count"] or
                metadata["team_count"] != len(cs2) or not metadata["latest_match_time"]):
            raise ValueError("EMPTY_DATASET")
        existing: dict = {}
        if target.exists():
            try:
                loaded = json.loads(target.read_text(encoding="utf-8"))
                existing = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                existing = {}
        scheduled = []
        fetch_scheduled = getattr(provider, "fetch_scheduled", None)
        if callable(fetch_scheduled):
            scheduled = list(fetch_scheduled(now.date(), (now + timedelta(days=7)).date()))
        scheduled_teams = {canonical_team("cs2", str(row.get(side) or ""))
                           for row in scheduled for side in ("team_a", "team_b") if row.get(side)}
        uncovered = sorted(team for team in scheduled_teams if team not in cs2)
        lower_uncovered = sorted({canonical_team("cs2", str(row.get(side) or ""))
                                  for row in scheduled if (row.get("tier_rank") or 0) > 2
                                  for side in ("team_a", "team_b") if row.get(side)} - set(cs2))
        coverage = (len(cs2) / expected_team_count) if expected_team_count else None
        meta = {**metadata, "generated_at": now.isoformat(), "provider": provider.name,
                "lookback_days": lookback_days, "coverage": coverage,
                "total_known_teams": expected_team_count,
                "covered_teams": len(cs2), "scheduled_team_count": len(scheduled_teams),
                "scheduled_covered_team_count": len(scheduled_teams) - len(uncovered),
                "uncovered_scheduled_teams": uncovered,
                "lower_tier_uncovered_teams": lower_uncovered,
                "upcoming_match_count": len(scheduled),
                "upcoming_match_covered_count": sum(
                    canonical_team("cs2", str(row.get("team_a") or "")) in cs2 and
                    canonical_team("cs2", str(row.get("team_b") or "")) in cs2 for row in scheduled)}
        payload = {**existing, "lol": existing.get("lol", {}), "cs2": cs2,
                   "_metadata": {**(existing.get("_metadata") or {}), "generated_at": now.isoformat(),
                                  "cs2": meta}}
        _atomic_json_write(target, payload)
        report.update({"status": "OK", "replaced": True, "last_success": now.isoformat(),
                       "error_category": None, **meta})
    except Exception as error:
        usable = False
        try:
            old = json.loads(target.read_text(encoding="utf-8"))
            usable = bool(isinstance(old, dict) and old.get("cs2"))
        except (OSError, json.JSONDecodeError):
            pass
        text = str(error).casefold()
        category = ("HTTP_403" if "403" in text else "HTTP_404" if "404" in text else
                    "HTTP_429" if "429" in text else "TIMEOUT" if "timeout" in text else
                    "CONNECTION" if any(term in text for term in ("connection", "name resolution", "reset")) else
                    "JSON_DECODE" if isinstance(error, json.JSONDecodeError) else
                    "EMPTY_DATASET" if str(error) == "EMPTY_DATASET" else type(error).__name__.upper())
        report.update({"status": "EMPTY" if str(error) == "EMPTY_DATASET" else "ERROR",
                       "error": type(error).__name__, "error_category": category,
                       "fallback_artifact_usable": usable})
    _atomic_json_write(status_target, report)
    return report


def model_team_count(path: str | Path) -> int:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return len((payload.get("model") or {}).get("team_ratings") or {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return 0
