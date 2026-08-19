"""Generate frozen LoL player-name and LoL/CS2 recent-form sidecars.

The sidecars are committed so the daily inference path can enrich pre-match
push messages without reading ``data/external`` or retraining models.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from prediction_agent.entities import canonical_team

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"

LOL_ROLES = {"top", "jng", "mid", "bot", "sup"}


def _parse_lol_time(value: str) -> datetime:
    value = value.strip().replace("Z", "+00:00")
    if value.endswith("+00:00") and len(value) > 10 and value[10] != "T":
        value = value[:10] + "T" + value[10:]
    for fmt in ("%m/%d/%y %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(value)


def _recent_from_results(by_team: dict[str, list[tuple[str, bool]]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for team, results in by_team.items():
        recent = sorted(results, key=lambda item: item[0])[-10:]
        out[team] = {
            "wins": sum(1 for _, won in recent if won),
            "losses": sum(1 for _, won in recent if not won),
            "last_n": len(recent),
        }
    return out


def _recent_from_lol_csvs(paths) -> dict[str, dict[str, int]]:
    """Read Oracle's Elixir CSV rows and return recent 10 BO3 form by teamname."""
    by_team: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("position") or "").casefold() not in LOL_ROLES:
                    continue
                game_id = str(row.get("gameid") or "").strip()
                team = str(row.get("teamname") or row.get("teamid") or "").strip()
                result = str(row.get("result") or "")
                date = str(row.get("date") or "").strip()
                if not game_id or not team or result not in {"0", "1", "0.0", "1.0"} or not date:
                    continue
                try:
                    played = _parse_lol_time(date)
                except ValueError:
                    continue
                won = bool(int(float(result)))
                by_team[canonical_team("lol", team)].append((played.isoformat(), won))
    return _recent_from_results(by_team)


def _recent_from_cs2_csvs() -> dict[str, dict[str, int]]:
    """Read CS2 history CSV files under data/external when available.

    If no CS2 CSV exists, keep this non-fatal and return an empty dict so the
    sidecar generation still succeeds for LoL.
    """
    candidates = []
    for pattern in (
        "data/external/**/*cs2*.csv",
        "data/external/**/*CS2*.csv",
        "data/external/valve_cs2_vrs/**/*.csv",
    ):
        candidates.extend(ROOT.glob(pattern))

    paths = sorted({path.resolve() for path in candidates if path.is_file()})
    if not paths:
        print(
            "WARNING: no CS2 history CSV files found under data/external; "
            "CS2 recent form will be empty.",
            file=sys.stderr,
        )
        return {}

    by_team: dict[str, list[tuple[str, bool]]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            if not {"team1Name", "team2Name"}.issubset(fields):
                print(f"WARNING: unsupported CS2 CSV schema, skipped: {path}", file=sys.stderr)
                continue
            for row in reader:
                team1 = str(row.get("team1Name") or "").strip()
                team2 = str(row.get("team2Name") or "").strip()
                raw_time = str(row.get("matchStartTime") or row.get("start_time") or row.get("date") or "").strip()
                raw_winner = str(row.get("winningTeam") or row.get("winner") or "").strip()
                if not team1 or not team2 or not raw_time or raw_winner not in {"0", "1", "team1", "team2"}:
                    continue
                try:
                    if raw_time.isdigit():
                        played = datetime.fromtimestamp(int(raw_time))
                    else:
                        played = _parse_lol_time(raw_time)
                except (ValueError, OSError):
                    continue
                team1_won = raw_winner in {"1", "team1"}
                by_team[canonical_team("cs2", team1)].append((played.isoformat(), team1_won))
                by_team[canonical_team("cs2", team2)].append((played.isoformat(), not team1_won))
    return _recent_from_results(by_team)


def _lol_csv_paths() -> list[Path]:
    candidates = [
        ROOT / "data" / "external" / "lol_analysis_2025" / "2025_LoL_esports_match_data_from_OraclesElixir.csv",
    ]
    return [path for path in candidates if path.exists()]


def main() -> None:
    player_names: dict[str, str] = {}
    lol_csvs = _lol_csv_paths()
    for path in lol_csvs:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                player_id = str(row.get("playerid") or "").strip()
                player_name = str(row.get("playername") or "").strip()
                if player_id and player_name:
                    player_names[player_id] = player_name

    recent_form = {
        "lol": _recent_from_lol_csvs(lol_csvs),
        "cs2": _recent_from_cs2_csvs(),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "player_names.json").write_text(
        json.dumps(player_names, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "recent_form.json").write_text(
        json.dumps(recent_form, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "lol_csvs": len(lol_csvs), "players": len(player_names),
        "lol_teams": len(recent_form["lol"]), "cs2_teams": len(recent_form["cs2"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
