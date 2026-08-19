"""Generate frozen LoL player-name and LoL/CS2 recent-form sidecars.

The sidecars are committed so the daily inference path can enrich pre-match
push messages without reading ``data/external`` or retraining models.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from prediction_agent.entities import canonical_team

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"


def _parse_lol_time(value: str):
    value = value.strip().replace("Z", "+00:00")
    if value.endswith("+00:00") and len(value) > 10 and value[10] != "T":
        value = value[:10] + "T" + value[10:]
    from datetime import datetime
    for fmt in ("%m/%d/%y %H:%M", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return datetime.fromisoformat(value)


def _recent_from_lol_csvs(paths) -> dict[str, dict[str, int]]:
    roles = {"top", "jng", "mid", "bot", "sup"}
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
        "cs2": {},
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
