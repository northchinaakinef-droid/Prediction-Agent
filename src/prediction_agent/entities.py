from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path


NBA = {
    "76ers": "Philadelphia 76ers", "Bucks": "Milwaukee Bucks", "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers", "Celtics": "Boston Celtics", "Clippers": "Los Angeles Clippers",
    "Grizzlies": "Memphis Grizzlies", "Hawks": "Atlanta Hawks", "Heat": "Miami Heat",
    "Hornets": "Charlotte Hornets", "Jazz": "Utah Jazz", "Kings": "Sacramento Kings",
    "Knicks": "New York Knicks", "Lakers": "Los Angeles Lakers", "Magic": "Orlando Magic",
    "Mavericks": "Dallas Mavericks", "Nets": "Brooklyn Nets", "Nuggets": "Denver Nuggets",
    "Pacers": "Indiana Pacers", "Pelicans": "New Orleans Pelicans", "Pistons": "Detroit Pistons",
    "Raptors": "Toronto Raptors", "Rockets": "Houston Rockets", "Spurs": "San Antonio Spurs",
    "Suns": "Phoenix Suns", "Thunder": "Oklahoma City Thunder", "Timberwolves": "Minnesota Timberwolves",
    "Trail Blazers": "Portland Trail Blazers", "Warriors": "Golden State Warriors",
    "Wizards": "Washington Wizards", "LA Clippers": "Los Angeles Clippers",
}

CS2 = {
    "NAVI": "Natus Vincere", "NaVi": "Natus Vincere", "Team Vitality": "Vitality",
    "Team Spirit": "Spirit", "FaZe Clan": "FaZe", "G2 Esports": "G2",
    "Team Liquid": "Liquid", "Team Falcons": "Falcons",
}


def normalized_name(name: str) -> str:
    value = unicodedata.normalize("NFKD", name).casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


@lru_cache(maxsize=1)
def team_entities() -> list[dict]:
    path = Path(__file__).resolve().parents[2] / "config" / "team_aliases.json"
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")))


def resolve_team(sport: str, name: str) -> tuple[str, str, float]:
    """Resolve a team using canonical names/aliases; never auto-accept weak fuzzy matches."""
    cleaned = " ".join(name.split())
    needle = normalized_name(cleaned)
    candidates = []
    for entity in team_entities():
        if entity.get("sport") != sport:
            continue
        names = [entity["team_id"], entity["canonical_name"], *entity.get("aliases", [])]
        if needle in {normalized_name(value) for value in names}:
            return str(entity["canonical_name"]), "alias", 1.0
        score = max(SequenceMatcher(None, needle, normalized_name(value)).ratio() for value in names)
        candidates.append((score, entity))
    if candidates:
        score, entity = max(candidates, key=lambda value: value[0])
        if score >= 0.92:
            return str(entity["canonical_name"]), "fuzzy", score
        if score >= 0.75:
            return cleaned, "review", score
    return cleaned, "normalized", 1.0


def canonical_team(sport: str, name: str) -> str:
    cleaned = " ".join(name.split())
    if sport == "nba":
        return NBA.get(cleaned, cleaned)
    if sport == "cs2":
        return CS2.get(cleaned, cleaned)
    if sport == "lol":
        return resolve_team(sport, cleaned)[0]
    return cleaned
