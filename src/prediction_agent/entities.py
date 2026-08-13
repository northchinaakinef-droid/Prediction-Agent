from __future__ import annotations


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


def canonical_team(sport: str, name: str) -> str:
    cleaned = " ".join(name.split())
    if sport == "nba":
        return NBA.get(cleaned, cleaned)
    if sport == "cs2":
        return CS2.get(cleaned, cleaned)
    return cleaned
