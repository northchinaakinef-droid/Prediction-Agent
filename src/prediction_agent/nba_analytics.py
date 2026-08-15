"""Basketball post-game analytics used for NBA reviews and future model features.

These helpers intentionally do not mutate the production model.  They turn an
NBA.com box score into the same four-factors / pace / efficiency dimensions
professional analysts use, so every finished game can be stored as a
structured training sample for later walk-forward model iteration.
"""
from __future__ import annotations

import re
from typing import Any


def duration_seconds(value: object) -> float:
    """Parse an ISO-8601 duration such as ``PT14M34.00S`` into seconds."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().upper()
    if not text or text == "PT":
        return 0.0
    match = re.match(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$", text)
    if not match:
        return 0.0
    minutes = int(match.group(1) or 0)
    seconds = float(match.group(2) or 0)
    return minutes * 60 + seconds


def _f(stats: dict[str, Any], *fields: str) -> float:
    for field in fields:
        value = stats.get(field)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _first(stats: dict[str, Any], *fields: str) -> float:
    return _f(stats, *fields)


def four_factors(team_stats: dict[str, Any], opponent_stats: dict[str, Any]) -> dict[str, float]:
    """Return Dean Oliver's four factors for one side."""
    fgm = _f(team_stats, "fieldGoalsMade")
    fga = _f(team_stats, "fieldGoalsAttempted")
    three_pm = _f(team_stats, "threePointersMade")
    ftm = _f(team_stats, "freeThrowsMade")
    fta = _f(team_stats, "freeThrowsAttempted")
    tov = _first(team_stats, "turnoversTotal", "turnovers")
    orb = _f(team_stats, "reboundsOffensive")
    opp_drb = _f(opponent_stats, "reboundsDefensive")
    efg = (fgm + 0.5 * three_pm) / fga if fga else 0.0
    tov_pct = tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) else 0.0
    orb_pct = orb / (orb + opp_drb) if (orb + opp_drb) else 0.0
    ftr = ftm / fga if fga else 0.0
    return {
        "effective_field_goal_pct": round(efg, 4),
        "turnover_pct": round(tov_pct, 4),
        "offensive_rebound_pct": round(orb_pct, 4),
        "free_throw_rate": round(ftr, 4),
    }


def estimate_possessions(team_stats: dict[str, Any], opponent_stats: dict[str, Any]) -> float:
    """Estimate possessions using the standard Basketball-Reference approximation."""
    def component(team: dict[str, Any], opp: dict[str, Any]) -> float:
        fga = _f(team, "fieldGoalsAttempted")
        fgm = _f(team, "fieldGoalsMade")
        fta = _f(team, "freeThrowsAttempted")
        tov = _first(team, "turnoversTotal", "turnovers")
        orb = _f(team, "reboundsOffensive")
        opp_drb = _f(opp, "reboundsDefensive")
        orb_pct = orb / (orb + opp_drb) if (orb + opp_drb) else 0.0
        return fga + 0.44 * fta - 1.07 * orb_pct * (fga - fgm) + tov

    return round(0.5 * (component(team_stats, opponent_stats) + component(opponent_stats, team_stats)), 2)


def pace(possessions: float, game_seconds: float) -> float:
    """Pace = possessions per 48 minutes."""
    minutes = game_seconds / 60 if game_seconds else 0.0
    if minutes <= 0:
        return 0.0
    return round(possessions * 48 / minutes, 2)


def ratings(team_points: float, opponent_points: float, possessions: float) -> dict[str, float]:
    if possessions <= 0:
        return {"offensive_rating": 0.0, "defensive_rating": 0.0, "net_rating": 0.0}
    off = team_points / possessions * 100
    defense = opponent_points / possessions * 100
    return {
        "offensive_rating": round(off, 1),
        "defensive_rating": round(defense, 1),
        "net_rating": round(off - defense, 1),
    }


def boxscore_metrics(box: dict[str, Any]) -> dict[str, Any]:
    """Derive analyst-grade metrics from a normalized NbaBoxscoreProvider dict."""
    home = box.get("home_team", {})
    away = box.get("away_team", {})
    home_stats = home.get("statistics", {})
    away_stats = away.get("statistics", {})
    home_factors = four_factors(home_stats, away_stats)
    away_factors = four_factors(away_stats, home_stats)
    possessions = estimate_possessions(home_stats, away_stats)
    duration = duration_seconds(box.get("duration"))
    team_minutes = max(duration_seconds(home_stats.get("minutes")), duration_seconds(away_stats.get("minutes")))
    game_seconds = duration or team_minutes or (48 * 60)
    game_pace = pace(possessions, game_seconds)
    home_ratings = ratings(float(home.get("score") or 0), float(away.get("score") or 0), possessions)
    away_ratings = ratings(float(away.get("score") or 0), float(home.get("score") or 0), possessions)
    return {
        "home_four_factors": home_factors,
        "away_four_factors": away_factors,
        "possessions": possessions,
        "pace": game_pace,
        "game_seconds": round(game_seconds, 1),
        "home_ratings": home_ratings,
        "away_ratings": away_ratings,
    }


def decisive_factors(box: dict[str, Any]) -> list[str]:
    """Return concise Chinese decisive factors for an NBA box score."""
    metrics = boxscore_metrics(box)
    home = box.get("home_team", {})
    away = box.get("away_team", {})
    home_stats = home.get("statistics", {})
    away_stats = away.get("statistics", {})
    home_name = str(home.get("team_name") or "主队")
    away_name = str(away.get("team_name") or "客队")
    rows: list[tuple[float, str]] = []

    for label, home_value, away_value in (
        ("有效命中率", metrics["home_four_factors"]["effective_field_goal_pct"],
         metrics["away_four_factors"]["effective_field_goal_pct"]),
        ("进攻篮板率", metrics["home_four_factors"]["offensive_rebound_pct"],
         metrics["away_four_factors"]["offensive_rebound_pct"]),
        ("失误率", metrics["home_four_factors"]["turnover_pct"],
         metrics["away_four_factors"]["turnover_pct"]),
    ):
        diff = float(home_value) - float(away_value)
        if abs(diff) < 1e-9:
            continue
        # For turnover rate lower is better.
        better = home_name if ((diff < 0) if label == "失误率" else (diff > 0)) else away_name
        rows.append((abs(diff), f"{label} {diff:+.1%}（{better}占优）"))

    for label, home_value, away_value, unit in (
        ("内线得分", _f(home_stats, "pointsInThePaint"), _f(away_stats, "pointsInThePaint"), ""),
        ("快攻得分", _f(home_stats, "pointsFastBreak"), _f(away_stats, "pointsFastBreak"), ""),
        ("二次进攻得分", _f(home_stats, "pointsSecondChance"), _f(away_stats, "pointsSecondChance"), ""),
        ("利用失误得分", _f(home_stats, "pointsFromTurnovers"), _f(away_stats, "pointsFromTurnovers"), ""),
        ("替补得分", _f(home_stats, "benchPoints"), _f(away_stats, "benchPoints"), ""),
        ("篮板总数", _f(home_stats, "reboundsTotal"), _f(away_stats, "reboundsTotal"), ""),
        ("总失误", _f(home_stats, "turnoversTotal"), _f(away_stats, "turnoversTotal"), ""),
    ):
        diff = float(home_value) - float(away_value)
        if abs(diff) < 0.5:
            continue
        better = home_name if diff > 0 else away_name
        if label == "总失误":
            better = away_name if diff > 0 else home_name
        rows.append((abs(diff), f"{label} {home_value:.0f}-{away_value:.0f}（{better}占优）"))

    rows.sort(key=lambda row: row[0], reverse=True)
    return [text for _, text in rows[:5]]


def flatten_nba_sample(game_sample: dict[str, Any]) -> dict[str, float | int | None]:
    """Return flat numeric features for future NBA model training.

    This function is intentionally read-only and is not called during the live
    scan or the 06:30 report job.
    """
    metrics = game_sample.get("metrics", {})
    home_factors = metrics.get("home_four_factors", {})
    away_factors = metrics.get("away_four_factors", {})
    home_ratings = metrics.get("home_ratings", {})
    away_ratings = metrics.get("away_ratings", {})

    def val(mapping: dict[str, Any], key: str) -> float | int | None:
        value = mapping.get(key)
        return value if isinstance(value, (int, float)) else None

    return {
        "home_score": game_sample.get("home_score"),
        "away_score": game_sample.get("away_score"),
        "winner_side_a": 1 if game_sample.get("winner_side") == "a" else 0,
        "pace": val(metrics, "pace"),
        "possessions": val(metrics, "possessions"),
        "home_efg": val(home_factors, "effective_field_goal_pct"),
        "away_efg": val(away_factors, "effective_field_goal_pct"),
        "home_tov_pct": val(home_factors, "turnover_pct"),
        "away_tov_pct": val(away_factors, "turnover_pct"),
        "home_orb_pct": val(home_factors, "offensive_rebound_pct"),
        "away_orb_pct": val(away_factors, "offensive_rebound_pct"),
        "home_ftr": val(home_factors, "free_throw_rate"),
        "away_ftr": val(away_factors, "free_throw_rate"),
        "home_off_rtg": val(home_ratings, "offensive_rating"),
        "away_off_rtg": val(away_ratings, "offensive_rating"),
        "home_def_rtg": val(home_ratings, "defensive_rating"),
        "away_def_rtg": val(away_ratings, "defensive_rating"),
        "home_net_rtg": val(home_ratings, "net_rating"),
        "away_net_rtg": val(away_ratings, "net_rating"),
    }
