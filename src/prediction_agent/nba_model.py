"""NBA Elo with pregame-only home court and rest context."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable

from .lol_model import LolGame


@dataclass
class NbaModel:
    ratings: dict[str, float]
    games: dict[str, int]
    last_game_dates: dict[str, str]
    trained_through: str
    samples: int
    k_factor: float = 20.0
    home_advantage: float = 60.0
    rest_day_value: float = 8.0
    seasonal_regression: float = .15
    base_rating: float = 1500.0
    efficiency_ratings: dict[str, float] = field(default_factory=dict)
    margin_k_factor: float = 6.0
    margin_cap: float = 15.0
    efficiency_weight: float = 0.3

    def _combined_rating(self, team: str) -> float:
        base = self.ratings.get(team, self.base_rating)
        return base + self.efficiency_weight * self.efficiency_ratings.get(team, 0.0)

    def game_probability(self, home_team: str, away_team: str,
                         played_at: datetime | None = None) -> float:
        adjustment = self.home_advantage
        if played_at is not None:
            adjustment += self.rest_day_value * (
                _rest_days(self.last_game_dates.get(home_team), played_at) -
                _rest_days(self.last_game_dates.get(away_team), played_at)
            )
        home = self._combined_rating(home_team) + adjustment
        away = self._combined_rating(away_team)
        return 1 / (1 + 10 ** ((away - home) / 400))

    def explain(self, home_team: str, away_team: str,
                played_at: datetime | None = None) -> list[str]:
        reasons = [
            f"赛前强度：{home_team} {self.ratings.get(home_team, self.base_rating):.0f}，"
            f"{away_team} {self.ratings.get(away_team, self.base_rating):.0f}",
            f"效率修正：{home_team} {self.efficiency_ratings.get(home_team, 0.0):+.1f}，"
            f"{away_team} {self.efficiency_ratings.get(away_team, 0.0):+.1f}",
            f"主场修正：{self.home_advantage:.0f} Elo",
        ]
        if played_at is not None:
            home_rest = _rest_days(self.last_game_dates.get(home_team), played_at)
            away_rest = _rest_days(self.last_game_dates.get(away_team), played_at)
            reasons.append(f"休息天数：{home_team} {home_rest} 天，{away_team} {away_rest} 天")
        return reasons


def _rest_days(last_value: str | None, played_at: datetime) -> int:
    if not last_value:
        return 3
    days = (played_at.date() - datetime.fromisoformat(last_value).date()).days - 1
    return min(3, max(0, days))


DEFAULT_ARENA_LOCATIONS: dict[str, tuple[float, float]] = {
    "Los Angeles Lakers": (34.0430, -118.2673),
    "Boston Celtics": (42.3662, -71.0621),
    "Golden State Warriors": (37.7680, -122.3877),
    "Chicago Bulls": (41.8807, -87.6742),
    "Miami Heat": (25.7814, -80.1870),
}


def back_to_back(last_value: str | None, played_at: datetime) -> bool:
    if not last_value:
        return False
    return (played_at.date() - datetime.fromisoformat(last_value).date()).days == 1


def travel_distance(team_a: str, team_b: str,
                    locations: dict[str, tuple[float, float]] | None = None) -> float:
    locations = locations or DEFAULT_ARENA_LOCATIONS
    a = locations.get(team_a)
    b = locations.get(team_b)
    if a is None or b is None:
        return 0.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 3959.0 * 2 * math.asin(math.sqrt(h))


def nba_schedule_fatigue(team: str, last_game_date: str | None,
                         last_location: str | None, current_location: str | None,
                         played_at: datetime,
                         locations: dict[str, tuple[float, float]] | None = None) -> dict:
    locations = locations or DEFAULT_ARENA_LOCATIONS
    b2b = back_to_back(last_game_date, played_at)
    distance = travel_distance(last_location or team, current_location or team, locations)
    return {"back_to_back": b2b, "travel_distance_miles": distance}


def _new(k_factor: float, home_advantage: float, rest_day_value: float,
         seasonal_regression: float = .15, efficiency_weight: float = .3,
         margin_k_factor: float = 6.0, margin_cap: float = 15.0) -> NbaModel:
    return NbaModel({}, {}, {}, "", 0, k_factor, home_advantage,
                    rest_day_value, seasonal_regression, 1500.0, {},
                    margin_k_factor, margin_cap, efficiency_weight)


def _update(model: NbaModel, game: LolGame) -> float:
    probability = min(1 - 1e-8, max(1e-8, model.game_probability(
        game.team_a, game.team_b, game.played_at)))
    rating_a = model.ratings.get(game.team_a, model.base_rating)
    rating_b = model.ratings.get(game.team_b, model.base_rating)
    change = model.k_factor * (game.team_a_won - probability)
    model.ratings[game.team_a] = rating_a + change
    model.ratings[game.team_b] = rating_b - change
    if game.margin is not None:
        clamped = max(-model.margin_cap, min(model.margin_cap, float(game.margin)))
        efficiency_delta = model.margin_k_factor * clamped
        model.efficiency_ratings[game.team_a] = model.efficiency_ratings.get(game.team_a, 0.0) + efficiency_delta
        model.efficiency_ratings[game.team_b] = model.efficiency_ratings.get(game.team_b, 0.0) - efficiency_delta
    for team in (game.team_a, game.team_b):
        model.games[team] = model.games.get(team, 0) + 1
        model.last_game_dates[team] = game.played_at.isoformat()
    model.samples += 1
    model.trained_through = game.played_at.date().isoformat()
    return probability


def fit_nba(games: Iterable[LolGame], *, k_factor: float = 20.0,
            home_advantage: float = 60.0, rest_day_value: float = 8.0,
            seasonal_regression: float = .15, efficiency_weight: float = .3,
            margin_k_factor: float = 6.0, margin_cap: float = 15.0) -> NbaModel:
    ordered = sorted(games, key=lambda game: (game.played_at, game.game_id))
    if not ordered:
        raise ValueError("no NBA games supplied")
    model = _new(k_factor, home_advantage, rest_day_value, seasonal_regression,
                 efficiency_weight, margin_k_factor, margin_cap)
    current_year = ordered[0].played_at.year
    for game in ordered:
        if game.played_at.year != current_year:
            model.ratings = {team: model.base_rating + (rating - model.base_rating) *
                             (1 - model.seasonal_regression)
                             for team, rating in model.ratings.items()}
            current_year = game.played_at.year
        _update(model, game)
    return model


def walk_forward_nba(games: Iterable[LolGame], *, k_factor: float = 20.0,
                     home_advantage: float = 60.0, rest_day_value: float = 8.0,
                     seasonal_regression: float = .15, efficiency_weight: float = .3,
                     margin_k_factor: float = 6.0, margin_cap: float = 15.0) -> list[dict[str, object]]:
    ordered = sorted(games, key=lambda game: (game.played_at, game.game_id))
    model = _new(k_factor, home_advantage, rest_day_value, seasonal_regression,
                 efficiency_weight, margin_k_factor, margin_cap)
    current_year = ordered[0].played_at.year if ordered else None
    rows = []
    for game in ordered:
        if current_year is not None and game.played_at.year != current_year:
            model.ratings = {team: model.base_rating + (rating - model.base_rating) *
                             (1 - model.seasonal_regression)
                             for team, rating in model.ratings.items()}
            current_year = game.played_at.year
        probability = model.game_probability(game.team_a, game.team_b, game.played_at)
        rows.append({"event_id": game.game_id, "played_at": game.played_at.date().isoformat(),
                     "team_a": game.team_a, "team_b": game.team_b,
                     "team_a_won": game.team_a_won, "model_probability_a": probability})
        _update(model, game)
    return rows


def _score(model: NbaModel, games: list[LolGame]) -> dict[str, float | int]:
    probabilities, outcomes = [], []
    current_year = datetime.fromisoformat(model.trained_through).year
    for game in games:
        if game.played_at.year != current_year:
            model.ratings = {team: model.base_rating + (rating - model.base_rating) *
                             (1 - model.seasonal_regression)
                             for team, rating in model.ratings.items()}
            current_year = game.played_at.year
        probabilities.append(_update(model, game))
        outcomes.append(game.team_a_won)
    return {
        "samples": len(games),
        "brier": mean((p-y) ** 2 for p, y in zip(probabilities, outcomes)),
        "log_loss": mean(-(y*math.log(p) + (1-y)*math.log(1-p))
                         for p, y in zip(probabilities, outcomes)),
        "accuracy": mean((p >= .5) == bool(y) for p, y in zip(probabilities, outcomes)),
    }


def evaluate_nba(games: Iterable[LolGame]) -> tuple[NbaModel, dict]:
    ordered = sorted(games, key=lambda game: (game.played_at, game.game_id))
    train = [game for game in ordered if game.played_at.year <= 2021]
    validation = [game for game in ordered if 2022 <= game.played_at.year <= 2023]
    final_test = [game for game in ordered if 2024 <= game.played_at.year <= 2025]
    candidates = [(k, home, rest) for k in (16., 20., 24.)
                  for home in (40., 60., 80.) for rest in (0., 8., 16.)]
    scored = []
    for k, home, rest in candidates:
        metrics = _score(fit_nba(train, k_factor=k, home_advantage=home,
                                 rest_day_value=rest), validation)
        scored.append((metrics["brier"], metrics["log_loss"], k, home, rest, metrics))
    _, _, k, home, rest, validation_metrics = min(scored)
    legacy_test_metrics = _score(fit_nba(train + validation, k_factor=k,
                                          home_advantage=home, rest_day_value=rest,
                                          efficiency_weight=0.0), final_test)
    efficiency_test_metrics = _score(fit_nba(train + validation, k_factor=k,
                                             home_advantage=home, rest_day_value=rest), final_test)
    test_metrics = efficiency_test_metrics
    production = fit_nba(ordered, k_factor=k, home_advantage=home, rest_day_value=rest)
    evaluation = {
        "protocol": {"train": "<=2021", "validation": "2022-2023",
                     "retrospective_test": "2024-2025", "new_unseen_lockbox": False},
        "selected_parameters": {"k_factor": k, "home_advantage": home,
                                "rest_day_value": rest},
        "data": {"samples": len(ordered), "train_samples": len(train),
                 "validation_samples": len(validation), "test_samples": len(final_test)},
        "validation": validation_metrics, "retrospective_test": test_metrics,
        "comparison": {"legacy_elo": legacy_test_metrics, "margin_efficiency": efficiency_test_metrics},
        "approved_for_probability_use": test_metrics["samples"] >= 500 and test_metrics["brier"] < .25,
        "approved_for_real_money": False,
        "limitations": [
            "2024-2025 is retrospective because earlier baseline results were already inspected",
            "2026 market outcomes have been inspected and cannot be reused as an unseen lockbox",
            "injuries, confirmed starters, travel distance, and player-level strength are not included",
        ],
    }
    return production, evaluation


def save_nba(model: NbaModel, evaluation: dict, path: str | Path) -> None:
    if "approved_for_real_money" not in evaluation or not evaluation.get("validation") or not evaluation.get("retrospective_test"):
        raise ValueError("refusing to write NBA model without a complete evaluation record")
    Path(path).write_text(json.dumps({"model": asdict(model), "evaluation": evaluation},
                                    ensure_ascii=False, indent=2), encoding="utf-8")


def load_nba(path: str | Path) -> tuple[NbaModel, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NbaModel(**payload["model"]), payload.get("evaluation", {})
