"""NBA Elo with pregame-only home court and rest context."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
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

    def game_probability(self, home_team: str, away_team: str,
                         played_at: datetime | None = None) -> float:
        adjustment = self.home_advantage
        if played_at is not None:
            adjustment += self.rest_day_value * (
                _rest_days(self.last_game_dates.get(home_team), played_at) -
                _rest_days(self.last_game_dates.get(away_team), played_at)
            )
        home = self.ratings.get(home_team, self.base_rating) + adjustment
        away = self.ratings.get(away_team, self.base_rating)
        return 1 / (1 + 10 ** ((away - home) / 400))

    def explain(self, home_team: str, away_team: str,
                played_at: datetime | None = None) -> list[str]:
        reasons = [
            f"赛前强度：{home_team} {self.ratings.get(home_team, self.base_rating):.0f}，"
            f"{away_team} {self.ratings.get(away_team, self.base_rating):.0f}",
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


def _new(k_factor: float, home_advantage: float, rest_day_value: float,
         seasonal_regression: float = .15) -> NbaModel:
    return NbaModel({}, {}, {}, "", 0, k_factor, home_advantage,
                    rest_day_value, seasonal_regression)


def _update(model: NbaModel, game: LolGame) -> float:
    probability = min(1 - 1e-8, max(1e-8, model.game_probability(
        game.team_a, game.team_b, game.played_at)))
    rating_a = model.ratings.get(game.team_a, model.base_rating)
    rating_b = model.ratings.get(game.team_b, model.base_rating)
    change = model.k_factor * (game.team_a_won - probability)
    model.ratings[game.team_a] = rating_a + change
    model.ratings[game.team_b] = rating_b - change
    for team in (game.team_a, game.team_b):
        model.games[team] = model.games.get(team, 0) + 1
        model.last_game_dates[team] = game.played_at.isoformat()
    model.samples += 1
    model.trained_through = game.played_at.date().isoformat()
    return probability


def fit_nba(games: Iterable[LolGame], *, k_factor: float = 20.0,
            home_advantage: float = 60.0, rest_day_value: float = 8.0,
            seasonal_regression: float = .15) -> NbaModel:
    ordered = sorted(games, key=lambda game: (game.played_at, game.game_id))
    if not ordered:
        raise ValueError("no NBA games supplied")
    model = _new(k_factor, home_advantage, rest_day_value, seasonal_regression)
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
                     seasonal_regression: float = .15) -> list[dict[str, object]]:
    ordered = sorted(games, key=lambda game: (game.played_at, game.game_id))
    model = _new(k_factor, home_advantage, rest_day_value, seasonal_regression)
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
    test_metrics = _score(fit_nba(train + validation, k_factor=k,
                                  home_advantage=home, rest_day_value=rest), final_test)
    production = fit_nba(ordered, k_factor=k, home_advantage=home, rest_day_value=rest)
    evaluation = {
        "protocol": {"train": "<=2021", "validation": "2022-2023",
                     "retrospective_test": "2024-2025", "new_unseen_lockbox": False},
        "selected_parameters": {"k_factor": k, "home_advantage": home,
                                "rest_day_value": rest},
        "data": {"samples": len(ordered), "train_samples": len(train),
                 "validation_samples": len(validation), "test_samples": len(final_test)},
        "validation": validation_metrics, "retrospective_test": test_metrics,
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
