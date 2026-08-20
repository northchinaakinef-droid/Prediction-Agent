from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable


def _clip(value: float) -> float:
    return min(1 - 1e-8, max(1e-8, value))


def series_probability(game_probability: float, best_of: int) -> float:
    """Probability of winning a best-of-N series under iid game odds."""
    if best_of < 1 or best_of % 2 == 0:
        raise ValueError("best_of must be a positive odd number")
    needed = best_of // 2 + 1
    return sum(math.comb(best_of, wins) * game_probability ** wins *
               (1 - game_probability) ** (best_of - wins)
               for wins in range(needed, best_of + 1))


@dataclass(frozen=True)
class LolGame:
    game_id: str
    played_at: datetime
    league: str
    team_a: str
    team_b: str
    team_a_won: int
    margin: float | None = None
    location: str | None = None


@dataclass
class EloModel:
    ratings: dict[str, float]
    games: dict[str, int]
    trained_through: str
    samples: int
    k_factor: float = 24.0
    base_rating: float = 1500.0

    def game_probability(self, team_a: str, team_b: str) -> float:
        a = self.ratings.get(team_a, self.base_rating)
        b = self.ratings.get(team_b, self.base_rating)
        return 1 / (1 + 10 ** ((b - a) / 400))

    def explain(self, team_a: str, team_b: str) -> list[str]:
        return [
            f"独立 Elo：{team_a} {self.ratings.get(team_a, self.base_rating):.0f}，"
            f"{team_b} {self.ratings.get(team_b, self.base_rating):.0f}",
            f"历史样本：{team_a} {self.games.get(team_a, 0)} 局，"
            f"{team_b} {self.games.get(team_b, 0)} 局；训练截至 {self.trained_through}",
        ]

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "EloModel":
        return cls(**value)


def player_level_team_rating(player_ratings: dict[str, float], roster: Iterable[str],
                               base_rating: float = 1500.0) -> float:
    players = tuple(roster)
    if not players:
        return base_rating
    return mean(player_ratings.get(player, base_rating) for player in players)


def roster_adjusted_probability(team_a_rating: float, team_b_rating: float,
                                roster_a_rating: float, roster_b_rating: float,
                                team_weight: float = 0.65) -> float:
    a = team_weight * team_a_rating + (1 - team_weight) * roster_a_rating
    b = team_weight * team_b_rating + (1 - team_weight) * roster_b_rating
    return 1 / (1 + 10 ** ((b - a) / 400))


def recent_form_rating(games: Iterable[LolGame], team: str, before: datetime,
                       window: int = 8) -> float:
    prior = sorted(
        (g for g in games if g.played_at < before and team in {g.team_a, g.team_b}),
        key=lambda g: g.played_at,
    )[-window:]
    if not prior:
        return 0.5
    wins = sum(1 for g in prior if (g.team_a == team and g.team_a_won) or (g.team_b == team and not g.team_a_won))
    return wins / len(prior)


def patch_adaptation_uncertainty(days_since_patch: float, adaptation_days: int = 7) -> float:
    if days_since_patch < 0:
        raise ValueError("days_since_patch must be non-negative")
    if days_since_patch >= adaptation_days:
        return 1.0
    return 0.5 + 0.5 * (days_since_patch / adaptation_days)


def side_advantage(patch: str, region: str,
                   table: dict[tuple[str, str], float] | None = None) -> float:
    table = table or {("14.1", "LCK"): 0.045, ("14.1", "LPL"): 0.030, ("14.2", "LCK"): 0.038}
    return table.get((patch, region), 0.0)


def load_oracle_elixir(paths: Iterable[str | Path]) -> list[LolGame]:
    games: list[LolGame] = []
    seen: set[str] = set()
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("position", "")).casefold() != "team":
                    continue
                game_id = str(row.get("gameid") or "").strip()
                if not game_id or game_id in seen:
                    continue
                # The blue/team-1 aggregate row is emitted before the opponent.
                side = str(row.get("side", "")).casefold()
                if side not in {"blue", "1"}:
                    continue
                opponent = None
                # Opponents are joined below in a second compact pass per file.
                # Storing rows first keeps ingestion compatible across OE schemas.
                seen.add(game_id)
                games.append(LolGame(
                    game_id=game_id,
                    played_at=_parse_date(row.get("date") or row.get("datetime")),
                    league=str(row.get("league") or "unknown"),
                    team_a=str(row.get("teamname") or row.get("teamid") or "unknown"),
                    team_b="",
                    team_a_won=int(float(row.get("result") or 0)),
                ))
        # Fill opponents with a small indexed pass so the main objects stay lean.
        opponents: dict[str, str] = {}
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("position", "")).casefold() == "team" and str(row.get("side", "")).casefold() in {"red", "2"}:
                    opponents[str(row.get("gameid") or "")] = str(row.get("teamname") or row.get("teamid") or "unknown")
        games = [LolGame(g.game_id, g.played_at, g.league, g.team_a, opponents.get(g.game_id, g.team_b), g.team_a_won)
                 if not g.team_b and g.game_id in opponents else g for g in games]
    return sorted((g for g in games if g.team_a and g.team_b), key=lambda g: (g.played_at, g.game_id))


def load_canonical_games(paths: Iterable[str | Path], league: str) -> list[LolGame]:
    """Load pre-game-safe results: event_id, played_at, team_a, team_b, team_a_won."""
    games: list[LolGame] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                games.append(LolGame(
                    game_id=str(row["event_id"]), played_at=_parse_date(row["played_at"]),
                    league=league, team_a=str(row["team_a"]), team_b=str(row["team_b"]),
                    team_a_won=int(row["team_a_won"]),
                ))
    return sorted(games, key=lambda game: (game.played_at, game.game_id))


def _parse_date(value: object) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def fit_elo(games: Iterable[LolGame], *, k_factor: float = 24.0,
            seasonal_regression: float = 0.15) -> EloModel:
    ordered = sorted(games, key=lambda g: (g.played_at, g.game_id))
    if not ordered:
        raise ValueError("no LoL games supplied")
    ratings: dict[str, float] = {}
    counts: dict[str, int] = {}
    current_year = ordered[0].played_at.year
    for game in ordered:
        if game.played_at.year != current_year:
            ratings = {team: 1500 + (rating - 1500) * (1 - seasonal_regression)
                       for team, rating in ratings.items()}
            current_year = game.played_at.year
        a = ratings.get(game.team_a, 1500.0)
        b = ratings.get(game.team_b, 1500.0)
        expected = 1 / (1 + 10 ** ((b - a) / 400))
        change = k_factor * (game.team_a_won - expected)
        ratings[game.team_a], ratings[game.team_b] = a + change, b - change
        counts[game.team_a] = counts.get(game.team_a, 0) + 1
        counts[game.team_b] = counts.get(game.team_b, 0) + 1
    return EloModel(ratings, counts, ordered[-1].played_at.date().isoformat(), len(ordered), k_factor)


def walk_forward_elo_probabilities(
    games: Iterable[LolGame], *, k_factor: float = 24.0,
    seasonal_regression: float = 0.15,
) -> list[dict[str, object]]:
    """Capture every probability before its result updates the Elo state."""
    ordered = sorted(games, key=lambda game: (game.played_at, game.game_id))
    ratings: dict[str, float] = {}
    counts: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    current_year = ordered[0].played_at.year if ordered else None
    for game in ordered:
        if current_year is not None and game.played_at.year != current_year:
            ratings = {team: 1500 + (rating - 1500) * (1 - seasonal_regression)
                       for team, rating in ratings.items()}
            current_year = game.played_at.year
        rating_a = ratings.get(game.team_a, 1500.0)
        rating_b = ratings.get(game.team_b, 1500.0)
        probability = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        rows.append({
            "event_id": game.game_id,
            "played_at": game.played_at.date().isoformat(),
            "team_a": game.team_a,
            "team_b": game.team_b,
            "team_a_won": game.team_a_won,
            "model_probability_a": probability,
            "prior_games_a": counts.get(game.team_a, 0),
            "prior_games_b": counts.get(game.team_b, 0),
        })
        change = k_factor * (game.team_a_won - probability)
        ratings[game.team_a], ratings[game.team_b] = rating_a + change, rating_b - change
        counts[game.team_a] = counts.get(game.team_a, 0) + 1
        counts[game.team_b] = counts.get(game.team_b, 0) + 1
    return rows


def evaluate_years(games: Iterable[LolGame], *, train_end: int = 2023,
                   validation_year: int = 2024, test_year: int = 2025) -> dict:
    """Strict chronological train/validation/final-test evaluation."""
    return evaluate_periods(games, train_end=train_end,
                            validation_start=validation_year, validation_end=validation_year,
                            test_start=test_year, test_end=test_year)


def evaluate_periods(games: Iterable[LolGame], *, train_end: int,
                     validation_start: int, validation_end: int,
                     test_start: int, test_end: int) -> dict:
    if not train_end < validation_start <= validation_end < test_start <= test_end:
        raise ValueError("periods must be chronological and non-overlapping")
    ordered = sorted(games, key=lambda g: (g.played_at, g.game_id))
    train = [g for g in ordered if g.played_at.year <= train_end]
    validation = [g for g in ordered if validation_start <= g.played_at.year <= validation_end]
    final_test = [g for g in ordered if test_start <= g.played_at.year <= test_end]
    if not train or not validation or not final_test:
        raise ValueError("train, validation, and final-test years must all contain games")
    model = fit_elo(train)
    val_metrics, model_after_val = _prequential(model, validation)
    test_metrics, _ = _prequential(model_after_val, final_test)
    approved = (test_metrics["samples"] >= 500 and test_metrics["brier"] < 0.25 and
                test_metrics["log_loss"] < math.log(2))
    return {
        "protocol": {
            "train": f"<= {train_end}", "validation": f"{validation_start}-{validation_end}",
            "final_test": f"{test_start}-{test_end}", "final_test_is_locked": True,
        },
        "train_samples": len(train), "validation": val_metrics, "final_test": test_metrics,
        "approved_for_probability_use": approved,
        "approved_for_real_money": False,
        "note": "真实建议展示状态由已结算虚拟场次与虚拟 ROI 规则计算。",
    }


def _prequential(model: EloModel, games: list[LolGame]) -> tuple[dict, EloModel]:
    ratings, counts = dict(model.ratings), dict(model.games)
    predictions, outcomes = [], []
    for game in games:
        current = EloModel(ratings, counts, model.trained_through, model.samples, model.k_factor)
        p = _clip(current.game_probability(game.team_a, game.team_b))
        predictions.append(p)
        outcomes.append(game.team_a_won)
        a, b = ratings.get(game.team_a, 1500.0), ratings.get(game.team_b, 1500.0)
        delta = model.k_factor * (game.team_a_won - p)
        ratings[game.team_a], ratings[game.team_b] = a + delta, b - delta
        counts[game.team_a] = counts.get(game.team_a, 0) + 1
        counts[game.team_b] = counts.get(game.team_b, 0) + 1
    metrics = {
        "samples": len(games),
        "brier": mean((p - y) ** 2 for p, y in zip(predictions, outcomes)),
        "log_loss": mean(-(y * math.log(p) + (1 - y) * math.log(1 - p))
                         for p, y in zip(predictions, outcomes)),
        "accuracy": mean((p >= .5) == bool(y) for p, y in zip(predictions, outcomes)),
    }
    end = games[-1].played_at.date().isoformat()
    return metrics, EloModel(ratings, counts, end, model.samples + len(games), model.k_factor)


def save_model(model: EloModel, path: str | Path, evaluation: dict | None = None) -> None:
    if evaluation is None or "approved_for_real_money" not in evaluation or not evaluation.get("validation"):
        raise ValueError("refusing to write model without a complete evaluation record")
    payload = {"model": model.as_dict(), "evaluation": evaluation}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(path: str | Path) -> tuple[EloModel, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return EloModel.from_dict(payload["model"]), payload.get("evaluation", {})
