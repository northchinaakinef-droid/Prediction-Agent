"""Roster-aware CS2 series model and strict chronological evaluation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class Cs2Series:
    event_id: str
    played_at: date
    team_a: str
    team_b: str
    roster_a: tuple[str, ...]
    roster_b: tuple[str, ...]
    team_a_won: int


@dataclass
class Cs2Model:
    team_ratings: dict[str, float]
    player_ratings: dict[str, float]
    team_games: dict[str, int]
    player_games: dict[str, int]
    trained_through: str
    samples: int
    team_weight: float = 0.65
    team_k: float = 24.0
    player_k: float = 8.0
    base_rating: float = 1500.0

    def _side_rating(self, team: str, roster: tuple[str, ...]) -> float:
        team_rating = self.team_ratings.get(team, self.base_rating)
        if not roster:
            return team_rating
        player_rating = mean(self.player_ratings.get(player, self.base_rating) for player in roster)
        return self.team_weight * team_rating + (1 - self.team_weight) * player_rating

    def probability(self, team_a: str, team_b: str,
                    roster_a: Iterable[str] = (), roster_b: Iterable[str] = ()) -> float:
        a = self._side_rating(team_a, tuple(roster_a))
        b = self._side_rating(team_b, tuple(roster_b))
        return 1 / (1 + 10 ** ((b - a) / 400))

    def as_dict(self) -> dict:
        return asdict(self)


_TEAM_RE = re.compile(r"^Team Name:\s*(.+?)<br", re.MULTILINE)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_valve_vrs(root: str | Path, *, start_year: int = 2024) -> list[Cs2Series]:
    """Parse Valve VRS snapshots without using future rank values as features."""
    perspectives: dict[tuple[str, str, str], dict[str, object]] = {}
    conflicts: set[tuple[str, str, str]] = set()
    for path in Path(root).glob("live/*/details/**/*.md"):
        text = path.read_text(encoding="utf-8", errors="replace")
        team_match = _TEAM_RE.search(text)
        if not team_match:
            continue
        team = team_match.group(1).strip()
        for line in text.splitlines():
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 12 or not _DATE_RE.match(cells[2]) or cells[4] not in {"W", "L"}:
                continue
            played = date.fromisoformat(cells[2])
            if played.year < start_year:
                continue
            opponent = cells[3].strip()
            if not opponent or opponent.casefold() == team.casefold():
                continue
            low, high = sorted((team, opponent), key=str.casefold)
            key = (played.isoformat(), low.casefold(), high.casefold())
            won = 1 if cells[4] == "W" else 0
            roster = tuple(sorted((p.strip() for p in cells[-1].split(",") if p.strip()), key=str.casefold))
            side = "a" if team.casefold() == low.casefold() else "b"
            current = perspectives.setdefault(key, {"names": (low, high)})
            previous = current.get(side)
            if previous is not None and previous[0] != won:  # type: ignore[index]
                conflicts.add(key)
            current[side] = (won, roster)

    games: list[Cs2Series] = []
    for key, sides in perspectives.items():
        if key in conflicts:
            continue
        low, high = sides["names"]  # type: ignore[misc]
        a = sides.get("a")
        b = sides.get("b")
        if a is None and b is None:
            continue
        team_a_won = a[0] if a is not None else 1 - b[0]  # type: ignore[index]
        if a is not None and b is not None and a[0] == b[0]:  # type: ignore[index]
            continue
        games.append(Cs2Series(
            event_id="|".join(key), played_at=date.fromisoformat(key[0]),
            team_a=low, team_b=high,
            roster_a=a[1] if a else (), roster_b=b[1] if b else (),  # type: ignore[index]
            team_a_won=team_a_won,
        ))
    return sorted(games, key=lambda game: (game.played_at, game.event_id))


def _new_model(team_weight: float, team_k: float, player_k: float) -> Cs2Model:
    return Cs2Model({}, {}, {}, {}, "", 0, team_weight, team_k, player_k)


def _update(model: Cs2Model, game: Cs2Series) -> float:
    probability = min(1 - 1e-8, max(1e-8, model.probability(
        game.team_a, game.team_b, game.roster_a, game.roster_b)))
    error = game.team_a_won - probability
    model.team_ratings[game.team_a] = model.team_ratings.get(game.team_a, 1500.0) + model.team_k * error
    model.team_ratings[game.team_b] = model.team_ratings.get(game.team_b, 1500.0) - model.team_k * error
    for player in game.roster_a:
        model.player_ratings[player] = model.player_ratings.get(player, 1500.0) + model.player_k * error
        model.player_games[player] = model.player_games.get(player, 0) + 1
    for player in game.roster_b:
        model.player_ratings[player] = model.player_ratings.get(player, 1500.0) - model.player_k * error
        model.player_games[player] = model.player_games.get(player, 0) + 1
    model.team_games[game.team_a] = model.team_games.get(game.team_a, 0) + 1
    model.team_games[game.team_b] = model.team_games.get(game.team_b, 0) + 1
    model.samples += 1
    model.trained_through = game.played_at.isoformat()
    return probability


def fit_cs2(games: Iterable[Cs2Series], *, team_weight: float = .65,
            team_k: float = 24.0, player_k: float = 8.0) -> Cs2Model:
    model = _new_model(team_weight, team_k, player_k)
    for game in sorted(games, key=lambda item: (item.played_at, item.event_id)):
        _update(model, game)
    if not model.samples:
        raise ValueError("no CS2 series supplied")
    return model


def _score(model: Cs2Model, games: list[Cs2Series]) -> dict:
    probabilities, outcomes = [], []
    for game in games:
        probabilities.append(_update(model, game))
        outcomes.append(game.team_a_won)
    return {
        "samples": len(games),
        "brier": mean((p - y) ** 2 for p, y in zip(probabilities, outcomes)),
        "log_loss": mean(-(y * math.log(p) + (1 - y) * math.log(1 - p))
                         for p, y in zip(probabilities, outcomes)),
        "accuracy": mean((p >= .5) == bool(y) for p, y in zip(probabilities, outcomes)),
    }


def evaluate_cs2(games: Iterable[Cs2Series], *, train_year: int = 2024,
                 validation_year: int = 2025, test_year: int = 2026) -> tuple[Cs2Model, dict]:
    ordered = sorted(games, key=lambda item: (item.played_at, item.event_id))
    train = [g for g in ordered if g.played_at.year == train_year]
    validation = [g for g in ordered if g.played_at.year == validation_year]
    test = [g for g in ordered if g.played_at.year == test_year]
    if not train or not validation or not test:
        raise ValueError("train, validation, and locked test years must all contain CS2 series")
    candidates = [(weight, team_k, player_k)
                  for weight in (.5, .65, .8) for team_k in (16., 24., 32.) for player_k in (4., 8., 12.)]
    tuned = []
    for weight, team_k, player_k in candidates:
        model = fit_cs2(train, team_weight=weight, team_k=team_k, player_k=player_k)
        metrics = _score(model, validation)
        tuned.append((metrics["brier"], metrics["log_loss"], weight, team_k, player_k, metrics))
    _, _, weight, team_k, player_k, validation_metrics = min(tuned)
    locked_model = fit_cs2(train + validation, team_weight=weight, team_k=team_k, player_k=player_k)
    test_metrics = _score(locked_model, test)
    production_model = fit_cs2(ordered, team_weight=weight, team_k=team_k, player_k=player_k)
    approved = test_metrics["samples"] >= 300 and test_metrics["brier"] < .25 and test_metrics["log_loss"] < math.log(2)
    evaluation = {
        "source": "ValveSoftware/counter-strike_regional_standings",
        "protocol": {"train": str(train_year), "validation": str(validation_year),
                     "final_test": str(test_year), "final_test_is_locked": True},
        "data": {"samples": len(ordered), "train_samples": len(train),
                 "validation_samples": len(validation), "test_samples": len(test),
                 "first_match": ordered[0].played_at.isoformat(), "last_match": ordered[-1].played_at.isoformat()},
        "selected_parameters": {"team_weight": weight, "team_k": team_k, "player_k": player_k},
        "validation": validation_metrics, "final_test": test_metrics,
        "approved_for_probability_use": approved,
        "approved_for_real_money": False,
        "limitations": [
            "baseline uses series results and roster identity; map pool and veto are not yet included",
            "no decision-time executable odds, so ROI has not been validated",
            "same-day rematches between identical teams are conservatively deduplicated",
        ],
    }
    return production_model, evaluation


def save_cs2(model: Cs2Model, evaluation: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"model": model.as_dict(), "evaluation": evaluation},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
