"""Patch-, roster-, player-, and champion-aware LoL game model."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Iterable


ROLES = ("top", "jng", "mid", "bot", "sup")


@dataclass(frozen=True)
class LolDraftGame:
    game_id: str
    played_at: datetime
    patch: str
    league: str
    blue_team: str
    red_team: str
    blue_players: tuple[str, ...]
    red_players: tuple[str, ...]
    blue_champions: tuple[str, ...]
    red_champions: tuple[str, ...]
    blue_won: int


@dataclass
class LolMetaModel:
    team_ratings: dict[str, float] = field(default_factory=dict)
    player_ratings: dict[str, float] = field(default_factory=dict)
    player_champion_ratings: dict[str, float] = field(default_factory=dict)
    patch_champion_ratings: dict[str, float] = field(default_factory=dict)
    patch_side_wins: dict[str, float] = field(default_factory=dict)
    patch_side_games: dict[str, int] = field(default_factory=dict)
    team_games: dict[str, int] = field(default_factory=dict)
    player_games: dict[str, int] = field(default_factory=dict)
    latest_team_rosters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    team_last_game: dict[str, str] = field(default_factory=dict)
    trained_through: str = ""
    samples: int = 0
    team_weight: float = .72
    team_k: float = 28.0
    player_k: float = 10.0
    proficiency_k: float = 6.0
    meta_k: float = 4.0
    base_rating: float = 1500.0

    def _average(self, values: Iterable[float]) -> float:
        data = tuple(values)
        return mean(data) if data else self.base_rating

    def _pre_rating(self, team: str, players: tuple[str, ...]) -> float:
        team_rating = self.team_ratings.get(team, self.base_rating)
        player_rating = self._average(self.player_ratings.get(player, self.base_rating) for player in players)
        return self.team_weight * team_rating + (1 - self.team_weight) * player_rating

    def _draft_adjustment(self, patch: str, players: tuple[str, ...], champions: tuple[str, ...]) -> float:
        if not players or len(players) != len(champions):
            return 0.0
        proficiency = self._average(
            self.player_champion_ratings.get(f"{player}|{champion}", self.base_rating)
            for player, champion in zip(players, champions)
        ) - self.base_rating
        meta = self._average(
            self.patch_champion_ratings.get(f"{patch}|{champion}", self.base_rating)
            for champion in champions
        ) - self.base_rating
        return .65 * proficiency + .35 * meta

    def _side_adjustment(self, patch: str) -> float:
        games = self.patch_side_games.get(patch, 0)
        wins = self.patch_side_wins.get(patch, 0.0)
        probability = (wins + 12) / (games + 24)
        probability = min(.62, max(.38, probability))
        return 200 * math.log10(probability / (1 - probability))

    @staticmethod
    def _elo_probability(rating_diff: float) -> float:
        return 1 / (1 + 10 ** (-rating_diff / 400))

    def predict_pre_draft(self, game: LolDraftGame) -> float:
        diff = (self._pre_rating(game.blue_team, game.blue_players)
                - self._pre_rating(game.red_team, game.red_players)
                + self._side_adjustment(game.patch))
        return self._elo_probability(diff)

    def predict_post_draft(self, game: LolDraftGame) -> float:
        diff = (self._pre_rating(game.blue_team, game.blue_players)
                - self._pre_rating(game.red_team, game.red_players)
                + self._draft_adjustment(game.patch, game.blue_players, game.blue_champions)
                - self._draft_adjustment(game.patch, game.red_players, game.red_champions)
                + self._side_adjustment(game.patch))
        return self._elo_probability(diff)

    def draft_readout(self, game: LolDraftGame) -> dict:
        """Structured lane-by-lane draft readout for research-oriented alerts."""
        post = self.predict_post_draft(game)
        blue_players = game.blue_players or ()
        red_players = game.red_players or ()
        blue_champions = game.blue_champions or ()
        red_champions = game.red_champions or ()
        lanes = []
        for index, role in enumerate(ROLES):
            bp = blue_players[index] if index < len(blue_players) else ""
            rp = red_players[index] if index < len(red_players) else ""
            bc = blue_champions[index] if index < len(blue_champions) else ""
            rc = red_champions[index] if index < len(red_champions) else ""
            blue_prof = (self.player_champion_ratings.get(f"{bp}|{bc}", self.base_rating) - self.base_rating
                         if bp and bc else 0.0)
            red_prof = (self.player_champion_ratings.get(f"{rp}|{rc}", self.base_rating) - self.base_rating
                        if rp and rc else 0.0)
            blue_meta = (self.patch_champion_ratings.get(f"{game.patch}|{bc}", self.base_rating) - self.base_rating
                         if bc else 0.0)
            red_meta = (self.patch_champion_ratings.get(f"{game.patch}|{rc}", self.base_rating) - self.base_rating
                        if rc else 0.0)
            blue_rating = .65 * blue_prof + .35 * blue_meta
            red_rating = .65 * red_prof + .35 * red_meta
            lanes.append({
                "role": role, "blue_champion": bc, "red_champion": rc,
                "blue_player": bp, "red_player": rp,
                "blue_proficiency": round(blue_prof, 1), "red_proficiency": round(red_prof, 1),
                "blue_meta": round(blue_meta, 1), "red_meta": round(red_meta, 1),
                "edge": round(blue_rating - red_rating, 1),
            })
        team_edge = self.team_ratings.get(game.blue_team, self.base_rating) - self.team_ratings.get(game.red_team, self.base_rating)
        return {
            "blue_team": game.blue_team, "red_team": game.red_team,
            "patch": game.patch, "post_draft_blue_win": round(post, 4),
            "team_edge": round(team_edge, 1), "lanes": lanes,
        }

    def as_dict(self) -> dict:
        return asdict(self)


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


def _patch(value: str) -> str:
    text = value.strip()
    if not text:
        return "unknown"
    parts = text.split(".")
    return f"{int(float(parts[0]))}.{int(float(parts[1])):02d}" if len(parts) > 1 else text


def load_oracle_drafts(paths: Iterable[str | Path]) -> list[LolDraftGame]:
    grouped: dict[str, dict] = {}
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                position = str(row.get("position") or "").casefold()
                if position not in ROLES:
                    continue
                game_id = str(row.get("gameid") or "").strip()
                side = str(row.get("side") or "").casefold()
                if not game_id or side not in {"blue", "red"}:
                    continue
                key = f"{row.get('league','')}|{game_id}"
                item = grouped.setdefault(key, {
                    "game_id": key, "played_at": _parse_time(str(row["date"])),
                    "patch": _patch(str(row.get("patch") or "")), "league": str(row.get("league") or "unknown"),
                    "blue": {}, "red": {}, "blue_team": "", "red_team": "", "blue_won": None,
                })
                role = ROLES.index(position)
                player = str(row.get("playerid") or row.get("playername") or "").strip()
                champion = str(row.get("champion") or "").strip()
                item[side][role] = (player, champion)
                item[f"{side}_team"] = str(row.get("teamname") or row.get("teamid") or "").strip()
                if side == "blue" and str(row.get("result") or "") in {"0", "1", "0.0", "1.0"}:
                    item["blue_won"] = int(float(row["result"]))
    games = []
    for item in grouped.values():
        if len(item["blue"]) != 5 or len(item["red"]) != 5 or item["blue_won"] is None:
            continue
        blue = [item["blue"][index] for index in range(5)]
        red = [item["red"][index] for index in range(5)]
        if not item["blue_team"] or not item["red_team"] or any(not p or not c for p, c in blue + red):
            continue
        games.append(LolDraftGame(
            item["game_id"], item["played_at"], item["patch"], item["league"],
            item["blue_team"], item["red_team"], tuple(p for p, _ in blue), tuple(p for p, _ in red),
            tuple(c for _, c in blue), tuple(c for _, c in red), item["blue_won"],
        ))
    return sorted(games, key=lambda game: (game.played_at, game.game_id))


def _update(model: LolMetaModel, game: LolDraftGame) -> tuple[float, float]:
    pre = min(1 - 1e-8, max(1e-8, model.predict_pre_draft(game)))
    post = min(1 - 1e-8, max(1e-8, model.predict_post_draft(game)))
    error = game.blue_won - post
    model.team_ratings[game.blue_team] = model.team_ratings.get(game.blue_team, 1500.0) + model.team_k * error
    model.team_ratings[game.red_team] = model.team_ratings.get(game.red_team, 1500.0) - model.team_k * error
    for player in game.blue_players:
        model.player_ratings[player] = model.player_ratings.get(player, 1500.0) + model.player_k * error
        model.player_games[player] = model.player_games.get(player, 0) + 1
    for player in game.red_players:
        model.player_ratings[player] = model.player_ratings.get(player, 1500.0) - model.player_k * error
        model.player_games[player] = model.player_games.get(player, 0) + 1
    for player, champion in zip(game.blue_players, game.blue_champions):
        key = f"{player}|{champion}"
        model.player_champion_ratings[key] = model.player_champion_ratings.get(key, 1500.0) + model.proficiency_k * error
        patch_key = f"{game.patch}|{champion}"
        model.patch_champion_ratings[patch_key] = model.patch_champion_ratings.get(patch_key, 1500.0) + model.meta_k * error
    for player, champion in zip(game.red_players, game.red_champions):
        key = f"{player}|{champion}"
        model.player_champion_ratings[key] = model.player_champion_ratings.get(key, 1500.0) - model.proficiency_k * error
        patch_key = f"{game.patch}|{champion}"
        model.patch_champion_ratings[patch_key] = model.patch_champion_ratings.get(patch_key, 1500.0) - model.meta_k * error
    model.patch_side_wins[game.patch] = model.patch_side_wins.get(game.patch, 0.0) + game.blue_won
    model.patch_side_games[game.patch] = model.patch_side_games.get(game.patch, 0) + 1
    model.team_games[game.blue_team] = model.team_games.get(game.blue_team, 0) + 1
    model.team_games[game.red_team] = model.team_games.get(game.red_team, 0) + 1
    model.latest_team_rosters[game.blue_team] = game.blue_players
    model.latest_team_rosters[game.red_team] = game.red_players
    model.team_last_game[game.blue_team] = game.played_at.isoformat()
    model.team_last_game[game.red_team] = game.played_at.isoformat()
    model.samples += 1
    model.trained_through = game.played_at.date().isoformat()
    return pre, post


def fit_lol_meta(games: Iterable[LolDraftGame], *, team_weight: float = .72,
                 team_k: float = 28.0, player_k: float = 10.0) -> LolMetaModel:
    model = LolMetaModel(team_weight=team_weight, team_k=team_k, player_k=player_k)
    for game in sorted(games, key=lambda item: (item.played_at, item.game_id)):
        _update(model, game)
    if not model.samples:
        raise ValueError("no complete LoL draft games supplied")
    return model


def _metrics(model: LolMetaModel, games: list[LolDraftGame]) -> dict:
    pre_values, post_values, outcomes = [], [], []
    for game in games:
        pre, post = _update(model, game)
        pre_values.append(pre)
        post_values.append(post)
        outcomes.append(game.blue_won)

    def score(values: list[float]) -> dict:
        return {
            "brier": mean((p - y) ** 2 for p, y in zip(values, outcomes)),
            "log_loss": mean(-(y * math.log(p) + (1-y) * math.log(1-p)) for p, y in zip(values, outcomes)),
            "accuracy": mean((p >= .5) == bool(y) for p, y in zip(values, outcomes)),
        }
    return {"samples": len(games), "pre_draft": score(pre_values), "post_draft": score(post_values)}


def evaluate_lol_meta(games: Iterable[LolDraftGame], *, validation_start: datetime,
                      test_start: datetime) -> tuple[LolMetaModel, dict]:
    ordered = sorted(games, key=lambda item: (item.played_at, item.game_id))
    train = [g for g in ordered if g.played_at < validation_start]
    validation = [g for g in ordered if validation_start <= g.played_at < test_start]
    test = [g for g in ordered if g.played_at >= test_start]
    if not train or not validation or not test:
        raise ValueError("train, validation, and locked test periods must all contain games")
    candidates = [(weight, team_k, player_k) for weight in (.6, .72, .84)
                  for team_k in (20., 28., 36.) for player_k in (6., 10., 14.)]
    tuned = []
    for weight, team_k, player_k in candidates:
        model = fit_lol_meta(train, team_weight=weight, team_k=team_k, player_k=player_k)
        metrics = _metrics(model, validation)
        tuned.append((metrics["post_draft"]["brier"], metrics["post_draft"]["log_loss"],
                      weight, team_k, player_k, metrics))
    _, _, weight, team_k, player_k, validation_metrics = min(tuned)
    locked = fit_lol_meta(train + validation, team_weight=weight, team_k=team_k, player_k=player_k)
    test_metrics = _metrics(locked, test)
    production = fit_lol_meta(ordered, team_weight=weight, team_k=team_k, player_k=player_k)
    post = test_metrics["post_draft"]
    approved = len(test) >= 1000 and post["brier"] < .25 and post["log_loss"] < math.log(2)
    report = {
        "source": "Oracle's Elixir (Riot/Bayes-supported community pipeline)",
        "protocol": {"train": f"< {validation_start.date()}",
                     "validation": f"{validation_start.date()} to {test_start.date()}",
                     "final_test": f">= {test_start.date()}", "final_test_is_locked": True},
        "data": {"samples": len(ordered), "train_samples": len(train),
                 "validation_samples": len(validation), "test_samples": len(test),
                 "first_game": ordered[0].played_at.isoformat(), "last_game": ordered[-1].played_at.isoformat()},
        "selected_parameters": {"team_weight": weight, "team_k": team_k, "player_k": player_k},
        "validation": validation_metrics, "final_test": test_metrics,
        "approved_for_probability_use": approved, "approved_for_real_money": False,
        "limitations": [
            "pre-draft and post-draft probabilities are separate and must not be interchanged",
            "historical executable odds are still required for ROI validation",
            "2026 is reserved for a second locked test after upstream quota/data-quality checks pass",
        ],
    }
    return production, report


def save_lol_meta(model: LolMetaModel, evaluation: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"model": model.as_dict(), "evaluation": evaluation},
                                 ensure_ascii=False, indent=2), encoding="utf-8")


def load_lol_meta(path: str | Path) -> tuple[LolMetaModel, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return LolMetaModel(**payload["model"]), payload.get("evaluation", {})
