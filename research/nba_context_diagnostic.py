"""Retrospective market diagnostic for the post-lockbox NBA context model.

The 2026 market outcomes were already inspected before this model was created,
so this report is never an unseen test and cannot approve production betting.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from prediction_agent.lol_model import load_canonical_games
from prediction_agent.nba_model import load_nba, walk_forward_nba

from nba_market_backtest import (DECISION_HOURS, END, START, TAG, evaluate,
                                 fill_histories, match_prediction, prediction_index,
                                 price_at)
from polymarket_walkforward import extract_games, fetch_events


def main() -> None:
    model, evaluation = load_nba("artifacts/nba_model.json")
    parameters = evaluation["selected_parameters"]
    rows = walk_forward_nba(load_canonical_games(["data/raw/nba/nba_games.csv"], "nba"), **parameters)
    index = prediction_index(rows)
    events = fetch_events(TAG, START, END, Path("data/research/nba_events.json"))
    games = extract_games("nba", events)
    joined = [(game, prediction) for game in games
              if (prediction := match_prediction(game, index)) is not None]
    histories = fill_histories([game for game, _ in joined], Path("data/research/nba_price_histories.json"))
    observations = []
    for game, prediction in joined:
        model_a = float(prediction["model_probability_a"])
        if game.team_a.casefold() != str(prediction["team_a"]).casefold():
            model_a = 1 - model_a
        for hours in DECISION_HOURS:
            price = price_at(histories.get(game.token_a, []), game.start_ts - hours * 3600)
            if price is not None:
                observations.append({
                    "start": datetime.fromtimestamp(game.start_ts, timezone.utc).isoformat(),
                    "decision_hours": hours, "winner": game.winner,
                    "model_probability_a": model_a, "market_probability_a": price,
                    "volume": game.volume,
                })
    results = {}
    for hours in DECISION_HOURS:
        window = [row for row in observations if row["decision_hours"] == hours]
        results[f"T-{hours}h"] = {
            "2025_diagnostic": evaluate([row for row in window if str(row["start"]).startswith("2025")]),
            "2026_already_seen_diagnostic": evaluate([row for row in window if str(row["start"]).startswith("2026")]),
        }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "RETROSPECTIVE_DIAGNOSTIC_ONLY",
        "approved_for_real_money": False,
        "reason": "2026 outcomes were inspected before this context model was developed",
        "model_parameters": parameters,
        "results": results,
    }
    target = Path("reports/nba_context_market_diagnostic.json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
