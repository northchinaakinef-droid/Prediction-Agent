from __future__ import annotations

import argparse

from prediction_agent.lol_model import load_canonical_games
from prediction_agent.nba_model import evaluate_nba, save_nba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("games", default="data/raw/nba/nba_games.csv", nargs="?")
    parser.add_argument("--output", default="artifacts/nba_model.json")
    args = parser.parse_args()
    model, evaluation = evaluate_nba(load_canonical_games([args.games], "nba"))
    save_nba(model, evaluation, args.output)
    print(evaluation)


if __name__ == "__main__":
    main()
