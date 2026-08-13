from __future__ import annotations

import argparse
import json

from prediction_agent.cs2_model import evaluate_cs2, load_valve_vrs, save_cs2


def main() -> None:
    parser = argparse.ArgumentParser(description="Train roster-aware CS2 baseline from Valve VRS snapshots")
    parser.add_argument("source", help="cloned Valve VRS repository")
    parser.add_argument("--output", default="artifacts/cs2_model.json")
    args = parser.parse_args()
    games = load_valve_vrs(args.source)
    model, evaluation = evaluate_cs2(games)
    save_cs2(model, evaluation, args.output)
    print(json.dumps({"saved": args.output, "evaluation": evaluation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
