from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from prediction_agent.lol_meta_model import evaluate_lol_meta, load_oracle_drafts, save_lol_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the patch/roster/champion-aware LoL model")
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--validation-start", default="2024-09-01T00:00:00Z")
    parser.add_argument("--test-start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--output", default="artifacts/lol_meta_model.json")
    args = parser.parse_args()
    parse = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    games = load_oracle_drafts(args.csv)
    model, evaluation = evaluate_lol_meta(
        games, validation_start=parse(args.validation_start), test_start=parse(args.test_start))
    save_lol_meta(model, evaluation, args.output)
    print(json.dumps({"saved": args.output, "evaluation": evaluation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
