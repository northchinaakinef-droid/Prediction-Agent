"""Refresh the frozen CS2 Recent Form artifact outside inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from prediction_agent.recent_form_refresh import (
    Bo3RecentFormProvider, model_team_count, refresh_cs2_recent_form,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=str(ROOT / "artifacts" / "recent_form.json"))
    parser.add_argument("--status", default=str(ROOT / "artifacts" / "recent_form_refresh_status.json"))
    parser.add_argument("--lookback-days", type=int, default=120)
    args = parser.parse_args()
    report = refresh_cs2_recent_form(
        provider=Bo3RecentFormProvider(), artifact_path=args.artifact, status_path=args.status,
        lookback_days=args.lookback_days,
        expected_team_count=model_team_count(ROOT / "artifacts" / "cs2_model.json"),
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())

