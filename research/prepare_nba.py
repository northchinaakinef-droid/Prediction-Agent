"""Convert the llimllib/stats.nba.com team gamelog parquet to canonical pregame rows."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


FRANCHISE_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Jersey Nets": "Brooklyn Nets",
    "New Orleans Hornets": "New Orleans Pelicans",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    frame = pd.read_parquet(args.source)
    frame["game_id"] = frame["game_id"].astype(str).str.zfill(10)
    frame = frame[frame["game_id"].str[:3].isin({"002", "004"})]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for game_id, group in frame.groupby("game_id", sort=False):
        if len(group) != 2 or set(group["wl"].dropna()) != {"W", "L"}:
            continue
        home = group[group["matchup"].str.contains(" vs. ", regex=False)]
        away = group[group["matchup"].str.contains(" @ ", regex=False)]
        if len(home) != 1 or len(away) != 1:
            continue
        home_row, away_row = home.iloc[0], away.iloc[0]
        rows.append({
            "event_id": game_id, "played_at": str(home_row["game_date"]),
            "team_a": FRANCHISE_ALIASES.get(str(home_row["team_name"]), str(home_row["team_name"])),
            "team_b": FRANCHISE_ALIASES.get(str(away_row["team_name"]), str(away_row["team_name"])),
            "team_a_won": 1 if home_row["wl"] == "W" else 0,
        })
    rows.sort(key=lambda row: (row["played_at"], row["event_id"]))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("event_id", "played_at", "team_a", "team_b", "team_a_won"))
        writer.writeheader()
        writer.writerows(rows)
    print({"games": len(rows), "first": rows[0]["played_at"], "last": rows[-1]["played_at"], "output": str(output)})


if __name__ == "__main__":
    main()
