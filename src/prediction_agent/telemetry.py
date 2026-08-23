from __future__ import annotations

import json
import logging


def event_log(event: str, *, match_id: str = "", sport: str = "", league: str = "",
              run_id: str = "", **fields) -> None:
    logging.info(json.dumps({
        "event": event, "match_id": match_id, "sport": sport, "league": league,
        "run_id": run_id, **fields,
    }, ensure_ascii=False, default=str, sort_keys=True))
