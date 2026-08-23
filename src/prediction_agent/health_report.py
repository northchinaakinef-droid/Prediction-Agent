from __future__ import annotations


def build_system_health(*, audits: dict, statuses: dict, data_health: dict,
                        notifications: dict | None = None, llm_provider: str | None = None,
                        llm_configured: bool = False) -> dict:
    schedule = {
        sport: {"discovered": int(audit.get("discovered") or 0),
                "expected": int(audit.get("expected") or 0),
                "status": audit.get("status")}
        for sport, audit in audits.items()
    }
    matched = sum(int(audit.get("market_matched") or 0) for audit in audits.values())
    total = sum(int(audit.get("expected") or 0) for audit in audits.values())
    return {
        "schedule": schedule,
        "market_match": {"matched": matched, "total": total},
        "models": {sport: status.get("model_status", "UNKNOWN") for sport, status in statuses.items()},
        "data": data_health,
        "notifications": notifications or {"PENDING": 0, "FAILED": 0, "DEAD": 0},
        "llm": {"provider": llm_provider or "none", "status": "OK" if llm_configured else "QUANT_FALLBACK"},
    }
