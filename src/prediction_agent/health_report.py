from __future__ import annotations


def build_system_health(*, audits: dict, statuses: dict, data_health: dict,
                        notifications: dict | None = None, llm_provider: str | None = None,
                        llm_configured: bool = False,
                        llm_telemetry: dict | None = None) -> dict:
    schedule = {
        sport: {"discovered": int(audit.get("discovered") or 0),
                "expected": int(audit.get("expected") or 0),
                "status": audit.get("status")}
        for sport, audit in audits.items()
    }
    matched = sum(int(audit.get("market_matched") or 0) for audit in audits.values())
    total = sum(int(audit.get("expected") or 0) for audit in audits.values())
    telemetry = dict(llm_telemetry or {})
    configured = bool(telemetry.get("configured", llm_configured))
    success = int(telemetry.get("success") or 0)
    failures = int(telemetry.get("request_failures") or 0) + int(telemetry.get("validation_failures") or 0)
    attempted = int(telemetry.get("attempted") or 0)
    if not configured:
        llm_status = "NOT_CONFIGURED"
    elif success and failures:
        llm_status = "DEGRADED"
    elif success:
        llm_status = "ACTIVE"
    elif attempted and failures:
        llm_status = "ERROR"
    else:
        llm_status = "QUANT_FALLBACK"
    llm = {"provider": telemetry.get("provider") or llm_provider or "none",
           "model": telemetry.get("model"), "configured": configured,
           "status": llm_status, **telemetry}
    llm["status"] = llm_status
    return {
        "schedule": schedule,
        "market_match": {"matched": matched, "total": total},
        "models": {sport: status.get("model_status", "UNKNOWN") for sport, status in statuses.items()},
        "data": data_health,
        "notifications": notifications or {"PENDING": 0, "FAILED": 0, "DEAD": 0},
        "llm": llm,
    }
