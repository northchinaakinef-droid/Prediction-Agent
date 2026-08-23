"""Read-only natural-language summaries for reports.

This module intentionally imports neither ``risk`` nor anything that mutates a
``Recommendation``.  It only reads plain row dictionaries after ``recommend()``
has already produced numeric outputs.
"""
from __future__ import annotations


def build_pre_match_summary(row: dict) -> str:
    event = row.get("event", "unknown")
    outcome = row.get("outcome")
    model = row.get("model_probability")
    market = row.get("market_fair_probability", row.get("market_probability"))
    execution = row.get("execution_price")
    edge = row.get("edge")
    ai = row.get("ai_analysis") or {}
    final = row.get("final_probability", model); adjusted = row.get("adjusted_edge", edge)
    lines = [str(event), "", "Quant：", f"{outcome or 'UNKNOWN'} {float(model):.1%}" if model is not None else "UNKNOWN",
             "", "AI Adjustment：", f"{float(row.get('ai_adjustment') or 0):+.1%}",
             "", "Final：", f"{float(final):.1%}" if final is not None else "UNKNOWN",
             "", "Market Fair：", f"{float(market):.1%}" if market is not None else "UNKNOWN",
             "", "Execution Price：", f"{float(execution):.1%}" if execution is not None else "UNKNOWN",
             "", "Adjusted Edge：", f"{float(adjusted):.1%}" if adjusted is not None else "UNKNOWN"]
    factors = ai.get("factors") or []
    if factors:
        lines.extend(["", "主要证据："])
        for index, factor in enumerate(factors[:3], 1):
            lines.append(f"{index}. [{factor.get('evidence_id')}] {factor.get('reason')}")
    risks = ai.get("risk_flags") or ([row.get("decision_gate")] if row.get("decision_gate") != "PASS" else [])
    if risks:
        lines.extend(["", "主要风险：", *[f"- {value}" for value in risks if value]])
    lines.extend(["", "数据质量：", str(row.get("data_quality_level") or "UNKNOWN"), "", "缺失数据："])
    missing = row.get("data_quality_missing") or []
    lines.extend([f"- {value}" for value in missing] or ["- 无"])
    lines.extend(["", "结论：", str(row.get("recommendation_state") or "WATCH")])
    if row.get("ai_status") != "LLM_ACTIVE":
        lines.extend(["", "LLM_NOT_ACTIVE：当前为 Quant fallback，不是 AI Analysis。"])
    return "\n".join(lines)


def build_post_match_summary(row: dict) -> str:
    event = row.get("event", "unknown")
    actual = row.get("actual_winner")
    predicted = row.get("predicted_winner")
    correct = row.get("prediction_correct")
    model = row.get("model_probability")
    parts = [f"{event}：实际胜者 {actual or '未知'}，赛前方向 {predicted or '未知'}。"]
    if correct is not None:
        parts.append("判断正确。" if correct else "判断错误。")
    if model is not None:
        parts.append(f"模型赛前胜率 {float(model):.1%}。")
    if row.get("brier_score") is not None:
        parts.append(f"Brier {float(row['brier_score']):.4f}。")
    return "".join(parts)
