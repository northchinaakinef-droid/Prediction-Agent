"""Read-only natural-language summaries for reports.

This module intentionally imports neither ``risk`` nor anything that mutates a
``Recommendation``.  It only reads plain row dictionaries after ``recommend()``
has already produced numeric outputs.
"""
from __future__ import annotations


def build_pre_match_summary(row: dict) -> str:
    sport = row.get("sport", "unknown")
    event = row.get("event", "unknown")
    outcome = row.get("outcome")
    model = row.get("model_probability")
    market = row.get("market_probability")
    edge = row.get("edge")
    reasons = row.get("reasons") or []
    parts = [f"{event} 赛前摘要：预测方向 {outcome or '未知'}。"]
    if model is not None and market is not None:
        parts.append(f"模型胜率 {float(model):.1%}，市场胜率 {float(market):.1%}。")
    if edge is not None:
        parts.append(f"净优势 {float(edge):.1%}。")
    if reasons:
        parts.append("主要依据：" + "；".join(str(item) for item in reasons[:3]) + "。")
    parts.append("该摘要为解释性文本，不参与任何数值或下注计算。")
    return "".join(parts)


def build_post_match_summary(row: dict) -> str:
    event = row.get("event", "unknown")
    actual = row.get("actual_winner")
    predicted = row.get("predicted_winner")
    correct = row.get("prediction_correct")
    model = row.get("model_probability")
    parts = [f"{event} 赛后摘要：实际胜者 {actual or '未知'}，赛前预测 {predicted or '未知'}。"]
    if correct is not None:
        parts.append("判断正确。" if correct else "判断错误。")
    if model is not None:
        parts.append(f"模型赛前胜率 {float(model):.1%}。")
    parts.append("该摘要为解释性文本，不参与任何数值或下注计算。")
    return "".join(parts)
