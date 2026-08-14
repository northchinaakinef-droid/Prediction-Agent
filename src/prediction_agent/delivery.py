from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from .live_engine import LiveAlert

from .providers.http import post_json


Transport = Callable[..., Any]


def _chunks(text: str, limit: int = 12_000) -> list[str]:
    if limit < 1:
        raise ValueError("limit must be positive")
    return [text[i:i + limit] for i in range(0, len(text), limit)] or [""]


def webhook_signature(timestamp: int, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}".encode("utf-8")
    digest = hmac.new(string_to_sign, digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("ascii")


@dataclass
class FeishuWebhookClient:
    webhook_url: str
    secret: str | None = None
    transport: Transport = post_json

    def send_text(self, text: str) -> list[dict[str, Any]]:
        responses = []
        for chunk in _chunks(text):
            payload: dict[str, object] = {"msg_type": "text", "content": {"text": chunk}}
            if self.secret:
                timestamp = int(time.time())
                payload.update({"timestamp": str(timestamp), "sign": webhook_signature(timestamp, self.secret)})
            response = self.transport(self.webhook_url, payload)
            if isinstance(response, dict) and response.get("code", response.get("StatusCode", 0)) != 0:
                raise RuntimeError(f"Feishu webhook rejected message: {response}")
            responses.append(response)
        return responses


@dataclass
class FeishuAppClient:
    app_id: str
    app_secret: str
    receive_id: str
    receive_id_type: str = "open_id"
    transport: Transport = post_json

    def _tenant_token(self) -> str:
        response = self.transport(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            {"app_id": self.app_id, "app_secret": self.app_secret},
        )
        if response.get("code", 0) != 0 or not response.get("tenant_access_token"):
            raise RuntimeError(f"failed to obtain Feishu tenant token: {response}")
        return str(response["tenant_access_token"])

    def send_text(self, text: str) -> list[dict[str, Any]]:
        token = self._tenant_token()
        url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={self.receive_id_type}"
        responses = []
        for chunk in _chunks(text):
            payload = {
                "receive_id": self.receive_id,
                "msg_type": "text",
                "content": json.dumps({"text": chunk}, ensure_ascii=False),
            }
            response = self.transport(url, payload, headers={"Authorization": f"Bearer {token}"})
            if response.get("code", 0) != 0:
                raise RuntimeError(f"Feishu API rejected message: {response}")
            responses.append(response)
        return responses


def format_daily_report(report: dict[str, Any], report_date: date | None = None) -> str:
    def percent(value: Any) -> str:
        return "不可比较" if value is None else f"{float(value):.1%}"

    def price(value: Any) -> str:
        return "不可用" if value is None else f"{float(value):.3f}"

    day = report_date or (date.fromisoformat(report["report_date"]) if report.get("report_date") else date.today())
    rows = sorted(report.get("recommendations", []), key=lambda row: row.get("action") != "BET")
    bankroll = report.get("bankroll_usdc")
    bankroll_line = f"折算本金：{float(bankroll):.2f} USDC" if bankroll is not None else "折算本金：等待当日汇率"
    opportunities = sum(row.get("action") == "BET" for row in rows)
    coverage = report.get("schedule_coverage", {})
    expected = sum(value.get("expected", 0) for value in coverage.values())
    watching = sum(value.get("watching", 0) for value in coverage.values())
    incomplete = bool(report.get("data_incomplete"))
    lines = [
        f"赛事研究日报｜{day.isoformat()}",
        "【赛事覆盖检查】",
        *(f"{sport.upper()}：已捕获 {value.get('discovered', 0)} / 预计 {value.get('expected', 0)}｜"
          f"市场 {value.get('market_matched', 0)}｜监控 {value.get('watching', 0)}｜"
          f"Schedule Coverage {float(value.get('coverage', 0)):.0%}｜"
          f"Market Coverage {float(value.get('market_coverage', 0)):.0%}"
          for sport, value in coverage.items()),
        f"总赛事：已捕获 {sum(value.get('discovered', 0) for value in coverage.values())} / 预计 {expected}｜正在监控 {watching}",
        *( ["🚨 DATA INCOMPLETE｜今日赛事存在未解释遗漏"] if incomplete else ["赛事覆盖：100% ✅"] ),
        bankroll_line,
        f"已生成研究分析：{len(rows)} 场｜合适机会：{opportunities} 场",
        "风控：单笔≤0.75%｜单日≤2.5%｜单赛事≤1.0%｜回撤10%熔断",
        "",
        "仅供研究，不保证收益。请以实际可成交盘口为准。",
        "",
    ]
    for sport, value in coverage.items():
        if value.get("source_warning"):
            lines.append(f"⚠️ {sport.upper()} 赛程源存在差异，已由市场层交叉验证；详见 schedule audit。")
        for missing in value.get("missing", []):
            lines.append(
                f"遗漏：{missing.get('league')}｜{missing.get('team_a')} vs {missing.get('team_b')}｜"
                f"阶段={missing.get('missing_stage') or missing.get('watcher_status')}｜"
                f"原因={missing.get('missing_reason') or 'watcher unavailable'}"
            )
    if coverage:
        lines.append("")
    status = report.get("model_status", {})
    if status:
        lines.extend([
            f"模型：独立胜率={'是' if status.get('independent_probability') else '否'}｜"
            f"概率验收={'通过' if status.get('probability_approved') else '未通过'}｜"
            f"真钱验收={'通过' if status.get('real_money_approved') else '未通过'}",
            f"训练截至：{status.get('trained_through', '-')}｜样本：{status.get('samples', 0)}",
            "",
        ])
    sport_status = report.get("sport_status", {})
    if sport_status:
        lines.append("模型状态：" + "｜".join(
            f"{sport.upper()}={'就绪' if value.get('ready') else '未就绪'}"
            for sport, value in sport_status.items()
        ))
        lines.append("")
    if not rows:
        lines.append("今日没有达到风控阈值的下注机会（NO BET）。")
    for index, row in enumerate(rows, 1):
        is_bet = row.get("action") == "BET"
        action = "观察 / 不下注" if not is_bet else "★ 合适机会"
        marker = "[重点]" if is_bet else "[观察]"
        lines.extend([
            f"{marker} {index}. [{str(row.get('sport', '')).upper()}] {row.get('event', row.get('event_id', '未知赛事'))}",
            f"方向：{row.get('outcome', '-')}｜结论：{action}",
            f"独立模型概率：{percent(row.get('model_probability'))}｜市场概率：{percent(row.get('market_probability'))}",
            f"风控后概率：{percent(row.get('decision_probability'))}｜原始优势：{percent(row.get('raw_edge'))}",
            f"成本后净优势：{percent(row.get('edge'))}｜净EV：{percent(row.get('expected_value'))}｜建议金额：{float(row.get('stake', 0)):.2f} USDC",
            f"市场可买价：{price(row.get('execution_price'))}",
            f"依据：{'；'.join(row.get('reasons', [])) or '无'}",
            "",
        ])
    risks = report.get("risk_notes", [])
    if risks:
        lines.append("风险提示：" + "；".join(str(x) for x in risks))
    return "\n".join(lines).strip()


def format_live_alert(alert: LiveAlert | dict[str, Any]) -> str:
    row = alert.as_dict() if isinstance(alert, LiveAlert) else alert
    icon = "🔴" if row.get("severity") == "EMERGENCY" else "🟠" if row.get("severity") == "IMPORTANT" else "🟡"
    category = {
        "MARKET_ANOMALY": "盘口异常", "MAJOR_EVENT": "重大事件", "PROBABILITY_CHANGE": "概率变化",
        "NEWS_ALERT": "阵容 / 新闻异常",
    }.get(str(row.get("category")), str(row.get("category")))
    return "\n".join([
        f"{icon}【{category}】",
        f"{str(row.get('sport', '')).upper()}｜{row.get('title')}",
        f"Alert Score：{float(row.get('alert_score', 0)):.0f}/100｜级别：{row.get('severity')}",
        str(row.get("summary") or ""),
        "关键原因：",
        *(f"• {reason}" for reason in row.get("reasons", [])),
        "",
        "研究监控信号，不构成下注建议。",
    ]).strip()
