from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

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
    day = report_date or date.today()
    rows = sorted(report.get("recommendations", []), key=lambda row: row.get("action") != "BET")
    bankroll = report.get("bankroll_usdc")
    bankroll_line = f"折算本金：{float(bankroll):.2f} USDC" if bankroll is not None else "折算本金：等待当日汇率"
    opportunities = sum(row.get("action") == "BET" for row in rows)
    lines = [
        f"赛事研究日报｜{day.isoformat()}",
        bankroll_line,
        f"覆盖赛事：{len(rows)} 场｜合适机会：{opportunities} 场",
        "风控：单笔≤0.75%｜单日≤2.5%｜单赛事≤1.0%｜回撤10%熔断",
        "",
        "仅供研究，不保证收益。请以实际可成交盘口为准。",
        "",
    ]
    if not rows:
        lines.append("今日没有达到风控阈值的下注机会（NO BET）。")
    for index, row in enumerate(rows, 1):
        is_bet = row.get("action") == "BET"
        action = "观察 / 不下注" if not is_bet else "★ 合适机会"
        marker = "[重点]" if is_bet else "[观察]"
        lines.extend([
            f"{marker} {index}. [{str(row.get('sport', '')).upper()}] {row.get('event', row.get('event_id', '未知赛事'))}",
            f"方向：{row.get('outcome', '-')}｜结论：{action}",
            f"模型概率：{float(row.get('model_probability', 0)):.1%}｜市场概率：{float(row.get('market_probability', 0)):.1%}",
            f"净优势：{float(row.get('edge', 0)):.1%}｜金额：{float(row.get('stake', 0)):.2f}",
            f"依据：{'；'.join(row.get('reasons', [])) or '无'}",
            "",
        ])
    risks = report.get("risk_notes", [])
    if risks:
        lines.append("风险提示：" + "；".join(str(x) for x in risks))
    return "\n".join(lines).strip()
