from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from .live_engine import LiveAlert

from .providers.http import post_json


Transport = Callable[..., Any]


def _validate_post_text(post: dict[str, Any]) -> None:
    """Reject text that was damaged by a shell/console encoding conversion."""
    suspicious_fragments = ("??", "\ufffd", "锛", "鈥", "馃")

    def walk(value: Any) -> None:
        if isinstance(value, str) and any(fragment in value for fragment in suspicious_fragments):
            raise ValueError("Feishu message contains corrupted text; refusing to send")
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(post)


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

    def send_post(self, post: dict[str, Any]) -> list[dict[str, Any]]:
        """Send one Feishu rich-text post through a v2 custom-bot webhook."""
        _validate_post_text(post)
        payload: dict[str, object] = {"msg_type": "post", "content": {"post": post}}
        if self.secret:
            timestamp = int(time.time())
            payload.update({"timestamp": str(timestamp), "sign": webhook_signature(timestamp, self.secret)})
        response = self.transport(self.webhook_url, payload)
        if isinstance(response, dict) and response.get("code", response.get("StatusCode", 0)) != 0:
            raise RuntimeError(f"Feishu webhook rejected message: {response}")
        return [response]


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

    def send_post(self, post: dict[str, Any]) -> list[dict[str, Any]]:
        """Send one Feishu rich-text post through the app messaging API."""
        _validate_post_text(post)
        token = self._tenant_token()
        url = f"https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type={self.receive_id_type}"
        payload = {
            "receive_id": self.receive_id,
            "msg_type": "post",
            "content": json.dumps(post, ensure_ascii=False),
        }
        response = self.transport(url, payload, headers={"Authorization": f"Bearer {token}"})
        if response.get("code", 0) != 0:
            raise RuntimeError(f"Feishu API rejected message: {response}")
        return [response]


SPORT_NAMES = {"lol": "英雄联盟", "cs2": "CS2", "nba": "NBA"}


def _sport_name(value: Any) -> str:
    key = str(value or "").casefold()
    return SPORT_NAMES.get(key, str(value or "未知项目").upper())


def _event_name(value: Any) -> str:
    text = str(value or "未知赛事")
    for prefix in ("LoL: ", "LOL: ", "CS2: ", "NBA: "):
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
    return text.replace(" vs ", " 对 ")


def _zh_reason(reason: Any) -> str:
    text = str(reason).strip()
    exact = {
        "confidence below threshold": "置信度未达到要求",
        "strategy not approved for trading": "策略尚未通过实盘验收",
        "cost-adjusted edge below threshold": "扣除成本后，优势未达到要求",
        "probability model is not approved": "概率模型尚未通过验收",
        "market comparison is invalid": "当前盘口不适合比较",
    }
    if text in exact:
        return exact[text]
    prefixes = {
        "uncertainty-adjusted edge=": "不确定性调整后优势：",
        "cost-adjusted edge=": "扣除成本后优势：",
        "net EV=": "净期望值：",
    }
    for prefix, replacement in prefixes.items():
        if text.startswith(prefix):
            return replacement + text.removeprefix(prefix)
    return text


def _key_reasons(row: dict[str, Any], limit: int = 3) -> list[str]:
    reasons = [_zh_reason(reason) for reason in row.get("reasons", []) if str(reason).strip()]
    rejection_markers = ("未达到", "尚未", "不适合", "不可", "缺失", "不足", "过期")
    selected = [reason for reason in reasons if any(marker in reason for marker in rejection_markers)]
    selected.extend(reason for reason in reasons if reason not in selected)
    return list(dict.fromkeys(selected))[:limit]


def _zh_live_text(value: Any) -> str:
    text = str(value or "")
    replacements = (
        (r"^model ([^ ]+) -> ([^;]+); market (.+)$", r"模型胜率：\1 → \2｜市场胜率：\3"),
        (r"^score differential ([^,]+), estimated regulation time remaining (.+)$", r"分差：\1｜预计常规时间剩余：\2"),
        (r"^BP/player-champion model changed prior (.+)$", r"选人及选手英雄模型调整：\1"),
        (r"^gold ([^,]+), kills ([^,]+), towers ([^,]+), dragons ([^,]+), barons (.+)$", r"经济差 \1｜击杀差 \2｜防御塔差 \3｜小龙差 \4｜大龙差 \5"),
        (r"^series maps ([^,]+), current-map rounds (.+)$", r"系列赛地图差 \1｜当前地图回合差 \2"),
        (r"^model-market divergence (.+)$", r"模型与市场分歧：\1"),
        (r"^market move since prior snapshot ([^;]+); spread (.+)$", r"市场概率变化：\1｜买卖价差：\2"),
        (r"^market volume increased by (.+)$", r"市场成交量增加：\1"),
    )
    for pattern, replacement in replacements:
        if re.match(pattern, text):
            text = re.sub(pattern, replacement, text)
            break
    return text.replace("unavailable", "暂无数据")


def format_daily_report(report: dict[str, Any], report_date: date | None = None) -> str:
    def percent(value: Any) -> str:
        return "不可比较" if value is None else f"{float(value):.1%}"

    def price(value: Any) -> str:
        return "不可用" if value is None else f"{float(value):.3f}"

    day = report_date or (date.fromisoformat(report["report_date"]) if report.get("report_date") else date.today())
    rows = sorted(report.get("recommendations", []), key=lambda row: row.get("action") != "BET")
    opportunities = sum(row.get("action") == "BET" for row in rows)
    coverage = report.get("schedule_coverage", {})
    expected = sum(value.get("expected", 0) for value in coverage.values())
    watching = sum(value.get("watching", 0) for value in coverage.values())
    incomplete = bool(report.get("data_incomplete"))
    lines = [
        f"每日赛事研究｜{day.isoformat()}",
        "【赛事覆盖】",
        *(f"{'✅' if float(value.get('coverage', 0)) >= 1 else '⚠️'} {_sport_name(sport)}　"
          f"预计 {value.get('expected', 0)}｜发现 {value.get('discovered', 0)}｜"
          f"市场 {value.get('market_matched', 0)}｜监控 {value.get('watching', 0)}"
          for sport, value in coverage.items()),
        f"合计　预计 {expected}｜已发现 {sum(value.get('discovered', 0) for value in coverage.values())}｜监控 {watching}",
        *( ["🚨 数据不完整：存在未解释的赛事遗漏"] if incomplete else ["✅ 今日目标赛事已全部覆盖"] ),
        "",
    ]
    for sport, value in coverage.items():
        if value.get("source_warning"):
            lines.append(f"⚠️ {_sport_name(sport)}：多个赛程源结果不一致，请查看赛程审计。")
        for missing in value.get("missing", []):
            lines.append(
                f"• 遗漏：{missing.get('league')}｜{missing.get('team_a')} 对 {missing.get('team_b')}｜"
                f"阶段：{missing.get('missing_stage') or missing.get('watcher_status') or '未知'}｜"
                f"原因：{missing.get('missing_reason') or '监控器不可用'}"
            )
    lines.extend(["", "【研究结论】", f"分析 {len(rows)} 场｜符合策略 {opportunities} 场｜其余仅观察"])
    if not rows:
        lines.append("暂无达到策略与风控要求的机会。")
    for index, row in enumerate(rows, 1):
        is_bet = row.get("action") == "BET"
        action = "暂不参与" if not is_bet else "符合策略"
        marker = "⭐" if is_bet else "▫️"
        reasons = _key_reasons(row)
        lines.extend([
            "",
            f"{marker} {index}. {_sport_name(row.get('sport'))}｜{_event_name(row.get('event', row.get('event_id', '未知赛事')))}",
            f"结论：{action}｜研究方向：{row.get('outcome', '-')}｜可买价：{price(row.get('execution_price'))}",
            f"模型胜率 {percent(row.get('model_probability'))}｜市场胜率 {percent(row.get('market_probability'))}｜净优势 {percent(row.get('edge'))}｜净期望值 {percent(row.get('expected_value'))}",
            *(f"• {reason}" for reason in reasons),
        ])
    bankroll = report.get("bankroll_usdc")
    bankroll_text = f"{float(bankroll):.2f} USDC" if bankroll is not None else "等待当日汇率"
    lines.extend([
        "",
        "【风控状态】",
        f"研究本金：{bankroll_text}",
        "单次上限 0.75%｜单日上限 2.5%｜单赛事上限 1.0%｜回撤 10% 暂停",
        "仅供研究与模拟记录，不构成投资或下注建议。",
    ])
    risks = report.get("risk_notes", [])
    if risks:
        lines.append("补充提示：" + "；".join(str(x) for x in risks[:2]))
    return "\n".join(lines).strip()


def format_daily_post(report: dict[str, Any], report_date: date | None = None) -> dict[str, Any]:
    """Build Feishu's localized rich-text post document from the readable text report."""
    lines = format_daily_report(report, report_date).splitlines()
    title = lines.pop(0) if lines else "每日赛事研究"
    content: list[list[dict[str, Any]]] = []
    for line in lines:
        if not line:
            continue
        element: dict[str, Any] = {"tag": "text", "text": line}
        content.append([element])
    return {"zh_cn": {"title": title, "content": content}}


def format_live_alert(alert: LiveAlert | dict[str, Any]) -> str:
    row = alert.as_dict() if isinstance(alert, LiveAlert) else alert
    icon = "🔴" if row.get("severity") == "EMERGENCY" else "🟠" if row.get("severity") == "IMPORTANT" else "🟡"
    severity = {"EMERGENCY": "紧急", "IMPORTANT": "重要", "OBSERVE": "关注", "NORMAL": "关注"}.get(
        str(row.get("severity")), str(row.get("severity") or "关注")
    )
    category = {
        "MARKET_ANOMALY": "盘口异常", "MAJOR_EVENT": "重大事件", "PROBABILITY_CHANGE": "概率变化",
        "NEWS_ALERT": "阵容 / 新闻异常",
        "PREMATCH_ANALYSIS": "赛前分析", "DRAFT_ANALYSIS": "BP 完成分析",
        "MATCH_START": "比赛开始", "PERIOD_UPDATE": "节次更新", "CLUTCH_TIME": "关键时段",
        "MATCH_FINISHED": "比赛结束", "WATCHER_MISSING": "监控缺失",
        "MONITORING_RECOVERY": "监控恢复",
    }.get(str(row.get("category")), str(row.get("category")))
    return "\n".join([
        f"{icon}【{category}】",
        f"{_sport_name(row.get('sport'))}｜{_event_name(row.get('title'))}",
        f"重要度：{float(row.get('alert_score', 0)):.0f}/100｜级别：{severity}",
        _zh_live_text(row.get("summary")),
        "关键原因：",
        *(f"• {_zh_live_text(reason)}" for reason in row.get("reasons", [])[:4]),
        "",
        "研究监控信号，不构成下注建议。",
    ]).strip()
