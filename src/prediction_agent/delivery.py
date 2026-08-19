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
from .context import format_lineup_section

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


def _display_team(value: Any) -> str:
    """Clean provider team labels for display only."""
    text = str(value or "").strip()
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]", "", text)
    text = re.sub(r"\s*\(BO\d+\)\s*-\s*.*$", "", text)
    text = re.sub(r"\.(CN|EU|KR|NA|TW|VN|JP|OCE|BR|TR|CIS|MENA|LATAM|SEA|AM|US|UK|AU)\b.*$", "", text)
    return " ".join(text.split()) or str(value or "未知队伍")


def _event_name(value: Any) -> str:
    text = str(value or "未知赛事")
    for prefix in ("LoL: ", "LOL: ", "CS2: ", "NBA: "):
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
    parts = re.split(r"\s*(?:vs\.?|VS\.?|对)\s*", text, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        return f"{_display_team(parts[0])} 对 {_display_team(parts[1])}"
    return _display_team(text)


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


def _event_team_pair(row: dict[str, Any]) -> tuple[str, str]:
    event_text = _event_name(row.get("event", row.get("event_id", "未知赛事")))
    parts = re.split(r"\s+对\s+", event_text, maxsplit=1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return event_text, "另一方"


def _two_sided_probability(model: Any, outcome: Any, team_a: str, team_b: str) -> str:
    if model is None:
        return "不可比较"
    model = float(model)
    outcome = _display_team(outcome)
    if outcome and outcome == team_a:
        return f"{team_a} {model:.1%}｜{team_b} {1 - model:.1%}"
    if outcome and outcome == team_b:
        return f"{team_a} {1 - model:.1%}｜{team_b} {model:.1%}"
    return f"{outcome or '未知方向'} {model:.1%}"


def format_daily_report(report: dict[str, Any], report_date: date | None = None) -> str:
    def percent(value: Any) -> str:
        return "不可比较" if value is None else f"{float(value):.1%}"

    day = report_date or (date.fromisoformat(report["report_date"]) if report.get("report_date") else date.today())
    rows = report.get("recommendations", [])
    bets = [row for row in rows if row.get("action") == "BET"]
    skipped_rows = [row for row in rows if row.get("action") != "BET"]
    skipped = len(skipped_rows)
    risk_status = report.get("risk_status") or {}
    circuit_breaker = bool(risk_status.get("circuit_breaker"))
    drawdown = risk_status.get("current_drawdown")

    def risk_status_text() -> str:
        if circuit_breaker:
            return "熔断（已暂停）"
        if risk_status.get("drawdown_level") == "warn":
            return "警戒（额度减半）"
        return "正常"

    lines = [f"【今日模拟下注】{len(bets)}场"]
    separator = "━━━━━━━━━━━━━━━━"
    if bets:
        lines.append(separator)
        for index, row in enumerate(bets, 1):
            event = row.get("event") or row.get("event_id") or "未知赛事"
            outcome = row.get("outcome") or "-"
            stake = float(row.get("stake") or 0)
            status_label = {"虚拟下注": "【虚拟下注】", "真实建议": "【下注建议】"}.get(
                str(row.get("bet_status") or ""), "")
            lines.append(f"[{index}]{status_label} {event} | {outcome} | {stake:.2f} USDC")
            lines.append(
                f"    EV: {percent(row.get('expected_value'))} | "
                f"模型: {percent(row.get('model_probability'))} vs 市场: {percent(row.get('market_probability'))}"
            )
            lines.append(f"    阵容状态: {row.get('lineup_status') or '未知'} | "
                         f"下注状态: {row.get('bet_status') or '跳过'}")
            lines.append(separator)

    if skipped_rows:
        lines.append("")
        lines.append(f"【今日跳过明细】{len(skipped_rows)}场")
        for index, row in enumerate(skipped_rows, 1):
            event = _event_name(row.get("event") or row.get("event_id") or "未知赛事")
            outcome = row.get("outcome") or "-"
            lines.append(f"[{index}] {event} | {outcome}")
            if circuit_breaker:
                lines.append("    卡住条件: 账户熔断（全部暂停）")
                continue
            ev = percent(row.get("expected_value"))
            model = percent(row.get("model_probability"))
            market = percent(row.get("market_probability"))
            reasons = _key_reasons(row, limit=3)
            reason_text = "；".join(reasons) if reasons else "无明确拒绝原因"
            lines.append(f"    EV: {ev} | 模型: {model} vs 市场: {market}")
            lines.append(f"    阵容: {row.get('lineup_status') or '未知'} | "
                         f"赛程匹配: {row.get('market_mapping_status') or 'NOT_IN_SCHEDULE'} | "
                         f"卡住条件: {reason_text}")

    bankroll = report.get("bankroll_usdc")
    paper_daily = report.get("paper_daily") or {}
    remaining = paper_daily.get("remaining_limit")
    if remaining is None and bankroll is not None:
        total_stake = sum(float(row.get("stake") or 0) for row in bets)
        remaining = max(0.0, float(bankroll) * 0.025 - total_stake)

    drawdown_text = f"{float(drawdown):.1%}" if isinstance(drawdown, (int, float)) else "0.0%"
    remaining_text = f"{float(remaining):.2f} USDC" if isinstance(remaining, (int, float)) else "0.00 USDC"
    lines.append("")
    if circuit_breaker:
        lines.append(f"【今日跳过】{skipped}场（账户熔断（全部暂停））")
    else:
        skip_parts = []
        for sport, stat in (paper_daily.get("by_sport") or {}).items():
            skip_count = int(stat.get("skipped") or 0)
            if skip_count:
                skip_parts.append(f"{_sport_name(sport)} {skip_count}场")
        skip_detail = "｜".join(skip_parts) if skip_parts else "无分项明细"
        lines.append(f"【今日跳过】{skipped}场（{skip_detail}；EV>5%、方向一致性或风险预算未达标）")
    lines.append(f"【虚拟账户历史回撤】{drawdown_text} | 剩余额度: {remaining_text}")
    virtual_betting = report.get("virtual_betting") or report.get("risk_status", {}).get("virtual_betting") or {}
    if virtual_betting:
        v_count = int(virtual_betting.get("count", 0))
        v_roi = virtual_betting.get("roi")
        v_balance = virtual_betting.get("balance")
        v_roi_text = "暂无" if v_roi is None else f"{float(v_roi):.1%}"
        v_balance_text = "暂无" if v_balance is None else f"{float(v_balance):.2f} USDC"
        lines.append(f"【虚拟积累】{v_count}/100场 | 虚拟ROI: {v_roi_text} | 虚拟余额: {v_balance_text}")
    lines.append(f"【风控状态】{risk_status_text()}")
    return "\n".join(lines).strip()

def format_paper_betting_summary(summary: dict[str, Any], report_date: str,
                                     bankroll: float) -> str:
    """Format one unified daily paper-betting summary for Feishu text delivery."""
    def roi_text(value: Any) -> str:
        return f"{float(value):.1%}" if isinstance(value, (int, float)) else "暂无"

    by_sport = summary.get("by_sport") or {}
    lines = [
        f"📊 虚拟投注日报｜{report_date}",
        "",
        f"研究本金：{float(bankroll):.2f} USDC",
        f"今日预测：{int(summary.get('predictions', 0))} 场｜"
        f"虚拟下注：{int(summary.get('bet_candidates', 0))} 场｜"
        f"已结算：{int(summary.get('settled_predictions', 0))} 场",
        f"今日模拟盈亏：{float(summary.get('paper_profit', 0)):+.2f} USDC｜"
        f"ROI：{roi_text(summary.get('paper_roi'))}",
    ]
    for sport, stat in by_sport.items():
        lines.append(
            f"{_sport_name(sport)}：下注 {int(stat.get('bet_candidates', 0))} 场｜"
            f"结算 {int(stat.get('settled_predictions', 0))} 场｜"
            f"盈亏 {float(stat.get('paper_profit', 0)):+.2f} USDC｜"
            f"ROI {roi_text(stat.get('paper_roi'))}"
        )
    lines.extend(["", "仅用于虚拟投注复盘与模型迭代，不涉及真实资金。"])
    return "\n".join(lines).strip()


def format_attribution_report(attribution: dict[str, Any], report_date: str | None = None) -> str:
    """Format the weekly cold-start attribution table for Feishu text delivery."""
    def win_rate_text(value: Any) -> str:
        return "暂无" if value is None else f"{float(value):.1%}"

    def roi_text(value: Any) -> str:
        return "暂无" if value is None else f"{float(value):+.1%}"

    day = report_date or date.today().isoformat()
    lines = [f"【每周归因报告】{day}", "━━━━━━━━━━━━━━━━"]
    sections = (
        ("按数据完整度", "by_data_quality", None),
        ("按阵容状态", "by_lineup_status", None),
        ("按 EV 等级", "by_ev_tier", None),
        ("按方向一致性", "by_direction_match", None),
        ("按赛事类型", "by_sport", _sport_name),
    )
    for title, key, name_mapper in sections:
        groups = attribution.get(key) or {}
        lines.append(f"【{title}】")
        if not groups:
            lines.append("暂无结算样本")
        for name, stat in groups.items():
            label = name_mapper(name) if name_mapper else str(name)
            lines.append(
                f"{label}：胜率 {win_rate_text(stat.get('win_rate'))} | "
                f"ROI {roi_text(stat.get('roi'))} | 样本 {int(stat.get('bets', 0))} 场"
            )
        lines.append("")
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


ROLE_CN = {"top": "上路", "jng": "打野", "mid": "中路", "bot": "下路", "sup": "辅助"}

CHAMPION_CN = {
    "Aatrox": "亚托克斯",
    "Ahri": "阿狸",
    "Akali": "阿卡丽",
    "Akshan": "阿克尚",
    "Alistar": "阿利斯塔",
    "Ambessa": "安蓓萨",
    "Amumu": "阿木木",
    "Anivia": "艾尼维亚",
    "Annie": "安妮",
    "Aphelios": "厄斐琉斯",
    "Ashe": "艾希",
    "Aurelion Sol": "奥瑞利安·索尔",
    "Aurora": "阿萝拉",
    "Azir": "阿兹尔",
    "Bard": "巴德",
    "Bel'Veth": "卑尔维斯",
    "Blitzcrank": "布里茨",
    "Brand": "布兰德",
    "Braum": "布隆",
    "Briar": "贝蕾亚",
    "Caitlyn": "凯特琳",
    "Camille": "卡蜜尔",
    "Cassiopeia": "卡西奥佩娅",
    "Cho'Gath": "科加斯",
    "Corki": "库奇",
    "Darius": "德莱厄斯",
    "Diana": "黛安娜",
    "Dr. Mundo": "蒙多医生",
    "Draven": "德莱文",
    "Ekko": "艾克",
    "Elise": "伊莉丝",
    "Evelynn": "伊芙琳",
    "Ezreal": "伊泽瑞尔",
    "Fiddlesticks": "费德提克",
    "Fiora": "菲奥娜",
    "Fizz": "菲兹",
    "Galio": "加里奥",
    "Gangplank": "普朗克",
    "Garen": "盖伦",
    "Gnar": "纳尔",
    "Gragas": "古拉加斯",
    "Graves": "格雷福斯",
    "Gwen": "格温",
    "Hecarim": "赫卡里姆",
    "Heimerdinger": "黑默丁格",
    "Hwei": "彗",
    "Illaoi": "俄洛伊",
    "Irelia": "艾瑞莉娅",
    "Ivern": "艾翁",
    "Janna": "迦娜",
    "Jarvan IV": "嘉文四世",
    "Jax": "贾克斯",
    "Jayce": "杰斯",
    "Jhin": "烬",
    "Jinx": "金克丝",
    "K'Sante": "奎桑提",
    "Kai'Sa": "卡莎",
    "Kalista": "卡莉丝塔",
    "Karma": "卡尔玛",
    "Karthus": "卡尔萨斯",
    "Kassadin": "卡萨丁",
    "Katarina": "卡特琳娜",
    "Kayle": "凯尔",
    "Kayn": "凯隐",
    "Kennen": "凯南",
    "Kha'Zix": "卡兹克",
    "Kindred": "千珏",
    "Kled": "克烈",
    "Kog'Maw": "克格莫",
    "LeBlanc": "乐芙兰",
    "Lee Sin": "李青",
    "Leona": "蕾欧娜",
    "Lillia": "莉莉娅",
    "Lissandra": "丽桑卓",
    "Lucian": "卢锡安",
    "Lulu": "璐璐",
    "Lux": "拉克丝",
    "Malphite": "墨菲特",
    "Malzahar": "玛尔扎哈",
    "Maokai": "茂凯",
    "Master Yi": "易大师",
    "Mel": "梅尔",
    "Milio": "米利欧",
    "Miss Fortune": "厄运小姐",
    "Mordekaiser": "莫德凯撒",
    "Morgana": "莫甘娜",
    "Naafiri": "娜菲芮",
    "Nami": "娜美",
    "Nasus": "内瑟斯",
    "Nautilus": "诺提勒斯",
    "Neeko": "妮蔻",
    "Nidalee": "奈德丽",
    "Nilah": "尼菈",
    "Nocturne": "魔腾",
    "Nunu & Willump": "努努和威朗普",
    "Olaf": "奥拉夫",
    "Orianna": "奥莉安娜",
    "Ornn": "奥恩",
    "Pantheon": "潘森",
    "Poppy": "波比",
    "Pyke": "派克",
    "Qiyana": "奇亚娜",
    "Quinn": "奎因",
    "Rakan": "洛",
    "Rammus": "拉莫斯",
    "Rek'Sai": "雷克塞",
    "Rell": "芮尔",
    "Renata Glasc": "蕾娜塔",
    "Renekton": "雷克顿",
    "Rengar": "雷恩加尔",
    "Riven": "锐雯",
    "Rumble": "兰博",
    "Ryze": "瑞兹",
    "Samira": "莎弥拉",
    "Sejuani": "瑟庄妮",
    "Senna": "赛娜",
    "Seraphine": "萨勒芬妮",
    "Sett": "瑟提",
    "Shaco": "萨科",
    "Shen": "慎",
    "Shyvana": "希瓦娜",
    "Singed": "辛吉德",
    "Sion": "赛恩",
    "Sivir": "希维尔",
    "Skarner": "斯卡纳",
    "Smolder": "斯莫德",
    "Sona": "娑娜",
    "Soraka": "索拉卡",
    "Swain": "斯维因",
    "Sylas": "塞拉斯",
    "Syndra": "辛德拉",
    "Tahm Kench": "塔姆·肯奇",
    "Taliyah": "塔莉垭",
    "Talon": "泰隆",
    "Taric": "塔里克",
    "Teemo": "提莫",
    "Thresh": "锤石",
    "Tristana": "崔丝塔娜",
    "Trundle": "特朗德尔",
    "Tryndamere": "泰达米尔",
    "Twisted Fate": "崔斯特",
    "Twitch": "图奇",
    "Udyr": "乌迪尔",
    "Urgot": "厄加特",
    "Varus": "韦鲁斯",
    "Vayne": "薇恩",
    "Veigar": "维迦",
    "Vel'Koz": "维克兹",
    "Vex": "薇古丝",
    "Vi": "蔚",
    "Viego": "佛耶戈",
    "Viktor": "维克托",
    "Vladimir": "弗拉基米尔",
    "Volibear": "沃利贝尔",
    "Warwick": "沃里克",
    "Wukong": "孙悟空",
    "Xayah": "霞",
    "Xerath": "泽拉斯",
    "Xin Zhao": "赵信",
    "Yasuo": "亚索",
    "Yone": "永恩",
    "Yorick": "约里克",
    "Yuumi": "悠米",
    "Zac": "扎克",
    "Zed": "劫",
    "Zeri": "泽丽",
    "Ziggs": "吉格斯",
    "Zilean": "基兰",
    "Zoe": "佐伊",
    "Zyra": "婕拉",
}


def _champion_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").casefold() if ch.isalnum())


_CHAMPION_CN_BY_KEY = {_champion_key(name): cn for name, cn in CHAMPION_CN.items()}


def _champion_cn(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "-":
        return text
    return _CHAMPION_CN_BY_KEY.get(_champion_key(text), text)




def _draft_edge_lines(lanes: list[dict]) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    weaknesses: list[str] = []
    for lane in lanes:
        edge = float(lane.get("edge") or 0)
        if abs(edge) < 15:
            continue
        role = ROLE_CN.get(str(lane.get("role")), str(lane.get("role")))
        blue_champion = _champion_cn(lane.get("blue_champion")) or "-"
        red_champion = _champion_cn(lane.get("red_champion")) or "-"
        if edge > 0:
            strengths.append(f"• {role}：蓝方 {blue_champion} 对 红方 {red_champion}｜蓝方净优势 +{edge:.0f}")
        else:
            weaknesses.append(f"• {role}：蓝方 {blue_champion} 对 红方 {red_champion}｜红方净优势 +{-edge:.0f}")
    return strengths, weaknesses


def _format_draft_alert(row: dict[str, Any]) -> str:
    icon = "🔴" if row.get("severity") == "EMERGENCY" else "🟠" if row.get("severity") == "IMPORTANT" else "🟡"
    severity = {"EMERGENCY": "紧急", "IMPORTANT": "重要", "OBSERVE": "关注", "NORMAL": "关注"}.get(
        str(row.get("severity")), str(row.get("severity") or "关注"))
    details = row.get("details") or {}
    readout = details.get("readout") or {}
    blue_champions = details.get("blue_champions") or []
    red_champions = details.get("red_champions") or []
    post = details.get("post_draft_probability")
    lanes = readout.get("lanes") or []
    team_edge = float(readout.get("team_edge") or 0)

    lines = [
        f"{icon}【BP 完成分析】",
        f"{_sport_name(row.get('sport'))}｜{_event_name(row.get('title'))}",
        f"重要度：{float(row.get('alert_score', 0)):.0f}/100｜级别：{severity}",
    ]
    if post is not None:
        lines.append(f"BP 后模型胜率：蓝方 {post:.1%}｜红方 {1-post:.1%}")
    team_a = _display_team(details.get("team_a") or "")
    team_b = _display_team(details.get("team_b") or "")
    bans_a = details.get("blue_bans") or []
    bans_b = details.get("red_bans") or []
    if team_a or team_b or bans_a or bans_b:
        lines.extend([
            "",
            "【BP结果】",
            f"{team_a or '蓝方'}：{'、'.join(_champion_cn(value) for value in bans_a) or '-'} / "
            f"选：{'、'.join(_champion_cn(value) for value in blue_champions) or '-'}",
            f"{team_b or '红方'}：{'、'.join(_champion_cn(value) for value in bans_b) or '-'} / "
            f"选：{'、'.join(_champion_cn(value) for value in red_champions) or '-'}",
        ])
    lines.extend([
        "",
        "【双方阵容】",
        f"• 蓝方：{'、'.join(_champion_cn(value) for value in blue_champions) or '-'}",
        f"• 红方：{'、'.join(_champion_cn(value) for value in red_champions) or '-'}",
        "",
        "【阵容优劣】",
    ])
    strengths, weaknesses = _draft_edge_lines(lanes)
    if team_edge > 20:
        strengths.append(f"• 队伍底蕴：蓝方整体略强 +{team_edge:.0f}")
    elif team_edge < -20:
        weaknesses.append(f"• 队伍底蕴：红方整体略强 +{-team_edge:.0f}")
    if strengths:
        lines.extend(strengths)
    if weaknesses:
        lines.extend(weaknesses)
    if not strengths and not weaknesses:
        lines.append("• 各线对位接近，没有明显的单线优势或劣势。")

    lines.extend(["", "【为什么】"])
    if strengths:
        lines.append("• 优势来自英雄熟练度、版本强度与队伍整体功底的叠加。")
    if weaknesses:
        lines.append("• 劣势主要集中在对位强度偏低，或被版本/熟练度克制的线路。")

    strong_blue = [lane for lane in lanes if float(lane.get("edge") or 0) >= 25]
    strong_red = [lane for lane in lanes if float(lane.get("edge") or 0) <= -25]
    lines.extend(["", "【配合与滚雪球】"])
    if strong_blue:
        roles = "、".join(ROLE_CN.get(str(lane.get("role")), str(lane.get("role"))) for lane in strong_blue)
        lines.append(f"• 蓝方应围绕{roles}建立节奏，打野优先保优势路，控先锋/小龙后滚雪球；红方需避战换资源拖发育。")
    elif strong_red:
        roles = "、".join(ROLE_CN.get(str(lane.get("role")), str(lane.get("role"))) for lane in strong_red)
        lines.append(f"• 红方应围绕{roles}建立节奏，打野优先保优势路，控先锋/小龙后滚雪球；蓝方需避战换资源拖发育。")
    else:
        lines.append("• 双方对位接近，优先围绕队伍底蕴更强的半区做资源交换，避免无把握的正面团战。")

    lines.extend(["", "【风险点】"])
    blue_weakest = min(lanes, key=lambda lane: float(lane.get("edge") or 0), default=None)
    red_weakest = max(lanes, key=lambda lane: float(lane.get("edge") or 0), default=None)
    if blue_weakest is not None and float(blue_weakest.get("edge") or 0) < -15:
        role = ROLE_CN.get(str(blue_weakest.get("role")), str(blue_weakest.get("role")))
        lines.append(f"• 蓝方{role}若被针对崩线，容易被滚雪球，最终全盘皆输。")
    if red_weakest is not None and float(red_weakest.get("edge") or 0) > 15:
        role = ROLE_CN.get(str(red_weakest.get("role")), str(red_weakest.get("role")))
        lines.append(f"• 红方{role}若被针对崩线，容易被滚雪球，最终全盘皆输。")
    if not (blue_weakest and red_weakest):
        lines.append("• 当前阵容解读样本有限，暂不给出极端风险判断。")
    lines.extend(["", "研究监控信号，不构成下注建议。"])
    return "\n".join(lines).strip()


def _format_prematch_alert(row: dict[str, Any]) -> str:
    icon = "🔴" if row.get("severity") == "EMERGENCY" else "🟠" if row.get("severity") == "IMPORTANT" else "🟡"
    severity = {"EMERGENCY": "紧急", "IMPORTANT": "重要", "OBSERVE": "关注", "NORMAL": "关注"}.get(
        str(row.get("severity")), str(row.get("severity") or "关注"))
    details = row.get("details") or {}
    sport = str(row.get("sport") or "")
    lines = [
        f"{icon}【赛前分析】",
        f"{_sport_name(sport)}｜{_event_name(row.get('title'))}",
        f"重要度：{float(row.get('alert_score', 0)):.0f}/100｜级别：{severity}",
    ]
    outcome = details.get("outcome")
    team_a = _display_team(details.get("team_a"))
    team_b = _display_team(details.get("team_b"))
    blue = details.get("blue_win_probability")
    red = details.get("red_win_probability")
    if blue is not None and red is not None:
        lines.append(f"赛前预测方向：{_display_team(outcome)}")
        if team_a and team_b:
            lines.append(f"模型胜率：{team_a} {float(blue):.1%}｜{team_b} {float(red):.1%}")
        else:
            lines.append(f"模型胜率：{float(blue):.1%}｜{float(red):.1%}")
    blue_market = details.get("blue_market_probability")
    red_market = details.get("red_market_probability")
    if blue_market is not None and red_market is not None:
        if team_a and team_b:
            lines.append(f"市场胜率：{team_a} {float(blue_market):.1%}｜{team_b} {float(red_market):.1%}")
        else:
            lines.append(f"市场胜率：{float(blue_market):.1%}｜{float(red_market):.1%}")
    else:
        lines.append("市场胜率：暂无对应报价")
    ev = details.get("ev")
    if ev is not None:
        lines.append(f"EV：{float(ev):.1%}")
    lines.extend(["", "【阵容信息】"])
    lines.append(format_lineup_section(
        sport,
        team_a,
        details.get("lineup_a") or [],
        team_b,
        details.get("lineup_b") or [],
    ))
    recent_a = details.get("recent_form_a")
    recent_b = details.get("recent_form_b")
    if recent_a or recent_b:
        lines.extend(["", "【近期状态】（最近10场BO3）"])
        for team, recent in ((team_a, recent_a), (team_b, recent_b)):
            if recent:
                wins = int(recent.get("wins", 0))
                losses = int(recent.get("losses", 0))
                total = max(1, wins + losses)
                lines.append(f"{team}：{wins}胜{losses}负，近期胜率 {wins / total:.1%}")
            else:
                lines.append(f"{team}：近期数据不可用")
    heroes = details.get("patch_meta_heroes") or []
    if sport == "lol" and heroes:
        coverage_a = details.get("meta_coverage_a")
        coverage_b = details.get("meta_coverage_b")
        lines.extend(["", "【版本关键英雄覆盖】（当前赛季强势英雄TOP5）"])
        lines.append("、".join(str(hero) for hero in heroes))
        def _coverage(value):
            return "暂无" if value is None else f"{float(value):.0f}%"
        lines.append(f"{team_a} 覆盖率：{_coverage(coverage_a)} | {team_b} 覆盖率：{_coverage(coverage_b)}")
    elif sport == "cs2":
        lines.extend(["", "【地图池】"])
        format_text = details.get("format") or "BO1"
        maps_a = details.get("map_strengths_a") or []
        maps_b = details.get("map_strengths_b") or []
        lines.append(f"{team_a} 强势图：{'、'.join(maps_a) or '暂无地图池数据'}")
        lines.append(f"{team_b} 强势图：{'、'.join(maps_b) or '暂无地图池数据'}")
        lines.append(f"本场赛制：{format_text}（BO1/BO3）")
    sample_a = details.get("sample_a")
    sample_b = details.get("sample_b")
    if sample_a is not None or sample_b is not None:
        lines.extend(["", "【历史样本】"])
        lines.append(f"{team_a} {sample_a if sample_a is not None else '暂无'} 局 | "
                     f"{team_b} {sample_b if sample_b is not None else '暂无'} 局")
    bet_status = details.get("bet_status") or "跳过"
    lines.extend(["", f"【下注状态】{bet_status}（虚拟下注 / 真实建议 / 跳过）"])
    analyst_count = int(details.get("analyst_count") or 0)
    if analyst_count:
        lines.append(f"分析师参考：已纳入 {analyst_count} 篇相关公开资料。")
    reasons = details.get("reasons") or row.get("reasons") or []
    lines.extend(["", "预测依据："])
    if reasons:
        lines.extend(f"• {_zh_reason(reason)}" for reason in reasons[:4])
    else:
        lines.append("• 暂无补充说明。")
    lines.extend(["", "研究监控信号，不构成下注建议。"])
    return "\n".join(lines).strip()


def _format_prematch_reference_alert(row: dict[str, Any]) -> str:
    details = row.get("details") or {}
    team_a = _display_team(details.get("team_a"))
    team_b = _display_team(details.get("team_b"))
    blue = details.get("blue_win_probability")
    red = details.get("red_win_probability")
    lines = [
        "🔵【赛前参考】（赛程未录入，仅供参考）",
        f"{_sport_name(row.get('sport'))}｜{_event_name(row.get('title'))}",
    ]
    if blue is not None and red is not None:
        if team_a and team_b:
            lines.append(f"模型胜率：{team_a} {float(blue):.1%} | {team_b} {float(red):.1%}")
        else:
            lines.append(f"模型胜率：{float(blue):.1%} | {float(red):.1%}")
    ev = details.get("ev")
    if ev is not None:
        lines.append(f"EV：{float(ev):.1%}")
    lines.extend(["注意：本场未在赛程库中命中，数据可信度下降，禁止下注。"])
    return "\n".join(lines).strip()


def _lol_resource_summary(game: dict[str, Any]) -> str:
    parts = []
    for label, key_a, key_b in (
        ("塔", "towers_a", "towers_b"),
        ("小龙", "dragons_a", "dragons_b"),
        ("大龙", "barons_a", "barons_b"),
        ("先锋", "rift_heralds_a", "rift_heralds_b"),
        ("高地", "inhibitors_a", "inhibitors_b"),
    ):
        value_a, value_b = game.get(key_a), game.get(key_b)
        if value_a is not None and value_b is not None:
            parts.append(f"{label} {value_a}-{value_b}")
    return "｜".join(parts)


def _lol_player_performance(game: dict[str, Any], limit: int = 3) -> list[str]:
    ranked = []
    for player in game.get("players") or []:
        try:
            kills = int(player.get("kills") or 0)
            deaths = int(player.get("deaths") or 0)
            assists = int(player.get("assists") or 0)
        except (TypeError, ValueError):
            continue
        name = str(player.get("player") or "").strip()
        if not name:
            continue
        impact = (kills + assists) / max(1, deaths)
        ranked.append((impact, kills, assists, -deaths, name, player))
    lines = []
    for _, kills, assists, neg_deaths, _, player in sorted(ranked, reverse=True)[:limit]:
        deaths = -neg_deaths
        team = _display_team(player.get("team"))
        champion = str(player.get("champion") or "").strip()
        gold = player.get("gold")
        cs = player.get("cs")
        extras = []
        if gold is not None:
            extras.append(f"金币 {gold}")
        if cs is not None:
            extras.append(f"补刀 {cs}")
        suffix = f"｜{'｜'.join(extras)}" if extras else ""
        lines.append(
            f"{team} {player.get('player')}（{champion or '英雄未知'}）"
            f"KDA {kills}/{deaths}/{assists}{suffix}"
        )
    return lines


def _lol_model_error_readout(details: dict[str, Any], games: list[dict[str, Any]]) -> str | None:
    actual_side = details.get("actual_side")
    bp_side = details.get("bp_side")
    if actual_side is None or bp_side is None or actual_side == bp_side:
        return None
    predicted = "蓝方" if bp_side == "a" else "红方"
    winner = "蓝方" if actual_side == "a" else "红方"
    evidence = []
    for game in games:
        for label, key_a, key_b in (
            ("塔", "towers_a", "towers_b"),
            ("小龙", "dragons_a", "dragons_b"),
            ("大龙", "barons_a", "barons_b"),
        ):
            value_a, value_b = game.get(key_a), game.get(key_b)
            if value_a is None or value_b is None or value_a == value_b:
                continue
            leading = "蓝方" if float(value_a) > float(value_b) else "红方"
            if leading == winner:
                evidence.append(f"{label}{value_a}-{value_b}")
    evidence_text = "、".join(dict.fromkeys(evidence))
    if evidence_text:
        return f"模型偏向{predicted}，但{winner}在可观测资源上取得 {evidence_text}，资源转化推翻了 BP 初始优势。"
    return f"模型偏向{predicted}但实际由{winner}获胜；当前逐局资源字段不足，不能进一步归因。"


def _format_lol_postmatch_evidence(details: dict[str, Any]) -> list[str]:
    games = list(details.get("game_samples") or [])
    if not games:
        return ["", "深度复盘：", "• 未取得逐局 BP、资源和选手数据，无法做阵容/过程/选手层面的可靠归因。"]
    lines = ["", "阵容与 BP："]
    for game in games:
        index = int(game.get("game_index") or 0)
        blue_players = list(game.get("blue_players") or [])
        red_players = list(game.get("red_players") or [])
        blue_champions = list(game.get("blue_champions") or [])
        red_champions = list(game.get("red_champions") or [])
        blue_draft = "、".join(
            f"{player}/{champion}" for player, champion in zip(blue_players, blue_champions)
        ) or "、".join(str(value) for value in blue_champions) or "数据缺失"
        red_draft = "、".join(
            f"{player}/{champion}" for player, champion in zip(red_players, red_champions)
        ) or "、".join(str(value) for value in red_champions) or "数据缺失"
        lines.append(f"• 第{index}局 蓝方：{blue_draft}")
        lines.append(f"• 第{index}局 红方：{red_draft}")

    lines.extend(["", "比赛过程："])
    for game in games:
        index = int(game.get("game_index") or 0)
        resources = _lol_resource_summary(game)
        winner_side = game.get("winner_side")
        winner = details.get("team_a") if winner_side == "a" else details.get("team_b") if winner_side == "b" else None
        parts = [f"第{index}局"]
        if winner:
            parts.append(f"胜者 {_display_team(winner)}")
        if resources:
            parts.append(resources)
        lines.append("• " + "｜".join(parts))

    player_lines = []
    for game in games:
        for line in _lol_player_performance(game):
            if line not in player_lines:
                player_lines.append(line)
    lines.extend(["", "选手表现："])
    if player_lines:
        lines.extend(f"• {line}" for line in player_lines[:5])
    else:
        lines.append("• 选手逐局数据缺失，无法可靠评价个人状态。")

    error_readout = _lol_model_error_readout(details, games)
    if error_readout:
        lines.extend(["", "模型偏差解释：", f"• {error_readout}"])
    return lines


def _format_postmatch_alert(row: dict[str, Any]) -> str:
    icon = "🔴" if row.get("severity") == "EMERGENCY" else "🟠" if row.get("severity") == "IMPORTANT" else "🟡"
    severity = {"EMERGENCY": "紧急", "IMPORTANT": "重要", "OBSERVE": "关注", "NORMAL": "关注"}.get(
        str(row.get("severity")), str(row.get("severity") or "关注"))
    details = row.get("details") or {}
    lines = [
        f"{icon}【赛后复盘】",
        f"{_sport_name(row.get('sport'))}｜{_event_name(row.get('title'))}",
        f"重要度：{float(row.get('alert_score', 0)):.0f}/100｜级别：{severity}",
    ]
    pre_match_status = details.get("pre_match_status")
    if pre_match_status:
        lines.append(f"赛前分析状态：{pre_match_status}")
    actual_winner = details.get("actual_winner")
    if actual_winner:
        lines.append(f"实际胜者：{_display_team(actual_winner)}")
    score_a = details.get("score_a")
    score_b = details.get("score_b")
    if score_a is not None and score_b is not None:
        lines.append(f"最终比分：{float(score_a):.0f} - {float(score_b):.0f}")
    actual_side = details.get("actual_side")
    prematch_side = details.get("prematch_side")
    prematch_team = details.get("prematch_team")
    if prematch_side is not None:
        result = "正确" if prematch_side == actual_side else "错误"
        lines.append(f"赛前预测：{_display_team(prematch_team)}｜判断：{result}")
    else:
        lines.append("赛前预测：未生成有效概率")
    bp_probability = details.get("bp_probability")
    if bp_probability is not None:
        bp_side = details.get("bp_side")
        result = "正确" if bp_side == actual_side else "错误"
        lines.append(f"BP后预测：蓝方 {float(bp_probability):.1%}｜红方 {1-float(bp_probability):.1%}｜判断：{result}")
    analyst_count = int(details.get("analyst_count") or 0)
    if analyst_count:
        lines.append(f"分析师参考：已纳入 {analyst_count} 篇相关公开资料。")
    decisive_factors = details.get("decisive_factors") or []
    if decisive_factors:
        lines.extend(["", "胜负手："])
        lines.extend(f"• {factor}" for factor in decisive_factors)
    if str(row.get("sport") or "").casefold() == "lol":
        lines.extend(_format_lol_postmatch_evidence(details))
    useful_reasons = []
    for reason in row.get("reasons", []):
        text = _zh_live_text(reason)
        if text.startswith(("赛前预测：", "BP后预测：", "复盘：")):
            continue
        if text not in useful_reasons:
            useful_reasons.append(text)
    if useful_reasons:
        lines.extend(["", "补充说明："])
        lines.extend(f"• {reason}" for reason in useful_reasons[:3])
    analyst_notes = details.get("analyst_notes") or []
    if analyst_notes:
        lines.extend(["", "公开资料："])
        lines.extend(f"• {note.get('source') or '公开来源'}：{note.get('title')}" for note in analyst_notes[:3])
    lines.extend(["", "研究监控信号，不构成下注建议。"])
    return "\n".join(lines).strip()


def format_live_alert(alert: LiveAlert | dict[str, Any]) -> str:
    row = alert.as_dict() if isinstance(alert, LiveAlert) else alert
    if str(row.get("category")) == "DRAFT_ANALYSIS" and row.get("details"):
        return _format_draft_alert(row)
    if str(row.get("category")) == "PREMATCH_ANALYSIS" and row.get("details"):
        return _format_prematch_alert(row)
    if str(row.get("category")) == "PREMATCH_REFERENCE" and row.get("details"):
        return _format_prematch_reference_alert(row)
    if str(row.get("category")) == "POSTMATCH_REVIEW" and row.get("details"):
        return _format_postmatch_alert(row)
    icon = "🔴" if row.get("severity") == "EMERGENCY" else "🟠" if row.get("severity") == "IMPORTANT" else "🟡"
    severity = {"EMERGENCY": "紧急", "IMPORTANT": "重要", "OBSERVE": "关注", "NORMAL": "关注"}.get(
        str(row.get("severity")), str(row.get("severity") or "关注")
    )
    category = {
        "MARKET_ANOMALY": "盘口异常", "MAJOR_EVENT": "重大事件", "PROBABILITY_CHANGE": "概率变化",
        "NEWS_ALERT": "阵容 / 新闻异常",
        "PREMATCH_ANALYSIS": "赛前分析", "PREMATCH_REFERENCE": "赛前参考",
        "LINEUP_MISSING": "阵容缺失提示", "DRAFT_ANALYSIS": "BP 完成分析",
        "MATCH_START": "比赛开始", "PERIOD_UPDATE": "节次更新", "CLUTCH_TIME": "关键时段",
        "MATCH_FINISHED": "比赛结束", "WATCHER_MISSING": "监控缺失",
        "MONITORING_RECOVERY": "监控恢复", "POSTMATCH_REVIEW": "赛后复盘",
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
