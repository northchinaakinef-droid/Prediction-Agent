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

    def price(value: Any) -> str:
        return "不可用" if value is None else f"{float(value):.3f}"

    day = report_date or (date.fromisoformat(report["report_date"]) if report.get("report_date") else date.today())
    rows = sorted(report.get("recommendations", []), key=lambda row: row.get("action") != "BET")
    opportunities = sum(row.get("action") == "BET" for row in rows)
    lines = [f"每日赛事研究｜{day.isoformat()}", "", "【今日总览】"]
    sport_order = ("nba", "lol", "cs2")
    statuses = report.get("sport_status", {})
    for sport in sport_order:
        sport_rows = [row for row in rows if row.get("sport") == sport]
        bets = sum(row.get("action") == "BET" for row in sport_rows)
        status = statuses.get(sport, {})
        ready = status.get("ready", True)
        today_markets = status.get("today_markets")
        suffix = f"｜今日市场 {today_markets} 场" if today_markets is not None else ""
        lines.append(f"{'✅' if ready else '⚠️'} {_sport_name(sport)}：分析 {len(sport_rows)} 场｜符合策略 {bets} 场{suffix}")
    other_rows = [row for row in rows if row.get("sport") not in sport_order]
    if other_rows:
        bets = sum(row.get("action") == "BET" for row in other_rows)
        lines.append(f"其他：分析 {len(other_rows)} 场｜符合策略 {bets} 场")
    lines.append(f"合计：分析 {len(rows)} 场｜符合策略 {opportunities} 场")

    coverage = report.get("schedule_coverage", {})
    if coverage:
        expected = sum(value.get("expected", 0) for value in coverage.values())
        watching = sum(value.get("watching", 0) for value in coverage.values())
        lines.extend(["", "【赛程覆盖】"])
        lines.extend(
            f"{'✅' if float(value.get('coverage', 0)) >= 1 else '⚠️'} {_sport_name(sport)}　"
            f"预计 {value.get('expected', 0)}｜发现 {value.get('discovered', 0)}｜"
            f"市场 {value.get('market_matched', 0)}｜监控 {value.get('watching', 0)}"
            for sport, value in coverage.items()
        )
        lines.append(f"合计：预计 {expected} 场｜监控 {watching} 场")
        if report.get("data_incomplete"):
            lines.append("🚨 数据不完整：存在未解释的赛事遗漏")
        for sport, value in coverage.items():
            if value.get("source_disagreement_warning"):
                lines.append(f"⚠️ {_sport_name(sport)}：多个赛程源结果不一致，请查看赛程审计。")
            elif value.get("source_unavailable_warning"):
                lines.append(f"⚠️ {_sport_name(sport)}：部分赛程源不可用，请查看赛程审计。")

    schedule_matched = sum(bool(row.get("schedule_matched")) for row in rows)
    market_only = len(rows) - schedule_matched
    suspicious = sum(not bool(row.get("probability_plausible", True)) for row in rows)
    if market_only or suspicious:
        lines.extend(["", "【数据提示】"])
        if market_only:
            lines.append(f"• 市场独有 {market_only} 场：仅 Polymarket 出现，未在赛程源确认，可能存在映射或队名不一致。")
        if suspicious:
            lines.append(f"• 概率可疑 {suspicious} 场：模型输出处于异常区间，建议复核。")

    if not rows:
        lines.extend(["", "暂无达到策略与风控要求的机会。"])

    for sport in (*sport_order, None):
        sport_rows = [row for row in rows if row.get("sport") == sport] if sport is not None else other_rows
        if not sport_rows:
            continue
        section_name = _sport_name(sport) if sport else "其他"
        lines.extend(["", f"【{section_name}】"])
        for index, row in enumerate(sport_rows, 1):
            is_bet = row.get("action") == "BET"
            action = "符合策略" if is_bet else "暂不参与"
            marker = "⭐" if is_bet else "▫️"
            team_a, team_b = _event_team_pair(row)
            outcome = row.get("outcome")
            model = row.get("model_probability")
            market = row.get("market_probability")
            reasons = _key_reasons(row)
            lines.extend([
                "",
                f"{marker} {index}. {team_a} 对 {team_b}",
                f"结论：{action}｜预测方向：{_display_team(outcome)}",
                f"模型胜率：{_two_sided_probability(model, outcome, team_a, team_b)}",
                f"市场胜率：{_two_sided_probability(market, outcome, team_a, team_b)}",
                f"净优势：{percent(row.get('edge'))}｜净期望值：{percent(row.get('expected_value'))}｜可买价：{price(row.get('execution_price'))}",
            ])
            if is_bet and row.get("stake") is not None:
                lines.append(f"模拟下注：{float(row['stake']):.2f} USDC")
            lines.extend(f"• {reason}" for reason in reasons)

    bankroll = report.get("bankroll_usdc")
    bankroll_text = f"{float(bankroll):.2f} USDC" if bankroll is not None else "等待当日汇率"
    paper_summary = report.get("paper_summary") or {}
    if paper_summary.get("predictions") is not None:
        by_sport = paper_summary.get("by_sport") or {}
        lines.extend(["", "【模拟资金】", f"研究本金：{bankroll_text}"])
        lines.append(f"累计预测 {paper_summary.get('predictions', 0)} 场｜已结算 {paper_summary.get('settled', 0)} 场")
        for sport, stat in by_sport.items():
            roi = stat.get("paper_roi")
            roi_text = f"{roi:.1%}" if isinstance(roi, (int, float)) else "暂无"
            lines.append(
                f"{_sport_name(sport)}：模拟利润 {float(stat.get('paper_profit', 0)):.2f} USDC｜ROI {roi_text}"
            )
        lines.append("说明：模拟盘仅用于学习与模型迭代，不涉及真实资金。")
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
    lines = [
        f"{icon}【赛前分析】",
        f"{_sport_name(row.get('sport'))}｜{_event_name(row.get('title'))}",
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
    lines.extend(["", "复盘说明："])
    for reason in row.get("reasons", [])[:4]:
        lines.append(f"• {_zh_live_text(reason)}")
    lines.extend(["", "研究监控信号，不构成下注建议。"])
    return "\n".join(lines).strip()


def format_live_alert(alert: LiveAlert | dict[str, Any]) -> str:
    row = alert.as_dict() if isinstance(alert, LiveAlert) else alert
    if str(row.get("category")) == "DRAFT_ANALYSIS" and row.get("details"):
        return _format_draft_alert(row)
    if str(row.get("category")) == "PREMATCH_ANALYSIS" and row.get("details"):
        return _format_prematch_alert(row)
    if str(row.get("category")) == "POSTMATCH_REVIEW" and row.get("details"):
        return _format_postmatch_alert(row)
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
