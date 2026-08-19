from datetime import datetime, timedelta, timezone
import unittest

from prediction_agent.anomaly import detect_market_anomalies
from prediction_agent.backtest import BacktestRow, run_backtest
from prediction_agent.delivery import (
    FeishuWebhookClient, _display_team, _event_name, _format_draft_alert, _format_postmatch_alert,
    _format_prematch_alert, format_daily_post, format_daily_report, format_live_alert, webhook_signature,
)
from prediction_agent.models import MarketSnapshot
from prediction_agent.risk import normalize_two_way, recommend


class CoreTests(unittest.TestCase):
    def test_remove_vig(self):
        a, b = normalize_two_way(0.55, 0.55)
        self.assertAlmostEqual(a, 0.5)
        self.assertAlmostEqual(b, 0.5)

    def test_no_bet_when_edge_is_weak(self):
        result = recommend(event_id="x", outcome="home", model_probability=0.51,
                           decimal_odds=2.0, bankroll=1000, confidence=0.7)
        self.assertEqual(result.action, "NO_BET")
        self.assertEqual(result.stake, 0)

    def test_stake_is_capped(self):
        result = recommend(event_id="x", outcome="home", model_probability=0.8,
                           decimal_odds=2.0, bankroll=1000, confidence=0.9,
                           spread=0.01, available_size=1000, trading_enabled=True)
        self.assertEqual(result.action, "BET")
        self.assertLessEqual(result.stake, 7.5)
        self.assertEqual(result.model_probability, 0.8)
        self.assertAlmostEqual(result.decision_probability, 0.77)
        self.assertAlmostEqual(result.raw_edge, 0.3)

    def test_explicit_no_trade_when_market_quality_is_missing(self):
        result = recommend(event_id="x", outcome="yes", model_probability=0.8,
                           decimal_odds=2.0, bankroll=1000, confidence=0.9,
                           require_market_quality=True)
        self.assertEqual(result.action, "NO_BET")
        self.assertEqual(result.decision, "NO TRADE")
        self.assertIn("spread/liquidity unavailable", result.reasons)

    def test_explicit_buy_direction_when_trade_passes(self):
        result = recommend(event_id="x", outcome="no", model_probability=0.8,
                           decimal_odds=2.0, bankroll=1000, confidence=0.9,
                           spread=0.01, available_size=1000, require_market_quality=True, trading_enabled=True)
        self.assertEqual(result.decision, "BUY NO")

    def test_anomaly_detection(self):
        now = datetime.now(timezone.utc)
        history = [MarketSnapshot("x", "yes", now + timedelta(minutes=i), .49, .51, 100, 100) for i in range(5)]
        history.append(MarketSnapshot("x", "yes", now + timedelta(minutes=6), .79, .90, 1000, 1))
        kinds = {a.kind for a in detect_market_anomalies(history)}
        self.assertTrue({"price_jump", "liquidity_drop", "book_imbalance"}.issubset(kinds))

    def test_backtest_rejects_lookahead(self):
        now = datetime.now(timezone.utc)
        row = BacktestRow("x", now, now - timedelta(hours=1), .7, 2.0, True, .9)
        with self.assertRaisesRegex(ValueError, "look-ahead"):
            run_backtest([row])

    def test_backtest_compares_model_to_market_and_reports_strata(self):
        now = datetime.now(timezone.utc)
        rows = [
            BacktestRow("a", now, now + timedelta(hours=1), .90, 2.0, True, .9),
            BacktestRow("b", now + timedelta(days=1), now + timedelta(days=1, hours=1), .55, 2.0, False, .9),
        ]
        report = run_backtest(rows)
        self.assertEqual(report.samples, 2)
        self.assertEqual(report.bets, 2)
        self.assertLess(report.brier_score, report.market_brier_score)
        self.assertEqual(len(report.edge_strata), 5)
        self.assertEqual(len(report.entry_price_strata), 7)

    def test_feishu_signature_is_stable(self):
        self.assertEqual(webhook_signature(1700000000, "secret"), webhook_signature(1700000000, "secret"))

    def test_feishu_webhook_payload(self):
        calls = []
        def transport(url, payload, **kwargs):
            calls.append((url, payload))
            return {"code": 0}
        FeishuWebhookClient("https://example.invalid/hook", transport=transport).send_text("hello")
        self.assertEqual(calls[0][1]["content"]["text"], "hello")

    def test_feishu_webhook_rich_post_payload(self):
        calls = []
        def transport(url, payload, **kwargs):
            calls.append((url, payload))
            return {"code": 0}
        post = format_daily_post({"report_date": "2026-08-14", "recommendations": []})
        FeishuWebhookClient("https://example.invalid/hook", transport=transport).send_post(post)
        self.assertEqual(calls[0][1]["msg_type"], "post")
        self.assertEqual(calls[0][1]["content"]["post"]["zh_cn"]["title"], "【今日模拟下注】0场")

    def test_feishu_rejects_corrupted_rich_post(self):
        client = FeishuWebhookClient("https://example.invalid/hook", transport=lambda *_args, **_kwargs: {"code": 0})
        with self.assertRaisesRegex(ValueError, "corrupted text"):
            client.send_post({"zh_cn": {"title": "????", "content": []}})

    def test_daily_report_supports_no_bet(self):
        message = format_daily_report({"recommendations": []})
        self.assertIn("【今日模拟下注】0场", message)
        self.assertIn("【今日跳过】0场", message)
        self.assertIn("【虚拟账户历史回撤】0.0%", message)
        self.assertIn("【风控状态】正常", message)
        self.assertNotIn("NO BET", message)

    def test_daily_report_highlights_bets_and_lists_skipped_details(self):
        message = format_daily_report({
            "bankroll_usdc": 1100,
            "recommendations": [
                {"event": "ordinary", "action": "NO_BET", "reasons": ["paper net EV below 5%"]},
                {"event": "opportunity", "action": "BET", "stake": 5.0,
                 "model_probability": .6, "market_probability": .55,
                 "expected_value": .08, "lineup_status": "完整"},
            ],
        })
        self.assertIn("opportunity", message)
        self.assertIn("ordinary", message)
        self.assertIn("【今日跳过明细】1场", message)
        self.assertIn("EV: 不可比较", message)
        self.assertIn("卡住条件", message)
        self.assertIn("【今日模拟下注】1场", message)
        self.assertIn("EV: 8.0%", message)
        self.assertIn("阵容状态: 完整", message)

    def test_daily_report_circuit_breaker_uses_breaker_label(self):
        message = format_daily_report({
            "bankroll_usdc": 1000,
            "risk_status": {"circuit_breaker": True, "current_drawdown": 0.111},
            "recommendations": [
                {"event": "frozen", "action": "NO_BET", "outcome": "A",
                 "expected_value": 0.02, "model_probability": 0.55,
                 "market_probability": 0.52, "reasons": ["paper net EV below 5%"]},
            ],
        })
        self.assertIn("【今日跳过】1场（账户熔断（全部暂停））", message)
        self.assertIn("账户熔断（全部暂停）", message)
        self.assertIn("【虚拟账户历史回撤】11.1%", message)
        self.assertIn("【风控状态】熔断（已暂停）", message)
        self.assertNotIn("EV:", message)
        self.assertNotIn("paper net EV below 5%", message)

    def test_daily_report_risk_status_warn(self):
        message = format_daily_report({
            "risk_status": {"drawdown_level": "warn", "current_drawdown": 0.12},
            "recommendations": [],
        })
        self.assertIn("【风控状态】警戒（额度减半）", message)
        self.assertIn("【虚拟账户历史回撤】12.0%", message)
        self.assertNotIn("熔断（已暂停）", message)


    def test_event_name_strips_team_region_suffix(self):
        self.assertEqual(_event_name("LNG Esports vs Ninjas in Pyjamas.CN"),
                         "LNG Esports 对 Ninjas in Pyjamas")
        self.assertEqual(_event_name("Ninjas in Pyjamas 对 LNG Esports"),
                         "Ninjas in Pyjamas 对 LNG Esports")
        self.assertEqual(_display_team("Ninjas in Pyjamas.CN"), "Ninjas in Pyjamas")
        self.assertEqual(_display_team("Team WE (BO3) - LPL Group Ascend"), "Team WE")
        self.assertEqual(_display_team("⁣Hooligans"), "Hooligans")

    def test_live_alert_text_is_clear(self):
        text = format_live_alert({
            "severity": "IMPORTANT", "alert_score": 60, "category": "DRAFT_ANALYSIS",
            "sport": "lol", "title": "LNG Esports vs Ninjas in Pyjamas.CN",
            "summary": "当前概率暂不可用，比赛仍保持监控。",
            "reasons": ["蓝方：K'Sante、Bel'Veth、Syndra、Lucian、Milio",
                        "红方：Jayce、Trundle、Diana、Xayah、Rakan"],
        })
        self.assertIn("BP 完成分析", text)
        self.assertIn("LNG Esports 对 Ninjas in Pyjamas", text)
        self.assertNotIn(".CN", text)


    def test_prematch_alert_formats_probability_and_reasons(self):
        text = _format_prematch_alert({
            "severity": "IMPORTANT", "alert_score": 55, "sport": "lol",
            "title": "BLG vs WE",
            "details": {
                "outcome": "BLG", "blue_win_probability": .63, "red_win_probability": .37,
                "blue_market_probability": .58, "red_market_probability": .42,
                "reasons": ["confidence below threshold"],
            },
        })
        self.assertIn("赛前分析", text)
        self.assertIn("赛前预测方向", text)
        self.assertIn("模型胜率", text)
        self.assertIn("置信度未达到要求", text)

    def test_postmatch_alert_compares_predictions(self):
        text = _format_postmatch_alert({
            "severity": "IMPORTANT", "alert_score": 70, "sport": "lol",
            "title": "T1 vs Gen.G",
            "details": {
                "actual_winner": "Gen.G", "actual_side": "b",
                "prematch_side": "a", "prematch_team": "T1",
                "bp_side": "b", "bp_probability": .52,
            },
            "reasons": ["赛前预测：T1；判断错误。", "BP后预测：蓝方 52.0%｜红方 48.0%；判断正确。"],
        })
        self.assertIn("赛后复盘", text)
        self.assertIn("实际胜者", text)
        self.assertIn("赛前预测", text)
        self.assertIn("判断：错误", text)
        self.assertIn("判断：正确", text)

    def test_lol_postmatch_alert_uses_game_evidence_and_drops_generic_repetition(self):
        text = _format_postmatch_alert({
            "severity": "IMPORTANT", "alert_score": 70, "sport": "lol",
            "title": "Anyone's Legend vs Top Esports",
            "details": {
                "actual_winner": "Top Esports", "actual_side": "b",
                "team_a": "Anyone's Legend", "team_b": "Top Esports",
                "prematch_side": None, "bp_side": "a", "bp_probability": .526,
                "decisive_factors": ["防御塔差 -5（红方占优）"],
                "game_samples": [{
                    "game_index": 1, "winner_side": "b",
                    "blue_players": ["Flandre", "Tarzan"],
                    "red_players": ["369", "Kanavi"],
                    "blue_champions": ["Renekton", "Sejuani"],
                    "red_champions": ["Gnar", "Wukong"],
                    "towers_a": 3, "towers_b": 8,
                    "dragons_a": 1, "dragons_b": 3,
                    "barons_a": 0, "barons_b": 2,
                    "players": [{
                        "team": "Top Esports", "player": "Kanavi", "champion": "Wukong",
                        "kills": 5, "deaths": 1, "assists": 9, "gold": 12800, "cs": 210,
                    }],
                }],
                "analyst_notes": [{"source": "公开赛评", "title": "TES 中野资源控制复盘"}],
            },
            "reasons": [
                "赛前预测：未生成有效概率。",
                "BP后预测：蓝方 52.6%｜红方 47.4%；判断错误。",
                "复盘：比赛走势偏离预测，需重点核查阵容克制、资源节奏或临场发挥。",
                "复盘：比赛走势偏离预测，需重点核查阵容克制、资源节奏或临场发挥。",
            ],
        })
        self.assertIn("阵容与 BP：", text)
        self.assertIn("Flandre/Renekton", text)
        self.assertIn("比赛过程：", text)
        self.assertIn("塔 3-8", text)
        self.assertIn("选手表现：", text)
        self.assertIn("Kanavi（Wukong）KDA 5/1/9", text)
        self.assertIn("模型偏差解释：", text)
        self.assertIn("资源转化推翻了 BP 初始优势", text)
        self.assertIn("公开赛评：TES 中野资源控制复盘", text)
        self.assertNotIn("需重点核查阵容克制", text)
        self.assertEqual(text.count("赛前预测：未生成有效概率"), 1)
        self.assertEqual(text.count("BP后预测：蓝方 52.6%"), 1)

    def test_lol_postmatch_alert_reports_missing_evidence_honestly(self):
        text = _format_postmatch_alert({
            "severity": "IMPORTANT", "alert_score": 70, "sport": "lol",
            "title": "A vs B",
            "details": {"actual_winner": "B", "actual_side": "b", "game_samples": []},
            "reasons": [],
        })
        self.assertIn("未取得逐局 BP、资源和选手数据", text)
        self.assertNotIn("比赛走势偏离预测", text)

    def test_draft_alert_includes_strength_and_risk_sections(self):
        text = _format_draft_alert({
            "severity": "IMPORTANT", "alert_score": 60, "sport": "lol",
            "title": "LNG Esports vs Ninjas in Pyjamas",
            "details": {
                "blue_champions": ["Aatrox", "Bel'Veth", "Syndra", "Lucian", "Milio"],
                "red_champions": ["Jayce", "Trundle", "Diana", "Xayah", "Rakan"],
                "post_draft_probability": 0.58,
                "readout": {
                    "blue_team": "LNG Esports", "red_team": "Ninjas in Pyjamas",
                    "team_edge": 45.0,
                    "lanes": [
                        {"role": "top", "blue_champion": "Aatrox", "red_champion": "Jayce", "edge": 20},
                        {"role": "jng", "blue_champion": "Bel'Veth", "red_champion": "Trundle", "edge": 35},
                        {"role": "mid", "blue_champion": "Syndra", "red_champion": "Diana", "edge": 55},
                        {"role": "bot", "blue_champion": "Lucian", "red_champion": "Xayah", "edge": -30},
                        {"role": "sup", "blue_champion": "Milio", "red_champion": "Rakan", "edge": -5},
                    ],
                },
            },
        })
        self.assertIn("BP 后模型胜率", text)
        self.assertIn("双方阵容", text)
        self.assertIn("阵容优劣", text)
        self.assertIn("配合与滚雪球", text)
        self.assertIn("风险点", text)
        self.assertIn("研究监控信号", text)
        self.assertIn("亚托克斯", text)
        self.assertIn("卑尔维斯", text)
        self.assertIn("辛德拉", text)
        self.assertIn("卢锡安", text)
        self.assertIn("米利欧", text)
        self.assertIn("杰斯", text)
        self.assertIn("特朗德尔", text)
        self.assertIn("黛安娜", text)
        self.assertIn("霞", text)
        self.assertIn("洛", text)
        self.assertNotIn("Aatrox", text)
        self.assertNotIn("K'Sante", text)
        self.assertNotIn("Bel'Veth", text)
        self.assertNotIn("Syndra", text)


if __name__ == "__main__":
    unittest.main()
