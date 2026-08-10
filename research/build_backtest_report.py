from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path


SOURCE = Path("reports/polymarket_walkforward.json")
OUTPUT = Path("reports/历史回测报告_2025-26.md")


def roi(rows: list[dict], slippage: float = 0.01, fees: bool = True) -> float | None:
    pnl = turnover = 0.0
    for row in rows:
        stake = float(row.get("stake") or 0)
        if stake <= 0:
            continue
        side = int(row["side"])
        raw = float(row["market_p_a"]) if side == 0 else 1 - float(row["market_p_a"])
        price = min(0.99, raw + slippage)
        shares = stake / price
        fee = shares * 0.03 * price * (1 - price) if fees else 0.0
        pnl += shares * (1.0 if row["won"] else 0.0) - stake - fee
        turnover += stake
    return pnl / turnover if turnover else None


def bootstrap_ci(rows: list[dict], seed: int = 20260810) -> tuple[float, float]:
    bets = [r for r in rows if float(r.get("stake") or 0) > 0]
    rng = random.Random(seed)
    samples = []
    for _ in range(10_000):
        sample = [bets[rng.randrange(len(bets))] for _ in bets]
        samples.append(roi(sample) or 0.0)
    samples.sort()
    return samples[249], samples[9749]


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    reports = payload["reports"]
    all_rows = [row for report in reports for row in report["records"]]
    all_bets = [row for row in all_rows if float(row.get("stake") or 0) > 0]
    total_turnover = sum(float(row["stake"]) for row in all_bets)
    total_profit = sum(float(row["pnl"]) for row in all_bets)
    total_wins = sum(bool(row["won"]) for row in all_bets)
    combined_ci = bootstrap_ci(all_rows)
    lines = [
        "# NBA、CBA、LoL 历史 Walk-forward 回测报告",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 结论摘要",
        "",
        "本次预先固定、跨联赛统一的 Elo 策略未通过盈利验收。三个锁箱测试均为负 ROI，不能投入实盘，也没有任何证据支持长期 ROI ≥60%。",
        "",
        f"合计 {len(all_bets)} 笔测试下注，盈利 {total_profit:.2f} USDC / 流水 {total_turnover:.2f} USDC，ROI {total_profit / total_turnover:.1%}，命中率 {total_wins / len(all_bets):.1%}；按比赛 bootstrap 的 95% ROI 区间为 {combined_ci[0]:.1%} 至 {combined_ci[1]:.1%}。",
        "",
        "## 测试设计",
        "",
        "- 数据：Polymarket Gamma 已结算赛事与 CLOB 历史价格；赛果取已结算 outcome。",
        "- 范围：NBA 2025-10-21 至 2026-06-30；CBA 2025-12-01 至 2026-06-30；LoL 2026-01-01 至 2026-06-30。",
        "- 决策价格：开赛前 60 分钟最后一个历史价格，仅使用当时以前的信息。",
        "- 时间切分：每个联赛前 60% 用作 Elo 热身，后 40% 完全锁箱测试；测试期间只用过去赛果在线更新。",
        "- 固定模型：初始 Elo 1500、K=20；模型概率向市场收缩 50%；净差至少 5 个百分点才考虑下注。",
        "- 成本：买入价加入 1 美分不利滑点，并按体育 taker fee 公式计费；最低市场总成交量 100 USDC。",
        "- 仓位：1/4 Kelly，单笔最多当时本金 0.75%。三个联赛使用完全相同参数，未按测试结果调参。",
        "",
        "## 数据来源",
        "",
        "- [Polymarket Gamma Events](https://docs.polymarket.com/api-reference/events/list-events-keyset-pagination)：赛事、比赛时间、市场 token 与结算结果。",
        "- [Polymarket CLOB Price History](https://docs.polymarket.com/api-reference/markets/get-prices-history)：开赛前历史价格。",
        "- [Polymarket Sports Metadata](https://docs.polymarket.com/api-reference/sports/get-sports-metadata-information)：NBA tag 745、CBA tag 103097、LoL tag 65。",
        "- [Polymarket Fees](https://docs.polymarket.com/trading/fees)：体育 taker fee 公式与费率。",
        "",
        "## 锁箱结果",
        "",
        "|联赛|全部比赛|训练/热身|有价格测试场次|价格覆盖|下注|命中率|ROI|本金回报|最大回撤|市场 Brier|Elo Brier|ROI 95% CI|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        ci = bootstrap_ci(report["records"])
        lines.append(
            f"|{report['league'].upper()}|{report['all_games']}|{report['train_games']}|{report['test_games_with_price']}|"
            f"{report['price_coverage']:.1%}|{report['bets']}|{pct(report['hit_rate'])}|{pct(report['roi'])}|"
            f"{pct(report['bankroll_return'])}|{pct(report['max_drawdown'])}|{report['market_brier']:.3f}|"
            f"{report['elo_brier']:.3f}|{ci[0]:.1%}～{ci[1]:.1%}|"
        )
    lines.extend([
        "",
        "Brier 越低越好。三个联赛的市场概率均优于 Elo，说明简单历史胜负强度没有提供足以覆盖价格与交易成本的增量信息。",
        "",
        "## 成本敏感性",
        "",
        "|联赛|零滑点/零手续费|1¢滑点+手续费（主结果）|2¢滑点+手续费|测试前半 ROI|测试后半 ROI|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for report in reports:
        rows = report["records"]
        midpoint = len(rows) // 2
        lines.append(
            f"|{report['league'].upper()}|{pct(roi(rows, 0, False))}|{pct(roi(rows, .01, True))}|"
            f"{pct(roi(rows, .02, True))}|{pct(roi(rows[:midpoint], .01, True))}|{pct(roi(rows[midpoint:], .01, True))}|"
        )
    lines.extend([
        "",
        "## 反过拟合措施",
        "",
        "- 在查看锁箱结果前固定参数，并对三个联赛使用同一组参数。",
        "- 未使用比赛中或赛后统计作为赛前特征，也未使用结算价格作为买入价格。",
        "- 按时间而非随机拆分，避免未来赛果进入过去特征。",
        "- 同时报告负结果、交易成本、回撤、概率校准和置信区间，不用单一 ROI 挑选最好看的子集。",
        "- 没有在本报告中继续搜索最佳 K、边际阈值或收缩比例；这样做会污染锁箱集。",
        "",
        "## 局限",
        "",
        "- 历史价格序列是成交/聚合价格，不是完整历史 ask 与可成交深度；1 美分滑点仅为保守近似，真实大额成交可能更差。",
        "- CBA 的 Polymarket 覆盖从赛季中段开始，不代表完整 2025–26 赛季。",
        "- LoL 汇总多个赛区，队名变更、跨赛区强度与 Bo1/Bo3/Bo5 差异未进入简单 Elo。",
        "- 未能历史化重建每场伤病、首发、版本新闻与博彩公司盘口；这些不能事后补录后声称为无泄漏回测。",
        "- Bootstrap 假设比赛近似独立，无法完整反映同日、同队和市场状态相关性。",
        "",
        "## 决策",
        "",
        "当前模型状态：拒绝上线、全部 NO_BET。下一模型必须新增有时间戳的伤病/阵容/版本特征与多博彩公司共识价格，并使用新的、从未参与开发的后续日期作为第二锁箱集。",
    ])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
