# Prediction Agent

面向 NBA、CBA、LoL、CS2 的赛事研究与风险控制骨架。它汇总概率、盘口、流动性和证据，输出 `BET` 或 `NO_BET`，但不承诺收益，也不会自动下单。

## 已实现

- Polymarket Gamma/CLOB/Data 公共读接口：市场、midpoint、order book、holders。
- The Odds API 适配器：当前与历史赔率快照（需要 API key）。
- 异常检测：价格跳变、点差突然扩大、订单簿失衡。
- 下注规模：模型概率向市场概率收缩，1/4 Kelly；当前风控为单笔 0.75%、单日 2.5%、同一赛事 1%、累计回撤 10% 熔断。
- 时间顺序回测：拒绝开赛后决策，报告 ROI、最大回撤、Brier 分数及 bootstrap 95% ROI 区间。
- 飞书推送：支持群自定义机器人 Webhook，或自建应用机器人直接推送个人账号。

## 快速运行

无需安装依赖：

```powershell
$env:PYTHONPATH="src"
python -m prediction_agent.cli recommend nba-demo yes 0.62 1.95 --confidence 0.8 --bankroll 1000 --spread 0.01 --available-size 500
python -m prediction_agent.cli backtest examples/backtest.csv
python -m prediction_agent.cli next-evaluate data/next_model.jsonl nba --output reports/nba_next_model.json
python -m unittest discover -s tests
```

## Next market-anchored model

The next model uses the decision-time Polymarket probability as a log-odds offset and learns only incremental corrections from independent information such as injuries, lineups, rest, patches, map pools, and multi-book consensus. NBA, CBA, LoL, and CS2 are always trained separately.

Input is JSONL. Every event requires `decision_at`, `start_at`, `settled_at`, market probability, result, and features carrying both `observed_at` and `source`. Features observed after the decision are rejected. See `examples/next_model_rows.jsonl`.

`next-evaluate` runs an expanding walk-forward where each test fold can only use labels settled before that fold begins. Paper-trading approval requires at least 200 OOS predictions, 3 folds, better Brier and Log Loss than market, improvement in two-thirds of folds, 80% execution-price coverage, 100 costed trades, positive ROI, Profit Factor above 1.05, and maximum drawdown at or below 20%. It never disables the production `NO TRADE` default.


## 飞书手机推送

最简单的方式是在一个只有你自己的飞书群中添加“自定义机器人”，把 Webhook 和可选签名密钥放入环境变量：

```powershell
$env:FEISHU_WEBHOOK_URL="https://open.feishu.cn/open-apis/bot/v2/hook/你的值"
$env:FEISHU_WEBHOOK_SECRET="你的签名密钥"
$env:PYTHONPATH="src"
python -m prediction_agent.cli notify examples/daily_report.json --dry-run
python -m prediction_agent.cli notify examples/daily_report.json
```

若需要机器人直接私聊你的手机账户，创建飞书自建应用、开启机器人能力及 `im:message:send_as_bot` 权限，然后设置 `FEISHU_APP_ID`、`FEISHU_APP_SECRET`、`FEISHU_RECEIVE_ID`。应用机器人必须覆盖目标用户。手机是否弹出系统通知仍取决于飞书客户端和手机的通知设置。

定时运行建议由 Windows Task Scheduler 在每天固定时间启动完整分析流程，只有分析成功且报告 JSON 校验通过后才执行 `notify`。不要把 Webhook、App Secret 或钱包私钥写进任务参数、源码或 Git。

联网查询 Polymarket：

```powershell
$env:PYTHONPATH="src"
python -m prediction_agent.cli polymarket NBA --limit 500
```

## 生产数据架构

1. 每 5–30 秒保存原始盘口快照，禁止只保存收盘价。
2. NBA 官方数据从 NBA.com/Stats 获取；伤病与首发必须记录发布时间和来源。
3. NBA 博彩盘由至少两个独立聚合源交叉检查。CBA、LoL、CS2 应签约有明确覆盖和历史快照的数据商。
4. 新闻/报告进入事件证据表，只允许使用 `published_at <= decision_at` 的内容。
5. 预测按联赛分模：NBA（球员可用性、休息、赛程、阵容）、CBA（外援/注册/赛程）、LoL（版本、首发、阵容、地图侧）、CS2（地图池、阵容、LAN/线上、赛制）。
6. 推荐层以去水后的市场概率为基准，只在校准后概率的净优势覆盖手续费、滑点和模型误差时下注。

## 回测验收标准

不要用“历史 ROI ≥60%”作为唯一验收条件。如此高的 ROI 往往来自样本过小、赔率时点错误、挑选偏差或泄漏。建议同时要求：

- 严格 walk-forward，训练/验证/测试按时间隔离；最后赛季完全锁箱。
- 至少 500 笔独立下注，并按联赛、赛季、赔率区间报告。
- ROI 的 95% 置信区间下界大于 0，而不是只看点估计。
- 概率校准（Brier/reliability）、CLV、最大回撤、成交率和容量均达标。
- 计入 vig、Polymarket fee、滑点、盘口延迟、限额和无法成交。
- 参数冻结后纸上交易 8–12 周，再考虑极小资金实盘。

## 下一阶段

- 建立 SQLite/PostgreSQL 快照库和定时采集器。
- 增加 NBA 官方统计、伤病、新闻及第二赔率商的实际凭证。
- 实现赛事实体对齐（同队异名、时区、Bo3/Bo5、让分/地图盘）。
- 加入按日总风险 3%、相关性暴露、连续亏损熔断和账户级回撤熔断。
- 提供只读日报/API；交易执行必须独立审批，不与预测进程共享私钥。
