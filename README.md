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

## NBA、CBA、LoL 独立胜率模型

三项运动必须分别训练、验证、锁箱测试和登记生产模型，不能混合训练，也不能由其中一个模型替代另一个。当前基础模型使用多年历史赛果构建赛前 Elo；LoL 再将单局概率换算成 BO3/BO5 系列赛概率。Polymarket 只在最后用于比较可成交价格、成本后净优势和 EV，不参与独立胜率计算。

推荐的数据划分固定为：2020–2023 训练、2024 验证、2025 最终锁箱测试，2026 只在验收完成后用于生产更新。最终测试会记录在模型工件中，但不会因此自动开放真钱建议；还必须另行提供决策时点的历史可成交盘口，通过含手续费、点差和滑点的 ROI 验收。

```powershell
$env:PYTHONPATH="src"
python -m prediction_agent.cli sport-train nba data/raw/nba/*.csv
python -m prediction_agent.cli sport-train cba data/raw/cba/*.csv
python -m prediction_agent.cli sport-train lol data/raw/lol/*.csv --format oracle-elixir
python -m prediction_agent.cli daily --model-dir artifacts --output reports/daily.json
python -m prediction_agent.cli schedule-audit --date 2026-08-14 --output reports/schedule_audit_2026-08-14.json
python -m prediction_agent.cli notify reports/daily.json --dry-run
```

NBA/CBA 的规范输入列为 `event_id,played_at,team_a,team_b,team_a_won`。这些字段都是赛前已知实体与赛后标签；任何比赛中的技术统计都不能作为同一场比赛的赛前特征。LoL 可直接读取 Oracle's Elixir 团队赛果行。

日报明确区分：独立模型概率、风控后概率、市场价格、原始优势、成本后净优势、净 EV 和建议金额。无法核验开赛时间、模型未验收、阵容证据缺失或市场质量不合格时输出 `NO BET` 并写明原因。

## 云服务器常驻运行

服务器购买与安全配置请先阅读 [`SERVER_SETUP.md`](SERVER_SETUP.md)。当前推荐腾讯云轻量应用服务器中国香港 2核2GB、Ubuntu 24.04，先按月购买验证；不需要 GPU、域名或数据库产品。

生产环境使用一台 Linux VPS/云服务器运行 Docker 容器，不依赖本机、Codex 桌面程序或 GitHub Actions。容器默认每天北京时间 06:30 执行：加载已冻结的 NBA/LoL/CS2 模型 → 获取当天赛程与市场 → 生成报告 → 推送飞书。CBA 当前暂停；CS2 已进入研究概率推送，但 2026 市场锁箱回测未通过，因此强制 `NO_BET`。`/health` 返回最近一次运行时间和错误状态。

日报中的 LoL 赛事先由两个独立赛程站点动态发现，再与完整分页后的 Polymarket moneyline 市场对账。内部时间统一为 UTC，日报自然日默认使用 `Asia/Singapore`。赛程源失败显示 `DATA_UNAVAILABLE`，市场缺失显示 `MISSING_MARKET`，不会被当作 0 场；审计历史追加到 `data/daily/schedule_audits/`，watcher 状态保存在 `data/daily/watcher_registry.json`。可用 `NEXTMATCH_SCHEDULE_URL`、`ESPORTAGENDA_LOL_URLS`、`REPORT_TIMEZONE` 覆盖数据源和时区。

模型训练与每日预测分开：训练任务只有在获得新赛季数据后人工触发，必须先通过验证和锁箱测试，再把模型文件放入服务器 `artifacts/`。每日任务只能读取模型，不能擅自改动参数或重训。

服务器部署：

```bash
cp .env.example .env
# 在 .env 中填写飞书密钥、本金和运行时间
docker compose build
docker compose up -d
curl http://127.0.0.1:8080/health
```

`compose.yaml` 把模型以只读方式挂载，报告和数据使用持久卷。容器异常退出后自动重启，并限制日志大小。生产调度仍由腾讯云容器负责；GitHub Actions 只在 `main` 更新后运行测试并部署新版本，不承担每天的比赛扫描。

云端开发通过私有 GitHub 仓库和 Codex Cloud 完成。所有修改先进入分支和 PR；合并到 `main` 后，`Deploy production` 工作流才会通过受限 SSH 密钥更新腾讯云。部署保留服务器上的 `.env`、`data/` 和前向纸面账本。GitHub `production` 环境需要配置 `DEPLOY_HOST`、`DEPLOY_USER`、`DEPLOY_SSH_KEY` 和 `DEPLOY_HOST_KEY` 四个 secrets。

服务器默认每 30 分钟静默扫描未来 30 小时赛事，把完整快照写入 `data/daily/paper.db`；飞书仍在每天设定时间汇总推送。账本是追加式 SQLite：保存生成时间、距开赛小时数、T−24h/T−6h/T−1h 最近窗口、比赛、模型概率、市场价格、执行价、阵容资格、拒绝理由和动作；相同运行重试不会重复写入。查看累计前向样本：

```bash
python -m prediction_agent.cli paper-summary --paper-db data/daily/paper.db
```

## 旧版市场锚定模型（非 LoL 生产概率）

The next model uses the decision-time Polymarket probability as a log-odds offset and learns only incremental corrections from independent information such as injuries, lineups, rest, patches, map pools, and multi-book consensus. NBA, LoL, and CS2 are always trained separately; CBA is currently paused.

## CS2 roster-aware baseline

The first CS2 baseline reads Valve's public Regional Standings snapshots and uses only pre-match team and five-player roster identity. It trains on 2024, tunes on 2025, and keeps 2026 as a locked chronological test. Its separate Polymarket market test replays every model prediction before updating with the result and samples prices at T−24h, T−6h, and T−1h. The 2026 proxy ROI was negative at all three windows (−9.2%, −6.8%, and −8.5%), so real-money approval remains false. See `reports/CS2_历史市场锁箱回测.md`.

HLTV pages are not scraped because their terms prohibit data mining/web scraping. For the next map/veto/LAN layer, the preferred source is GRID Open Access (official CS2 telemetry, application required) or a licensed commercial feed. GRID access and license approval are a data prerequisite, not a server prerequisite; never put its token in Git.

## NBA market lockbox result

The chronological NBA Elo baseline was joined to 1,252 of 1,334 Polymarket moneyline games. Its 2026 proxy ROI was negative at T−24h, T−6h, and T−1h (−12.3%, −5.2%, and −8.5%), and the market Brier score beat the model in every window. Real-money approval therefore remains false. Since these 2026 outcomes have now been inspected, they are diagnostic data for future model changes rather than a reusable unseen lockbox. See `reports/NBA_历史市场锁箱回测.md`.

```bash
prediction-agent cs2-train data/external/valve_cs2_vrs
```

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
