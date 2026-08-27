# V2.4 Signal Quality & Live Monitoring Reliability Report

Date: 2026-08-27  
Branch: `codex/v2.4-signal-quality`  
Safety mode: Paper-only / Shadow research

## Executive result

V2.4 separates candidates from canonical virtual bets, removes real-money-advice wording,
stabilizes live match identity and missing/recovery transitions, and gates alert severity by
market tradability. Quant probability, Evidence gates, Risk/Kelly/stake controls, canonical
samples, provider promotion, roster freshness, and the LLM adjustment cap were not changed.

## Required answers

1. **“今日模拟下注”统计错误是否修复？** Yes. `action=BET` is now a candidate. Only
   `final_status=VIRTUAL_BET`, a recognized paper-bet status, or a canonical virtual-bet ID is
   counted as an actual virtual bet. Candidate details show direction, probability, EV, market
   and data quality, blocker, and final status.
2. **“真实建议”误导文案是否移除？** Yes. Delivery and progress copy uses 研究建议、虚拟研究
   and Paper-only terminology. Regression coverage rejects 真实建议、真实下注、真实执行 while
   `REAL_TRADING_DISABLED=true`.
3. **同一比赛不同 alias 是否统一 identity？** Yes. `canonical_live_match_id` prefers stable
   reconciled/provider IDs and otherwise uses sport-isolated, order-independent canonical teams
   with optional tournament/start context. Hyperion/GAL/TLNP/BFX/Nongshim variants are covered.
   Academy/youth/challenger names remain distinct.
4. **Missing/Recovery flapping 是否得到控制？** Yes. Persistent SQLite state implements
   HEALTHY → SUSPECT → MISSING → RECOVERING → HEALTHY. Defaults require two missing scans and two
   recovery scans; configurable grace/start windows are supported. Each real transition emits once.
5. **Market quality 如何影响 alert severity？** Alert-only quality is classified as GOOD,
   LOW_LIQUIDITY, WIDE_SPREAD, STALE, or UNTRADEABLE. Poor quality caps alerts at research/OBSERVE
   severity and places the market-quality reason before the probability difference. Model and
   canonical market snapshot values remain unchanged.
6. **超大 spread / 极低流动性市场是否降噪？** Yes. A 64% spread is UNTRADEABLE and classified
   as MARKET_MICROSTRUCTURE_ANOMALY rather than urgent value. Low liquidity is classified as
   LOW_LIQUIDITY_NOISE and downgraded.
7. **实际 sent alerts 比 raw generated alerts 减少多少？** The persisted audit now records raw
   generated, sent, deduped, flapping-suppressed, low-quality-suppressed, confirmed missing and
   recovery counts. A production reduction percentage must be reported after the first deployed
   full scan; no fabricated pre-deployment number is used here.
8. **是否影响 Quant/Evidence/Risk/LLM Shadow？** No. No probability model, Evidence/Data Quality
   Gate, Risk/Kelly/stake rule, LLM ±5% cap, sample logic, provider promotion, or roster freshness
   code was changed. PandaScore registered rosters remain outside formal Evidence when freshness
   cannot be proved.
9. **所有回归测试结果？** PASS: unittest 223/223; pytest 238/238. V2.4-specific tests 11/11.
10. **新 frozen manifest 是什么？**
    `5bbd9db4a469a5cf9b9d2968ed12625075147385f561aa04b8f3815bbe120706`.

## Deployment status

Deployment is intentionally gated by the repository rule: feature branch → pull request → merge
to `main` → protected GitHub Actions deployment. The activation script preserves `.env`, backs up
production code and SQLite databases, rebuilds/recreates the container, and verifies health,
HTTP, scheduler, SQLite and provider freshness. Production scan observations will be appended after
the PR is merged and the workflow completes.
