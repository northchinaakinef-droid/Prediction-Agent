from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from .backtest import load_csv, run_backtest
from .cs2_model import evaluate_cs2, load_valve_vrs, save_cs2
from .delivery import FeishuAppClient, FeishuWebhookClient, format_daily_report
from .next_model import DEFAULT_FEATURES, load_jsonl, walk_forward_evaluate
from .lol_daily import run_daily
from .lol_model import evaluate_periods, evaluate_years, fit_elo, load_oracle_elixir, save_model
from .lol_model import load_canonical_games
from .sports_daily import run_all
from .providers.polymarket import PolymarketClient
from .paper_store import record_report, settle_pending, summary as paper_summary
from .risk import recommend


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Sports market research and risk-control CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    poly = sub.add_parser("polymarket", help="search active Polymarket markets")
    poly.add_argument("query")
    poly.add_argument("--limit", type=int, default=100)
    rec = sub.add_parser("recommend", help="size a model edge")
    rec.add_argument("event_id")
    rec.add_argument("outcome")
    rec.add_argument("model_probability", type=float)
    rec.add_argument("decimal_odds", type=float)
    rec.add_argument("--confidence", type=float, default=0.7)
    rec.add_argument("--bankroll", type=float, default=float(os.getenv("BANKROLL", "1000")))
    rec.add_argument("--spread", type=float)
    rec.add_argument("--available-size", type=float)
    rec.add_argument("--estimated-cost", type=float, default=0.0)
    rec.add_argument("--enable-trading", action="store_true")
    bt = sub.add_parser("backtest", help="backtest timestamped prediction rows")
    bt.add_argument("csv")
    bt.add_argument("--bankroll", type=float, default=1000)
    nxt = sub.add_parser("next-evaluate", help="walk-forward a market-anchored league model")
    nxt.add_argument("jsonl", help="timestamped next-model JSONL dataset")
    nxt.add_argument("league", choices=sorted(DEFAULT_FEATURES))
    nxt.add_argument("--features", help="comma-separated feature names; defaults are league-specific")
    nxt.add_argument("--initial-train", type=int, default=120)
    nxt.add_argument("--test-size", type=int, default=50)
    nxt.add_argument("--min-edge", type=float, default=0.03)
    nxt.add_argument("--output", help="optional JSON report path")
    notify = sub.add_parser("notify", help="send a prepared daily JSON report to Feishu")
    notify.add_argument("report", help="JSON file containing recommendations")
    notify.add_argument("--dry-run", action="store_true", help="print the exact message without sending")
    lol_train = sub.add_parser("lol-train", help="train an independent LoL model from multi-year Oracle's Elixir CSVs")
    lol_train.add_argument("csv", nargs="+", help="chronological Oracle's Elixir CSV files")
    lol_train.add_argument("--output", default="artifacts/lol_model.json")
    lol_train.add_argument("--train-end", type=int, default=2023)
    lol_train.add_argument("--validation-year", type=int, default=2024)
    lol_train.add_argument("--test-year", type=int, default=2025)
    cs2_train = sub.add_parser("cs2-train", help="train the roster-aware CS2 model from Valve VRS snapshots")
    cs2_train.add_argument("source", help="cloned ValveSoftware/counter-strike_regional_standings repository")
    cs2_train.add_argument("--output", default="artifacts/cs2_model.json")
    daily = sub.add_parser("lol-daily", help="build today's independent LoL market report")
    daily.add_argument("--model", default="artifacts/lol_model.json")
    daily.add_argument("--output", default="reports/lol_daily.json")
    sport_train = sub.add_parser("sport-train", help="train one independent NBA/CBA/LoL model")
    sport_train.add_argument("sport", choices=("nba", "cba", "lol"))
    sport_train.add_argument("csv", nargs="+")
    sport_train.add_argument("--format", choices=("canonical", "oracle-elixir"), default="canonical")
    sport_train.add_argument("--output-dir", default="artifacts")
    sport_train.add_argument("--train-end", type=int, default=2023)
    sport_train.add_argument("--validation-start", type=int, default=2024)
    sport_train.add_argument("--validation-end", type=int, default=2024)
    sport_train.add_argument("--test-start", type=int, default=2025)
    sport_train.add_argument("--test-end", type=int, default=2025)
    all_daily = sub.add_parser("daily", help="analyze upcoming NBA, LoL, and CS2 markets (CBA paused)")
    all_daily.add_argument("--model-dir", default="artifacts")
    all_daily.add_argument("--output", default="reports/daily.json")
    all_daily.add_argument("--paper-db", help="append this genuinely forward run to a SQLite ledger")
    audit = sub.add_parser("schedule-audit", help="discover schedules, reconcile markets, and report coverage")
    audit.add_argument("--date", help="report date in Asia/Singapore (YYYY-MM-DD)")
    audit.add_argument("--model-dir", default="artifacts")
    audit.add_argument("--output", default="reports/schedule_audit.json")
    paper = sub.add_parser("paper-summary", help="summarize the append-only forward prediction ledger")
    paper.add_argument("--paper-db", default="data/daily/paper.db")
    settle = sub.add_parser("paper-settle", help="settle prior ledger events from Polymarket")
    settle.add_argument("--paper-db", default="data/daily/paper.db")
    args = parser.parse_args()
    if args.command == "polymarket":
        rows = PolymarketClient().search_sports(args.query, limit=args.limit)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.command == "recommend":
        result = recommend(event_id=args.event_id, outcome=args.outcome,
                           model_probability=args.model_probability, decimal_odds=args.decimal_odds,
                           bankroll=args.bankroll, confidence=args.confidence,
                           spread=args.spread, available_size=args.available_size,
                           estimated_cost=args.estimated_cost,
                           trading_enabled=args.enable_trading)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    elif args.command == "backtest":
        report = run_backtest(load_csv(args.csv), initial_bankroll=args.bankroll)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    elif args.command == "next-evaluate":
        rows = [row for row in load_jsonl(args.jsonl) if row.league == args.league]
        if not rows:
            parser.error(f"no {args.league} rows in dataset")
        names = tuple(name.strip() for name in args.features.split(",") if name.strip()) if args.features else None
        report = walk_forward_evaluate(
            rows, names, initial_train=args.initial_train, test_size=args.test_size, min_edge=args.min_edge,
        )
        payload = json.dumps(report.as_dict(), ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(payload, encoding="utf-8")
        print(payload)
    elif args.command == "lol-train":
        games = load_oracle_elixir(args.csv)
        evaluation = evaluate_years(games, train_end=args.train_end,
                                    validation_year=args.validation_year, test_year=args.test_year)
        # Production forecasts use every game available before the daily run;
        # the untouched historical final-test result remains recorded separately.
        model = fit_elo(games)
        save_model(model, args.output, evaluation)
        print(json.dumps({"saved": args.output, "model": model.as_dict(),
                          "evaluation": evaluation}, ensure_ascii=False, indent=2))
    elif args.command == "lol-daily":
        report = run_daily(args.model, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "cs2-train":
        games = load_valve_vrs(args.source)
        model, evaluation = evaluate_cs2(games)
        save_cs2(model, evaluation, args.output)
        print(json.dumps({"saved": args.output, "evaluation": evaluation}, ensure_ascii=False, indent=2))
    elif args.command == "sport-train":
        if args.format == "oracle-elixir":
            if args.sport != "lol":
                parser.error("oracle-elixir format is only valid for LoL")
            games = load_oracle_elixir(args.csv)
        else:
            games = load_canonical_games(args.csv, args.sport)
        evaluation = evaluate_periods(
            games, train_end=args.train_end, validation_start=args.validation_start,
            validation_end=args.validation_end, test_start=args.test_start, test_end=args.test_end,
        )
        model = fit_elo(games)
        target = Path(args.output_dir) / f"{args.sport}_model.json"
        save_model(model, target, evaluation)
        print(json.dumps({"saved": str(target), "evaluation": evaluation}, ensure_ascii=False, indent=2))
    elif args.command == "daily":
        report = run_all(args.model_dir, args.output)
        if args.paper_db:
            report["paper_store"] = record_report(args.paper_db, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "schedule-audit":
        from datetime import date as calendar_date
        report_day = calendar_date.fromisoformat(args.date) if args.date else None
        report = run_all(args.model_dir, args.output, report_day=report_day)
        payload = json.dumps(report["schedule_coverage"], ensure_ascii=False, indent=2)
        Path(args.output).write_text(payload, encoding="utf-8")
        print(payload)
    elif args.command == "paper-summary":
        print(json.dumps(paper_summary(args.paper_db), ensure_ascii=False, indent=2))
    elif args.command == "paper-settle":
        print(json.dumps(settle_pending(args.paper_db), ensure_ascii=False, indent=2))
    else:
        report_data = json.loads(Path(args.report).read_text(encoding="utf-8"))
        message = format_daily_report(report_data)
        if args.dry_run:
            print(message)
            return
        webhook = os.getenv("FEISHU_WEBHOOK_URL")
        if webhook:
            client = FeishuWebhookClient(webhook, os.getenv("FEISHU_WEBHOOK_SECRET") or None)
        else:
            required = ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_RECEIVE_ID"]
            missing = [name for name in required if not os.getenv(name)]
            if missing:
                parser.error("missing Feishu configuration: " + ", ".join(missing))
            client = FeishuAppClient(
                os.environ["FEISHU_APP_ID"], os.environ["FEISHU_APP_SECRET"],
                os.environ["FEISHU_RECEIVE_ID"], os.getenv("FEISHU_RECEIVE_ID_TYPE", "open_id"),
            )
        result = client.send_text(message)
        print(json.dumps({"sent": True, "parts": len(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
