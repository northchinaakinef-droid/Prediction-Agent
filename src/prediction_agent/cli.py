from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .backtest import load_csv, run_backtest
from .delivery import FeishuAppClient, FeishuWebhookClient, format_daily_report
from .providers.polymarket import PolymarketClient
from .risk import recommend


def main() -> None:
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
    bt = sub.add_parser("backtest", help="backtest timestamped prediction rows")
    bt.add_argument("csv")
    bt.add_argument("--bankroll", type=float, default=1000)
    notify = sub.add_parser("notify", help="send a prepared daily JSON report to Feishu")
    notify.add_argument("report", help="JSON file containing recommendations")
    notify.add_argument("--dry-run", action="store_true", help="print the exact message without sending")
    args = parser.parse_args()
    if args.command == "polymarket":
        rows = PolymarketClient().search_sports(args.query, limit=args.limit)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.command == "recommend":
        result = recommend(event_id=args.event_id, outcome=args.outcome,
                           model_probability=args.model_probability, decimal_odds=args.decimal_odds,
                           bankroll=args.bankroll, confidence=args.confidence)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    elif args.command == "backtest":
        report = run_backtest(load_csv(args.csv), initial_bankroll=args.bankroll)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
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
