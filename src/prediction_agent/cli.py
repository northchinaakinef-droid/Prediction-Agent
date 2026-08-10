from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from .backtest import load_csv, run_backtest
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
    args = parser.parse_args()
    if args.command == "polymarket":
        rows = PolymarketClient().search_sports(args.query, limit=args.limit)
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    elif args.command == "recommend":
        result = recommend(event_id=args.event_id, outcome=args.outcome,
                           model_probability=args.model_probability, decimal_odds=args.decimal_odds,
                           bankroll=args.bankroll, confidence=args.confidence)
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    else:
        report = run_backtest(load_csv(args.csv), initial_bankroll=args.bankroll)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

