"""Append-only SQLite ledger for genuinely forward sports predictions."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .providers.polymarket import PolymarketClient
from .costs import estimate_cost
from .narrative import build_post_match_summary


SCHEMA = """
CREATE TABLE IF NOT EXISTS report_runs (
    run_id TEXT PRIMARY KEY,
    generated_at TEXT NOT NULL,
    report_date TEXT NOT NULL,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES report_runs(run_id),
    generated_at TEXT NOT NULL,
    sport TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event TEXT,
    scheduled_start TEXT,
    outcome TEXT NOT NULL,
    model_probability REAL NOT NULL,
    market_probability REAL,
    execution_price REAL,
    closing_price REAL,
    clv REAL,
    action TEXT NOT NULL,
    stake REAL NOT NULL,
    probability_eligible INTEGER NOT NULL,
    real_money_approved INTEGER NOT NULL,
    market_started INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_event_idx ON predictions(sport, event_id, generated_at);
CREATE TABLE IF NOT EXISTS post_match_reviews (
    review_id TEXT PRIMARY KEY,
    sport TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event TEXT,
    generated_at TEXT NOT NULL,
    actual_winner TEXT NOT NULL,
    predicted_winner TEXT,
    prediction_correct INTEGER,
    model_probability REAL,
    bp_probability REAL,
    decisive_factors_json TEXT NOT NULL,
    review_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settlements (
    sport TEXT NOT NULL,
    event_id TEXT NOT NULL,
    winner TEXT NOT NULL,
    settled_at TEXT NOT NULL,
    source TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (sport, event_id)
);
"""


def _id(*values: object) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _migrate_predictions(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
    if "closing_price" not in columns:
        connection.execute("ALTER TABLE predictions ADD COLUMN closing_price REAL")
    if "clv" not in columns:
        connection.execute("ALTER TABLE predictions ADD COLUMN clv REAL")


def record_report(path: str | Path, report: dict[str, Any]) -> dict[str, int | str]:
    """Persist one immutable report and its rows; retries are idempotent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    generated = str(report["generated_at"])
    run_id = _id(generated, report.get("report_date"))
    inserted = 0
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        _migrate_predictions(connection)
        cursor = connection.execute(
            "INSERT OR IGNORE INTO report_runs VALUES (?, ?, ?, ?)",
            (run_id, generated, str(report.get("report_date", "")),
             json.dumps(report, ensure_ascii=False)),
        )
        new_run = cursor.rowcount == 1
        if new_run:
            for row in report.get("recommendations", []):
                prediction_id = _id(run_id, row.get("sport"), row.get("event_id"), row.get("outcome"))
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO predictions
                    (prediction_id, run_id, generated_at, sport, event_id, event,
                     scheduled_start, outcome, model_probability, market_probability,
                     execution_price, closing_price, clv, action, stake,
                     probability_eligible, real_money_approved, market_started, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (prediction_id, run_id, str(row.get("generated_at", generated)),
                     str(row.get("sport", "")), str(row.get("event_id", "")), row.get("event"),
                     row.get("scheduled_start"), str(row.get("outcome", "")),
                     row.get("model_probability"), row.get("market_probability"),
                     row.get("execution_price"), row.get("closing_price"), row.get("clv"),
                     str(row.get("action", "NO_BET")), float(row.get("stake") or 0),
                     int(bool(row.get("probability_eligible"))),
                     int(bool(row.get("real_money_approved"))), int(bool(row.get("market_started"))),
                     json.dumps(row, ensure_ascii=False)),
                )
                inserted += int(cursor.rowcount == 1)
        connection.commit()
        total = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return {"run_id": run_id, "new_run": int(new_run), "inserted_predictions": inserted,
            "total_predictions": int(total)}


def record_post_match_review(path: str | Path, review: dict[str, Any]) -> None:
    """Persist one post-match review row; repeated writes are idempotent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    review_id = _id(review.get("sport"), review.get("event_id"),
                    review.get("generated_at"), review.get("actual_winner"))
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        connection.execute(
            """INSERT OR IGNORE INTO post_match_reviews VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (review_id, str(review.get("sport", "")), str(review.get("event_id", "")),
             review.get("event"), str(review.get("generated_at", "")),
             str(review.get("actual_winner", "")), review.get("predicted_winner"),
             int(bool(review.get("prediction_correct"))), review.get("model_probability"),
             review.get("bp_probability"),
             json.dumps(review.get("decisive_factors", []), ensure_ascii=False),
             json.dumps(review, ensure_ascii=False)),
        )
        connection.commit()


def summary(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"runs": 0, "predictions": 0, "settled": 0, "by_sport": {}}
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        _migrate_predictions(connection)
        runs = connection.execute("SELECT COUNT(*) FROM report_runs").fetchone()[0]
        predictions = connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        settled = connection.execute("SELECT COUNT(*) FROM settlements").fetchone()[0]
        rows = connection.execute(
            "SELECT sport, COUNT(*), SUM(market_started=0), SUM(probability_eligible), SUM(action='BET') "
            "FROM predictions GROUP BY sport"
        ).fetchall()
        evaluated = connection.execute(
            """SELECT p.sport, p.outcome, p.model_probability, p.action, p.stake,
                      p.execution_price, s.winner, p.clv
               FROM predictions p JOIN settlements s
                 ON p.sport=s.sport AND p.event_id=s.event_id
               WHERE p.market_started=0"""
        ).fetchall()
    by_sport = {sport: {"predictions": count, "valid_forward_predictions": forward,
                        "probability_eligible": eligible,
                        "bet_candidates": bets, "settled_predictions": 0,
                        "model_brier": None, "paper_profit": 0.0, "paper_roi": None}
                for sport, count, forward, eligible, bets in rows}
    grouped: dict[str, list[tuple]] = {}
    for row in evaluated:
        grouped.setdefault(row[0], []).append(row)
    for sport, sport_rows in grouped.items():
        squared, profit, turnover, clv_values = [], 0.0, 0.0, []
        for _, outcome, probability, action, stake, execution, winner, clv in sport_rows:
            won = outcome == winner
            squared.append((float(probability) - float(won)) ** 2)
            if action == "BET" and float(stake) > 0 and execution:
                shares = float(stake) / float(execution)
                fee = estimate_cost(float(execution), float(stake))
                profit += shares * int(won) - float(stake) - fee
                turnover += float(stake)
            if clv is not None:
                clv_values.append(float(clv))
        by_sport[sport].update({"settled_predictions": len(sport_rows),
                                "model_brier": sum(squared) / len(squared),
                                "paper_profit": profit,
                                "paper_roi": profit / turnover if turnover else None,
                                "mean_clv": sum(clv_values) / len(clv_values) if clv_values else None})
    return {"runs": runs, "predictions": predictions, "settled": settled,
            "by_sport": by_sport}


def committed_exposure(path: str | Path, report_date: str, bankroll: float) -> tuple[float, dict[str, float]]:
    """Return already-committed BET stake fractions for a report date."""
    target = Path(path)
    if not target.exists() or bankroll <= 0:
        return 0.0, {}
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        rows = connection.execute(
            "SELECT p.event_id, SUM(p.stake) FROM predictions p JOIN report_runs r ON p.run_id = r.run_id WHERE p.action = 'BET' AND r.report_date = ? GROUP BY p.event_id",
            (report_date,),
        ).fetchall()
    event_stakes = {str(event_id): float(stake) for event_id, stake in rows if stake is not None}
    daily = sum(event_stakes.values()) / bankroll
    events = {event_id: stake / bankroll for event_id, stake in event_stakes.items()}
    return daily, events


def current_drawdown(path: str | Path, bankroll: float) -> float:
    """Return current peak-to-trough drawdown fraction from settled paper bets.

    Real-money mode would replace this with an external account-balance feed;
    the paper ledger already stores the forward settlement data needed for a
    conservative enforcement of MAX_DRAWDOWN_FRACTION today.
    """
    target = Path(path)
    if not target.exists() or bankroll <= 0:
        return 0.0
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        rows = connection.execute(
            """SELECT p.stake, p.execution_price, p.outcome, s.winner, s.settled_at
               FROM predictions p JOIN settlements s
                 ON p.sport=s.sport AND p.event_id=s.event_id
               WHERE p.action='BET' AND p.stake > 0 AND p.execution_price IS NOT NULL
               ORDER BY s.settled_at"""
        ).fetchall()
    equity = peak = float(bankroll)
    for stake, execution, outcome, winner, _ in rows:
        stake = float(stake)
        execution = float(execution)
        shares = stake / execution
        won = 1.0 if outcome == winner else 0.0
        pnl = shares * won - stake - estimate_cost(execution, stake)
        equity += pnl
        peak = max(peak, equity)
    return (peak - equity) / peak if peak > 0 else 0.0


def record_closing_line(path: str | Path, sport: str, event_id: str, closing_price: float) -> None:
    """Record the T-0 closing price once the market closes.

    CLV is signed such that a positive value means the recorded execution price
    was better than the final closing consensus.
    """
    target = Path(path)
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        _migrate_predictions(connection)
        connection.execute(
            """UPDATE predictions
               SET closing_price = ?,
                   clv = CASE WHEN execution_price IS NOT NULL THEN ? - execution_price ELSE NULL END
               WHERE sport = ? AND event_id = ?""",
            (float(closing_price), float(closing_price), sport, event_id),
        )
        connection.commit()


def paper_review(path: str | Path, since: str = "1970-01-01") -> list[dict[str, Any]]:
    """Return settled post-match reviews sorted by biggest miss first."""
    target = Path(path)
    if not target.exists():
        return []
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        rows = connection.execute(
            """SELECT sport, event_id, event, generated_at, actual_winner, predicted_winner,
                      model_probability, prediction_correct
               FROM post_match_reviews
               WHERE generated_at >= ?
               ORDER BY ABS(model_probability - prediction_correct) DESC""",
            (since,),
        ).fetchall()
    return [
        {
            "sport": sport, "event_id": event_id, "event": event, "generated_at": generated_at,
            "actual_winner": actual_winner, "predicted_winner": predicted_winner,
            "model_probability": model_probability, "prediction_correct": bool(prediction_correct),
            "miss": abs(float(model_probability) - float(prediction_correct)),
        }
        for sport, event_id, event, generated_at, actual_winner, predicted_winner,
            model_probability, prediction_correct in rows
    ]


def paper_review_detail(path: str | Path, since: str = "1970-01-01") -> list[dict[str, Any]]:
    """Return full post-match review JSON for downstream model iteration."""
    target = Path(path)
    if not target.exists():
        return []
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        rows = connection.execute(
            "SELECT review_json FROM post_match_reviews WHERE generated_at >= ? ORDER BY generated_at",
            (since,),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def _list(value: Any) -> list[Any]:
    return json.loads(value) if isinstance(value, str) else list(value or [])


def settle_pending(path: str | Path, client: PolymarketClient | None = None) -> dict[str, int]:
    """Resolve prior events from Polymarket; unresolved events remain pending."""
    target = Path(path)
    if not target.exists():
        return {"checked": 0, "settled": 0, "pending": 0}
    client = client or PolymarketClient(timeout=30)
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        _migrate_predictions(connection)
        pending = connection.execute(
            """SELECT DISTINCT p.sport, p.event_id FROM predictions p
               LEFT JOIN settlements s ON p.sport=s.sport AND p.event_id=s.event_id
               WHERE s.event_id IS NULL AND p.market_started=0"""
        ).fetchall()
        settled_count = errors = 0
        for sport, event_id in pending:
            try:
                event = client.event(str(event_id))
            except Exception:
                errors += 1
                logging.exception('settle_pending: unexpected error contacting Polymarket for %s', event_id)
                continue
            winner = None
            market_payload = None
            for market in event.get("markets", []):
                outcomes = [str(value) for value in _list(market.get("outcomes"))]
                prices = [float(value) for value in _list(market.get("outcomePrices"))]
                if len(outcomes) == len(prices) == 2 and sorted(prices) == [0.0, 1.0]:
                    if market.get("sportsMarketType") in (None, "moneyline") and market.get("gameStartTime"):
                        winner = outcomes[prices.index(1.0)]
                        market_payload = market
                        break
            if winner is None:
                continue
            settled_at = str(event.get("closedTime") or event.get("endDate") or event.get("updatedAt") or "")
            pred = connection.execute(
                "SELECT generated_at, event, outcome, model_probability, market_probability "
                "FROM predictions WHERE sport=? AND event_id=? AND market_started=0 LIMIT 1",
                (sport, event_id),
            ).fetchone()
            connection.execute(
                "INSERT OR IGNORE INTO settlements VALUES (?, ?, ?, ?, ?, ?)",
                (sport, event_id, winner, settled_at, "Polymarket Gamma",
                 json.dumps(market_payload, ensure_ascii=False)),
            )
            if pred:
                generated_at, event_name, predicted_winner, model_probability, market_probability = pred
                review = {
                    "sport": sport,
                    "event_id": event_id,
                    "event": event_name,
                    "generated_at": generated_at,
                    "actual_winner": winner,
                    "predicted_winner": predicted_winner,
                    "prediction_correct": int(predicted_winner == winner),
                    "model_probability": model_probability,
                    "bp_probability": market_probability,
                    "decisive_factors": [],
                }
                review["narrative_summary"] = build_post_match_summary(review)
                connection.execute(
                    """INSERT OR IGNORE INTO post_match_reviews VALUES
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (_id(sport, event_id, generated_at, winner), sport, event_id, event_name,
                     generated_at, winner, predicted_winner, review["prediction_correct"],
                     model_probability, market_probability,
                     json.dumps(review["decisive_factors"], ensure_ascii=False),
                     json.dumps(review, ensure_ascii=False)),
                )
            settled_count += 1
        connection.commit()
    return {"checked": len(pending), "settled": settled_count,
            "pending": len(pending) - settled_count, "errors": errors}
