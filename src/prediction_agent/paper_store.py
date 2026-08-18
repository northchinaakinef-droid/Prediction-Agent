"""Append-only SQLite ledger for genuinely forward sports predictions."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
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
    lineup_status TEXT,
    ev_tier TEXT,
    direction_match INTEGER,
    settled_profit REAL,
    settled_winner TEXT,
    payload_json TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0
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
CREATE TABLE IF NOT EXISTS paper_summary_sends (
    report_date TEXT PRIMARY KEY,
    sent_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS weekly_attribution_sends (
    report_week TEXT PRIMARY KEY,
    sent_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS account_events (
    event_id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    reason TEXT,
    equity_before REAL,
    drawdown_before REAL,
    bankroll REAL,
    payload_json TEXT NOT NULL
);
"""


def _id(*values: object) -> str:
    return hashlib.sha256("|".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _paper_bet_profit(stake, execution_price, won):
    # Return settled profit for one paper bet, using the shared cost model.
    stake = float(stake or 0)
    execution = float(execution_price) if execution_price is not None else 0.0
    if stake <= 0 or execution <= 0:
        return 0.0
    shares = stake / execution
    return shares * int(bool(won)) - stake - estimate_cost(execution, stake)


def _attribution_ev_tier(value: Any) -> str | None:
    """Compute the paper-betting EV tier from an expected_value."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value > 0.15:
        return "高"
    if value > 0.05:
        return "中"
    return "低"


def _attribution_direction(market_probability: Any) -> int | None:
    """Derive direction_match from the chosen side's stored market probability."""
    if market_probability is None:
        return None
    try:
        market_probability = float(market_probability)
    except (TypeError, ValueError):
        return None
    return int(market_probability >= 0.5)


def _migrate_predictions(connection: sqlite3.Connection) -> None:
    columns = {row[1] for row in connection.execute("PRAGMA table_info(predictions)")}
    for column, definition in (
        ("closing_price", "closing_price REAL"),
        ("clv", "clv REAL"),
        ("lineup_status", "lineup_status TEXT"),
        ("ev_tier", "ev_tier TEXT"),
        ("direction_match", "direction_match INTEGER"),
        ("settled_profit", "settled_profit REAL"),
        ("settled_winner", "settled_winner TEXT"),
        ("archived", "archived INTEGER NOT NULL DEFAULT 0"),
    ):
        if column not in columns:
            connection.execute(f"ALTER TABLE predictions ADD COLUMN {definition}")


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
                lineup_status = row.get("lineup_status") or "未知"
                ev_tier = row.get("ev_tier") or _attribution_ev_tier(row.get("expected_value")) or "未知"
                direction_match = row.get("direction_match")
                if direction_match is None:
                    direction_match = _attribution_direction(row.get("market_probability"))
                cursor = connection.execute(
                    """INSERT OR IGNORE INTO predictions
                    (prediction_id, run_id, generated_at, sport, event_id, event,
                     scheduled_start, outcome, model_probability, market_probability,
                     execution_price, closing_price, clv, action, stake,
                     probability_eligible, real_money_approved, market_started,
                     lineup_status, ev_tier, direction_match, settled_profit, settled_winner,
                     payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (prediction_id, run_id, str(row.get("generated_at", generated)),
                     str(row.get("sport", "")), str(row.get("event_id", "")), row.get("event"),
                     row.get("scheduled_start"), str(row.get("outcome", "")),
                     row.get("model_probability"), row.get("market_probability"),
                     row.get("execution_price"), row.get("closing_price"), row.get("clv"),
                     str(row.get("action", "NO_BET")), float(row.get("stake") or 0),
                     int(bool(row.get("probability_eligible"))),
                     int(bool(row.get("real_money_approved"))), int(bool(row.get("market_started"))),
                     lineup_status, ev_tier,
                     int(bool(direction_match)) if direction_match is not None else None,
                     None, None,
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


def paper_daily_summary(path: str | Path, report_date: str) -> dict[str, Any]:
    """Return one report date's paper-betting summary instead of all-time totals."""
    target = Path(path)
    if not target.exists():
        return {"report_date": report_date, "predictions": 0, "bet_candidates": 0,
                "settled_predictions": 0, "paper_profit": 0.0, "paper_roi": None,
                "by_sport": {}}
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        _migrate_predictions(connection)
        rows = connection.execute(
            """SELECT p.sport, COUNT(*), SUM(p.market_started=0),
                      SUM(p.probability_eligible), SUM(p.action='BET')
               FROM predictions p JOIN report_runs r ON p.run_id = r.run_id
               WHERE r.report_date = ?
               GROUP BY p.sport""",
            (report_date,),
        ).fetchall()
        evaluated = connection.execute(
            """SELECT p.sport, p.outcome, p.model_probability, p.action, p.stake,
                      p.execution_price, s.winner, p.clv
               FROM predictions p
               JOIN report_runs r ON p.run_id = r.run_id
               JOIN settlements s ON p.sport = s.sport AND p.event_id = s.event_id
               WHERE r.report_date = ? AND p.market_started = 0""",
            (report_date,),
        ).fetchall()
    by_sport = {str(sport): {
        "predictions": int(count or 0),
        "valid_forward_predictions": int(forward or 0),
        "probability_eligible": int(eligible or 0),
        "bet_candidates": int(bets or 0),
        "settled_predictions": 0,
        "model_brier": None,
        "paper_profit": 0.0,
        "paper_roi": None,
    } for sport, count, forward, eligible, bets in rows}
    grouped: dict[str, list[tuple]] = {}
    for row in evaluated:
        grouped.setdefault(str(row[0]), []).append(row)
    total_profit = 0.0
    total_turnover = 0.0
    for sport, sport_rows in grouped.items():
        squared, profit, turnover, clv_values = [], 0.0, 0.0, []
        for _, outcome, probability, action, stake, execution, winner, clv in sport_rows:
            won = str(outcome or "").strip() == str(winner or "").strip()
            squared.append((float(probability) - float(won)) ** 2)
            if action == "BET" and float(stake) > 0 and execution:
                profit += _paper_bet_profit(stake, execution, won)
                turnover += float(stake)
            if clv is not None:
                clv_values.append(float(clv))
        by_sport.setdefault(sport, {
            "predictions": 0, "valid_forward_predictions": 0,
            "probability_eligible": 0, "bet_candidates": 0,
        })
        by_sport[sport].update({
            "settled_predictions": len(sport_rows),
            "model_brier": sum(squared) / len(squared),
            "paper_profit": profit,
            "paper_roi": profit / turnover if turnover else None,
            "mean_clv": sum(clv_values) / len(clv_values) if clv_values else None,
        })
        total_profit += profit
        total_turnover += turnover
    total_predictions = sum(stat["predictions"] for stat in by_sport.values())
    total_bets = sum(stat["bet_candidates"] for stat in by_sport.values())
    total_settled = sum(stat["settled_predictions"] for stat in by_sport.values())
    return {
        "report_date": report_date,
        "predictions": total_predictions,
        "bet_candidates": total_bets,
        "settled_predictions": total_settled,
        "paper_profit": total_profit,
        "paper_roi": total_profit / total_turnover if total_turnover else None,
        "by_sport": by_sport,
    }


def paper_summary_sent(path: str | Path, report_date: str) -> bool:
    target = Path(path)
    if not target.exists():
        return False
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        row = connection.execute(
            "SELECT 1 FROM paper_summary_sends WHERE report_date = ?",
            (report_date,),
        ).fetchone()
    return row is not None


def mark_paper_summary_sent(path: str | Path, report_date: str, sent_at: str) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        cursor = connection.execute(
            """INSERT OR IGNORE INTO paper_summary_sends(report_date, sent_at)
               VALUES (?, ?)""",
            (report_date, sent_at),
        )
        connection.commit()
    return cursor.rowcount == 1


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
            won = str(outcome or "").strip() == str(winner or "").strip()
            squared.append((float(probability) - float(won)) ** 2)
            if action == "BET" and float(stake) > 0 and execution:
                profit += _paper_bet_profit(stake, execution, won)
                turnover += float(stake)
            if clv is not None:
                clv_values.append(float(clv))
        by_sport[sport].update({"settled_predictions": len(sport_rows),
                                "model_brier": sum(squared) / len(squared),
                                "paper_profit": profit,
                                "paper_roi": profit / turnover if turnover else None,
                                "paper_turnover": turnover,
                                "mean_clv": sum(clv_values) / len(clv_values) if clv_values else None})
    total_profit = sum(float(stat["paper_profit"]) for stat in by_sport.values())
    total_turnover = sum(float(stat.get("paper_turnover", 0.0)) for stat in by_sport.values())
    return {
        "runs": runs,
        "predictions": predictions,
        "settled": settled,
        "paper_profit": total_profit,
        "paper_roi": total_profit / total_turnover if total_turnover else None,
        "settled_bets": sum(int(stat["bet_candidates"]) for stat in by_sport.values()),
        "by_sport": by_sport,
    }


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
               WHERE p.action='BET' AND p.archived=0 AND p.stake > 0
                 AND p.execution_price IS NOT NULL
               ORDER BY s.settled_at"""
        ).fetchall()
    equity = peak = float(bankroll)
    for stake, execution, outcome, winner, _ in rows:
        pnl = _paper_bet_profit(stake, execution, str(outcome or "").strip() == str(winner or "").strip())
        equity += pnl
        peak = max(peak, equity)
    return (peak - equity) / peak if peak > 0 else 0.0


def reset_paper_account(path: str | Path, bankroll: float, reason: str = "manual_reset") -> dict[str, Any]:
    """Archive all existing BET rows and record a RESET account event.

    Reset does not delete history.  Every existing BET prediction is marked
    ``archived=1``, a RESET event captures the pre-reset equity and drawdown,
    and ``current_drawdown()`` starts over from the new bankroll baseline.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    reset_at = datetime.now(timezone.utc).isoformat()
    reset_id = _id("reset", reset_at, reason)
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        _migrate_predictions(connection)
        rows = connection.execute(
            """SELECT p.stake, p.execution_price, p.outcome, s.winner, s.settled_at
               FROM predictions p JOIN settlements s
                 ON p.sport=s.sport AND p.event_id=s.event_id
               WHERE p.action='BET' AND p.archived=0 AND p.stake > 0
                 AND p.execution_price IS NOT NULL
               ORDER BY s.settled_at"""
        ).fetchall()
        equity = peak = float(bankroll)
        for stake, execution, outcome, winner, _ in rows:
            won = str(outcome or "").strip() == str(winner or "").strip()
            equity += _paper_bet_profit(stake, execution, won)
            peak = max(peak, equity)
        equity_before = equity
        drawdown_before = (peak - equity) / peak if peak > 0 else 0.0
        archived = connection.execute(
            "UPDATE predictions SET archived=1 WHERE action='BET'"
        ).rowcount
        payload = {
            "reset_at": reset_at,
            "reason": reason,
            "bankroll": float(bankroll),
            "equity_before": equity_before,
            "drawdown_before": drawdown_before,
            "archived_bet_rows": int(archived),
        }
        connection.execute(
            """INSERT OR IGNORE INTO account_events
               (event_id, action, occurred_at, reason, equity_before, drawdown_before,
                bankroll, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (reset_id, "RESET", reset_at, reason, equity_before, drawdown_before,
             float(bankroll), json.dumps(payload, ensure_ascii=False)),
        )
        connection.commit()
    return {"reset_id": reset_id, "reset_at": reset_at, "reason": reason,
            "equity_before": equity_before, "drawdown_before": drawdown_before,
            "archived_bet_rows": int(archived), "bankroll": float(bankroll)}


def attribution(path: str | Path, *, since: str | None = None) -> dict[str, Any]:
    """Group settled paper bets by the new cold-start attribution fields."""
    target = Path(path)
    empty = {"by_lineup_status": {}, "by_ev_tier": {}, "by_direction_match": {},
             "by_sport": {}, "by_data_quality": {}, "sample_count": 0,
             "legacy_sample_count": 0, "total_sample_count": 0}
    if not target.exists():
        return empty
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        _migrate_predictions(connection)
        sql = """SELECT p.sport, p.lineup_status, p.ev_tier, p.direction_match,
                        p.outcome, p.action, p.stake, p.execution_price,
                        COALESCE(p.settled_winner, s.winner) AS winner
                 FROM predictions p JOIN settlements s
                   ON p.sport = s.sport AND p.event_id = s.event_id
                 WHERE p.market_started = 0 AND p.action = 'BET'"""
        params: list[Any] = []
        if since:
            sql += " AND s.settled_at >= ?"
            params.append(since)
        rows = connection.execute(sql, params).fetchall()

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        bets = len(items)
        wins = sum(
            1 for item in items
            if str(item.get("outcome") or "").strip() == str(item.get("winner") or "").strip()
        )
        profit = sum(
            _paper_bet_profit(
                item.get("stake"), item.get("execution_price"),
                str(item.get("outcome") or "").strip() == str(item.get("winner") or "").strip(),
            )
            for item in items
        )
        turnover = sum(float(item.get("stake") or 0) for item in items)
        return {
            "bets": bets,
            "wins": wins,
            "win_rate": wins / bets if bets else None,
            "profit": profit,
            "turnover": turnover,
            "roi": profit / turnover if turnover else None,
        }

    by_lineup: dict[str, list[dict[str, Any]]] = {}
    by_ev: dict[str, list[dict[str, Any]]] = {}
    by_dir: dict[str, list[dict[str, Any]]] = {}
    by_sport: dict[str, list[dict[str, Any]]] = {}
    complete_items: list[dict[str, Any]] = []
    legacy_items: list[dict[str, Any]] = []
    for row in rows:
        sport = str(row[0])
        lineup = row[1]
        ev_tier = row[2]
        raw_dir = row[3]
        item = {
            "outcome": row[4],
            "stake": row[6],
            "execution_price": row[7],
            "winner": row[8],
        }
        if lineup is None and ev_tier is None and raw_dir is None:
            legacy_items.append(item)
            continue
        complete_items.append(item)
        if raw_dir in (1, True):
            dir_key = "一致"
        elif raw_dir in (0, False):
            dir_key = "相反"
        else:
            dir_key = "未知"
        by_lineup.setdefault(str(lineup or "未知"), []).append(item)
        by_ev.setdefault(str(ev_tier or "未知"), []).append(item)
        by_dir.setdefault(dir_key, []).append(item)
        by_sport.setdefault(sport, []).append(item)

    return {
        "by_lineup_status": {key: stats(value) for key, value in by_lineup.items()},
        "by_ev_tier": {key: stats(value) for key, value in by_ev.items()},
        "by_direction_match": {key: stats(value) for key, value in by_dir.items()},
        "by_sport": {key: stats(value) for key, value in by_sport.items()},
        "by_data_quality": {
            "完整归因字段": stats(complete_items),
            "历史数据（字段缺失）": stats(legacy_items),
        },
        "sample_count": len(complete_items),
        "legacy_sample_count": len(legacy_items),
        "total_sample_count": len(rows),
    }


def weekly_attribution_sent(path: str | Path, report_week: str) -> bool:
    target = Path(path)
    if not target.exists():
        return False
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        row = connection.execute(
            "SELECT 1 FROM weekly_attribution_sends WHERE report_week = ?",
            (report_week,),
        ).fetchone()
    return row is not None


def mark_weekly_attribution_sent(path: str | Path, report_week: str, sent_at: str) -> bool:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        cursor = connection.execute(
            """INSERT OR IGNORE INTO weekly_attribution_sends(report_week, sent_at)
               VALUES (?, ?)""",
            (report_week, sent_at),
        )
        connection.commit()
    return cursor.rowcount == 1


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
        return {"checked": 0, "settled": 0, "pending": 0, "updated_predictions": 0, "errors": 0}
    client = client or PolymarketClient(timeout=30)
    retries = max(0, int(os.getenv("PAPER_SETTLE_RETRIES", "2")))
    settled_count = updated_predictions = errors = 0
    with closing(sqlite3.connect(target)) as connection:
        connection.executescript(SCHEMA)
        _migrate_predictions(connection)
        pending = connection.execute(
            """SELECT DISTINCT p.sport, p.event_id FROM predictions p
               LEFT JOIN settlements s ON p.sport=s.sport AND p.event_id=s.event_id
               WHERE s.event_id IS NULL AND p.market_started=0"""
        ).fetchall()
        for sport, event_id in pending:
            event = None
            for attempt in range(retries + 1):
                try:
                    event = client.event(str(event_id))
                    break
                except Exception:
                    if attempt >= retries:
                        errors += 1
                        logging.exception('settle_pending: unexpected error contacting Polymarket for %s', event_id)
                        break
                    time.sleep(min(2 ** attempt, 5))
            if event is None:
                continue
            winner = None
            market_payload = None
            for market in event.get("markets", []):
                outcomes = [str(value).strip() for value in _list(market.get("outcomes"))]
                prices = [float(value) for value in _list(market.get("outcomePrices"))]
                if len(outcomes) == len(prices) == 2 and sorted(prices) == [0.0, 1.0]:
                    if market.get("sportsMarketType") in (None, "moneyline") and market.get("gameStartTime"):
                        winner = outcomes[prices.index(1.0)]
                        market_payload = market
                        break
            if winner is None:
                continue
            settled_at = str(event.get("closedTime") or event.get("endDate") or event.get("updatedAt") or "")
            connection.execute(
                "INSERT OR IGNORE INTO settlements VALUES (?, ?, ?, ?, ?, ?)",
                (sport, event_id, winner, settled_at, "Polymarket Gamma",
                 json.dumps(market_payload, ensure_ascii=False)),
            )
            pred = connection.execute(
                "SELECT generated_at, event, outcome, model_probability, market_probability "
                "FROM predictions WHERE sport=? AND event_id=? AND market_started=0 LIMIT 1",
                (sport, event_id),
            ).fetchone()
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
            prediction_rows = connection.execute(
                "SELECT prediction_id, outcome, action, stake, execution_price "
                "FROM predictions WHERE sport=? AND event_id=? AND market_started=0",
                (sport, event_id),
            ).fetchall()
            for prediction_id, predicted_outcome, action, stake, execution_price in prediction_rows:
                won = str(predicted_outcome or "").strip() == str(winner or "").strip()
                profit = _paper_bet_profit(stake, execution_price, won) if action == "BET" else 0.0
                connection.execute(
                    "UPDATE predictions SET settled_winner=?, settled_profit=? WHERE prediction_id=?",
                    (winner, profit, prediction_id),
                )
                updated_predictions += 1
            settled_count += 1
        connection.commit()
    return {"checked": len(pending), "settled": settled_count,
            "pending": len(pending) - settled_count, "errors": errors,
            "updated_predictions": updated_predictions}
