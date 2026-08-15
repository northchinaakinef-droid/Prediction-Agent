from __future__ import annotations

import json
import logging
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .entities import canonical_team, normalized_name
from .providers.live_data import LiveState
from .providers.polymarket import PolymarketClient


def _logit(value: float) -> float:
    value = min(.999, max(.001, value))
    return math.log(value / (1 - value))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-20, min(20, value))))


@dataclass
class ProbabilityUpdate:
    pre_match_probability: float | None
    current_probability: float | None
    probability_change: float | None
    method: str
    calibration_status: str
    reasons: list[str]
    confidence: str = "LOW"


@dataclass
class MarketState:
    market_id: str | None
    outcome_a: str | None
    probability_a: float | None
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    depth: float | None
    imbalance: float | None
    observed_at: datetime
    available: bool
    error: str | None = None
    mid_price: float | None = None
    last_price: float | None = None
    volume: float | None = None


@dataclass
class LiveAlert:
    match_key: str
    sport: str
    severity: str
    alert_score: float
    category: str
    title: str
    summary: str
    reasons: list[str]
    observed_at: datetime
    dedupe_key: str
    details: dict[str, Any] | None = None

    def as_dict(self) -> dict:
        row = asdict(self)
        row["observed_at"] = self.observed_at.isoformat()
        return row


class DynamicProbabilityEngine:
    """Transparent research-only live updates; never grants trading approval."""

    def update(self, state: LiveState, prior: float | None) -> ProbabilityUpdate:
        if prior is None:
            return ProbabilityUpdate(None, None, None, "UNAVAILABLE", "NO_PRIOR",
                                     ["independent pre-match probability is unavailable"])
        score = _logit(prior)
        reasons = []
        f = state.features
        if state.sport == "nba":
            if state.score_a is None or state.score_b is None:
                return ProbabilityUpdate(prior, None, None, "UNAVAILABLE", "MISSING_LIVE_SCORE",
                                         ["NBA live score is unavailable"])
            period = int(f.get("period") or 0)
            clock = float(f.get("game_clock_seconds") or 0)
            remaining = max(30.0, max(0, 4 - period) * 720 + clock)
            differential = state.score_a - state.score_b
            leverage = math.sqrt(2880 / remaining)
            score += differential * .115 * leverage
            reasons.append(f"score differential {differential:+.0f}, estimated regulation time remaining {remaining:.0f}s")
        elif state.sport == "lol":
            if f.get("post_draft_probability") is not None:
                draft_probability = float(f["post_draft_probability"])
                score = _logit(draft_probability)
                reasons.append(f"BP/player-champion model changed prior {prior:.1%}→{draft_probability:.1%}")
            required = ("gold_a", "gold_b", "kills_a", "kills_b", "towers_a", "towers_b")
            if any(f.get(name) is None for name in required):
                if f.get("post_draft_probability") is not None:
                    current = _sigmoid(score)
                    return ProbabilityUpdate(
                        prior, current, current - prior, "LOL_POST_DRAFT_MODEL",
                        "UNVALIDATED_RESEARCH_ONLY", reasons,
                    )
                return ProbabilityUpdate(prior, None, None, "UNAVAILABLE", "MISSING_LIVE_FEATURES",
                                         ["LoL gold/kills/towers live fields are incomplete"])
            gold = float(f["gold_a"]) - float(f["gold_b"])
            kills = float(f["kills_a"]) - float(f["kills_b"])
            towers = float(f["towers_a"]) - float(f["towers_b"])
            dragons = float(f.get("dragons_a") or 0) - float(f.get("dragons_b") or 0)
            barons = float(f.get("barons_a") or 0) - float(f.get("barons_b") or 0)
            inhibitors = float(f.get("inhibitors_a") or 0) - float(f.get("inhibitors_b") or 0)
            score += gold * .000075 + kills * .075 + towers * .16 + dragons * .14 + barons * .42 + inhibitors * .3
            reasons.append(f"gold {gold:+.0f}, kills {kills:+.0f}, towers {towers:+.0f}, dragons {dragons:+.0f}, barons {barons:+.0f}")
        elif state.sport == "cs2":
            series = float(f.get("series_score_a") or state.score_a or 0) - float(
                f.get("series_score_b") or state.score_b or 0)
            rounds = float(f.get("round_score_a") or 0) - float(f.get("round_score_b") or 0)
            score += series * .7 + rounds * .12
            reasons.append(f"series maps {series:+.0f}, current-map rounds {rounds:+.0f}")
        else:
            return ProbabilityUpdate(prior, None, None, "UNAVAILABLE", "UNSUPPORTED_SPORT", [])
        current = _sigmoid(score)
        return ProbabilityUpdate(
            prior, current, current - prior, "TRANSPARENT_LIVE_HEURISTIC",
            "UNVALIDATED_RESEARCH_ONLY", reasons,
        )


def _field(value: Any) -> list:
    return json.loads(value) if isinstance(value, str) else list(value or [])


class PolymarketMonitor:
    def __init__(self, client: PolymarketClient | None = None):
        self.client = client or PolymarketClient(timeout=10)

    @staticmethod
    def _market(state: LiveState, events: list[dict]) -> tuple[dict, list[str]] | None:
        wanted = {normalized_name(canonical_team(state.sport, state.team_a)),
                  normalized_name(canonical_team(state.sport, state.team_b))}
        for event in events:
            for market in event.get("markets", []):
                if market.get("sportsMarketType") != "moneyline":
                    continue
                outcomes = [canonical_team(state.sport, str(value)) for value in _field(market.get("outcomes"))]
                if len(outcomes) == 2 and {normalized_name(value) for value in outcomes} == wanted:
                    return market, outcomes
        return None

    def snapshot(self, state: LiveState, events: list[dict]) -> MarketState:
        now = datetime.now(timezone.utc)
        found = self._market(state, events)
        if not found:
            return MarketState(None, None, None, None, None, None, None, None, now, False,
                               "moneyline market not found")
        market, outcomes = found
        prices = [float(value) for value in _field(market.get("outcomePrices"))]
        index = 0 if normalized_name(canonical_team(state.sport, outcomes[0])) == normalized_name(
            canonical_team(state.sport, state.team_a)) else 1
        probability = prices[index] if len(prices) == 2 else None
        bid = float(market["bestBid"]) if market.get("bestBid") is not None else None
        ask = float(market["bestAsk"]) if market.get("bestAsk") is not None else None
        if index == 1:
            bid, ask = (1 - ask if ask is not None else None), (1 - bid if bid is not None else None)
        depth = float(market.get("liquidity") or 0)
        imbalance = None
        try:
            token_ids = self.client.token_ids(market)
            if len(token_ids) == 2:
                book = self.client.order_book(token_ids[index])
                bids = [(float(row["price"]), float(row["size"])) for row in book.get("bids", [])]
                asks = [(float(row["price"]), float(row["size"])) for row in book.get("asks", [])]
                bid = max((row[0] for row in bids), default=bid)
                ask = min((row[0] for row in asks), default=ask)
                bid_depth = sum(size for _, size in sorted(bids, reverse=True)[:5])
                ask_depth = sum(size for _, size in sorted(asks)[:5])
                depth = bid_depth + ask_depth
                imbalance = ((bid_depth - ask_depth) / depth) if depth else None
        except Exception:
            logging.debug("order-book enrichment unavailable for %s", market.get("id"), exc_info=True)
        spread = ask - bid if bid is not None and ask is not None else None
        mid = (bid + ask) / 2 if bid is not None and ask is not None else probability
        last = float(market.get("lastTradePrice")) if market.get("lastTradePrice") is not None else probability
        if index == 1 and last is not None:
            last = 1 - last
        volume = float(market.get("volumeNum") or market.get("volume") or 0)
        return MarketState(str(market.get("id")), outcomes[index], probability, bid, ask, spread,
                           depth, imbalance, now, True, mid_price=mid, last_price=last, volume=volume)


class LiveStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init(self) -> None:
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS live_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT, match_key TEXT NOT NULL, observed_at TEXT NOT NULL,
              sport TEXT NOT NULL, source TEXT NOT NULL, state_json TEXT NOT NULL,
              pre_match_probability REAL, current_probability REAL, market_probability REAL,
              probability_method TEXT NOT NULL, calibration_status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS live_snapshot_match_time ON live_snapshots(match_key, observed_at);
            CREATE TABLE IF NOT EXISTS market_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT, match_key TEXT NOT NULL, observed_at TEXT NOT NULL,
              market_id TEXT, market_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS live_alerts (
              id INTEGER PRIMARY KEY AUTOINCREMENT, dedupe_key TEXT NOT NULL, observed_at TEXT NOT NULL,
              alert_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS live_alert_dedupe ON live_alerts(dedupe_key, observed_at);
            """)

    def previous(self, match_key: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM live_snapshots WHERE match_key=? ORDER BY observed_at DESC LIMIT 1", (match_key,)
            ).fetchone()
            market = db.execute(
                "SELECT market_json FROM market_snapshots WHERE match_key=? ORDER BY observed_at DESC LIMIT 1",
                (match_key,),
            ).fetchone()
        result = dict(row) if row else None
        if result is not None and market:
            result["_market"] = json.loads(market["market_json"])
        return result

    def save(self, match_key: str, state: LiveState, probability: ProbabilityUpdate,
             market: MarketState) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO live_snapshots(match_key,observed_at,sport,source,state_json,pre_match_probability,current_probability,market_probability,probability_method,calibration_status) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (match_key, state.observed_at.isoformat(), state.sport, state.source,
                 json.dumps(state.as_dict(), ensure_ascii=False), probability.pre_match_probability,
                 probability.current_probability, market.probability_a, probability.method,
                 probability.calibration_status),
            )
            db.execute(
                "INSERT INTO market_snapshots(match_key,observed_at,market_id,market_json) VALUES(?,?,?,?)",
                (match_key, market.observed_at.isoformat(), market.market_id,
                 json.dumps(asdict(market), ensure_ascii=False, default=str)),
            )

    def legacy_postmatch_dedupe_keys(self) -> list[dict[str, str]]:
        with self.connect() as db:
            rows = db.execute(
                """SELECT dedupe_key, MIN(observed_at) AS observed_at
                   FROM live_alerts
                   WHERE dedupe_key LIKE '%:POSTMATCH_REVIEW:%'
                   GROUP BY dedupe_key"""
            ).fetchall()
        return [dict(row) for row in rows]

    def legacy_prematch_dedupe_rows(self) -> list[dict[str, str]]:
        """Return pre-match alerts stored under the old match_id-based key."""
        with self.connect() as db:
            rows = db.execute(
                """SELECT dedupe_key, MIN(observed_at) AS observed_at, alert_json
                   FROM live_alerts
                   WHERE dedupe_key LIKE '%:PREMATCH_ANALYSIS'
                   GROUP BY dedupe_key"""
            ).fetchall()
        return [dict(row) for row in rows]

    def ensure_alert_marker(self, dedupe_key: str, observed_at: str) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO live_alerts(dedupe_key, observed_at, alert_json)
                   SELECT ?, ?, '{}'
                   WHERE NOT EXISTS (
                       SELECT 1 FROM live_alerts WHERE dedupe_key = ?
                   )""",
                (dedupe_key, observed_at, dedupe_key),
            )

    def alert_recent(self, dedupe_key: str, cooldown_seconds: int) -> bool:
        cutoff = datetime.now(timezone.utc).timestamp() - cooldown_seconds
        with self.connect() as db:
            rows = db.execute("SELECT observed_at FROM live_alerts WHERE dedupe_key=? ORDER BY observed_at DESC LIMIT 1",
                              (dedupe_key,)).fetchall()
        return bool(rows and datetime.fromisoformat(rows[0]["observed_at"]).timestamp() >= cutoff)

    def alert_exists(self, dedupe_key: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT 1 FROM live_alerts WHERE dedupe_key=? LIMIT 1", (dedupe_key,)).fetchone()
        return row is not None

    def save_alert(self, alert: LiveAlert) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO live_alerts(dedupe_key,observed_at,alert_json) VALUES(?,?,?)",
                       (alert.dedupe_key, alert.observed_at.isoformat(),
                        json.dumps(alert.as_dict(), ensure_ascii=False)))

    def history(self, match_key: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM live_snapshots WHERE match_key=? ORDER BY observed_at", (match_key,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["state"] = json.loads(item.pop("state_json"))
            result.append(item)
        return result


def match_key(state: LiveState) -> str:
    teams = sorted((normalized_name(canonical_team(state.sport, state.team_a)),
                    normalized_name(canonical_team(state.sport, state.team_b))))
    return f"{state.sport}:{teams[0]}:{teams[1]}"


class AlertEngine:
    def evaluate(self, state: LiveState, probability: ProbabilityUpdate, market: MarketState,
                 previous: dict | None) -> LiveAlert | None:
        if probability.current_probability is None:
            return None
        previous_probability = float(previous["current_probability"]) if previous and previous.get(
            "current_probability") is not None else probability.pre_match_probability
        probability_move = abs(probability.current_probability - (previous_probability or probability.current_probability))
        divergence = abs(probability.current_probability - market.probability_a) if market.probability_a is not None else 0
        previous_market = float(previous["market_probability"]) if previous and previous.get(
            "market_probability") is not None else market.probability_a
        market_move = abs((market.probability_a or 0) - (previous_market or market.probability_a or 0))
        previous_state = json.loads(previous["state_json"]) if previous else {}
        prior_features = previous_state.get("features", {})
        objective_weight = 0.0
        key_events = []
        for name, weight in (("barons_a", 18), ("barons_b", 18), ("dragons_a", 7), ("dragons_b", 7),
                             ("towers_a", 4), ("towers_b", 4), ("round_score_a", 3), ("round_score_b", 3)):
            before, after = prior_features.get(name), state.features.get(name)
            if before is not None and after is not None and float(after) > float(before):
                objective_weight += weight
                key_events.append(f"{name} {before}→{after}")
        news_weight = float(state.features.get("news_importance") or 0)
        key_events.extend(state.key_events)
        previous_volume = float(previous.get("_market", {}).get("volume") or 0) if previous else 0
        volume_change = max(0.0, float(market.volume or 0) - previous_volume)
        liquidity_weight = (8 if market.spread is not None and market.spread >= .08 else 0) + (
            10 if previous_volume and volume_change / previous_volume >= .5 else 0)
        score = min(100.0, probability_move * 300 + market_move * 220 + divergence * 120 +
                    objective_weight + news_weight + liquidity_weight)
        if score < 30:
            return None
        severity = "EMERGENCY" if score >= 80 else "IMPORTANT" if score >= 60 else "OBSERVE"
        category = ("NEWS_ALERT" if news_weight >= 25 else "MARKET_ANOMALY" if divergence >= .12 else
                    "MAJOR_EVENT" if objective_weight >= 15 else "PROBABILITY_CHANGE")
        key = match_key(state)
        reasons = list(probability.reasons)
        if market.probability_a is not None:
            reasons.append(f"model-market divergence {probability.current_probability-market.probability_a:+.1%}")
            reasons.append(f"market move since prior snapshot {market_move:+.1%}; spread " +
                           (f"{market.spread:.1%}" if market.spread is not None else "unavailable"))
            if volume_change:
                reasons.append(f"market volume increased by {volume_change:.0f}")
        reasons.extend(key_events)
        return LiveAlert(
            key, state.sport, severity, score, category, f"{state.team_a} vs {state.team_b}",
            f"model {probability.pre_match_probability:.1%} -> {probability.current_probability:.1%}; market " +
            (f"{market.probability_a:.1%}" if market.probability_a is not None else "unavailable"),
            reasons, state.observed_at, f"{key}:{category}",
        )


class LiveAnalysisEngine:
    def __init__(self, store: LiveStore, market: PolymarketMonitor | None = None):
        self.store = store
        self.market = market or PolymarketMonitor()
        self.probabilities = DynamicProbabilityEngine()
        self.alerts = AlertEngine()

    def process(self, states: list[LiveState], market_events: dict[str, list[dict]],
                priors: dict[str, float], *, alert_sports: set[str] | None = None) -> list[LiveAlert]:
        emitted = []
        for state in states:
            key = match_key(state)
            previous = self.store.previous(key)
            probability = self.probabilities.update(state, priors.get(key))
            market = self.market.snapshot(state, market_events.get(state.sport, []))
            alert = self.alerts.evaluate(state, probability, market, previous)
            self.store.save(key, state, probability, market)
            alerts_enabled = alert_sports is None or state.sport in alert_sports
            if alert and alerts_enabled and not self.store.alert_recent(alert.dedupe_key, 10 * 60):
                self.store.save_alert(alert)
                emitted.append(alert)
        return emitted
