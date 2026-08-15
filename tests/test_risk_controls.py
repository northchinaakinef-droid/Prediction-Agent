import os
from unittest import TestCase
from unittest.mock import patch

from prediction_agent.costs import estimate_cost, estimate_cost_rate
from prediction_agent.risk import RiskBudgetLedger, RiskConfig, recommend


def make_bet(ledger, event_id, group_key, **overrides):
    cfg = ledger.config
    cap = ledger.cap_for(event_id, group_key)
    params = dict(
        event_id=event_id,
        outcome="A",
        model_probability=0.8,
        decimal_odds=1 / 0.61,
        bankroll=ledger.bankroll,
        confidence=0.9,
        spread=0.01,
        available_size=1000,
        trading_enabled=True,
        max_bet_fraction=cap,
        kelly_scale=cfg.kelly_scale,
        min_edge=cfg.min_edge,
        min_confidence=cfg.min_confidence,
        max_spread=cfg.max_spread,
        min_available_size=cfg.min_available_size,
        max_depth_fraction=cfg.max_depth_fraction,
        risk_reasons=ledger.exhausted_reasons(event_id, group_key),
    )
    params.update(overrides)
    rec = recommend(**params)
    if rec.action == "BET":
        ledger.commit(event_id, group_key, rec.stake_fraction)
    return rec


class RiskConfigTests(TestCase):
    def test_from_env_wires_five_phantom_config_values(self):
        with patch.dict(os.environ, {
            "MAX_BET_FRACTION": "0.02",
            "MAX_DAILY_RISK_FRACTION": "0.05",
            "MAX_EVENT_RISK_FRACTION": "0.03",
            "MAX_DRAWDOWN_FRACTION": "0.25",
            "KELLY_FRACTION": "0.5",
        }, clear=False):
            cfg = RiskConfig.from_env()
        self.assertEqual(cfg.max_bet_fraction, 0.02)
        self.assertEqual(cfg.max_daily_risk_fraction, 0.05)
        self.assertEqual(cfg.max_event_risk_fraction, 0.03)
        self.assertEqual(cfg.max_drawdown_fraction, 0.25)
        self.assertEqual(cfg.kelly_scale, 0.5)


class CostModelTests(TestCase):
    def test_price_dependent_cost_is_shared_formula(self):
        self.assertAlmostEqual(estimate_cost_rate(0.60), 0.03 * 0.40)
        self.assertAlmostEqual(estimate_cost(0.60, 100), 100 * 0.03 * 0.40)


class RiskLedgerTests(TestCase):
    def test_daily_budget_caps_total_across_qualifying_bets(self):
        cfg = RiskConfig()
        ledger = RiskBudgetLedger(cfg, 1000)
        stakes = []
        for i in range(5):
            rec = make_bet(ledger, f"event-{i}", "nba:2026-08-15")
            stakes.append(rec.stake_fraction)
        self.assertLessEqual(sum(stakes), cfg.max_daily_risk_fraction + 1e-9)

    def test_event_budget_caps_correlated_markets_on_same_event(self):
        cfg = RiskConfig()
        ledger = RiskBudgetLedger(cfg, 1000)
        first = make_bet(ledger, "event-1", "nba:2026-08-15")
        second = make_bet(ledger, "event-1", "nba:2026-08-15")
        self.assertEqual(first.action, "BET")
        self.assertEqual(second.action, "BET")
        self.assertLessEqual(first.stake_fraction + second.stake_fraction, cfg.max_event_risk_fraction + 1e-9)

    def test_correlation_group_budget_caps_total(self):
        cfg = RiskConfig(max_correlation_group_fraction=0.02)
        ledger = RiskBudgetLedger(cfg, 1000)
        first = make_bet(ledger, "event-1", "nba:2026-08-15")
        second = make_bet(ledger, "event-2", "nba:2026-08-15")
        third = make_bet(ledger, "event-3", "nba:2026-08-15")
        total = first.stake_fraction + second.stake_fraction + third.stake_fraction
        self.assertLessEqual(total, cfg.max_correlation_group_fraction + 1e-9)

    def test_drawdown_breaker_forces_no_bet(self):
        cfg = RiskConfig()
        ledger = RiskBudgetLedger(cfg, 1000)
        ledger.breaker_reason = "account drawdown circuit breaker triggered"
        rec = make_bet(ledger, "event-1", "nba:2026-08-15")
        self.assertEqual(rec.action, "NO_BET")
        self.assertEqual(rec.stake, 0)
        self.assertIn("account drawdown circuit breaker triggered", rec.reasons)


class DepthCapacityTests(TestCase):
    def test_stake_is_capped_to_visible_depth_fraction(self):
        rec = recommend(
            event_id="event-1", outcome="A", model_probability=0.8,
            decimal_odds=1 / 0.61, bankroll=1000, confidence=0.9,
            spread=0.01, available_size=1000, trading_enabled=True,
            max_depth_fraction=0.001,
        )
        self.assertEqual(rec.action, "BET")
        self.assertLessEqual(rec.stake_fraction, 0.001 + 1e-9)
        self.assertIn("stake capped to 0% of available depth", rec.reasons)
