import unittest

from prediction_agent.risk import RiskBudgetLedger, RiskConfig
from prediction_agent.sports_daily import _select_llm_execution


class V23ShadowModeTests(unittest.TestCase):
    def test_shadow_executes_quant_but_records_final_simulation(self):
        quant = {"action": "NO_BET", "stake": 0, "expected_value": .02}
        final = {"action": "BET", "stake": 5, "expected_value": .12, "risk_calculation": {"final_stake": 5}}
        row = _select_llm_execution(quant, final, shadow_mode=True, baseline=.6)
        self.assertEqual(row["action"], "NO_BET")
        self.assertEqual(row["stake"], 0)
        self.assertEqual(row["shadow_final_action"], "BET")
        self.assertEqual(row["llm_decision_mode"], "SHADOW")

    def test_active_executes_final(self):
        quant = {"action": "NO_BET", "stake": 0, "expected_value": .02}
        final = {"action": "BET", "stake": 5, "expected_value": .12}
        row = _select_llm_execution(quant, final, shadow_mode=False, baseline=.6)
        self.assertEqual((row["action"], row["stake"]), ("BET", 5))
        self.assertEqual(row["quant_action"], "NO_BET")

    def test_only_selected_candidate_is_committed(self):
        ledger = RiskBudgetLedger(RiskConfig(), 1000)
        row = _select_llm_execution({"action": "BET", "stake": 5, "stake_fraction": .005},
                                    {"action": "BET", "stake": 7, "stake_fraction": .007},
                                    shadow_mode=True, baseline=.6)
        ledger.commit("e", "g", row["stake_fraction"])
        self.assertEqual(ledger.daily_committed, .005)


if __name__ == "__main__": unittest.main()
