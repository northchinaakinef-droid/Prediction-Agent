import unittest

from prediction_agent.delivery import format_daily_report


class V25FeishuShadowAuditTests(unittest.TestCase):
    def test_daily_report_renders_shadow_telemetry_and_attribution(self):
        report = {
            "system_health": {
                "models": {"nba": "OK"},
                "data": {"cs2_recent_form": "OK"},
                "notifications": {},
                "llm": {
                    "status": "ACTIVE", "mode": "SHADOW", "provider": "fake", "model": "audit",
                    "eligible_matches": 2, "attempted": 2, "success": 1, "fallback": 1,
                    "cache_hits": 1, "evidence_gate_fallback": 0, "validation_failures": 1,
                    "request_failures": 0, "average_adjustment": .01,
                    "positive_adjustments": 1, "negative_adjustments": 0, "zero_adjustments": 0,
                },
            },
            "llm_attribution": {"overall": {
                "samples": 1, "quant_brier": .16, "final_brier": .09, "brier_delta": -.07,
                "quant_log_loss": .51, "final_log_loss": .36, "log_loss_delta": -.15,
            }},
        }

        rendered = format_daily_report(report)

        self.assertIn("【LLM审计】模式 SHADOW", rendered)
        self.assertIn("fake/audit", rendered)
        self.assertIn("【LLM结算审计】样本 1", rendered)
        self.assertNotIn("真实建议", rendered)
        self.assertNotIn("真实下注", rendered)
        self.assertNotIn("真实执行", rendered)


if __name__ == "__main__":
    unittest.main()
