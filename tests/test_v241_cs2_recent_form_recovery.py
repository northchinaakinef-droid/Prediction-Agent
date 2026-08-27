from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from prediction_agent.recent_form_refresh import Cs2MatchResult, refresh_cs2_recent_form
from prediction_agent.context import recent_form_artifact_health


NOW = datetime(2026, 8, 25, tzinfo=timezone.utc)


class Provider:
    name = "fake"
    def __init__(self, rows=None, error=None): self.rows, self.error = rows or [], error
    def fetch_finished(self, _start, _end):
        if self.error: raise self.error
        return list(self.rows)


def matches():
    return [Cs2MatchResult("1", NOW, "NAVI", "G2", "NAVI"),
            Cs2MatchResult("2", NOW, "G2", "Vitality", "Vitality")]


class V23RecentFormRefreshTests(unittest.TestCase):
    def test_refresh_success_writes_health_metadata_and_real_records(self):
        with TemporaryDirectory() as tmp:
            artifact, status = Path(tmp) / "recent.json", Path(tmp) / "status.json"
            artifact.write_text(json.dumps({"lol": {"T1": {"wins": 1}}}), encoding="utf-8")
            result = refresh_cs2_recent_form(provider=Provider(matches()), artifact_path=artifact,
                                             status_path=status, now=NOW, expected_team_count=4)
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "OK")
            self.assertEqual(payload["cs2"]["Natus Vincere"]["last_n"], 1)
            self.assertEqual(payload["_metadata"]["cs2"]["record_count"], 2)
            self.assertEqual(payload["_metadata"]["cs2"]["coverage"], .75)
            if os.name != "nt":
                self.assertEqual(artifact.stat().st_mode & 0o777, 0o644)

    def test_provider_failure_and_stale_fallback_preserve_artifact(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"
            original = json.dumps({"cs2": {"Old": {"wins": 1, "losses": 0, "last_n": 1}}})
            artifact.write_text(original, encoding="utf-8")
            result = refresh_cs2_recent_form(provider=Provider(error=RuntimeError("down")),
                                             artifact_path=artifact, now=NOW)
            self.assertEqual(result["status"], "ERROR")
            self.assertTrue(result["fallback_artifact_usable"])
            self.assertEqual(artifact.read_text(encoding="utf-8"), original)

    def test_http_error_preserves_artifact(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"
            original = '{"cs2":{"Old":{"last_n":1}}}'
            artifact.write_text(original, encoding="utf-8")
            error = HTTPError("https://api.bo3.gg", 403, "Forbidden", {}, None)
            result = refresh_cs2_recent_form(provider=Provider(error=error), artifact_path=artifact, now=NOW)
            self.assertEqual(result["error_category"], "HTTP_403")
            self.assertEqual(artifact.read_text(encoding="utf-8"), original)

    def test_malformed_json_error_preserves_artifact(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"; artifact.write_text('{"cs2":{"Old":{}}}', encoding="utf-8")
            original = artifact.read_bytes()
            error = json.JSONDecodeError("invalid", "x", 0)
            result = refresh_cs2_recent_form(provider=Provider(error=error), artifact_path=artifact, now=NOW)
            self.assertEqual(result["error_category"], "JSON_DECODE")
            self.assertEqual(artifact.read_bytes(), original)

    def test_schema_mismatch_rows_preserve_artifact(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"; artifact.write_text('{"cs2":{"Old":{}}}', encoding="utf-8")
            original = artifact.read_bytes()
            invalid = Cs2MatchResult("1", NOW, "A", "B", "neither")
            result = refresh_cs2_recent_form(provider=Provider([invalid]), artifact_path=artifact, now=NOW)
            self.assertEqual(result["status"], "EMPTY")
            self.assertEqual(artifact.read_bytes(), original)

    def test_corrupt_artifact_is_replaced_only_after_success(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"
            artifact.write_text("{bad", encoding="utf-8")
            failed = refresh_cs2_recent_form(provider=Provider(error=RuntimeError("down")),
                                             artifact_path=artifact, now=NOW)
            self.assertFalse(failed["fallback_artifact_usable"])
            self.assertEqual(artifact.read_text(encoding="utf-8"), "{bad")
            succeeded = refresh_cs2_recent_form(provider=Provider(matches()), artifact_path=artifact, now=NOW)
            self.assertEqual(succeeded["status"], "OK")
            self.assertIn("cs2", json.loads(artifact.read_text(encoding="utf-8")))

    def test_empty_dataset_does_not_replace(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"; artifact.write_text('{"cs2":{"Old":{}}}', encoding="utf-8")
            result = refresh_cs2_recent_form(provider=Provider([]), artifact_path=artifact, now=NOW)
            self.assertEqual(result["status"], "EMPTY")
            self.assertIn("Old", artifact.read_text(encoding="utf-8"))

    def test_zero_teams_does_not_replace(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"; artifact.write_text('{"cs2":{"Old":{}}}', encoding="utf-8")
            original = artifact.read_bytes()
            invalid = Cs2MatchResult("1", NOW, "Same", "Same", "Same")
            refresh_cs2_recent_form(provider=Provider([invalid]), artifact_path=artifact, now=NOW)
            self.assertEqual(artifact.read_bytes(), original)

    def test_partial_temporary_file_is_never_promoted(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"; artifact.write_text('{"cs2":{"Old":{}}}', encoding="utf-8")
            original = artifact.read_bytes()
            with patch("prediction_agent.recent_form_refresh.json.dump", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    from prediction_agent.recent_form_refresh import _atomic_json_write
                    _atomic_json_write(artifact, {"cs2": {"New": {}}})
            self.assertEqual(artifact.read_bytes(), original)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_atomic_replace_is_used(self):
        with TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "recent.json"
            with patch("prediction_agent.recent_form_refresh.os.replace", wraps=os.replace) as replace:
                result = refresh_cs2_recent_form(provider=Provider(matches()), artifact_path=artifact, now=NOW)
            self.assertEqual(result["status"], "OK")
            self.assertGreaterEqual(replace.call_count, 2)  # artifact plus refresh status
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    def test_stale_fallback_health_is_not_green(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "recent_form.json").write_text(json.dumps({
                "cs2": {"Old": {"wins": 1, "losses": 0, "last_n": 1}},
                "_metadata": {"generated_at": "2020-01-01T00:00:00+00:00", "cs2": {"record_count": 1}},
            }), encoding="utf-8")
            (root / "recent_form_refresh_status.json").write_text(
                json.dumps({"status": "ERROR", "error": "provider down"}), encoding="utf-8")
            with patch("prediction_agent.context.ARTIFACT_DIR", root):
                health = recent_form_artifact_health()
            self.assertEqual(health["artifact_status"], "STALE")
            self.assertEqual(health["refresh_status"], "ERROR")

    def test_deployment_preserves_runtime_recent_form_artifacts(self):
        workflow = Path(".github/workflows/deploy-production.yml").read_text(encoding="utf-8")
        activation = Path("scripts/activate_release.sh").read_text(encoding="utf-8")
        self.assertIn("--exclude='artifacts/recent_form.json'", workflow)
        self.assertIn("recent_form(_refresh_status)?", activation)
        self.assertIn("runtime-artifacts", activation)

    def test_systemd_and_container_use_the_same_host_artifact(self):
        refresh_script = Path("scripts/refresh_cs2_recent_form.py").read_text(encoding="utf-8")
        compose = Path("compose.yaml").read_text(encoding="utf-8")
        self.assertIn('ROOT / "artifacts" / "recent_form.json"', refresh_script)
        self.assertIn("./artifacts:/app/artifacts:ro", compose)


if __name__ == "__main__": unittest.main()
