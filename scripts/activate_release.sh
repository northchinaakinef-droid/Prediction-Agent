#!/usr/bin/env bash
set -euo pipefail

archive="${1:-/tmp/prediction-agent-release.tgz}"
app_dir="/opt/prediction-agent"
backup_root="/home/ubuntu/prediction-agent-backups"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_dir="$backup_root/$stamp"
release_dir=""
old_image_id=""
old_image_ref=""

cleanup() {
  [[ -z "$release_dir" ]] || rm -rf -- "$release_dir"
  rm -f -- "$archive" /tmp/activate_prediction_agent_release.sh
}
trap cleanup EXIT

fail() {
  echo "DEPLOYMENT_FAILED: $1" >&2
  if [[ -n "$old_image_id" && -n "$old_image_ref" ]]; then
    echo "Rolling back to image $old_image_id" >&2
    tar -xzf "$backup_dir/code.tgz" -C "$app_dir"
    docker image tag "$old_image_id" "$old_image_ref"
    cd "$app_dir"
    docker compose up -d --no-build --force-recreate
    docker compose ps
  fi
  exit 1
}

[[ -f "$archive" ]] || fail "release archive not found: $archive"
[[ -f "$app_dir/.env" ]] || fail "production .env missing: $app_dir/.env"

if tar -tzf "$archive" | grep -Eq '(^|/)\.env$|(^|/)data/|(^|/)reports/daily\.json$'; then
  fail "release archive contains protected runtime state"
fi

install -d -m 700 -o ubuntu -g ubuntu "$backup_dir"
cp -a "$app_dir/.env" "$backup_dir/.env"
chmod 600 "$backup_dir/.env"
tar -C "$app_dir" --exclude=.env --exclude=data --exclude=reports -czf "$backup_dir/code.tgz" .
if [[ -f "$app_dir/compose.yaml" ]]; then
  cd "$app_dir"
  current_container="$(docker compose ps -q prediction-agent 2>/dev/null || true)"
  if [[ -n "$current_container" ]]; then
    old_image_id="$(docker inspect "$current_container" --format '{{.Image}}')"
    old_image_ref="$(docker inspect "$current_container" --format '{{.Config.Image}}')"
    docker inspect "$current_container" > "$backup_dir/container-inspect.json"
  fi
fi

python3 - "$app_dir/data" "$backup_dir/sqlite" <<'PY'
import sqlite3, sys
from pathlib import Path
source, target_root = Path(sys.argv[1]), Path(sys.argv[2])
for path in source.rglob("*.db"):
    target = target_root / path.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
PY

# Safety is explicit on every release and server-owned secrets remain untouched.
python3 - "$app_dir/.env" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
result, found = [], False
for line in lines:
    if line.startswith("REAL_TRADING_DISABLED="):
        result.append("REAL_TRADING_DISABLED=true"); found = True
    else:
        result.append(line)
if not found: result.append("REAL_TRADING_DISABLED=true")
path.write_text("\n".join(result) + "\n", encoding="utf-8")
PY
chown 10001:10001 "$app_dir/.env"
chmod 600 "$app_dir/.env"

release_dir="$(mktemp -d /tmp/prediction-agent-release.XXXXXX)"
tar -xzf "$archive" -C "$release_dir"
cp -a "$release_dir/." "$app_dir/"

cd "$app_dir"
docker compose build || fail "docker compose build failed"
docker compose up -d || fail "docker compose up failed"

ready=false
for _ in {1..36}; do
  if curl --fail --silent http://127.0.0.1:8080/health >/tmp/prediction-agent-health.json 2>/dev/null && \
     docker compose exec -T prediction-agent python - <<'PY' >/dev/null 2>&1
import os, sqlite3, sys
from datetime import datetime, timezone
if os.getenv('REAL_TRADING_DISABLED', '').lower() != 'true': sys.exit(1)
with sqlite3.connect('/app/data/daily/paper.db') as db:
    if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok': sys.exit(1)
    latest = db.execute('SELECT MAX(observed_at) FROM provider_health_observations').fetchone()[0]
    if not latest: sys.exit(1)
    age = datetime.now(timezone.utc) - datetime.fromisoformat(latest.replace('Z', '+00:00'))
    if age.total_seconds() > 900: sys.exit(1)
with sqlite3.connect('/app/data/daily/notifications.db') as db:
    db.execute('CREATE TABLE IF NOT EXISTS deployment_health_checks(checked_at TEXT PRIMARY KEY)')
    db.execute('INSERT OR REPLACE INTO deployment_health_checks VALUES(?)', (datetime.now(timezone.utc).isoformat(),))
PY
  then
    scanner_ok="$(python3 - <<'PY'
import json
row=json.load(open('/tmp/prediction-agent-health.json'))
s=row.get('prematch_scanner') or {}
print('yes' if s.get('last_ok') and s.get('error') is None and s.get('interval_seconds') == 300 else 'no')
PY
)"
    if [[ "$scanner_ok" == yes ]]; then ready=true; break; fi
  fi
  sleep 5
done

[[ "$ready" == true ]] || fail "health, prematch scanner, SQLite, or provider freshness verification failed"
container="$(docker compose ps -q prediction-agent)"
health_status="$(docker inspect "$container" --format '{{.State.Health.Status}}')"
[[ "$health_status" == healthy ]] || fail "container health is $health_status"
docker compose ps
echo "BACKUP_DIR=$backup_dir"
echo "RELEASE_GIT_SHA=$(cat RELEASE_GIT_SHA 2>/dev/null || echo unknown)"
echo "CONTAINER_HEALTH=$health_status"
echo "PredictionAgent automatic deployment complete."
