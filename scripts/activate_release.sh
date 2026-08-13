#!/usr/bin/env bash
set -euo pipefail

archive="${1:-/tmp/prediction-agent-release.tgz}"
app_dir="/opt/prediction-agent"

if [[ ! -f "$archive" ]]; then
  echo "Release archive not found: $archive" >&2
  exit 1
fi
if [[ ! -f "$app_dir/.env" ]]; then
  echo "Production .env is missing: $app_dir/.env" >&2
  exit 1
fi

release_dir="$(mktemp -d /tmp/prediction-agent-release.XXXXXX)"
cleanup() {
  rm -rf -- "$release_dir"
  rm -f -- "$archive"
  rm -f -- /tmp/activate_prediction_agent_release.sh
}
trap cleanup EXIT

tar -xzf "$archive" -C "$release_dir"

# Preserve server-owned secrets and persistent state. The archive excludes
# `.env`, `data/`, and `reports/daily.json`; cp does not remove server-only files.
cp -a "$release_dir/." "$app_dir/"

cd "$app_dir"
docker compose build
docker compose up -d
docker compose ps
sleep 5
curl --fail --silent --show-error http://127.0.0.1:8080/health
echo
echo "PredictionAgent deployment complete."
