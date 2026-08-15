#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "Missing .env. Run: cp .env.example .env && chmod 600 .env"
  echo "Then edit only the Feishu webhook values; never paste them into chat."
  exit 1
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io docker-compose-v2 ca-certificates
sudo systemctl enable --now docker

mkdir -p reports data/daily
sudo chown -R 10001:10001 reports data
sudo chown 10001:10001 .env
chmod 600 .env

sudo docker compose build
sudo docker compose up -d
sudo docker compose ps
sleep 5
curl --fail --silent --show-error http://127.0.0.1:8080/health
echo
echo "PredictionAgent is running."
