# Automatic Production Deployment

```text
AUTOMATIC_DEPLOYMENT=PASS
workflow=.github/workflows/deploy-production.yml
trigger=push to main, workflow_dispatch
remote=/opt/prediction-agent
service=prediction-agent
```

## Pipeline

```text
checkout
→ Python 3.12
→ install package + pytest
→ build protected release archive
→ SCP through production SSH key
→ server backup
→ Docker Compose build/up
→ health + scheduler + SQLite + Provider freshness verification
```

The production concurrency group is `prediction-agent-production` with cancellation disabled, so only one deployment can mutate production at a time.

## Required GitHub secrets

- `PROD_HOST`: `43.128.1.180`
- `PROD_USER`: `ubuntu`
- `PROD_SSH_KEY`: the private deployment key stored only in GitHub Actions secrets

The private key and server `.env` must never be committed.

## Preserved state and safety

The archive excludes `.env`, `data/`, and `reports/daily.json`. The activation script explicitly rewrites `REAL_TRADING_DISABLED=true` without changing other server-owned secrets. It does not enable LLM or real trading.

Each activation creates `/home/ubuntu/prediction-agent-backups/<UTC timestamp>/` containing:

- `.env`
- previous code archive
- SQLite-consistent online backups
- previous container inspection metadata

## Verification and rollback

Deployment fails if tests, upload, Docker build/up, HTTP health, container health, prematch scanner, SQLite quick-check/write, or Provider freshness validation fails.

Before building, the previous image ID and Compose image reference are retained. A deployment failure restores the previous code archive, retags the previous image, and recreates the old Compose container. Backups are not deleted by successful deployments.

The workflow does not send Feishu test messages. Normal production Outbox behavior remains unchanged.

## Runtime validation

The first restored main-branch deployment completed through GitHub Actions run `32646475487`:

```text
release_git_sha=56534a0b993b588dc918c912722472fbf8b3748f
previous_image=sha256:d02eb020e30c58b07d0429a40cc835498996a38bb2ee1ed10c66a6aa006d9922
deployed_image=sha256:a8a93dd12536ab475dfe409f2720834cdb975afb7b6989b88c880aaa663f3448
container_health=healthy
restart_policy=unless-stopped
prematch_interval_seconds=300
prematch_error=None
sqlite_quick_check=ok
provider_health_timestamp=2026-08-23T14:48:08.451676+00:00
automatic_backup=/home/ubuntu/prediction-agent-backups/20260823T144715Z
```

The cloud server remains the only Shadow runtime. No deployment test message is sent to Feishu.
