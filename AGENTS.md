# PredictionAgent contributor guide

## Safety boundaries

- This project provides research probabilities and paper-betting evaluation. Do not add automatic real-money trading or wallet handling.
- Keep `real_money_approved` false unless a separate, genuinely unseen chronological lockbox passes the documented acceptance criteria.
- Never commit `.env`, Feishu webhooks, SSH credentials, API secrets, wallet material, `data/`, or the live paper-trading database.
- CBA is paused. Production analysis currently covers NBA, LoL, and CS2.
- Daily inference must load frozen artifacts; do not retrain models during the 30-minute scan or 06:30 report job.
- Trade execution is isolated by design: this repository must never import a wallet/exchange SDK or read a private key. Any future execution service must be separate and consume only this repo's `NO_BET`/`BET` JSON output.

## Validation

Run the full suite before committing:

```bash
python -m unittest discover -s tests -v
```

For changes to probability models, preserve chronological train/validation/test boundaries and compute every walk-forward probability before applying that event's result.

## Delivery

- Work on a branch and open a pull request. Do not push feature changes directly to `main`.
- A merge to `main` may deploy to the production Tencent Cloud host through GitHub Actions.
- Do not modify or overwrite the production `.env`, `data/`, or persistent reports during deployment.
