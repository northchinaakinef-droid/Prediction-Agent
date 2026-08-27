# V2.5 LLM Preflight Audit & Shadow Restoration

## Result

`LLM_PREFLIGHT_READY`

This result means the code path is ready for a later, separately authorized credential and Shadow activation step. No LLM credential was read, written, or configured; no live LLM request was made; and this change is not authorized for production deployment in this phase.

## Audited baseline

- GitHub `main` SHA: `a291ce7cf172cf8569fbda4bbcb402fa42b8d6fd`
- Baseline source manifest: 83 tracked `src/`, `scripts/`, and `tests/` files; aggregate SHA-256 `f1640cbece48ada05def0cc6a4690e531d64839fdae4a3d3475b23f6f19da241`
- Actual baseline tests: unittest 238 passed; pytest 253 passed
- Restored source manifest: 90 files; aggregate SHA-256 `7d616e7c8ef3e2dc75f5bc5501ef89523e812b1aaa0701420227f7ccb2a619ac`

The historical report did not describe the checked-out GitHub `main`: the reported Shadow implementation and tests were absent from tracked `main`, while similar files existed only in an unrelated, untracked extracted workspace. Git history available in this checkout does not prove whether they were never merged or were later lost, so no stronger causal claim is made.

## LLM_PRE_FLIGHT_MATRIX

| Capability | Current main before restoration | Expected | Status after restoration |
| --- | --- | --- | --- |
| OpenAI client | Present | OpenAI-compatible client | PASS |
| DeepSeek client | Present through compatible config | Explicit provider identity | PASS |
| Anthropic client | Present | Anthropic client | PASS |
| Evidence minimum gate | Present | Existing sport requirements unchanged | PASS |
| ±5% adjustment cap | Present | Unchanged | PASS |
| Semantic Evidence hash | Missing | Ignore scan-only time drift; include semantic state | PASS |
| Provider/model isolated cache | Missing | Provider and model in identity | PASS |
| Quant baseline in cache identity | Missing | Baseline in identity | PASS |
| Final Probability recompute EV | Missing | Recompute from Final probability | PASS |
| Final Probability recompute action | Missing | Recompute from Final probability | PASS |
| Final Probability recompute stake | Missing | Recompute from Final probability | PASS |
| Shadow Quant Candidate | Missing | Independently calculated formal candidate | PASS |
| Shadow Final Candidate | Missing | Independently calculated audit candidate | PASS |
| Shadow single ledger commit | Missing | Commit selected candidate exactly once | PASS |
| Quant vs Final attribution | Missing | Canonical-sample comparison | PASS |
| Brier comparison | Missing | Quant, Final, and Final-minus-Quant delta | PASS |
| LogLoss comparison | Missing | Quant, Final, and Final-minus-Quant delta | PASS |
| LLM telemetry | Partial | Eligibility, attempts, cache, fallback and adjustment detail | PASS |
| LLM health ACTIVE/DEGRADED/ERROR | Partial | Include NOT_CONFIGURED and QUANT_FALLBACK | PASS |
| Feishu Shadow audit | Missing | Shadow telemetry and settled attribution, render only | PASS |

## Restored behavior

The post-AI helper now recalculates edge, EV, expected ROI, Kelly-derived stake, action, Data Quality gating, and Risk gating from one probability input. Quant and Final candidates therefore cannot display a new Final probability while retaining an old Quant action or stake.

With `LLM_SHADOW_MODE=true` (the safe default), the formal action, stake, virtual bet, and sole risk-ledger commit use the Quant candidate. The Final candidate is retained only for shadow telemetry and attribution. The active simulation path selects Final and also commits only once; it was tested without enabling production LLM activity.

Cache identity now contains match ID, semantic evidence fingerprint, Quant baseline probability, prompt version, provider, model, and analyst/schema version. `observed_at` and continuous freshness seconds do not churn the cache; material evidence changes and freshness-state transitions do.

LLM request, validation, or evidence-reference failures fall back to the Quant probability without crashing. Evidence-gate ineligibility is recorded separately and is not counted as an API request failure.

Canonical settlement attribution stores Quant and Final Brier and LogLoss, action, EV, stake, adjustment, provider, and model. Its deltas are `Final - Quant`, so negative scoring deltas mean Final performed better; duplicate raw scans sharing a canonical sample key are not counted twice.

Feishu rendering now exposes Shadow mode telemetry and canonical Quant-vs-Final settlement metrics without sending a test message. The V2.4 wording protections remain intact: it does not introduce `真实建议`, `真实下注`, or `真实执行`.

## Safety and invariant review

- Evidence minimum requirements and roster freshness blocking were not lowered.
- The LLM adjustment cap remains ±5 percentage points.
- Existing Data Quality, Risk, Kelly/stake, live-monitor, CS2 recent-form, roster-provider, provider-promotion, and model-training logic was not changed.
- Shadow provider data is not promoted into formal Evidence.
- Real-money trading remains disabled; no wallet/exchange SDK or private-key path was added.
- No `.env`, credential, API key, token, webhook, database, data directory, or production artifact is included.

## Verification

- `python -m unittest discover -s tests -q`: **253 passed**
- `python -m pytest -q`: **268 passed**
- Fake LLM cases cover +5%, -5%, request fallback, invalid evidence reference, semantic cache hits/misses, provider/model/baseline isolation, Shadow/active selection, attribution deduplication, and Feishu audit rendering.
- `python scripts/check_llm_preflight.py`: all nine checks PASS; final output `LLM_PREFLIGHT_READY`.

## Readiness answer

The code is ready for a later controlled credential configuration and Shadow-only activation. It is **not** permission to configure a key, call a live provider, switch `LLM_SHADOW_MODE=false`, merge/deploy this PR, or alter any trading boundary. Those actions require the user's next explicit approval.
