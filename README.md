# CreditPulse

Predict next-month credit default and prioritize a limited human review queue.

**Status:** trained and evaluated independent portfolio implementation. The public browser demo scores with the checked-in trained model. FastAPI runs locally or in the provided container configuration. This is not a regulated production deployment.

[Live dashboard](https://nikhil-chandekar-data-portfolio.goofyvole8.chatgpt.site/projects/creditpulse.html) · [Model card](MODEL_CARD.md) · [Metrics](metrics.json)

## Project summary

Credit operations have finite review capacity. This model ranks existing accounts by next-month default risk. A human decides any intervention; prediction does not prove that a default can be prevented.

18,000 training accounts; 3,000 for model selection/calibration; 3,000 for policy selection; 6,000 held-out test accounts. Imputation is fit on training only. Demographic fields and account ID are excluded. Logistic regression is the baseline; XGBoost is the selected challenger.

## Measured results

| Metric | Held-out result |
|---|---|
| ROC-AUC | 0.777 |
| Average precision | 0.561 |
| Capture at top 10% | 31.7% |

Test size: **6,000**. See metrics.json for baselines, partition sizes and policy thresholds. Results were computed locally on Python 3.14.7. Source: [official download](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip); checksum in manifest.json. No realized financial ROI is claimed.

## Run the trained project

Python 3.14 is the tested version. From a fresh clone:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/ for the interactive dashboard and http://127.0.0.1:8000/docs for validated API requests. Use the example inputs in examples.json. No API key is needed.

```bash
python train.py
python monitor.py
# Optional, requires Docker installed:
docker compose up --build
```

Training downloads the official dataset into ignored .data/. It can take time and requires an internet connection. `python train.py --source /path/to/source.zip` reuses an existing official archive. Reproduction records the new checksum in manifest.json; compare it to the committed checksum before comparing results. Raw data and private text are not checked in. Use the committed artifact to run predictions immediately without downloading training data.

## Code map

```text
data.py             Download, schema checks and provenance
features.py         Documented feature/text transformations
train.py            Deterministic split, baselines, fitting and export
evaluate.py         Statistical and business-policy metrics
policy.py           Review threshold and uncertainty helpers
inference.py        Safe JSON model loading / portable prediction
api.py              Pydantic validation, endpoints and SQLite audit
monitor.py          Request counts, review rate and latency summary
test_project.py     Edge cases, API contracts and model-export parity
schema.sql          Advanced SQL reference and data mart
model.json          Trained portable model, not a placeholder
metrics.json        Actual frozen-test results
examples.json       Scoring/parity examples
split_manifest.csv  Stable source IDs / partitions (no narrative text)
manifest.json       Exact source URL, checksum and scope
dashboard.html/js   Public interactive scoring and metrics
MODEL_CARD.md       Intended use, limitations and release/runbook
requirements.txt    Exact tested dependency versions
Dockerfile          Non-root API container configuration
compose.yaml        API health check and persistent audit volume
```

The flat module layout is deliberate: this is a small, readable application. Each file has a clear responsibility without framework boilerplate. SQLite stores operational audit data locally. The SQL analytical mart is separate from inference; see SQL_NOTES.md for database setup and validation scope.

## Key insights and conclusion

XGBoost improved test average precision from 0.503 to 0.561. However, capture at a 10% review budget was 31.7%, versus 31.3% for a latest-status rule. The small capture difference is not evidence of a material business improvement. The strongest conclusion is improved ranking quality across thresholds, not proven loss reduction.

## Engineering and operations

- The trained artifact is JSON, avoiding unsafe pickle deserialization. Python and JavaScript predictions are parity checked.
- Pydantic checks the request contract; invalid inputs return 422. Missing/corrupt model artifacts prevent startup.
- Requests are audited with random ID, version, score, review flag and timing. Raw narratives and account histories are not logged.
- Run `python monitor.py` to summarize audit counts, review share and inference p95; this is not a full observability platform.
- Version model, policy, schema and service together. To roll back, restore the previous reviewed Git commit/artifact, restart, and run tests plus a synthetic request.
- In a real deployment, add TLS, authentication where required, request limits, alerting and delayed-label performance checks. Do not expose sensitive user data through the public demonstration.

## Limitations

Historical 2005 Taiwanese snapshot; no genuine out-of-time account cohort. Intervention effectiveness and avoidable loss are unobserved. Calibration and subgroup behavior may change in new populations.

Docker configuration is provided; a container build is not claimed tested unless Docker verification is recorded. The measured FastAPI benchmark uses a local test client and is not a cloud throughput/SLA claim. See VALIDATION.md for exact checks and environment.

## Interview talking points

1. Why this business problem needs a review policy as well as a classifier.
2. How the split prevents target or future-data leakage.
3. Why the baseline matters and where the selected model still fails.
4. How offline predictions match deployed inference.
5. What must be measured before claiming business ROI.

## License and attribution

MIT applies to original code. Dataset rights remain with the original publisher. See the official source and manifest for provenance. This is an independent educational portfolio project, not employment or client work.
