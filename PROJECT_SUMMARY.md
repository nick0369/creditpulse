# CreditPulse — project summary

Credit operations have finite review capacity. This model ranks existing accounts by next-month default risk. A human decides any intervention; prediction does not prove that a default can be prevented.

18,000 training accounts; 3,000 for model selection/calibration; 3,000 for policy selection; 6,000 held-out test accounts. Imputation is fit on training only. Demographic fields and account ID are excluded. Logistic regression is the baseline; XGBoost is the selected challenger.

## Measured results

| Metric | Held-out result |
|---|---|
| ROC-AUC | 0.777 |
| Average precision | 0.561 |
| Capture at top 10% | 31.7% |

Test size: **6,000**. See metrics.json for baselines, partition sizes and policy thresholds. Results were computed locally on Python 3.14.7. Source: [official download](https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip); checksum in manifest.json. No realized financial ROI is claimed.


## Deliverables

Commented Python, trained JSON model, measured holdout evaluation, interactive browser demo, tested FastAPI service, SQL, unit/integration checks, data provenance, Docker configuration, monitoring summary and honest operational limitations.
