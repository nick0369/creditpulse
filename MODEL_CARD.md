# CreditPulse model card

## Intended use
Predict next-month credit default and prioritize a limited human review queue. Decision support and educational benchmarking with historical public data.

## Data and training
18,000 training accounts; 3,000 for model selection/calibration; 3,000 for policy selection; 6,000 held-out test accounts. Imputation is fit on training only. Demographic fields and account ID are excluded. Logistic regression is the baseline; XGBoost is the selected challenger.

Artifact: model.json. Scope and source SHA-256: manifest.json. Reproducible partitions: split_manifest.csv. Weights were fitted; no results are simulated. Dataset itself is not redistributed.

## Evaluation
| ROC-AUC | 0.777 |
| Average precision | 0.561 |
| Capture at top 10% | 31.7% |

See metrics.json for full details. Credit intervals use account-level bootstrap; text accepted-precision uncertainty uses Wilson bounds, which do not fully model correlated templates. No fairness certification or operational approval is claimed.

## Conclusion
XGBoost improved test average precision from 0.503 to 0.561. However, capture at a 10% review budget was 31.7%, versus 31.3% for a latest-status rule. The small capture difference is not evidence of a material business improvement. The strongest conclusion is improved ranking quality across thresholds, not proven loss reduction.

## Risks and limitations
Historical 2005 Taiwanese snapshot; no genuine out-of-time account cohort. Intervention effectiveness and avoidable loss are unobserved. Calibration and subgroup behavior may change in new populations.

## Monitoring and rollback
Immediate: input validation, inference failures, audit-store failures. Monitor latency and review rate with monitor.py. Future delayed labels are required to measure real model drift. Restore a reviewed previous artifact and restart the API to roll back; never silently retrain or change review thresholds in production.

## Release checklist
Run pytest, model-export parity, JavaScript parity, one synthetic API request, /ready and dashboard smoke checks. Maintain the data checksum, dependency pins, threshold and metrics with the model release. The public demo executes inference in the visitor browser and does not retain submitted text.
