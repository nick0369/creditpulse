"""Metrics with explicit holdouts, baselines and uncertainty."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    f1_score,
    confusion_matrix,
    classification_report,
)
from policy import wilson_lower


def capture(y, p, fraction=0.1):
    y, p = np.asarray(y), np.asarray(p)
    # Stable ordering makes ties deterministic for a fixed split manifest.
    selected = np.argsort(-p, kind="stable")[: max(1, int(np.ceil(len(y) * fraction)))]
    return float(y[selected].sum() / max(1, y.sum()))


def credit_metrics(y, p, threshold):
    y = np.asarray(y)
    return dict(
        roc_auc=float(roc_auc_score(y, p)),
        average_precision=float(average_precision_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        f1=float(f1_score(y, np.asarray(p) >= threshold)),
        prevalence=float(y.mean()),
        capture_at_10pct=capture(y, p),
        confusion_matrix=confusion_matrix(y, np.asarray(p) >= threshold).tolist(),
    )


def text_metrics(y, probabilities, classes, threshold):
    y = np.asarray(y)
    guess = np.asarray(classes)[probabilities.argmax(axis=1)]
    accepted = probabilities.max(axis=1) >= threshold
    n = int(accepted.sum())
    correct = int((guess[accepted] == y[accepted]).sum())
    return dict(
        macro_f1=float(f1_score(y, guess, average="macro", zero_division=0)),
        coverage=float(accepted.mean()),
        accepted_cases=n,
        accepted_precision=float(correct / n) if n else None,
        accepted_precision_wilson_lower=wilson_lower(correct, n),
        unknown_label_rate=float((~np.isin(y, classes)).mean()),
        per_class=classification_report(y, guess, zero_division=0, output_dict=True),
        confusion_matrix=confusion_matrix(y, guess, labels=classes).tolist(),
    )


def bootstrap_ap(y, p, iterations=300):
    rng = np.random.default_rng(42)
    y = np.asarray(y)
    p = np.asarray(p)
    values = []
    for _ in range(iterations):
        ix = rng.integers(0, len(y), len(y))
        if np.unique(y[ix]).size == 2:
            values.append(average_precision_score(y[ix], p[ix]))
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]
