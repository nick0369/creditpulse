"""Small business-policy helpers kept separate from statistical modeling."""

import numpy as np
from statistics import NormalDist


def wilson_lower(correct, total, confidence=0.95):
    """Two-sided Wilson lower endpoint; safe for small accepted subsets."""
    if total == 0:
        return 0.0
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = correct / total
    return (
        p
        + z * z / (2 * total)
        - z * ((p * (1 - p) + z * z / (4 * total)) / total) ** 0.5
    ) / (1 + z * z / total)


def choose_threshold(probabilities, correct, required_precision=0.90, min_cases=100):
    """Maximize validation coverage subject to a conservative precision floor."""
    best = (0.0, 1.01)  # Above one means all cases require human review.
    for threshold in np.linspace(0.35, 0.99, 100):
        accepted = probabilities >= threshold
        n = int(accepted.sum())
        if (
            n >= min_cases
            and wilson_lower(int(correct[accepted].sum()), n) >= required_precision
        ):
            best = max(best, (n / len(correct), float(threshold)))
    return best[1]
