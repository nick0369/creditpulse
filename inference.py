"""Portable JSON inference, shared by the API and offline contract tests."""

import math


def sigmoid(x):
    x = max(-700, min(700, float(x)))
    return 1 / (1 + math.exp(-x))


def validate_artifact(model):
    if (
        not {"task", "version", "features", "kind", "threshold", "calibration"}
        <= model.keys()
    ):
        raise ValueError("Invalid model artifact")
    if model["task"] != "credit" or model["kind"] not in {"xgboost", "logistic"}:
        raise ValueError("Unsupported credit model")


def tree_score(node, values):
    if "leaf" in node:
        return node["leaf"]
    index = int(node["split"].removeprefix("f"))
    value = values[index]
    child = (
        node["missing"]
        if value is None
        else (node["yes"] if value < node["split_condition"] else node["no"])
    )
    return tree_score(next(n for n in node["children"] if n["nodeid"] == child), values)


def predict(payload, model):
    values = [
        model["medians"][n] if payload[n] is None else payload[n]
        for n in model["features"]
    ]
    if any(v is None or not math.isfinite(float(v)) for v in values):
        raise ValueError("Provide finite numeric features")
    if payload["limit_bal"] <= 0 or payload["latest_status"] > payload["max_status"]:
        raise ValueError("Invalid balance limit or inconsistent repayment statuses")
    if model["kind"] == "xgboost":
        margin = model["base_margin"] + sum(
            tree_score(t, values) for t in model["trees"]
        )
    else:
        margin = model["intercept"] + sum(
            c * (v - m) / s
            for c, v, m, s in zip(
                model["coef"], values, model["means"], model["scales"]
            )
        )
    a, b = model["calibration"]
    p = sigmoid(a * margin + b)
    return {
        "default_probability": p,
        "confidence": p,
        "needs_review": p >= model["threshold"],
        "decision": (
            "Human review candidate"
            if p >= model["threshold"]
            else "Standard monitoring"
        ),
    }
