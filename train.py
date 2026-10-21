"""Reproduce the frozen 60/10/10/20 experiment; no test-driven model selection."""

import argparse, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from data import load_data
from features import build_features, FEATURES
from evaluate import credit_metrics, bootstrap_ap, capture
from inference import tree_score, predict


def main(source=None):
    start = time.perf_counter()
    raw = load_data(source)
    X = build_features(raw)
    y = raw.target.to_numpy()
    dev, test = train_test_split(
        np.arange(len(raw)), test_size=0.2, random_state=42, stratify=y
    )
    train, rest = train_test_split(
        dev, test_size=0.25, random_state=42, stratify=y[dev]
    )
    calibration, policy = train_test_split(
        rest, test_size=0.5, random_state=42, stratify=y[rest]
    )
    # Fit missing-ratio imputation on training accounts only.
    medians = X.iloc[train].median()
    X = X.fillna(medians)
    split = pd.DataFrame({"account_id": raw.ID, "split": "train"})
    for name, ix in [("calibration", calibration), ("policy", policy), ("test", test)]:
        split.loc[ix, "split"] = name
    split.to_csv("split_manifest.csv", index=False)
    baseline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    baseline.fit(X.iloc[train], y[train])
    challenger = XGBClassifier(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=5,
        n_jobs=2,
        random_state=42,
    )
    challenger.fit(X.iloc[train].to_numpy(), y[train])
    cal_base = baseline.predict_proba(X.iloc[calibration])[:, 1]
    cal_xgb = challenger.predict_proba(X.iloc[calibration].to_numpy())[:, 1]
    use_xgb = average_precision_score(
        y[calibration], cal_xgb
    ) > average_precision_score(y[calibration], cal_base)

    def margin(ix):
        return (
            challenger.predict(X.iloc[ix].to_numpy(), output_margin=True)
            if use_xgb
            else baseline.decision_function(X.iloc[ix])
        )

    calibrator = LogisticRegression(C=1000, max_iter=1000).fit(
        margin(calibration).reshape(-1, 1), y[calibration]
    )
    pp = calibrator.predict_proba(margin(policy).reshape(-1, 1))[:, 1]
    threshold = float(np.quantile(pp, 0.9))
    model = dict(
        task="credit",
        version="creditpulse-v1",
        features=FEATURES,
        threshold=threshold,
        medians=medians.to_dict(),
        calibration=[float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])],
    )
    if use_xgb:
        trees = [
            json.loads(t) for t in challenger.get_booster().get_dump(dump_format="json")
        ]
        row = X.iloc[train[0]].to_numpy().tolist()
        base = float(margin(train[:1])[0]) - sum(tree_score(t, row) for t in trees)
        model.update(kind="xgboost", trees=trees, base_margin=base)
    else:
        scaler = baseline[1]
        lr = baseline[2]
        model.update(
            kind="logistic",
            coef=lr.coef_[0].tolist(),
            intercept=float(lr.intercept_[0]),
            means=scaler.mean_.tolist(),
            scales=scaler.scale_.tolist(),
        )
    # Exported scoring should match the original fitted estimators.
    p = calibrator.predict_proba(margin(test).reshape(-1, 1))[:, 1]
    for i, expected in zip(test[:100], p[:100]):
        assert abs(predict(X.iloc[i].to_dict(), model)["confidence"] - expected) < 1e-5
    Path("model.json").write_text(
        json.dumps(model, separators=(",", ":")), encoding="utf-8"
    )
    metrics = credit_metrics(y[test], p, threshold)
    metrics.update(
        n_train=len(train),
        n_calibration=len(calibration),
        n_policy=len(policy),
        n_test=len(test),
        ap_95pct_bootstrap=bootstrap_ap(y[test], p),
        selected_model=model["kind"],
        threshold=threshold,
        baseline_logistic=credit_metrics(
            y[test], baseline.predict_proba(X.iloc[test])[:, 1], 0.5
        ),
        baseline_latest_status_capture=capture(
            y[test], X.iloc[test].latest_status.to_numpy()
        ),
        challenger_selection="Calibration-partition AP; policy and test untouched during selection",
        train_seconds=time.perf_counter() - start,
        python=platform.python_version(),
    )
    examples = []
    for i in test[:8]:
        payload = X.iloc[i].to_dict()
        if not all(np.isfinite(v) for v in payload.values()):
            continue
        examples.append(
            {
                "input": payload,
                "expected_confidence": predict(payload, model)["confidence"],
            }
        )
    Path("examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    main(parser.parse_args().source)
