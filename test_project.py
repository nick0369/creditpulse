"""Edge cases, artifact parity and API behavior, not arbitrary accuracy asserts."""

import json, math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from inference import predict, validate_artifact
from policy import wilson_lower

ROOT = Path(__file__).parent
MODEL = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
EXAMPLES = json.loads((ROOT / "examples.json").read_text(encoding="utf-8"))


def test_artifact_schema():
    validate_artifact(MODEL)


def test_corrupt_artifact_rejected():
    with pytest.raises(ValueError):
        validate_artifact({})


def test_empty_wilson():
    assert wilson_lower(0, 0) == 0


def test_wilson_not_point_estimate():
    assert 0.9 < wilson_lower(100, 100) < 1


@pytest.mark.parametrize("case", EXAMPLES)
def test_exported_inference_parity(case):
    assert predict(case["input"], MODEL)["confidence"] == pytest.approx(
        case["expected_confidence"], abs=1e-5
    )


def test_api_contract(tmp_path, monkeypatch):
    import api

    monkeypatch.setattr(api, "DB", tmp_path / "audit.db")
    with TestClient(api.app) as client:
        assert client.get("/ready").status_code == 200
        r = client.post("/predict", json=EXAMPLES[0]["input"])
        assert r.status_code == 200 and r.json()["model_version"] == MODEL["version"]
        assert r.json()["confidence"] == pytest.approx(
            EXAMPLES[0]["expected_confidence"], abs=1e-5
        )
        assert client.post("/predict", json={}).status_code == 422
        assert (
            client.post(
                "/predict", json={**EXAMPLES[0]["input"], "unexpected": 1}
            ).status_code
            == 422
        )
        assert client.get("/assets/not-allowed").status_code == 404
    assert (tmp_path / "audit.db").exists()


def test_empty_or_inconsistent_input():
    bad = (
        {"text": " " * 40}
        if MODEL["task"] == "text"
        else {**EXAMPLES[0]["input"], "limit_bal": 0}
    )
    with pytest.raises(ValueError):
        predict(bad, MODEL)


def test_monitor_without_db(tmp_path):
    from monitor import summarize

    assert summarize(tmp_path / "missing.db")["requests"] == 0


def test_split_is_account_disjoint():
    import pandas as pd

    splits = pd.read_csv(ROOT / "split_manifest.csv")
    assert splits.account_id.is_unique
    assert splits.split.value_counts().to_dict() == {
        "train": 18000,
        "test": 6000,
        "calibration": 3000,
        "policy": 3000,
    }


def test_sql_grain_and_zero_bill():
    import sqlite3

    with sqlite3.connect(":memory:") as db:
        db.executescript((ROOT / "schema.sql").read_text())
        db.execute("INSERT INTO accounts VALUES (1,1000)")
        db.executemany(
            "INSERT INTO repayment_history VALUES (?,?,?,?,?)",
            [(1, i, 0, 0, 0) for i in range(1, 7)],
        )
        row = db.execute(
            "SELECT delinquent_months,payment_to_bill,balance_change FROM credit_features"
        ).fetchone()
        assert row == (0, None, 0)
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO repayment_history VALUES (1,6,0,0,0)")


def test_pay_zero_is_latest():
    import pandas as pd
    from features import build_features

    raw = {"ID": [1], "LIMIT_BAL": [1000]}
    raw.update(
        {
            k: [v]
            for k, v in zip(
                ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"],
                [2, 1, 0, 0, 0, 0],
            )
        }
    )
    raw.update({f"BILL_AMT{i}": [100] for i in range(1, 7)})
    raw.update({f"PAY_AMT{i}": [10] for i in range(1, 7)})
    out = build_features(pd.DataFrame(raw)).iloc[0]
    assert (
        out.latest_status == 2
        and out.delinquent_months == 2
        and out.payment_to_bill == pytest.approx(0.1)
    )
