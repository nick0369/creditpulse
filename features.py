"""Point-in-time account features. PAY_0 is the latest month, not PAY_1."""

import numpy as np
import pandas as pd

FEATURES = [
    "limit_bal",
    "latest_status",
    "max_status",
    "delinquent_months",
    "avg_utilization",
    "payment_to_bill",
    "balance_change",
]


def build_features(raw):
    if not raw["ID"].is_unique:
        raise ValueError("Account IDs must be unique")
    status = raw[["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]]
    bill = raw[[f"BILL_AMT{i}" for i in range(1, 7)]]
    pay = raw[[f"PAY_AMT{i}" for i in range(1, 7)]]
    # Undefined zero-denominator ratios stay missing; imputation is fit on training only.
    out = pd.DataFrame(index=raw.index)
    out["limit_bal"] = raw.LIMIT_BAL
    out["latest_status"] = raw.PAY_0
    out["max_status"] = status.max(axis=1)
    out["delinquent_months"] = (status > 0).sum(axis=1)
    out["avg_utilization"] = bill.mean(axis=1) / raw.LIMIT_BAL.replace(0, np.nan)
    out["payment_to_bill"] = pay.sum(axis=1) / bill.sum(axis=1).replace(0, np.nan)
    out["balance_change"] = raw.BILL_AMT1 - raw.BILL_AMT2
    return out[FEATURES]
