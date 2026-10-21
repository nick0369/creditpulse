"""Run `python monitor.py` after scoring; no raw user input is logged."""

import json, os, sqlite3
from pathlib import Path
import numpy as np


def summarize(path=None):
    db = Path(path or os.getenv("AUDIT_DB", "state/audit.db"))
    if not db.exists():
        return {"requests": 0, "status": "no audit data yet"}
    with sqlite3.connect(db) as con:
        rows = con.execute("SELECT latency_ms,needs_review FROM predictions").fetchall()
    if not rows:
        return {"requests": 0}
    a = np.asarray(rows)
    return {
        "requests": len(rows),
        "inference_p95_ms": float(np.quantile(a[:, 0], 0.95)),
        "review_rate": float(a[:, 1].mean()),
        "note": "Quality requires delayed ground-truth labels; drift is not proof of degradation.",
    }


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2))
