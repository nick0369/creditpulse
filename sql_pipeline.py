"""Materialize and validate the SQL feature mart on the full UCI snapshot."""

import argparse, json, sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_data
from features import build_features, FEATURES


def main(source=None):
    raw = load_data(source)
    Path(".data").mkdir(exist_ok=True)
    with sqlite3.connect(".data/analytics.db") as db:
        db.executescript(Path("schema.sql").read_text())
        db.execute("DELETE FROM repayment_history")
        db.execute("DELETE FROM accounts")
        db.executemany(
            "INSERT INTO accounts VALUES (?,?)",
            raw[["ID", "LIMIT_BAL"]].itertuples(index=False, name=None),
        )
        for month in range(1, 7):
            # Original month 1 is latest; SQL month_index 6 is latest.
            status = "PAY_0" if month == 1 else f"PAY_{month}"
            rows = [
                (int(i), 7 - month, int(s), float(b), float(p))
                for i, s, b, p in raw[
                    ["ID", status, f"BILL_AMT{month}", f"PAY_AMT{month}"]
                ].itertuples(index=False, name=None)
            ]
            db.executemany("INSERT INTO repayment_history VALUES (?,?,?,?,?)", rows)
        sql = pd.read_sql_query("SELECT * FROM credit_features ORDER BY account_id", db)
        python = build_features(raw.sort_values("ID"))
        assert np.allclose(
            sql[FEATURES].to_numpy(dtype=float), python.to_numpy(), equal_nan=True
        )
        report = {
            "accounts": len(sql),
            "history_rows": db.execute(
                "SELECT COUNT(*) FROM repayment_history"
            ).fetchone()[0],
            "sql_python_parity": "passed for all seven features and all accounts",
            "engine": "SQLite",
            "postgresql_runtime_tested": False,
        }
    Path("sql_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    main(parser.parse_args().source)
