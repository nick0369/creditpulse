# SQL implementation and validation

schema.sql defines the analytical schema, constraints, and a window-function view. It is separate from the API audit schema, which is created automatically by api.py.

The local implementation uses SQLite, available without extra installation. It demonstrates SQL CTEs, window functions, conditional aggregates, null-safe division, explicit grain, indexes, and constraints. PostgreSQL is a suitable enterprise migration, not a dependency falsely claimed to have run here.

Run `python sql_pipeline.py` to download the source and materialize .data/analytics.db. For an existing archive use `python sql_pipeline.py --source /path/to/credit.zip`.

The full run validated 30,000 account rows, 180,000 history rows and agreement between every SQL/Python feature value. See sql_validation.json. Month index 1 is oldest; 6 is latest. Zero bill totals produce NULL ratios before training-only imputation.
