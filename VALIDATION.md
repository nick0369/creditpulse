# Validation report — CreditPulse

- 18 pytest cases passed locally on Windows / Python 3.14.7.
- Tests include exported-model parity, invalid requests, durable audit writes, SQL edge cases, and split contracts.
- 200 sequential FastAPI TestClient predictions, after 10 warmup requests: 0 errors; p95 20.11 ms. This includes a SQLite audit write and excludes network latency; it is not a throughput/SLA benchmark.
- Python / browser-JavaScript scoring parity is verified against examples.json before release.
- FastAPI dashboard, /docs, /ready and model-asset endpoints returned HTTP 200 in local checks.
- Black formatting was applied to Python sources.
- Two installed-environment deprecation warnings were observed (Starlette's httpx transition and pytest-asyncio's Python event-loop policy); tests completed successfully.
- Docker is not installed locally. GitHub Actions verified Linux dependency installation, all tests, Docker image build and container readiness: https://github.com/nick0369/creditpulse/actions/runs/39664778003 .
- SQL is executed and tested with SQLite. PostgreSQL syntax is intended to be compatible, but a PostgreSQL server was not available for a runtime test.

Re-run `python -m pytest -q` after any change. Do not substitute this report for your own checks when changing dependencies or model data. CI configuration runs tests and the container build in GitHub Actions when enabled.
