"""Validated inference, startup checks, and a minimal durable SQLite audit log."""

import json, logging, os, sqlite3, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from inference import predict, validate_artifact

ROOT = Path(__file__).parent
DB = Path(os.getenv("AUDIT_DB", str(ROOT / "state/audit.db")))
logger = logging.getLogger("inference")


class CreditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    limit_bal: float = Field(gt=0, le=100000000)
    latest_status: int = Field(ge=-2, le=9)
    max_status: int = Field(ge=-2, le=9)
    delinquent_months: int = Field(ge=0, le=6)
    avg_utilization: float = Field(ge=-100, le=100)
    payment_to_bill: float = Field(ge=-100, le=100)
    balance_change: float = Field(ge=-100000000, le=100000000)


class TextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=20, max_length=20000)


@asynccontextmanager
async def lifespan(app):
    app.state.model = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))
    validate_artifact(app.state.model)
    DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS predictions (request_id TEXT PRIMARY KEY, scored_at TEXT DEFAULT CURRENT_TIMESTAMP, model_version TEXT, score REAL, needs_review INTEGER, latency_ms REAL)"
        )
    yield


app = FastAPI(title=ROOT.name + " model API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if not hasattr(app.state, "model"):
        raise HTTPException(503, "Model unavailable")
    return {"status": "ready", "model_version": app.state.model["version"]}


def score(payload):
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    try:
        result = predict(payload, app.state.model)
        latency = (time.perf_counter() - start) * 1000
        with sqlite3.connect(DB, timeout=5) as con:
            con.execute(
                "INSERT INTO predictions(request_id,model_version,score,needs_review,latency_ms) VALUES (?,?,?,?,?)",
                (
                    request_id,
                    app.state.model["version"],
                    result["confidence"],
                    int(result["needs_review"]),
                    latency,
                ),
            )
        return dict(
            request_id=request_id, model_version=app.state.model["version"], **result
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except Exception:
        logger.error("prediction_failed request_id=%s", request_id)
        raise HTTPException(503, "Inference or audit store unavailable") from None


# Task is fixed by the checked-in model manifest; use a separate service per repo.
TASK = json.loads((ROOT / "model.json").read_text(encoding="utf-8"))["task"]
if TASK == "credit":

    @app.post("/predict")
    def credit(payload: CreditRequest):
        return score(payload.model_dump())

else:

    @app.post("/predict")
    def text(payload: TextRequest):
        return score(payload.model_dump())


@app.get("/")
def dashboard():
    return FileResponse(ROOT / "dashboard.html")


@app.get("/assets/{name}")
def assets(name: str):
    if name not in {"dashboard.js", "model.json", "metrics.json"}:
        raise HTTPException(404)
    return FileResponse(ROOT / name)
