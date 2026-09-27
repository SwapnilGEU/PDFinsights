import json
import sys
import time
from pathlib import Path
from uuid import uuid4

# Make app/'s own folder importable directly, matching how app/*.py
# imports itself internally (bare `from config import ...` etc.)
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

# Bare imports on purpose: app/rag.py imports `metrics` and `logging_utils`
# by bare name. Importing them here as `app.metrics` would load a SECOND
# copy of each module, so rag.py's retrieval/LLM stats went into a
# different MetricsCollector (and log buffer) than the one /metrics and the
# per-request logs read -> retrieval/LLM latency always showed 0.
from logging_utils import log_event
from metrics import metrics
from pydantic import BaseModel
from rag import rag_stream

app = FastAPI(title="Advanced RAG API")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Advanced RAG API is running"}


@app.post("/query")
def query_rag(request: QueryRequest, http_request: Request):
    """Streams the answer back as newline-delimited JSON (NDJSON): one
    {"type": "token", "text": ...} line per chunk of the answer, then one
    final {"type": "done", "sources": [...], "rag_section": {...},
    "llm_section": {...}} line, or {"type": "error", ...} if the pipeline
    fails partway through. This replaces the old single-JSON-blob
    response -- see rag_stream() in app/rag.py for why.
    """
    request_id = http_request.headers.get("X-Request-ID") or str(uuid4())
    request_start = time.perf_counter()

    log_event(
        "request_started",
        request_id=request_id,
        query=request.question,
        endpoint="/query",
        method="POST",
        query_length=len(request.question),
    )

    def event_stream():
        final_chunk = {}

        try:
            for chunk in rag_stream(request.question, request_id=request_id):
                if chunk["type"] == "done":
                    final_chunk = chunk

                yield json.dumps(chunk) + "\n"

            latency_ms = (time.perf_counter() - request_start) * 1000
            metrics.record_request(latency_ms, success=True)
            metrics.save_snapshot()

            log_event(
                "request_completed",
                request_id=request_id,
                status_code=200,
            )

            # Structured per-request JSON: data_logs/<date>/metrics/<id>.json
            metrics.save_request_snapshot(
                request_id,
                request_section={
                    "endpoint": "/query",
                    "question": request.question,
                    "status": "success",
                    "total_latency_ms": round(latency_ms, 2),
                },
                rag_section=final_chunk.get("rag_section", {}),
                llm_section=final_chunk.get("llm_section", {}),
            )

        except Exception:  # noqa: BLE001 - last-resort boundary around the whole stream; must still emit an error chunk and log, not crash the response mid-stream
            latency_ms = (time.perf_counter() - request_start) * 1000
            metrics.record_request(latency_ms, success=False)
            metrics.save_snapshot()

            log_event(
                "request_failed",
                request_id=request_id,
                endpoint="/query",
            )

            metrics.save_request_snapshot(
                request_id,
                request_section={
                    "endpoint": "/query",
                    "question": request.question,
                    "status": "failed",
                    "total_latency_ms": round(latency_ms, 2),
                },
                rag_section={},
                llm_section={},
            )

            yield json.dumps({"type": "error", "request_id": request_id}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/metrics")
def get_metrics():
    return metrics.summary()
