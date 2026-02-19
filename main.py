"""
DocuAgent — Upstage 기반 문서 분석 서비스
Solar · Document Parse · Information Extract
"""

import asyncio
import base64
import datetime
import html
import io
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
import threading
import uuid
import zipfile
from collections import OrderedDict
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def _load_env(path: str = ".env") -> None:
    """Load simple KEY=VALUE lines into environment (if not already set)."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

# ─── Config ───
_load_env()
DEFAULT_UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"
DEFAULT_SOLAR_MODEL = "solar-pro2"
LOGGER = logging.getLogger("docuagent")


def _read_int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _read_csv_env(name: str, default: list[str]) -> list[str]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or default


MAX_UPLOAD_BYTES = _read_int_env(
    "DOCUAGENT_MAX_UPLOAD_BYTES",
    20 * 1024 * 1024,
    minimum=1024,
    maximum=200 * 1024 * 1024,
)
MAX_DOCS_IN_MEMORY = _read_int_env("DOCUAGENT_MAX_DOCS", 25, minimum=1, maximum=500)
MAX_CHAT_HISTORY_MESSAGES = _read_int_env(
    "DOCUAGENT_MAX_CHAT_HISTORY_MESSAGES",
    40,
    minimum=2,
    maximum=400,
)
MAX_CHAT_QUESTION_CHARS = _read_int_env(
    "DOCUAGENT_MAX_CHAT_QUESTION_CHARS",
    1000,
    minimum=16,
    maximum=20000,
)
MAX_RUNTIME_SESSIONS = _read_int_env(
    "DOCUAGENT_MAX_RUNTIME_SESSIONS",
    200,
    minimum=10,
    maximum=5000,
)
MAX_IE_PAGES = _read_int_env(
    "DOCUAGENT_IE_MAX_PAGES",
    3,
    minimum=1,
    maximum=8,
)
RATE_LIMIT_WINDOW_SEC = _read_int_env(
    "DOCUAGENT_RATE_LIMIT_WINDOW_SEC",
    60,
    minimum=1,
    maximum=3600,
)
RATE_LIMIT_ANALYZE_MAX = _read_int_env(
    "DOCUAGENT_RATE_LIMIT_ANALYZE_MAX",
    12,
    minimum=1,
    maximum=1000,
)
RATE_LIMIT_CHAT_MAX = _read_int_env(
    "DOCUAGENT_RATE_LIMIT_CHAT_MAX",
    120,
    minimum=1,
    maximum=10000,
)
RATE_LIMIT_RUNTIME_MAX = _read_int_env(
    "DOCUAGENT_RATE_LIMIT_RUNTIME_MAX",
    60,
    minimum=1,
    maximum=2000,
)
MAX_RATE_LIMIT_SESSIONS = _read_int_env(
    "DOCUAGENT_MAX_RATE_LIMIT_SESSIONS",
    2000,
    minimum=100,
    maximum=20000,
)
MAX_ANALYSIS_JOBS = _read_int_env(
    "DOCUAGENT_MAX_ANALYSIS_JOBS",
    500,
    minimum=50,
    maximum=10000,
)
MAX_ACTIVE_ANALYSIS_JOBS = _read_int_env(
    "DOCUAGENT_MAX_ACTIVE_ANALYSIS_JOBS",
    12,
    minimum=1,
    maximum=200,
)
MAX_ACTIVE_ANALYSIS_JOBS_PER_SESSION = _read_int_env(
    "DOCUAGENT_MAX_ACTIVE_ANALYSIS_JOBS_PER_SESSION",
    4,
    minimum=1,
    maximum=50,
)
DOC_TTL_SEC = _read_int_env(
    "DOCUAGENT_DOC_TTL_SEC",
    60 * 60 * 24,
    minimum=60,
    maximum=60 * 60 * 24 * 30,
)
ANALYSIS_JOB_TTL_SEC = _read_int_env(
    "DOCUAGENT_ANALYSIS_JOB_TTL_SEC",
    60 * 60 * 6,
    minimum=60,
    maximum=60 * 60 * 24 * 14,
)
METRICS_SAMPLE_SIZE = _read_int_env(
    "DOCUAGENT_METRICS_SAMPLE_SIZE",
    200,
    minimum=20,
    maximum=5000,
)
MAX_METRICS_PATHS = _read_int_env(
    "DOCUAGENT_MAX_METRICS_PATHS",
    256,
    minimum=20,
    maximum=4000,
)
SESSION_COOKIE_NAME = "docuagent_sid"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30
REQUEST_UPSTAGE_API_KEY: ContextVar[str] = ContextVar("request_upstage_api_key", default="")
ALLOWED_UPLOAD_EXTENSIONS = {
    ext.lower()
    for ext in _read_csv_env(
        "DOCUAGENT_ALLOWED_EXTENSIONS",
        [".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"],
    )
}


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


SESSION_COOKIE_SECURE = _truthy(os.getenv("DOCUAGENT_COOKIE_SECURE"))
SESSION_COOKIE_SAMESITE = (os.getenv("DOCUAGENT_COOKIE_SAMESITE") or "lax").strip().lower()
if SESSION_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    SESSION_COOKIE_SAMESITE = "lax"

CORS_ORIGINS = _read_csv_env(
    "DOCUAGENT_CORS_ORIGINS",
    [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
)
CORS_ALLOW_ALL = "*" in CORS_ORIGINS
UPSTREAM_TIMEOUT_SEC = _read_int_env(
    "DOCUAGENT_UPSTREAM_TIMEOUT_SEC",
    90,
    minimum=5,
    maximum=300,
)
UPSTREAM_RETRY_TOTAL = _read_int_env(
    "DOCUAGENT_UPSTREAM_RETRY_TOTAL",
    2,
    minimum=0,
    maximum=10,
)


def _build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=UPSTREAM_RETRY_TOTAL,
        connect=UPSTREAM_RETRY_TOTAL,
        read=UPSTREAM_RETRY_TOTAL,
        status=UPSTREAM_RETRY_TOTAL,
        backoff_factor=0.4,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"GET", "POST"}),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "DocuAgent/1.1"})
    return session


HTTP_SESSION = _build_http_session()


def _is_placeholder_key(value: str) -> bool:
    return str(value or "").strip().lower() in {"your_api_key_here", "change_me"}


def _get_upstage_api_key() -> str:
    runtime_key = (REQUEST_UPSTAGE_API_KEY.get() or "").strip()
    if runtime_key:
        return runtime_key
    return (os.getenv("UPSTAGE_API_KEY") or "").strip()


def _get_upstage_base_url() -> str:
    return (os.getenv("UPSTAGE_BASE_URL") or DEFAULT_UPSTAGE_BASE_URL).rstrip("/")


def _get_solar_model() -> str:
    return (os.getenv("UPSTAGE_SOLAR_MODEL") or DEFAULT_SOLAR_MODEL).strip()


def _is_demo_mode(api_key: Optional[str] = None) -> bool:
    # Force demo mode for reviewers: no paid keys required.
    if (os.getenv("DOCUAGENT_MODE") or "").strip().lower() in {"demo", "stub"}:
        return True

    if _truthy(os.getenv("DOCUAGENT_DEMO_MODE")):
        return True

    key = _get_upstage_api_key() if api_key is None else str(api_key or "").strip()
    if not key:
        return True

    # Common placeholder values.
    if _is_placeholder_key(key):
        return True

    return False


_client_cache: "OrderedDict[str, OpenAI]" = OrderedDict()


def _get_upstage_client() -> OpenAI:
    key = _get_upstage_api_key()
    if not key:
        raise RuntimeError("UPSTAGE_API_KEY is not set")

    base_url = _get_upstage_base_url()
    cache_key = f"{base_url}|{key}"
    if cache_key in _client_cache:
        _client_cache.move_to_end(cache_key)
        return _client_cache[cache_key]

    client = OpenAI(api_key=key, base_url=base_url)
    _client_cache[cache_key] = client
    while len(_client_cache) > 32:
        _client_cache.popitem(last=False)
    return client

app = FastAPI(title="DocuAgent", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ALLOW_ALL else CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=not CORS_ALLOW_ALL,
)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
    session_id, is_new_session = _resolve_session_id(request.cookies.get(SESSION_COOKIE_NAME))
    runtime_settings = _get_runtime_settings(session_id)
    runtime_key = str(runtime_settings.get("upstage_api_key") or "").strip()
    key_token = REQUEST_UPSTAGE_API_KEY.set(runtime_key)
    request.state.request_id = request_id
    request.state.session_id = session_id
    request.state.runtime_settings = runtime_settings
    started = time.perf_counter()
    response = None
    try:
        response = await call_next(request)
    finally:
        REQUEST_UPSTAGE_API_KEY.reset(key_token)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        route = request.scope.get("route")
        route_path = getattr(route, "path", request.url.path)
        status_code = int(getattr(response, "status_code", 500))
        _record_request_metric(request.method, route_path, status_code, elapsed_ms)
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = str(elapsed_ms)
    response.headers.setdefault("x-content-type-options", "nosniff")
    response.headers.setdefault("x-frame-options", "DENY")
    response.headers.setdefault("referrer-policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("cross-origin-opener-policy", "same-origin")
    response.headers.setdefault("permissions-policy", "camera=(), microphone=(), geolocation=()")
    if request.url.scheme == "https":
        response.headers.setdefault("strict-transport-security", "max-age=31536000; includeSubDomains")
    if is_new_session:
        cookie_secure = SESSION_COOKIE_SECURE or request.url.scheme == "https"
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            samesite=SESSION_COOKIE_SAMESITE,
            secure=cookie_secure,
            path="/",
        )
    if request.url.path.startswith("/api/") or request.url.path in {"/healthz"}:
        response.headers["cache-control"] = "no-store"
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", f"req-{uuid.uuid4().hex[:12]}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id},
        headers={"x-request-id": request_id, "cache-control": "no-store"},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", f"req-{uuid.uuid4().hex[:12]}")
    LOGGER.exception("Unhandled exception request_id=%s path=%s", request_id, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal server error", "request_id": request_id},
        headers={"x-request-id": request_id, "cache-control": "no-store"},
    )

# Serve static assets (logo, one-page image, etc.)
assets_dir = Path(__file__).parent / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# In-memory stores (single-process demo/service mode)
SERVICE_STARTED_AT = time.time()
ACTIVE_JOB_STATUSES = {"queued", "running", "canceling"}
TERMINAL_JOB_STATUSES = {"completed", "failed", "canceled"}
doc_store: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
runtime_store: "OrderedDict[str, dict[str, str]]" = OrderedDict()
rate_limit_store: "OrderedDict[str, dict[str, list[float]]]" = OrderedDict()
analysis_job_store: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
request_metrics_store: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
doc_store_lock = threading.RLock()
runtime_store_lock = threading.RLock()
rate_limit_lock = threading.RLock()
analysis_job_lock = threading.RLock()
request_metrics_lock = threading.RLock()
DOC_OWNER_KEY = "_session_id"


def _parse_utc_to_epoch(value: str) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return 0.0


def _doc_created_ts(doc: dict[str, Any]) -> float:
    raw = doc.get("_created_ts")
    if isinstance(raw, (int, float)):
        return float(raw)
    parsed = _parse_utc_to_epoch(str(doc.get("created_at_utc") or ""))
    if parsed > 0:
        doc["_created_ts"] = parsed
        return parsed
    now_ts = time.time()
    doc["_created_ts"] = now_ts
    return now_ts


def _job_updated_ts(job: dict[str, Any]) -> float:
    raw = job.get("_updated_ts")
    if isinstance(raw, (int, float)):
        return float(raw)
    parsed = _parse_utc_to_epoch(str(job.get("updated_at_utc") or job.get("created_at_utc") or ""))
    if parsed > 0:
        job["_updated_ts"] = parsed
        return parsed
    now_ts = time.time()
    job["_updated_ts"] = now_ts
    return now_ts


def _prune_doc_store(now_ts: Optional[float] = None) -> int:
    safe_now = float(now_ts if now_ts is not None else time.time())
    cutoff = safe_now - DOC_TTL_SEC
    removed = 0
    with doc_store_lock:
        stale_doc_ids = [
            doc_id
            for doc_id, doc in list(doc_store.items())
            if _doc_created_ts(doc) < cutoff
        ]
        for doc_id in stale_doc_ids:
            if doc_store.pop(doc_id, None) is not None:
                removed += 1
    return removed


def _prune_analysis_jobs_locked(now_ts: Optional[float] = None) -> int:
    safe_now = float(now_ts if now_ts is not None else time.time())
    cutoff = safe_now - ANALYSIS_JOB_TTL_SEC
    stale_job_ids = []
    for job_id, job in list(analysis_job_store.items()):
        status = str(job.get("status") or "")
        if status in TERMINAL_JOB_STATUSES and _job_updated_ts(job) < cutoff:
            stale_job_ids.append(job_id)
    removed = 0
    for job_id in stale_job_ids:
        if analysis_job_store.pop(job_id, None) is not None:
            removed += 1
    return removed


def _prune_analysis_jobs() -> int:
    with analysis_job_lock:
        return _prune_analysis_jobs_locked()


def _count_active_analysis_jobs_locked(session_id: Optional[str] = None) -> int:
    target_session = str(session_id or "")
    count = 0
    for job in analysis_job_store.values():
        status = str(job.get("status") or "")
        if status not in ACTIVE_JOB_STATUSES:
            continue
        if target_session and str(job.get("session_id") or "") != target_session:
            continue
        count += 1
    return count


def _record_request_metric(method: str, path: str, status_code: int, elapsed_ms: int) -> None:
    safe_method = str(method or "GET").upper()
    safe_path = str(path or "/")
    key = f"{safe_method} {safe_path}"
    with request_metrics_lock:
        metric = request_metrics_store.get(key)
        if metric is None:
            metric = {
                "method": safe_method,
                "path": safe_path,
                "count": 0,
                "count_4xx": 0,
                "count_5xx": 0,
                "last_status": 0,
                "last_elapsed_ms": 0,
                "latencies_ms": [],
            }
            request_metrics_store[key] = metric
        request_metrics_store.move_to_end(key)
        while len(request_metrics_store) > MAX_METRICS_PATHS:
            request_metrics_store.popitem(last=False)
        metric["count"] = int(metric["count"]) + 1
        if 400 <= int(status_code) < 500:
            metric["count_4xx"] = int(metric["count_4xx"]) + 1
        if int(status_code) >= 500:
            metric["count_5xx"] = int(metric["count_5xx"]) + 1
        metric["last_status"] = int(status_code)
        metric["last_elapsed_ms"] = int(max(0, elapsed_ms))
        latencies = metric["latencies_ms"]
        latencies.append(int(max(0, elapsed_ms)))
        if len(latencies) > METRICS_SAMPLE_SIZE:
            del latencies[: len(latencies) - METRICS_SAMPLE_SIZE]


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return int(ordered[0])
    idx = int(round((len(ordered) - 1) * max(0.0, min(1.0, p))))
    idx = max(0, min(len(ordered) - 1, idx))
    return int(ordered[idx])


def _metrics_snapshot() -> dict[str, Any]:
    with request_metrics_lock:
        items = []
        total_requests = 0
        total_4xx = 0
        total_5xx = 0
        for metric in request_metrics_store.values():
            latencies = [int(v) for v in metric.get("latencies_ms", [])]
            samples = len(latencies)
            avg_ms = int(sum(latencies) / samples) if samples else 0
            p50_ms = _percentile(latencies, 0.50)
            p95_ms = _percentile(latencies, 0.95)
            entry = {
                "method": str(metric.get("method") or ""),
                "path": str(metric.get("path") or ""),
                "count": int(metric.get("count", 0)),
                "count_4xx": int(metric.get("count_4xx", 0)),
                "count_5xx": int(metric.get("count_5xx", 0)),
                "last_status": int(metric.get("last_status", 0)),
                "last_elapsed_ms": int(metric.get("last_elapsed_ms", 0)),
                "samples": samples,
                "avg_ms": avg_ms,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
            }
            total_requests += entry["count"]
            total_4xx += entry["count_4xx"]
            total_5xx += entry["count_5xx"]
            items.append(entry)

    items.sort(key=lambda item: (item["p95_ms"], item["count"]), reverse=True)
    return {
        "uptime_sec": int(max(0, time.time() - SERVICE_STARTED_AT)),
        "paths": len(items),
        "totals": {
            "requests": total_requests,
            "errors_4xx": total_4xx,
            "errors_5xx": total_5xx,
        },
        "items": items[:100],
    }


def _put_doc(session_id: str, doc_id: str, doc: dict[str, Any]) -> None:
    _prune_doc_store()
    with doc_store_lock:
        created_ts = doc.get("_created_ts")
        if not isinstance(created_ts, (int, float)):
            created_ts = time.time()
        doc["_created_ts"] = float(created_ts)
        doc[DOC_OWNER_KEY] = session_id
        doc_store[doc_id] = doc
        doc_store.move_to_end(doc_id)
        while len(doc_store) > MAX_DOCS_IN_MEMORY:
            doc_store.popitem(last=False)


def _resolve_session_id(cookie_value: Optional[str]) -> tuple[str, bool]:
    candidate = str(cookie_value or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{20,128}", candidate):
        return candidate, False
    return secrets.token_urlsafe(24), True


def _get_runtime_settings(session_id: str) -> dict[str, str]:
    with runtime_store_lock:
        settings = runtime_store.get(session_id)
        if settings is None:
            settings = {}
            runtime_store[session_id] = settings
        runtime_store.move_to_end(session_id)
        while len(runtime_store) > MAX_RUNTIME_SESSIONS:
            runtime_store.popitem(last=False)
        return settings


def _is_session_doc(doc: dict[str, Any], session_id: str) -> bool:
    return str(doc.get(DOC_OWNER_KEY) or "") == str(session_id or "")


def _iter_session_docs(session_id: str):
    _prune_doc_store()
    with doc_store_lock:
        docs = [
            (doc_id, doc)
            for doc_id, doc in reversed(doc_store.items())
            if _is_session_doc(doc, session_id)
        ]
    for item in docs:
        yield item


def _enforce_rate_limit(session_id: str, action: str, limit: int) -> None:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    with rate_limit_lock:
        bucket = rate_limit_store.get(session_id)
        if bucket is None:
            bucket = {}
            rate_limit_store[session_id] = bucket
        rate_limit_store.move_to_end(session_id)
        while len(rate_limit_store) > MAX_RATE_LIMIT_SESSIONS:
            rate_limit_store.popitem(last=False)
        hits = [ts for ts in bucket.get(action, []) if ts >= window_start]
        if len(hits) >= limit:
            retry_after = int(max(1, RATE_LIMIT_WINDOW_SEC - (now - hits[0])))
            raise HTTPException(
                429,
                f"요청이 너무 많습니다. 잠시 후 다시 시도해주세요. (retry_after={retry_after}s)",
            )
        hits.append(now)
        bucket[action] = hits


def _mask_secret(value: str) -> str:
    secret = str(value or "").strip()
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * (len(secret) - 8)}{secret[-4:]}"


# ─── Helpers ───


def _safe_error_text(text: str, limit: int = 400) -> str:
    value = str(text or "").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _validate_upload_input(filename: str, file_bytes: bytes) -> None:
    name = str(filename or "").strip()
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_UPLOAD_EXTENSIONS))
        raise HTTPException(400, f"지원하지 않는 파일 형식입니다. 허용 형식: {allowed}")
    if not file_bytes:
        raise HTTPException(400, "빈 파일입니다.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"파일이 너무 큽니다. (최대 {MAX_UPLOAD_BYTES} bytes)")


def _doc_detail_payload(doc_id: str, doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": doc_id,
        "filename": str(doc.get("filename") or ""),
        "created_at_utc": str(doc.get("created_at_utc") or ""),
        "demo_mode": str(doc.get("mode", "demo")) == "demo",
        "pages": int(doc.get("pages", 0) or 0),
        "parsed_markdown": str(doc.get("parsed_markdown") or ""),
        "extracted_data": doc.get("extracted_data", {}) or {},
        "summary": str(doc.get("summary") or ""),
        "edu_pack": doc.get("edu_pack", {}) or {},
        "audience": str(doc.get("audience") or ""),
        "goal": str(doc.get("goal") or ""),
        "tags": list(doc.get("tags", []) or []),
        "pipeline_metrics": doc.get("pipeline_metrics", {}) or {},
    }


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class AnalysisJobCanceled(Exception):
    pass


def _analysis_job_summary(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": str(job.get("status") or "queued"),
        "step": int(job.get("step", 0) or 0),
        "progress_label": str(job.get("progress_label") or ""),
        "filename": str(job.get("filename") or ""),
        "created_at_utc": str(job.get("created_at_utc") or ""),
        "updated_at_utc": str(job.get("updated_at_utc") or ""),
        "error": str(job.get("error") or ""),
        "doc_id": str(job.get("doc_id") or ""),
        "cancel_requested": bool(job.get("cancel_requested", False)),
    }


def _build_job_payload(session_id: str, job_id: str) -> dict[str, Any]:
    with analysis_job_lock:
        _prune_analysis_jobs_locked()
        job = analysis_job_store.get(job_id)
        if not job or str(job.get("session_id") or "") != str(session_id or ""):
            raise HTTPException(404, "분석 작업을 찾을 수 없습니다.")
        payload = _analysis_job_summary(job_id, job)

    doc_id = payload.get("doc_id")
    if payload["status"] == "completed" and doc_id:
        _prune_doc_store()
        with doc_store_lock:
            doc = doc_store.get(doc_id)
            if doc and _is_session_doc(doc, session_id):
                payload["result"] = _doc_detail_payload(doc_id, doc)
    return payload


def _put_analysis_job(job_id: str, job: dict[str, Any]) -> None:
    with analysis_job_lock:
        now_ts = time.time()
        created_ts = job.get("_created_ts")
        updated_ts = job.get("_updated_ts")
        job["_created_ts"] = float(created_ts) if isinstance(created_ts, (int, float)) else now_ts
        job["_updated_ts"] = float(updated_ts) if isinstance(updated_ts, (int, float)) else now_ts
        job.setdefault("created_at_utc", _utc_now_iso())
        job.setdefault("updated_at_utc", job.get("created_at_utc", _utc_now_iso()))
        job.setdefault("cancel_requested", False)
        _prune_analysis_jobs_locked(now_ts)
        analysis_job_store[job_id] = job
        analysis_job_store.move_to_end(job_id)
        while len(analysis_job_store) > MAX_ANALYSIS_JOBS:
            analysis_job_store.popitem(last=False)


def _update_analysis_job(job_id: str, **updates: Any) -> None:
    with analysis_job_lock:
        _prune_analysis_jobs_locked()
        job = analysis_job_store.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at_utc"] = _utc_now_iso()
        job["_updated_ts"] = time.time()
        analysis_job_store.move_to_end(job_id)


def _list_session_analysis_jobs(
    session_id: str,
    *,
    limit: int,
    status_filter: set[str],
) -> list[dict[str, Any]]:
    with analysis_job_lock:
        _prune_analysis_jobs_locked()
        items = []
        for job_id, job in reversed(analysis_job_store.items()):
            if str(job.get("session_id") or "") != str(session_id or ""):
                continue
            status = str(job.get("status") or "queued")
            if status_filter and status not in status_filter:
                continue
            items.append(_analysis_job_summary(job_id, job))
            if len(items) >= limit:
                break
        return items


def _create_analysis_job_record(session_id: str, filename: str) -> str:
    with analysis_job_lock:
        now_ts = time.time()
        _prune_analysis_jobs_locked(now_ts)
        active_total = _count_active_analysis_jobs_locked()
        if active_total >= MAX_ACTIVE_ANALYSIS_JOBS:
            raise HTTPException(
                503,
                "현재 분석 작업이 많아 새 작업을 받을 수 없습니다. 잠시 후 다시 시도해주세요.",
            )
        active_session = _count_active_analysis_jobs_locked(session_id)
        if active_session >= MAX_ACTIVE_ANALYSIS_JOBS_PER_SESSION:
            raise HTTPException(
                429,
                "현재 세션의 동시 분석 작업 한도를 초과했습니다. 실행 중 작업이 끝난 뒤 다시 시도해주세요.",
            )
        job_id = secrets.token_urlsafe(10)
        now_iso = _utc_now_iso()
        analysis_job_store[job_id] = {
            "session_id": session_id,
            "status": "queued",
            "step": 0,
            "progress_label": "queued",
            "filename": filename,
            "created_at_utc": now_iso,
            "updated_at_utc": now_iso,
            "error": "",
            "doc_id": "",
            "cancel_requested": False,
            "_created_ts": now_ts,
            "_updated_ts": now_ts,
        }
        analysis_job_store.move_to_end(job_id)
        while len(analysis_job_store) > MAX_ANALYSIS_JOBS:
            analysis_job_store.popitem(last=False)
        return job_id


def _request_cancel_analysis_job(session_id: str, job_id: str) -> dict[str, Any]:
    with analysis_job_lock:
        _prune_analysis_jobs_locked()
        job = analysis_job_store.get(job_id)
        if not job or str(job.get("session_id") or "") != str(session_id or ""):
            raise HTTPException(404, "분석 작업을 찾을 수 없습니다.")
        status = str(job.get("status") or "")
        if status in TERMINAL_JOB_STATUSES:
            return _analysis_job_summary(job_id, job)
        job["cancel_requested"] = True
        if status == "queued":
            job["status"] = "canceled"
            job["progress_label"] = "canceled"
            job["error"] = "사용자가 작업을 취소했습니다."
        else:
            job["status"] = "canceling"
            job["progress_label"] = "canceling"
            if not str(job.get("error") or "").strip():
                job["error"] = "취소 요청됨"
        job["updated_at_utc"] = _utc_now_iso()
        job["_updated_ts"] = time.time()
        analysis_job_store.move_to_end(job_id)
        return _analysis_job_summary(job_id, job)


def _job_cancel_requested(job_id: str) -> bool:
    with analysis_job_lock:
        job = analysis_job_store.get(job_id)
        if not job:
            return True
        return bool(job.get("cancel_requested", False))

def _extract_json(content: str) -> Optional[dict]:
    """Best-effort JSON extraction from model responses."""
    text = (content or "").strip()
    if not text:
        return None

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    def try_load(raw: str) -> Optional[dict]:
        try:
            loaded = json.loads(raw)
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            return None

    loaded = try_load(text)
    if loaded is not None:
        return loaded

    # Repair the most common drift: trailing commas.
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    return try_load(repaired)


# ─── Upstage API Wrappers ───

def call_document_parse(file_bytes: bytes, filename: str) -> dict:
    """Step 1 (Live): Document Parse — 문서를 구조화된 마크다운으로 변환"""
    if _is_demo_mode():
        return _demo_document_parse(file_bytes, filename)

    api_key = _get_upstage_api_key()
    if not api_key:
        raise HTTPException(400, "UPSTAGE_API_KEY is not set.")

    url = f"{_get_upstage_base_url()}/document-ai/document-parse"
    headers = {"Authorization": f"Bearer {api_key}"}

    mime, _ = mimetypes.guess_type(filename)
    if not mime and filename.lower().endswith((".tif", ".tiff")):
        mime = "image/tiff"
    mime = mime or "application/octet-stream"

    files = {"document": (filename, file_bytes, mime)}
    data = {"output_format": "markdown"}

    try:
        resp = HTTP_SESSION.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=UPSTREAM_TIMEOUT_SEC,
        )
    except requests.RequestException as e:
        raise HTTPException(502, f"Document Parse 요청 실패: {e}")
    if resp.status_code != 200:
        raise HTTPException(
            502,
            f"Document Parse 실패({resp.status_code}): {_safe_error_text(resp.text)}",
        )

    try:
        result = resp.json()
    except Exception:
        raise HTTPException(502, "Document Parse 응답이 JSON 형식이 아닙니다.")

    return {
        "markdown": result.get("content", {}).get("markdown", ""),
        "elements": result.get("elements", []),
        "pages": result.get("usage", {}).get("pages", 0),
    }


def call_information_extract(file_bytes: bytes, filename: str, schema: dict) -> dict:
    """Step 2 (Live): Information Extract — 문서에서 구조화된 데이터 추출"""
    if _is_demo_mode():
        props = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(props, dict) and props:
            extracted = {str(k): "" for k in props.keys()}
        else:
            extracted = {}
        if "title" in extracted and not extracted["title"]:
            extracted["title"] = Path(filename or "document").stem
        extracted.setdefault("mode", "demo")
        return extracted

    api_key = _get_upstage_api_key()
    if not api_key:
        raise HTTPException(400, "UPSTAGE_API_KEY is not set.")

    # IE는 이미지 입력이므로, PDF/이미지를 PNG로 정규화한다.
    if filename.lower().endswith(".pdf"):
        images = _pdf_to_png_images(file_bytes, max_pages=MAX_IE_PAGES)
    else:
        images = [_image_to_png_bytes(file_bytes)]

    content_items = []
    for img_bytes in images:
        b64_data = base64.b64encode(img_bytes).decode()
        content_items.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_data}"},
            }
        )

    # IE API 호출 (Upstage OpenAI-compatible style)
    ie_payload = {
        "model": "information-extract",
        "messages": [{
            "role": "user",
            "content": content_items,
        }],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "document_extraction",
                "schema": schema
            }
        }
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        resp = HTTP_SESSION.post(
            f"{_get_upstage_base_url()}/information-extraction",
            headers=headers,
            json=ie_payload,
            timeout=UPSTREAM_TIMEOUT_SEC,
        )
    except requests.RequestException as e:
        raise HTTPException(502, f"Information Extract 요청 실패: {e}")
    
    if resp.status_code != 200:
        raise HTTPException(
            502,
            f"Information Extract 실패({resp.status_code}): {_safe_error_text(resp.text)}",
        )

    try:
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(502, "Information Extract 응답 파싱 실패 (choices/message/content).")
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}


def _image_to_png_bytes(file_bytes: bytes) -> bytes:
    """Convert arbitrary image bytes into PNG bytes for consistent downstream handling."""
    try:
        from PIL import Image
    except Exception as e:
        raise HTTPException(500, f"Pillow가 필요합니다: {e}")

    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(400, f"이미지 파일을 열 수 없습니다: {e}")

    if img.mode in {"RGBA", "LA"}:
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _pdf_to_png_images(file_bytes: bytes, max_pages: int = 3) -> list[bytes]:
    """Convert PDF pages to PNG bytes using poppler or pdfium."""
    safe_pages = max(1, min(int(max_pages), 8))

    if shutil.which("pdftoppm"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        out_prefix = tmp_path.replace(".pdf", "_page")
        out_dir = Path(tmp_path).parent
        out_base = Path(out_prefix).name
        image_paths: list[Path] = []
        try:
            result = subprocess.run(
                ["pdftoppm", tmp_path, out_prefix, "-png", "-f", "1", "-l", str(safe_pages)],
                capture_output=True,
            )
            if result.returncode != 0:
                err = result.stderr.decode("utf-8", errors="ignore").strip()
                raise HTTPException(500, f"PDF 변환 실패: {err or 'pdftoppm 실행 오류'}")

            image_paths = sorted(out_dir.glob(f"{out_base}-*.png"))[:safe_pages]
            if not image_paths:
                single = Path(out_prefix + ".png")
                if single.exists():
                    image_paths = [single]
            if not image_paths:
                raise HTTPException(500, "PDF 변환 실패: 변환된 PNG 파일이 없습니다.")

            images: list[bytes] = []
            for image_path in image_paths:
                with image_path.open("rb") as f:
                    images.append(f.read())
            return images
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            for image_path in image_paths:
                try:
                    image_path.unlink()
                except OSError:
                    pass

    try:
        import pypdfium2 as pdfium
    except Exception:
        raise HTTPException(
            400,
            "PDF 처리를 위해 poppler(pdftoppm) 또는 pypdfium2가 필요합니다. "
            "macOS: brew install poppler 또는 pip install pypdfium2 pillow"
        )

    try:
        pdf = pdfium.PdfDocument(file_bytes)
        images: list[bytes] = []
        page_count = min(len(pdf), safe_pages)
        for idx in range(page_count):
            page = pdf[idx]
            bitmap = page.render(scale=2)
            pil_image = bitmap.to_pil()
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            page.close()
            images.append(buf.getvalue())
        pdf.close()
        if not images:
            raise HTTPException(400, "PDF 페이지를 찾을 수 없습니다.")
        return images
    except Exception as e:
        raise HTTPException(500, f"PDF 변환 실패: {e}")


def _pdf_to_png_bytes(file_bytes: bytes) -> bytes:
    """Backward-compatible helper for single-page conversion."""
    return _pdf_to_png_images(file_bytes, max_pages=1)[0]


def call_solar_chat(system_prompt: str, user_message: str, history: list = None) -> str:
    """Solar — 문서 기반 Q&A"""
    if _is_demo_mode():
        return (
            "Demo mode: Solar responses are stubbed. "
            "Configure UPSTAGE_API_KEY to enable real generation."
        )

    client = _get_upstage_client()
    model = _get_solar_model()

    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1500,
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(502, f"Solar 호출 실패: {_safe_error_text(e)}")
    
    return response.choices[0].message.content


def auto_detect_schema(parsed_markdown: str) -> dict:
    """Solar로 문서 유형을 분석하고 자동으로 추출 스키마 생성"""
    # Demo mode: deterministic fallback schema (no network call).
    if _is_demo_mode():
        return {
            "type": "object",
            "properties": {
                "document_type": {"type": "string", "description": "문서 유형"},
                "title": {"type": "string", "description": "문서 제목"},
                "date": {"type": "string", "description": "날짜"},
                "author_or_issuer": {"type": "string", "description": "작성자/발행자"},
                "key_content": {"type": "string", "description": "핵심 내용"},
                "amounts_or_numbers": {"type": "string", "description": "금액/수치"},
            },
        }

    client = _get_upstage_client()
    model = _get_solar_model()
    
    prompt = """당신은 문서 분석 전문가입니다. 아래 문서 내용을 보고, 이 문서에서 추출해야 할 핵심 필드를 JSON Schema 형태로 생성하세요.

규칙:
1. 필드는 5~10개 정도로 제한
2. 모든 필드의 type은 "string"
3. 각 필드에 description 포함 (한국어)
4. JSON만 출력 (다른 텍스트 없이)

출력 형식:
{
  "type": "object",
  "properties": {
    "field_name": {"type": "string", "description": "설명"}
  }
}

문서 내용:
""" + parsed_markdown[:2000]
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0,
        )
    except Exception:
        # 폴백 스키마는 아래에서 반환
        response = None
    
    if response is not None:
        content = response.choices[0].message.content.strip()
        parsed = _extract_json(content)
        if parsed:
            return parsed

    # 폴백 스키마
    return {
        "type": "object",
        "properties": {
            "document_type": {"type": "string", "description": "문서 유형"},
            "title": {"type": "string", "description": "문서 제목"},
            "date": {"type": "string", "description": "날짜"},
            "author_or_issuer": {"type": "string", "description": "작성자/발행자"},
            "key_content": {"type": "string", "description": "핵심 내용"},
            "amounts_or_numbers": {"type": "string", "description": "금액/수치"},
        }
    }


def generate_edu_pack(
    parsed_markdown: str,
    extracted_data: dict,
    audience: str,
    goal: str
) -> dict:
    """교육 콘텐츠 패키지 생성 (학습 목표/핵심 개념/퀴즈/플래시카드 등)"""
    # Demo mode: return a helpful placeholder so the UI flow works end-to-end
    # without external API calls.
    if _is_demo_mode():
        title = ""
        if isinstance(extracted_data, dict):
            title = str(extracted_data.get("title") or extracted_data.get("document_type") or "")
        title = title.strip()

        key_concepts: list[str] = []
        if title:
            key_concepts.append(title)

        if isinstance(extracted_data, dict):
            for k, v in extracted_data.items():
                if k in {"mode", "raw"}:
                    continue
                v_str = str(v or "").strip()
                if not v_str:
                    continue
                if len(v_str) > 40:
                    v_str = v_str[:37] + "..."
                if v_str and v_str not in key_concepts:
                    key_concepts.append(v_str)
                if len(key_concepts) >= 6:
                    break

        if not key_concepts:
            key_concepts = ["문서 구조", "핵심 개념", "용어 정리"]

        demo_summary = (
            "Demo mode: learning pack generation is stubbed. "
            "Configure UPSTAGE_API_KEY to enable real generation."
        )

        return {
            "learning_objectives": [
                "문서의 목적과 핵심 내용을 파악한다",
                "핵심 용어/개념을 정리한다",
                "문서 기반 질문에 답할 수 있다",
            ],
            "key_concepts": key_concepts,
            "summary": demo_summary,
            "quiz": [
                {
                    "question": "이 문서의 핵심 주제는 무엇인가요?",
                    "answer": title or "문서 내용을 기반으로 정리합니다.",
                }
            ],
            "flashcards": [{"front": "핵심 개념", "back": ", ".join(key_concepts[:3])}],
            "activities": [
                "문서에서 중요한 문장을 3개 선택해 근거와 함께 정리하기",
                "핵심 개념 3개를 예시와 함께 설명하기",
            ],
        }

    client = _get_upstage_client()
    model = _get_solar_model()

    prompt = f"""당신은 교육 콘텐츠 설계 전문가입니다. 아래 문서 내용과 추출 정보를 바탕으로 교육용 패키지를 JSON으로만 출력하세요.

학습자 수준: {audience}
학습 목표: {goal}

요구 사항:
1) 문서 근거 기반으로 작성 (없는 내용은 추정하지 말 것)
2) 한국어로 간결하게 작성
3) JSON 외의 텍스트는 출력하지 말 것

출력 형식:
{{
  "learning_objectives": ["..."],
  "key_concepts": ["..."],
  "summary": "3~5문장",
  "quiz": [{{"question": "...", "answer": "..."}}],
  "flashcards": [{{"front": "...", "back": "..."}}],
  "activities": ["..."]
}}

[문서 내용]
{parsed_markdown[:2500]}

[추출 정보]
{json.dumps(extracted_data, ensure_ascii=False, indent=2)}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2,
        )
        content = response.choices[0].message.content.strip()
        parsed = _extract_json(content)
        if parsed:
            return parsed
    except Exception:
        pass

    return {
        "learning_objectives": ["문서의 핵심을 이해한다", "핵심 개념을 정리한다", "문서 기반 질문에 답한다"],
        "key_concepts": [],
        "summary": "",
        "quiz": [],
        "flashcards": [],
        "activities": []
    }


def _get_doc(session_id: str, doc_id: str) -> dict:
    _prune_doc_store()
    with doc_store_lock:
        if not doc_store:
            raise HTTPException(400, "먼저 문서를 업로드해주세요.")

        # Backward-compat alias: "current" means "most recently analyzed document".
        if not doc_id or doc_id == "current":
            for _, doc in reversed(doc_store.items()):
                if _is_session_doc(doc, session_id):
                    return doc
            raise HTTPException(400, "현재 세션에 분석된 문서가 없습니다. 먼저 문서를 업로드해주세요.")

        doc = doc_store.get(doc_id)
        if not doc or not _is_session_doc(doc, session_id):
            raise HTTPException(404, "문서를 찾을 수 없습니다. 다시 업로드해주세요.")
        return doc


def _normalize_chat_question(question: str) -> str:
    safe_question = str(question or "").strip()
    if not safe_question:
        raise HTTPException(400, "질문을 입력해주세요.")
    if len(safe_question) > MAX_CHAT_QUESTION_CHARS:
        raise HTTPException(400, f"질문이 너무 깁니다. {MAX_CHAT_QUESTION_CHARS}자 이하로 입력해주세요.")
    return safe_question


def _append_chat_exchange(doc: dict, question: str, answer: str) -> None:
    with doc_store_lock:
        history = doc.get("chat_history")
        if not isinstance(history, list):
            history = []
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > MAX_CHAT_HISTORY_MESSAGES:
            history = history[-MAX_CHAT_HISTORY_MESSAGES:]
        doc["chat_history"] = history


def _doc_summary(doc_id: str, doc: dict) -> dict:
    return {
        "doc_id": doc_id,
        "filename": str(doc.get("filename", "")),
        "created_at_utc": str(doc.get("created_at_utc", "")),
        "pages": int(doc.get("pages", 0) or 0),
        "mode": str(doc.get("mode", "demo")),
        "audience": str(doc.get("audience", "")),
        "goal": str(doc.get("goal", "")),
        "tags": list(doc.get("tags", []) or []),
    }


def _parse_tags(tag_text: str) -> list:
    if not tag_text:
        return []
    raw = tag_text.replace("\n", ",").replace("#", " ").split(",")
    tags = []
    seen = set()
    for t in raw:
        t = t.strip()
        if not t:
            continue
        if t not in seen:
            tags.append(t)
            seen.add(t)
    return tags[:12]


def _extract_pdf_text_local(file_bytes: bytes, max_pages: int = 3, max_chars: int = 8000) -> tuple[str, int]:
    """Best-effort PDF text extraction for demo mode (no network calls)."""
    try:
        from pypdf import PdfReader
    except Exception:
        return "", 0

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        page_count = len(reader.pages)
        chunks: list[str] = []
        for i in range(min(page_count, max_pages)):
            try:
                chunks.append(reader.pages[i].extract_text() or "")
            except Exception:
                continue
        text = "\n".join(chunks).strip()
        if len(text) > max_chars:
            text = text[:max_chars]
        return text, page_count
    except Exception:
        return "", 0


def _demo_document_parse(file_bytes: bytes, filename: str) -> dict:
    """Demo-mode local 'parse' so reviewers can run end-to-end without an API key."""
    suffix = Path(filename).suffix.lower()
    stem = Path(filename).stem or "document"

    if suffix == ".pdf":
        extracted_text, pages = _extract_pdf_text_local(file_bytes)
        if extracted_text:
            md = (
                "# Demo Document Parse\n\n"
                f"**Filename:** {html.escape(filename or 'document')}\n\n"
                f"## {stem}\n\n"
                "### Extracted Text (Local)\n\n"
                f"{extracted_text}\n"
            )
        else:
            md = (
                "# Demo Document Parse\n\n"
                f"**Filename:** {html.escape(filename or 'document')}\n\n"
                f"## {stem}\n\n"
                "### Note\n\n"
                "텍스트를 추출하지 못했습니다. (스캔 PDF/이미지 기반 PDF일 수 있습니다)\n"
                "데모 모드에서는 OCR을 수행하지 않습니다.\n"
            )
        return {"markdown": md, "elements": [], "pages": pages or 1}

    # Images (png/jpg/tiff): no OCR in demo mode
    return {
        "markdown": (
            "# Demo Document Parse\n\n"
            f"**Filename:** {html.escape(filename or 'document')}\n\n"
            f"## {stem}\n\n"
            "이미지 파일이 업로드되었습니다. 데모 모드에서는 OCR을 수행하지 않습니다.\n"
        ),
        "elements": [],
        "pages": 1,
    }


def _demo_information_extract(parsed_markdown: str, filename: str, schema: dict) -> dict:
    plain = re.sub(r"[#>*_`]+", " ", parsed_markdown or "")
    plain = re.sub(r"\s+", " ", plain).strip()

    stem = Path(filename).stem or "document"
    title = stem
    for line in (parsed_markdown or "").splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip() or title
            break

    def find_first(patterns: list[str]) -> str:
        for pat in patterns:
            m = re.search(pat, plain, flags=re.IGNORECASE)
            if m:
                return m.group(0).strip()
        return ""

    date = find_first(
        [
            r"\b20\d{2}[./-]\d{1,2}[./-]\d{1,2}\b",
            r"\b20\d{2}년\s*\d{1,2}월\s*\d{1,2}일\b",
        ]
    )
    amounts = find_first(
        [
            r"(?:₩|\$|USD|KRW)\s*\d{1,3}(?:,\d{3})+(?:\.\d+)?",
            r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
        ]
    )

    lowered = plain.lower()
    if any(k in lowered for k in ["invoice", "영수증", "청구", "견적"]):
        doc_type = "청구/정산 문서"
    elif any(k in lowered for k in ["계약", "agreement", "nda"]):
        doc_type = "계약 문서"
    elif any(k in lowered for k in ["학습", "lecture", "course", "교육"]):
        doc_type = "학습 자료"
    else:
        doc_type = "일반 문서"

    key_content = plain[:280] + ("..." if len(plain) > 280 else "")
    data = {
        "document_type": doc_type,
        "title": title,
        "date": date,
        "author_or_issuer": "DocuAgent Demo",
        "key_content": key_content,
        "amounts_or_numbers": amounts,
    }

    props = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(props, dict) and props:
        # keep only schema keys, but fill missing ones with empty string for UI stability
        return {k: str(data.get(k, "")) for k in props.keys()}

    return data


def _demo_summary(extracted: dict) -> str:
    doc_type = str(extracted.get("document_type") or "-")
    title = str(extracted.get("title") or "-")
    date = str(extracted.get("date") or "-")
    key = str(extracted.get("key_content") or "").strip()
    key_line = key[:180] + ("..." if len(key) > 180 else "")
    lines = [
        f"문서 유형: {doc_type}",
        f"제목: {title}",
        f"날짜: {date}",
    ]
    if key_line:
        lines.append(f"핵심: {key_line}")
    return "\n".join(lines[:5]).strip()


def _demo_chat_answer(doc: dict, question: str) -> str:
    q = (question or "").strip()
    if not q:
        return "질문을 입력해주세요."

    q_lower = q.lower()
    if "요약" in q or "summary" in q_lower:
        return str(doc.get("summary") or "")

    extracted = doc.get("extracted_data", {}) or {}
    aliases = {
        "title": ["제목", "타이틀", "title"],
        "date": ["날짜", "일자", "date"],
        "document_type": ["문서 유형", "문서종류", "type", "doctype"],
        "author_or_issuer": ["작성자", "발행자", "issuer", "author"],
        "amounts_or_numbers": ["금액", "수치", "amount", "number"],
        "key_content": ["핵심", "내용", "요점", "key"],
    }
    for field, keys in aliases.items():
        if any(k.lower() in q_lower for k in keys):
            val = extracted.get(field)
            if val:
                return f"{field}: {val}"

    return (
        "데모 모드에서는 LLM 호출 없이 규칙 기반 답변만 제공합니다.\n"
        "UPSTAGE_API_KEY를 설정하면 Solar 기반 Q&A가 동작합니다."
    )


def _export_html(doc: dict) -> str:
    title = html.escape(doc.get("filename", "DocuAgent 결과"))
    summary = html.escape(doc.get("summary", "-"))
    audience = html.escape(doc.get("audience", "-"))
    goal = html.escape(doc.get("goal", "-"))
    pages = html.escape(str(doc.get("pages", "-")))
    tags = ", ".join(doc.get("tags", []) or [])
    tags = html.escape(tags) if tags else "-"
    generated = html.escape(datetime.datetime.utcnow().isoformat())
    parsed = html.escape(doc.get("parsed_markdown", "-"))
    edu = doc.get("edu_pack", {}) or {}

    def list_items(items):
        if not items:
            return "<li>없음</li>"
        return "".join(f"<li>{html.escape(str(i))}</li>" for i in items)

    def qa_items(items, q_key, a_key):
        if not items:
            return "<div class='qa'>없음</div>"
        blocks = []
        for i, item in enumerate(items, start=1):
            q = html.escape(str(item.get(q_key, "")))
            a = html.escape(str(item.get(a_key, "")))
            blocks.append(f"<div class='qa'><strong>Q{i}.</strong> {q}<br><span class='muted'>A.</span> {a}</div>")
        return "".join(blocks)

    extracted = doc.get("extracted_data", {})
    if isinstance(extracted, dict):
        extracted_html = "".join(
            f"<li><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</li>"
            for k, v in extracted.items() if k != "raw"
        ) or "<li>없음</li>"
    else:
        extracted_html = "<li>없음</li>"

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body{{font-family:Arial, sans-serif; line-height:1.6; padding:32px; color:#111;}}
    h1,h2,h3{{margin:0 0 12px;}}
    h2{{margin-top:26px; border-bottom:1px solid #ddd; padding-bottom:6px;}}
    .meta{{font-size:12px; color:#555;}}
    .card{{background:#f8f8fb; border:1px solid #e6e6ee; padding:12px 14px; border-radius:8px;}}
    .qa{{margin:8px 0; padding:8px 10px; border:1px solid #e3e3ee; border-radius:8px;}}
    .muted{{color:#666;}}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">
    <div>페이지 수: {pages}</div>
    <div>학습자 수준: {audience}</div>
    <div>학습 목표: {goal}</div>
    <div>태그: {tags}</div>
    <div>생성일: {generated}</div>
  </div>

  <h2>요약</h2>
  <div class="card">{summary}</div>

  <h2>추출 정보</h2>
  <ul>{extracted_html}</ul>

  <h2>교육 콘텐츠</h2>
  <h3>학습 목표</h3>
  <ul>{list_items(edu.get("learning_objectives", []))}</ul>
  <h3>핵심 개념</h3>
  <ul>{list_items(edu.get("key_concepts", []))}</ul>
  <h3>교육용 요약</h3>
  <div class="card">{html.escape(edu.get("summary", "-"))}</div>
  <h3>퀴즈</h3>
  {qa_items(edu.get("quiz", []), "question", "answer")}
  <h3>플래시카드</h3>
  {qa_items(edu.get("flashcards", []), "front", "back")}
  <h3>활동/과제</h3>
  <ul>{list_items(edu.get("activities", []))}</ul>

  <h2>원문 파싱</h2>
  <div class="card"><pre>{parsed}</pre></div>
</body>
</html>
"""


def _build_scorm_zip(doc: dict) -> bytes:
    title = html.escape(doc.get("filename", "DocuAgent 결과"))
    html_body = _export_html(doc)
    manifest = f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="MANIFEST-1" version="1.2"
  xmlns="http://www.imsproject.org/xsd/imscp_rootv1p1p2"
  xmlns:adlcp="http://www.adlnet.org/xsd/adlcp_rootv1p2"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.imsproject.org/xsd/imscp_rootv1p1p2 imscp_rootv1p1p2.xsd
    http://www.adlnet.org/xsd/adlcp_rootv1p2 adlcp_rootv1p2.xsd">
  <metadata>
    <schema>ADL SCORM</schema>
    <schemaversion>1.2</schemaversion>
  </metadata>
  <organizations default="ORG-1">
    <organization identifier="ORG-1">
      <title>{title}</title>
      <item identifier="ITEM-1" identifierref="RES-1">
        <title>{title}</title>
      </item>
    </organization>
  </organizations>
  <resources>
    <resource identifier="RES-1" type="webcontent" adlcp:scormtype="sco" href="index.html">
      <file href="index.html"/>
    </resource>
  </resources>
</manifest>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr("index.html", html_body)
    return buf.getvalue()


def _build_imscc_zip(doc: dict) -> bytes:
    html_body = _export_html(doc)
    manifest = """<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="MANIFEST-IMSCC"
  xmlns="http://www.imsglobal.org/xsd/imscp_v1p1"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.imsglobal.org/xsd/imscp_v1p1 imscp_v1p1.xsd">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.1.0</schemaversion>
  </metadata>
  <organizations/>
  <resources>
    <resource identifier="RES-1" type="webcontent" href="webcontent/index.html">
      <file href="webcontent/index.html"/>
    </resource>
  </resources>
</manifest>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr("webcontent/index.html", html_body)
    return buf.getvalue()


def _build_scorm2004_zip(doc: dict) -> bytes:
    title = html.escape(doc.get("filename", "DocuAgent 결과"))
    html_body = _export_html(doc)
    manifest = f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="MANIFEST-2004" version="1.0"
  xmlns="http://www.imsglobal.org/xsd/imscp_v1p1"
  xmlns:adlcp="http://www.adlnet.org/xsd/adlcp_v1p3"
  xmlns:adlseq="http://www.adlnet.org/xsd/adlseq_v1p3"
  xmlns:adlnav="http://www.adlnet.org/xsd/adlnav_v1p3"
  xmlns:imsss="http://www.imsglobal.org/xsd/imsss"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.imsglobal.org/xsd/imscp_v1p1 imscp_v1p1.xsd
    http://www.adlnet.org/xsd/adlcp_v1p3 adlcp_v1p3.xsd
    http://www.adlnet.org/xsd/adlseq_v1p3 adlseq_v1p3.xsd
    http://www.adlnet.org/xsd/adlnav_v1p3 adlnav_v1p3.xsd
    http://www.imsglobal.org/xsd/imsss imsss_v1p0.xsd">
  <metadata>
    <schema>ADL SCORM</schema>
    <schemaversion>2004 4th Edition</schemaversion>
  </metadata>
  <organizations default="ORG-1">
    <organization identifier="ORG-1" structure="hierarchical">
      <title>{title}</title>
      <item identifier="ITEM-1" identifierref="RES-1" isvisible="true">
        <title>{title}</title>
        <imsss:sequencing>
          <imsss:controlMode choice="true" flow="true"/>
        </imsss:sequencing>
      </item>
    </organization>
  </organizations>
  <resources>
    <resource identifier="RES-1" type="webcontent" adlcp:scormType="sco" href="index.html">
      <file href="index.html"/>
    </resource>
  </resources>
</manifest>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr("index.html", html_body)
    return buf.getvalue()


def _build_imscc13_zip(doc: dict) -> bytes:
    title = html.escape(doc.get("filename", "DocuAgent 결과"))
    html_body = _export_html(doc)
    edu = doc.get("edu_pack", {}) or {}
    keywords = []
    for tag in (doc.get("tags", []) or [])[:8]:
        keywords.append(str(tag))
    for concept in (edu.get("key_concepts", []) or [])[:8]:
        keywords.append(str(concept))
    seen = set()
    keywords = [k for k in keywords if k and not (k in seen or seen.add(k))]
    if not keywords:
        keywords = ["DocuAgent"]
    keyword_xml = "".join(
        f"<lom:keyword><lom:string language=\"ko\">{html.escape(str(k))}</lom:string></lom:keyword>"
        for k in keywords
    )

    manifest = f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="MANIFEST-IMSCC-1P3"
  xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
  xmlns:lom="http://ltsc.ieee.org/xsd/LOM"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1
    http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscp_v1p2_v1p0.xsd">
  <metadata>
    <schema>1EdTech Common Cartridge</schema>
    <schemaversion>1.3.0</schemaversion>
    <lom:lom>
      <lom:general>
        <lom:title>
          <lom:string language="ko">{title}</lom:string>
        </lom:title>
        <lom:description>
          <lom:string language="ko">DocuAgent가 생성한 문서 기반 교육 콘텐츠 패키지</lom:string>
        </lom:description>
        {keyword_xml}
      </lom:general>
      <lom:educational>
        <lom:learningResourceType>
          <lom:source>LOMv1.0</lom:source>
          <lom:value>narrative text</lom:value>
        </lom:learningResourceType>
        <lom:learningResourceType>
          <lom:source>LOMv1.0</lom:source>
          <lom:value>exercise</lom:value>
        </lom:learningResourceType>
        <lom:intendedEndUserRole>
          <lom:source>LOMv1.0</lom:source>
          <lom:value>learner</lom:value>
        </lom:intendedEndUserRole>
      </lom:educational>
    </lom:lom>
  </metadata>
  <organizations/>
  <resources>
    <resource identifier="RES-1" type="webcontent" href="webcontent/index.html">
      <file href="webcontent/index.html"/>
    </resource>
  </resources>
</manifest>"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("imsmanifest.xml", manifest)
        zf.writestr("webcontent/index.html", html_body)
    return buf.getvalue()


# ─── API Endpoints ───


def _run_analysis_pipeline(
    *,
    session_id: str,
    filename: str,
    file_bytes: bytes,
    audience: str,
    goal: str,
    tags: str,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> tuple[str, dict[str, Any]]:
    demo_mode = _is_demo_mode()
    metrics: dict[str, int] = {}
    started_total = time.perf_counter()

    def check_cancel() -> None:
        if cancel_check:
            cancel_check()

    def mark_progress(step: int, label: str) -> None:
        if progress_callback:
            try:
                progress_callback(step, label)
            except Exception:
                pass

    check_cancel()
    mark_progress(1, "Document Parse")
    started = time.perf_counter()
    if demo_mode:
        parsed = _demo_document_parse(file_bytes, filename)
    else:
        parsed = call_document_parse(file_bytes, filename)
    metrics["document_parse_ms"] = int((time.perf_counter() - started) * 1000)

    check_cancel()
    mark_progress(2, "Schema + Information Extract")
    started = time.perf_counter()
    schema = auto_detect_schema(parsed["markdown"])
    metrics["schema_ms"] = int((time.perf_counter() - started) * 1000)

    check_cancel()
    started = time.perf_counter()
    if demo_mode:
        extracted = _demo_information_extract(parsed["markdown"], filename, schema)
    else:
        extracted = call_information_extract(file_bytes, filename, schema)
    metrics["information_extract_ms"] = int((time.perf_counter() - started) * 1000)

    check_cancel()
    mark_progress(3, "Solar Summary")
    started = time.perf_counter()
    if demo_mode:
        summary = _demo_summary(extracted)
    else:
        summary_prompt = f"""아래 문서 내용과 추출된 정보를 바탕으로 한국어로 간결한 분석 요약을 작성하세요.

[파싱된 문서 내용]
{parsed['markdown'][:3000]}

[추출된 정보]
{json.dumps(extracted, ensure_ascii=False, indent=2)}

요약은 3~5줄로 핵심만 작성하세요."""
        summary = call_solar_chat(
            "당신은 문서 분석 도우미입니다. 정확하고 간결하게 답변합니다.",
            summary_prompt,
        )
    metrics["summary_ms"] = int((time.perf_counter() - started) * 1000)

    check_cancel()
    mark_progress(4, "Education Pack")
    started = time.perf_counter()
    edu_pack = generate_edu_pack(parsed["markdown"], extracted, audience, goal)
    metrics["edu_pack_ms"] = int((time.perf_counter() - started) * 1000)
    metrics["total_ms"] = int((time.perf_counter() - started_total) * 1000)

    check_cancel()
    tag_list = _parse_tags(tags)
    doc_id = secrets.token_urlsafe(8)
    doc = {
        "filename": filename,
        "created_at_utc": _utc_now_iso(),
        "_created_ts": time.time(),
        "parsed_markdown": parsed["markdown"],
        "extracted_data": extracted,
        "summary": summary,
        "edu_pack": edu_pack,
        "pages": parsed["pages"],
        "chat_history": [],
        "audience": audience,
        "goal": goal,
        "tags": tag_list,
        "mode": "demo" if demo_mode else "live",
        "pipeline_metrics": metrics,
    }
    check_cancel()
    _put_doc(session_id, doc_id, doc)
    return doc_id, doc


async def _execute_analysis_job(
    *,
    job_id: str,
    session_id: str,
    filename: str,
    file_bytes: bytes,
    audience: str,
    goal: str,
    tags: str,
) -> None:
    if _job_cancel_requested(job_id):
        _update_analysis_job(
            job_id,
            status="canceled",
            progress_label="canceled",
            error="사용자가 작업을 취소했습니다.",
        )
        return

    _update_analysis_job(job_id, status="running", step=1, progress_label="Document Parse")

    def _progress(step: int, label: str) -> None:
        if _job_cancel_requested(job_id):
            return
        _update_analysis_job(job_id, status="running", step=step, progress_label=label)

    def _cancel_check() -> None:
        if _job_cancel_requested(job_id):
            raise AnalysisJobCanceled()

    try:
        doc_id, _doc = await asyncio.to_thread(
            _run_analysis_pipeline,
            session_id=session_id,
            filename=filename,
            file_bytes=file_bytes,
            audience=audience,
            goal=goal,
            tags=tags,
            progress_callback=_progress,
            cancel_check=_cancel_check,
        )
    except AnalysisJobCanceled:
        _update_analysis_job(
            job_id,
            status="canceled",
            progress_label="canceled",
            error="사용자가 작업을 취소했습니다.",
        )
        return
    except HTTPException as exc:
        _update_analysis_job(
            job_id,
            status="failed",
            error=str(exc.detail),
            progress_label="failed",
        )
        return
    except Exception as exc:
        LOGGER.exception("analysis job failed job_id=%s", job_id)
        _update_analysis_job(
            job_id,
            status="failed",
            error=f"internal server error: {_safe_error_text(exc)}",
            progress_label="failed",
        )
        return

    if _job_cancel_requested(job_id):
        with doc_store_lock:
            doc = doc_store.get(doc_id)
            if doc and _is_session_doc(doc, session_id):
                doc_store.pop(doc_id, None)
        _update_analysis_job(
            job_id,
            status="canceled",
            step=4,
            progress_label="canceled",
            doc_id="",
            error="사용자가 작업을 취소했습니다.",
        )
        return

    _update_analysis_job(
        job_id,
        status="completed",
        step=5,
        progress_label="completed",
        doc_id=doc_id,
        error="",
    )


def _runtime_config_payload(session_id: str) -> dict[str, Any]:
    settings = _get_runtime_settings(session_id)
    runtime_key = str(settings.get("upstage_api_key") or "").strip()
    env_key = (os.getenv("UPSTAGE_API_KEY") or "").strip()
    effective_key = runtime_key or env_key
    runtime_configured = bool(runtime_key) and not _is_placeholder_key(runtime_key)
    effective_configured = bool(effective_key) and not _is_placeholder_key(effective_key)
    return {
        "runtime_key_configured": runtime_configured,
        "effective_key_configured": effective_configured,
        "demo_mode": _is_demo_mode(api_key=effective_key),
        "masked_runtime_key": _mask_secret(runtime_key),
    }

@app.get("/healthz")
async def healthz() -> dict:
    converter = "pdftoppm" if shutil.which("pdftoppm") else "pypdfium2"
    key = _get_upstage_api_key()
    _prune_doc_store()
    _prune_analysis_jobs()
    with doc_store_lock:
        doc_count = len(doc_store)
    with runtime_store_lock:
        runtime_count = len(runtime_store)
    with analysis_job_lock:
        job_count = len(analysis_job_store)
        active_job_count = _count_active_analysis_jobs_locked()
    metric_summary = _metrics_snapshot()
    return {
        "status": "ok",
        "demo_mode": _is_demo_mode(),
        "upstage_key_configured": bool(key) and not _is_placeholder_key(key),
        "pdf_converter": converter,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "in_memory": {
            "docs": doc_count,
            "runtime_sessions": runtime_count,
            "analysis_jobs": job_count,
            "active_analysis_jobs": active_job_count,
        },
        "allowed_extensions": sorted(ALLOWED_UPLOAD_EXTENSIONS),
        "rate_limits": {
            "window_sec": RATE_LIMIT_WINDOW_SEC,
            "analyze": RATE_LIMIT_ANALYZE_MAX,
            "chat": RATE_LIMIT_CHAT_MAX,
            "runtime": RATE_LIMIT_RUNTIME_MAX,
        },
        "analysis_limits": {
            "max_total_jobs": MAX_ANALYSIS_JOBS,
            "max_active_jobs": MAX_ACTIVE_ANALYSIS_JOBS,
            "max_active_jobs_per_session": MAX_ACTIVE_ANALYSIS_JOBS_PER_SESSION,
        },
        "retention": {
            "doc_ttl_sec": DOC_TTL_SEC,
            "analysis_job_ttl_sec": ANALYSIS_JOB_TTL_SEC,
        },
        "metrics": {
            "tracked_paths": metric_summary["paths"],
            "requests": metric_summary["totals"]["requests"],
            "errors_4xx": metric_summary["totals"]["errors_4xx"],
            "errors_5xx": metric_summary["totals"]["errors_5xx"],
            "uptime_sec": metric_summary["uptime_sec"],
        },
        "version": app.version,
    }


@app.get("/api/metrics")
async def read_metrics(request: Request):
    _enforce_rate_limit(request.state.session_id, "runtime", RATE_LIMIT_RUNTIME_MAX)
    return _metrics_snapshot()


@app.get("/api/runtime/config")
async def read_runtime_config(request: Request):
    _enforce_rate_limit(request.state.session_id, "runtime", RATE_LIMIT_RUNTIME_MAX)
    return _runtime_config_payload(request.state.session_id)


@app.post("/api/runtime/config")
async def update_runtime_config(
    request: Request,
    upstage_api_key: str = Form(default=""),
):
    _enforce_rate_limit(request.state.session_id, "runtime", RATE_LIMIT_RUNTIME_MAX)
    key = str(upstage_api_key or "").strip()
    if key and len(key) < 12:
        raise HTTPException(400, "API 키 형식이 올바르지 않습니다. (최소 12자)")

    settings = _get_runtime_settings(request.state.session_id)
    if key:
        settings["upstage_api_key"] = key
    else:
        settings.pop("upstage_api_key", None)

    return _runtime_config_payload(request.state.session_id)


@app.post("/api/analyze")
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
    audience: str = Form("일반"),
    goal: str = Form("핵심 이해"),
    tags: str = Form("")
):
    """풀 파이프라인: Document Parse → Schema 자동 생성 → Information Extract → Solar 요약"""
    _enforce_rate_limit(request.state.session_id, "analyze", RATE_LIMIT_ANALYZE_MAX)
    
    file_bytes = await file.read()
    filename = file.filename or "document.pdf"
    _validate_upload_input(filename, file_bytes)
    doc_id, doc = _run_analysis_pipeline(
        session_id=request.state.session_id,
        filename=filename,
        file_bytes=file_bytes,
        audience=audience,
        goal=goal,
        tags=tags,
        progress_callback=None,
    )
    return _doc_detail_payload(doc_id, doc)

@app.post("/api/analyze/jobs")
async def create_analysis_job(
    request: Request,
    file: UploadFile = File(...),
    audience: str = Form("일반"),
    goal: str = Form("핵심 이해"),
    tags: str = Form(""),
):
    _enforce_rate_limit(request.state.session_id, "analyze", RATE_LIMIT_ANALYZE_MAX)

    file_bytes = await file.read()
    filename = file.filename or "document.pdf"
    _validate_upload_input(filename, file_bytes)
    job_id = _create_analysis_job_record(request.state.session_id, filename)

    asyncio.create_task(
        _execute_analysis_job(
            job_id=job_id,
            session_id=request.state.session_id,
            filename=filename,
            file_bytes=file_bytes,
            audience=audience,
            goal=goal,
            tags=tags,
        )
    )
    return {
        "job_id": job_id,
        "status": "queued",
        "poll_url": f"/api/analyze/jobs/{job_id}",
    }


@app.get("/api/analyze/jobs")
async def list_analysis_jobs(
    request: Request,
    limit: int = 20,
    status: str = "",
):
    safe_limit = max(1, min(int(limit), 100))
    valid_status = ACTIVE_JOB_STATUSES | TERMINAL_JOB_STATUSES
    status_filter = {
        token.strip().lower()
        for token in str(status or "").split(",")
        if token.strip()
    }
    invalid_status = sorted(token for token in status_filter if token not in valid_status)
    if invalid_status:
        raise HTTPException(400, f"유효하지 않은 status 값입니다: {', '.join(invalid_status)}")
    items = _list_session_analysis_jobs(
        request.state.session_id,
        limit=safe_limit,
        status_filter=status_filter,
    )
    return {
        "limit": safe_limit,
        "status_filter": sorted(status_filter),
        "items": items,
    }


@app.get("/api/analyze/jobs/{job_id}")
async def get_analysis_job(request: Request, job_id: str):
    return _build_job_payload(request.state.session_id, job_id)


@app.post("/api/analyze/jobs/{job_id}/cancel")
async def cancel_analysis_job(request: Request, job_id: str):
    return _request_cancel_analysis_job(request.state.session_id, job_id)


@app.get("/api/docs")
async def list_docs(
    request: Request,
    limit: int = 20,
    offset: int = 0,
):
    safe_limit = max(1, min(int(limit), 100))
    safe_offset = max(0, int(offset))

    doc_items = list(_iter_session_docs(request.state.session_id))
    page = doc_items[safe_offset : safe_offset + safe_limit]
    items = [_doc_summary(doc_id, doc) for doc_id, doc in page]
    return {
        "total": len(doc_items),
        "limit": safe_limit,
        "offset": safe_offset,
        "items": items,
    }


@app.get("/api/docs/{doc_id}")
async def get_doc_detail(request: Request, doc_id: str):
    doc = _get_doc(request.state.session_id, doc_id)
    return _doc_detail_payload(doc_id, doc)


@app.delete("/api/docs/{doc_id}")
async def delete_doc(request: Request, doc_id: str):
    doc = _get_doc(request.state.session_id, doc_id)
    with doc_store_lock:
        removed = doc_store.pop(doc_id, None)
    if removed is None:
        raise HTTPException(404, "문서를 찾을 수 없습니다. 다시 시도해주세요.")
    return {
        "deleted": True,
        "doc_id": doc_id,
        "filename": str(doc.get("filename") or ""),
    }


@app.post("/api/docs/clear")
async def clear_session_docs(request: Request):
    target_ids = [doc_id for doc_id, _ in _iter_session_docs(request.state.session_id)]
    deleted = 0
    with doc_store_lock:
        for doc_id in target_ids:
            if doc_store.pop(doc_id, None) is not None:
                deleted += 1
    return {"deleted": deleted}


@app.post("/api/chat")
async def chat_with_document(
    request: Request,
    question: str = Form(...),
    doc_id: str = Form(default="current")
):
    """문서 기반 Q&A — Solar"""
    _enforce_rate_limit(request.state.session_id, "chat", RATE_LIMIT_CHAT_MAX)
    doc = _get_doc(request.state.session_id, doc_id)
    safe_question = _normalize_chat_question(question)

    if doc.get("mode") == "demo":
        answer = _demo_chat_answer(doc, safe_question)
        _append_chat_exchange(doc, safe_question, answer)
        return {"answer": answer}
    
    system_prompt = f"""당신은 DocuAgent 문서 분석 도우미입니다.
아래 문서 내용과 추출된 정보를 참고하여 사용자의 질문에 정확히 답변하세요.
문서에 없는 내용은 "문서에서 확인할 수 없습니다"라고 답변하세요.
한국어로 답변합니다.

[문서: {doc['filename']}]
{doc['parsed_markdown'][:4000]}

[추출된 핵심 정보]
{json.dumps(doc['extracted_data'], ensure_ascii=False, indent=2)}"""
    
    answer = call_solar_chat(system_prompt, safe_question, doc.get("chat_history", [])[-10:])
    
    # 대화 기록 저장
    _append_chat_exchange(doc, safe_question, answer)
    
    return {"answer": answer}


@app.get("/api/export/scorm")
async def export_scorm(request: Request, doc_id: str = "current"):
    doc = _get_doc(request.state.session_id, doc_id)
    payload = _build_scorm_zip(doc)
    filename = f"{Path(doc.get('filename', 'docuagent')).stem}_scorm.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)


@app.get("/api/export/ims")
async def export_ims(request: Request, doc_id: str = "current"):
    doc = _get_doc(request.state.session_id, doc_id)
    payload = _build_imscc_zip(doc)
    filename = f"{Path(doc.get('filename', 'docuagent')).stem}_imscc.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)


@app.get("/api/export/scorm2004")
async def export_scorm2004(request: Request, doc_id: str = "current"):
    doc = _get_doc(request.state.session_id, doc_id)
    payload = _build_scorm2004_zip(doc)
    filename = f"{Path(doc.get('filename', 'docuagent')).stem}_scorm2004.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)


@app.get("/api/export/ims13")
async def export_ims13(request: Request, doc_id: str = "current"):
    doc = _get_doc(request.state.session_id, doc_id)
    payload = _build_imscc13_zip(doc)
    filename = f"{Path(doc.get('filename', 'docuagent')).stem}_imscc13.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)


@app.post("/api/update-tags")
async def update_tags(
    request: Request,
    doc_id: str = Form(...),
    tags: str = Form("")
):
    doc = _get_doc(request.state.session_id, doc_id)
    tag_list = _parse_tags(tags)
    with doc_store_lock:
        doc["tags"] = tag_list
    return {"tags": tag_list}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    # Allow override via env vars to avoid restricted binds.
    host = os.getenv("DOCUAGENT_HOST", "127.0.0.1")
    port = int(os.getenv("DOCUAGENT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
