import io
import sys
from pathlib import Path
import time
import zipfile
import asyncio

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as appmod


def _make_png_bytes() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (32, 32), (120, 90, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def _request_async(method: str, url: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=appmod.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, url, **kwargs)


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    return asyncio.run(_request_async(method, url, **kwargs))


def _new_session_headers() -> dict[str, str]:
    health = _request("GET", "/healthz")
    assert health.status_code == 200, health.text
    session_id = health.cookies.get("docuagent_sid")
    assert session_id
    return {"Cookie": f"docuagent_sid={session_id}"}


def test_extract_json_repairs_trailing_commas() -> None:
    raw = "```json\n{ \"a\": 1, }\n```"
    assert appmod._extract_json(raw) == {"a": 1}


def test_healthz_smoke() -> None:
    r = _request("GET", "/healthz")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "demo_mode" in data
    assert "pdf_converter" in data
    assert "analysis_limits" in data
    assert "retention" in data
    assert "metrics" in data


def test_runtime_api_key_config_is_session_scoped(monkeypatch) -> None:
    monkeypatch.delenv("DOCUAGENT_MODE", raising=False)
    monkeypatch.delenv("DOCUAGENT_DEMO_MODE", raising=False)
    monkeypatch.delenv("UPSTAGE_API_KEY", raising=False)

    session_headers = _new_session_headers()

    config = _request(
        "POST",
        "/api/runtime/config",
        data={"upstage_api_key": "upstage_test_key_123456"},
        headers=session_headers,
    )
    assert config.status_code == 200, config.text
    payload = config.json()
    assert payload["runtime_key_configured"] is True
    assert payload["demo_mode"] is False

    health_live = _request("GET", "/healthz", headers=session_headers)
    assert health_live.status_code == 200, health_live.text
    health_live_payload = health_live.json()
    assert health_live_payload["upstage_key_configured"] is True
    assert health_live_payload["demo_mode"] is False

    clear = _request(
        "POST",
        "/api/runtime/config",
        data={"upstage_api_key": ""},
        headers=session_headers,
    )
    assert clear.status_code == 200, clear.text
    assert clear.json()["runtime_key_configured"] is False

    health_demo = _request("GET", "/healthz", headers=session_headers)
    assert health_demo.status_code == 200, health_demo.text
    health_demo_payload = health_demo.json()
    assert health_demo_payload["upstage_key_configured"] is False
    assert health_demo_payload["demo_mode"] is True


def test_demo_analyze_chat_and_exports(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    session_headers = _new_session_headers()

    # Analyze a local PNG in demo mode (no external calls).
    r = _request(
        "POST",
        "/api/analyze",
        files={"file": ("demo.png", _make_png_bytes(), "image/png")},
        data={"audience": "대학생", "goal": "핵심 이해", "tags": "계약서,리스크"},
        headers=session_headers,
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["demo_mode"] is True
    doc_id = payload["doc_id"]
    assert doc_id
    assert "summary" in payload

    # Document listing should include newly analyzed docs for UI selectors.
    docs_res = _request("GET", "/api/docs", params={"limit": 10, "offset": 0}, headers=session_headers)
    assert docs_res.status_code == 200, docs_res.text
    docs_payload = docs_res.json()
    assert docs_payload["total"] >= 1
    assert any(item["doc_id"] == doc_id for item in docs_payload["items"])

    # Chat should work in demo mode (rule-based response).
    r2 = _request("POST", "/api/chat", data={"question": "요약 보여줘", "doc_id": doc_id}, headers=session_headers)
    assert r2.status_code == 200, r2.text
    assert r2.json().get("answer")

    too_long = "가" * (appmod.MAX_CHAT_QUESTION_CHARS + 1)
    r2_long = _request("POST", "/api/chat", data={"question": too_long, "doc_id": doc_id}, headers=session_headers)
    assert r2_long.status_code == 400

    for i in range(appmod.MAX_CHAT_HISTORY_MESSAGES + 6):
        rr = _request("POST", "/api/chat", data={"question": f"Q{i}", "doc_id": doc_id}, headers=session_headers)
        assert rr.status_code == 200, rr.text

    with appmod.doc_store_lock:
        history = list(appmod.doc_store[doc_id]["chat_history"])
    assert len(history) <= appmod.MAX_CHAT_HISTORY_MESSAGES
    assert history[-2]["role"] == "user"
    assert history[-1]["role"] == "assistant"

    # Exports should be valid ZIPs.
    r3 = _request("GET", "/api/export/scorm", params={"doc_id": doc_id}, headers=session_headers)
    assert r3.status_code == 200, r3.text
    with zipfile.ZipFile(io.BytesIO(r3.content), "r") as zf:
        names = set(zf.namelist())
        assert "imsmanifest.xml" in names
        assert "index.html" in names


def test_document_access_is_session_isolated(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    owner_headers = _new_session_headers()
    other_headers = _new_session_headers()

    analyzed = _request(
        "POST",
        "/api/analyze",
        files={"file": ("private.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": ""},
        headers=owner_headers,
    )
    assert analyzed.status_code == 200, analyzed.text
    doc_id = analyzed.json()["doc_id"]

    docs_other = _request("GET", "/api/docs", params={"limit": 20, "offset": 0}, headers=other_headers)
    assert docs_other.status_code == 200, docs_other.text
    assert all(item["doc_id"] != doc_id for item in docs_other.json()["items"])

    denied_chat = _request(
        "POST",
        "/api/chat",
        data={"question": "요약", "doc_id": doc_id},
        headers=other_headers,
    )
    assert denied_chat.status_code == 404

    denied_export = _request(
        "GET",
        "/api/export/scorm",
        params={"doc_id": doc_id},
        headers=other_headers,
    )
    assert denied_export.status_code == 404


def test_doc_detail_delete_and_clear_endpoints(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    headers = _new_session_headers()

    first = _request(
        "POST",
        "/api/analyze",
        files={"file": ("first.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": "a,b"},
        headers=headers,
    )
    assert first.status_code == 200, first.text
    first_id = first.json()["doc_id"]

    detail = _request("GET", f"/api/docs/{first_id}", headers=headers)
    assert detail.status_code == 200, detail.text
    detail_payload = detail.json()
    assert detail_payload["doc_id"] == first_id
    assert "parsed_markdown" in detail_payload
    assert "edu_pack" in detail_payload

    deleted = _request("DELETE", f"/api/docs/{first_id}", headers=headers)
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["deleted"] is True

    detail_after = _request("GET", f"/api/docs/{first_id}", headers=headers)
    assert detail_after.status_code == 404

    second = _request(
        "POST",
        "/api/analyze",
        files={"file": ("second.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": ""},
        headers=headers,
    )
    assert second.status_code == 200, second.text

    third = _request(
        "POST",
        "/api/analyze",
        files={"file": ("third.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": ""},
        headers=headers,
    )
    assert third.status_code == 200, third.text

    cleared = _request("POST", "/api/docs/clear", headers=headers)
    assert cleared.status_code == 200, cleared.text
    assert int(cleared.json().get("deleted", 0)) >= 2

    docs = _request("GET", "/api/docs", params={"limit": 20, "offset": 0}, headers=headers)
    assert docs.status_code == 200, docs.text
    assert docs.json()["total"] == 0


def test_runtime_rate_limit(monkeypatch) -> None:
    monkeypatch.delenv("DOCUAGENT_MODE", raising=False)
    monkeypatch.delenv("DOCUAGENT_DEMO_MODE", raising=False)
    monkeypatch.delenv("UPSTAGE_API_KEY", raising=False)
    monkeypatch.setattr(appmod, "RATE_LIMIT_WINDOW_SEC", 60)
    monkeypatch.setattr(appmod, "RATE_LIMIT_RUNTIME_MAX", 1)
    appmod.rate_limit_store.clear()

    headers = _new_session_headers()
    first = _request("GET", "/api/runtime/config", headers=headers)
    assert first.status_code == 200, first.text

    second = _request("GET", "/api/runtime/config", headers=headers)
    assert second.status_code == 429


def test_upload_extension_validation(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    headers = _new_session_headers()
    res = _request(
        "POST",
        "/api/analyze",
        files={"file": ("bad.exe", b"fake-bytes", "application/octet-stream")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": ""},
        headers=headers,
    )
    assert res.status_code == 400
    assert "지원하지 않는 파일 형식" in res.json().get("detail", "")


def test_async_analysis_job_lifecycle_and_isolation(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    owner_headers = _new_session_headers()
    other_headers = _new_session_headers()

    created = _request(
        "POST",
        "/api/analyze/jobs",
        files={"file": ("job.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": "async"},
        headers=owner_headers,
    )
    assert created.status_code == 200, created.text
    job_id = created.json().get("job_id")
    assert job_id

    denied = _request("GET", f"/api/analyze/jobs/{job_id}", headers=other_headers)
    assert denied.status_code == 404

    final = None
    for _ in range(80):
        poll = _request("GET", f"/api/analyze/jobs/{job_id}", headers=owner_headers)
        assert poll.status_code == 200, poll.text
        payload = poll.json()
        if payload.get("status") == "completed":
            final = payload
            break
        if payload.get("status") == "failed":
            raise AssertionError(f"job failed unexpectedly: {payload}")
        time.sleep(0.05)

    assert final is not None, "job did not complete in time"
    assert final["status"] == "completed"
    result = final.get("result", {})
    assert result.get("doc_id")
    assert result.get("summary") is not None


def test_metrics_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    headers = _new_session_headers()
    for _ in range(2):
        health = _request("GET", "/healthz", headers=headers)
        assert health.status_code == 200, health.text

    metrics = _request("GET", "/api/metrics", headers=headers)
    assert metrics.status_code == 200, metrics.text
    payload = metrics.json()
    assert payload["totals"]["requests"] >= 2
    assert any(item.get("path") == "/healthz" for item in payload.get("items", []))


def test_analysis_job_cancel_and_list(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    headers = _new_session_headers()

    def _slow_pipeline(*, progress_callback=None, cancel_check=None, **_kwargs):
        if progress_callback:
            progress_callback(1, "Document Parse")
        for _ in range(200):
            if cancel_check:
                cancel_check()
            time.sleep(0.01)
        raise AssertionError("job should have been canceled before completion")

    monkeypatch.setattr(appmod, "_run_analysis_pipeline", _slow_pipeline)

    created = _request(
        "POST",
        "/api/analyze/jobs",
        files={"file": ("cancel.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": "cancel"},
        headers=headers,
    )
    assert created.status_code == 200, created.text
    job_id = created.json()["job_id"]

    listed = _request("GET", "/api/analyze/jobs", params={"limit": 20}, headers=headers)
    assert listed.status_code == 200, listed.text
    assert any(item.get("job_id") == job_id for item in listed.json().get("items", []))

    canceled = _request("POST", f"/api/analyze/jobs/{job_id}/cancel", headers=headers)
    assert canceled.status_code == 200, canceled.text
    assert canceled.json()["status"] in {"canceling", "canceled"}

    final = None
    for _ in range(80):
        polled = _request("GET", f"/api/analyze/jobs/{job_id}", headers=headers)
        assert polled.status_code == 200, polled.text
        payload = polled.json()
        if payload.get("status") in {"canceling", "canceled"}:
            final = payload
            break
        time.sleep(0.03)

    assert final is not None, "job did not reflect cancel request"
    assert final["status"] in {"canceling", "canceled"}
    assert final.get("cancel_requested") is True


def test_active_analysis_job_limit(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    monkeypatch.setattr(appmod, "MAX_ACTIVE_ANALYSIS_JOBS_PER_SESSION", 1)
    monkeypatch.setattr(appmod, "MAX_ACTIVE_ANALYSIS_JOBS", 10)
    with appmod.analysis_job_lock:
        appmod.analysis_job_store.clear()

    headers = _new_session_headers()

    def _slow_pipeline(*, progress_callback=None, cancel_check=None, **_kwargs):
        if progress_callback:
            progress_callback(1, "Document Parse")
        for _ in range(180):
            if cancel_check:
                cancel_check()
            time.sleep(0.01)
        return "should-not-complete", {}

    monkeypatch.setattr(appmod, "_run_analysis_pipeline", _slow_pipeline)

    first = _request(
        "POST",
        "/api/analyze/jobs",
        files={"file": ("first-limit.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": ""},
        headers=headers,
    )
    assert first.status_code == 200, first.text
    first_job_id = first.json()["job_id"]

    second = _request(
        "POST",
        "/api/analyze/jobs",
        files={"file": ("second-limit.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": ""},
        headers=headers,
    )
    assert second.status_code == 429, second.text

    canceled = _request("POST", f"/api/analyze/jobs/{first_job_id}/cancel", headers=headers)
    assert canceled.status_code == 200, canceled.text


def test_doc_ttl_prune(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")
    monkeypatch.setattr(appmod, "DOC_TTL_SEC", 1)
    headers = _new_session_headers()

    analyzed = _request(
        "POST",
        "/api/analyze",
        files={"file": ("ttl.png", _make_png_bytes(), "image/png")},
        data={"audience": "일반", "goal": "핵심 이해", "tags": ""},
        headers=headers,
    )
    assert analyzed.status_code == 200, analyzed.text
    doc_id = analyzed.json()["doc_id"]

    with appmod.doc_store_lock:
        appmod.doc_store[doc_id]["_created_ts"] = time.time() - 120

    docs = _request("GET", "/api/docs", params={"limit": 20, "offset": 0}, headers=headers)
    assert docs.status_code == 200, docs.text
    assert docs.json()["total"] == 0
