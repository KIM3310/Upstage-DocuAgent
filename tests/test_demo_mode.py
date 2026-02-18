import io
import sys
from pathlib import Path
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


def test_demo_analyze_chat_and_exports(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")

    # Analyze a local PNG in demo mode (no external calls).
    r = _request(
        "POST",
        "/api/analyze",
        files={"file": ("demo.png", _make_png_bytes(), "image/png")},
        data={"audience": "대학생", "goal": "핵심 이해", "tags": "계약서,리스크"},
    )
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["demo_mode"] is True
    doc_id = payload["doc_id"]
    assert doc_id
    assert "summary" in payload

    # Document listing should include newly analyzed docs for UI selectors.
    docs_res = _request("GET", "/api/docs", params={"limit": 10, "offset": 0})
    assert docs_res.status_code == 200, docs_res.text
    docs_payload = docs_res.json()
    assert docs_payload["total"] >= 1
    assert any(item["doc_id"] == doc_id for item in docs_payload["items"])

    # Chat should work in demo mode (rule-based response).
    r2 = _request("POST", "/api/chat", data={"question": "요약 보여줘", "doc_id": doc_id})
    assert r2.status_code == 200, r2.text
    assert r2.json().get("answer")

    too_long = "가" * (appmod.MAX_CHAT_QUESTION_CHARS + 1)
    r2_long = _request("POST", "/api/chat", data={"question": too_long, "doc_id": doc_id})
    assert r2_long.status_code == 400

    for i in range(appmod.MAX_CHAT_HISTORY_MESSAGES + 6):
        rr = _request("POST", "/api/chat", data={"question": f"Q{i}", "doc_id": doc_id})
        assert rr.status_code == 200, rr.text

    history = appmod.doc_store[doc_id]["chat_history"]
    assert len(history) <= appmod.MAX_CHAT_HISTORY_MESSAGES
    assert history[-2]["role"] == "user"
    assert history[-1]["role"] == "assistant"

    # Exports should be valid ZIPs.
    r3 = _request("GET", "/api/export/scorm", params={"doc_id": doc_id})
    assert r3.status_code == 200, r3.text
    with zipfile.ZipFile(io.BytesIO(r3.content), "r") as zf:
        names = set(zf.namelist())
        assert "imsmanifest.xml" in names
        assert "index.html" in names
