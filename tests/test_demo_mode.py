import io
import sys
from pathlib import Path
import zipfile

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import main as appmod


client = TestClient(appmod.app)


def _make_png_bytes() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (32, 32), (120, 90, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_extract_json_repairs_trailing_commas() -> None:
    raw = "```json\n{ \"a\": 1, }\n```"
    assert appmod._extract_json(raw) == {"a": 1}


def test_healthz_smoke() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "demo_mode" in data
    assert "pdf_converter" in data


def test_demo_analyze_chat_and_exports(monkeypatch) -> None:
    monkeypatch.setenv("DOCUAGENT_DEMO_MODE", "1")

    # Analyze a local PNG in demo mode (no external calls).
    r = client.post(
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

    # Chat should work in demo mode (rule-based response).
    r2 = client.post("/api/chat", data={"question": "요약 보여줘", "doc_id": doc_id})
    assert r2.status_code == 200, r2.text
    assert r2.json().get("answer")

    # Exports should be valid ZIPs.
    r3 = client.get("/api/export/scorm", params={"doc_id": doc_id})
    assert r3.status_code == 200, r3.text
    with zipfile.ZipFile(io.BytesIO(r3.content), "r") as zf:
        names = set(zf.namelist())
        assert "imsmanifest.xml" in names
        assert "index.html" in names
