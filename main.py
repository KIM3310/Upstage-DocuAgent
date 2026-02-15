"""
DocuAgent — Upstage 기반 문서 분석 서비스
Solar · Document Parse · Information Extract
"""

import base64
import datetime
import html
import io
import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import requests
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

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

MAX_UPLOAD_BYTES = int(os.getenv("DOCUAGENT_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_DOCS_IN_MEMORY = int(os.getenv("DOCUAGENT_MAX_DOCS", "25"))


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_upstage_api_key() -> str:
    return (os.getenv("UPSTAGE_API_KEY") or "").strip()


def _get_upstage_base_url() -> str:
    return (os.getenv("UPSTAGE_BASE_URL") or DEFAULT_UPSTAGE_BASE_URL).rstrip("/")


def _get_solar_model() -> str:
    return (os.getenv("UPSTAGE_SOLAR_MODEL") or DEFAULT_SOLAR_MODEL).strip()


def _is_demo_mode() -> bool:
    # Force demo mode for reviewers: no paid keys required.
    if (os.getenv("DOCUAGENT_MODE") or "").strip().lower() in {"demo", "stub"}:
        return True

    if _truthy(os.getenv("DOCUAGENT_DEMO_MODE")):
        return True

    key = _get_upstage_api_key()
    if not key:
        return True

    # Common placeholder values.
    if key.lower() in {"your_api_key_here", "change_me"}:
        return True

    return False


_client: Optional[OpenAI] = None


def _get_upstage_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    key = _get_upstage_api_key()
    if not key:
        raise RuntimeError("UPSTAGE_API_KEY is not set")

    _client = OpenAI(api_key=key, base_url=_get_upstage_base_url())
    return _client

app = FastAPI(title="DocuAgent", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static assets (logo, one-page image, etc.)
assets_dir = Path(__file__).parent / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# In-memory document store (per session, demo purpose)
doc_store: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


def _put_doc(doc_id: str, doc: dict[str, Any]) -> None:
    doc_store[doc_id] = doc
    doc_store.move_to_end(doc_id)
    while len(doc_store) > MAX_DOCS_IN_MEMORY:
        doc_store.popitem(last=False)


# ─── Helpers ───

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
        raise HTTPException(
            400,
            "Demo mode is enabled (no external API calls). "
            "Set UPSTAGE_API_KEY and unset DOCUAGENT_DEMO_MODE to run live parsing.",
        )

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

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=90)
    if resp.status_code != 200:
        raise HTTPException(502, f"Document Parse 실패({resp.status_code}): {resp.text}")

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
        raise HTTPException(
            400,
            "Demo mode is enabled (no external API calls). "
            "Set UPSTAGE_API_KEY and unset DOCUAGENT_DEMO_MODE to run live extraction.",
        )

    api_key = _get_upstage_api_key()
    if not api_key:
        raise HTTPException(400, "UPSTAGE_API_KEY is not set.")

    # IE는 이미지 입력이므로, PDF/이미지를 PNG로 정규화한다.
    if filename.lower().endswith(".pdf"):
        img_bytes = _pdf_to_png_bytes(file_bytes)
    else:
        img_bytes = _image_to_png_bytes(file_bytes)
    b64_data = base64.b64encode(img_bytes).decode()

    # IE API 호출 (Upstage OpenAI-compatible style)
    ie_payload = {
        "model": "information-extract",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64_data}"}
            }]
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
    
    resp = requests.post(
        f"{_get_upstage_base_url()}/information-extraction",
        headers=headers,
        json=ie_payload,
        timeout=90,
    )
    
    if resp.status_code != 200:
        raise HTTPException(502, f"Information Extract 실패({resp.status_code}): {resp.text}")

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


def _pdf_to_png_bytes(file_bytes: bytes) -> bytes:
    """Convert first page of PDF to PNG bytes using poppler or pdfium."""
    if shutil.which("pdftoppm"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        out_prefix = tmp_path.replace(".pdf", "_page")
        result = subprocess.run(
            ["pdftoppm", tmp_path, out_prefix, "-png", "-singlefile"],
            capture_output=True
        )
        if result.returncode != 0:
            err = result.stderr.decode("utf-8", errors="ignore").strip()
            os.unlink(tmp_path)
            raise HTTPException(500, f"PDF 변환 실패: {err or 'pdftoppm 실행 오류'}")

        img_path = out_prefix + ".png"
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        os.unlink(tmp_path)
        os.unlink(img_path)
        return img_bytes

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
        page = pdf[0]
        bitmap = page.render(scale=2)
        pil_image = bitmap.to_pil()
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        page.close()
        pdf.close()
        return buf.getvalue()
    except Exception as e:
        raise HTTPException(500, f"PDF 변환 실패: {e}")


def call_solar_chat(system_prompt: str, user_message: str, history: list = None) -> str:
    """Solar — 문서 기반 Q&A"""
    if _is_demo_mode():
        raise HTTPException(
            400,
            "Demo mode is enabled (no external API calls). "
            "Set UPSTAGE_API_KEY and unset DOCUAGENT_DEMO_MODE to use Solar.",
        )

    client = _get_upstage_client()
    model = _get_solar_model()

    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1500,
        temperature=0.3,
    )
    
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
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0,
    )
    
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
    # Demo mode: simple deterministic output (no network call).
    if _is_demo_mode():
        return {
            "learning_objectives": [
                "문서의 핵심을 이해한다",
                "추출된 필드를 기준으로 내용을 구조화한다",
                "문서 기반 질문에 답할 수 있다",
            ],
            "key_concepts": (extracted_data.get("document_type") and [str(extracted_data.get("document_type"))]) or [],
            "summary": str(extracted_data.get("key_content") or "")[:220],
            "quiz": [],
            "flashcards": [],
            "activities": [],
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

    return {
        "learning_objectives": ["문서의 핵심을 이해한다", "핵심 개념을 정리한다", "문서 기반 질문에 답한다"],
        "key_concepts": [],
        "summary": "",
        "quiz": [],
        "flashcards": [],
        "activities": []
    }


def _get_doc(doc_id: str) -> dict:
    if not doc_store:
        raise HTTPException(400, "먼저 문서를 업로드해주세요.")

    # Backward-compat alias: "current" means "most recently analyzed document".
    if not doc_id or doc_id == "current":
        last_key = next(reversed(doc_store))
        return doc_store[last_key]

    if doc_id not in doc_store:
        raise HTTPException(400, "문서를 찾을 수 없습니다. 다시 업로드해주세요.")
    return doc_store[doc_id]


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
            md = f"# {stem}\n\n## Extracted Text (Local)\n\n{extracted_text}\n"
        else:
            md = (
                f"# {stem}\n\n"
                "## Note\n\n"
                "텍스트를 추출하지 못했습니다. (스캔 PDF/이미지 기반 PDF일 수 있습니다)\n"
                "데모 모드에서는 OCR을 수행하지 않습니다.\n"
            )
        return {"markdown": md, "elements": [], "pages": pages or 1}

    # Images (png/jpg/tiff): no OCR in demo mode
    return {
        "markdown": f"# {stem}\n\n이미지 파일이 업로드되었습니다. 데모 모드에서는 OCR을 수행하지 않습니다.\n",
        "elements": [],
        "pages": 1,
    }


def _demo_information_extract(parsed_markdown: str, filename: str, schema: dict) -> dict:
    plain = re.sub(r"[#>*_`]+", " ", parsed_markdown or "")
    plain = re.sub(r"\\s+", " ", plain).strip()

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
            r"\\b20\\d{2}[./-]\\d{1,2}[./-]\\d{1,2}\\b",
            r"\\b20\\d{2}년\\s*\\d{1,2}월\\s*\\d{1,2}일\\b",
        ]
    )
    amounts = find_first(
        [
            r"(?:₩|\\$|USD|KRW)\\s*\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?",
            r"\\b\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?\\b",
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

@app.get("/healthz")
async def healthz() -> dict:
    converter = "pdftoppm" if shutil.which("pdftoppm") else "pypdfium2"
    key = _get_upstage_api_key()
    return {
        "status": "ok",
        "demo_mode": _is_demo_mode(),
        "upstage_key_configured": bool(key) and key.lower() not in {"your_api_key_here", "change_me"},
        "pdf_converter": converter,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "version": app.version,
    }


@app.post("/api/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    audience: str = Form("일반"),
    goal: str = Form("핵심 이해"),
    tags: str = Form("")
):
    """풀 파이프라인: Document Parse → Schema 자동 생성 → Information Extract → Solar 요약"""
    
    file_bytes = await file.read()
    filename = file.filename or "document.pdf"

    if not file_bytes:
        raise HTTPException(400, "빈 파일입니다.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"파일이 너무 큽니다. (최대 {MAX_UPLOAD_BYTES} bytes)")

    demo_mode = _is_demo_mode()
    
    # Step 1: Document Parse
    if demo_mode:
        parsed = _demo_document_parse(file_bytes, filename)
    else:
        parsed = call_document_parse(file_bytes, filename)
    
    # Step 2: Solar로 스키마 자동 생성
    schema = auto_detect_schema(parsed["markdown"])
    
    # Step 3: Information Extract
    if demo_mode:
        extracted = _demo_information_extract(parsed["markdown"], filename, schema)
    else:
        extracted = call_information_extract(file_bytes, filename, schema)
    
    # Step 4: Solar로 문서 요약 생성
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

    # 교육 콘텐츠 패키지 생성
    edu_pack = generate_edu_pack(parsed["markdown"], extracted, audience, goal)
    tag_list = _parse_tags(tags)
    
    # 저장
    doc_id = secrets.token_urlsafe(8)
    doc = {
        "filename": filename,
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
    }
    _put_doc(doc_id, doc)
    
    return {
        "doc_id": doc_id,
        "demo_mode": demo_mode,
        "filename": filename,
        "pages": parsed["pages"],
        "parsed_markdown": parsed["markdown"],
        "extracted_data": extracted,
        "summary": summary,
        "edu_pack": edu_pack,
        "audience": audience,
        "goal": goal,
        "tags": tag_list
    }


@app.post("/api/chat")
async def chat_with_document(
    question: str = Form(...),
    doc_id: str = Form(default="current")
):
    """문서 기반 Q&A — Solar"""
    doc = _get_doc(doc_id)

    if doc.get("mode") == "demo":
        answer = _demo_chat_answer(doc, question)
        doc["chat_history"].append({"role": "user", "content": question})
        doc["chat_history"].append({"role": "assistant", "content": answer})
        return {"answer": answer}
    
    system_prompt = f"""당신은 DocuAgent 문서 분석 도우미입니다.
아래 문서 내용과 추출된 정보를 참고하여 사용자의 질문에 정확히 답변하세요.
문서에 없는 내용은 "문서에서 확인할 수 없습니다"라고 답변하세요.
한국어로 답변합니다.

[문서: {doc['filename']}]
{doc['parsed_markdown'][:4000]}

[추출된 핵심 정보]
{json.dumps(doc['extracted_data'], ensure_ascii=False, indent=2)}"""
    
    answer = call_solar_chat(system_prompt, question, doc["chat_history"][-10:])
    
    # 대화 기록 저장
    doc["chat_history"].append({"role": "user", "content": question})
    doc["chat_history"].append({"role": "assistant", "content": answer})
    
    return {"answer": answer}


@app.get("/api/export/scorm")
async def export_scorm(doc_id: str = "current"):
    doc = _get_doc(doc_id)
    payload = _build_scorm_zip(doc)
    filename = f"{Path(doc.get('filename', 'docuagent')).stem}_scorm.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)


@app.get("/api/export/ims")
async def export_ims(doc_id: str = "current"):
    doc = _get_doc(doc_id)
    payload = _build_imscc_zip(doc)
    filename = f"{Path(doc.get('filename', 'docuagent')).stem}_imscc.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)


@app.get("/api/export/scorm2004")
async def export_scorm2004(doc_id: str = "current"):
    doc = _get_doc(doc_id)
    payload = _build_scorm2004_zip(doc)
    filename = f"{Path(doc.get('filename', 'docuagent')).stem}_scorm2004.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)


@app.get("/api/export/ims13")
async def export_ims13(doc_id: str = "current"):
    doc = _get_doc(doc_id)
    payload = _build_imscc13_zip(doc)
    filename = f"{Path(doc.get('filename', 'docuagent')).stem}_imscc13.zip"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type="application/zip", headers=headers)


@app.post("/api/update-tags")
async def update_tags(
    doc_id: str = Form(...),
    tags: str = Form("")
):
    doc = _get_doc(doc_id)
    tag_list = _parse_tags(tags)
    doc["tags"] = tag_list
    return {"tags": tag_list}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn
    # Allow override via env vars to avoid restricted binds.
    host = os.getenv("DOCUAGENT_HOST", "0.0.0.0")
    port = int(os.getenv("DOCUAGENT_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
