"""
DocuAgent — Upstage 기반 문서 분석 서비스
Solar · Document Parse · Information Extract
"""

import os, json, base64, tempfile, subprocess, io, zipfile, datetime, html, shutil
from pathlib import Path
from typing import Optional

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
API_KEY = os.getenv("UPSTAGE_API_KEY", "your_api_key_here")
BASE_URL = "https://api.upstage.ai/v1"
SOLAR_MODEL = "solar-pro2"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

app = FastAPI(title="DocuAgent", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static assets (logo, one-page image, etc.)
assets_dir = Path(__file__).parent / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

# In-memory document store (per session, demo purpose)
doc_store: dict = {}


# ─── Helpers ───

def _extract_json(content: str) -> Optional[dict]:
    """Best-effort JSON extraction from model responses."""
    text = (content or "").strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ─── Upstage API Wrappers ───

def call_document_parse(file_bytes: bytes, filename: str) -> dict:
    """Step 1: Document Parse — 문서를 구조화된 마크다운으로 변환"""
    url = f"{BASE_URL}/document-ai/document-parse"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    mime = "application/pdf" if filename.lower().endswith(".pdf") else "image/png"
    files = {"document": (filename, file_bytes, mime)}
    data = {"output_format": "markdown"}
    
    resp = requests.post(url, headers=headers, files=files, data=data)
    if resp.status_code != 200:
        raise HTTPException(500, f"Document Parse 실패: {resp.text}")
    
    result = resp.json()
    return {
        "markdown": result.get("content", {}).get("markdown", ""),
        "elements": result.get("elements", []),
        "pages": result.get("usage", {}).get("pages", 0),
    }


def call_information_extract(file_bytes: bytes, filename: str, schema: dict) -> dict:
    """Step 2: Information Extract — 문서에서 구조화된 데이터 추출"""
    
    # PDF → 이미지 변환 필요 (IE는 이미지 입력)
    if filename.lower().endswith(".pdf"):
        img_bytes = _pdf_to_png_bytes(file_bytes)
        b64_data = base64.b64encode(img_bytes).decode()
    else:
        b64_data = base64.b64encode(file_bytes).decode()
    
    # IE API 호출
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
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    resp = requests.post(
        f"{BASE_URL}/information-extraction",
        headers=headers,
        json=ie_payload
    )
    
    if resp.status_code != 200:
        raise HTTPException(500, f"Information Extract 실패: {resp.text}")
    
    result = resp.json()
    content = result["choices"][0]["message"]["content"]
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}


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
    messages = [{"role": "system", "content": system_prompt}]
    
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_message})
    
    response = client.chat.completions.create(
        model=SOLAR_MODEL,
        messages=messages,
        max_tokens=1500,
        temperature=0.3,
    )
    
    return response.choices[0].message.content


def auto_detect_schema(parsed_markdown: str) -> dict:
    """Solar로 문서 유형을 분석하고 자동으로 추출 스키마 생성"""
    
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
        model=SOLAR_MODEL,
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
        model=SOLAR_MODEL,
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
    if doc_id not in doc_store:
        raise HTTPException(400, "먼저 문서를 업로드해주세요.")
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
    title = html.escape(doc.get("filename", "DocuAgent 결과"))
    html_body = _export_html(doc)
    manifest = f"""<?xml version="1.0" encoding="UTF-8"?>
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
    
    # Step 1: Document Parse
    parsed = call_document_parse(file_bytes, filename)
    
    # Step 2: Solar로 스키마 자동 생성
    schema = auto_detect_schema(parsed["markdown"])
    
    # Step 3: Information Extract
    extracted = call_information_extract(file_bytes, filename, schema)
    
    # Step 4: Solar로 문서 요약 생성
    summary_prompt = f"""아래 문서 내용과 추출된 정보를 바탕으로 한국어로 간결한 분석 요약을 작성하세요.

[파싱된 문서 내용]
{parsed['markdown'][:3000]}

[추출된 정보]
{json.dumps(extracted, ensure_ascii=False, indent=2)}

요약은 3~5줄로 핵심만 작성하세요."""
    
    summary = call_solar_chat(
        "당신은 문서 분석 도우미입니다. 정확하고 간결하게 답변합니다.",
        summary_prompt
    )

    # 교육 콘텐츠 패키지 생성
    edu_pack = generate_edu_pack(parsed["markdown"], extracted, audience, goal)
    tag_list = _parse_tags(tags)
    
    # 저장
    doc_id = "current"
    doc_store[doc_id] = {
        "filename": filename,
        "parsed_markdown": parsed["markdown"],
        "extracted_data": extracted,
        "summary": summary,
        "edu_pack": edu_pack,
        "pages": parsed["pages"],
        "chat_history": [],
        "audience": audience,
        "goal": goal,
        "tags": tag_list
    }
    
    return {
        "doc_id": doc_id,
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
