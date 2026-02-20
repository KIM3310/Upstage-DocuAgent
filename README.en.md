# DocuAgent

## Overview
DocuAgent parses documents, extracts key data, generates summaries and learning materials, and supports document-grounded Q&A. It uses Upstage Solar, Document Parse, and Information Extract.

## Features
- Parsing: Convert PDF/images to Markdown
- Schema generation: Create extraction schemas by document type
- Information extraction: Schema-based JSON output
- Summary and Q&A: Document-grounded responses
- Learning pack: Objectives, key concepts, quizzes, flashcards, activities
- Customization: Audience level, learning goal, and tags
- Export: Markdown, HTML, PDF, SCORM 1.2/2004, IMS CC 1.1/1.3
- LOM tags: User tags + key concepts mapped to IMS CC 1.3 keywords
- Recommended tags: One-click application of suggested tags
- Optional Ollama provider for text-generation stages (schema/summary/chat/learning pack)

## Pipeline
1) Document Parse
2) Schema generation
3) Information Extract
4) Learning pack generation
5) Summary and Q&A

## Upstage APIs
- Solar (solar-pro2): schema generation, summary, learning pack, Q&A
- Document Parse: document structure conversion
- Information Extract: schema-based extraction

## Setup
### 1) Install dependencies
```
pip install -r requirements.txt
```

### 2) Run (Demo mode, no API key)
DocuAgent can run end-to-end without paid keys so reviewers can try the full UI locally.
```
DOCUAGENT_DEMO_MODE=1 python main.py
```

### 3) Run (Live mode, Upstage API)
```
export UPSTAGE_API_KEY="your_api_key_here"
```
Or use .env
If you skip the key, DocuAgent runs in **demo mode** (stubbed outputs) so you can still try the UI end-to-end.

### 4) Optional: Live mode with Ollama provider
```
export UPSTAGE_API_KEY="your_api_key_here"
export DOCUAGENT_LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://127.0.0.1:11434"
export OLLAMA_MODEL="llama3.2:latest"
python main.py
```

### 5) Open UI
```
python main.py
```
Open http://localhost:8000
Health check: http://localhost:8000/healthz

## Usage
1) Upload a document
   - Or click the sample buttons (KO/EN) on the UI
2) Select audience level and learning goal
3) Enter LOM tags (optional)
4) (Optional) Enter Runtime settings in the UI (Upstage key / provider / Ollama)
   - Runtime settings are applied to the current session only (not persisted in localStorage)
5) Review results
6) Export to Markdown/HTML/PDF/SCORM/IMS
7) Ask questions about the document
8) Re-open or delete previous session docs from the `세션 문서` panel

## Export Formats
- Markdown: Notion-friendly
- HTML: LMS-friendly
- PDF: print to PDF
- SCORM 1.2/2004: ZIP package
- IMS CC 1.1/1.3: ZIP package

## LOM Tags
- User tags + key concepts are mapped to IMS CC 1.3 keywords
- Recommended tags can be applied with one click

## Project Structure
- main.py: FastAPI backend, Upstage/Ollama integration, package builders
- index.html: frontend UI
- requirements.txt: dependencies
- LMS_IMPORT_CHECKLIST.md: LMS import checklist
- LMS_IMPORT_LOG_TEMPLATE.md: issue log template

## Notes
- PDF input: poppler (pdftoppm) is preferred. If not available, pypdfium2 + Pillow are used.
- Demo mode uses best-effort local PDF text extraction via pypdf (no OCR).
- Information Extract uses up to the first 3 PDF pages by default (`DOCUAGENT_IE_MAX_PAGES`).
- Documents and runtime keys are session-scoped in-memory state; one session cannot access another session's `doc_id`.
- Uploads are validated by extension allowlist (`DOCUAGENT_ALLOWED_EXTENSIONS`).
- Upstage upstream calls use retry/timeout controls (`DOCUAGENT_UPSTREAM_RETRY_TOTAL`, `DOCUAGENT_UPSTREAM_TIMEOUT_SEC`).
- Per-session rate limits protect analyze/chat/runtime APIs (`DOCUAGENT_RATE_LIMIT_*`).
- Frontend analyze flow uses async job polling (`/api/analyze/jobs`) and supports cancel requests (`/api/analyze/jobs/{job_id}/cancel`).
- Async jobs now include per-session/global active job limits and TTL cleanup (`DOCUAGENT_MAX_ACTIVE_ANALYSIS_JOBS*`, `DOCUAGENT_ANALYSIS_JOB_TTL_SEC`).
- In-memory document retention is controlled by `DOCUAGENT_DOC_TTL_SEC`.
- Lightweight request metrics are available at `/api/metrics` (rolling latency/error aggregates).

## Security-related Env
- `DOCUAGENT_CORS_ORIGINS`: comma-separated origin allowlist
- `DOCUAGENT_COOKIE_SECURE`: force secure cookies (`1`/`true`)
- `DOCUAGENT_COOKIE_SAMESITE`: `lax` / `strict` / `none`
- `DOCUAGENT_HOST`: defaults to `127.0.0.1`
- `DOCUAGENT_MAX_ACTIVE_ANALYSIS_JOBS` / `DOCUAGENT_MAX_ACTIVE_ANALYSIS_JOBS_PER_SESSION`: async analysis concurrency guardrails
- `DOCUAGENT_ANALYSIS_JOB_TTL_SEC` / `DOCUAGENT_DOC_TTL_SEC`: in-memory retention windows
- `DOCUAGENT_METRICS_SAMPLE_SIZE` / `DOCUAGENT_MAX_METRICS_PATHS`: request metrics sample bounds

## Dev / Tests
```
pip install -r requirements-dev.txt
pytest -q
```
