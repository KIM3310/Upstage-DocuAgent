# DocuAgent

## Summary
DocuAgent is a document analysis service that turns an upload into:
- Structured content (parse)
- Key information extraction (schema-based JSON)
- Summary + document-grounded Q&A
- Learning content (objectives, quiz, flashcards, activities)
- LMS-ready exports (SCORM / IMS CC)

## Docs
- English: `README.en.md`
- 한국어: `README.ko.md`

## Quickstart
1) Install
```bash
python3 -m pip install -r requirements.txt
```

2) Run (Demo mode, no paid keys)
```bash
DOCUAGENT_DEMO_MODE=1 python3 main.py
```

3) Optional: Run (Live mode, Upstage API)
```bash
UPSTAGE_API_KEY="your_api_key_here" python3 main.py
```

Open `http://localhost:8000`
- Click sample buttons on the UI: `샘플 불러오기 (KO)` / `Sample (EN)`
- Health check: `http://localhost:8000/healthz`

## Project Structure
- `main.py`: FastAPI backend + Upstage integrations + export builders
- `index.html`: frontend UI
- `assets/`: demo documents + UI assets
- `requirements.txt`: runtime dependencies
- `requirements-dev.txt`: dev/test tooling

## Demo Assets
- `assets/demo-learning-ko.pdf`
- `assets/demo-learning.pdf`

## Ops Artifacts (Portfolio)
- `RUNBOOK.md`: local demo runbook
- `POSTMORTEM_TEMPLATE.md`: incident postmortem template
- `.github/workflows/ci.yml`: CI (compile + tests)

## Dev / Tests
```bash
python3 -m pip install -r requirements-dev.txt
ruff check --select F .
pytest -q
```

