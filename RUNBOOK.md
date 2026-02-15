# DocuAgent Runbook (Local Demo)

This is a local demo app. If `UPSTAGE_API_KEY` is configured, it calls the Upstage API.
If not, it runs in **demo mode** (stubbed outputs) so the end-to-end flow is still runnable.

## Prerequisites
- Python 3.9+ (3.11 recommended)

## Setup
1. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Configure environment:
   - Copy `.env.example` to `.env` (optional)
   - Optional: set `UPSTAGE_API_KEY` to enable real API calls
   - Optional: force demo mode with `DOCUAGENT_MODE=demo` or `DOCUAGENT_DEMO_MODE=1`

## Run
```bash
python3 main.py
```
Open `http://localhost:8000`.

## Health Check
- `GET /` should render the UI.
- `GET /healthz` should return JSON status (`demo_mode`, `pdf_converter`, etc).
- `GET /docs` should show FastAPI docs.

## Demo Script (3 minutes)
1. Upload a demo PDF from `assets/`.
2. Run Document Parse and confirm Markdown output is created (stubbed in demo mode).
3. Run Information Extract and confirm JSON extraction result (stubbed in demo mode).
4. Generate learning content (quiz/flashcards/activities).
5. Ask a follow-up question and confirm the answer references the uploaded content.

## Troubleshooting
- `UPSTAGE_API_KEY` not set:
  - Export it in the shell or set it in `.env`.
- PDF conversion errors:
  - Install poppler for `pdftoppm` (macOS: `brew install poppler`) or rely on `pypdfium2`.
