# DocuAgent

## Summary
DocuAgent is a document analysis service that turns an uploaded document into:
- Structured content
- Key information extraction
- Summary
- Learning content (quiz/flashcards/activities)
- Document-grounded Q&A

## Docs
- Korean: `README.ko.md`
- English: `README.en.md`

## Quickstart
1. Install:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Configure API key (optional):
   ```bash
   export UPSTAGE_API_KEY="your_api_key_here"
   ```
   Or copy `.env.example` to `.env` and set the key there.
   - If you skip the key, the app runs in **demo mode** (stubbed outputs) so you can still try the UI.

3. Run:
   ```bash
   python3 main.py
   ```
   Open `http://localhost:8000`

## Project Structure
- `main.py`: FastAPI backend + Upstage API integration
- `index.html`: frontend UI
- `assets/`: images and demo documents
- `requirements.txt`: Python dependencies

## Demo Assets
- `assets/demo-learning-ko.pdf`
- `assets/demo-learning.pdf`

## Ops Artifacts (Portfolio)
- `RUNBOOK.md` (local demo runbook)
- `POSTMORTEM_TEMPLATE.md` (incident postmortem template)
- `.github/workflows/ci.yml` (CI: install + compile + smoke tests)
