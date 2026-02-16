[Personal Project] DocuAgent — Document -> Learning Content -> LMS Package Automation (Upstage API)

Most “document AI demos” stop at summarization. The real pain is turning PDFs/screenshots into reusable assets that teams can ship into training/onboarding workflows (and doing it in a way that’s reviewable, reproducible, and exportable).

I built DocuAgent to run an end-to-end pipeline from a single upload:
1) Document Parse (PDF/image → Markdown)
2) Schema generation (document-type aware extraction fields)
3) Information Extract (schema-based JSON)
4) Summary + learning pack (objectives, key concepts, quiz, flashcards, activities)
5) Document-grounded Q&A
6) Export: Markdown/HTML/PDF + SCORM 1.2/2004 + IMS CC 1.1/1.3 ZIP packages

Engineering decisions / troubleshooting I worked through
- Reviewer-friendly demo mode: the full UI runs without paid keys (DOCUAGENT_DEMO_MODE=1) and exposes /healthz + sample buttons for fast reproduction.
- PDF/image robustness: poppler(pdftoppm) preferred, with a pypdfium2 fallback; images normalized to PNG for stable downstream extraction.
- Structured-output hardening: best-effort JSON extraction + repair so the UI consistently receives parseable objects.
- LMS realism: exports are minimal but valid, with manifests and an import checklist/log template for validation.

Repo:
https://github.com/KIM3310/Upstage-DocuAgent

