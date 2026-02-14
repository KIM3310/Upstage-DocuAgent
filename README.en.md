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

### 2) Set API key
```
export UPSTAGE_API_KEY="your_api_key_here"
```
Or use .env
If you skip the key, DocuAgent runs in **demo mode** (stubbed outputs) so you can still try the UI end-to-end.

### 3) Run
```
python main.py
```
Open http://localhost:8000

## Usage
1) Upload a document
2) Select audience level and learning goal
3) Enter LOM tags (optional)
4) Review results
5) Export to Markdown/HTML/PDF/SCORM/IMS
6) Ask questions about the document

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
- main.py: FastAPI backend, Upstage integration, package builders
- index.html: frontend UI
- requirements.txt: dependencies
- LMS_IMPORT_CHECKLIST.md: LMS import checklist
- LMS_IMPORT_LOG_TEMPLATE.md: issue log template

## Notes
- PDF input requires poppler (pdftoppm) or pypdfium2+pillow for conversion.
