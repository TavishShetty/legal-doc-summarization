# LegalDocs AI - Document Anonymizer & Summarizer

A web app for anonymizing and summarizing legal/personal documents using AI.

## Structure

- `frontend/`: Static files for GitHub Pages.
- `backend/`: Flask app for Render with AI model integration.

## Setup

1. **Front-End**: Push `frontend/` to GitHub Pages.
2. **Back-End**: Deploy `backend/` on Render with `gunicorn app:app`.

## Features

- Upload PDFs, DOCX, TXT for processing.
- Process sample documents (invoice, agreement, bank, medical).
- Anonymize and summarize with customizable options.
