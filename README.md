# Calgary Hacks 2026

## Topic 2: Preserve Today for Tomorrow: Why Archives Matter

## Inspiration
	- Our inspiration comes from when documents are being digitalized they’re oftentime not accessible by prioritizing a people first design, screen readers can’t read image only PDFs, search doesn’t work, and sharing is really messy. We wanted an archive tool that treats accessibility as the default—OCR for readability, text-to-speech, translation, and export formats so more people can actually use preserved information.

## What does Aion do?

1. Aion turns documents (photos/scans/PDFs) into a searchable, shareable archive with accessibility in mind.
2. Upload/capture documents and run OCR to extract text.
3. Store metadata (title, tags, date, source) for organization.
4. Provide a reader experience with tools like text-to-speech and export options (where supported).
5. Generate share access (links/codes) for easy viewing.
6. Maintain version history so updates don’t overwrite the past.

## How Aion was built?
- We built Aion as a simple pipeline:

1. Capture/Upload a file (image/PDF/text)

2. Preprocess (clean up scans, handle PDFs)

3. Extract text using OCR / PDF text extraction

4. Store the file + extracted text + metadata in a database

5. Index/Search so users can find documents by content + tags

6. Serve a UI to browse libraries, view records, and share them


# Aion (Flask)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
	- `pip install -r requirements.txt`

## Run
- `python app.py`

Open http://127.0.0.1:5000/ in your browser.

