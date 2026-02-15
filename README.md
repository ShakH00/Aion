# Calgary Hacks 2026

## Topic 2: Preserve Today for Tomorrow: Why Archives Matter

# Getting Started with Aion (Flask)

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
	- `pip install -r requirements.txt`

## Run
- `python app.py`

Open http://127.0.0.1:5000/ in your browser.

## Inspiration
- Our inspiration comes from when documents are being digitalized they’re oftentime not accessible by prioritizing a people first design, screen readers can’t read image only PDFs, search doesn’t work, and sharing is really messy. We wanted an archive tool that treats accessibility as the default—OCR for readability, text-to-speech, translation, and export formats so more people can actually use preserved information.

## What does Aion do?

1. Aion turns documents (photos/scans/PDFs) into a searchable, shareable archive with accessibility in mind.
2. Upload/capture documents and run OCR to extract text.
3. Store metadata (title, tags, date, source) for organization.
4. Provide a reader experience with tools like text-to-speech and export options (where supported).
5. Generate share access (links/codes) for easy viewing.
6. Maintain version history so updates don’t overwrite the past.

## How Aion was built

We built Aion as a simple pipeline:

1. Capture/Upload a file (image/PDF/text)
2. Preprocess (clean up scans, handle PDFs)
3. Extract text using OCR / PDF text extraction
4. Store the file + extracted text + metadata in a database
5. Index/Search so users can find documents by content + tags
6. Serve a UI to browse libraries, view records, and share them

## Challenges we ran into

One of our biggest challenges was **scoping and differentiation**. We initially committed to **Topic 3**, developing a game around the prompt **“Impermanence.”** During early prototyping, we realized our concept wasn’t distinct enough and risked being too surface-level.

We made the decision to pivot to **Topic 2**, where the challenge of preserving and accessing archives offered deeper technical and design problems. The pivot forced rapid re-planning under time pressure, but it ultimately improved the clarity, usefulness, and impact of our final build.

We also ran into a number of issues while integrating the **Raspberry Pi** into our project. Hardware introduced a lot of uncertainty—device setup, connectivity, and reliably triggering our capture/scan flow—so we spent time debugging ports, dependencies, and inconsistent behavior across machines. Getting the Pi to work smoothly with the rest of our pipeline was a challenge, especially under hackathon time constraints.

## Accomplishments that we're proud of

- Shipped a full pipeline from scanning to search in a hackathon timeframe
- Made documents searchable through OCR + metadata
- Built sharing + version history to keep archives useful and trustworthy
- Prioritized accessibility features instead of treating them as an afterthought
- Successfully integrated Raspberry Pi scanning into the workflow (despite hardware pain)

## What we learned

- **Scoping matters as much as coding**: choosing the right problem early (and pivoting when needed) made the project stronger.
- **OCR is only as good as the input**: scan quality, lighting, and preprocessing heavily affect accuracy.
- **Data + metadata design is everything**: storing files is easy—making them searchable and organized requires a clean model.
- **Hardware adds uncertainty**: integrating the Raspberry Pi introduced debugging time we didn’t fully anticipate (connectivity, ports, reliability).
- **Accessibility can’t be an afterthought**: building for real users means designing for readability and multiple ways to consume information.

## What's next for Aion

- Better scanning + OCR accuracy
- Multi-page workflows and bulk uploads
- Smarter search + filters
- Version diffs and audit trails
- Stronger accessibility and export options
- Cleaner Raspberry Pi setup and reliability


