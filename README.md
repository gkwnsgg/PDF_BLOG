# PDF to Blog Converter

## Overview
This project extracts specific "Article" content from magazine PDFs and uploads it to a blog (WordPress). It automates the extraction of narrative content while strictly excluding advertisements, covers, tables of contents, and other non-article elements.

## Architecture
- `src/`: Core logic
    - `classifier.py`: Classification logic (Article vs Ad).
    - `layout.py`: Reading order reconstruction and layout analysis.
    - `extractor.py`: Text and image extraction.
    - `uploader.py`: WordPress API integration (stub).
- `tests/`: Verification tests.
- `output/`: Results.

## Requirements
- Python 3.10+
- PyMuPDF
- Pillow
- pandas
- numpy
