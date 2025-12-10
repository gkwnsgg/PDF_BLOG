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

## Usage
The refactored extraction logic is available via `process_magazine.py`.

```bash
python3 process_magazine.py <path_to_pdf> --output <output_directory>
```

Example:
```bash
python3 process_magazine.py magazine.pdf --output my_blog_content
```

This will create:
- `my_blog_content/articles/`: Extracted text (Markdown).
- `my_blog_content/images/`: Extracted images.

### Key Features
- **Ad Exclusion**: Automatically filters out pages without watermarks/page numbers.
- **Reading Order**: Reconstructs Left-to-Right columns from scrambled PDF text.
- **Text-Over-Image**: Extracts background images separately from text overlays.
- **Mixed Language**: Handles both English and Korean content.
