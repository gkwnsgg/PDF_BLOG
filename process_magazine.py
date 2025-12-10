import argparse
import fitz
import os
from pathlib import Path
from src.classifier import PageClassifier
from src.extractor import ContentExtractor
from src.uploader import WordPressUploader

def main():
    parser = argparse.ArgumentParser(description="PDF to Blog Converter")
    parser.add_argument("pdf_path", help="Path to the magazine PDF")
    parser.add_argument("--output", default="output", help="Output directory")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    output_dir = Path(args.output)

    # Create subdirectories
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "articles").mkdir(parents=True, exist_ok=True)

    print(f"Processing {pdf_path}...")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    classifier = PageClassifier()
    extractor = ContentExtractor(str(output_dir))
    # uploader = WordPressUploader(str(output_dir)) # Disabled for now, focusing on extraction

    for page_num, page in enumerate(doc):
        real_page_num = page_num + 1
        print(f"Analyzing Page {real_page_num}...")

        # Check aspect ratio to determine if spread or single
        is_spread = page.rect.width > page.rect.height

        regions = []
        if is_spread:
            l_rect, r_rect = classifier.get_page_halves(page)
            regions.append(('L', l_rect))
            regions.append(('R', r_rect))
        else:
            regions.append(('Single', page.rect))

        page_markdown = []

        for side, rect in regions:
            classification = classifier.classify_region(page, rect)
            print(f"  [{side}] Type: {classification['type']} ({classification['reason']})")

            # White List Logic: Only process 'article' types
            if classification['type'] == 'article':
                content = extractor.extract_content(page, rect, real_page_num)
                if content.strip():
                     page_markdown.append(f"### Page {real_page_num} ({side})\n\n{content}")

        if page_markdown:
            full_text = "\n\n---\n\n".join(page_markdown)

            # Save to Markdown file
            md_filename = f"article_p{real_page_num:03d}.md"
            with open(output_dir / "articles" / md_filename, "w", encoding="utf-8") as f:
                f.write(full_text)

            print(f"  -> Extracted to {md_filename}")

    print("Done.")

if __name__ == "__main__":
    main()
