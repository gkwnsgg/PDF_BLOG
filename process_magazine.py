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
    output_dir.mkdir(exist_ok=True)

    print(f"Processing {pdf_path}...")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return

    classifier = PageClassifier()
    extractor = ContentExtractor(str(output_dir))
    uploader = WordPressUploader(str(output_dir))

    for page_num, page in enumerate(doc):
        real_page_num = page_num + 1
        print(f"Analyzing Page {real_page_num}...")

        # Split page into halves (Left/Right)
        # Note: Some pages might be single pages (Covers), but for now we assume spreads as per req.
        # Ideally we should detect spread vs single.
        # For this prototype, we'll try to treat the whole page first,
        # but the requirement says "Mostly 2-page spreads".
        # Let's split unconditionally for now, as typical magazine PDFs are spreads.
        # But wait, if it's a cover, it might be single.
        # Let's check aspect ratio?
        # A spread is usually wider than tall. A single page is taller than wide.

        is_spread = page.rect.width > page.rect.height

        regions = []
        if is_spread:
            l_rect, r_rect = classifier.get_page_halves(page)
            regions.append(('L', l_rect))
            regions.append(('R', r_rect))
        else:
            regions.append(('Single', page.rect))

        page_content = []

        for side, rect in regions:
            classification = classifier.classify_region(page, rect)
            print(f"  [{side}] Type: {classification['type']} ({classification['reason']})")

            if classification['type'] == 'article':
                # Extract content
                md = extractor.extract_content(page, rect, real_page_num)
                if md.strip():
                    page_content.append(md)

            elif classification['type'] == 'article_image':
                # It's an article page but mostly image/decorative.
                # We should probably extract the image.
                # For now, let's treat it as article to capture the image.
                 md = extractor.extract_content(page, rect, real_page_num)
                 if md.strip():
                    page_content.append(md)

        if page_content:
            full_text = "\n\n---\n\n".join(page_content)
            # Create a post for this page (or append to a running article?)
            # The requirement implies extracting "Articles".
            # An article might span multiple pages.
            # However, for this MVP, let's upload per-page-spread chunks
            # or we need a way to group them.
            # Given the complexity of multi-page grouping without Title matching,
            # we will output per-page content for now.

            title = f"Extracted Content - Page {real_page_num}"
            # Try to find a real title in the content? (First bold line?)

            uploader.upload_post(title, full_text)

    print("Done.")

if __name__ == "__main__":
    main()
