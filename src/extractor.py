import os
import fitz
import re
from pathlib import Path
from typing import List, Dict, Any
from src.layout import LayoutEngine
from src.classifier import WM_TOP_PAT, BOTTOM_NUM_RE, SECTION_LABELS_UP, _BANNER_ASCII_RE

class ContentExtractor:
    """
    Extracts content (text and images) from a page region and formats it as Markdown.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.layout_engine = LayoutEngine()

    def _save_image(self, page: fitz.Page, img_meta: Dict[str, Any], page_num: int, img_idx: int) -> str:
        """
        Extracts and saves an image to disk. Returns the relative path.
        """
        try:
            # Try getting bytes from dictionary first
            image_data = img_meta.get("image")
            ext = img_meta.get("ext", "png")

            # If no bytes, check if we have a valid xref in the original block?
            # get_text("dict") doesn't always provide xref in the "image" field (it provides bytes).
            # But let's check robustness: if image_data is None/empty, we can't save.

            if not isinstance(image_data, bytes):
                # Fallback or just skip for this MVP if extraction is complex without XREF
                return ""

            filename = f"p{page_num:03d}_img{img_idx:02d}.{ext}"
            filepath = self.images_dir / filename

            with open(filepath, "wb") as f:
                f.write(image_data)

            return f"images/{filename}"
        except Exception as e:
            print(f"Error saving image: {e}")
            return ""

    def _is_artifact(self, text: str, y_pos: float, page_h: float, rect: fitz.Rect) -> bool:
        """
        Determines if a span text is an artifact (watermark, page number).
        Uses geometric bands + regex.
        """
        t = text.strip()
        if not t: return True

        # Check Top Band (12%)
        top_y = rect.y0 + page_h * 0.12
        if y_pos <= top_y:
            if WM_TOP_PAT.match(t) or "BOM" in t.upper():
                return True
            # Check banner-like
            up = t.upper().replace("│", "|").replace("︱", "|").replace("｜", "|")
            if up in SECTION_LABELS_UP or up.replace("L ", "") in SECTION_LABELS_UP:
                return True

        # Check Bottom Band (12%)
        bot_y = rect.y1 - page_h * 0.12
        if y_pos >= bot_y:
             if BOTTOM_NUM_RE.search(t):
                 return True

        return False

    def extract_content(self, page: fitz.Page, rect: fitz.Rect, page_num: int) -> str:
        """
        Returns a Markdown string of the content in the rect.
        """
        # 1. Get ordered text spans
        ordered_spans = self.layout_engine.extract_reading_order(page, rect)

        # 2. Get images
        images = self.layout_engine.detect_images(page, rect)

        # 3. Merge streams
        mixed_stream = []
        page_h = page.rect.height

        for s in ordered_spans:
            # Filter artifacts before adding
            y_center = (s["bbox"][1] + s["bbox"][3]) / 2.0
            if self._is_artifact(s["text"], y_center, page_h, rect):
                continue
            mixed_stream.append({'type': 'text', 'obj': s, 'y': s['bbox'][1]})

        for idx, img in enumerate(images):
            mixed_stream.append({'type': 'image', 'obj': img, 'y': img['y_pos'], 'idx': idx})

        # Sort by Y (primary) and X (secondary)
        mixed_stream.sort(key=lambda x: (x['y'], x['obj']['bbox'][0]))

        # 4. Generate Markdown
        md_output = []
        current_paragraph = []

        def flush_paragraph():
            if current_paragraph:
                text = " ".join(current_paragraph).replace("  ", " ")
                md_output.append(text + "\n\n")
                current_paragraph.clear()

        for item in mixed_stream:
            if item['type'] == 'image':
                flush_paragraph()
                img_path = self._save_image(page, item['obj'], page_num, item['idx'])
                if img_path:
                    md_output.append(f"![Image]({img_path})\n\n")
            else:
                span = item['obj']
                text = span['text'].strip()
                if not text: continue

                font_flags = span.get('flags', 0)
                if font_flags & 16: # Bold
                    text = f"**{text}**"

                current_paragraph.append(text)

        flush_paragraph()

        return "".join(md_output)
