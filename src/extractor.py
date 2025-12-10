import os
import fitz
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.layout import LayoutEngine
from src.classifier import WM_TOP_PAT, BOTTOM_NUM_RE, SECTION_LABELS_UP, _BANNER_ASCII_RE

class ContentExtractor:
    """
    Extracts content (text and images) from a page region and formats it as Markdown.
    Handles "Text-Over-Image" by detecting background images.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.layout_engine = LayoutEngine()

    def _save_image(self, img_meta: Dict[str, Any], page_num: int, img_idx: int) -> str:
        """
        Extracts and saves an image to disk. Returns the relative path.
        """
        try:
            image_data = img_meta.get("image")
            ext = img_meta.get("ext", "png")

            if not isinstance(image_data, bytes):
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
        """
        t = text.strip()
        if not t: return True

        # Check Top Band (15%)
        top_y = rect.y0 + page_h * 0.15
        if y_pos <= top_y:
            if WM_TOP_PAT.match(t) or "BOM" in t.upper():
                return True
            up = t.upper().replace("│", "|").replace("︱", "|").replace("｜", "|")
            if up in SECTION_LABELS_UP or up.replace("L ", "") in SECTION_LABELS_UP:
                return True

        # Check Bottom Band (15%)
        bot_y = rect.y1 - page_h * 0.15
        if y_pos >= bot_y:
             if BOTTOM_NUM_RE.search(t):
                 return True
             if re.fullmatch(r"\d{1,3}", t):
                 return True

        return False

    def _dedup_overlapping_spans(self, spans: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        """
        Removes duplicate text spans (common in PDFs with shadow layers).
        Ported simplified logic from main.py
        """
        if not spans: return spans

        # Sort by Y, then X
        srt = sorted(spans, key=lambda s: (s["bbox"][1], s["bbox"][0]))

        keep = []
        for s in srt:
            if not keep:
                keep.append(s)
                continue

            last = keep[-1]
            # Check intersection
            def get_area(b): return (b[2]-b[0])*(b[3]-b[1])
            def get_inter(a, b):
                ix0 = max(a[0], b[0]); iy0 = max(a[1], b[1])
                ix1 = min(a[2], b[2]); iy1 = min(a[3], b[3])
                return max(0, ix1-ix0) * max(0, iy1-iy0)

            bbox_s = s["bbox"]
            bbox_l = last["bbox"]

            inter = get_inter(bbox_s, bbox_l)
            area_s = get_area(bbox_s)

            # If significant overlap (same text or shadow)
            if inter > 0.5 * area_s:
                if s["text"].strip() == last["text"].strip():
                    continue # Duplicate

            keep.append(s)

        return keep

    def extract_content(self, page: fitz.Page, rect: fitz.Rect, page_num: int) -> str:
        """
        Returns a Markdown string of the content in the rect.
        Handles Text-Over-Image by extracting background images separately.
        """
        page_h = page.rect.height
        page_area = (rect.x1 - rect.x0) * (rect.y1 - rect.y0)

        # 1. Get ordered text spans
        ordered_spans = self.layout_engine.extract_reading_order(page, rect)
        ordered_spans = self._dedup_overlapping_spans(ordered_spans)

        # 2. Get images
        images = self.layout_engine.detect_images(page, rect)

        # 3. Detect Background Images (Text-Over-Image)
        # Rule: If an image covers > 60% of the logical page area, treat it as background.
        background_images = []
        inline_images = []

        for img in images:
            if img["area"] > (page_area * 0.60):
                background_images.append(img)
            else:
                inline_images.append(img)

        md_output = []

        # Process Background Images first (Output them at the top or bottom, or just note them)
        # Requirement: "background image extracted separately"
        for idx, img in enumerate(background_images):
            img_path = self._save_image(img, page_num, 90 + idx) # Use high index for bg
            if img_path:
                md_output.append(f"![Background Image]({img_path})\n\n")

        # 4. Merge Inline Images and Text
        mixed_stream = []

        for s in ordered_spans:
            # Filter artifacts
            y_center = (s["bbox"][1] + s["bbox"][3]) / 2.0
            if self._is_artifact(s["text"], y_center, page_h, rect):
                continue
            mixed_stream.append({'type': 'text', 'obj': s, 'y': s['bbox'][1]})

        for idx, img in enumerate(inline_images):
            mixed_stream.append({'type': 'image', 'obj': img, 'y': img['y_pos'], 'idx': idx})

        # Sort by Y (primary) and X (secondary)
        mixed_stream.sort(key=lambda x: (x['y'], x['obj']['bbox'][0]))

        current_paragraph = []

        def flush_paragraph():
            if current_paragraph:
                text = " ".join(current_paragraph).replace("  ", " ")
                # Fix spacing around punctuation
                text = text.replace(" ,", ",").replace(" .", ".")
                md_output.append(text + "\n\n")
                current_paragraph.clear()

        for item in mixed_stream:
            if item['type'] == 'image':
                flush_paragraph()
                img_path = self._save_image(item['obj'], page_num, item['idx'])
                if img_path:
                    md_output.append(f"![Image]({img_path})\n\n")
            else:
                span = item['obj']
                text = span['text'].strip()
                if not text: continue

                # Check for headers (Bold + Large?)
                # Simplified markdown formatting
                font_flags = span.get('flags', 0)
                is_bold = (font_flags & 16) or (font_flags & 2) # fitz flags vary

                if is_bold:
                    text = f"**{text}**"

                current_paragraph.append(text)

        flush_paragraph()

        return "".join(md_output)
