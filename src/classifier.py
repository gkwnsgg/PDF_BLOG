import re
import fitz
from typing import List, Dict, Any, Tuple

# Constants ported from main.py
WM_TOP_PAT = re.compile(
    r"""^BOM\s*([|│\|l]\s*(NEW\s*CAR|NEWS|CAR|MOTORRAD|TRAVEL|GOLF|CULTURE))?$""",
    re.IGNORECASE
)
BOTTOM_NUM_RE = re.compile(r"\bBOM\s*\d{1,3}\b", re.IGNORECASE)
SECTION_LABELS_UP = {"TRAVEL", "GOLF", "CAR", "NEWS", "MOTORRAD", "CULTURE", "BMW", "MINI"}
_BANNER_ASCII_RE = re.compile(r"^[A-Z0-9\s\|\-\/\.\,]+$")

class PageClassifier:
    """
    Classifies a PDF page or region (rect) as 'Article', 'Ad', or 'Excluded'.
    """

    def __init__(self):
        pass

    def _upper_ratio(self, s: str) -> float:
        letters = [ch for ch in s if ch.isalpha()]
        if not letters:
            return 0.0
        uppers = [ch for ch in letters if ch.isupper()]
        return (len(uppers) / len(letters)) if letters else 0.0

    def _is_wm_banner_like(self, raw_text: str) -> bool:
        t = (raw_text or "").strip()
        if not t:
            return False
        up = t.upper().replace("│", "|").replace("︱", "|").replace("｜", "|")
        if up == "BOM" or re.fullmatch(r"BOM\s*\d{1,3}", up):
            return True
        if up.replace("L ", "") in SECTION_LABELS_UP:
            return True
        if "BOM|" in up or "|BOM" in up or "BOM |" in up:
            return True
        if _BANNER_ASCII_RE.match(up) and (self._upper_ratio(up) >= 0.8) and (("|" in up) or ("/" in up) or (" -" in up) or ("—" in up)) and (len(up) <= 40):
            return True
        return False

    def get_page_halves(self, page: fitz.Page) -> Tuple[fitz.Rect, fitz.Rect]:
        """Splits a page into Left and Right rectangles."""
        R = page.rect
        xm = (R.x0 + R.x1) / 2.0
        return fitz.Rect(R.x0, R.y0, xm, R.y1), fitz.Rect(xm, R.y0, R.x1, R.y1)

    def is_decorative_page(self, spans: List[Dict[str, Any]]) -> bool:
        """
        Classify a page as decorative if none of the spans contain Korean case-marking particles.
        """
        if not spans:
            return True # No text means likely image-only or decorative
        particles = ["은", "는", "이", "가", "을", "를"]
        for s in spans:
            txt = s.get("text", "")
            if any(p in txt for p in particles):
                return False
        return True

    def classify_region(self, page: fitz.Page, rect: fitz.Rect) -> Dict[str, Any]:
        """
        Analyzes a specific region (rect) of the page.
        Returns a dictionary with classification results:
        {
            'type': 'article' | 'ad' | 'decorative',
            'confidence': float,
            'details': ...
        }
        """
        # Extract text from the region
        text_blocks = page.get_text("dict", clip=rect, flags=fitz.TEXTFLAGS_SEARCH | fitz.TEXT_PRESERVE_LIGATURES).get("blocks", [])

        spans = []
        for b in text_blocks:
            if b.get("type", 0) != 0: continue
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    spans.append(s)

        # 1. Check for Watermarks/Page Numbers (Presence suggests Article, Absence suggests Ad)
        # However, Ads explicitly LACK these.
        # We look for "BOM" markers or page numbers in top/bottom bands.

        has_watermark = False
        page_h = page.rect.height

        # Check top 12% and bottom 12% for watermarks
        top_y = rect.y0 + page_h * 0.12
        bot_y = rect.y1 - page_h * 0.12

        for s in spans:
            y = (s["bbox"][1] + s["bbox"][3]) / 2.0
            txt = s["text"].strip()

            if y <= top_y:
                if WM_TOP_PAT.match(txt) or "BOM" in txt.upper() or self._is_wm_banner_like(txt):
                    has_watermark = True
                    break
            if y >= bot_y:
                 if BOTTOM_NUM_RE.search(txt) or self._is_wm_banner_like(txt):
                    has_watermark = True
                    break

        # 2. Text Density & Particles
        is_decorative = self.is_decorative_page(spans)

        # Heuristic Logic
        # If it has watermarks/page numbers, it's likely part of the magazine flow (Article or TOC).
        # If it lacks them AND has low text density or no particles, it's likely an Ad or Image Page.
        # If it has text but no particles, it's "decorative" (Article w/o body text, or Ad).

        # Article Detection Strong Signal: Consistent Title Font?
        # (This usually requires comparing against other pages, but for single page we use heuristics)

        if not spans:
             return {'type': 'ad', 'reason': 'empty'}

        if has_watermark:
            if is_decorative:
                # Watermark present but no particles -> Likely TOC, Intro, or Image-heavy Article page
                return {'type': 'article_image', 'reason': 'watermark_but_decorative'}
            else:
                return {'type': 'article', 'reason': 'watermark_and_text'}
        else:
            # No watermark
            if is_decorative:
                return {'type': 'ad', 'reason': 'no_watermark_decorative'}
            else:
                # Text present but no watermark. Could be a full-page Ad with text.
                # Or a full-bleed article page (rare to have NO page number).
                # Ads usually don't have page numbers.
                return {'type': 'ad', 'reason': 'no_watermark'}
