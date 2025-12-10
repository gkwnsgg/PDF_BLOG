import re
import fitz
from typing import List, Dict, Any, Tuple

# Regex patterns ported/refined from main.py
WM_TOP_PAT = re.compile(
    r"""^BOM\s*([|│\|l]\s*(NEW\s*CAR|NEWS|CAR|MOTORRAD|TRAVEL|GOLF|CULTURE))?$""",
    re.IGNORECASE
)
BOTTOM_NUM_RE = re.compile(r"\bBOM\s*\d{1,3}\b", re.IGNORECASE)
SECTION_LABELS_UP = {"TRAVEL", "GOLF", "CAR", "NEWS", "MOTORRAD", "CULTURE", "BMW", "MINI"}
_BANNER_ASCII_RE = re.compile(r"^[A-Z0-9\s\|\-\/\.\,]+$")
PAGE_NUM_RE = re.compile(r"^\d{1,3}$")

class PageClassifier:
    """
    Classifies a PDF page or region (rect) as 'Article', 'Ad', or 'Excluded'
    based on the presence of Watermarks/Page Numbers and Text Density.
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
        """
        Checks if text looks like a watermark banner (e.g. 'BOM | NEWS').
        """
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
        Classify a page as decorative if it lacks substantial narrative text.
        For Mixed English/Korean, we check for:
        1. Korean particles (strong signal for Korean text).
        2. English text density (stop words or sentence structure).

        If NEITHER is found, it's decorative/image-only.
        """
        if not spans:
            return True

        # 1. Korean Check
        particles = ["은", "는", "이", "가", "을", "를", "의", "에", "로", "으로"]
        korean_signal = False
        full_text = " ".join([s.get("text", "") for s in spans])

        if any(p in full_text for p in particles):
            korean_signal = True

        # 2. English Check (Basic Density)
        # If no Korean signal, check if we have enough English words.
        # We look for common English stop words or just a count of words > 3 chars.
        english_words = [w for w in full_text.split() if re.match(r"^[a-zA-Z]{3,}$", w)]
        english_signal = len(english_words) > 10 # Arbitrary threshold for "substantial" text

        if not korean_signal and not english_signal:
            return True

        return False

    def classify_region(self, page: fitz.Page, rect: fitz.Rect) -> Dict[str, Any]:
        """
        Analyzes a specific region (rect) of the page.
        Returns: { 'type': 'article'|'ad'|'excluded', 'reason': str }
        """
        # Extract text from the region
        text_blocks = page.get_text("dict", clip=rect, flags=fitz.TEXTFLAGS_SEARCH | fitz.TEXT_PRESERVE_LIGATURES).get("blocks", [])

        spans = []
        for b in text_blocks:
            if b.get("type", 0) != 0: continue
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    spans.append(s)

        if not spans:
             return {'type': 'ad', 'reason': 'empty_text'} # Or excluded? User said Ad = No watermark. Empty = Ad/Image.

        # 1. Check for Watermarks/Page Numbers (The "White List" Key)
        has_watermark = False
        page_h = page.rect.height

        # Check top 15% and bottom 15% for watermarks/page numbers
        top_y = rect.y0 + page_h * 0.15
        bot_y = rect.y1 - page_h * 0.15

        for s in spans:
            y = (s["bbox"][1] + s["bbox"][3]) / 2.0
            txt = s["text"].strip()
            if not txt: continue

            # Top Band Check
            if y <= top_y:
                if WM_TOP_PAT.match(txt) or "BOM" in txt.upper() or self._is_wm_banner_like(txt):
                    has_watermark = True
                    break

            # Bottom Band Check
            if y >= bot_y:
                 # Check for "BOM 123" or just "123"
                 if BOTTOM_NUM_RE.search(txt) or self._is_wm_banner_like(txt):
                    has_watermark = True
                    break
                 if PAGE_NUM_RE.fullmatch(txt): # Simple page number
                    has_watermark = True
                    break

        # 2. Classification Logic

        # Rule: Ads strictly exclude Watermarks & Page Numbers.
        # So, Has Watermark -> Likely Article (or TOC/Info).
        # No Watermark -> Likely Ad (or Cover).

        if has_watermark:
            # It's an Article candidate.
            # Secondary check: Is it just an Info/TOC page?
            # User said: "Info/TOC ... distinct structures (lists, dense data)".
            # For now, we assume if it has a watermark and "substantial text", it's an Article.

            if self.is_decorative_page(spans):
                # Has watermark but almost no text -> Likely an image-heavy Article page.
                # We should keep it as Article to extract the image.
                return {'type': 'article', 'reason': 'watermark_present'}
            else:
                return {'type': 'article', 'reason': 'watermark_and_text'}

        else:
            # No Watermark -> Ad, Cover, or Image-only.
            # User said: "Ads: Characterized by absence of both Watermarks and Page Numbers."
            # "Covers: Single-page layouts... no narrative text."
            # We treat all "No Watermark" pages as "Ad/Excluded" for the Article Extraction goal.

            return {'type': 'ad', 'reason': 'no_watermark'}
