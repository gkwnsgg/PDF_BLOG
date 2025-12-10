import unittest
from unittest.mock import MagicMock, patch
import fitz
from src.classifier import PageClassifier
from src.layout import LayoutEngine
from src.extractor import ContentExtractor

class TestMagazineProcessing(unittest.TestCase):

    def setUp(self):
        self.classifier = PageClassifier()
        self.layout = LayoutEngine()
        self.extractor = ContentExtractor("test_output")

    def create_mock_span(self, text, bbox, font="Arial", flags=0):
        return {
            "text": text,
            "bbox": bbox,
            "font": font,
            "flags": flags,
            "size": 12.0
        }

    def test_classifier_ad_vs_article(self):
        # Scenario 1: Ad (No watermark, sparse text)
        page = MagicMock()
        page.rect = fitz.Rect(0, 0, 600, 800)

        # Mock text blocks: Just some decorative text, no watermark
        mock_blocks = [
            {"lines": [{"spans": [
                self.create_mock_span("BUY NOW", (100, 100, 200, 120)),
                self.create_mock_span("BEST CAR", (100, 150, 200, 170))
            ]}]}
        ]
        page.get_text.return_value = {"blocks": mock_blocks}

        # Should be classified as Ad (no watermark)
        result = self.classifier.classify_region(page, page.rect)
        self.assertEqual(result['type'], 'ad')
        self.assertEqual(result['reason'], 'no_watermark')

    def test_classifier_article_with_watermark(self):
        # Scenario 2: Article (Has watermark "BOM | NEWS")
        page = MagicMock()
        page.rect = fitz.Rect(0, 0, 600, 800)

        mock_blocks = [
            {"lines": [{"spans": [
                self.create_mock_span("BOM | NEWS", (50, 20, 150, 40)), # Top band
                self.create_mock_span("The New BMW", (50, 100, 300, 120)),
                self.create_mock_span("Driving experience...", (50, 130, 300, 150))
            ]}]}
        ]
        page.get_text.return_value = {"blocks": mock_blocks}

        result = self.classifier.classify_region(page, page.rect)
        self.assertEqual(result['type'], 'article')
        self.assertIn('watermark', result['reason'])

    def test_classifier_mixed_language_decorative(self):
        # Scenario 3: Mixed language decorative check
        # Case A: English text, no Korean particles -> Should NOT be decorative if enough text
        spans_english = [self.create_mock_span("This is a long sentence about cars.", (0,0,10,10))] * 20
        self.assertFalse(self.classifier.is_decorative_page(spans_english))

        # Case B: Korean text with particles -> Should NOT be decorative
        spans_korean = [self.create_mock_span("자동차는 달린다.", (0,0,10,10))]
        self.assertFalse(self.classifier.is_decorative_page(spans_korean))

        # Case C: Sparse text, no particles -> Decorative
        spans_sparse = [self.create_mock_span("WOW", (0,0,10,10))]
        self.assertTrue(self.classifier.is_decorative_page(spans_sparse))

    def test_layout_column_splitting(self):
        # Scenario 4: Two columns
        # Col 1: x=50..150, Col 2: x=200..300
        spans = [
            self.create_mock_span("Col1-Line1", (50, 100, 150, 120)),
            self.create_mock_span("Col2-Line1", (200, 100, 300, 120)),
            self.create_mock_span("Col1-Line2", (50, 130, 150, 150)),
            self.create_mock_span("Col2-Line2", (200, 130, 300, 150))
        ]

        columns = self.layout.group_spans_into_columns(spans)
        self.assertEqual(len(columns), 2)
        # Check order: Col1 then Col2
        self.assertEqual(columns[0][0]['text'], "Col1-Line1")
        self.assertEqual(columns[1][0]['text'], "Col2-Line1")

    def test_extractor_text_over_image(self):
        # Scenario 5: Text Over Background Image
        page = MagicMock()
        page.rect = fitz.Rect(0, 0, 600, 800)
        rect = fitz.Rect(0, 0, 600, 800)

        # Mock Image Block (Large background)
        # Type 1 = Image
        img_block = {
            "type": 1,
            "bbox": (0, 0, 600, 800), # Full page
            "image": b"fake_bytes",
            "ext": "png"
        }

        # Mock Text Block
        txt_block = {
            "type": 0,
            "lines": [{"spans": [
                self.create_mock_span("Overlay Text", (100, 100, 300, 120))
            ]}]
        }

        page.get_text.return_value = {"blocks": [img_block, txt_block]}

        # Run extraction
        # Note: We need to patch open() since it tries to write files
        with patch("builtins.open", new_callable=MagicMock):
            content = self.extractor.extract_content(page, rect, 1)

        # Should contain text
        self.assertIn("Overlay Text", content)
        # Should contain Background Image markdown
        self.assertIn("![Background Image]", content)

if __name__ == '__main__':
    unittest.main()
