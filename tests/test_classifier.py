import fitz
import pytest
from src.classifier import PageClassifier

@pytest.fixture
def classifier():
    return PageClassifier()

def test_page_halves(classifier):
    # Create a dummy page 100x100
    doc = fitz.open()
    page = doc.new_page(width=100, height=100)

    l_rect, r_rect = classifier.get_page_halves(page)

    assert l_rect.x0 == 0
    assert l_rect.x1 == 50
    assert r_rect.x0 == 50
    assert r_rect.x1 == 100

def test_banner_detection(classifier):
    assert classifier._is_wm_banner_like("BOM")
    assert classifier._is_wm_banner_like("BOM 12")
    assert classifier._is_wm_banner_like("TRAVEL")
    assert classifier._is_wm_banner_like("GOLF")
    assert not classifier._is_wm_banner_like("This is a normal sentence.")

def test_decorative_check(classifier):
    # Korean particles: "은", "는", "이", "가", "을", "를"
    spans_article = [{"text": "자동차는 달린다."}]
    spans_ad = [{"text": "LUXURY BRAND NEW ARRIVAL"}]

    assert not classifier.is_decorative_page(spans_article) # Has particle -> Not decorative (Article)
    assert classifier.is_decorative_page(spans_ad)      # No particle -> Decorative (Ad)
