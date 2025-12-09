import fitz
import pytest
from src.layout import LayoutEngine

@pytest.fixture
def layout_engine():
    return LayoutEngine()

def test_column_detection(layout_engine):
    # Create artificial spans simulating two columns with a significant gap
    # Column 1: x=10..40 (width 30)
    # Column 2: x=100..130 (width 30)
    # Gap = 60 (larger than default min_col_width 40)
    spans = []

    # Col 1 spans
    for y in range(10, 100, 10):
        spans.append({"bbox": (10, y, 40, y+5), "text": f"Col1-Line{y}"})

    # Col 2 spans
    for y in range(10, 100, 10):
        spans.append({"bbox": (100, y, 130, y+5), "text": f"Col2-Line{y}"})

    columns = layout_engine.group_spans_into_columns(spans)

    assert len(columns) == 2

    # Check content of Col 1
    col1_texts = [s["text"] for s in columns[0]]
    assert "Col1-Line10" in col1_texts

    # Check content of Col 2
    col2_texts = [s["text"] for s in columns[1]]
    assert "Col2-Line10" in col2_texts

def test_reading_order_integration(layout_engine):
    # Mock FitZ page not easily possible without a file,
    # but we can test the internal sorting logic if we mock the input of group_spans_into_columns
    pass
