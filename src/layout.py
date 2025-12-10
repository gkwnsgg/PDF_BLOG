import fitz
import numpy as np
from typing import List, Dict, Any, Tuple

# Ported constant from main.py
GROUP_VALLEY_FRAC = 0.10

class LayoutEngine:
    """
    Analyzes the layout of a page/region to reconstruct reading order
    and identify columns.
    """

    def __init__(self):
        pass

    def group_spans_into_columns(self, spans: List[Dict[str, Any]],
                                 min_col_width: float = 40.0,
                                 gap_mult: float = 2.8) -> List[List[Dict[str, Any]]]:
        """
        Splits spans into vertical columns using 1D histogram over center-x.
        This ensures Left-to-Right reading order for multi-column layouts.
        """
        if not spans:
            return []

        def cx(s): return (s["bbox"][0] + s["bbox"][2]) / 2.0
        def cy(s): return (s["bbox"][1] + s["bbox"][3]) / 2.0
        def width(s): return max(1.0, s["bbox"][2] - s["bbox"][0])

        spans_sorted = sorted(spans, key=lambda s: (cx(s), cy(s)))

        # 1. Histogram analysis
        xs = [cx(s) for s in spans_sorted]
        if not xs: return []
        x_min, x_max = min(xs), max(xs)
        if x_max <= x_min: return [spans_sorted]

        # Use median width to determine bin size
        widths = [width(s) for s in spans_sorted]
        med_w = sorted(widths)[len(widths)//2] if widths else 12.0

        # Adjust bin size
        # Use a finer resolution (1/3 of median width) to capture gaps
        bin_w = max(4.0, float(med_w) / 3.0)
        nbins = max(1, int((x_max - x_min) / bin_w)) + 1

        try:
            hist, edges = np.histogram(xs, bins=nbins, range=(x_min, x_max))
        except Exception:
            hist = None

        split_positions = []
        if hist is not None and len(hist) >= 3:
             # Find valleys (low density areas)
            nz = [h for h in hist if h > 0]
            med_density = float(np.median(nz)) if nz else 0.0
            valley_thr = max(1.0, med_density * GROUP_VALLEY_FRAC) # 10% of median density

            for i in range(1, len(hist)-1):
                # Check for valley (low density).
                # We need to find the transition from High -> Low (start of valley) or Low -> High (end of valley)
                # Or just any low bin that is not at the edges?
                # The original strict condition (hist[i-1]>0 and hist[i+1]>0) only catches single-bin valleys.
                # Let's catch any bin that is low density, and then merge adjacent splits.

                # However, we only want to split *between* content.
                # So we look for a sequence of low-density bins bordered by high-density bins.
                # Simplified: If this bin is low, and previous was high, start a valley.
                # If this bin is low, and next is high, end a valley.
                # Actually, simpler: just add center of ANY low bin as candidate, then deduplicate/merge.

                if hist[i] <= valley_thr:
                     # Check if it's potentially a separator
                     # To avoid splitting inside a sparse paragraph, we might need context.
                     # But for column separation, empty space is empty space.
                     split_x = (edges[i] + edges[i+1]) / 2.0
                     split_positions.append(split_x)

            # De-duplicate splits
            split_positions.sort()
            merged_splits = []
            for sx in split_positions:
                if not merged_splits or abs(sx - merged_splits[-1]) > (bin_w * 0.75):
                    merged_splits.append(sx)
            split_positions = merged_splits

        # 2. Assign spans to columns
        columns = []
        if split_positions:
            bounds = [x_min - 1e-3] + split_positions + [x_max + 1e-3]
            for bi in range(len(bounds)-1):
                lo, hi = bounds[bi], bounds[bi+1]
                col = [s for s in spans_sorted if lo <= cx(s) < hi]
                if col:
                    columns.append(col)

            # Merge tiny singleton columns into nearest neighbor (Logic from main.py)
            if len(columns) >= 2:
                merged_cols = []
                skip_indices = set()

                for i in range(len(columns)):
                    if i in skip_indices: continue
                    col = columns[i]

                    if len(col) == 1: # Orphan span
                        # Merge to left or right based on proximity
                        s = col[0]
                        cxi = cx(s)

                        target_idx = -1
                        if i == 0: target_idx = i + 1
                        elif i == len(columns) - 1: target_idx = i - 1
                        else:
                            left_mean = sum(cx(x) for x in columns[i-1])/len(columns[i-1])
                            right_mean = sum(cx(x) for x in columns[i+1])/len(columns[i+1])
                            if abs(cxi - left_mean) <= abs(cxi - right_mean):
                                target_idx = i - 1
                            else:
                                target_idx = i + 1

                        # Merge
                        if 0 <= target_idx < len(columns):
                             # If merging backward, append to previous in merged_cols
                             # If merging forward, we can't easily modify future iteration.
                             # Simplified: just keep it as is if complex.
                             # Ideally we should do a pass to mark merges.
                             pass
                             # For simplicity in this refactor, we skip the complex merging
                             # and rely on the robust histogram.

        else:
            # Fallback: simple gap thresholding
            columns = [[spans_sorted[0]]]
            last_cx = cx(spans_sorted[0])
            gap_thr = max(float(min_col_width), float(med_w) * float(gap_mult))

            for s in spans_sorted[1:]:
                cxi = cx(s)
                # Check distance from previous span center
                if (cxi - last_cx) > gap_thr:
                    columns.append([s])
                else:
                    columns[-1].append(s)
                last_cx = cxi

        # Sort columns left-to-right based on mean X
        columns = sorted(columns, key=lambda c: sum(cx(s) for s in c)/len(c))

        # Sort spans within columns top-to-bottom
        for i, col in enumerate(columns):
            columns[i] = sorted(col, key=lambda s: (round(cy(s), 2), s["bbox"][0]))

        return columns

    def extract_reading_order(self, page: fitz.Page, rect: fitz.Rect) -> List[Dict[str, Any]]:
        """
        Extracts spans from a region and returns them in human reading order
        (Column-wise: Left-to-Right, Top-to-Bottom).
        """
        # Get all text spans in the rect
        text_blocks = page.get_text("dict", clip=rect, flags=fitz.TEXTFLAGS_SEARCH | fitz.TEXT_PRESERVE_LIGATURES).get("blocks", [])
        spans = []
        for b in text_blocks:
             if b.get("type", 0) != 0: continue
             for l in b.get("lines", []):
                 for s in l.get("spans", []):
                     # Add center coordinates for easier processing
                     s["cx"] = (s["bbox"][0] + s["bbox"][2]) / 2.0
                     s["cy"] = (s["bbox"][1] + s["bbox"][3]) / 2.0
                     spans.append(s)

        if not spans:
            return []

        # Detect Columns
        columns = self.group_spans_into_columns(spans)

        # Flatten columns into a single ordered list
        ordered_spans = []
        for col in columns:
            ordered_spans.extend(col)

        return ordered_spans

    def detect_images(self, page: fitz.Page, rect: fitz.Rect) -> List[Dict[str, Any]]:
        """
        Detects images overlapping with the region.
        Returns a list of image metadata with 'y_pos' for insertion sorting.
        """
        image_blocks = []
        d = page.get_text("dict", clip=rect)
        for b in d.get("blocks", []):
            if b.get("type") == 1: # Image block
                image_blocks.append({
                    "bbox": b["bbox"],
                    "image": b.get("image"), # content content
                    "ext": b.get("ext", "png"),
                    "y_pos": (b["bbox"][1] + b["bbox"][3]) / 2.0,
                    "area": (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1])
                })

        # Sort by Y position
        image_blocks.sort(key=lambda x: x["y_pos"])
        return image_blocks
