#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf_to_blog_v3.py — 이미지 캡처 + 제목 추출 + 본문 추출(워터마크 완전 배제 & exclude 일치)

- 워터마크 스캔/템플릿 매칭 로직: 제거
- exclude(예: 9pr, 37p, 40pl): 캡처/제목/본문 모두 제외
- 제목 추출 규칙:
  * 허용된 반쪽(rect) 내부 텍스트만 사용
  * 상단 스트립에서 워터마크(예: "BOM | CAR/NEWS/...","BOM") 글꼴 크기 측정 → wm_max_pt
  * --title-base-size 지정 시: [기준값 ± title-size-tol] 범위의 글꼴만 제목 후보
    - WM 패턴 및 'BOM' 문구는 무조건 제외
    - 줄바꿈/굵기 섞여도 결합 (line-join-gap, block-down-tol)
    - 이중 레이어 중복 제거
- page_decisions.csv: title_side/title_text/title_max_pt/wm_max_pt/excluded/exclude_reason 기록
  + title_base_size, title_size_tol 기록
- page_articles.csv: 기사 본문 텍스트(제목/워터마크/장식문구는 제외, exclude 페이지 제외)
"""

import re, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

 # Global gap log for diagnostics (set in main)
GAP_LOG: list = []
# Global gap log counter for per-(page,side) cap
GAP_LOG_COUNT: dict = {}

# Column valley threshold (fraction of median non-zero bin count) – can be overridden by --col-valley-frac
GROUP_VALLEY_FRAC = 0.10

import fitz  # PyMuPDF
import pandas as pd
import numpy as np

# Optional pdfplumber import for rescue engine
try:
    import pdfplumber  # optional rescue engine for lead-band ordering
except Exception:
    pdfplumber = None

WM_TOP_PAT = re.compile(
    r"""^BOM\s*([|│\|l]\s*(NEW\s*CAR|NEWS|CAR|MOTORRAD|TRAVEL|GOLF|CULTURE))?$""",
    re.IGNORECASE
)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_excludes(tokens: List[str]) -> Dict[int, str]:
    """'7pr' → {7:'R'}, '37p' → {37:'ALL'}"""
    out={}
    for t in tokens:
        t=t.strip().lower()
        m=re.match(r"^(\d+)\s*(p|pl|pr)?$", t)
        if not m: continue
        page=int(m.group(1))
        suf=m.group(2)
        if suf=="pl": out[page]="L"
        elif suf=="pr": out[page]="R"
        else: out[page]="ALL"
    return out

def page_halves_rect(page: fitz.Page) -> Tuple[fitz.Rect, fitz.Rect]:
    R=page.rect
    xm=(R.x0+R.x1)/2.0
    return fitz.Rect(R.x0,R.y0,xm,R.y1), fitz.Rect(xm,R.y0,R.x1,R.y1)

def render_region_png(page: fitz.Page, rect: fitz.Rect, scale: float) -> bytes:
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    return pix.tobytes("png")

def gather_spans_in_rect(page: fitz.Page, rect: fitz.Rect) -> List[Dict[str,Any]]:
    d = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH | fitz.TEXT_PRESERVE_LIGATURES)
    out=[]
    for b in d.get("blocks", []):
        if b.get("type",0)!=0: continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                txt=(s.get("text") or "").strip()
                if not txt: continue
                bb=s.get("bbox") or l.get("bbox") or b.get("bbox")
                cx=(bb[0]+bb[2])/2.0; cy=(bb[1]+bb[3])/2.0
                if rect.contains(fitz.Point(cx,cy)):
                    out.append({
                        "text": txt,
                        "size": float(s.get("size",0.0)),
                        "flags": int(s.get("flags",0)),
                        "font": s.get("font",""),
                        "bbox": tuple(bb)
                    })
    return out


def gather_words_in_rect(page: fitz.Page, rect: fitz.Rect) -> List[Dict[str, Any]]:
    words = page.get_text("words") or []
    out = []
    rx0, ry0, rx1, ry1 = rect
    for w in words:
        x0, y0, x1, y1, t, bno, lno, wno = w
        if x0 >= rx1 or x1 <= rx0 or y0 >= ry1 or y1 <= ry0:
            continue
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if not rect.contains(fitz.Point(cx, cy)):
            continue
        t = (t or "").strip()
        if not t:
            continue
        out.append({
            "text": t,
            "bbox": (float(x0), float(y0), float(x1), float(y1)),
            "block": int(bno),
            "line": int(lno),
            "word": int(wno),
        })
    out.sort(key=lambda d: (d["block"], d["line"], d["word"]))
    return out

def _bbox_intersection(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

def annotate_words_with_span_attrs(words: List[Dict[str, Any]],
                                   spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not words:
        return words
    if not spans:
        for w in words:
            w["size"] = 0.0
            w["flags"] = 0
            w["font"] = ""
        return words

    for w in words:
        wb = w["bbox"]
        best_iou = 0.0
        best = None
        for s in spans:
            sb = s.get("bbox", (0,0,0,0))
            inter = _bbox_intersection(wb, sb)
            if inter <= 0.0:
                continue
            sa = max(1.0, (sb[2]-sb[0])*(sb[3]-sb[1]))
            wa = max(1.0, (wb[2]-wb[0])*(wb[3]-wb[1]))
            iou = inter / (sa + wa - inter)
            if iou > best_iou:
                best_iou = iou
                best = s
        if best is None:
            cx = (wb[0]+wb[2])/2.0; cy=(wb[1]+wb[3])/2.0
            for s in spans:
                sb = s.get("bbox", (0,0,0,0))
                if sb[0] <= cx <= sb[2] and sb[1] <= cy <= sb[3]:
                    best = s; break
        if best is not None:
            w["size"] = float(best.get("size", 0.0) or 0.0)
            w["flags"] = int(best.get("flags", 0))
            w["font"] = best.get("font", "")
        else:
            w["size"] = 0.0
            w["flags"] = 0
            w["font"] = ""
    return words

# --- 단어 레벨 중복 제거 (겹친 텍스트레이어 제거) ---
def dedup_words_by_overlap(words: List[Dict[str, Any]],
                           iou_tol: float = 0.85,
                           center_tol: float = 1.0) -> List[Dict[str, Any]]:
    """
    Collapse nearly-identical word boxes (duplicate text layers).
    - Prefer the visually stronger style (larger size), BUT
    - Preserve the earliest stream position (min of block/line/word) to avoid reordering
      the first sentence of a paragraph to later positions.
    """
    # allow runtime override via attributes set in main()
    iou_tol = float(getattr(dedup_words_by_overlap, "_iou_tol", iou_tol))
    center_tol = float(getattr(dedup_words_by_overlap, "_center_tol", center_tol))
    if not words:
        return words

    def _cx(b): return (b[0] + b[2]) / 2.0
    def _cy(b): return (b[1] + b[3]) / 2.0

    used = [False] * len(words)
    out: List[Dict[str, Any]] = []
    for i, wi in enumerate(words):
        if used[i]:
            continue
        bi = wi["bbox"]
        ti = (wi.get("text") or "").strip()
        # Track best visual style and earliest stream coords across the group
        best = wi
        earliest_block = wi.get("block", 0)
        earliest_line  = wi.get("line", 0)
        earliest_word  = wi.get("word", 0)
        used[i] = True
        for j in range(i + 1, len(words)):
            if used[j]:
                continue
            wj = words[j]
            tj = (wj.get("text") or "").strip()
            if not ti or ti != tj:
                continue
            bj = wj["bbox"]
            dx = abs(_cx(bi) - _cx(bj))
            dy = abs(_cy(bi) - _cy(bj))
            inter = _bbox_intersection(bi, bj)
            ai = max(1.0, (bi[2] - bi[0]) * (bi[3] - bi[1]))
            aj = max(1.0, (bj[2] - bj[0]) * (bj[3] - bj[1]))
            iou = inter / (ai + aj - inter) if (ai + aj - inter) > 0 else 0.0
            if (dx <= center_tol and dy <= center_tol) or (iou >= iou_tol):
                # Update best by larger font size (visual strength)
                if float(wj.get("size", 0.0) or 0.0) > float(best.get("size", 0.0) or 0.0):
                    best = wj
                # Track the earliest stream coordinates among duplicates
                earliest_block = min(earliest_block, wj.get("block", 0))
                earliest_line  = min(earliest_line,  wj.get("line", 0))
                earliest_word  = min(earliest_word,  wj.get("word", 0))
                used[j] = True

        # Preserve earliest stream order while keeping the best visual attributes
        best_mut = dict(best)
        best_mut["block"] = earliest_block
        best_mut["line"]  = earliest_line
        best_mut["word"]  = earliest_word
        out.append(best_mut)

    # Re-sort by stream order (now anchored to earliest of the group)
    out.sort(key=lambda d: (d.get("block", 0), d.get("line", 0), d.get("word", 0)))
    return out

def _build_text_from_stream_words(words: List[Dict[str, Any]],
                                  inline_gap_mult: float = 3.0) -> str:
    if not words:
        return ""
    lines: Dict[tuple, List[Dict[str,Any]]] = {}
    for w in words:
        key = (w["block"], w["line"])
        lines.setdefault(key, []).append(w)

    out_lines: List[str] = []
    for key in sorted(lines.keys()):
        ws = lines[key]
        ws.sort(key=lambda d: (d.get("word", 0), d["bbox"][0]))
        gaps = []
        for i in range(len(ws)-1):
            x0p, _, x1p, _ = ws[i]["bbox"]
            x0c, _, _, _ = ws[i+1]["bbox"]
            gaps.append(max(0.0, x0c - x1p))
        widths = [max(1.0, (w["bbox"][2]-w["bbox"][0])) for w in ws]
        med_w = sorted(widths)[len(widths)//2] if widths else 10.0
        pos_gaps = [g for g in gaps if g > 0.0]
        if pos_gaps:
            try:
                small_ref = float(np.percentile(pos_gaps, 20.0))
            except Exception:
                sg = sorted(pos_gaps)
                small_ref = float(sg[max(0, int(0.2*(len(sg)-1)))])
        else:
            small_ref = 0.0
        # make splitting less aggressive so words on the same visual line stay together
        base_thr = max(6.0, med_w * 1.4)
        dyn_thr = small_ref * float(inline_gap_mult) if small_ref > 0.0 else 0.0
        thr = max(base_thr, dyn_thr)

        chunks: List[List[Dict[str,Any]]] = []
        cur: List[Dict[str,Any]] = []
        for w in ws:
            if not cur:
                cur.append(w); continue
            prev = cur[-1]
            gap = max(0.0, w["bbox"][0] - prev["bbox"][2])
            if gap >= thr:
                chunks.append(cur); cur = [w]
            else:
                cur.append(w)
        if cur:
            chunks.append(cur)

        joined_chunks = [" ".join(x["text"] for x in ch).strip() for ch in chunks if ch]
        line_text = "\n".join([c for c in joined_chunks if c])
        if line_text:
            out_lines.append(line_text)
    return "\n".join(out_lines).strip()

# --- collapse adjacent duplicate tokens within a line (stutter/echo) ---
def _collapse_stutter_lines(text: str, min_pairs: int = 2) -> str:
    """
    Collapse adjacent duplicated tokens within a line (e.g., '프로 프로즌' → '프로즌' does not apply,
    but '오더는 오더는 오더는' → '오더는'). Also handles half-echo duplication where the second
    half of a line exactly repeats the first half.
    Heuristics:
      - Only trigger when a line contains at least `min_pairs` adjacent duplicate pairs.
      - Tokenization is whitespace-based to work robustly with CJK and Latin text.
    """
    if not text:
        return text
    out_lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            out_lines.append(raw)
            continue
        toks = line.split()
        n = len(toks)
        # half-echo: first half == second half → keep only first half
        if n >= 8 and (n % 2 == 0) and toks[: n // 2] == toks[n // 2 :]:
            toks = toks[: n // 2]
            n = len(toks)
        # count immediate duplicate pairs
        pairs = 0
        i = 0
        collapsed = []
        while i < n:
            if i + 1 < n and toks[i] == toks[i + 1]:
                pairs += 1
                # keep a single token
                collapsed.append(toks[i])
                # skip all identical repeats ahead
                j = i + 2
                while j < n and toks[j] == toks[i]:
                    j += 1
                i = j
            else:
                collapsed.append(toks[i])
                i += 1
        if pairs >= max(1, int(min_pairs)):
            out_lines.append(" ".join(collapsed))
        else:
            out_lines.append(raw)
    return "\n".join(out_lines)

# --- sentence reflow (safe splitting by punctuation) ---
def _reflow_sentences(text: str,
                      kind: str = "body",
                      min_len: int = 22,
                      allow_quote_split: bool = False,
                      keep_qa_prefix: bool = True) -> str:
    """
    Reflow lines into sentences. Split on '.', '!', '?', '…' with *fixed-width* lookbehind.
    - keep_qa_prefix: treat leading 'Q.' / 'A.' as a prefix (do NOT split there).
    - min_len: merge very short sentences with the next one to avoid choppy line breaks.
    - kind: 'body' (apply splitting), 'subtitle' (light touch; collapse spaces only).
    """
    if not text:
        return text

    # For subtitles, just squash spaces and return
    if kind == "subtitle":
        return re.sub(r"\s+", " ", text).strip()

    # Merge lines and collapse spaces for stable splitting
    merged = " ".join(ln.strip() for ln in text.splitlines() if ln.strip())
    merged = re.sub(r"\s+", " ", merged).strip()

    # Protect 'Q.' / 'A.' tokens so they aren't treated as sentence terminators.
    # We use a placeholder (§) temporarily and restore later.
    if keep_qa_prefix:
        merged = re.sub(r'(^|\s)([QqAa])\.', r'\1\2§', merged)

    # Fixed-width lookbehind on a single char is safe in Python's re engine.
    # Split AFTER ., !, ?, … followed by whitespace.
    parts = re.split(r'(?<=[\.!?…])\s+', merged)

    # Merge too-short segments with their successor to reduce over-breaking.
    result = []
    for seg in parts:
        if not seg:
            continue
        if not result:
            result.append(seg)
            continue
        if len(result[-1]) < max(1, int(min_len)):
            result[-1] = (result[-1] + " " + seg).strip()
        else:
            result.append(seg)

    out = "\n".join(s.strip() for s in result if s.strip())

    # Restore Q./A. markers
    if keep_qa_prefix:
        out = out.replace("Q§", "Q.").replace("A§", "A.")

    return out

# === 이중 레이어(겹침/유사 텍스트) 중복 스팬 제거 ===
def _bbox_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    iw = max(0.0, ix1 - max(ax0, bx0))
    inter = iw * ih
    if inter <= 0.0: return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _norm_text_for_cmp(t: str) -> str:
    return re.sub(r"\s+", "", (t or "").strip().lower())

def _token_jaccard(a: str, b: str) -> float:
    sa = set(re.sub(r"\W+", " ", a.lower()).split())
    sb = set(re.sub(r"\W+", " ", b.lower()).split())
    if not sa or not sb: return 0.0
    return len(sa & sb) / len(sa | sb)

def dedup_overlapping_spans(spans: List[Dict[str,Any]],
                            iou_tol: float = 0.85,
                            center_tol: float = 1.2,
                            text_sim_tol: float = 0.90) -> List[Dict[str,Any]]:
    if not spans: return spans
    def cy(s): bb=s["bbox"]; return (bb[1]+bb[3])/2.0
    def cx(s): bb=s["bbox"]; return (bb[0]+bb[2])/2.0
    srt = sorted(spans, key=lambda s:(round(cy(s),2), round(cx(s),2), -s.get("size",0)))

    used = [False]*len(srt)
    groups: List[List[int]] = []
    for i, si in enumerate(srt):
        if used[i]: continue
        gi = [i]; used[i] = True
        bb_i = si["bbox"]; cx_i=(bb_i[0]+bb_i[2])/2.0; cy_i=(bb_i[1]+bb_i[3])/2.0
        ti_norm = _norm_text_for_cmp(si["text"])
        for j in range(i+1, len(srt)):
            if used[j]: continue
            sj = srt[j]
            bb_j = sj["bbox"]; cx_j=(bb_j[0]+bb_j[2])/2.0; cy_j=(bb_j[1]+bb_j[3])/2.0
            dx = abs(cx_i - cx_j); dy = abs(cy_i - cy_j)
            iou = _bbox_iou(bb_i, bb_j)
            if (dx <= center_tol and dy <= center_tol) or (iou >= iou_tol):
                tj_norm = _norm_text_for_cmp(sj["text"])
                if ti_norm and tj_norm and (ti_norm == tj_norm or _token_jaccard(si["text"], sj["text"]) >= text_sim_tol):
                    gi.append(j); used[j] = True
        groups.append(gi)

    result: List[Dict[str,Any]] = []
    def score(span):
        size = float(span.get("size",0.0) or 0.0)
        bold = 1 if (int(span.get("flags",0)) & 2) != 0 else 0
        bb = span["bbox"]; height = (bb[3]-bb[1])
        return (size, bold, height)
    for gi in groups:
        best_idx = max(gi, key=lambda k: score(srt[k]))
        result.append(srt[best_idx])
    result.sort(key=lambda s:(round(cy(s),2), round(cx(s),2)))
    return result

def percentile(vals: List[float], p: float) -> float:
    if not vals: return 0.0
    vals=sorted(vals)
    k=max(0, min(len(vals)-1, int(round((p/100.0)*(len(vals)-1)))))
    return float(vals[k])

def detect_wm_font_size(spans: List[Dict[str,Any]], page_h: float, top_percent: float) -> float:
    y1 = page_h * (top_percent/100.0)
    cand=[s["size"] for s in spans
          if (s["bbox"][1] <= y1) and (WM_TOP_PAT.match(s["text"]) or "BOM" in s["text"].upper())]
    return max(cand) if cand else 0.0

def compute_body_stats(spans: List[Dict[str,Any]], page_h: float, top_percent: float) -> Dict[str, float]:
    y1 = page_h * (top_percent/100.0)
    body = [s for s in spans if s["bbox"][1] > y1]
    if not body:
        return {"all_p90": 0.0, "reg_p90": 0.0, "bold_p90": 0.0}
    sizes_all = [s["size"] for s in body]
    sizes_reg = [s["size"] for s in body if (int(s.get("flags",0)) & 2) == 0]
    sizes_bold = [s["size"] for s in body if (int(s.get("flags",0)) & 2) != 0]
    def p90(vs):
        if not vs: return 0.0
        vs = sorted(vs)
        k = max(0, min(len(vs)-1, int(round(0.90*(len(vs)-1)))))
        return float(vs[k])
    return {"all_p90": p90(sizes_all), "reg_p90": p90(sizes_reg), "bold_p90": p90(sizes_bold)}

def cluster_and_join(spans: List[Dict[str,Any]], join_gap: float, down_tol: float) -> str:
    if not spans: return ""
    def cy(s): bb=s["bbox"]; return (bb[1]+bb[3])/2.0
    spans=sorted(spans, key=lambda s:(round(cy(s),2), s["bbox"][0]))

    blocks=[[spans[0]]]
    for s in spans[1:]:
        prev=blocks[-1][-1]
        same_line = abs(cy(s)-cy(prev)) <= join_gap
        next_line = (cy(s) > cy(prev)) and ((cy(s)-cy(prev)) <= join_gap*down_tol)
        if same_line or next_line:
            blocks[-1].append(s)
        else:
            blocks.append([s])

    lines=[]
    # determine inline gap multiplier (default 3.0, configurable via attribute)
    inline_mult = getattr(cluster_and_join, "_inline_gap_mult", 3.0)

    def _split_line_chunks(sp_list):
        # --- begin gap logging context
        gap_logger = getattr(cluster_and_join, "_gap_logger", None)
        page_ctx   = getattr(cluster_and_join, "_gap_page", None)
        side_ctx   = getattr(cluster_and_join, "_gap_side", None)
        # sort spans left→right
        sp_list = sorted(sp_list, key=lambda s: s["bbox"][0])
        if not sp_list:
            return []

        # compute adjacent gaps based on bbox edges
        gaps = []
        for i in range(len(sp_list)-1):
            prev = sp_list[i]["bbox"]; cur = sp_list[i+1]["bbox"]
            gap = max(0.0, cur[0] - prev[2])
            gaps.append(gap)

        # robust small-gap baseline: 20th percentile of positive gaps
        pos_gaps = [g for g in gaps if g > 0.0]
        small_ref = 0.0
        if pos_gaps:
            try:
                small_ref = float(np.percentile(pos_gaps, 20.0))
            except Exception:
                pos_gaps_sorted = sorted(pos_gaps)
                idx = max(0, int(0.20 * (len(pos_gaps_sorted)-1)))
                small_ref = float(pos_gaps_sorted[idx])

        # fallback: use median character/box width as proxy
        widths = [max(1.0, s["bbox"][2]-s["bbox"][0]) for s in sp_list]
        med_w = sorted(widths)[len(widths)//2] if widths else 10.0

        # final threshold: max(absolute floor, small-gap * multiplier, median width * 1.25)
        inline_mult = getattr(cluster_and_join, "_inline_gap_mult", 3.0)
        thr = max(6.0, small_ref * float(inline_mult), med_w * 1.25)

        # helper to record line center y
        def _line_center_y(splist):
            if not splist: return None
            bb0 = splist[0]["bbox"]; bb1 = splist[-1]["bbox"]
            return ( (bb0[1]+bb0[3])/2.0 + (bb1[1]+bb1[3])/2.0 ) / 2.0

        # split at large gaps (→ parallel paragraph chunks on the same row)
        chunks = [[sp_list[0]]]
        for i in range(1, len(sp_list)):
            prev_bb = sp_list[i-1]["bbox"]; cur_bb = sp_list[i]["bbox"]
            gap = max(0.0, cur_bb[0] - prev_bb[2])
            if gap >= thr:
                chunks.append([sp_list[i]])
            else:
                chunks[-1].append(sp_list[i])

        # log per-line gap metrics (if enabled)
        if gap_logger is not None:
            try:
                # throttle: at most args.gap_max_lines per (page, side)
                from builtins import str as _str  # safe no-op import
                key = (page_ctx, side_ctx)
                global GAP_LOG_COUNT
                if not isinstance(GAP_LOG_COUNT, dict):
                    GAP_LOG_COUNT = {}
                cnt = GAP_LOG_COUNT.get(key, 0)
                max_lines = getattr(cluster_and_join, "_gap_max_lines", 60)
                if cnt < max_lines:
                    GAP_LOG.append({
                        "page": page_ctx,
                        "side": side_ctx,
                        "line_cy": _line_center_y(sp_list),
                        "num_spans": len(sp_list),
                        "num_chunks": len([c for c in chunks if c]),
                        "gaps": [float(g) for g in gaps],
                        "pos_gaps": [float(g) for g in pos_gaps],
                        "small_ref": float(small_ref),
                        "median_char_w": float(med_w),
                        "inline_gap_mult": float(inline_mult),
                        "threshold": float(thr),
                    })
                    GAP_LOG_COUNT[key] = cnt + 1
            except Exception:
                pass

        # join each chunk into its own string
        joined = [" ".join(s["text"] for s in chunk).strip() for chunk in chunks if chunk]
        return [j for j in joined if j]

    # Set default no-op logger/context if not present
    if not hasattr(cluster_and_join, "_gap_logger"):
        cluster_and_join._gap_logger = None
    if not hasattr(cluster_and_join, "_gap_page"):
        cluster_and_join._gap_page = None
    if not hasattr(cluster_and_join, "_gap_side"):
        cluster_and_join._gap_side = None

    for blk in blocks:
        line_chunks = _split_line_chunks(blk)
        if line_chunks:
            # join chunks with newline to indicate separate parallel paragraphs on same row
            lines.append("\n".join(line_chunks))

    seen=set(); dedup=[]
    for ln in lines:
        # if a line contains multiple chunks separated by '\n', dedup each chunk
        for subln in ln.split("\n"):
            sub = subln.strip()
            if not sub: continue
            norm=re.sub(r"\s+"," ", sub.lower())
            if norm in seen: 
                continue
            seen.add(norm)
            dedup.append(sub)

    def jaccard(a,b):
        sa=set(a.lower().split()); sb=set(b.lower().split())
        if not sa or not sb: return 0.0
        return len(sa&sb)/len(sa|sb)

    final=[]
    for ln in dedup:
        if final and jaccard(final[-1], ln) >= 0.7:
            continue
        final.append(ln)

    def is_substring_similar(a,b):
        a_low=a.lower(); b_low=b.lower()
        if a_low in b_low or b_low in a_low:
            tokens_a=set(a_low.split()); tokens_b=set(b_low.split())
            if tokens_a and tokens_b:
                j = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                if j >= 0.85:
                    return True
        return False

    filtered=[]
    for ln in final:
        if filtered and is_substring_similar(filtered[-1], ln):
            continue
        filtered.append(ln)

    return "\n".join(filtered).strip()

def extract_title_from_side(page: fitz.Page,
                            side_rect: fitz.Rect,
                            title_percentile: float,
                            title_base_size: Optional[float],
                            title_size_tol: float,
                            wm_top_percent: float,
                            wm_margin: float,
                            join_gap: float,
                            down_tol: float) -> Tuple[str, float, float, Optional[Tuple[float,float,float,float]]]:
    spans = gather_spans_in_rect(page, side_rect)
    spans = dedup_overlapping_spans(spans, iou_tol=0.85, center_tol=1.2, text_sim_tol=0.90)
    if not spans:
        return ("", 0.0, 0.0, None)
    sizes = [s["size"] for s in spans]

    wm_max_pt = detect_wm_font_size(spans, page.rect.height, wm_top_percent)
    body_stats = compute_body_stats(spans, page.rect.height, wm_top_percent)
    body_reg_p90 = body_stats["reg_p90"]
    body_bold_p90 = body_stats["bold_p90"]

    perc_pt = percentile(sizes, title_percentile)
    use_base = (title_base_size is not None)

    if use_base:
        low = title_base_size - title_size_tol
        high = title_base_size + title_size_tol
        cand = [s for s in spans
                if (low <= s["size"] <= high)
                and (not WM_TOP_PAT.match(s["text"]))
                and ("BOM" not in s["text"].upper())]
        is_max_mode = False
    else:
        body_margin = getattr(extract_title_from_side, "_body_margin", 0.8)
        bold_guard  = getattr(extract_title_from_side, "_bold_guard", 0.5)
        dyn_min_pt = max(
            perc_pt,
            wm_max_pt + wm_margin,
            body_reg_p90 + body_margin,
            body_bold_p90 + bold_guard
        )
        cand = [s for s in spans
                if (s["size"] >= dyn_min_pt)
                and (not WM_TOP_PAT.match(s["text"]))
                and ("BOM" not in s["text"].upper())]
        is_max_mode = (title_percentile >= 99.5)

    if not cand:
        return ("", 0.0, wm_max_pt, None)

    joined_text = cluster_and_join(cand, join_gap, down_tol)
    lines = joined_text.split("\n") if joined_text else []
    if not lines: return ("", 0.0, wm_max_pt, None)

    def is_bold_only_line(line_text: str) -> bool:
        tokens = line_text.strip()
        if not tokens: return False
        for s in cand:
            if s["text"].strip() == tokens and (int(s.get("flags",0)) & 2) != 0:
                return True
        bold_texts = "".join(s["text"].strip() for s in cand if (int(s.get("flags",0)) & 2) != 0)
        return bold_texts.replace(" ","") == tokens.replace(" ","")

    if is_max_mode:
        all_bold = all((int(s.get("flags",0)) & 2) != 0 for s in cand)
        total_len = sum(len(s["text"].strip()) for s in cand)
        if all_bold and total_len < 15:
            return ("", 0.0, wm_max_pt, None)

    filtered_lines = []
    for ln in lines:
        if len(ln.strip()) < 15 and is_bold_only_line(ln):
            continue
        filtered_lines.append(ln)

    if not filtered_lines:
        return ("", 0.0, wm_max_pt, None)

    def jaccard_tokens(a, b):
        sa = set(re.sub(r"\W+", " ", a.lower()).split())
        sb = set(re.sub(r"\W+", " ", b.lower()).split())
        if not sa or not sb: return 0.0
        return len(sa & sb) / len(sa | sb)

    def is_highly_similar(a, b):
        jac = jaccard_tokens(a, b)
        if jac >= 0.85: return True
        a_low = a.lower().strip(); b_low = b.lower().strip()
        if a_low in b_low or b_low in a_low:
            return jac >= 0.7
        return False

    collapsed = []
    for ln in filtered_lines:
        if collapsed and is_highly_similar(collapsed[-1], ln):
            continue
        collapsed.append(ln)

    deduped = []
    for ln in collapsed:
        if any(is_highly_similar(prev, ln) for prev in deduped):
            continue
        deduped.append(ln)

    title = "\n".join(deduped).strip()
    max_pt = max([s["size"] for s in cand], default=0.0)
    xs0 = min(s["bbox"][0] for s in cand); ys0 = min(s["bbox"][1] for s in cand)
    xs1 = max(s["bbox"][2] for s in cand); ys1 = max(s["bbox"][3] for s in cand)
    title_bbox = (float(xs0), float(ys0), float(xs1), float(ys1))
    return (title, max_pt, wm_max_pt, title_bbox)


# --- 장식문구 페이지 보수적 판정 ---
def is_decorative_page(spans: List[Dict[str,Any]]) -> bool:
    """
    Classify a page as decorative if none of the spans contain Korean case-marking particles
    ("은", "는", "이", "가", "을", "를").
    Returns True if the page is decorative, False otherwise.
    """
    if not spans or len(spans) < 1:
        return False
    particles = ["은", "는", "이", "가", "을", "를"]
    for s in spans:
        txt = s.get("text", "")
        if any(p in txt for p in particles):
            return False
    return True

# --- 워터마크 라인 제거 유틸 ---
def _normalize_wm_label(s: str) -> str:
    if not s: return ""
    t = s.strip().lower()
    t = t.replace("︱","|").replace("｜","|").replace("│","|").replace("¦","|")
    t = t.replace("l", "|")
    t = re.sub(r"\s+", "", t)
    return t

# --- whitespace-collapsing helper for simple comparisons ---
def _norm_simple(s: str) -> str:
    """Collapse whitespace and lowercase for simple comparisons."""
    return re.sub(r"\s+", "", (s or "").strip().lower())

def _norm_for_match(s: str) -> str:
    if not s:
        return ""
    t = str(s)
    t = (t.replace("“","\"").replace("”","\"")
           .replace("‘","'").replace("’","'")
           .replace("│","|").replace("︱","|").replace("｜","|"))
    t = re.sub(r"\s+", " ", t).strip().lower()
    # 가벼운 문장부호/기호 제거(한글/영문 단어는 보존)
    t = re.sub(r"[\"'·…‐-–—\-]|(?<=\w)[\.,](?=\w)", "", t)
    return t.strip()

BOTTOM_NUM_RE = re.compile(r"\bBOM\s*\d{1,3}\b", re.IGNORECASE)

# --- 추가 워터마크 배너 감지 유틸 ---
SECTION_LABELS_UP = {"TRAVEL", "GOLF", "CAR", "NEWS", "MOTORRAD", "CULTURE", "BMW", "MINI"}

def _upper_ratio(s: str) -> float:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 0.0
    uppers = [ch for ch in letters if ch.isupper()]
    return (len(uppers) / len(letters)) if letters else 0.0

_BANNER_ASCII_RE = re.compile(r"^[A-Z0-9\s\|\-\/\.\,]+$")

def _is_wm_banner_like(raw_text: str) -> bool:
    """
    라인 텍스트가 워터마크/섹션 배너 성격이면 True.
    - 'BOM', 'BOM 12' 등
    - 'TRAVEL', 'GOLF' 등 섹션 라벨, 또는 'l TRAVEL'
    - 대문자/ASCII 위주 + '|' 또는 '/' 포함된 짧은 배너형 문자열
    """
    t = (raw_text or "").strip()
    if not t:
        return False
    up = t.upper().replace("│", "|").replace("︱", "|").replace("｜", "|")
    # 자주 나오는 고정 패턴
    if up == "BOM" or re.fullmatch(r"BOM\s*\d{1,3}", up):
        return True
    if up.replace("L ", "") in SECTION_LABELS_UP:
        return True
    if "BOM|" in up or "|BOM" in up or "BOM |" in up:
        return True
    # ASCII 배너 형태(대문자 비율 높고 파이프/슬래시 포함, 길이 짧음)
    if _BANNER_ASCII_RE.match(up) and (_upper_ratio(up) >= 0.8) and (("|" in up) or ("/" in up) or (" -" in up) or ("—" in up)) and (len(up) <= 40):
        return True
    return False


def _rescue_lead_with_plumber(
    args,
    page_num: int,
    side_rect: fitz.Rect,
    title_bbox: Optional[Tuple[float, float, float, float]],
    band_frac: float,
    content_words: List[Dict[str, Any]],
    raw_words_stream: Optional[List[Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Use pdfplumber within the title-below 'lead band' to identify the top-most human-reading-order line,
    then re-anchor the matching PyMuPDF words to the very front so they appear first.

    Improvements:
      - Auto-expand lead band when no groups/low similarity (attempts: [band_frac, band_frac*2 ≤ 0.35]).
      - Earliest-stream fallback: if similarity is low, pick the earliest (block,line) group inside the band.
      - Keep stream-order semantics (no y-center global sorting).
    """
    if getattr(args, "rescue_engine", "off") != "plumber":
        return None
    if pdfplumber is None:
        return None
    pdf_path = getattr(args, "_pdf_path", None)
    if not pdf_path:
        return None

    def _wyc(bb):
        return (bb[1] + bb[3]) / 2.0

    try:
        with pdfplumber.open(pdf_path) as ppdf:
            if page_num - 1 < 0 or page_num - 1 >= len(ppdf.pages):
                return None
            ppage = ppdf.pages[page_num - 1]
            page_h = float(ppage.height)

            # attempt list: original band + expanded band
            attempts = [float(band_frac)]
            expanded = min(float(band_frac) * 2.0, 0.35)
            if expanded > band_frac + 1e-6:
                attempts.append(expanded)

            for attempt_idx, bf in enumerate(attempts):
                # Lead band top in PyMuPDF coordinates (origin: top-left)
                if title_bbox:
                    tx0, ty0, tx1, ty1 = title_bbox
                    band_top_py = float(ty1)
                else:
                    band_top_py = float(min(page_h * 0.12, page_h))
                band_bot_py = float(min(band_top_py + bf * page_h, page_h))

                # Convert to pdfplumber crop box (origin: bottom-left)
                x0_pl = float(side_rect.x0)
                x1_pl = float(side_rect.x1)
                y0_pl = page_h - band_bot_py
                y1_pl = page_h - band_top_py

                region = ppage.crop((x0_pl, y0_pl, x1_pl, y1_pl))
                words = region.extract_words() or []
                if not words:
                    # try next attempt
                    continue

                # Group words into lines by similar 'top'
                lines_map: Dict[float, List[dict]] = {}
                for w in words:
                    top_val = float(w.get("top", w.get("y0", 0.0)))
                    key = round(top_val, 1)
                    lines_map.setdefault(key, []).append(w)
                if not lines_map:
                    continue

                # Choose visually first line
                first_key = sorted(lines_map.keys())[0]
                first_line_words = sorted(lines_map[first_key], key=lambda d: float(d.get("x0", 0.0)))
                plumber_line = " ".join(w.get("text", "").strip() for w in first_line_words if w.get("text"))
                plumber_norm = _norm_for_match(plumber_line)
                if not plumber_norm:
                    continue

                # Base words for matching: use content_words if any exist in the band, else raw_words_stream
                has_content_in_band = any(band_top_py <= _wyc(w["bbox"]) <= band_bot_py for w in content_words)
                base_words = content_words if has_content_in_band else (raw_words_stream or [])
                if not base_words:
                    if getattr(args, "rescue_engine", "off") == "plumber":
                        print(f"[rescue] plumber skip p{page_num}: no base words in lead band (attempt {attempt_idx+1}/{len(attempts)})")
                    continue

                # Build groups in the band
                groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
                for w in base_words:
                    cy = _wyc(w["bbox"])
                    if band_top_py <= cy <= band_bot_py:
                        groups.setdefault((w.get("block", 0), w.get("line", 0)), []).append(w)

                # Build content_groups in the band (to avoid re-inserting filtered words)
                groups_content: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
                for w in content_words:
                    cy = _wyc(w["bbox"])
                    if band_top_py <= cy <= band_bot_py:
                        groups_content.setdefault((w.get("block", 0), w.get("line", 0)), []).append(w)

                if not groups:
                    continue

                # Scoring helpers
                def norm_join(ws: List[Dict[str, Any]]) -> str:
                    ws_sorted = sorted(ws, key=lambda d: (d.get("word", 0), d["bbox"][0]))
                    return _norm_for_match(" ".join(x.get("text", "").strip() for x in ws_sorted if x.get("text")))

                def jaccard(a: str, b: str) -> float:
                    sa = set(re.sub(r"\W+", " ", a.lower()).split())
                    sb = set(re.sub(r"\W+", " ", b.lower()).split())
                    if not sa or not sb:
                        return 0.0
                    return len(sa & sb) / len(sa | sb)

                # Pick best by similarity; if too low, fall back to earliest (block,line)
                best_key: Optional[Tuple[int, int]] = None
                best_score = -1.0
                for key, ws in groups.items():
                    cand = norm_join(ws)
                    if not cand:
                        continue
                    score = 1.0 if (cand == plumber_norm or cand in plumber_norm or plumber_norm in cand) else jaccard(cand, plumber_norm)
                    if score > best_score:
                        best_score = score
                        best_key = key

                # Earliest-stream fallback when similarity is too low
                if best_key is None or best_score < 0.20:
                    earliest_keys = sorted(groups.keys(), key=lambda k: (k[0], k[1]))
                    if earliest_keys:
                        best_key = earliest_keys[0]
                        best_score = max(best_score, 0.20)

                if best_key is None:
                    continue

                chosen_ws = groups_content.get(best_key) or groups.get(best_key) or []
                if not chosen_ws:
                    continue

                # Keep only tokens that still exist in content_words (avoid re-adding filtered tokens)
                cw_index = {(w.get("text", ""), w.get("bbox")) for w in content_words}
                chosen_ws = [w for w in chosen_ws if (w.get("text", ""), w.get("bbox")) in cw_index]
                if not chosen_ws:
                    continue

                # Re-anchor chosen group to front by assigning a smaller block id
                min_block = min(w.get("block", 0) for w in content_words) if content_words else 0
                new_block = min_block - 10

                # Keep other band groups + outside words as-is
                groups_from_content: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
                for w in content_words:
                    cy = _wyc(w["bbox"])
                    if band_top_py <= cy <= band_bot_py:
                        groups_from_content.setdefault((w.get("block", 0), w.get("line", 0)), []).append(w)

                rescued: List[Dict[str, Any]] = []
                others: List[Dict[str, Any]] = []
                for key, ws in groups_from_content.items():
                    if key == best_key:
                        for w in chosen_ws:
                            w2 = dict(w)
                            w2["block"] = new_block
                            w2["line"] = 0
                            rescued.append(w2)
                    else:
                        others.extend(ws)

                outside: List[Dict[str, Any]] = []
                for w in content_words:
                    cy = _wyc(w["bbox"])
                    if not (band_top_py <= cy <= band_bot_py):
                        outside.append(w)

                updated = rescued + others + outside
                updated.sort(key=lambda d: (d.get("block", 0), d.get("line", 0), d.get("word", 0), d["bbox"][0]))
                if getattr(args, "rescue_engine", "off") == "plumber":
                    band_pct = f"{bf:.2f}"
                    print(f"[rescue] plumber applied on page {page_num} (attempt {attempt_idx+1}/{len(attempts)}, band={band_pct}, score={best_score:.2f})")
                return updated

            # all attempts failed
            if getattr(args, "rescue_engine", "off") == "plumber":
                print(f"[rescue] plumber skip p{page_num}: no grouped lines / low score after attempts")
            return None

    except Exception:
        return None


def _strip_residual_wm_lines(text: str) -> str:
    """
    Post-filter already reflowed text to drop any residual watermark-like lines that slipped through:
      - 'BOM' or 'BOM 12' style tokens,
      - section labels such as (TRAVEL, GOLF, CAR, NEWS, MOTORRAD, CULTURE) optionally prefixed with 'l'/'L',
      - lines made only of bars/dashes,
      - lone 'l'/'I' artifacts.
    """
    if not text:
        return text
    lines = text.splitlines()
    out = []
    WM_LINE2_RE = re.compile(r'^\s*(?:[lL]\s*)?(TRAVEL|GOLF|CAR|NEWS|MOTORRAD|CULTURE)\s*$')
    BOM_RE      = re.compile(r'^\s*BOM(?:\s*\d{1,3})?\s*$', re.IGNORECASE)
    BARS_RE     = re.compile(r'^[\|\│︱｜\-–—\s]+$')
    LONE_LI_RE  = re.compile(r'^\s*[lLiI]\s*$')
    for ln in lines:
        raw = (ln or "").strip()
        if not raw:
            continue
        up = raw.upper().replace("│", "|").replace("︱", "|").replace("｜", "|")
        if BOM_RE.match(up):
            continue
        if WM_LINE2_RE.match(up):
            continue
        if BARS_RE.match(up):
            continue
        if LONE_LI_RE.match(up):
            continue
        out.append(ln)
    return "\n".join(out).strip()

def remove_wm_lines(spans: List[Dict[str,Any]], page_rect: fitz.Rect,
                    top_percent: float, bottom_percent: float,
                    line_join_gap: float = 6.0) -> List[Dict[str,Any]]:
    """상단/하단 워터마크 라인을 라인 단위로 제거."""
    if not spans: return spans

    def cy(s): bb=s["bbox"]; return (bb[1]+bb[3])/2.0
    spans_sorted = sorted(spans, key=lambda s:(round(cy(s),2), s["bbox"][0]))

    lines: List[List[Dict[str,Any]]] = []
    for s in spans_sorted:
        if not lines:
            lines.append([s]); continue
        prev = lines[-1][-1]
        same_line = abs(cy(s) - cy(prev)) <= line_join_gap
        if same_line: lines[-1].append(s)
        else: lines.append([s])

    H = float(page_rect.height)
    top_y = page_rect.y0 + H * (top_percent/100.0)
    bot_y0 = page_rect.y1 - H * (bottom_percent/100.0)

    keep_spans: List[Dict[str,Any]] = []
    for ln in lines:
        ys = [cy(x) for x in ln]
        line_cy = sum(ys)/len(ys)
        ln_sorted = sorted(ln, key=lambda s:s["bbox"][0])
        raw_text = " ".join(s.get("text","").strip() for s in ln_sorted if s.get("text","").strip())
        norm = _normalize_wm_label(raw_text)

        # 라인 전체 폭 비율(페이지 대비)
        try:
            lx0 = min(s["bbox"][0] for s in ln_sorted)
            lx1 = max(s["bbox"][2] for s in ln_sorted)
            line_width_ratio = (lx1 - lx0) / float(page_rect.width if page_rect.width else 1.0)
        except Exception:
            line_width_ratio = 0.0

        # 배너성 워터마크 여부(텍스트 패턴 기반)
        banner_like = _is_wm_banner_like(raw_text)

        # Additional normalization and flags
        raw_strip = raw_text.strip()
        raw_upper = raw_strip.upper()
        is_page_num = bool(re.fullmatch(r"\d{1,3}", raw_strip))
        is_travel_band = bool(re.fullmatch(r"[lL]\s*TRAVEL", raw_upper)) or (raw_upper == "TRAVEL")
        is_section_band = raw_upper in {"GOLF", "MOTORRAD", "CULTURE", "CAR", "NEWS"}

        is_top_zone = (line_cy <= top_y)
        is_bot_zone = (line_cy >= bot_y0)

        drop = False
        if is_top_zone:
            if WM_TOP_PAT.match(raw_text) or norm.startswith("bom|") or norm == "bom":
                drop = True
        if (not drop) and is_bot_zone:
            if BOTTOM_NUM_RE.search(raw_text):
                drop = True
        # extra: page number or section strip (e.g. "50", "54", "l TRAVEL") that sits in top/bottom band or near top
        if (not drop) and (is_page_num or is_travel_band or is_section_band):
            # allow a slightly deeper top band for these decorative labels
            if (line_cy <= page_rect.y0 + H * 0.35) or is_top_zone or is_bot_zone:
                drop = True

        # 추가 배너/워터마크 판단: 상단/하단 대역이 아니더라도,
        # 배너성 텍스트이거나(ASCII + 파이프/슬래시/섹션) 라인 폭이 넓은 경우 제거
        if (not drop) and banner_like and (is_top_zone or is_bot_zone or line_width_ratio >= 0.60):
            drop = True

        if not drop:
            keep_spans.extend(ln)

    keep_spans.sort(key=lambda s:(round(cy(s),2), s["bbox"][0]))
    return keep_spans



def group_spans_into_columns(spans: List[Dict[str,Any]],
                             min_col_width: float = 40.0,
                             gap_mult: float = 2.8) -> List[List[Dict[str,Any]]]:
    """
    Split spans into vertical columns using a hybrid approach:
      1) 1D histogram over center-x to find density valleys (preferred)
      2) fallback to center-x gap thresholding
    Returns list of columns (left→right), each is a list of spans.
    """
    if not spans:
        return []

    # --- helpers
    def cx(s): 
        bb = s["bbox"]; 
        return (bb[0]+bb[2])/2.0
    def cy(s):
        bb = s["bbox"]
        return (bb[1]+bb[3])/2.0
    def width(s):
        bb = s["bbox"]
        return max(1.0, bb[2]-bb[0])

    spans_sorted = sorted(spans, key=lambda s: (cx(s), cy(s)))

    # typical span width & x-range
    widths = [width(s) for s in spans_sorted]
    widths_sorted = sorted(widths)
    med_w = widths_sorted[len(widths_sorted)//2] if widths_sorted else 12.0
    # conservative threshold (fallback)
    gap_thr = max(float(min_col_width), float(med_w) * float(gap_mult))

    xs = [cx(s) for s in spans_sorted]
    x_min, x_max = min(xs), max(xs)
    if x_max <= x_min:
        return [spans_sorted]

    # --- 1) histogram valley split
    # bin size ~ median width for stability
    bin_w = max(8.0, float(med_w))
    nbins = max(1, int((x_max - x_min) / bin_w))
    try:
        hist, edges = np.histogram(xs, bins=nbins, range=(x_min, x_max))
    except Exception:
        hist, edges = None, None

    split_positions: List[float] = []
    if hist is not None and edges is not None and len(hist) >= 3:
        # identify valleys: bins with very low density (<= configurable fraction of median non-zero)
        nz = [h for h in hist if h > 0]
        med_density = float(np.median(nz)) if nz else 0.0
        # use configurable valley fraction
        valley_thr = max(1.0, med_density * float(GROUP_VALLEY_FRAC)) if med_density > 0 else 1.0

        for i in range(1, len(hist)-1):
            if hist[i] <= valley_thr and hist[i-1] > 0 and hist[i+1] > 0:
                # candidate valley between edges[i] ~ edges[i+1]
                split_x = (edges[i] + edges[i+1]) / 2.0
                split_positions.append(split_x)

        # de-duplicate close splits
        split_positions.sort()
        merged_splits: List[float] = []
        for sx in split_positions:
            if not merged_splits or abs(sx - merged_splits[-1]) > (bin_w * 0.75):
                merged_splits.append(sx)
        split_positions = merged_splits

    columns: List[List[Dict[str,Any]]] = []
    if split_positions:
        # Assign by nearest split boundaries (left→right bins)
        bounds = [x_min - 1e-3] + split_positions + [x_max + 1e-3]
        for bi in range(len(bounds)-1):
            lo, hi = bounds[bi], bounds[bi+1]
            col = [s for s in spans_sorted if lo <= cx(s) < hi]
            if col:
                columns.append(col)

        # merge tiny singleton columns into nearest neighbor
        if len(columns) >= 2:
            merged: List[List[Dict[str,Any]]] = []
            for i, col in enumerate(columns):
                if len(col) == 1:
                    # merge to neighbor with closer mean x
                    cxs = [cx(s) for s in col]
                    mean_x = sum(cxs)/len(cxs)
                    # choose neighbor
                    if i == 0:
                        columns[i+1].extend(col)
                    elif i == len(columns)-1:
                        columns[i-1].extend(col)
                    else:
                        left_mean = sum(cx(s) for s in columns[i-1]) / max(1, len(columns[i-1]))
                        right_mean = sum(cx(s) for s in columns[i+1]) / max(1, len(columns[i+1]))
                        if abs(mean_x - left_mean) <= abs(mean_x - right_mean):
                            columns[i-1].extend(col)
                        else:
                            columns[i+1].extend(col)
                else:
                    merged.append(col)
            if merged:
                columns = merged

        # final ordering by mean x
        columns = sorted(columns, key=lambda c: sum(cx(s) for s in c)/len(c))

    else:
        # --- 2) fallback: gap thresholding
        columns = [[spans_sorted[0]]]
        last_cx = cx(spans_sorted[0])
        for s in spans_sorted[1:]:
            cxi = cx(s)
            if (cxi - last_cx) > gap_thr:
                columns.append([s])
            else:
                columns[-1].append(s)
            last_cx = cxi

        # drop tiny columns by merging to nearest
        if len(columns) >= 2:
            clean_cols: List[List[Dict[str,Any]]] = []
            for i, col in enumerate(columns):
                if len(col) == 1:
                    if i == 0:
                        columns[i+1].extend(col)
                    else:
                        columns[i-1].extend(col)
                else:
                    clean_cols.append(col)
            if clean_cols:
                columns = clean_cols

        # ensure left→right by mean x
        columns = sorted(columns, key=lambda c: sum(cx(s) for s in c)/len(c))

    # inside each column, sort top→bottom for stability
    for i, col in enumerate(columns):
        columns[i] = sorted(col, key=lambda s: (round(cy(s),2), s["bbox"][0]))

    return columns


def extract_article_texts(
    doc,
    out_dir,
    start_idx,
    stop_idx,
    excludes,
    start_anchor,
    stop_anchor,
    title_base_size,
    title_size_tol,
    title_percentile,
    title_top_percent,
    line_join_gap,
    block_down_tol,
    wm_margin,
    article_wm_top_percent,
    wm_bottom_percent,
    col_gap_mult,
    min_col_width,
    articles_filename,
    args,
):
    """기사 본문 텍스트를 추출하여 page_articles.csv/JSON/MD 저장(제목/워터마크/장식문구/제외페이지 제외).
       출력 순서: Left-Top → Left-Bottom → Right-Top → Right-Bottom
       col_gap_mult, min_col_width: 열 분할 동작 제어 파라미터
    """
    records = []
    md_lines = ["# Extracted Articles\n"]
    json_records = []


    for p in range(start_idx, stop_idx):
        page_num = p + 1
        ex = excludes.get(page_num)

        # Only exclude the whole page if explicitly marked ALL.
        if ex == "ALL":
            rec = {"pdf_page": page_num, "excluded": True, "exclude_reason": "ALL", "article_text": ""}
            records.append(rec)
            json_records.append(rec)
            md_lines.append(f"## Page {page_num}\n- excluded: **ALL**\n")
            continue

        page = doc.load_page(p)
        L_rect, R_rect = page_halves_rect(page)

        # respect side-specific excludes (but do not drop entire page)
        skip_L = (ex == "L")
        skip_R = (ex == "R")

        # Helper to clean spans (dedup + WM line removal), detect decorative, and build text
        def build_side_text(raw_spans: List[Dict[str,Any]], side_tag: str, _args=args) -> Tuple[str, bool]:
            # --- subtitle / body discriminator (for title-bottom band) ---
            SUBTITLE_MAX_CHARS = 120  # subtitle은 대략 1~2줄, 너무 길면 본문으로 간주
            SUBTITLE_MAX_DOTS  = 1    # 마침표가 2개 이상이면 본문으로 간주
            def _is_subtitle_like(line_text: str) -> bool:
                t = (line_text or "").strip()
                if not t:
                    return False
                # 너무 길면 본문
                if len(t) > SUBTITLE_MAX_CHARS:
                    return False
                # 마침표/물음표/느낌표 개수 체크
                dots = t.count(".") + t.count("?") + t.count("!") + t.count("…")
                if dots > SUBTITLE_MAX_DOTS:
                    return False
                # 쉼표가 아주 많아도 본문일 확률이 높다
                if t.count(",") >= 3:
                    return False
                return True
            # gap logger
            if getattr(_args, "dump_gaps", False):
                cluster_and_join._gap_logger = lambda rec: GAP_LOG.append(rec)
                cluster_and_join._gap_page = page_num
                cluster_and_join._gap_side = side_tag
            else:
                cluster_and_join._gap_logger = None
                cluster_and_join._gap_page = None
                cluster_and_join._gap_side = None

            side_rect = (L_rect if side_tag == "L" else R_rect)

            if not raw_spans:
                return ("", True)

            # 1) span dedup + wm line 제거
            cleaned_spans = dedup_overlapping_spans(
                raw_spans, iou_tol=0.85, center_tol=1.2, text_sim_tol=0.90
            )
            cleaned_spans = remove_wm_lines(
                cleaned_spans,
                page_rect=page.rect,
                top_percent=article_wm_top_percent,
                bottom_percent=wm_bottom_percent,
                line_join_gap=6.0,
            )

            # 2) word stream + span attrs
            words_stream = gather_words_in_rect(page, side_rect)
            words_stream = annotate_words_with_span_attrs(words_stream, cleaned_spans)
            # 2-1) 겹치는 텍스트레이어(동일 텍스트, 거의 같은 위치)를 1개로 정리
            words_stream = dedup_words_by_overlap(words_stream, iou_tol=0.85, center_tol=1.0)
            words_stream_snapshot = [dict(w) for w in words_stream]

            # --- (A) 라인 지문 기반 중복 제거 + 리드 밴드 보호 ---
            # 제목 bbox를 먼저 조회(있으면 사용)
            title_bbox_map = getattr(_args, "_title_bbox_map", {})
            title_bbox = title_bbox_map.get((page_num, side_tag)) if title_bbox_map else None

            # 라인으로 묶어 최초 등장만 유지
            _lines_map: Dict[Tuple[int,int], List[Dict[str,Any]]] = {}
            for _w in words_stream:
                _lines_map.setdefault((_w.get("block",0), _w.get("line",0)), []).append(_w)

            # 리드 밴드 계산: 제목 bbox 아래 ~ (subtitle_band_frac + 0.04) 페이지 높이
            if title_bbox:
                _lead_top = title_bbox[3]
            else:
                _lead_top = page.rect.y0 + page.rect.height * 0.12
            _lead_bot = min(page.rect.y1, _lead_top + page.rect.height *
                            (float(getattr(_args, "subtitle_band_frac", 0.12)) + 0.04))

            def _norm_line_txt(_t: str) -> str:
                return re.sub(r"\s+", " ", (_t or "").strip().lower())

            _first_seen: Dict[str, Tuple[int,int]] = {}
            _drop_keys: set = set()

            for _key, _ws in sorted(_lines_map.items()):
                _ws.sort(key=lambda d: (d.get("word", 0), d["bbox"][0]))
                _line_txt = " ".join(wi["text"] for wi in _ws).strip()
                if not _line_txt:
                    continue
                _norm = _norm_line_txt(_line_txt)
                _wy = float(np.mean([(wi["bbox"][1] + wi["bbox"][3]) / 2.0 for wi in _ws]))
                _in_lead = (_lead_top <= _wy <= _lead_bot)
                if _norm not in _first_seen:
                    _first_seen[_norm] = _key
                    # 리드 밴드에서 최초 관측되면 보호 플래그
                    if _in_lead:
                        for wi in _ws:
                            wi["_lead_guard"] = True
                else:
                    # 같은 문장이 다시 나오면(이중 레이어/복제본) 뒤쪽은 삭제
                    _drop_keys.add(_key)

            if _drop_keys:
                _new_words_stream: List[Dict[str,Any]] = []
                for _key, _ws in _lines_map.items():
                    if _key in _drop_keys:
                        continue
                    _new_words_stream.extend(_ws)
                words_stream = _new_words_stream
            # --- (A) 끝 ---

            # 3) decorate check (원 버전은 여기서 spans로 했지만, 스트림으로 해도 됨)
            deco = is_decorative_page(raw_spans)

            # 4) caption/bullet pruning (rollback to basic rule)
            lines_by_key: Dict[Tuple[int,int], List[Dict[str,Any]]] = {}
            for w in words_stream:
                key = (w["block"], w["line"])
                lines_by_key.setdefault(key, []).append(w)
            body_sizes = [float(w.get("size",0.0) or 0.0) for w in words_stream]
            body_med = float(np.median(body_sizes)) if body_sizes else 0.0
            CAP_BULLET_RE = re.compile(r"^[\s\-–·•●◦▲△▷▶▸❖◆■▪]+")
            cap_small_diff = getattr(_args, "caption_small_diff", 1.1)
            cap_max_chars  = getattr(_args, "caption_max_chars", 28)
            # --- extra: watermark-like line filters (mid-page, not only top/bottom bands) ---
            WM_LINE_RE = re.compile(r"^(?:[lL]\s*)?(TRAVEL|GOLF|CAR|NEWS|MOTORRAD)$")
            WM_LINE_BAR_RE = re.compile(r"^\|+$")

            pruned_words: List[Dict[str,Any]] = []
            for key, ws in sorted(lines_by_key.items()):
                ws.sort(key=lambda d: (d.get("word",0), d["bbox"][0]))
                line_text = " ".join(w["text"] for w in ws).strip()
                if not line_text:
                    continue
                # 리드 밴드에서 최초로 관측된 라인은 보호(캡션/워터마크 필터를 넘어 통과)
                if any(w.get("_lead_guard") for w in ws):
                    pruned_words.extend(ws)
                    continue
                # 텍스트 패턴만으로 워터마크/배너로 판정되면 즉시 제거
                if _is_wm_banner_like(line_text):
                    continue
                avg_size = float(np.mean([float(w.get("size",0.0) or 0.0) for w in ws]))
                short_enough = len(re.sub(r"\s+","", line_text)) <= cap_max_chars
                bullet_like  = bool(CAP_BULLET_RE.match(line_text))
                tiny_vs_body = (body_med > 0 and (avg_size <= max(0.0, body_med - cap_small_diff)))
                # drop obvious watermark/section labels and stray band artifacts regardless of Y-band
                raw_up = line_text.strip().upper().replace("│","|").replace("︱","|").replace("｜","|")
                raw_strip = line_text.strip()
                # lone band artifacts like "l", "I", or just vertical bars
                if raw_up in {"L", "I", "|"} or WM_LINE_BAR_RE.match(raw_up):
                    continue
                # also catch lowercase 'l' residue
                if raw_strip in {"l", "i"}:
                    continue
                # BOM page number or BOM token alone
                if raw_up == "BOM" or re.fullmatch(r"BOM\s*\d{1,3}", raw_up):
                    continue
                # section strips like "TRAVEL", "l TRAVEL", "GOLF", etc.
                if WM_LINE_RE.fullmatch(raw_up):
                    continue
                # bare page number lines (short and small-ish), keep a small safety bound
                if re.fullmatch(r"\d{1,3}", line_text.strip()) and (avg_size <= body_med + 0.25) and len(ws) <= 3:
                    continue
                # Q. 머리는 살린다
                if re.match(r"^\s*Q[\.\s]", line_text):
                    pruned_words.extend(ws)
                    continue
                if (short_enough and tiny_vs_body) or (bullet_like and (avg_size <= body_med)):
                    # decorative/caption → drop
                    continue
                pruned_words.extend(ws)
            words_stream = pruned_words

            # (1) 이 페이지/사이드의 제목 bbox 가져오기
            title_bbox_map = getattr(_args, "_title_bbox_map", {})
            title_bbox = title_bbox_map.get((page_num, side_tag)) if title_bbox_map else None

            # (2) 제목 크기 대역 계산
            if title_base_size is not None:
                low = title_base_size - title_size_tol
                high = title_base_size + title_size_tol
            else:
                sizes_all = [float(w.get("size", 0.0) or 0.0) for w in words_stream]
                base = percentile(sizes_all, title_percentile) if sizes_all else 0.0
                low = base - title_size_tol
                high = base + title_size_tol

            # (3) 제목을 본문에 넣을지 말지 플래그
            # --keep-title-in-body 를 주면 True가 돼서 제목급도 본문에 남김
            drop_title_band = not getattr(_args, "keep_title_in_body", False)

            # (4) 제목 bbox를 못 잡은 페이지는 "위쪽 28%"를 한번 더 깎아서 제목/띠지를 날린다
            if drop_title_band and title_bbox is None:
                strict_top_y = page.rect.y0 + page.rect.height * 0.28
            else:
                strict_top_y = None

            # (5) 위/아래 워터마크 대역
            page_h = float(page.rect.height)
            top_band_y = page.rect.y0 + page_h * (article_wm_top_percent / 100.0)
            bot_band_y = page.rect.y1 - page_h * (wm_bottom_percent / 100.0)

            PAGE_NUM_RE = re.compile(r"^\d{1,3}$")
            SECTION_LABELS = {"CAR", "NEWS", "TRAVEL", "GOLF", "CULTURE"}

            # --- 6) 실제 본문 단어 만들기 ---
            content_words: List[Dict[str, Any]] = []
            for w in words_stream:
                sz = float(w.get("size", 0.0) or 0.0)
                txt = (w.get("text", "") or "").strip()
                if not txt:
                    continue

                # Lead-band guard: if this token belongs to a protected first-line group, keep it unconditionally.
                if w.get("_lead_guard"):
                    content_words.append(w)
                    continue

                # (6-1) 제목 bbox 미검출 시 상단 컷(완화: 28%). 제목 대역 글꼴이라도 lead_guard가 아니면만 제거
                if strict_top_y is not None:
                    wy0 = w["bbox"][1]
                    if wy0 <= strict_top_y and (low <= sz <= high):
                        continue

                # (6-2) 일반적인 제목대역 버리기 (옵션이 꺼져있을 때만)
                if drop_title_band and (low <= sz <= high):
                    continue

                wyc = (w["bbox"][1] + w["bbox"][3]) / 2.0
                in_wm_band = (wyc <= top_band_y) or (wyc >= bot_band_y)

                # 워터마크 / BOM 계열 제거
                nt = _normalize_wm_label(txt)
                if nt == "bom" or nt.startswith("bom|") or WM_TOP_PAT.match(txt) or "BOM" in txt.upper():
                    continue

                # 페이지번호가 워터마크 띠 안에 있을 때
                if PAGE_NUM_RE.match(txt) and in_wm_band:
                    continue

                # 상단/하단 띠에 있는 섹션 이름도 제거
                if in_wm_band and txt.upper() in SECTION_LABELS:
                    continue
                norm_up = txt.upper().replace("L ", "").replace("│", "|").strip()
                if in_wm_band and norm_up in SECTION_LABELS:
                    continue

                content_words.append(w)

            # (6-3) 한 번 더 레이어 중복 제거
            content_words = dedup_words_by_overlap(content_words, iou_tol=0.90, center_tol=1.0)

            # (6-4) 제목 박스 안에 반쯤이라도 겹친 건 무조건 빼기
            if title_bbox:
                tx0, ty0, tx1, ty1 = title_bbox
                def _bbox_overlap_ratio_word(bb):
                    ix0 = max(tx0, bb[0]); iy0 = max(ty0, bb[1])
                    ix1 = min(tx1, bb[2]); iy1 = min(ty1, bb[3])
                    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                    area = max(1.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))
                    return inter / area
                content_words = [
                    w for w in content_words
                    if _bbox_overlap_ratio_word(w["bbox"]) < 0.5
                ]

            # (6-5) 제목 바로 아래 리드 밴드 보정: pdfplumber로 먼저 구조 구제 시도 → 실패하면 기존 규칙
            if not getattr(_args, "_disable_subtitle_band", False):
                band_frac = float(getattr(_args, "subtitle_band_frac", 0.12))

                # ① pdfplumber rescue (human reading first-line in lead band)
                rescued = _rescue_lead_with_plumber(
                    args=_args,
                    page_num=page_num,
                    side_rect=side_rect,
                    title_bbox=title_bbox,
                    band_frac=band_frac,
                    content_words=content_words,
                    raw_words_stream=words_stream_snapshot
                )
                if rescued is not None:
                    content_words = rescued
                else:
                    # ② fallback: 기존 부제목-성향 라인만 앞으로 당기는 로직 (title_bbox 없으면 top 12%를 리드 시작으로 가정)
                    if title_bbox:
                        lead_top = float(title_bbox[3])
                    else:
                        lead_top = float(page.rect.y0 + page.rect.height * 0.12)
                    band_y = float(min(page.rect.y1, lead_top + page.rect.height * band_frac))
                    upper_words: List[Dict[str, Any]] = []
                    lower_words: List[Dict[str, Any]] = []
                    for w in content_words:
                        wyc = (w["bbox"][1] + w["bbox"][3]) / 2.0
                        if lead_top <= wyc <= band_y:
                            upper_words.append(w)
                        else:
                            lower_words.append(w)
                    # If nothing landed in the initial band, expand once and try again.
                    if not upper_words:
                        band_frac2 = min(band_frac * 2.0, 0.35)
                        band_y2 = float(min(page.rect.y1, lead_top + page.rect.height * band_frac2))
                        for w in content_words:
                            wyc2 = (w["bbox"][1] + w["bbox"][3]) / 2.0
                            if lead_top <= wyc2 <= band_y2:
                                upper_words.append(w)
                            else:
                                # keep existing lower_words entries and avoid duplicates
                                if w not in lower_words:
                                    lower_words.append(w)
                    if upper_words:
                        tmp_by_key: Dict[Tuple[int,int], List[Dict[str,Any]]] = {}
                        for uw in upper_words:
                            tmp_by_key.setdefault((uw.get("block",0), uw.get("line",0)), []).append(uw)
                        subtitle_lines: List[List[Dict[str,Any]]] = []
                        for (_bk,_ln), uw_list in tmp_by_key.items():
                            uw_list.sort(key=lambda d: d["bbox"][0])
                            line_txt = " ".join(u["text"] for u in uw_list).strip()
                            if _is_subtitle_like(line_txt):
                                subtitle_lines.append(uw_list)
                            else:
                                lower_words.extend(uw_list)
                        if subtitle_lines:
                            if lower_words:
                                min_block = min(w.get("block", 0) for w in lower_words)
                            else:
                                min_block = 0
                            subtitle_block = min_block - 10
                            ordered_subs: List[Dict[str,Any]] = []
                            for idx, uw_list in enumerate(sorted(subtitle_lines, key=lambda lst: (((lst[0]["bbox"][1]+lst[0]["bbox"][3])/2.0), lst[0]["bbox"][0]))):
                                for j, uw in enumerate(uw_list):
                                    uw["block"] = subtitle_block
                                    uw["line"]  = idx
                                    ordered_subs.append(uw)
                            content_words = ordered_subs + lower_words
                        else:
                            content_words = upper_words + lower_words
            # 6) 최종 합성
            if getattr(_args, "use_stream_order", True):
                text = _build_text_from_stream_words(
                    content_words,
                    inline_gap_mult=getattr(_args, "inline_gap_mult", 3.0)
                )
                # collapse adjacent duplicated tokens or half-echo lines, if enabled
                if getattr(_args, "stutter_fix", True):
                    text = _collapse_stutter_lines(
                        text,
                        min_pairs=getattr(_args, "stutter_min_pairs", 2)
                    )
                # reflow into readable sentences (body mode)
                text = _reflow_sentences(
                    text,
                    kind="body",
                    min_len=getattr(_args, "min_sentence_chars", 22),
                    allow_quote_split=getattr(_args, "allow_quote_split", False),
                    keep_qa_prefix=not getattr(_args, "no_keep_qa_prefix", False),
                )
                # post-filter residual watermark/band artifacts that slipped through
                text = _strip_residual_wm_lines(text)
                return (text.strip(), deco)
            else:
                # 기존 열 분할 경로 (나중에 필요하면 쓰도록 남겨둠)
                content_spans = [{
                    "text": w["text"],
                    "bbox": w["bbox"],
                    "size": float(w.get("size",0.0) or 0.0),
                    "flags": int(w.get("flags",0)),
                    "font": w.get("font","")
                } for w in content_words]
                if not content_spans:
                    return ("", deco)
                columns = group_spans_into_columns(
                    content_spans,
                    min_col_width=getattr(build_side_text, "_min_col_width", 40.0),
                    gap_mult=getattr(build_side_text, "_col_gap_mult", 2.8)
                )
                col_texts: List[str] = []
                for col in columns:
                    para_txt = cluster_and_join(col, line_join_gap, block_down_tol)
                    if para_txt.strip():
                        col_texts.append(para_txt.strip())
                geom_text = "\n\n".join(col_texts).strip()
                if getattr(_args, "stutter_fix", True):
                    geom_text = _collapse_stutter_lines(
                        geom_text,
                        min_pairs=getattr(_args, "stutter_min_pairs", 2)
                    )
                geom_text = _reflow_sentences(
                    geom_text,
                    kind="body",
                    min_len=getattr(_args, "min_sentence_chars", 22),
                    allow_quote_split=getattr(_args, "allow_quote_split", False),
                    keep_qa_prefix=not getattr(_args, "no_keep_qa_prefix", False),
                )
                # post-filter residual watermark/band artifacts that slipped through
                geom_text = _strip_residual_wm_lines(geom_text)
                return (geom_text, deco)
            
        # Helper: global de-duplication of highly similar lines (for both stream and geometry routes)
        def _dedup_lines_glob(text: str, sim: float = 0.95) -> str:
            if not text:
                return text
            lines = [ln.strip() for ln in text.splitlines()]
            out = []
            seen_norm = set()
            def _norm(s: str) -> str:
                return re.sub(r"\s+", " ", s.lower()).strip()
            def _jac(a: str, b: str) -> float:
                ta = set(re.sub(r"\W+", " ", a.lower()).split())
                tb = set(re.sub(r"\W+", " ", b.lower()).split())
                if not ta or not tb:
                    return 0.0
                return len(ta & tb) / len(ta | tb)
            for ln in lines:
                if not ln:
                    continue
                n = _norm(ln)
                if n in seen_norm:
                    continue
                if any(_jac(ln, prev) >= sim for prev in out):
                    continue
                out.append(ln)
                seen_norm.add(n)
            return "\n".join(out)

        # Build left and right texts (unless excluded)
        L_text, L_deco = ("", True)
        R_text, R_deco = ("", True)

        # Build left side
        if not skip_L:
            L_raw = gather_spans_in_rect(page, L_rect)
            L_text, L_deco = build_side_text(L_raw, "L", args)

        if not skip_R:
            R_raw = gather_spans_in_rect(page, R_rect)
            R_text, R_deco = build_side_text(R_raw, "R", args)

        # If all non-skipped sides are decorative, optionally skip the page (disabled in v7)
        if getattr(args, "_disable_decorative_skip", False):
            all_effective_deco = False
        else:
            all_effective_deco = ((skip_L or L_deco) and (skip_R or R_deco))

        if all_effective_deco:
            rec = {"pdf_page": page_num, "excluded": True, "exclude_reason": "decorative-auto", "article_text": ""}
            records.append(rec)
            json_records.append(rec)
            md_lines.append(f"## Page {page_num}\n- decorative-only: **skipped**\n")
            continue

        # Concatenate in human reading order: LEFT (all) then RIGHT (all)
        parts = []
        if L_text:
            parts.append(L_text)
        if R_text:
            # separate halves by a blank line if both present
            if parts:
                parts.append("")
            parts.append(R_text)
        article_text = "\n".join(parts).strip()


        rec = {"pdf_page": page_num, "excluded": False, "exclude_reason": (ex or ""), "article_text": article_text}
        records.append(rec)
        json_records.append(rec)

        # Markdown section: just the joined article text
        md_lines.append(f"## Page {page_num}\n")
        if article_text:
            md_lines.append(article_text + "\n")


    # === 저장: CSV + JSON + MD ===
    df = pd.DataFrame(records)
    if df.empty:
        df = pd.DataFrame(columns=["pdf_page","excluded","exclude_reason","article_text"])

    csv_path = out_dir / (articles_filename if articles_filename.endswith(".csv") else "page_articles.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[articles] saved CSV: {csv_path}")

    json_path = out_dir / (Path(articles_filename).with_suffix(".json").name if articles_filename.endswith(".csv") else "page_articles.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, ensure_ascii=False, indent=2)
    print(f"[articles] saved JSON: {json_path}")

    md_path = out_dir / (Path(articles_filename).with_suffix(".md").name if articles_filename.endswith(".csv") else "page_articles.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines).strip() + "\n")
    print(f"[articles] saved MD: {md_path}")

    # Flush gap log to CSV if enabled
    if getattr(args, "dump_gaps", False) and GAP_LOG:
        import csv
        gaps_csv = out_dir / getattr(args, "gaps_filename", "line_gap_metrics.csv")
        with open(gaps_csv, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["pdf_page","side","line_cy","num_spans","num_chunks","gaps","pos_gaps","small_ref","median_char_w","inline_gap_mult","threshold"])
            for rec in GAP_LOG:
                w.writerow([
                    rec.get("page",""),
                    rec.get("side",""),
                    round(rec.get("line_cy",0.0),2) if rec.get("line_cy") is not None else "",
                    rec.get("num_spans",""),
                    rec.get("num_chunks",""),
                    ";".join(str(x) for x in rec.get("gaps",[])),
                    ";".join(str(x) for x in rec.get("pos_gaps",[])),
                    rec.get("small_ref",""),
                    rec.get("median_char_w",""),
                    rec.get("inline_gap_mult",""),
                    rec.get("threshold",""),
                ])
        print(f"[diag] saved gap metrics: {gaps_csv}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o","--out", default="out_pdf49blog")
    ap.add_argument("--render-scale", type=float, default=2.5)
    ap.add_argument("--start-anchor", default="")
    ap.add_argument("--stop-anchor",  default="")
    # 제목 추출 파라미터
    ap.add_argument("--title-percentile", type=float, default=97.5)
    ap.add_argument("--title-top-percent", type=float, default=12.0)
    ap.add_argument("--wm-margin", type=float, default=0.5)
    ap.add_argument("--line-join-gap", type=float, default=9.0)
    ap.add_argument("--block-down-tol", type=float, default=1.2)
    ap.add_argument("--body-margin", type=float, default=0.8)
    ap.add_argument("--bold-guard",  type=float, default=0.5)
    # 제목 폰트 고정값 모드
    ap.add_argument("--title-base-size", type=float, default=None)
    ap.add_argument("--title-size-tol", type=float, default=0.8)
    # 제외
    ap.add_argument("--exclude", nargs="*", default=[])
    # 본문 추출 전용 워터마크 제거 스트립
    ap.add_argument("--article-wm-top-percent", type=float, default=18.0,
                    help="본문 추출 시 상단 워터마크 제거용 스트립 높이(%)")
    ap.add_argument("--wm-bottom-percent", type=float, default=12.0,
                    help="본문 추출 시 하단 BOM 넘버링 제거용 스트립 높이(%)")
    # 기사 본문 CSV 파일명
    ap.add_argument("--articles-filename", default="page_articles.csv")
    ap.add_argument("--col-gap-mult", type=float, default=2.8,
                    help="열 분할용 X-간격 임계 = median(span width) * 이 값")
    ap.add_argument("--min-col-width", type=float, default=40.0,
                    help="열 분할 최소 간격(PDF 좌표 단위). 이보다 좁으면 같은 열로 간주")
    ap.add_argument("--inline-gap-mult", type=float, default=3.0,
                    help="Split side-by-side paragraphs on the same row when X-gap >= median_small_gap * this (default 3.0)")
    ap.add_argument("--subtitle-band-frac", type=float, default=0.12,
                    help="Height fraction below detected title bbox to prioritize subtitle lines (default 0.12 = 12% of page height)")
    ap.add_argument("--first-body-band-frac", type=float, default=0.08,
                    help="(compat) Height fraction below subtitle band considered as early first-body region. Currently accepted for backward compatibility.")
    ap.add_argument("--col-valley-frac", type=float, default=0.10,
                    help="Histogram valley fraction for column splitting (default 0.10)")
    # --- 추가: 문단 분리/클러스터링 관련 ---
    # optional knobs for caption filter (safe defaults keep current behaviour)
    ap.add_argument("--caption-small-diff", type=float, default=1.1,
                    help="Consider a line caption-like if its avg font <= body_median - this value (pt)")
    ap.add_argument("--caption-max-chars", type=int, default=28,
                    help="Max character count for a line to be considered caption-like")
    # --- gap logging diagnostics ---
    ap.add_argument("--dump-gaps", action="store_true",
                    help="Log per-line inline X-gaps and thresholds for diagnostics")
    ap.add_argument("--gaps-filename", default="line_gap_metrics.csv",
                    help="Output CSV filename for --dump-gaps")
    ap.add_argument("--gap-max-lines", type=int, default=60, help="When --dump-gaps, maximum lines to log per (page,side) to avoid huge CSVs. Default 60")
    ap.add_argument("--no-stream-order", action="store_true",
                    help="Disable stream-order reconstruction and fall back to geometry-based column grouping.")
    ap.add_argument("--keep-title-in-body", action="store_true",
                    help="When set, keep title/subtitle sized words in article extraction (except watermark).")
    ap.add_argument("--min-sentence-chars", type=int, default=22,
                    help="Do not break a sentence if it would be shorter than this many characters (reflow).")
    ap.add_argument("--allow-quote-split", action="store_true",
                    help="Allow splitting at punctuation inside quotes during reflow (default: off).")
    ap.add_argument("--no-keep-qa-prefix", action="store_true",
                    help="Treat 'Q.'/'A.' like normal sentence terminators (default: keep as prefix).")

    ap.add_argument("--rescue-engine", choices=["off","plumber"], default="off",
                    help="Lead-band first-line rescue engine (default: off). Use 'plumber' to fix pages where the first line after title appears later in the stream.")

    # --- duplicate text mitigation knobs ---
    ap.add_argument("--dup-iou-tol", type=float, default=0.82,
                    help="IoU threshold for duplicate word boxes in overlap dedup (default 0.82)")
    ap.add_argument("--dup-center-tol", type=float, default=2.2,
                    help="Center-distance tolerance (PDF units) for duplicate word boxes (default 2.2)")
    ap.add_argument("--stutter-fix", dest="stutter_fix", action="store_true",
                    help="Collapse immediate duplicate tokens within a line (default on)")
    ap.add_argument("--no-stutter-fix", dest="stutter_fix", action="store_false")
    ap.set_defaults(stutter_fix=True)
    ap.add_argument("--stutter-min-pairs", type=int, default=2,
                    help="Minimum adjacent duplicate pairs needed in a line to trigger stutter collapse (default 2)")

    args = ap.parse_args()
    # propagate duplicate-removal thresholds to helpers
    dedup_words_by_overlap._iou_tol = float(getattr(args, "dup_iou_tol", 0.82))
    dedup_words_by_overlap._center_tol = float(getattr(args, "dup_center_tol", 2.2))
    print(f"[cfg] rescue-engine={getattr(args, 'rescue_engine', 'off')}")
    
    # 사용자가 --no-stream-order 안 줬으면 스트림 사용
    use_stream_order = (not args.no_stream_order)
    args.use_stream_order = use_stream_order

    setattr(args, "_force_no_geometry", True)
    setattr(args, "_disable_decorative_skip", False)  # enable decorative-page auto skip
    setattr(args, "_disable_caption_prune", True)
    setattr(args, "_disable_subtitle_band", False)

    # Reset global gap log to avoid cross-run contamination
    global GAP_LOG
    GAP_LOG = []

    # apply runtime knobs for inline chunk splitting and column valley detection
    cluster_and_join._inline_gap_mult = float(args.inline_gap_mult)
    # set gap log cap and reset counter
    cluster_and_join._gap_max_lines = int(args.gap_max_lines)
    global GAP_LOG_COUNT
    GAP_LOG_COUNT = {}
    global GROUP_VALLEY_FRAC
    GROUP_VALLEY_FRAC = float(args.col_valley_frac)

    # Ensure runtime attributes are visible for nested function
    setattr(args, "subtitle_band_frac", float(args.subtitle_band_frac))

    pdf_path=Path(args.pdf).expanduser().resolve()
    setattr(args, "_pdf_path", str(pdf_path))
    out_dir=Path(args.out).expanduser().resolve(); ensure_dir(out_dir)

    doc=fitz.open(pdf_path)

    # 시작/정지 앵커
    def compact_text(p: fitz.Page) -> str:
        return re.sub(r"\s+","", p.get_text("text") or "")
    start_idx=0
    if args.start_anchor:
        want=re.sub(r"\s+","", args.start_anchor)
        for i in range(doc.page_count):
            if want in compact_text(doc.load_page(i)):
                start_idx=i+1; break
    stop_idx=doc.page_count
    if args.stop_anchor:
        want=re.sub(r"\s+","", args.stop_anchor)
        for j in range(start_idx, doc.page_count):
            if want in compact_text(doc.load_page(j)):
                stop_idx=j; break
    if start_idx>=stop_idx:
        start_idx, stop_idx = 0, doc.page_count

    excludes = parse_excludes(args.exclude)

    # 제목 동적 임계 계산 시 내부 가드값
    extract_title_from_side._body_margin = float(args.body_margin)
    extract_title_from_side._bold_guard  = float(args.bold_guard)

    decisions=[]
    for p in range(start_idx, stop_idx):
        page=doc.load_page(p)
        L_rect, R_rect = page_halves_rect(page)

        ex = excludes.get(p+1)
        exL = (ex in ("L","ALL"))
        exR = (ex in ("R","ALL"))

        images=[]; title_text=""; title_side=""; title_max_pt=0.0; wm_max_pt=0.0

        if not exL and not exR:
            png = render_region_png(page, page.rect, args.render_scale)
            (out_dir / f"page-{p+1:03d}.png").write_bytes(png)
            images.append(f"page-{p+1:03d}.png")

            tL, ptL, wmL, tbL = extract_title_from_side(
                page, L_rect,
                args.title_percentile,
                args.title_base_size, args.title_size_tol,
                args.title_top_percent,
                args.wm_margin, args.line_join_gap, args.block_down_tol
            )
            tR, ptR, wmR, tbR = extract_title_from_side(
                page, R_rect,
                args.title_percentile,
                args.title_base_size, args.title_size_tol,
                args.title_top_percent,
                args.wm_margin, args.line_join_gap, args.block_down_tol
            )
            cand=[]
            if tL: cand.append(("L", tL, ptL, wmL, tbL))
            if tR: cand.append(("R", tR, ptR, wmR, tbR))
            if cand:
                cand.sort(key=lambda x:(x[2], len(x[1])), reverse=True)
                title_side, title_text, title_max_pt, wm_max_pt, title_bbox = cand[0]
            else:
                title_bbox = None
            # store title bbox keyed by (page, side), for use in body extraction ordering
            if title_side:
                setattr(args, "_title_bbox_map", getattr(args, "_title_bbox_map", {}))
                args._title_bbox_map[(p+1, title_side)] = title_bbox

            if title_side == "L":
                _spans = gather_spans_in_rect(page, L_rect)
            elif title_side == "R":
                _spans = gather_spans_in_rect(page, R_rect)
            else:
                _spans = []
            _wm_pt = detect_wm_font_size(_spans, page.rect.height, args.title_top_percent) if _spans else 0.0
            _stats = compute_body_stats(_spans, page.rect.height, args.title_top_percent) if _spans else {"reg_p90":0.0,"bold_p90":0.0}
            if args.title_base_size is not None:
                _thr = args.title_base_size - args.title_size_tol
            else:
                _perc  = percentile([s["size"] for s in _spans], args.title_percentile) if _spans else 0.0
                _thr   = max(_perc, _wm_pt + args.wm_margin, _stats["reg_p90"] + args.body_margin, _stats.get("bold_p90",0.0) + args.bold_guard)
        else:
            if ex!="ALL":
                if not exL:
                    png = render_region_png(page, L_rect, args.render_scale)
                    fn=f"page-{p+1:03d}-left.png"; (out_dir/fn).write_bytes(png); images.append(fn)
                    tL, ptL, wmL, tbL = extract_title_from_side(
                        page, L_rect,
                        args.title_percentile,
                        args.title_base_size, args.title_size_tol,
                        args.title_top_percent,
                        args.wm_margin, args.line_join_gap, args.block_down_tol
                    )
                    if tL:
                        title_side, title_text, title_max_pt, wm_max_pt, title_bbox = "L", tL, ptL, wmL, tbL
                        # store title bbox for this page/side
                        setattr(args, "_title_bbox_map", getattr(args, "_title_bbox_map", {}))
                        args._title_bbox_map[(p+1, title_side)] = title_bbox
                    if args.title_base_size is not None:
                        _thr = args.title_base_size - args.title_size_tol
                        _wm_pt = wmL
                        _stats = compute_body_stats(gather_spans_in_rect(page, L_rect), page.rect.height, args.title_top_percent)
                    else:
                        _spans = gather_spans_in_rect(page, L_rect)
                        _wm_pt = detect_wm_font_size(_spans, page.rect.height, args.title_top_percent) if _spans else 0.0
                        _stats = compute_body_stats(_spans, page.rect.height, args.title_top_percent) if _spans else {"reg_p90":0.0,"bold_p90":0.0}
                        _perc  = percentile([s["size"] for s in _spans], args.title_percentile) if _spans else 0.0
                        _thr   = max(_perc, _wm_pt + args.wm_margin, _stats["reg_p90"] + args.body_margin, _stats.get("bold_p90",0.0) + args.bold_guard)
                if not exR:
                    png = render_region_png(page, R_rect, args.render_scale)
                    fn=f"page-{p+1:03d}-right.png"; (out_dir/fn).write_bytes(png); images.append(fn)
                    tR, ptR, wmR, tbR = extract_title_from_side(
                        page, R_rect,
                        args.title_percentile,
                        args.title_base_size, args.title_size_tol,
                        args.title_top_percent,
                        args.wm_margin, args.line_join_gap, args.block_down_tol
                    )
                    if tR and (ptR>title_max_pt or (ptR==title_max_pt and len(tR)>len(title_text))):
                        title_side, title_text, title_max_pt, wm_max_pt, title_bbox = "R", tR, ptR, wmR, tbR
                        setattr(args, "_title_bbox_map", getattr(args, "_title_bbox_map", {}))
                        args._title_bbox_map[(p+1, title_side)] = title_bbox
                    if args.title_base_size is not None:
                        _thr = args.title_base_size - args.title_size_tol
                        _wm_pt = wmR
                        _stats = compute_body_stats(gather_spans_in_rect(page, R_rect), page.rect.height, args.title_top_percent)
                    else:
                        _spans = gather_spans_in_rect(page, R_rect)
                        _wm_pt = detect_wm_font_size(_spans, page.rect.height, args.title_top_percent) if _spans else 0.0
                        _stats = compute_body_stats(_spans, page.rect.height, args.title_top_percent) if _spans else {"reg_p90":0.0,"bold_p90":0.0}
                        _perc  = percentile([s["size"] for s in _spans], args.title_percentile) if _spans else 0.0
                        _thr   = max(_perc, _wm_pt + args.wm_margin, _stats["reg_p90"] + args.body_margin, _stats.get("bold_p90",0.0) + args.bold_guard)
            else:
                _wm_pt = 0.0; _stats = {"reg_p90":0.0,"bold_p90":0.0}; _thr = 0.0

        decisions.append({
            "pdf_page": p+1,
            "excluded": bool(ex is not None),
            "exclude_reason": (ex or ""),
            "images": ";".join(images),
            "title_side": title_side,
            "title_text": title_text,
            "title_max_pt": round(title_max_pt,2),
            "wm_max_pt": round(wm_max_pt,2),
            "title_thresh_pt": round(_thr,2),
            "body_reg_p90": round(_stats.get("reg_p90",0.0),2),
            "body_bold_p90": round(_stats.get("bold_p90",0.0),2),
            "title_base_size": None if args.title_base_size is None else round(float(args.title_base_size), 2),
            "title_size_tol": round(float(args.title_size_tol), 2),
        })

    pd.DataFrame(decisions).to_csv(out_dir/"page_decisions.csv", index=False, encoding="utf-8-sig")

    # 본문 추출 (워터마크 라인 제거 + 제목 크기 대역 제외 + 장식문구 제외)
    # pass column parameters into build_side_text via attributes
    # (the nested function will read these if present)
    # NOTE: we attach to the function object after its definition, so set again inside scope.
    extract_article_texts(
        doc=doc,
        out_dir=out_dir,
        start_idx=start_idx,
        stop_idx=stop_idx,
        excludes=excludes,
        start_anchor=args.start_anchor,
        stop_anchor=args.stop_anchor,
        title_base_size=args.title_base_size,
        title_size_tol=args.title_size_tol,
        title_percentile=args.title_percentile,
        title_top_percent=args.title_top_percent,
        line_join_gap=args.line_join_gap,
        block_down_tol=args.block_down_tol,
        wm_margin=args.wm_margin,
        article_wm_top_percent=args.article_wm_top_percent,
        wm_bottom_percent=args.wm_bottom_percent,
        col_gap_mult=args.col_gap_mult,
        min_col_width=args.min_col_width,
        articles_filename=args.articles_filename,
        args=args,
    )

    print(f"[done] saved to: {out_dir}")
    print(f"[pdf_to_blog] v6.1 — stream-order + lead-band rescue={args.rescue_engine}")

if __name__ == "__main__":
    main()