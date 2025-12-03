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
# optional clustering (DBSCAN) for paragraph blocks
import numpy as np
try:
    # Optional: only used if scikit-learn is installed
    from sklearn.cluster import DBSCAN  # type: ignore
    _SKLEARN_OK = True
except ImportError:
    DBSCAN = None  # fallback stub
    _SKLEARN_OK = False

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

from math import isfinite  # ← imports 근처에 추가해도 됨

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
    """같은 텍스트가 거의 같은 위치(bbox)로 두 번 이상 찍힌 경우 하나로 합친다.
    PDF에 텍스트 레이어가 2중으로 올라가 있는 경우를 강제로 정리하기 위한 단계다.
    
    기준:
    - text 가 동일하고
    - 중심점 차이가 center_tol 이하이거나, bbox IoU가 iou_tol 이상이면 중복으로 본다.
    - 중복일 때는 size(글꼴 크기)가 더 큰 쪽을 keep 한다.
    - 마지막에 (block, line, word) 순으로 다시 정렬해 스트림 순서를 유지한다.
    """
    if not words:
        return words

    def _cx(b):
        return (b[0] + b[2]) / 2.0

    def _cy(b):
        return (b[1] + b[3]) / 2.0

    used = [False] * len(words)
    out: List[Dict[str, Any]] = []
    for i, wi in enumerate(words):
        if used[i]:
            continue
        bi = wi["bbox"]
        ti = (wi.get("text") or "").strip()
        keep = wi
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
                # 중복 → 더 큰 글꼴/상자를 남긴다
                if float(wj.get("size", 0.0) or 0.0) > float(keep.get("size", 0.0) or 0.0):
                    keep = wj
                used[j] = True
        out.append(keep)
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

from statistics import median

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

BOTTOM_NUM_RE = re.compile(r"\bBOM\s*\d{1,3}\b", re.IGNORECASE)

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

        if not drop:
            keep_spans.extend(ln)

    keep_spans.sort(key=lambda s:(round(cy(s),2), s["bbox"][0]))
    return keep_spans

def _split_spans_quadrants(spans: List[Dict[str,Any]], page_rect: fitz.Rect) -> Dict[str, List[Dict[str,Any]]]:
    """
    Split spans into quadrants based on center point:
      - 'LT': Left-Top
      - 'LB': Left-Bottom
      - 'RT': Right-Top
      - 'RB': Right-Bottom
    """
    quads = {"LT": [], "LB": [], "RT": [], "RB": []}
    xm = (page_rect.x0 + page_rect.x1) / 2.0
    ym = (page_rect.y0 + page_rect.y1) / 2.0

    for s in spans:
        bb = s.get("bbox", (0,0,0,0))
        cx = (bb[0] + bb[2]) / 2.0
        cy = (bb[1] + bb[3]) / 2.0
        if cx <= xm and cy <= ym:
            quads["LT"].append(s)
        elif cx <= xm and cy > ym:
            quads["LB"].append(s)
        elif cx > xm and cy <= ym:
            quads["RT"].append(s)
        else:
            quads["RB"].append(s)
    return quads

def _span_center_x(s): 
    bb = s["bbox"]; 
    return (bb[0]+bb[2])/2.0

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

# --- Paragraph splitting helpers ---
def _median_line_gap(spans: List[Dict[str,Any]]) -> float:
    if not spans or len(spans) < 2:
        return 12.0
    def cy(s): bb=s["bbox"]; return (bb[1]+bb[3])/2.0
    ys = sorted([cy(s) for s in spans])
    gaps = [ys[i+1]-ys[i] for i in range(len(ys)-1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps:
        return 12.0
    gaps.sort()
    return gaps[len(gaps)//2]

def split_into_paragraphs_by_y(spans: List[Dict[str,Any]],
                               gap_mult: float = 1.6) -> List[List[Dict[str,Any]]]:
    "Y-간격 기반 문단 나누기 (위→아래 정렬 후, 큰 간격에서 단락 분리)"
    if not spans:
        return []
    def cy(s): bb=s["bbox"]; return (bb[1]+bb[3])/2.0
    spans = sorted(spans, key=lambda s: (round(cy(s),2), s["bbox"][0]))
    med_gap = _median_line_gap(spans)
    thr = med_gap * gap_mult
    paras: List[List[Dict[str,Any]]] = [[spans[0]]]
    for s in spans[1:]:
        prev = paras[-1][-1]
        if (cy(s) - cy(prev)) > thr:
            paras.append([s])
        else:
            paras[-1].append(s)
    return paras

def split_into_paragraphs_dbscan(spans: List[Dict[str,Any]],
                                 eps: float = 28.0,
                                 min_samples: int = 5) -> List[List[Dict[str,Any]]]:
    "DBSCAN으로 (cx, cy) 공간에서 문단 클러스터링. scikit-learn 없으면 휴리스틱으로 폴백."
    if not spans:
        return []
    if not _SKLEARN_OK:
        return split_into_paragraphs_by_y(spans, gap_mult=1.6)
    def cx(s): bb=s["bbox"]; return (bb[0]+bb[2])/2.0
    def cy(s): bb=s["bbox"]; return (bb[1]+bb[3])/2.0
    pts = np.array([[cx(s), cy(s)] for s in spans], dtype=float)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pts)
    clusters: Dict[int, List[Dict[str,Any]]] = {}
    for lab, s in zip(labels, spans):
        clusters.setdefault(int(lab), []).append(s)
    # -1 label is noise: attach to nearest non-noise cluster or keep as single paragraph
    if -1 in clusters and len(clusters) > 1:
        noise = clusters.pop(-1)
        for s in noise:
            # attach to cluster with nearest center
            centers = {k: np.mean([[cx(x), cy(x)] for x in v], axis=0) for k,v in clusters.items()}
            if not centers:
                clusters.setdefault(0, []).append(s); continue
            dists = {k: float(np.linalg.norm(np.array([cx(s),cy(s)])-centers[k])) for k in centers}
            best = min(dists, key=dists.get)
            clusters[best].append(s)
    # order clusters: left→right by mean x, inside each: top→bottom
    ordered = sorted(clusters.values(),
                     key=lambda cl: sum((s["bbox"][0]+s["bbox"][2])/2.0 for s in cl)/len(cl))
    for i, cl in enumerate(ordered):
        ordered[i] = sorted(cl, key=lambda s: ((s["bbox"][1]+s["bbox"][3])/2.0, s["bbox"][0]))
    # finally, split each cluster into Y-paragraphs (to avoid over-merge)
    paras_all: List[List[Dict[str,Any]]] = []
    for cl in ordered:
        paras_all.extend(split_into_paragraphs_by_y(cl, gap_mult=1.2))
    return paras_all

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

    samples_fh = None
    if getattr(args, "dump_layout_samples", False):
        samples_path = out_dir / getattr(args, "samples_filename", "layout_samples.jsonl")
        samples_fh = open(samples_path, "a", encoding="utf-8")

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
        def build_side_text(raw_spans: List[Dict[str,Any]], side_tag: str) -> Tuple[str, bool]:
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
            if getattr(args, "dump_gaps", False):
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
            cap_small_diff = getattr(args, "caption_small_diff", 1.1)
            cap_max_chars  = getattr(args, "caption_max_chars", 28)

            pruned_words: List[Dict[str,Any]] = []
            for key, ws in sorted(lines_by_key.items()):
                ws.sort(key=lambda d: (d.get("word",0), d["bbox"][0]))
                line_text = " ".join(w["text"] for w in ws).strip()
                if not line_text:
                    continue
                avg_size = float(np.mean([float(w.get("size",0.0) or 0.0) for w in ws]))
                short_enough = len(re.sub(r"\s+","", line_text)) <= cap_max_chars
                bullet_like  = bool(CAP_BULLET_RE.match(line_text))
                tiny_vs_body = (body_med > 0 and (avg_size <= max(0.0, body_med - cap_small_diff)))
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
            title_bbox_map = getattr(args, "_title_bbox_map", {})
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
            drop_title_band = not getattr(args, "keep_title_in_body", False)

            # (4) 제목 bbox를 못 잡은 페이지는 "위쪽 35%"를 한번 더 깎아서 제목/띠지를 날린다
            if drop_title_band and title_bbox is None:
                strict_top_y = page.rect.y0 + page.rect.height * 0.35
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

                # (6-1) 제목 bbox 못 잡았고, 위쪽 35% 안에 있고, 크기가 제목대역이면 버림
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

            # (6-5) 제목 바로 아래 밴드를 강제로 제일 앞으로 보내기 (4pL 문제 해결부)
            if title_bbox and not getattr(args, "_disable_subtitle_band", False):
                band_frac = float(getattr(args, "subtitle_band_frac", 0.12))
                band_y = min(page.rect.y1, title_bbox[3] + page.rect.height * band_frac)
                upper_words: List[Dict[str, Any]] = []
                lower_words: List[Dict[str, Any]] = []
                # 먼저 band 영역을 전부 나눠 담는다
                for w in content_words:
                    wyc = (w["bbox"][1] + w["bbox"][3]) / 2.0
                    if title_bbox[3] <= wyc <= band_y:
                        upper_words.append(w)
                    else:
                        lower_words.append(w)
                # band 안에 뭔가 있으면 subtitle-like 인지 먼저 검사
                if upper_words:
                    # 같은 줄(=block,line) 단위로 합쳐서 부제목 성향을 본다
                    tmp_by_key: Dict[Tuple[int,int], List[Dict[str,Any]]] = {}
                    for uw in upper_words:
                        tmp_by_key.setdefault((uw.get("block",0), uw.get("line",0)), []).append(uw)
                    # 부제목처럼 보이는 라인만 추림
                    subtitle_lines: List[List[Dict[str,Any]]] = []
                    for (_bk,_ln), uw_list in tmp_by_key.items():
                        uw_list.sort(key=lambda d: d["bbox"][0])
                        line_txt = " ".join(u["text"] for u in uw_list).strip()
                        if _is_subtitle_like(line_txt):
                            subtitle_lines.append(uw_list)
                        else:
                            # 부제목이 아니면 본문 쪽으로 돌려보낸다
                            lower_words.extend(uw_list)
                    if subtitle_lines:
                        # 실제로 부제목처럼 보이는 라인만 앞으로 당긴다
                        if lower_words:
                            min_block = min(w.get("block", 0) for w in lower_words)
                        else:
                            min_block = 0
                        subtitle_block = min_block - 10
                        ordered_subs: List[Dict[str,Any]] = []
                        for idx, uw_list in enumerate(sorted(subtitle_lines, key=lambda lst: ( (lst[0]["bbox"][1]+lst[0]["bbox"][3])/2.0, lst[0]["bbox"][0] ))):
                            for j, uw in enumerate(uw_list):
                                uw["block"] = subtitle_block
                                uw["line"] = idx  # 같은 부제목 묶음은 같은 line으로
                                ordered_subs.append(uw)
                        content_words = ordered_subs + lower_words
                    else:
                        # 부제목처럼 보이는 게 없으면 원래 순서 유지
                        content_words = upper_words + lower_words
                else:
                    content_words = content_words
            # 6) 최종 합성
            if getattr(args, "use_stream_order", True):
                text = _build_text_from_stream_words(
                    content_words,
                    inline_gap_mult=getattr(args, "inline_gap_mult", 3.0)
                )
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
                return ("\n\n".join(col_texts).strip(), deco)
            
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
            L_text, L_deco = build_side_text(L_raw, "L")

        if not skip_R:
            R_raw = gather_spans_in_rect(page, R_rect)
            R_text, R_deco = build_side_text(R_raw, "R")

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

    # close samples file if it was opened
    if samples_fh is not None:
        samples_fh.close()

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
    ap.add_argument("--big-col-min-spans", type=int, default=3,
                    help="Minimum spans to consider a column 'big' when deciding geometry mode (default 3)")
    ap.add_argument("--col-valley-frac", type=float, default=0.10,
                    help="Histogram valley fraction for column splitting (default 0.10)")
    # --- 추가: 문단 분리/클러스터링 관련 ---
    ap.add_argument("--layout-mode", choices=["heuristic","dbscan"], default="heuristic",
                    help="문단 분리 알고리즘: heuristic(기본) 또는 dbscan(옵션, scikit-learn 필요)")
    ap.add_argument("--para-gap-mult", type=float, default=1.6,
                    help="문단 분리용 Y-간격 배수 (기본 1.6, 값이 크면 문단이 덜 쪼개짐)")
    ap.add_argument("--dbscan-eps", type=float, default=28.0,
                    help="DBSCAN epsilon (PDF 좌표 단위)")
    ap.add_argument("--dbscan-min-samples", type=int, default=5,
                    help="DBSCAN min_samples")
    # optional: dump layout samples for future learning (OFF by default)
    ap.add_argument("--dump-layout-samples", action="store_true",
                    help="If set, dump per-line layout features/weak labels to JSONL for training.")
    ap.add_argument("--samples-filename", default="layout_samples.jsonl",
                    help="Filename for dumped layout samples (JSONL).")
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
    ap.add_argument("--single-flow-fallback", action="store_true", help="If most lines look like a single flow (no stable split), bypass column splitting for that side.")
    ap.add_argument("--single-flow-min-onechunk", type=float, default=0.72, help="Trigger single-flow when ratio of lines with exactly one chunk >= this (0~1). Default 0.72")
    ap.add_argument("--gap-max-lines", type=int, default=60, help="When --dump-gaps, maximum lines to log per (page,side) to avoid huge CSVs. Default 60")
    ap.add_argument("--no-stream-order", action="store_true",
                    help="Disable stream-order reconstruction and fall back to geometry-based column grouping.")
    ap.add_argument("--keep-title-in-body", action="store_true",
                    help="When set, keep title/subtitle sized words in article extraction (except watermark).")
    args = ap.parse_args()

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
    setattr(args, "big_col_min_spans", int(args.big_col_min_spans))

    pdf_path=Path(args.pdf).expanduser().resolve()
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

    # expose column params to build_side_text
    extract_article_texts_build_side_min_col_w = float(args.min_col_width)
    extract_article_texts_build_side_gap_mult = float(args.col_gap_mult)

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

if __name__=="__main__":
    main()