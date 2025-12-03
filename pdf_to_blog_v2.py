#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf_to_blog_v2.py — 이미지 캡처 버전 + 제목 추출(워터마크 완전 배제 & exclude 일치)

- 워터마크 스캔/템플릿 매칭 로직: 제거
- exclude(예: 9pr, 37p, 40pl): 캡처/제목 모두 제외
- 제목 추출 규칙:
  * 허용된 반쪽(rect) 내부 텍스트만 사용
  * 상단 스트립에서 워터마크(예: "BOM | CAR/NEWS/...","BOM") 글꼴 크기 측정 → wm_max_pt
  * (신규) --title-base-size 가 지정되면: [기준값 ± title-size-tol] 범위의 글꼴만 제목 후보
    - WM 패턴 및 'BOM' 문구는 무조건 제외
    - 줄바꿈/굵기 섞여도 결합 (line-join-gap, block-down-tol)
    - 중복 라인/문장 제거
  * (미지정 시) 기존 퍼센타일 기반 동작 유지
- page_decisions.csv: title_side/title_text/title_max_pt/wm_max_pt/excluded/exclude_reason 기록
  + (신규) title_base_size, title_size_tol 기록
  * (신규) 이중 레이어 중복 제거: 같은 위치/유사 텍스트 스팬은 하나만 유지(IoU/중심거리/토큰유사 기반)
"""

import re, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import fitz  # PyMuPDF
import pandas as pd

WM_TOP_PAT = re.compile(
    r"""^BOM\s*([|│\|l]\s*(NEW\s*CAR|NEWS|CAR|MOTORRAD|TRAVEL|GOLF|CULTURE))?$""",
    re.IGNORECASE
)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def parse_excludes(tokens: List[str]) -> Dict[int, str]:
    """
    tokens 예: ["7pr","9pr","11pr","19pr","21pr","23pr","37p","39pr"]
    반환: {7:"R", 9:"R", 11:"R", 19:"R", 21:"R", 23:"R", 37:"ALL", 39:"R"}
    """
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

# === 이중 레이어(겹침/유사 텍스트) 중복 스팬 제거 ===
def _bbox_iou(a: Tuple[float,float,float,float], b: Tuple[float,float,float,float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0); ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

def _norm_text_for_cmp(t: str) -> str:
    # 대소문자/공백/연속 공백 제거, 유사 구분자는 그대로 둠
    return re.sub(r"\s+", "", (t or "").strip().lower())

def _token_jaccard(a: str, b: str) -> float:
    sa = set(re.sub(r"\W+", " ", a.lower()).split())
    sb = set(re.sub(r"\W+", " ", b.lower()).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def dedup_overlapping_spans(spans: List[Dict[str,Any]],
                            iou_tol: float = 0.85,
                            center_tol: float = 1.2,
                            text_sim_tol: float = 0.90) -> List[Dict[str,Any]]:
    """
    PDF 이중 레이어(같은 위치/내용이 겹치는 스팬) 제거용 전처리.
    - bbox IoU가 높거나 중심점이 거의 같은 스팬끼리 묶음
    - 텍스트가 동일(공백무시)하거나 토큰 Jaccard 유사도가 높은 경우만 병합
    - 대표 스팬 선택 규칙: size 큰 것 > 볼드(flags&2) > bbox 높이 큰 것
    """
    if not spans:
        return spans
    # 정렬(위→아래, 좌→우)
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
            # 중심이 거의 같거나, IoU가 매우 클 때만 후보
            if (dx <= center_tol and dy <= center_tol) or (iou >= iou_tol):
                # 텍스트 유사성 확인
                tj_norm = _norm_text_for_cmp(sj["text"])
                if ti_norm and tj_norm and (ti_norm == tj_norm or _token_jaccard(si["text"], sj["text"]) >= text_sim_tol):
                    gi.append(j); used[j] = True
        groups.append(gi)

    result: List[Dict[str,Any]] = []
    for gi in groups:
        if len(gi) == 1:
            result.append(srt[gi[0]])
            continue
        # 대표 스팬 선택
        best_idx = gi[0]
        def score(span):
            size = float(span.get("size",0.0) or 0.0)
            bold = 1 if (int(span.get("flags",0)) & 2) != 0 else 0
            bb = span["bbox"]; height = (bb[3]-bb[1])
            # size 우선, 다음 bold, 다음 bbox 높이
            return (size, bold, height)
        for j in gi[1:]:
            if score(srt[j]) > score(srt[best_idx]):
                best_idx = j
        result.append(srt[best_idx])

    # 원래 순서 근사 유지: y,x 기준으로 재정렬
    result.sort(key=lambda s:(round(cy(s),2), round(cx(s),2)))
    return result

def percentile(vals: List[float], p: float) -> float:
    if not vals: return 0.0
    vals=sorted(vals)
    k=max(0, min(len(vals)-1, int(round((p/100.0)*(len(vals)-1)))))
    return float(vals[k])

def detect_wm_font_size(spans: List[Dict[str,Any]], page_h: float, top_percent: float) -> float:
    """
    상단 top_percent% 스트립 안에서 WM_TOP_PAT 매치 or 'BOM' 포함 텍스트의 최대 글꼴 크기
    """
    y1 = page_h * (top_percent/100.0)
    cand=[s["size"] for s in spans
          if (s["bbox"][1] <= y1) and (WM_TOP_PAT.match(s["text"]) or "BOM" in s["text"].upper())]
    return max(cand) if cand else 0.0

def compute_body_stats(spans: List[Dict[str,Any]], page_h: float, top_percent: float) -> Dict[str, float]:
    """
    워터마크 상단 스트립을 제외한 본문 영역의 글꼴 크기 통계:
    - all_p90: 전체 span size의 90퍼센타일
    - reg_p90: 일반(비볼드) span size의 90퍼센타일
    - bold_p90: 볼드 span size의 90퍼센타일
    """
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
    return {
        "all_p90": p90(sizes_all),
        "reg_p90": p90(sizes_reg),
        "bold_p90": p90(sizes_bold)
    }

def cluster_and_join(spans: List[Dict[str,Any]], join_gap: float, down_tol: float) -> str:
    """
    y(상단) 기준 정렬 후, 인접 라인을 병합. 줄바꿈 유지.
    join_gap: 같은 줄/이웃 줄로 묶을 가로·세로 허용폭(픽셀/포인트 혼용 → bbox 기준)
    down_tol: 다음 줄의 y가 현재 줄보다 얼마나 내려가 있어도 같은 블록으로 볼지 계수
    """
    if not spans: return ""
    # 라인 중심 y로 정렬
    def cy(s): bb=s["bbox"]; return (bb[1]+bb[3])/2.0
    spans=sorted(spans, key=lambda s:(round(cy(s),2), s["bbox"][0]))

    blocks=[[spans[0]]]
    for s in spans[1:]:
        prev=blocks[-1][-1]
        same_line = abs(cy(s)-cy(prev)) <= join_gap
        next_line = (cy(s) > cy(prev)) and ((cy(s)-cy(prev)) <= join_gap*down_tol)
        same_row = same_line or next_line
        if same_row:
            blocks[-1].append(s)
        else:
            blocks.append([s])

    lines=[]
    for blk in blocks:
        blk = sorted(blk, key=lambda s:s["bbox"][0])
        line = " ".join(s["text"] for s in blk)
        lines.append(line.strip())

    # 1) 완전 동일 중복 제거  2) 토큰 유사도 높은 중복 제거(간단)
    seen=set(); dedup=[]
    for ln in lines:
        norm=re.sub(r"\s+"," ", ln.strip().lower())
        if norm in seen: continue
        seen.add(norm)
        dedup.append(ln)
    # 토큰 유사도(70% 이상) 중복 제거
    final=[]
    def jaccard(a,b):
        sa=set(a.lower().split()); sb=set(b.lower().split())
        if not sa or not sb: return 0.0
        return len(sa&sb)/len(sa|sb)
    for ln in dedup:
        if final and jaccard(final[-1], ln) >= 0.7:
            continue
        final.append(ln)

    # 추가: 강력한 중복 필터링 (부분문자열 포함 및 거의 동일)
    def is_substring_similar(a,b):
        a_low=a.lower()
        b_low=b.lower()
        if a_low in b_low or b_low in a_low:
            tokens_a=set(a_low.split())
            tokens_b=set(b_low.split())
            if not tokens_a or not tokens_b:
                return False
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
                            down_tol: float) -> Tuple[str, float, float]:
    """
    반환: (title_text, title_max_pt, wm_max_pt)
    - (신규) title_base_size 지정 시: [base ± tol] 사이즈만 후보
    - 미지정 시: 기존 퍼센타일 기반 + 동적 최소치 사용
    """
    spans = gather_spans_in_rect(page, side_rect)
    # 이중 레이어 중복 스팬 제거(좌표/텍스트 유사 기반)
    spans = dedup_overlapping_spans(spans, iou_tol=0.85, center_tol=1.2, text_sim_tol=0.90)
    if not spans:
        return ("", 0.0, 0.0)
    sizes = [s["size"] for s in spans]

    # 워터마크 최대 폰트
    wm_max_pt = detect_wm_font_size(spans, page.rect.height, wm_top_percent)

    # 본문(상단 스트립 제외) 통계
    body_stats = compute_body_stats(spans, page.rect.height, wm_top_percent)
    body_reg_p90 = body_stats["reg_p90"]
    body_bold_p90 = body_stats["bold_p90"]

    # 기본 퍼센타일
    perc_pt = percentile(sizes, title_percentile)

    # base-size 모드 여부
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
        # 동적 최소 제목 임계 계산(기존 로직 유지)
        body_margin = getattr(extract_title_from_side, "_body_margin", 0.8)
        bold_guard  = getattr(extract_title_from_side, "_bold_guard", 0.5)
        dyn_min_pt = max(
            perc_pt,
            wm_max_pt + wm_margin,
            body_reg_p90 + body_margin,
            body_bold_p90 + (bold_guard)
        )
        cand = [s for s in spans
                if (s["size"] >= dyn_min_pt)
                and (not WM_TOP_PAT.match(s["text"]))
                and ("BOM" not in s["text"].upper())]
        is_max_mode = (title_percentile >= 99.5)

    if not cand:
        return ("", 0.0, wm_max_pt)

    joined_text = cluster_and_join(cand, join_gap, down_tol)
    lines = joined_text.split("\n") if joined_text else []
    if not lines:
        return ("", 0.0, wm_max_pt)

    # 볼드-only + 짧은 라인 제거(25p 부제목 억제용)
    def is_bold_only_line(line_text: str) -> bool:
        tokens = line_text.strip()
        if not tokens:
            return False
        for s in cand:
            if s["text"].strip() == tokens and (int(s.get("flags",0)) & 2) != 0:
                return True
        bold_texts = "".join(s["text"].strip() for s in cand if (int(s.get("flags",0)) & 2) != 0)
        if bold_texts.replace(" ","") == tokens.replace(" ",""):
            return True
        return False

    # max 모드일 때 전체가 볼드-only이면서 매우 짧으면 통째로 배제
    if is_max_mode:
        all_bold = all((int(s.get("flags",0)) & 2) != 0 for s in cand)
        total_len = sum(len(s["text"].strip()) for s in cand)
        if all_bold and total_len < 15:
            return ("", 0.0, wm_max_pt)

    filtered_lines = []
    for ln in lines:
        if len(ln.strip()) < 15 and is_bold_only_line(ln):
            continue
        filtered_lines.append(ln)

    if not filtered_lines:
        return ("", 0.0, wm_max_pt)

    # --- 추가: 라인 중복(부분문자열 포함, 높은 유사도) 및 연속 유사 라인 병합 ---
    def jaccard_tokens(a, b):
        sa = set(re.sub(r"\W+", " ", a.lower()).split())
        sb = set(re.sub(r"\W+", " ", b.lower()).split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    def is_highly_similar(a, b):
        # Jaccard 유사도 0.85 이상이거나, 한쪽이 다른쪽의 부분문자열이면서 토큰 유사도 높음
        jac = jaccard_tokens(a, b)
        if jac >= 0.85:
            return True
        a_low = a.lower().strip()
        b_low = b.lower().strip()
        if a_low in b_low or b_low in a_low:
            # 토큰 유사도 체크
            if jac >= 0.7:
                return True
        return False

    # 1. 연속적으로 거의 동일한 라인 merge (collapse)
    collapsed = []
    for ln in filtered_lines:
        if collapsed and is_highly_similar(collapsed[-1], ln):
            continue
        collapsed.append(ln)

    # 2. 전체 중복(부분문자열 유사 포함) 제거 (이전 라인들과 중복이면 skip)
    deduped = []
    for ln in collapsed:
        is_dup = False
        for prev in deduped:
            if is_highly_similar(prev, ln):
                is_dup = True
                break
        if not is_dup:
            deduped.append(ln)

    title = "\n".join(deduped).strip()
    max_pt = max([s["size"] for s in cand], default=0.0)
    return (title, max_pt, wm_max_pt)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o","--out", default="out_pdf49blog")
    ap.add_argument("--render-scale", type=float, default=2.5)
    ap.add_argument("--start-anchor", default="")
    ap.add_argument("--stop-anchor",  default="")
    # 제목 추출 파라미터
    ap.add_argument("--title-percentile", type=float, default=97.5, help="제목으로 볼 최소 글꼴 상위 퍼센타일(%)")
    ap.add_argument("--title-top-percent", type=float, default=12.0, help="워터마크 측정용 상단 스트립 높이(%)")
    ap.add_argument("--wm-margin", type=float, default=0.5, help="워터마크보다 최소 몇 pt 커야 제목으로 인정할지")
    ap.add_argument("--line-join-gap", type=float, default=9.0, help="줄 병합 허용 간격")
    ap.add_argument("--block-down-tol", type=float, default=1.2, help="아래 줄 병합 계수")
    ap.add_argument("--body-margin", type=float, default=0.8, help="본문(비볼드) 상한 대비 최소 여유 pt")
    ap.add_argument("--bold-guard",  type=float, default=0.5, help="본문 볼드 상한 대비 최소 여유 pt(볼드 인플레 방지)")
    # (신규) 제목 폰트 고정값 모드
    ap.add_argument("--title-base-size", type=float, default=None, help="제목 기준 폰트 크기(pt). 지정하면 퍼센타일 대신 이 값을 기준으로 추출")
    ap.add_argument("--title-size-tol", type=float, default=0.8, help="기준 폰트 크기 대비 허용 오차(pt)")
    # 제외
    ap.add_argument("--exclude", nargs="*", default=[], help="예: 7pr 9pr 37p 40pl")
    args=ap.parse_args()

    pdf_path=Path(args.pdf).expanduser().resolve()
    out_dir=Path(args.out).expanduser().resolve(); ensure_dir(out_dir)

    doc=fitz.open(pdf_path)

    # 시작/정지 앵커(텍스트 레이어 포함 검색, 없으면 전체)
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

    # 동적 임계 내부에서 사용할 가드값 주입
    extract_title_from_side._body_margin = float(args.body_margin)
    extract_title_from_side._bold_guard  = float(args.bold_guard)

    decisions=[]
    for p in range(start_idx, stop_idx):
        page=doc.load_page(p)
        L_rect, R_rect = page_halves_rect(page)

        ex = excludes.get(p+1)  # 1-based
        exL = (ex in ("L","ALL"))
        exR = (ex in ("R","ALL"))

        images=[]; title_text=""; title_side=""; title_max_pt=0.0; wm_max_pt=0.0
        exclude_reason=""

        if not exL and not exR:
            # 둘 다 허용 → 전체 캡처 1장
            png = render_region_png(page, page.rect, args.render_scale)
            (out_dir / f"page-{p+1:03d}.png").write_bytes(png)
            images.append(f"page-{p+1:03d}.png")

            # 좌/우 각각 추출 후 더 “제목다운” 쪽 채택
            tL, ptL, wmL = extract_title_from_side(
                page, L_rect,
                args.title_percentile,
                args.title_base_size, args.title_size_tol,
                args.title_top_percent,
                args.wm_margin, args.line_join_gap, args.block_down_tol
            )
            tR, ptR, wmR = extract_title_from_side(
                page, R_rect,
                args.title_percentile,
                args.title_base_size, args.title_size_tol,
                args.title_top_percent,
                args.wm_margin, args.line_join_gap, args.block_down_tol
            )
            cand = []
            if tL: cand.append(("L", tL, ptL, wmL))
            if tR: cand.append(("R", tR, ptR, wmR))
            if cand:
                cand.sort(key=lambda x:(x[2], len(x[1])), reverse=True)
                title_side, title_text, title_max_pt, wm_max_pt = cand[0]

            # 디버깅용 임계 기록 (선택된 쪽 기준; base-size 모드면 low를 기록)
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
            # 한쪽/전체 제외 상태
            if ex!="ALL":
                if not exL:
                    png = render_region_png(page, L_rect, args.render_scale)
                    fn=f"page-{p+1:03d}-left.png"; (out_dir/fn).write_bytes(png); images.append(fn)
                    tL, ptL, wmL = extract_title_from_side(
                        page, L_rect,
                        args.title_percentile,
                        args.title_base_size, args.title_size_tol,
                        args.title_top_percent,
                        args.wm_margin, args.line_join_gap, args.block_down_tol
                    )
                    if tL: title_side, title_text, title_max_pt, wm_max_pt = "L", tL, ptL, wmL
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
                    tR, ptR, wmR = extract_title_from_side(
                        page, R_rect,
                        args.title_percentile,
                        args.title_base_size, args.title_size_tol,
                        args.title_top_percent,
                        args.wm_margin, args.line_join_gap, args.block_down_tol
                    )
                    if tR and (ptR>title_max_pt or (ptR==title_max_pt and len(tR)>len(title_text))):
                        title_side, title_text, title_max_pt, wm_max_pt = "R", tR, ptR, wmR
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
                if exL and exR:
                    # 논리상 ALL과 동일, 안전 처리
                    _wm_pt = 0.0; _stats = {"reg_p90":0.0,"bold_p90":0.0}; _thr = 0.0
            else:
                exclude_reason="all"
                # 이미지/제목 모두 스킵
                _wm_pt = 0.0; _stats = {"reg_p90":0.0,"bold_p90":0.0}; _thr = 0.0

        decisions.append({
            "pdf_page": p+1,
            "excluded": bool(ex is not None),
            "exclude_reason": excludes.get(p+1,""),
            "images": ";".join(images),
            "title_side": title_side,
            "title_text": title_text,
            "title_max_pt": round(title_max_pt,2),
            "wm_max_pt": round(wm_max_pt,2),
            "title_thresh_pt": round(_thr,2),
            "body_reg_p90": round(_stats.get("reg_p90",0.0),2),
            "body_bold_p90": round(_stats.get("bold_p90",0.0),2),
            # 디버깅용 (신규)
            "title_base_size": None if args.title_base_size is None else round(float(args.title_base_size), 2),
            "title_size_tol": round(float(args.title_size_tol), 2),
        })

    pd.DataFrame(decisions).to_csv(out_dir/"page_decisions.csv", index=False, encoding="utf-8-sig")
    print(f"[done] saved to: {out_dir}")

if __name__=="__main__":
    main()