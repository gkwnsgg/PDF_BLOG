#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
measure_title_base.py — PDF에서 제목 기준 폰트 크기(title-base-size) 측정기

사용 개요
- 입력 PDF에서 사용자가 지정한 면(예: 4pl, 5pl, 9pr …)의 텍스트 스팬을 조사
- 상단 워터마크 스트립(top_percent %)은 제외
- 워터마크 텍스트(예: 'BOM | CAR/NEWS/…', 'BOM 12' 등)는 제외
- 각 면에서 '최대 폰트 크기'를 뽑고, 그 최대값들의 '평균'을 title-base-size로 출력

설치
  pip install pymupdf pandas

예)
  python measure_title_base.py "BOM_25_여름_내지.pdf" 4pl 5pl 6pl 8pl 9pl 24pl 28pl 32pl 36pl 38pl 39pl \
    --top-percent 12.0 -o base_probe.csv

결과
- 콘솔: TITLE_BASE_SIZE_PT=xx.xx
- CSV: 입력 토큰별(max_pt, 샘플 수 등) 상세 기록
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import fitz  # PyMuPDF
import pandas as pd

# 상단 워터마크 패턴 (BOM | CAR/NEWS/… 혹은 BOM 숫자)
WM_TOP_PAT = re.compile(
    r"""^BOM\s*([|│\|l]\s*(NEW\s*CAR|NEWS|CAR|MOTORRAD|TRAVEL|GOLF|CULTURE))?$""",
    re.IGNORECASE
)
WM_FOOT_PAT = re.compile(r"^BOM\s*\d{1,3}\s*$", re.IGNORECASE)

def parse_side_token(tok: str) -> Tuple[int, str]:
    """
    '4pl' -> (4, 'L'), '9pr' -> (9, 'R'), '30p'/'30' -> (30, 'BOTH')
    잘못된 토큰은 (0, '') 반환
    """
    tok = tok.strip().lower()
    m = re.match(r"^(\d+)\s*(p|pl|pr)?$", tok)
    if not m:
        return (0, "")
    page = int(m.group(1))
    suf = (m.group(2) or "").lower()
    if suf == "pl": return (page, "L")
    if suf == "pr": return (page, "R")
    return (page, "BOTH")

def page_halves_rect(page: fitz.Page) -> Tuple[fitz.Rect, fitz.Rect]:
    R = page.rect
    xm = (R.x0 + R.x1) / 2.0
    return fitz.Rect(R.x0, R.y0, xm, R.y1), fitz.Rect(xm, R.y0, R.x1, R.y1)

def gather_spans_in_rect(page: fitz.Page, rect: fitz.Rect) -> List[Dict[str, Any]]:
    d = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH | fitz.TEXT_PRESERVE_LIGATURES)
    out = []
    for b in d.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                txt = (s.get("text") or "").strip()
                if not txt:
                    continue
                bb = s.get("bbox") or l.get("bbox") or b.get("bbox")
                cx = (bb[0] + bb[2]) / 2.0
                cy = (bb[1] + bb[3]) / 2.0
                if rect.contains(fitz.Point(cx, cy)):
                    out.append({
                        "text": txt,
                        "size": float(s.get("size", 0.0) or 0.0),
                        "bbox": tuple(bb),
                        "flags": int(s.get("flags", 0) or 0),
                        "font": s.get("font", "")
                    })
    return out

def measure_title_base(pdf_path: Path,
                       probes: List[str],
                       top_percent: float = 12.0) -> Dict[str, Any]:
    """
    probes: ["4pl","5pl","9pr", ...]
    top_percent: 상단 워터마크 스트립 비율(%). 이 영역의 텍스트는 제외.
    """
    doc = fitz.open(pdf_path)
    rows = []

    for tok in probes:
        page_no, side = parse_side_token(tok)
        if page_no <= 0 or page_no > doc.page_count or not side:
            rows.append({"token": tok, "page": page_no, "side": side or "-", "max_pt": 0.0, "spans": 0, "used": 0})
            continue

        page = doc.load_page(page_no - 1)
        L_rect, R_rect = page_halves_rect(page)
        rects = [L_rect, R_rect] if side == "BOTH" else ([L_rect] if side == "L" else [R_rect])

        page_h = float(page.rect.height)
        y_wm = page_h * (top_percent / 100.0)

        max_pt = 0.0
        total_spans = 0
        used_spans = 0

        for R in rects:
            spans = gather_spans_in_rect(page, R)
            total_spans += len(spans)
            for s in spans:
                y_top = s["bbox"][1]
                txt = s["text"]
                # 상단 워터마크 스트립 제외
                if y_top <= y_wm:
                    continue
                # 워터마크 텍스트 제외
                if WM_TOP_PAT.match(txt) or WM_FOOT_PAT.match(txt) or ("BOM" in txt.upper()):
                    continue
                used_spans += 1
                if s["size"] > max_pt:
                    max_pt = s["size"]

        rows.append({"token": tok, "page": page_no, "side": side, "max_pt": round(max_pt, 2),
                     "spans": total_spans, "used": used_spans})

    # 유효한 max_pt 평균
    vals = [r["max_pt"] for r in rows if r["max_pt"] > 0]
    avg = round(sum(vals) / len(vals), 2) if vals else 0.0

    return {"rows": rows, "avg_pt": avg}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="입력 PDF 경로")
    ap.add_argument("probes", nargs="+", help="제목이 있는 면 토큰들 (예: 4pl 5pl 9pl ...)")
    ap.add_argument("--top-percent", type=float, default=12.0,
                    help="상단 워터마크 스트립 비율(%%). 이 영역은 제외하고 측정")
    ap.add_argument("-o", "--out", default="", help="결과 CSV 저장 경로(옵션)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    result = measure_title_base(pdf_path, args.probes, args.top_percent)

    # 콘솔 출력
    print("TITLE_BASE_SIZE_PT={:.2f}".format(result["avg_pt"]))

    # CSV 옵션 저장
    if args.out:
        df = pd.DataFrame(result["rows"])
        df.to_csv(Path(args.out).expanduser().resolve(), index=False, encoding="utf-8-sig")
        print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()