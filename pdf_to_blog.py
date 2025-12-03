#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pdf_to_blog.py — 워터마크 무시, 사용자 지정 제외 페이지 기반 캡처

기능:
- 시작/종료 앵커로 본문 범위 제한
- --exclude 로 직접 페이지 제외 지정 가능:
    * 37p  → 37쪽 전체 제외
    * 9pr  → 9쪽 오른쪽 제외
    * 40pl → 40쪽 왼쪽 제외
- 제외 지정이 없으면 모든 페이지/반쪽을 캡처
"""

import os, re, argparse, json, datetime as dt
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
from PIL import Image


# ---------------------- 유틸 ----------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_str():
    return dt.datetime.now().isoformat(timespec="seconds")

def pix_to_pil(pix: "fitz.Pixmap") -> Image.Image:
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


# ---------------------- 렌더링 ----------------------

def render_full_png(page: "fitz.Page", scale: float = 2.5) -> Image.Image:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix_to_pil(pix)


# ---------------------- 메인 파이프라인 ----------------------

def process_pdf(pdf_path: Path, out_dir: Path,
                start_anchor: str = "",
                stop_anchor: str = "",
                render_scale: float = 2.5,
                exclude_tokens: List[str] = None):

    ensure_dir(out_dir)
    doc = fitz.open(pdf_path)

    # 앵커 기반 시작/종료 페이지 결정
    start_page = 0
    end_page   = doc.page_count
    if start_anchor:
        for i in range(doc.page_count):
            txt = doc.load_page(i).get_text("text")
            if start_anchor.replace(" ", "") in txt.replace(" ", ""):
                start_page = max(0, i+1)
                break
    if stop_anchor:
        for j in range(start_page+1, doc.page_count):
            txt = doc.load_page(j).get_text("text")
            if stop_anchor.replace(" ", "") in txt.replace(" ", ""):
                end_page = j
                break
    if start_page >= end_page:
        start_page, end_page = 0, doc.page_count

    exclude_set = set(exclude_tokens or [])

    decisions = []
    pages_bundle = []

    for p in range(start_page, end_page):
        page_no = p + 1
        token_full = f"{page_no}p"
        token_left = f"{page_no}pl"
        token_right= f"{page_no}pr"

        # 전체 제외
        if token_full in exclude_set:
            decisions.append({
                "pdf_page": page_no,
                "excluded": True,
                "excluded_reason": "full",
                "images": ""
            })
            continue

        # 렌더링
        pil = render_full_png(doc.load_page(p), render_scale)
        W, H = pil.size
        xm   = W // 2

        allow_left  = token_left not in exclude_set
        allow_right = token_right not in exclude_set

        images = []
        if allow_left and allow_right:
            tag = f"page-{page_no:03d}"
            out_png = out_dir / f"{tag}.png"
            pil.save(out_png, "PNG")
            images.append(out_png.name)
            allowed_rects = "FULL"
        else:
            if allow_left:
                tag = f"page-{page_no:03d}-left"
                out_png = out_dir / f"{tag}.png"
                pil.crop((0, 0, xm, H)).save(out_png, "PNG")
                images.append(out_png.name)
            if allow_right:
                tag = f"page-{page_no:03d}-right"
                out_png = out_dir / f"{tag}.png"
                pil.crop((xm, 0, W, H)).save(out_png, "PNG")
                images.append(out_png.name)
            allowed_rects = ";".join(images) if images else ""

        decisions.append({
            "pdf_page": page_no,
            "excluded": False,
            "excluded_reason": "",
            "images": ";".join(images),
            "allowed_rects": allowed_rects
        })

        if images:
            pages_bundle.append({
                "page_no": page_no,
                "images": images
            })

    # page_decisions.csv 저장
    import pandas as pd
    pd.DataFrame(decisions).to_csv(out_dir/"page_decisions.csv", index=False, encoding="utf-8-sig")

    # manifest.json 저장
    manifest = [{
        "index": i+1,
        "title": f"Page {p['page_no']}",
        "slug": f"page-{p['page_no']}",
        "pages": [p["page_no"]],
        "images": p["images"]
    } for i, p in enumerate(pages_bundle)]
    (out_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"pages": pages_bundle, "decisions": decisions, "manifest": manifest}


# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="입력 PDF 경로")
    ap.add_argument("-o", "--out", default="out_bom", help="출력 폴더")
    ap.add_argument("--start-anchor", default="", help="시작 앵커 텍스트")
    ap.add_argument("--stop-anchor", default="", help="종료 앵커 텍스트")
    ap.add_argument("--render-scale", type=float, default=2.5, help="렌더 배율")
    ap.add_argument("--exclude", nargs="*", default=[], help="제외할 페이지 (예: 7pr 9pr 37p 40pl)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir  = Path(args.out).expanduser().resolve()

    result = process_pdf(
        pdf_path=pdf_path,
        out_dir=out_dir,
        start_anchor=args.start_anchor,
        stop_anchor=args.stop_anchor,
        render_scale=args.render_scale,
        exclude_tokens=args.exclude
    )

    print(f"[done] pages: {len(result['pages'])}")
    print(f"[done] decisions csv: {out_dir/'page_decisions.csv'}")
    print(f"[done] manifest json: {out_dir/'manifest.json'}")


if __name__ == "__main__":
    main()
#이미지 추출 버전