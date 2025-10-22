"""Extract text from PDFs using PyMuPDF and PaddleOCR"""
import argparse
import io
import os
import sys
import re
from typing import List, Optional
import fitz
import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
    from unidecode import unidecode
    from spellchecker import SpellChecker
except:
    PaddleOCR = unidecode = SpellChecker = None


def extract_embedded_text(pdf_path):
    doc = fitz.open(pdf_path)
    parts = [page.get_text("text") for page in doc if page.get_text("text")]
    doc.close()
    return "\n".join(parts).strip()


def render_page_to_pil(page, zoom=2.0):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes(output="png"))).convert("RGB")


def auto_clean_text(text):
    if not (unidecode and SpellChecker):
        return text
    text = unidecode(text)
    text = re.sub(r"\bI'", "l'", text)
    text = re.sub(r"(?<=\w)0(?=\w)", "o", text)
    text = re.sub(r"(?<=\w)1(?=\w)", "l", text)
    spell = SpellChecker(language='fr')
    tokens = re.split(r'(\W+)', text)
    corrected = []
    for token in tokens:
        if not token.strip() or re.match(r"[\d@/\\]", token) or re.match(r"\W+$", token):
            corrected.append(token)
            continue
        word = token.lower()
        if word not in spell:
            suggestion = spell.correction(word)
            if suggestion:
                token = suggestion.capitalize(
                ) if token[0].isupper() else suggestion
        corrected.append(token)
    clean_text = "".join(corrected)
    clean_text = re.sub(r"\s{2,}", " ", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    return clean_text


def ocr_image(paddle_ocr, pil_image):
    img_np = np.array(pil_image)
    result = paddle_ocr.ocr(img_np)
    lines = []
    if not result:
        return ""
    for res in result:
        if not res:
            continue
        for line in res:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                    lines.append(str(line[1][0]))
    return "\n".join(lines)


def extract_text_from_pdf(pdf_path, pages=None, zoom=2.0):
    embedded = extract_embedded_text(pdf_path)
    if len(embedded) > 100:
        return embedded
    if not PaddleOCR:
        raise RuntimeError("PaddleOCR unavailable")
    ocr = PaddleOCR(use_angle_cls=True, lang='fr')
    doc = fitz.open(pdf_path)
    all_text = []
    page_count = doc.page_count
    pages_to_process = range(page_count) if pages is None else [
        p for p in pages if 0 <= p < page_count]
    for pno in pages_to_process:
        print(f"Processing page {pno + 1} ...")
        page = doc.load_page(pno)
        pil_img = render_page_to_pil(page, zoom=zoom)
        text = ocr_image(ocr, pil_img)
        all_text.append(f"-- PAGE {pno + 1} --\n{text}")
    doc.close()
    return "\n\n".join(all_text).strip()


def parse_page_list(s):
    if not s:
        return []
    out = set()
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-', 1))
            for i in range(a, b + 1):
                out.add(i - 1)
        else:
            out.add(int(part) - 1)
    return sorted(out)


def main():
    p = argparse.ArgumentParser(
        description="Extract text from PDFs using PaddleOCR")
    p.add_argument("pdf", help="Input PDF file")
    p.add_argument("-o", "--output", help="Output text file")
    p.add_argument("--pages", help="Pages to OCR (e.g. '1,3-5')")
    p.add_argument("--zoom", type=float, default=2.0, help="Render zoom")
    args = p.parse_args()
    if not os.path.exists(args.pdf):
        print(f"ERROR: file not found: {args.pdf}")
        sys.exit(2)
    out_path = args.output or os.path.splitext(args.pdf)[0] + '.txt'
    pages = parse_page_list(args.pages) if args.pages else None
    try:
        text = extract_text_from_pdf(args.pdf, pages=pages, zoom=args.zoom)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Wrote text to: {out_path}")


if __name__ == '__main__':
    main()

