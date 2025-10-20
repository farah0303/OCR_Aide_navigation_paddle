"""Unified OCR for PDFs and Images with PaddleOCR"""
import argparse
import glob
import os
import sys
import traceback
from typing import List, Optional

try:
    from extract_text_pdf import extract_text_from_pdf, parse_page_list
except:
    extract_text_from_pdf = parse_page_list = None

try:
    from extract_text_image import extract_text_from_image, is_image_file, SUPPORTED_EXTENSIONS
except:
    extract_text_from_image = is_image_file = SUPPORTED_EXTENSIONS = None


def detect_file_type(fp):
    if not os.path.exists(fp):
        return 'unknown'
    ext = os.path.splitext(fp)[1].lower()
    return 'pdf' if ext == '.pdf' else ('image' if is_image_file and is_image_file(fp) else 'unknown')


def categorize_files(fps):
    pdf, img, unk = [], [], []
    for f in fps:
        t = detect_file_type(f)
        (pdf if t == 'pdf' else img if t == 'image' else unk).append(f)
    return pdf, img, unk


def find_all_files(d="."):
    files = glob.glob(os.path.join(d, "*.pdf")) + \
        glob.glob(os.path.join(d, "*.PDF"))
    if SUPPORTED_EXTENSIONS:
        for e in SUPPORTED_EXTENSIONS:
            files.extend(glob.glob(os.path.join(
                d, f"*{e}")) + glob.glob(os.path.join(d, f"*{e.upper()}")))
    return sorted(set(files))


def choose_files(fps, multi=False):
    if not fps:
        print("No files found.")
        sys.exit(2)
    for i, p in enumerate(fps, 1):
        print(f"  [{i}] {os.path.basename(p)} ({detect_file_type(p).upper()})")
    if not multi:
        while True:
            c = input(f"Select (1-{len(fps)}) or 'q': ")
            if c.lower() == 'q':
                sys.exit(0)
            try:
                if 1 <= int(c) <= len(fps):
                    return fps[int(c)-1]
            except:
                pass
    else:
        print("Enter: numbers (1,3,5), range (1-5), 'all', or 'q'")
        while True:
            c = input("Selection: ").strip()
            if c.lower() == 'q':
                sys.exit(0)
            if c.lower() == 'all':
                return fps
            try:
                sel = []
                for p in c.split(','):
                    p = p.strip()
                    if '-' in p:
                        s, e = map(int, p.split('-', 1))
                        sel.extend(fps[i-1]
                                   for i in range(s, e+1) if 1 <= i <= len(fps))
                    else:
                        i = int(p)
                        if 1 <= i <= len(fps):
                            sel.append(fps[i-1])
                if sel:
                    return list(set(sel))
            except:
                pass


def process_file(fp, lang='en', use_angle=True, clean=False, pages=None, zoom=2.0):
    ft = detect_file_type(fp)
    if ft == 'pdf':
        if not extract_text_from_pdf:
            raise RuntimeError("PDF module unavailable")
        print(f"Processing PDF: {os.path.basename(fp)}")
        return extract_text_from_pdf(fp, pages=pages, zoom=zoom)
    elif ft == 'image':
        if not extract_text_from_image:
            raise RuntimeError("Image module unavailable")
        print(f"Processing Image: {os.path.basename(fp)}")
        return extract_text_from_image(fp, lang=lang, use_angle_cls=use_angle, clean_text=clean)
    raise ValueError(f"Unsupported: {fp}")


def process_multi(fps, lang='en', use_angle=True, clean=False, zoom=2.0):
    pdfs, imgs, unk = categorize_files(fps)
    if unk:
        print(f"\nSkipping {len(unk)} unsupported file(s)")
    texts = []
    for pdf in pdfs:
        try:
            texts.append(extract_text_from_pdf(pdf, zoom=zoom))
        except Exception as e:
            print(f"ERROR {pdf}: {e}")
    for img in imgs:
        try:
            texts.append(extract_text_from_image(img, lang=lang,
                         use_angle_cls=use_angle, clean_text=clean))
        except Exception as e:
            print(f"ERROR {img}: {e}")
    return "\n\n".join(texts).strip()


def main():
    p = argparse.ArgumentParser(description="Unified OCR for PDFs and Images")
    p.add_argument("-f", "--file", nargs='+', help="File(s) to process")
    p.add_argument("-o", "--output", help="Output file")
    p.add_argument("-l", "--lang", default='en', help="OCR language")
    p.add_argument("-n", "--no-angle", action='store_true',
                   help="Disable angle classification")
    p.add_argument("-c", "--clean", action='store_true',
                   help="Apply text cleaning")
    p.add_argument("-b", "--batch", action='store_true',
                   help="Process multiple files")
    p.add_argument("-p", "--pages", help="PDF pages (e.g. '1,3-5')")
    p.add_argument("-z", "--zoom", type=float, default=2.0, help="PDF zoom")
    args = p.parse_args()

    fps = args.file if args.file else ([choose_files(find_all_files(
    ), args.batch)] if not args.batch else choose_files(find_all_files(), True))
    if isinstance(fps, str):
        fps = [fps]
    fps = [f for f in fps if os.path.exists(f)]
    if not fps:
        print("No valid files.")
        sys.exit(2)

    try:
        pages = parse_page_list(
            args.pages) if args.pages and parse_page_list else None
        text = process_file(fps[0], args.lang, not args.no_angle, args.clean, pages, args.zoom) if len(
            fps) == 1 else process_multi(fps, args.lang, not args.no_angle, args.clean, args.zoom)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\n✓ Saved to: {args.output}")
        else:
            print(f"\n{'='*60}\nEXTRACTED TEXT:\n{'='*60}\n{text}\n{'='*60}")
            default = os.path.splitext(
                fps[0])[0] + '.txt' if len(fps) == 1 else 'extracted_text.txt'
            if input(f"\nSave? (y/n): ").strip().lower() in ('y', 'yes', ''):
                out = input(f"Filename [{default}]: ").strip() or default
                with open(out, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"✓ Saved to: {out}")
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
