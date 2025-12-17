"""Extract text from images using PaddleOCR"""
import argparse
import os
import sys
import re
from typing import List, Optional
import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
    from spellchecker import SpellChecker
    from unidecode import unidecode
except:
    PaddleOCR = SpellChecker = unidecode = None

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff',
                        '.tif', '.webp', '.gif', '.ppm', '.pgm', '.pbm', '.pnm'}


def is_image_file(filepath):
    return os.path.splitext(filepath)[1].lower() in SUPPORTED_EXTENSIONS


def load_image(image_path):
    try:
        img = Image.open(image_path)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to load {image_path}: {e}")


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


def auto_clean_text(text):
    if not (SpellChecker and unidecode):
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


def extract_text_from_image(image_path, lang='fr', use_angle_cls=True, clean_text=False, use_gpu=False):
    if not PaddleOCR:
        raise RuntimeError("PaddleOCR unavailable")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not is_image_file(image_path):
        raise ValueError(f"Unsupported format: {image_path}")
    print(f"Processing {os.path.basename(image_path)} ...")
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu)
    pil_img = load_image(image_path)
    text = ocr_image(ocr, pil_img)
    if clean_text and text:
        try:
            text = auto_clean_text(text)
        except Exception as e:
            print(f"Warning: cleaning failed: {e}")
    return text


def extract_text_from_images(image_paths, lang='fr', use_angle_cls=True, clean_text=False, use_gpu=False):
    if not PaddleOCR:
        raise RuntimeError("PaddleOCR unavailable")
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu)
    all_text = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: not found, skipping: {img_path}")
            continue
        if not is_image_file(img_path):
            print(f"Warning: unsupported format, skipping: {img_path}")
            continue
        print(f"Processing {os.path.basename(img_path)} ...")
        try:
            pil_img = load_image(img_path)
            text = ocr_image(ocr, pil_img)
            if clean_text and text:
                try:
                    text = auto_clean_text(text)
                except Exception as e:
                    print(f"Warning: cleaning failed for {img_path}: {e}")
            all_text.append(f"-- {os.path.basename(img_path)} --\n{text}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            all_text.append(f"-- {os.path.basename(img_path)} --\nERROR: {e}")
    return "\n\n".join(all_text).strip()


def main():
    p = argparse.ArgumentParser(
        description="Extract text from images using PaddleOCR")
    p.add_argument("images", nargs='+', help="Input image file(s)")
    p.add_argument("-o", "--output", help="Output text file")
    p.add_argument("--lang", default='fr', help="OCR language")
    p.add_argument("--no-angle", action='store_true',
                   help="Disable angle classification")
    p.add_argument("--clean", action='store_true', help="Apply text cleaning")
    args = p.parse_args()
    valid_images = [img for img in args.images if os.path.exists(img)]
    if not valid_images:
        print("ERROR: No valid image files.")
        sys.exit(2)
    try:
        if len(valid_images) == 1:
            text = extract_text_from_image(
                valid_images[0], lang=args.lang, use_angle_cls=not args.no_angle, clean_text=args.clean)
            out_path = args.output or os.path.splitext(valid_images[0])[
                0] + '.txt'
        else:
            text = extract_text_from_images(
                valid_images, lang=args.lang, use_angle_cls=not args.no_angle, clean_text=args.clean)
            out_path = args.output or 'extracted_text.txt'
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    if args.output or len(valid_images) > 1:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"\nExtracted text written to: {out_path}")
    else:
        print("\n" + "="*50)
        print(text)
        print("="*50)


if __name__ == '__main__':
    main()