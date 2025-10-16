"""
extract_text_image.py

Extract text from images using PaddleOCR.
- Automatically detects common image formats (JPG, PNG, BMP, TIFF, WEBP, etc.)
- Applies OCR with optional text cleaning
- Supports batch processing of multiple images

Usage:
    python extract_text_image.py input.jpg -o output.txt

Requires: paddleocr, pillow, numpy
"""

import os
import sys
from typing import List, Optional

import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

# Supported image extensions (case-insensitive)
SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
    '.webp', '.gif', '.ppm', '.pgm', '.pbm', '.pnm'
}


def is_image_file(filepath: str) -> bool:
    """Check if a file is a supported image format."""
    _, ext = os.path.splitext(filepath)
    return ext.lower() in SUPPORTED_EXTENSIONS


def load_image(image_path: str) -> Image.Image:
    """
    Load an image file without prior knowledge of its format.
    PIL automatically detects the format.
    """
    try:
        img = Image.open(image_path)
        # Convert to RGB if necessary (e.g., for RGBA or grayscale)
        if img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to load image {image_path}: {e}")


def ocr_image(paddle_ocr, pil_image: Image.Image) -> str:
    """Run PaddleOCR on a PIL image and return concatenated text."""
    img_np = np.array(pil_image)
    result = paddle_ocr.ocr(img_np)

    lines: List[str] = []
    if not result:
        return ""  # Nothing detected in the image

    for res in result:
        if not res:
            continue
        for line in res:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                if isinstance(line[1], (list, tuple)) and len(line[1]) >= 1:
                    lines.append(str(line[1][0]))
    return "\n".join(lines)


def extract_text_from_image(
    image_path: str,
    lang: str = 'fr',
    use_angle_cls: bool = True,
    clean_text: bool = False
) -> str:
    """
    Extract text from a single image file.

    Args:
        image_path: Path to the image file
        lang: Language for OCR ('en', 'fr', 'ch', etc.)
        use_angle_cls: Whether to use angle classification for rotated text
        clean_text: Whether to apply automatic text cleaning (requires additional packages)

    Returns:
        Extracted text as a string
    """
    if PaddleOCR is None:
        raise RuntimeError(
            "PaddleOCR package not available. Install with `pip install paddleocr`.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if not is_image_file(image_path):
        raise ValueError(
            f"File does not appear to be a supported image format: {image_path}")

    print(f"Processing {os.path.basename(image_path)} ...")

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    # Load and process image
    pil_img = load_image(image_path)
    text = ocr_image(ocr, pil_img)

    # Optional: apply text cleaning
    if clean_text and text:
        try:
            text = auto_clean_text(text)
        except Exception as e:
            print(f"Warning: Text cleaning failed: {e}")

    return text


def extract_text_from_images(
    image_paths: List[str],
    lang: str = 'fr',
    use_angle_cls: bool = True,
    clean_text: bool = False
) -> str:
    """
    Extract text from multiple image files.

    Args:
        image_paths: List of paths to image files
        lang: Language for OCR
        use_angle_cls: Whether to use angle classification
        clean_text: Whether to apply automatic text cleaning

    Returns:
        Combined extracted text from all images
    """
    if PaddleOCR is None:
        raise RuntimeError(
            "PaddleOCR package not available. Install with `pip install paddleocr`.")

    # Initialize PaddleOCR once for all images
    ocr = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang)

    all_text: List[str] = []

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: File not found, skipping: {img_path}")
            continue

        if not is_image_file(img_path):
            print(
                f"Warning: Not a supported image format, skipping: {img_path}")
            continue

        print(f"Processing {os.path.basename(img_path)} ...")

        try:
            pil_img = load_image(img_path)
            text = ocr_image(ocr, pil_img)

            if clean_text and text:
                try:
                    text = auto_clean_text(text)
                except Exception as e:
                    print(f"Warning: Text cleaning failed for {img_path}: {e}")

            all_text.append(f"-- {os.path.basename(img_path)} --\n{text}")
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            all_text.append(f"-- {os.path.basename(img_path)} --\nERROR: {e}")

    return "\n\n".join(all_text).strip()


# Optional text cleaning function (same as in extract_text_paddle.py)
def auto_clean_text(text: str) -> str:
    """
    Automatically correct common OCR errors (0/o, I/l, accents, etc.)
    and apply general spell checking in French.

    Note: Requires spellchecker and unidecode packages.
    """
    import re
    try:
        from spellchecker import SpellChecker
        from unidecode import unidecode
    except ImportError:
        print("Warning: spellchecker or unidecode not installed. Skipping text cleaning.")
        return text

    # Step 1: Basic cleaning (remove strange characters)
    text = unidecode(text)

    # Step 2: Correct character confusions (I/l, 0/o)
    text = re.sub(r"\bI'", "l'", text)
    # Replace 0 in the middle of a word with o
    text = re.sub(r"(?<=\w)0(?=\w)", "o", text)
    text = re.sub(r"(?<=\w)1(?=\w)", "l", text)

    # Step 3: Global spell checking
    spell = SpellChecker(language='fr')
    tokens = re.split(r'(\W+)', text)  # Preserve punctuation and spaces
    corrected = []

    for token in tokens:
        # Don't correct email addresses, numbers, or punctuation
        if (
            not token.strip()
            or re.match(r"[\d@/\\]", token)
            or re.match(r"\W+$", token)
        ):
            corrected.append(token)
            continue

        # Suggest correction if word doesn't exist
        word = token.lower()
        if word not in spell:
            suggestion = spell.correction(word)
            if suggestion:
                token = suggestion.capitalize(
                ) if token[0].isupper() else suggestion
        corrected.append(token)

    # Step 4: Reconstruct text
    clean_text = "".join(corrected)

    # Final cleaning
    clean_text = re.sub(r"\s{2,}", " ", clean_text)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    return clean_text


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Extract text from images using PaddleOCR"
    )
    p.add_argument("images", nargs='+', help="Input image file(s)")
    p.add_argument("-o", "--output",
                   help="Output text file (default: <image>.txt)")
    p.add_argument("--lang", default='fr',
                   help="OCR language: 'en', 'fr', 'ch', etc. (default: en)")
    p.add_argument("--no-angle", action='store_true',
                   help="Disable angle classification for rotated text")
    p.add_argument("--clean", action='store_true',
                   help="Apply automatic text cleaning (French spell check)")

    args = p.parse_args()

    # Check if files exist
    valid_images = []
    for img in args.images:
        if os.path.exists(img):
            valid_images.append(img)
        else:
            print(f"Warning: File not found: {img}")

    if not valid_images:
        print("ERROR: No valid image files provided.")
        sys.exit(2)

    # Extract text
    try:
        if len(valid_images) == 1:
            text = extract_text_from_image(
                valid_images[0],
                lang=args.lang,
                use_angle_cls=not args.no_angle,
                clean_text=args.clean
            )
            out_path = args.output or os.path.splitext(valid_images[0])[
                0] + '.txt'
        else:
            text = extract_text_from_images(
                valid_images,
                lang=args.lang,
                use_angle_cls=not args.no_angle,
                clean_text=args.clean
            )
            out_path = args.output or 'extracted_text.txt'
    except Exception as e:
        print(f"ERROR during extraction: {e}")
        sys.exit(1)

    # Save or print results
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
