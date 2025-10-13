

import argparse
import glob
import os
import sys
from typing import List

try:
    from extract_text_paddle import extract_text_from_pdf, parse_page_list
except Exception as e:
    print("ERROR: couldn't import extract_text_paddle module:", e)
    print("Make sure you're running from the project root where extract_text_paddle.py exists.")
    raise


def choose_pdf_interactively(pdf_paths: List[str]) -> str:
    if not pdf_paths:
        print("No PDF files found in the current directory.")
        sys.exit(2)
    print("PDF files found:")
    for i, p in enumerate(pdf_paths, start=1):
        print(f"  [{i}] {os.path.basename(p)}")
    while True:
        choice = input(f"Select a file by number (1-{len(pdf_paths)}), or 'q' to quit: ")
        if choice.lower() == 'q':
            sys.exit(0)
        try:
            idx = int(choice)
            if 1 <= idx <= len(pdf_paths):
                return pdf_paths[idx - 1]
        except Exception:
            pass
        print("Invalid choice, try again.")


def main():
    parser = argparse.ArgumentParser(description="OCR PDF with PaddleOCR")
    parser.add_argument("--file", help="PDF file to process")
    parser.add_argument("--output", help="Output text file")
    parser.add_argument("--pages", help="Pages to extract (e.g. '1,3-5')")
    parser.add_argument("--zoom", type=float, default=2.0, help="Zoom factor for rendering")
    args = parser.parse_args()

    # Choisir le PDF
    if args.file:
        pdf_path = args.file
    else:
        pdf_paths = glob.glob("*.pdf")
        pdf_path = choose_pdf_interactively(pdf_paths)

    # Convertir les pages si spécifiées
    pages = parse_page_list(args.pages) if args.pages else None

    # Extraire le texte
    text = extract_text_from_pdf(pdf_path, pages=pages, zoom=args.zoom)

    # Sauvegarder ou afficher
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
