import argparse
import glob
import os
import sys
from typing import List

try:
    from extract_text_image import extract_text_from_image, extract_text_from_images, is_image_file
except Exception as e:
    print("ERROR: couldn't import extract_text_image module:", e)
    print("Make sure you're running from the img_module directory where extract_text_image.py exists.")
    raise


def find_image_files(directory: str = ".") -> List[str]:
    """Find all supported image files in the given directory."""
    image_files = []

    # Common image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif',
                  '*.webp', '*.gif', '*.ppm', '*.pgm', '*.pbm', '*.pnm']

    for ext in extensions:
        pattern = os.path.join(directory, ext)
        image_files.extend(glob.glob(pattern))
        # Also search for uppercase extensions
        pattern_upper = os.path.join(directory, ext.upper())
        image_files.extend(glob.glob(pattern_upper))

    # Remove duplicates and sort
    return sorted(list(set(image_files)))


def choose_image_interactively(image_paths: List[str]) -> str:
    """Let the user select an image file interactively."""
    if not image_paths:
        print("No image files found in the current directory.")
        print("Supported formats: JPG, PNG, BMP, TIFF, WEBP, GIF, PPM, PGM, PBM, PNM")
        sys.exit(2)

    print("Image files found:")
    for i, p in enumerate(image_paths, start=1):
        print(f"  [{i}] {os.path.basename(p)}")

    while True:
        choice = input(
            f"Select a file by number (1-{len(image_paths)}), or 'q' to quit: ")
        if choice.lower() == 'q':
            sys.exit(0)
        try:
            idx = int(choice)
            if 1 <= idx <= len(image_paths):
                return image_paths[idx - 1]
        except Exception:
            pass
        print("Invalid choice, try again.")


def choose_multiple_images_interactively(image_paths: List[str]) -> List[str]:
    """Let the user select multiple image files interactively."""
    if not image_paths:
        print("No image files found in the current directory.")
        sys.exit(2)

    print("Image files found:")
    for i, p in enumerate(image_paths, start=1):
        print(f"  [{i}] {os.path.basename(p)}")

    print("\nSelect images to process:")
    print("  - Enter numbers separated by commas (e.g., '1,3,5')")
    print("  - Enter a range with hyphen (e.g., '1-5')")
    print("  - Enter 'all' to process all images")
    print("  - Enter 'q' to quit")

    while True:
        choice = input("Your selection: ").strip()

        if choice.lower() == 'q':
            sys.exit(0)

        if choice.lower() == 'all':
            return image_paths

        try:
            selected = []
            parts = choice.split(',')

            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle range
                    start, end = part.split('-', 1)
                    start_idx = int(start.strip())
                    end_idx = int(end.strip())
                    for i in range(start_idx, end_idx + 1):
                        if 1 <= i <= len(image_paths):
                            selected.append(image_paths[i - 1])
                else:
                    # Handle single number
                    idx = int(part)
                    if 1 <= idx <= len(image_paths):
                        selected.append(image_paths[idx - 1])

            if selected:
                return list(set(selected))  # Remove duplicates
        except Exception:
            pass

        print("Invalid selection, try again.")


def main():
    parser = argparse.ArgumentParser(
        description="OCR Images with PaddleOCR",
        epilog="Examples:\n"
               "  python main.py --file image.jpg\n"
               "  python main.py --file img1.png img2.jpg --output results.txt\n"
               "  python main.py --batch --lang fr\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--file", nargs='+', help="Image file(s) to process")
    parser.add_argument("--output", help="Output text file")
    parser.add_argument("--lang", default='en',
                        help="OCR language: 'en', 'fr', 'ch', etc. (default: en)")
    parser.add_argument("--no-angle", action='store_true',
                        help="Disable angle classification for rotated text")
    parser.add_argument("--clean", action='store_true',
                        help="Apply automatic text cleaning (French spell check)")
    parser.add_argument("--batch", action='store_true',
                        help="Process multiple images interactively")

    args = parser.parse_args()

    # Choose image(s)
    if args.file:
        image_paths = args.file
        # Validate files
        valid_images = [f for f in image_paths if os.path.exists(
            f) and is_image_file(f)]
        if not valid_images:
            print("ERROR: No valid image files provided.")
            sys.exit(2)
        image_paths = valid_images
    else:
        # Interactive mode
        available_images = find_image_files()

        if args.batch:
            image_paths = choose_multiple_images_interactively(
                available_images)
        else:
            image_paths = [choose_image_interactively(available_images)]

    # Extract text
    try:
        print(f"\nStarting OCR with language: {args.lang}")
        print(
            f"Angle classification: {'disabled' if args.no_angle else 'enabled'}")
        print(f"Text cleaning: {'enabled' if args.clean else 'disabled'}\n")

        if len(image_paths) == 1:
            text = extract_text_from_image(
                image_paths[0],
                lang=args.lang,
                use_angle_cls=not args.no_angle,
                clean_text=args.clean
            )
        else:
            print(f"Processing {len(image_paths)} images...\n")
            text = extract_text_from_images(
                image_paths,
                lang=args.lang,
                use_angle_cls=not args.no_angle,
                clean_text=args.clean
            )
    except Exception as e:
        print(f"ERROR during extraction: {e}")
        sys.exit(1)

    # Save or display results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nExtracted text saved to: {args.output}")
    else:
        print("\n" + "="*60)
        print("EXTRACTED TEXT:")
        print("="*60)
        print(text)
        print("="*60)

        # Offer to save
        if len(image_paths) == 1:
            default_output = os.path.splitext(image_paths[0])[0] + '.txt'
        else:
            default_output = 'extracted_text.txt'

        save = input(
            f"\nSave to file? (y/n, default: {default_output}): ").strip().lower()
        if save in ('y', 'yes', ''):
            output_path = input(
                f"Output filename [{default_output}]: ").strip() or default_output
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()
