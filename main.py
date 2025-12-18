"""Unified OCR for PDFs and Images with PaddleOCR (French only, outputs in 'outputs/')"""
import argparse
import csv
import glob
import os
import sys
import time
import traceback
import fitz
from utils_drive.drive_utils import upload_to_box

# Try to import AI correction, but make it optional
try:
    from advanced_correction.advanced_correction import corriger_texte_ai
    AI_CORRECTION_AVAILABLE = True
except:
    AI_CORRECTION_AVAILABLE = False
    print("⚠️  AI correction not available (OpenAI API key missing or module error)")

try:
    from extract_text_pdf import extract_text_from_pdf
except:
    extract_text_from_pdf = None

try:
    from extract_text_image import extract_text_from_image, is_image_file, SUPPORTED_EXTENSIONS
except:
    extract_text_from_image = is_image_file = SUPPORTED_EXTENSIONS = None


def supported_image_exts():
    """Return a set of supported image extensions, with sensible defaults if module import failed."""
    if SUPPORTED_EXTENSIONS:
        return {e.lower() for e in SUPPORTED_EXTENSIONS}
    return {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def detect_file_type(fp):
    if not os.path.exists(fp):
        return 'unknown'
    ext = os.path.splitext(fp)[1].lower()
    return 'pdf' if ext == '.pdf' else ('image' if is_image_file and is_image_file(fp) else 'unknown')


def find_example_files():
    pdf_dir = os.path.join("example", "pdf")
    img_dir = os.path.join("example", "img")
    files = []

    files.extend(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    files.extend(glob.glob(os.path.join(pdf_dir, "*.PDF")))

    if SUPPORTED_EXTENSIONS:
        for e in SUPPORTED_EXTENSIONS:
            files.extend(glob.glob(os.path.join(img_dir, f"*{e}")))
            files.extend(glob.glob(os.path.join(img_dir, f"*{e.upper()}")))

    return sorted(set(files))


def choose_file(fps):
    if not fps:
        print("❌ Aucun fichier trouvé.")
        sys.exit(2)

    print("\nFichiers disponibles :\n")
    for i, p in enumerate(fps, 1):
        print(f"  [{i}] {os.path.basename(p)} ({detect_file_type(p).upper()})")

    while True:
        c = input(f"\nChoisissez un fichier (1-{len(fps)}) ou 'q' pour quitter : ").strip()
        if c.lower() == 'q':
            sys.exit(0)
        try:
            index = int(c)
            if 1 <= index <= len(fps):
                return fps[index - 1]
        except:
            pass
        print("Sélection invalide.")


def gather_input_files(inputs, recursive=False):
    """Expand files and folders into a de-duplicated list of supported documents."""
    exts = supported_image_exts() | {".pdf"}
    collected = []

    for path in inputs:
        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in exts:
                collected.append(path)
            else:
                print(f"⚠️  Ignored unsupported file: {path}")
        elif os.path.isdir(path):
            for root, _, filenames in os.walk(path):
                for name in filenames:
                    ext = os.path.splitext(name)[1].lower()
                    if ext in exts:
                        collected.append(os.path.join(root, name))
                if not recursive:
                    break
        else:
            print(f"⚠️  Path not found: {path}")

    return sorted(set(collected))


def build_output_path(fp, used_names):
    """Create a unique output path alongside the source file to avoid collisions."""
    base_dir = os.path.dirname(fp)
    stem = os.path.splitext(os.path.basename(fp))[0]
    candidate = os.path.join(base_dir, f"{stem}.txt")
    suffix = 1

    while candidate in used_names:
        candidate = os.path.join(base_dir, f"{stem}_{suffix}.txt")
        suffix += 1

    used_names.add(candidate)
    return candidate


def default_output_path(fp):
    base_dir = os.path.dirname(fp)
    stem = os.path.splitext(os.path.basename(fp))[0]
    return os.path.join(base_dir, f"{stem}.txt")


def load_existing_results(csv_path):
    if not os.path.exists(csv_path):
        return {}
    existing = {}
    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fp = row.get("filepath")
                if fp:
                    existing[fp] = row
    except Exception as e:
        print(f"⚠️  Could not read existing results.csv: {e}")
    return existing


def get_page_count(fp):
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".pdf":
        try:
            doc = fitz.open(fp)
            count = doc.page_count
            doc.close()
            return count
        except Exception:
            return 0
    elif detect_file_type(fp) == "image":
        return 1
    return 0


def process_file(fp, use_angle=True, zoom=2.0, use_gpu=False):
    """Process file with unified text and table extraction.

    Returns:
        tuple[str, dict]: extracted text and metadata (tables, pages)
    """
    try:
        from extract_text_unified import extract_document_with_tables
        # Use unified extraction (automatically handles tables if present)
        text, details = extract_document_with_tables(fp, zoom=zoom, use_gpu=use_gpu, return_details=True)
        return text, details
    except ImportError as e:
        # Fallback to old method if unified extractor not available
        print(f"⚠️  Unified extractor not available ({e}), using standard text extraction")
        ft = detect_file_type(fp)
        pages = get_page_count(fp)
        if ft == 'pdf':
            if not extract_text_from_pdf:
                raise RuntimeError("Module PDF indisponible")
            text = extract_text_from_pdf(fp, zoom=zoom)
            return text, {"tables": 0, "pages": pages}
        elif ft == 'image':
            if not extract_text_from_image:
                raise RuntimeError("Module image indisponible")
            text = extract_text_from_image(fp, lang='fr', use_angle_cls=use_angle, clean_text=True, use_gpu=use_gpu)
            return text, {"tables": 0, "pages": pages or 1}
        raise ValueError(f"Format non supporté : {fp}")


def ensure_output_folder():
    out_dir = "outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="OCR for PDFs and images (batch-friendly)")
    parser.add_argument("inputs", nargs="*", help="Files or folders to process. If empty, uses interactive sample picker.")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recurse into subfolders when a folder is provided.")
    parser.add_argument("-z", "--zoom", type=float, default=2.0, help="PDF render zoom (quality vs speed).")
    parser.add_argument("-n", "--no-angle", action="store_true", help="Disable angle detection for images (faster).")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for OCR and table extraction.")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI correction even if available.")
    parser.add_argument("--no-upload", action="store_true", help="Skip Box uploads for batch runs.")
    parser.add_argument("--resume", action="store_true", help="Skip files already processed (results.csv or existing output txt).")
    parser.add_argument("--serve-api", action="store_true", help="Start the HTTP API server instead of running a batch.")
    parser.add_argument("--api-host", default="0.0.0.0", help="Host for the API server (default: 0.0.0.0).")
    parser.add_argument("--api-port", type=int, default=5000, help="Port for the API server (default: 5000).")

    args = parser.parse_args()

    try:
        if args.serve_api:
            from api import app

            print(f"🌐 Starting API server on {args.api_host}:{args.api_port} ...")
            app.run(host=args.api_host, port=args.api_port)
            return

        if args.inputs:
            files = gather_input_files(args.inputs, recursive=args.recursive)
            if not files:
                print("❌ Aucun fichier valide trouvé dans les chemins fournis.")
                sys.exit(2)
        else:
            files = find_example_files()
            fp = choose_file(files)
            files = [fp]

        output_dir = ensure_output_folder()
        used_output_paths = set()
        csv_path = os.path.join(output_dir, "results.csv")
        existing_results = load_existing_results(csv_path) if args.resume else {}

        results = []
        total_files = len(files)

        for idx, fp in enumerate(files, 1):
            print(f"\n📝 ({idx}/{total_files}) Extraction OCR pour : {fp}")
            try:
                rel_input = os.path.relpath(fp)
                candidate_output = default_output_path(fp)

                if args.resume:
                    already_logged = rel_input in existing_results
                    already_output = os.path.exists(candidate_output)
                    if already_logged or already_output:
                        print("⏭️  Resume: skipping (already processed)")
                        continue

                t0 = time.perf_counter()
                ocr_text, details = process_file(fp, use_angle=not args.no_angle, zoom=args.zoom, use_gpu=args.gpu)
                elapsed = time.perf_counter() - t0

                if AI_CORRECTION_AVAILABLE and not args.no_ai:
                    print("\n🤖 Correction AI...")
                    try:
                        corrected = corriger_texte_ai(ocr_text)
                    except Exception as e:
                        print(f"⚠️  AI correction failed: {e}")
                        print("📝 Using uncorrected OCR text instead...")
                        corrected = ocr_text
                else:
                    reason = "not available" if not AI_CORRECTION_AVAILABLE else "flag --no-ai set"
                    print(f"\n⏭️  Skipping AI correction ({reason})")
                    corrected = ocr_text

                output_path = build_output_path(fp, used_output_paths)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(corrected)

                print(f"✅ Texte enregistré dans : {output_path}\n")

                tables = details.get("tables", 0) if isinstance(details, dict) else 0
                pages = details.get("pages", 0) if isinstance(details, dict) else get_page_count(fp)
                results.append({
                    "filepath": rel_input,
                    "pages": pages,
                    "tables": tables,
                    "seconds": round(elapsed, 3)
                })

                if args.no_upload:
                    continue

                # Upload to Box (optional, will fail gracefully if credentials missing)
                try:
                    print("⬆️ Upload du fichier original sur Box ...")
                    original_link = upload_to_box(fp, folder_id="349750293522")

                    print("⬆️ Upload du texte sur Box ...")
                    text_link = upload_to_box(output_path, folder_id="349750293522")

                    print(f"Original file link: {original_link}")
                    print(f"Result text link: {text_link}")
                except Exception as e:
                    print(f"⚠️  Box upload skipped: {e}")

            except Exception as e:
                print(f"❌ Échec du traitement pour {fp} : {e}")
                traceback.print_exc()

        # Write results CSV summary
        if results:
            header = ["filepath", "pages", "tables", "seconds"]
            merged = existing_results if args.resume else {}
            for row in results:
                merged[row["filepath"]] = {k: str(v) for k, v in row.items()}

            with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                writer.writeheader()
                writer.writerows(merged.values())
            print(f"🧾 Résumé écrit dans : {csv_path}")

    except Exception as e:
        print(f"\nERREUR : {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()