"""Unified OCR for PDFs and Images with PaddleOCR (French only, outputs in 'outputs/')"""
import glob
import os
import sys
import traceback
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


def process_file(fp, use_angle=True, zoom=2.0):
    """Process file with unified text and table extraction"""
    try:
        from extract_text_unified import extract_document_with_tables
        # Use unified extraction (automatically handles tables if present)
        return extract_document_with_tables(fp, zoom=zoom, use_gpu=False)
    except ImportError as e:
        # Fallback to old method if unified extractor not available
        print(f"⚠️  Unified extractor not available ({e}), using standard text extraction")
        ft = detect_file_type(fp)
        if ft == 'pdf':
            if not extract_text_from_pdf:
                raise RuntimeError("Module PDF indisponible")
            return extract_text_from_pdf(fp, zoom=zoom)
        elif ft == 'image':
            if not extract_text_from_image:
                raise RuntimeError("Module image indisponible")
            return extract_text_from_image(fp, lang='fr', use_angle_cls=use_angle, clean_text=True)
        raise ValueError(f"Format non supporté : {fp}")


def ensure_output_folder():
    out_dir = "outputs"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    return out_dir


def main():
    try:
        files = find_example_files()
        fp = choose_file(files)

        print("\n📝 Extraction OCR...")
        ocr_text = process_file(fp)

        # Apply AI correction only if available
        if AI_CORRECTION_AVAILABLE:
            print("\n🤖 Correction AI...")
            try:
                corrected = corriger_texte_ai(ocr_text)
            except Exception as e:
                print(f"⚠️  AI correction failed: {e}")
                print("📝 Using uncorrected OCR text instead...")
                corrected = ocr_text
        else:
            print("\n⏭️  Skipping AI correction (not available)")
            corrected = ocr_text

        output_dir = ensure_output_folder()
        output_name = os.path.splitext(os.path.basename(fp))[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)

        # Save the text (corrected or not)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(corrected)

        print(f"\n✅ Texte enregistré dans : {output_path}\n")

        # Upload to Box (optional, will fail gracefully if credentials missing)
        try:
            print("⬆️ Upload du fichier original sur Box ...")
            original_link = upload_to_box(fp, folder_id="349750293522")

            print("⬆️ Upload du texte sur Box ...")
            text_link = upload_to_box(output_path, folder_id="349750293522")

            print(f"\nOriginal file link: {original_link}")
            print(f"Result text link: {text_link}")
        except Exception as e:
            print(f"⚠️  Box upload skipped: {e}")

    except Exception as e:
        print(f"\nERREUR : {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()