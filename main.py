"""Unified OCR for PDFs and Images with PaddleOCR (French only, outputs in 'outputs/')"""
import glob
import os
import sys
import traceback

try:
    from extract_text_pdf import extract_text_from_pdf
except:
    extract_text_from_pdf = None

try:
    from extract_text_image import extract_text_from_image, is_image_file, SUPPORTED_EXTENSIONS
except:
    extract_text_from_image = is_image_file = SUPPORTED_EXTENSIONS = None

# 🔹 Import pour Google Drive
from utils_drive.drive_utils import upload_to_drive, ORIGINALS_FOLDER_ID, TEXTS_FOLDER_ID


def detect_file_type(fp):
    if not os.path.exists(fp):
        return 'unknown'
    ext = os.path.splitext(fp)[1].lower()
    return 'pdf' if ext == '.pdf' else ('image' if is_image_file and is_image_file(fp) else 'unknown')


def find_example_files():
    pdf_dir = os.path.join("example", "pdf")
    img_dir = os.path.join("example", "img")
    files = []

    # PDFs
    files.extend(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    files.extend(glob.glob(os.path.join(pdf_dir, "*.PDF")))

    # Images
    if SUPPORTED_EXTENSIONS:
        for e in SUPPORTED_EXTENSIONS:
            files.extend(glob.glob(os.path.join(img_dir, f"*{e}")))
            files.extend(glob.glob(os.path.join(img_dir, f"*{e.upper()}")))

    return sorted(set(files))


def choose_file(fps):
    if not fps:
        print("❌ Aucun fichier trouvé dans 'example/pdf' ou 'example/img'.")
        sys.exit(2)

    print("\nFichiers disponibles pour OCR :\n")
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
        print("Sélection invalide, essayez encore.")


def process_file(fp, use_angle=True, zoom=2.0):
    ft = detect_file_type(fp)
    if ft == 'pdf':
        if not extract_text_from_pdf:
            raise RuntimeError("Module PDF indisponible")
        print(f"\nTraitement du PDF : {os.path.basename(fp)}")
        return extract_text_from_pdf(fp, zoom=zoom)
    elif ft == 'image':
        if not extract_text_from_image:
            raise RuntimeError("Module image indisponible")
        print(f"\nTraitement de l'image : {os.path.basename(fp)}")
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
        text = process_file(fp, use_angle=True)

        output_dir = ensure_output_folder()
        output_name = os.path.splitext(os.path.basename(fp))[0] + ".txt"
        output_path = os.path.join(output_dir, output_name)

        # 🔹 Sauvegarde du texte OCR localement
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"\n✅ OCR terminé. Résultat enregistré dans : {output_path}\n")

        # 🔹 Upload sur Google Drive
        print("⬆️ Upload du fichier original sur Drive ...")
        upload_to_drive(fp, folder_id=ORIGINALS_FOLDER_ID)

        print("⬆️ Upload du fichier texte OCR sur Drive ...")
        upload_to_drive(output_path, folder_id=TEXTS_FOLDER_ID)

    except Exception as e:
        print(f"\nERREUR : {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
