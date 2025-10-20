# OCR with PaddleOCR and PyMuPDF

This small utility extracts text from PDF files. It will try to use embedded/selectable text first and fall back to rendering pages and running PaddleOCR on images for scanned PDFs.

Quick install (PowerShell):

```powershell
# activate your venv first (if using the provided venv)
.\ven\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
# Note: for better speed/accuracy you may want to install a PaddlePaddle wheel matching your machine (see https://www.paddlepaddle.org.cn/install/quick)
```

Usage:

### Single File Processing

```powershell
# PDF with all pages
python main.py --file document.pdf -o output.txt

# PDF with specific pages only
python main.py --file document.pdf -o out.txt --pages 1,3-5

# Image file
python main.py --file scan.jpg -o output.txt --lang en

# Image with text cleaning
python main.py --file photo.png -o out.txt --clean
```

### Batch Processing

```powershell
# Process multiple files at once (no --batch flag needed)
python main.py --file doc1.pdf image1.jpg doc2.pdf -o combined.txt

# Interactive selection mode
python main.py --batch
```

### Advanced Options

```powershell
# High-quality PDF rendering with 3x zoom
python main.py -f scan.pdf -z 3.0 -o output.txt

# Disable angle detection for faster processing
python main.py -f image.jpg -n -o out.txt

# French language OCR with cleaning
python main.py -f french_doc.png -l fr -c -o output.txt

# Super compact: PDF pages 1-3, zoom 3x, output to file
python main.py -f doc.pdf -p 1-3 -z 3 -o out.txt
```

## Command-Line Options

| Option       | Shorthand | Description                          | Default                  |
| ------------ | --------- | ------------------------------------ | ------------------------ |
| `--file`     | `-f`      | File(s) to process (PDFs or images)  | Interactive selection    |
| `--output`   | `-o`      | Output text file                     | None (prints to console) |
| `--lang`     | `-l`      | OCR language (en, fr, ch, etc.)      | `en`                     |
| `--pages`    | `-p`      | PDF pages to process (e.g., `1,3-5`) | All pages                |
| `--zoom`     | `-z`      | PDF render quality multiplier        | `2.0`                    |
| `--clean`    | `-c`      | Apply text cleaning/formatting       | `False`                  |
| `--no-angle` | `-n`      | Disable text angle detection         | `False`                  |
| `--batch`    | `-b`      | Interactive mode for multiple files  | `False`                  |

## How It Works

### PDF Processing Flow

1. **Check for embedded text** → Extract if available (fastest)
2. **If scanned/no text** → Render pages as images
3. **Apply PaddleOCR** → Extract text from rendered images
4. **Combine results** → Merge text from all pages

### Image Processing Flow

1. **Detect image format** → Validate supported extension
2. **Load image** → Preprocess if needed
3. **Run PaddleOCR** → Detect text regions and recognize characters
4. **Post-process** → Apply angle correction and cleaning if requested
5. **Return text** → Formatted and cleaned output

## Notes

- **First Run**: PaddleOCR downloads model files (~100MB) on first execution
- **Performance**: PDFs with embedded text are processed instantly; OCR is slower
- **Memory**: Higher `--zoom` values improve quality but use more RAM
- **Batch Jobs**: For very large batches, consider splitting work or adding multiprocessing
- **Supported Formats**:
  - PDFs: `.pdf`
  - Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`

## Troubleshooting

**Issue**: "No module named paddleocr"

- **Fix**: Run `pip install paddleocr`

**Issue**: Slow processing on images

- **Fix**: Use `--no-angle` flag to disable angle detection

**Issue**: Poor OCR quality

- **Fix**: For PDFs, increase `--zoom` value

**Issue**: Wrong language detected

- **Fix**: Specify language explicitly with `--lang` (e.g., `--lang en` for English)
