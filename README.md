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

```powershell
python extract_text_paddle.py input.pdf -o output.txt
# OCR only pages 1 and 3-4:
python extract_text_paddle.py input.pdf -o out.txt --pages 1,3-4
```

Notes:

- The script uses `paddleocr` which will download model files on first run.
- If your PDFs already contain selectable text, the script returns that instead of running OCR (faster, preserves original text).
- Increase `--zoom` for higher-quality renders at the cost of CPU and memory.
- For large batches consider splitting work per-file or adding multiprocessing.
