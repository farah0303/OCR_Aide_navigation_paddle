# Image OCR Module

This module provides OCR (Optical Character Recognition) functionality for images using PaddleOCR. It's designed to work with various image formats without requiring prior knowledge of the file type.

## Usage

### Basic Usage

Process a single image:

```bash
python main.py --file image.jpg
```

Save output to a specific file:

```bash
python main.py --file image.jpg --output results.txt
```

### Multiple Images

Process multiple images:

```bash
python main.py --file img1.png img2.jpg img3.bmp --output combined.txt
```

Interactive batch mode:

```bash
python main.py --batch
```

### Language Options

Specify OCR language (default is French):

```bash
python main.py --file document.jpg --lang en  # English
python main.py --file document.jpg --lang ch  # Chinese
```

### Additional Options

Apply text cleaning (spell check for French):

```bash
python main.py --file scan.jpg --clean
```

Disable angle classification (faster but less accurate for rotated text):

```bash
python main.py --file image.jpg --no-angle
```

### Interactive Mode

Run without arguments for interactive file selection:

```bash
python main.py
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- WEBP (.webp)
- GIF (.gif)
- PPM, PGM, PBM, PNM (.ppm, .pgm, .pbm, .pnm)

## Module Structure

- `main.py`: Interactive CLI application
- `extract_text_image.py`: Core OCR functionality and utilities

## Examples

### Example 1: Quick OCR

```bash
python main.py --file receipt.jpg
```

### Example 2: Batch Processing with English Language

```bash
python main.py --batch --lang en --clean --output en_docs.txt
```

### Example 3: Process All Images in Directory

```bash
python main.py --file *.jpg --output all_text.txt
```

## API Usage

You can also import and use the functions directly in your Python code:

```python
from extract_text_image import extract_text_from_image, extract_text_from_images

# Single image
text = extract_text_from_image('photo.jpg', lang='en')
print(text)

# Multiple images
images = ['img1.jpg', 'img2.png', 'img3.bmp']
combined_text = extract_text_from_images(images, lang='fr', clean_text=True)
print(combined_text)
```
