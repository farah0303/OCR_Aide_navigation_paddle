"""
Unified text and table extraction module.

This module combines OCR text extraction with table detection and extraction,
providing a single interface for processing documents with both text and tables.
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image, ImageDraw
import fitz

try:
    from paddleocr import PaddleOCR
    from table_extractor import TableExtractionPipeline, ExtractedTable
    from table_extractor.config import Config
except ImportError as e:
    print(f"Warning: Import failed - {e}")
    PaddleOCR = TableExtractionPipeline = ExtractedTable = Config = None


# Global shared instances (initialized once per session)
_table_pipeline = None
_table_pipeline_use_gpu = None
_ocr_engine = None
_ocr_engine_use_gpu = None


def get_table_pipeline(use_gpu=False):
    """Get or initialize the shared table extraction pipeline (lazy initialization)."""
    global _table_pipeline, _table_pipeline_use_gpu
    if _table_pipeline is None or _table_pipeline_use_gpu != use_gpu:
        if _table_pipeline is not None:
            print("ℹ️  Reinitializing table pipeline for new GPU setting...")
        print("🔧 Initializing table extraction pipeline (this may take 60-120 seconds)...")
        config = Config(
            pdf_dpi=200,  # Match OCR zoom level (zoom=2.0 → ~200 DPI)
            ocr_lang='fr',
            table_conf_threshold=0.5,
            save_html=False,
            save_csv=False,
            save_json=False,
            save_images=False
        )
        _table_pipeline = TableExtractionPipeline(config=config, use_gpu=use_gpu)
        _table_pipeline_use_gpu = use_gpu
        print("✅ Table extraction pipeline initialized")
    return _table_pipeline


def get_ocr_engine(lang='fr', use_gpu=False):
    """Get or initialize the shared OCR engine (lazy initialization)."""
    global _ocr_engine, _ocr_engine_use_gpu
    if _ocr_engine is None or _ocr_engine_use_gpu != use_gpu:
        if not PaddleOCR:
            raise RuntimeError("PaddleOCR unavailable")
        print("🔧 Initializing OCR engine...")
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
        _ocr_engine_use_gpu = use_gpu
        print("✅ OCR engine initialized")
    return _ocr_engine


# Removed detect_tables_in_document - not needed for scanned documents
# The table pipeline will do proper image-based detection


def mask_table_regions(image, table_bboxes):
    """
    Create a masked version of the image with table regions painted white.
    
    Args:
        image: PIL Image
        table_bboxes: List of (x1, y1, x2, y2) bounding boxes
        
    Returns:
        PIL Image with table regions masked
    """
    if not table_bboxes:
        return image
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Create a copy
    masked = img_array.copy()
    
    # Paint table regions white
    for bbox in table_bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, masked.shape[1] - 1))
        x2 = max(0, min(x2, masked.shape[1]))
        y1 = max(0, min(y1, masked.shape[0] - 1))
        y2 = max(0, min(y2, masked.shape[0]))
        
        if x2 > x1 and y2 > y1:
            masked[y1:y2, x1:x2] = 255
    
    # Convert back to PIL Image
    return Image.fromarray(masked)


def ocr_image_with_exclusions(ocr_engine, image, exclude_bboxes=None):
    """
    Extract text from image, excluding specified regions.
    
    Args:
        ocr_engine: Initialized PaddleOCR instance
        image: PIL Image
        exclude_bboxes: List of (x1, y1, x2, y2) regions to exclude
        
    Returns:
        str: Extracted text
    """
    # Mask table regions
    if exclude_bboxes:
        image = mask_table_regions(image, exclude_bboxes)
    
    # Run OCR
    img_np = np.array(image)
    result = ocr_engine.ocr(img_np)
    
    # Extract text lines
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


def format_table_as_markdown(table):
    """
    Convert an ExtractedTable to markdown format.
    
    Args:
        table: ExtractedTable object with dataframe
        
    Returns:
        str: Markdown representation of the table
    """
    if table.dataframe is None or table.dataframe.empty:
        return "[Empty Table]"
    
    try:
        # Use pandas to_markdown if available
        if hasattr(table.dataframe, 'to_markdown'):
            return table.dataframe.to_markdown(index=False)
        else:
            # Fallback: simple text table
            return format_table_as_simple_text(table)
    except Exception as e:
        print(f"Warning: Could not format table as markdown: {e}")
        return f"[Table with {len(table.dataframe)} rows × {len(table.dataframe.columns)} columns]"


def format_table_as_simple_text(table):
    """
    Convert an ExtractedTable to simple text format with borders.
    
    Args:
        table: ExtractedTable object with dataframe
        
    Returns:
        str: Text representation of the table
    """
    if table.dataframe is None or table.dataframe.empty:
        return "[Empty Table]"
    
    df = table.dataframe
    
    # Calculate column widths
    col_widths = []
    for col in df.columns:
        max_width = max(
            len(str(col)),
            df[col].astype(str).map(len).max() if len(df) > 0 else 0
        )
        col_widths.append(max_width)
    
    # Build header
    header = ' | '.join(str(col).ljust(w) for col, w in zip(df.columns, col_widths))
    separator = '-+-'.join('-' * w for w in col_widths)
    
    # Build rows
    rows = []
    for _, row in df.iterrows():
        row_str = ' | '.join(str(val).ljust(w) for val, w in zip(row, col_widths))
        rows.append(row_str)
    
    return '\n'.join([header, separator] + rows)


def extract_document_with_tables(file_path, zoom=2.0, use_gpu=False, return_details=False):
    """
    Extract text and tables from a document (PDF or image).
    
    This function:
    1. Always initializes table pipeline (needed for scanned documents)
    2. Extracts tables using PP-StructureV3 image analysis
    3. Extracts text from non-table regions
    4. Combines text and tables in reading order
    
    Args:
        file_path: Path to PDF or image file
        zoom: PDF rendering zoom factor (default 2.0 → ~200 DPI)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        str: Combined text with embedded tables in markdown format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"📄 Processing: {os.path.basename(file_path)}")
    pages_count = 1
    if file_path.lower().endswith('.pdf'):
        try:
            doc_probe = fitz.open(file_path)
            pages_count = doc_probe.page_count
            doc_probe.close()
        except Exception as e:
            print(f"⚠️  Could not read page count: {e}")
    
    # Initialize table pipeline (required for scanned documents)
    print("🔧 Initializing table extraction pipeline...")
    table_pipeline = get_table_pipeline(use_gpu)
    
    # Extract tables using PP-StructureV3 (detects tables in images)
    print("🔍 Detecting and extracting tables...")
    try:
        tables = table_pipeline.extract(file_path, output_dir=None)
        if tables:
            print(f"✅ Found {len(tables)} table(s)")
        else:
            print("ℹ️  No tables detected in document")
    except Exception as e:
        print(f"⚠️  Table extraction failed: {e}")
        import traceback
        traceback.print_exc()
        print("   Falling back to text-only extraction")
        return extract_text_only(file_path, zoom, use_gpu=use_gpu, return_details=return_details, pages_hint=pages_count)
    
    # If no tables found, use standard text extraction
    tables_count = len(tables) if tables else 0

    if not tables:
        print("   Using standard text extraction for non-table content")
        return extract_text_only(file_path, zoom, use_gpu=use_gpu, return_details=return_details, pages_hint=pages_count)
    
    # Organize tables by page
    tables_by_page = {}
    for table in tables:
        page_num = table.page_number
        if page_num not in tables_by_page:
            tables_by_page[page_num] = []
        tables_by_page[page_num].append(table)
    
    # Initialize OCR engine
    ocr = get_ocr_engine(lang='fr', use_gpu=use_gpu)
    
    # Process document
    if file_path.lower().endswith('.pdf'):
        text = extract_pdf_with_tables(file_path, tables_by_page, ocr, zoom, pages_count)
    else:
        text = extract_image_with_tables(file_path, tables_by_page, ocr, pages_count)

    if return_details:
        return text, {"tables": tables_count, "pages": pages_count}
    return text


def extract_text_only(file_path, zoom=2.0, use_gpu=False, return_details=False, pages_hint=None):
    """
    Extract text without table detection (fallback).
    
    Args:
        file_path: Path to PDF or image file
        zoom: PDF rendering zoom factor
        
    Returns:
        str: Extracted text
    """
    pages_count = pages_hint if pages_hint is not None else (1 if not file_path.lower().endswith('.pdf') else None)

    if file_path.lower().endswith('.pdf'):
        from extract_text_pdf import extract_text_from_pdf
        if pages_count is None:
            try:
                probe = fitz.open(file_path)
                pages_count = probe.page_count
                probe.close()
            except Exception:
                pages_count = 0
        text = extract_text_from_pdf(file_path, pages=None, zoom=zoom)
    else:
        from extract_text_image import extract_text_from_image
        pages_count = 1 if pages_count is None else pages_count
        text = extract_text_from_image(file_path, lang='fr', use_angle_cls=True, clean_text=True, use_gpu=use_gpu)

    if return_details:
        return text, {"tables": 0, "pages": pages_count}
    return text


def extract_pdf_with_tables(pdf_path, tables_by_page, ocr, zoom=2.0, pages_count=None):
    """
    Extract text and tables from PDF, merging them in reading order.
    
    Args:
        pdf_path: Path to PDF file
        tables_by_page: Dict mapping page_number -> List[ExtractedTable]
        ocr: Initialized PaddleOCR instance
        zoom: Rendering zoom factor
        
    Returns:
        str: Combined text with embedded tables
    """
    doc = fitz.open(pdf_path)
    if pages_count is None:
        pages_count = doc.page_count
    all_text = []
    
    for page_num in range(doc.page_count):
        page_number = page_num + 1  # 1-indexed for display
        print(f"  📄 Processing page {page_number}/{doc.page_count}...")
        
        page = doc.load_page(page_num)
        
        # Render page to image
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Get tables for this page
        page_tables = tables_by_page.get(page_number, [])
        
        # Extract table bounding boxes (in image coordinates)
        table_bboxes = []
        if page_tables:
            # Tables are extracted at 200 DPI, we're rendering at zoom*72 DPI
            # Scale factor: (zoom * 72) / 200
            scale_factor = (zoom * 72) / 200
            
            for table in page_tables:
                x1, y1, x2, y2 = table.bbox
                # Scale bbox to match our rendering
                scaled_bbox = (
                    int(x1 * scale_factor),
                    int(y1 * scale_factor),
                    int(x2 * scale_factor),
                    int(y2 * scale_factor)
                )
                table_bboxes.append(scaled_bbox)
        
        # Extract text excluding table regions
        text = ocr_image_with_exclusions(ocr, pil_img, table_bboxes)
        
        # Build page output
        page_output = [f"-- PAGE {page_number} --"]
        
        if text.strip():
            page_output.append(text)
        
        # Add tables inline
        if page_tables:
            for table in sorted(page_tables, key=lambda t: t.table_number):
                page_output.append("")
                page_output.append(f"[TABLE {table.table_number}]")
                page_output.append(format_table_as_markdown(table))
                page_output.append("")
        
        all_text.append("\n".join(page_output))
    
    doc.close()
    return "\n\n".join(all_text).strip()


def extract_image_with_tables(image_path, tables_by_page, ocr, pages_count=1):
    """
    Extract text and tables from image, merging them in reading order.
    
    Args:
        image_path: Path to image file
        tables_by_page: Dict mapping page_number -> List[ExtractedTable]
        ocr: Initialized PaddleOCR instance
        
    Returns:
        str: Combined text with embedded tables
    """
    from extract_text_image import load_image
    
    pil_img = load_image(image_path)
    
    # Get tables (should be page 1)
    page_tables = tables_by_page.get(1, [])
    
    # Extract table bounding boxes
    table_bboxes = [table.bbox for table in page_tables]
    
    # Extract text excluding table regions
    text = ocr_image_with_exclusions(ocr, pil_img, table_bboxes)
    
    # Build output
    output = [f"-- {os.path.basename(image_path)} --"]
    
    if text.strip():
        output.append(text)
    
    # Add tables inline
    if page_tables:
        for table in sorted(page_tables, key=lambda t: t.table_number):
            output.append("")
            output.append(f"[TABLE {table.table_number}]")
            output.append(format_table_as_markdown(table))
            output.append("")
    
    return "\n".join(output).strip()


# Convenience function for backward compatibility
def extract_unified(file_path, zoom=2.0, use_gpu=False):
    """
    Main entry point for unified extraction.
    
    Args:
        file_path: Path to PDF or image file
        zoom: PDF rendering zoom factor (default 2.0)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        str: Extracted text with embedded tables
    """
    return extract_document_with_tables(file_path, zoom=zoom, use_gpu=use_gpu)
