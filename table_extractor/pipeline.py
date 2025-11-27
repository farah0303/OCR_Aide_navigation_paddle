"""
Main table extraction pipeline orchestrating all components.
"""

from dataclasses import dataclass
from typing import List, Optional, Union
from pathlib import Path
import logging
from PIL import Image

from .config import Config, DEFAULT_CONFIG
from .utils import setup_logger, ensure_dir, crop_image
from .pdf_loader import PDFLoader
from .table_detector import TableDetector
from .table_structure import TableStructureRecognizer
from .ocr_engine import OCREngine
from .html_parser import HTMLParser
from .postprocessing import TablePostprocessor
import pandas as pd


logger = setup_logger(__name__)


@dataclass
class ExtractedTable:
    """Container for extracted table data."""
    page_number: int
    table_number: int
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    html: str
    dataframe: Optional[pd.DataFrame]
    json_data: dict
    image_crop: Image.Image
    cell_bboxes: Optional[List[dict]] = None  # List of cell bounding boxes with row/col info


class TableExtractionPipeline:
    """
    Complete pipeline for extracting tables from PDFs and images.
    
    This pipeline:
    1. Loads PDF/images
    2. Detects table regions
    3. Recognizes table structure
    4. Performs OCR on cells
    5. Parses HTML to DataFrame
    6. Applies postprocessing
    7. Saves results
    """
    
    def __init__(self, config: Optional[Config] = None, use_gpu: bool = False):
        """
        Initialize extraction pipeline.
        
        Args:
            config: Configuration object (uses DEFAULT_CONFIG if None)
            use_gpu: Whether to use GPU acceleration
        """
        self.config = config or DEFAULT_CONFIG
        self.config.use_gpu = use_gpu
        
        logger.info("Initializing Table Extraction Pipeline...")
        logger.info(f"GPU enabled: {use_gpu}")
        
        # Initialize shared PPStructure instance once (saves ~60-120 seconds and ~500MB memory)
        # This instance is reused by both detector and structure recognizer
        from paddleocr import PPStructure
        logger.info("Initializing shared PP-Structure instance...")
        self._shared_structure = PPStructure()
        logger.info("Shared PP-Structure initialized")
        
        # Initialize components with shared instance
        self.pdf_loader = PDFLoader(self.config)
        self.table_detector = TableDetector(self.config, shared_analyzer=self._shared_structure)
        self.structure_recognizer = TableStructureRecognizer(self.config, shared_engine=self._shared_structure)
        self.ocr_engine = None  # Lazy initialization (PPStructureV3 already includes OCR)
        self.html_parser = HTMLParser()
        self.postprocessor = TablePostprocessor(self.config)
        
        logger.info("Pipeline initialized successfully")
    
    def _ensure_ocr_engine(self):
        """Lazy initialize OCR engine only if needed."""
        if self.ocr_engine is None:
            logger.info("Initializing OCR engine (lazy)...")
            self.ocr_engine = OCREngine(self.config)
    
    def extract(self, input_path: Union[str, Path], 
                output_dir: Optional[Union[str, Path]] = None) -> List[ExtractedTable]:
        """
        Extract all tables from input file.
        
        Args:
            input_path: Path to PDF or image file
            output_dir: Optional output directory for saving results
            
        Returns:
            List of ExtractedTable objects
        """
        input_path = Path(input_path)
        logger.info(f"Starting extraction from: {input_path}")
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Load images
        try:
            images = self.pdf_loader.load(input_path)
            logger.info(f"Loaded {len(images)} page(s)")
        except Exception as e:
            logger.error(f"Failed to load input: {e}")
            raise
        
        # Extract tables from all pages
        all_tables = []
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            page_tables = self._extract_from_page(image, page_num)
            all_tables.extend(page_tables)
            
            logger.info(f"Found {len(page_tables)} table(s) on page {page_num}")
        
        logger.info(f"Total tables extracted: {len(all_tables)}")
        
        # Save results if output directory specified
        if output_dir:
            self._save_results(all_tables, output_dir)
        
        return all_tables
    
    def _extract_from_page(self, image: Image.Image, page_number: int) -> List[ExtractedTable]:
        """
        Extract all tables from a single page.
        
        Args:
            image: PIL Image of page
            page_number: Page number
            
        Returns:
            List of ExtractedTable objects
        """
        # Apply preprocessing to entire page
        preprocessed_image = self.postprocessor.detect_and_correct_skew(image)
        preprocessed_image = self.postprocessor.preprocess_image(preprocessed_image)
        
        # Detect table regions
        detected_tables = self.table_detector.detect_tables(preprocessed_image)
        
        if not detected_tables:
            logger.info(f"No tables detected on page {page_number}")
            return []
        
        # Filter overlapping detections
        detected_tables = self.table_detector.filter_overlapping_tables(detected_tables)
        
        # Extract each table
        extracted_tables = []
        for table_idx, table_info in enumerate(detected_tables, 1):
            try:
                extracted = self._extract_single_table(
                    preprocessed_image,
                    table_info,
                    page_number,
                    table_idx
                )
                if extracted:
                    extracted_tables.append(extracted)
            except Exception as e:
                logger.error(f"Failed to extract table {table_idx} on page {page_number}: {e}")
                continue
        
        return extracted_tables
    
    def _extract_single_table(self, image: Image.Image, table_info: dict,
                             page_number: int, table_number: int) -> Optional[ExtractedTable]:
        """
        Extract a single table.
        
        Args:
            image: PIL Image of page
            table_info: Table detection info with bbox
            page_number: Page number
            table_number: Table number on page
            
        Returns:
            ExtractedTable object or None if extraction fails
        """
        bbox = table_info['bbox']
        confidence = table_info['confidence']
        
        logger.debug(f"Extracting table {table_number} at {bbox}")
        
        # Crop table region
        table_image = crop_image(image, bbox)
        
        # Apply table-specific enhancements
        table_image = self.postprocessor.enhance_table_region(table_image)
        
        # Check if HTML is already provided from detection (PaddleOCR 3.x PPStructureV3)
        html = table_info.get('html', '')
        cell_bboxes = None
        
        if html:
            logger.debug(f"Using HTML from PPStructureV3 detection for table {table_number}")
        else:
            # Fallback: Recognize table structure if not provided
            logger.debug(f"Running structure recognition for table {table_number}")
            structure_result = self.structure_recognizer.recognize_structure(table_image)
            
            if not structure_result['success']:
                logger.warning(f"Failed to recognize structure for table {table_number}")
                return None
            
            html = structure_result['html']
            cell_bboxes = structure_result.get('cells', [])
        
        # Validate HTML
        if not html or not self.structure_recognizer.validate_html(html):
            logger.warning(f"Invalid HTML for table {table_number}")
            return None
        
        # Parse HTML to DataFrame
        dataframe = self.html_parser.html_to_dataframe(html)
        
        if dataframe is None:
            logger.warning(f"Failed to parse HTML to DataFrame for table {table_number}")
            return None
        
        # Apply postprocessing to DataFrame
        dataframe = self.postprocessor.clean_dataframe(dataframe)
        
        # Validate table quality
        if not self.postprocessor.validate_table(dataframe):
            logger.warning(f"Table {table_number} failed quality validation")
            return None
        
        # Convert to JSON
        json_data = self.html_parser.html_to_json(html)
        
        # Create ExtractedTable object
        extracted = ExtractedTable(
            page_number=page_number,
            table_number=table_number,
            bbox=bbox,
            confidence=confidence,
            html=html,
            dataframe=dataframe,
            json_data=json_data,
            image_crop=table_image,
            cell_bboxes=cell_bboxes
        )
        
        logger.debug(f"Successfully extracted table {table_number} with shape {dataframe.shape}")
        return extracted
    
    def _save_results(self, tables: List[ExtractedTable], output_dir: Union[str, Path]):
        """
        Save extraction results to directory.
        
        Args:
            tables: List of ExtractedTable objects
            output_dir: Output directory path
        """
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        
        logger.info(f"Saving results to: {output_dir}")
        
        for table in tables:
            # Create directory for this table
            table_dir = output_dir / f"page_{table.page_number}" / f"table_{table.table_number}"
            ensure_dir(table_dir)
            
            # Save HTML
            if self.config.save_html:
                html_path = table_dir / "table.html"
                try:
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(table.html)
                    logger.debug(f"Saved HTML: {html_path}")
                except Exception as e:
                    logger.error(f"Failed to save HTML: {e}")
            
            # Save CSV
            if self.config.save_csv and table.dataframe is not None:
                csv_path = table_dir / "table.csv"
                try:
                    self.html_parser.save_csv(table.dataframe, str(csv_path))
                except Exception as e:
                    logger.error(f"Failed to save CSV: {e}")
            
            # Save JSON
            if self.config.save_json:
                json_path = table_dir / "table.json"
                try:
                    self.html_parser.save_json(table.json_data, str(json_path))
                except Exception as e:
                    logger.error(f"Failed to save JSON: {e}")
            
            # Save image
            if self.config.save_images:
                image_path = table_dir / "table.png"
                try:
                    table.image_crop.save(image_path)
                    logger.debug(f"Saved image: {image_path}")
                except Exception as e:
                    logger.error(f"Failed to save image: {e}")
        
        logger.info(f"Successfully saved {len(tables)} table(s)")
    
    def extract_from_image(self, image: Image.Image) -> List[ExtractedTable]:
        """
        Extract tables from a single image (convenience method).
        
        Args:
            image: PIL Image
            
        Returns:
            List of ExtractedTable objects
        """
        return self._extract_from_page(image, page_number=1)
