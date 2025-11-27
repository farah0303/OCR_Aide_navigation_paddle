"""
Table structure recognition using PaddleOCR PP-StructureV3.
"""

from typing import Dict, Any, Optional, Tuple
from PIL import Image
import logging

from .utils import setup_logger, image_to_numpy
from .config import Config


logger = setup_logger(__name__)


class TableStructureRecognizer:
    """Recognizes table structure using PP-StructureV3."""
    
    def __init__(self, config: Config, shared_engine=None):
        """
        Initialize table structure recognizer.
        
        Args:
            config: Configuration object
            shared_engine: Optional pre-initialized PPStructureV3 instance to reuse
        """
        self.config = config
        self.table_engine = shared_engine
        if shared_engine is None:
            self._initialize_table_engine()
    
    def _initialize_table_engine(self):
        """Initialize PP-StructureV3 table recognition engine."""
        try:
            from paddleocr import PPStructureV3
            
            logger.info("Initializing PP-StructureV3 table recognition engine...")
            
            # Initialize PP-Structure with minimal parameters
            self.table_engine = PPStructureV3()
            
            logger.info("PP-StructureV3 table recognition engine initialized successfully")
            
        except ImportError:
            logger.error("PaddleOCR is not installed. Install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize table engine: {e}")
            raise
    
    def recognize_structure(self, table_image: Image.Image) -> Dict[str, Any]:
        """
        Recognize table structure and extract HTML.
        
        Args:
            table_image: PIL Image of table region
            
        Returns:
            Dictionary containing:
                - html: HTML representation with rowspan/colspan
                - cells: List of cell information
                - confidence: Overall confidence score
        """
        try:
            img_array = image_to_numpy(table_image)
            
            logger.debug(f"Running table structure recognition on image of size {table_image.size}")
            
            # Run table recognition using predict() method
            result = self.table_engine.predict(img_array)
            
            if not result:
                logger.warning("No table structure detected")
                return self._empty_result()
            
            # Extract table information
            table_html = None
            cells = []
            confidence = 0.0
            
            for item in result:
                if item['type'] == 'table':
                    # Extract HTML
                    table_html = item.get('res', {}).get('html', '')
                    
                    # Extract cell information
                    if 'res' in item and 'cell_bbox' in item['res']:
                        cells = item['res']['cell_bbox']
                    
                    # Get confidence
                    confidence = item.get('score', item.get('confidence', 0.0))
                    
                    logger.debug(f"Extracted table with confidence {confidence:.2f}")
                    break
            
            if not table_html:
                logger.warning("No HTML table extracted")
                return self._empty_result()
            
            return {
                'html': table_html,
                'cells': cells,
                'confidence': confidence,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Table structure recognition failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'html': '',
            'cells': [],
            'confidence': 0.0,
            'success': False
        }
    
    def extract_cell_images(self, table_image: Image.Image, 
                           cells: list) -> Dict[Tuple[int, int], Image.Image]:
        """
        Extract individual cell images from table.
        
        Args:
            table_image: PIL Image of table
            cells: List of cell bounding boxes
            
        Returns:
            Dictionary mapping (row, col) to cell image
        """
        from .utils import crop_image, normalize_bbox
        
        cell_images = {}
        
        for cell in cells:
            try:
                bbox = cell.get('bbox', cell.get('cell_bbox'))
                if not bbox:
                    continue
                
                # Normalize bbox
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    normalized = normalize_bbox(
                        [int(x1), int(y1), int(x2), int(y2)],
                        table_image.width,
                        table_image.height
                    )
                    
                    # Get row/col position
                    row = cell.get('row', cell.get('row_start', 0))
                    col = cell.get('col', cell.get('col_start', 0))
                    
                    # Crop cell image
                    cell_img = crop_image(table_image, normalized)
                    cell_images[(row, col)] = cell_img
                    
            except Exception as e:
                logger.warning(f"Failed to extract cell image: {e}")
                continue
        
        return cell_images
    
    def validate_html(self, html: str) -> bool:
        """
        Validate HTML table structure.
        
        Args:
            html: HTML string
            
        Returns:
            True if valid, False otherwise
        """
        if not html or not html.strip():
            return False
        
        # Check for basic table tags
        required_tags = ['<table', '</table>', '<tr', '</tr>']
        return all(tag in html.lower() for tag in required_tags)
