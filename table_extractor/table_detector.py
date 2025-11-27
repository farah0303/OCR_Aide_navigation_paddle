"""
Table detection using PaddleOCR PP-StructureV3 layout analysis.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import logging

from .utils import setup_logger, image_to_numpy, normalize_bbox, sort_bboxes_top_to_bottom
from .config import Config


logger = setup_logger(__name__)


class TableDetector:
    """Detects table regions using PP-StructureV3 layout analysis."""
    
    def __init__(self, config: Config, shared_analyzer=None):
        """
        Initialize table detector.
        
        Args:
            config: Configuration object
            shared_analyzer: Optional pre-initialized PPStructureV3 instance to reuse
        """
        self.config = config
        self.layout_analyzer = shared_analyzer
        if shared_analyzer is None:
            self._initialize_layout_analyzer()
    
    def _initialize_layout_analyzer(self):
        """Initialize PaddleOCR PP-Structure layout analyzer."""
        try:
            from paddleocr import PPStructure
            
            logger.info("Initializing PP-Structure layout analyzer...")
            
            # Initialize PP-Structure for layout analysis with minimal parameters
            self.layout_analyzer = PPStructure()
            
            logger.info("PP-Structure layout analyzer initialized successfully")
            
        except ImportError:
            logger.error("PaddleOCR is not installed. Install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize layout analyzer: {e}")
            raise
    
    def detect_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect all table regions in image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of detected tables with bboxes and metadata
            Format: [
                {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': 0.95,
                    'type': 'table'
                },
                ...
            ]
        """
        try:
            img_array = image_to_numpy(image)
            
            logger.debug(f"Running layout analysis on image of size {image.size}")
            
            # Run layout analysis using __call__() method (PaddleOCR 2.7.3)
            result = self.layout_analyzer(img_array)
            
            if not result or not isinstance(result, list) or len(result) == 0:
                logger.warning("No layout elements detected")
                return []
            
            # PaddleOCR 2.7.3 PPStructure returns list of dicts with 'type', 'bbox', 'res', etc.
            tables = []
            for idx, region in enumerate(result):
                # region is a dict with keys: 'type', 'bbox', 'img', 'res', 'img_idx'
                if isinstance(region, dict) and region.get('type') == 'table':
                    bbox = region.get('bbox')
                    # bbox is [x1, y1, x2, y2] in image coordinates
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        normalized_bbox = normalize_bbox(
                            [int(x1), int(y1), int(x2), int(y2)],
                            image.width,
                            image.height
                        )
                        
                        # Get confidence score - PPStructure doesn't provide confidence for layout
                        confidence = 1.0
                        
                        if confidence >= self.config.table_conf_threshold:
                            # Get HTML content from 'res' if available (table recognition result)
                            html_content = region.get('res', {}).get('html', '') if isinstance(region.get('res'), dict) else ''
                            
                            tables.append({
                                'bbox': normalized_bbox,
                                'confidence': confidence,
                                'type': 'table',
                                'html': html_content,  # Store HTML content from PPStructure
                                'region_id': idx
                            })
                            logger.debug(f"Detected table at {normalized_bbox} with confidence {confidence:.2f}")
            
            # Sort tables from top to bottom
            tables.sort(key=lambda t: (t['bbox'][1], t['bbox'][0]))
            
            logger.info(f"Detected {len(tables)} table(s) in image")
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []
    
    def detect_all_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect all layout regions (not just tables).
        
        Args:
            image: PIL Image
            
        Returns:
            List of all detected regions with types
        """
        try:
            img_array = image_to_numpy(image)
            
            # Run layout analysis
            result = self.layout_analyzer(img_array)
            
            if not result:
                return []
            
            regions = []
            for region in result:
                bbox = region['bbox']
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    normalized_bbox = normalize_bbox(
                        [int(x1), int(y1), int(x2), int(y2)],
                        image.width,
                        image.height
                    )
                    
                    regions.append({
                        'bbox': normalized_bbox,
                        'type': region['type'],
                        'confidence': region.get('score', region.get('confidence', 1.0))
                    })
            
            return regions
            
        except Exception as e:
            logger.error(f"Region detection failed: {e}")
            return []
    
    def filter_overlapping_tables(self, tables: List[Dict[str, Any]], 
                                  iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Remove overlapping table detections using NMS.
        
        Args:
            tables: List of detected tables
            iou_threshold: IoU threshold for considering overlap
            
        Returns:
            Filtered list of tables
        """
        if len(tables) <= 1:
            return tables
        
        from .utils import calculate_iou
        
        # Sort by confidence
        sorted_tables = sorted(tables, key=lambda t: t['confidence'], reverse=True)
        
        kept_tables = []
        while sorted_tables:
            current = sorted_tables.pop(0)
            kept_tables.append(current)
            
            # Remove overlapping tables
            sorted_tables = [
                t for t in sorted_tables
                if calculate_iou(current['bbox'], t['bbox']) < iou_threshold
            ]
        
        # Re-sort by position
        kept_tables.sort(key=lambda t: (t['bbox'][1], t['bbox'][0]))
        
        return kept_tables
