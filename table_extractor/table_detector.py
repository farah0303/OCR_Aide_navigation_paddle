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
    
    def _detect_tables_on_image(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run PP-Structure detection on a single orientation of the image."""
        try:
            img_array = image_to_numpy(image)

            logger.debug(f"Running layout analysis on image of size {image.size}")

            # Run layout analysis using __call__() method (PaddleOCR 2.7.3)
            result = self.layout_analyzer(img_array)

            if not result or not isinstance(result, list) or len(result) == 0:
                logger.warning("No layout elements detected")
                return []

            tables = []
            for idx, region in enumerate(result):
                # region is a dict with keys: 'type', 'bbox', 'img', 'res', 'img_idx'
                if isinstance(region, dict) and region.get('type') == 'table':
                    bbox = region.get('bbox')
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        normalized_bbox = normalize_bbox(
                            [int(x1), int(y1), int(x2), int(y2)],
                            image.width,
                            image.height
                        )

                        confidence = 1.0  # PPStructure layout does not expose scores

                        if confidence >= self.config.table_conf_threshold:
                            html_content = region.get('res', {}).get('html', '') if isinstance(region.get('res'), dict) else ''

                            tables.append({
                                'bbox': normalized_bbox,
                                'confidence': confidence,
                                'type': 'table',
                                'html': html_content,
                                'region_id': idx
                            })
                            logger.debug(f"Detected table at {normalized_bbox} with confidence {confidence:.2f}")

            tables.sort(key=lambda t: (t['bbox'][1], t['bbox'][0]))
            logger.info(f"Detected {len(tables)} table(s) in image")
            return tables

        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []

    def _rotate_image(self, image: Image.Image, angle: int) -> Image.Image:
        """Rotate image by 0/90/180/270 degrees using transpose to preserve bounds."""
        if angle == 0:
            return image
        if angle == 90:
            return image.transpose(Image.ROTATE_90)
        if angle == 180:
            return image.transpose(Image.ROTATE_180)
        if angle == 270:
            return image.transpose(Image.ROTATE_270)
        raise ValueError(f"Unsupported angle: {angle}")

    def _map_bbox_to_original(self, bbox: Tuple[int, int, int, int], angle: int,
                               orig_w: int, orig_h: int) -> Tuple[int, int, int, int]:
        """Map a bbox from a rotated image back to original orientation."""
        x1, y1, x2, y2 = bbox
        points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        mapped = []

        for x, y in points:
            if angle == 0:
                nx, ny = x, y
            elif angle == 90:
                nx, ny = orig_w - 1 - y, x
            elif angle == 180:
                nx, ny = orig_w - 1 - x, orig_h - 1 - y
            elif angle == 270:
                nx, ny = y, orig_h - 1 - x
            else:
                raise ValueError(f"Unsupported angle: {angle}")
            mapped.append((nx, ny))

        xs = [p[0] for p in mapped]
        ys = [p[1] for p in mapped]
        return (
            int(max(0, min(xs))),
            int(max(0, min(ys))),
            int(min(orig_w, max(xs))),
            int(min(orig_h, max(ys)))
        )

    def detect_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect tables assuming upright orientation (backwards-compatible)."""
        return self._detect_tables_on_image(image)

    def detect_tables_multi_orientation(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect tables trying 0/90/180/270 rotations and map back to original."""
        orig_w, orig_h = image.width, image.height
        angles = [0, 90, 180, 270]
        best_tables: List[Dict[str, Any]] = []
        best_score = -1
        best_angle = 0

        for angle in angles:
            rotated_img = self._rotate_image(image, angle)
            detected = self._detect_tables_on_image(rotated_img)

            if not detected:
                continue

            # Map back to original coordinates
            mapped = []
            for tbl in detected:
                mapped_bbox = self._map_bbox_to_original(tbl['bbox'], angle, orig_w, orig_h)
                mapped.append({**tbl, 'bbox': mapped_bbox, 'rotation': angle})

            # Prefer orientations that produce more tables and richer HTML output
            total_html_chars = sum(len(tbl.get('html', '') or '') for tbl in detected)
            score = len(mapped) + 0.001 * total_html_chars

            if score > best_score:
                best_score = score
                best_tables = mapped
                best_angle = angle

        if best_tables:
            logger.info(f"Using orientation {best_angle}° with {len(best_tables)} detected table(s)")

        # Sort mapped tables top-to-bottom
        best_tables.sort(key=lambda t: (t['bbox'][1], t['bbox'][0]))
        return best_tables
    
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
