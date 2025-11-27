"""
OCR engine wrapper for PaddleOCR.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import logging

from .utils import setup_logger, image_to_numpy
from .config import Config


logger = setup_logger(__name__)


class OCREngine:
    """Wrapper for PaddleOCR text recognition."""
    
    def __init__(self, config: Config):
        """
        Initialize OCR engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.ocr = None
        self._initialize_ocr()
    
    def _initialize_ocr(self):
        """Initialize PaddleOCR instance."""
        try:
            from paddleocr import PaddleOCR
            
            logger.info("Initializing PaddleOCR engine...")
            
            # Initialize with configuration
            # Note: PaddleOCR 3.x API changes:
            # - Removed: use_gpu, enable_mkldnn, cpu_threads, use_angle_cls, det_limit_side_len, rec_batch_num, show_log
            # - GPU usage is determined by installing paddlepaddle-gpu vs paddlepaddle
            # - lang parameter is valid but uses different parameter name: 'lang' becomes part of model selection
            # - Using minimal initialization with no parameters for compatibility
            self.ocr = PaddleOCR(
                lang=self.config.ocr_lang
            )
            
            logger.info("PaddleOCR engine initialized successfully")
            
        except ImportError:
            logger.error("PaddleOCR is not installed. Install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    def recognize_text(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Perform OCR on image.
        
        Args:
            image: PIL Image
            
        Returns:
            List of detected text regions with bboxes, text, and confidence
            Format: [
                {
                    'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                    'text': 'recognized text',
                    'confidence': 0.95
                },
                ...
            ]
        """
        try:
            # Convert to numpy array
            img_array = image_to_numpy(image)
            
            # Run OCR
            result = self.ocr.ocr(img_array, cls=self.config.ocr_use_angle_cls)
            
            if result is None or len(result) == 0:
                logger.warning("No text detected in image")
                return []
            
            # Parse results
            text_regions = []
            for line in result[0] if result[0] else []:
                if line is None:
                    continue
                    
                bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (text, confidence)
                
                text_regions.append({
                    'bbox': bbox,
                    'text': text_info[0],
                    'confidence': text_info[1]
                })
            
            logger.debug(f"Detected {len(text_regions)} text regions")
            return text_regions
            
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return []
    
    def recognize_cell(self, cell_image: Image.Image) -> Tuple[str, float]:
        """
        Recognize text in a single table cell.
        
        Args:
            cell_image: PIL Image of table cell
            
        Returns:
            Tuple of (text, confidence)
        """
        try:
            if cell_image.width < 5 or cell_image.height < 5:
                return "", 0.0
            
            text_regions = self.recognize_text(cell_image)
            
            if not text_regions:
                return "", 0.0
            
            # Combine all text regions in cell
            texts = [r['text'] for r in text_regions]
            confidences = [r['confidence'] for r in text_regions]
            
            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return combined_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Cell OCR failed: {e}")
            return "", 0.0
    
    def detect_rotation(self, image: Image.Image) -> float:
        """
        Detect image rotation angle.
        
        Args:
            image: PIL Image
            
        Returns:
            Rotation angle in degrees (0, 90, 180, 270)
        """
        try:
            # Use angle classifier if available
            if not self.config.ocr_use_angle_cls:
                return 0.0
            
            img_array = image_to_numpy(image)
            
            # Run OCR with angle detection
            result = self.ocr.ocr(img_array, cls=True)
            
            # Extract rotation info from first detection
            if result and len(result) > 0 and result[0] and len(result[0]) > 0:
                # PaddleOCR returns angle info, but we'll use a simple heuristic
                # based on the orientation of detected text
                return 0.0  # Placeholder - would need actual angle extraction
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Rotation detection failed: {e}")
            return 0.0
    
    def correct_rotation(self, image: Image.Image) -> Image.Image:
        """
        Automatically detect and correct image rotation.
        
        Args:
            image: PIL Image
            
        Returns:
            Rotated image
        """
        try:
            if not self.config.detect_rotation:
                return image
            
            angle = self.detect_rotation(image)
            
            if abs(angle) < 1.0:
                return image
            
            logger.info(f"Rotating image by {angle} degrees")
            return image.rotate(angle, expand=True)
            
        except Exception as e:
            logger.error(f"Rotation correction failed: {e}")
            return image
