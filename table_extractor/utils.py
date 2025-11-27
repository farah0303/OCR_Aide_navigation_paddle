"""
Utility functions for table extraction pipeline.
"""

import os
import logging
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def ensure_dir(path: str) -> str:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
        
    Returns:
        Absolute path to directory
    """
    os.makedirs(path, exist_ok=True)
    return os.path.abspath(path)


def normalize_bbox(bbox: List[int], img_width: int, img_height: int) -> Tuple[int, int, int, int]:
    """
    Normalize bounding box coordinates to ensure they're within image bounds.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2] or [x1, y1, w, h]
        img_width: Image width
        img_height: Image height
        
    Returns:
        Normalized bbox as (x1, y1, x2, y2)
    """
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        # Check if bbox is in [x, y, w, h] format
        if x2 < x1 or y2 < y1:
            x2 = x1 + x2
            y2 = y1 + y2
    else:
        raise ValueError(f"Invalid bbox format: {bbox}")
    
    # Clamp to image bounds
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    return (int(x1), int(y1), int(x2), int(y2))


def calculate_iou(bbox1: Tuple[int, int, int, int], 
                  bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bbox as (x1, y1, x2, y2)
        bbox2: Second bbox as (x1, y1, x2, y2)
        
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def draw_bboxes(image: Image.Image, 
                bboxes: List[Tuple[int, int, int, int]], 
                labels: Optional[List[str]] = None,
                color: str = "red",
                width: int = 3) -> Image.Image:
    """
    Draw bounding boxes on image.
    
    Args:
        image: PIL Image
        bboxes: List of bounding boxes as (x1, y1, x2, y2)
        labels: Optional labels for each bbox
        color: Box color
        width: Line width
        
    Returns:
        Image with drawn bboxes
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        if labels and i < len(labels):
            label = labels[i]
            # Draw label background
            text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 25), label, fill="white", font=font)
    
    return img_copy


def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """
    Crop image using bounding box.
    
    Args:
        image: PIL Image
        bbox: Bounding box as (x1, y1, x2, y2)
        
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


def image_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
        
    Returns:
        Numpy array in RGB format
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def numpy_to_image(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array
        
    Returns:
        PIL Image
    """
    return Image.fromarray(array.astype(np.uint8))


def sort_bboxes_top_to_bottom(bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """
    Sort bounding boxes from top to bottom, left to right.
    
    Args:
        bboxes: List of bounding boxes as (x1, y1, x2, y2)
        
    Returns:
        Sorted list of bboxes
    """
    return sorted(bboxes, key=lambda b: (b[1], b[0]))


def validate_bbox(bbox: Tuple[int, int, int, int], 
                  min_width: int = 10, 
                  min_height: int = 10) -> bool:
    """
    Validate if bbox meets minimum size requirements.
    
    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        min_width: Minimum width
        min_height: Minimum height
        
    Returns:
        True if valid, False otherwise
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return width >= min_width and height >= min_height
