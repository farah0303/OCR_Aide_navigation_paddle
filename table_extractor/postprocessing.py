"""
Postprocessing utilities for table extraction.
"""

from typing import Optional
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import logging

from .utils import setup_logger, image_to_numpy, numpy_to_image
from .config import Config


logger = setup_logger(__name__)


class TablePostprocessor:
    """Handles postprocessing of extracted tables and images."""
    
    def __init__(self, config: Config):
        """
        Initialize postprocessor.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean DataFrame by removing empty rows/columns and trimming whitespace.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            return df
        
        try:
            df_cleaned = df.copy()
            
            # Trim whitespace from all string columns
            if self.config.trim_whitespace:
                for col in df_cleaned.columns:
                    if df_cleaned[col].dtype == 'object':
                        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
            
            # Remove completely empty rows
            if self.config.clean_empty_rows:
                df_cleaned = df_cleaned.replace('', np.nan)
                df_cleaned = df_cleaned.dropna(how='all')
            
            # Remove completely empty columns
            if self.config.clean_empty_cols:
                df_cleaned = df_cleaned.dropna(axis=1, how='all')
            
            # Reset index
            df_cleaned = df_cleaned.reset_index(drop=True)
            
            logger.debug(f"Cleaned DataFrame from shape {df.shape} to {df_cleaned.shape}")
            return df_cleaned
            
        except Exception as e:
            logger.error(f"DataFrame cleaning failed: {e}")
            return df
    
    def validate_table(self, df: pd.DataFrame) -> bool:
        """
        Validate if extracted table meets quality criteria.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if valid, False otherwise
        """
        if df is None or df.empty:
            return False
        
        try:
            # Check minimum dimensions
            if df.shape[0] < self.config.min_table_rows:
                logger.warning(f"Table has {df.shape[0]} rows, minimum is {self.config.min_table_rows}")
                return False
            
            if df.shape[1] < self.config.min_table_cols:
                logger.warning(f"Table has {df.shape[1]} columns, minimum is {self.config.min_table_cols}")
                return False
            
            # Check empty cell ratio
            total_cells = df.shape[0] * df.shape[1]
            empty_cells = df.isna().sum().sum() + (df == '').sum().sum()
            empty_ratio = empty_cells / total_cells if total_cells > 0 else 1.0
            
            if empty_ratio > self.config.max_empty_cell_ratio:
                logger.warning(f"Table has {empty_ratio:.2%} empty cells, maximum is {self.config.max_empty_cell_ratio:.2%}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Table validation failed: {e}")
            return False

    def score_table(self, df: pd.DataFrame) -> float:
        """
        Compute a simple quality score for an extracted table.
        Higher is better; 0 means unusable, 1 means very confident.
        """
        if df is None or df.empty:
            return 0.0

        try:
            total_cells = df.shape[0] * df.shape[1]
            empty_cells = df.isna().sum().sum() + (df == '').sum().sum()
            empty_ratio = empty_cells / total_cells if total_cells > 0 else 1.0
            non_empty_ratio = max(0.0, 1.0 - empty_ratio)

            # Use average text length as a weak signal of OCR success (shorter often means missing text)
            lengths = []
            for col in df.columns:
                lengths.extend(df[col].astype(str).str.len().tolist())
            avg_len = float(np.nanmean(lengths)) if lengths else 0.0
            length_score = min(1.0, avg_len / 10.0)

            score = 0.7 * non_empty_ratio + 0.3 * length_score
            return float(max(0.0, min(1.0, score)))

        except Exception as e:
            logger.error(f"Table scoring failed: {e}")
            return 0.0
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing to improve OCR quality.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        if not self.config.enable_preprocessing:
            return image
        
        try:
            img_array = image_to_numpy(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply denoising
            if self.config.apply_denoise:
                gray = cv2.fastNlMeansDenoising(
                    gray,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
                logger.debug("Applied denoising")
            
            # Apply adaptive thresholding
            if self.config.apply_adaptive_threshold:
                gray = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2
                )
                logger.debug("Applied adaptive thresholding")
            
            # Convert back to RGB
            if len(gray.shape) == 2:
                processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            else:
                processed = gray
            
            return numpy_to_image(processed)
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def detect_and_correct_skew(self, image: Image.Image) -> Image.Image:
        """
        Detect and correct image skew/rotation.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Deskewed PIL Image
        """
        if not self.config.apply_deskew:
            return image
        
        try:
            img_array = image_to_numpy(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            
            if lines is None or len(lines) == 0:
                return image
            
            # Calculate average angle
            angles = []
            for line in lines[:50]:  # Use first 50 lines
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                if -45 < angle < 45:
                    angles.append(angle)
            
            if not angles:
                return image
            
            median_angle = np.median(angles)
            
            # Only correct if angle is significant
            if abs(median_angle) < 0.5:
                return image
            
            logger.info(f"Detected skew angle: {median_angle:.2f} degrees")
            
            # Rotate image
            (h, w) = img_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                img_array,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            return numpy_to_image(rotated)
            
        except Exception as e:
            logger.error(f"Skew correction failed: {e}")
            return image
    
    def enhance_table_region(self, image: Image.Image) -> Image.Image:
        """
        Apply enhancements specifically for table regions.
        
        Args:
            image: Table region image
            
        Returns:
            Enhanced image
        """
        try:
            img_array = image_to_numpy(image)
            
            # Increase contrast
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            logger.debug("Applied contrast enhancement")
            return numpy_to_image(enhanced)
            
        except Exception as e:
            logger.error(f"Table enhancement failed: {e}")
            return image
    
    def detect_rotation_angle(self, image: Image.Image) -> float:
        """
        Detect if image needs rotation (0, 90, 180, 270 degrees).
        
        Args:
            image: Input image
            
        Returns:
            Rotation angle in degrees
        """
        if not self.config.detect_rotation:
            return 0.0
        
        try:
            # This is a placeholder for rotation detection
            # In practice, you might use text orientation detection
            # or analyze the aspect ratio and content distribution
            return 0.0
            
        except Exception as e:
            logger.error(f"Rotation detection failed: {e}")
            return 0.0
    
    def normalize_cell_content(self, text: str) -> str:
        """
        Normalize text content from table cells.
        
        Args:
            text: Raw cell text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        try:
            # Remove multiple spaces
            normalized = ' '.join(text.split())
            
            # Remove special characters that might be OCR artifacts
            # Keep alphanumeric, punctuation, and common symbols
            # This is conservative to avoid removing legitimate content
            
            return normalized.strip()
            
        except Exception as e:
            logger.error(f"Text normalization failed: {e}")
            return text
