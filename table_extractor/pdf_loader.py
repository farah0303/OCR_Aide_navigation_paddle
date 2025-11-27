"""
PDF to image conversion and image loading utilities.
"""

from typing import List, Union, Optional
from pathlib import Path
import logging
from PIL import Image
import os

from .utils import setup_logger
from .config import Config


logger = setup_logger(__name__)


class PDFLoader:
    """Handles PDF to image conversion and image loading."""
    
    def __init__(self, config: Config):
        """
        Initialize PDF loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.dpi = config.pdf_dpi
        self.format = config.pdf_format
    
    def load(self, input_path: Union[str, Path]) -> List[Image.Image]:
        """
        Load images from PDF or image file.
        
        Args:
            input_path: Path to PDF or image file
            
        Returns:
            List of PIL Images (one per page)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        suffix = input_path.suffix.lower()
        
        if suffix == '.pdf':
            return self._load_pdf(input_path)
        elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return self._load_image(input_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _load_pdf(self, pdf_path: Path) -> List[Image.Image]:
        """
        Convert PDF to images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of images, one per page
        """
        try:
            from pdf2image import convert_from_path
            
            logger.info(f"Converting PDF to images: {pdf_path}")
            logger.info(f"Using DPI: {self.dpi}")
            
            # Convert PDF to images
            images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                fmt=self.format.lower()
            )
            
            logger.info(f"Successfully loaded {len(images)} pages from PDF")
            return images
            
        except ImportError:
            logger.error("pdf2image is not installed. Install with: pip install pdf2image")
            logger.error("Also ensure poppler is installed on your system")
            raise
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            # Try to load with lower DPI as fallback
            if self.dpi > 150:
                logger.info("Attempting to load with lower DPI (150)...")
                try:
                    from pdf2image import convert_from_path
                    images = convert_from_path(
                        str(pdf_path),
                        dpi=150,
                        fmt=self.format.lower()
                    )
                    logger.info(f"Successfully loaded {len(images)} pages with lower DPI")
                    return images
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    raise e
            raise
    
    def _load_image(self, image_path: Path) -> List[Image.Image]:
        """
        Load single image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List containing single image
        """
        try:
            logger.info(f"Loading image: {image_path}")
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode not in ['RGB', 'L']:
                logger.info(f"Converting image from {image.mode} to RGB")
                image = image.convert('RGB')
            elif image.mode == 'L':
                logger.info("Converting grayscale to RGB")
                image = image.convert('RGB')
            
            logger.info(f"Successfully loaded image with size {image.size}")
            return [image]
            
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def save_images(self, images: List[Image.Image], output_dir: Union[str, Path], 
                   prefix: str = "page") -> List[Path]:
        """
        Save images to directory.
        
        Args:
            images: List of PIL Images
            output_dir: Output directory
            prefix: Filename prefix
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, image in enumerate(images, 1):
            output_path = output_dir / f"{prefix}_{i:04d}.png"
            image.save(output_path)
            saved_paths.append(output_path)
            logger.debug(f"Saved image: {output_path}")
        
        logger.info(f"Saved {len(saved_paths)} images to {output_dir}")
        return saved_paths
