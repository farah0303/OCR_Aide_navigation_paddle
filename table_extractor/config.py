"""
Configuration settings for the table extraction pipeline.
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Config:
    """Central configuration for table extraction pipeline."""
    
    # PDF conversion settings
    pdf_dpi: int = 300
    pdf_format: str = "RGB"
    
    # PaddleOCR model settings
    ocr_lang: str = "en"
    ocr_use_angle_cls: bool = True
    ocr_det_limit_side_len: int = 1280
    ocr_rec_batch_num: int = 6
    
    # PP-StructureV3 settings
    structure_table_max_len: int = 1536
    structure_table_model_dir: Optional[str] = None
    structure_table_char_dict_path: Optional[str] = None
    
    # Detection thresholds
    table_conf_threshold: float = 0.5
    layout_conf_threshold: float = 0.5
    
    # Preprocessing settings
    enable_preprocessing: bool = True
    apply_denoise: bool = True
    apply_deskew: bool = True
    apply_adaptive_threshold: bool = False
    denoise_kernel_size: int = 3
    
    # Rotation detection
    detect_rotation: bool = True
    rotation_threshold: float = 0.5
    
    # Postprocessing settings
    clean_empty_rows: bool = True
    clean_empty_cols: bool = True
    trim_whitespace: bool = True
    min_cell_confidence: float = 0.3
    
    # Output settings
    save_html: bool = True
    save_csv: bool = True
    save_json: bool = True
    save_images: bool = True
    output_dir: str = "output"
    
    # Performance settings
    # Note: These settings are kept for backwards compatibility but are not used in PaddleOCR 3.x
    # GPU usage in PaddleOCR 3.x is determined by installing paddlepaddle-gpu vs paddlepaddle
    use_gpu: bool = False  # Deprecated in PaddleOCR 3.x
    enable_mkldnn: bool = True  # Deprecated in PaddleOCR 3.x
    cpu_threads: int = 10  # Deprecated in PaddleOCR 3.x
    
    # Validation settings
    min_table_rows: int = 1
    min_table_cols: int = 1
    max_empty_cell_ratio: float = 0.9
    min_table_score: float = 0.4
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.pdf_dpi < 72:
            raise ValueError("pdf_dpi must be at least 72")
        if not 0 <= self.table_conf_threshold <= 1:
            raise ValueError("table_conf_threshold must be between 0 and 1")
        if not 0 <= self.layout_conf_threshold <= 1:
            raise ValueError("layout_conf_threshold must be between 0 and 1")


# Default configuration instance
DEFAULT_CONFIG = Config()
