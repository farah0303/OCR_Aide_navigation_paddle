"""
Table Extraction System using PaddleOCR PP-StructureV3

A production-grade system for extracting tables from multi-page scanned PDFs,
including rotated tables, tables in complex layouts, and tables embedded in
maps, blueprints, and diagrams.
"""

from .pipeline import TableExtractionPipeline, ExtractedTable

__version__ = "1.0.0"
__all__ = ["TableExtractionPipeline", "ExtractedTable"]
