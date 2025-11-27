"""
HTML table parsing and conversion to structured formats.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import logging
import json
from io import StringIO
import re

from .utils import setup_logger


logger = setup_logger(__name__)


class HTMLParser:
    """Parses HTML tables and converts to various formats."""
    
    def __init__(self):
        """Initialize HTML parser."""
        pass
    
    def html_to_dataframe(self, html: str) -> Optional[pd.DataFrame]:
        """
        Convert HTML table to pandas DataFrame.
        
        Args:
            html: HTML table string
            
        Returns:
            DataFrame or None if parsing fails
        """
        try:
            if not html or not html.strip():
                logger.warning("Empty HTML provided")
                return None
            
            # Try pandas built-in parser first
            try:
                dfs = pd.read_html(StringIO(html), encoding='utf-8')
                if dfs:
                    df = dfs[0]
                    logger.debug(f"Parsed table with shape {df.shape}")
                    return df
            except Exception as e:
                logger.warning(f"pandas.read_html failed: {e}, trying fallback parser")
            
            # Fallback to custom parser
            df = self._fallback_html_parser(html)
            if df is not None:
                logger.debug(f"Fallback parser succeeded with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to parse HTML to DataFrame: {e}")
            return None
    
    def _fallback_html_parser(self, html: str) -> Optional[pd.DataFrame]:
        """
        Fallback HTML parser that handles rowspan/colspan manually.
        
        Args:
            html: HTML table string
            
        Returns:
            DataFrame or None
        """
        try:
            from html.parser import HTMLParser as BaseHTMLParser
            
            class TableHTMLParser(BaseHTMLParser):
                def __init__(self):
                    super().__init__()
                    self.tables = []
                    self.current_table = []
                    self.current_row = []
                    self.current_cell = []
                    self.in_table = False
                    self.in_row = False
                    self.in_cell = False
                    self.cell_attrs = {}
                
                def handle_starttag(self, tag, attrs):
                    attrs_dict = dict(attrs)
                    if tag == 'table':
                        self.in_table = True
                        self.current_table = []
                    elif tag == 'tr' and self.in_table:
                        self.in_row = True
                        self.current_row = []
                    elif tag in ('td', 'th') and self.in_row:
                        self.in_cell = True
                        self.current_cell = []
                        self.cell_attrs = {
                            'rowspan': int(attrs_dict.get('rowspan', 1)),
                            'colspan': int(attrs_dict.get('colspan', 1))
                        }
                
                def handle_endtag(self, tag):
                    if tag == 'table':
                        self.in_table = False
                        if self.current_table:
                            self.tables.append(self.current_table)
                    elif tag == 'tr' and self.in_row:
                        self.in_row = False
                        if self.current_row:
                            self.current_table.append(self.current_row)
                    elif tag in ('td', 'th') and self.in_cell:
                        self.in_cell = False
                        cell_text = ''.join(self.current_cell).strip()
                        self.current_row.append({
                            'text': cell_text,
                            'rowspan': self.cell_attrs.get('rowspan', 1),
                            'colspan': self.cell_attrs.get('colspan', 1)
                        })
                
                def handle_data(self, data):
                    if self.in_cell:
                        self.current_cell.append(data)
            
            parser = TableHTMLParser()
            parser.feed(html)
            
            if not parser.tables:
                logger.warning("No tables found in HTML")
                return None
            
            # Convert first table to DataFrame
            table_data = parser.tables[0]
            
            # Handle rowspan/colspan by expanding cells
            expanded_table = self._expand_merged_cells(table_data)
            
            if not expanded_table:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(expanded_table)
            
            return df
            
        except Exception as e:
            logger.error(f"Fallback parser failed: {e}")
            return None
    
    def _expand_merged_cells(self, table_data: List[List[Dict]]) -> List[List[str]]:
        """
        Expand cells with rowspan/colspan into full grid.
        
        Args:
            table_data: List of rows, each containing cell dictionaries
            
        Returns:
            2D list representing expanded table
        """
        if not table_data:
            return []
        
        # Find maximum dimensions
        max_cols = max(
            sum(cell.get('colspan', 1) for cell in row)
            for row in table_data
        )
        
        # Create grid
        grid = []
        
        for row_idx, row in enumerate(table_data):
            if row_idx >= len(grid):
                grid.append([None] * max_cols)
            
            col_idx = 0
            for cell in row:
                # Find next available column
                while col_idx < max_cols and grid[row_idx][col_idx] is not None:
                    col_idx += 1
                
                if col_idx >= max_cols:
                    break
                
                text = cell.get('text', '')
                rowspan = cell.get('rowspan', 1)
                colspan = cell.get('colspan', 1)
                
                # Fill grid with cell value
                for r in range(rowspan):
                    row_pos = row_idx + r
                    # Ensure row exists
                    while row_pos >= len(grid):
                        grid.append([None] * max_cols)
                    
                    for c in range(colspan):
                        col_pos = col_idx + c
                        if col_pos < max_cols:
                            grid[row_pos][col_pos] = text
                
                col_idx += colspan
        
        # Replace None with empty strings
        for row in grid:
            for i in range(len(row)):
                if row[i] is None:
                    row[i] = ''
        
        return grid
    
    def html_to_json(self, html: str) -> Dict[str, Any]:
        """
        Convert HTML table to JSON preserving structure.
        
        Args:
            html: HTML table string
            
        Returns:
            Dictionary with table structure
        """
        try:
            df = self.html_to_dataframe(html)
            
            if df is None:
                return {'rows': [], 'columns': [], 'data': []}
            
            return {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': df.columns.tolist(),
                'data': df.values.tolist(),
                'records': df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Failed to convert HTML to JSON: {e}")
            return {'rows': 0, 'columns': 0, 'data': []}
    
    def dataframe_to_html(self, df: pd.DataFrame) -> str:
        """
        Convert DataFrame to HTML table.
        
        Args:
            df: pandas DataFrame
            
        Returns:
            HTML string
        """
        try:
            return df.to_html(index=False, escape=False)
        except Exception as e:
            logger.error(f"Failed to convert DataFrame to HTML: {e}")
            return ""
    
    def save_json(self, data: Dict[str, Any], output_path: str):
        """
        Save JSON data to file.
        
        Args:
            data: Dictionary to save
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved JSON to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
    
    def save_csv(self, df: pd.DataFrame, output_path: str):
        """
        Save DataFrame to CSV.
        
        Args:
            df: pandas DataFrame
            output_path: Output file path
        """
        try:
            df.to_csv(output_path, index=False, encoding='utf-8')
            logger.debug(f"Saved CSV to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
