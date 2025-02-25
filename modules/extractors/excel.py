import pandas as pd
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from .base import BaseExtractor

logger = logging.getLogger(__name__)

class ExcelExtractor(BaseExtractor):
    """Extractor para archivos Excel."""
    
    def extract(self, file_path: str, interactive: bool = False) -> pd.DataFrame:
        """
        Extrae datos de un archivo Excel.
        
        Args:
            file_path: Ruta al archivo Excel
            interactive: Si debe ser interactivo
            
        Returns:
            DataFrame con actividades y precios
        """
        try:
            # Intentar leer todas las hojas
            xls = pd.ExcelFile(file_path)
            sheets = xls.sheet_names
            
            if len(sheets) == 1:
                # Si solo hay una hoja, usarla directamente
                df = pd.read_excel(file_path)
                return self._process_tabular_data(df, interactive)
            else:
                # Si hay múltiples hojas, intentar encontrar la correcta
                logger.info(f"Archivo con múltiples hojas: {sheets}")
                
                # Intentar cada hoja hasta encontrar una con datos válidos
                for sheet in sheets:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet)
                        if not df.empty and self._is_valid_data(df):
                            logger.info(f"Usando hoja: {sheet}")
                            return self._process_tabular_data(df, interactive)
                    except Exception as e:
                        logger.warning(f"Error al procesar hoja {sheet}: {str(e)}")
                        continue
                
                # Si llegamos aquí, no encontramos datos válidos
                logger.error(f"No se encontraron datos válidos en ninguna hoja")
                return pd.DataFrame()
                
        except Exception as e:
            logger.exception(f"Error al procesar archivo Excel {file_path}: {str(e)}")
            raise
            
    def _process_tabular_data(self, df: pd.DataFrame, interactive: bool) -> pd.DataFrame:
        """
        Procesa los datos tabulares para extraer información de precios.
        
        Args:
            df: DataFrame con datos crudos
            interactive: Si debe ser interactivo
            
        Returns:
            DataFrame procesado
        """
        try:
            # 1. Limpiar datos
            df = df.dropna(how='all')  # Eliminar filas completamente vacías
            
            # 2. Detectar columnas relevantes
            column_map = self._map_columns(df)
            
            # 3. Renombrar columnas
            df = df.rename(columns=column_map)
            
            # 4. Seleccionar solo columnas necesarias
            required_cols = ['actividades', 'costo_unitario', 'cantidad', 'costo_total']
            df = df.reindex(columns=required_cols)
            
            # 5. Normalizar columnas numéricas
            for col in ['costo_unitario', 'cantidad', 'costo_total']:
                if col in df.columns:
                    df[col] = self._normalize_price_column(df[col])
            
            return df
            
        except Exception as e:
            logger.exception(f"Error al procesar datos tabulares: {str(e)}")
            return pd.DataFrame()
    
    def _is_valid_data(self, df: pd.DataFrame) -> bool:
        """
        Verifica si el DataFrame contiene datos válidos de precios.
        
        Args:
            df: DataFrame a validar
            
        Returns:
            bool: True si contiene datos válidos
        """
        if df.empty:
            return False
            
        # Verificar si hay al menos una columna que parezca de precios
        price_patterns = ['precio', 'costo', 'valor', 'total', 'unitario', 'rate']
        has_price_col = any(
            any(pattern in col.lower() for pattern in price_patterns)
            for col in df.columns
        )
        
        # Verificar si hay al menos una columna que parezca de actividades
        activity_patterns = ['actividad', 'descripcion', 'item', 'concepto', 'material']
        has_activity_col = any(
            any(pattern in col.lower() for pattern in activity_patterns)
            for col in df.columns
        )
        
        return has_price_col and has_activity_col
    
    def _map_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Mapea las columnas del DataFrame a los nombres estandarizados.
        
        Args:
            df: DataFrame con columnas originales
            
        Returns:
            Dict con mapeo de nombres de columnas
        """
        column_map = {}
        
        # Patrones para cada tipo de columna
        patterns = {
            'actividades': ['actividad', 'descripcion', 'item', 'concepto', 'material'],
            'costo_unitario': ['precio unitario', 'costo unitario', 'valor unitario', 'precio', 'costo'],
            'cantidad': ['cantidad', 'qty', 'unidades', 'volume'],
            'costo_total': ['total', 'importe', 'valor total', 'subtotal']
        }
        
        # Buscar coincidencias para cada columna requerida
        for target_col, search_patterns in patterns.items():
            for col in df.columns:
                col_lower = str(col).lower()
                if any(pattern in col_lower for pattern in search_patterns):
                    column_map[col] = target_col
                    break
        
        return column_map
