import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from .base import BaseExtractor

logger = logging.getLogger(__name__)

class CSVExtractor(BaseExtractor):
    """Extractor para archivos CSV."""
    
    def extract(self, file_path: str, interactive: bool = False) -> pd.DataFrame:
        """
        Extrae datos de un archivo CSV.
        
        Args:
            file_path: Ruta al archivo CSV
            interactive: Si debe ser interactivo
            
        Returns:
            DataFrame con actividades y precios
        """
        try:
            # Intentar diferentes codificaciones y delimitadores
            df, encoding, delimiter = self._detect_format(file_path)
            
            if df is not None:
                logger.info(f"CSV leído con encoding={encoding}, delimiter={delimiter}")
                return self._process_tabular_data(df, interactive)
            else:
                logger.error(f"No se pudo determinar el formato del CSV: {file_path}")
                raise ValueError(f"No se pudo determinar el formato del CSV: {file_path}")
            
        except Exception as e:
            logger.exception(f"Error al procesar archivo CSV {file_path}: {str(e)}")
            raise
    
    def _detect_format(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str, str]:
        """
        Detecta el formato correcto del CSV probando diferentes codificaciones y delimitadores.
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            Tuple con (DataFrame, encoding, delimiter) o (None, None, None) si no se detecta
        """
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        delimiters = [',', ';', '\t', '|']
        
        for encoding in encodings:
            # Primero intentar detectar el delimitador
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    sample = f.read(1024)  # Leer una muestra
                    
                # Contar ocurrencias de cada delimitador
                delimiter_counts = {d: sample.count(d) for d in delimiters}
                
                # Ordenar por frecuencia
                sorted_delimiters = sorted(
                    delimiters,
                    key=lambda d: delimiter_counts[d],
                    reverse=True
                )
                
                # Intentar con los delimitadores más probables primero
                for delimiter in sorted_delimiters:
                    if delimiter_counts[delimiter] > 0:
                        try:
                            df = pd.read_csv(
                                file_path,
                                encoding=encoding,
                                delimiter=delimiter,
                                error_bad_lines=False,
                                warn_bad_lines=True
                            )
                            
                            # Verificar si parece válido
                            if len(df.columns) > 1 and self._is_valid_data(df):
                                return df, encoding, delimiter
                                
                        except:
                            continue
                            
            except:
                continue
        
        return None, None, None
    
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
