import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import BaseExtractor
from ..dockling_processor import DocklingProcessor

logger = logging.getLogger(__name__)

class DocklingExtractor(BaseExtractor):
    """
    Extractor que utiliza el servicio Dockling para procesar documentos.
    Este extractor es más avanzado y puede manejar cualquier tipo de documento,
    incluyendo formatos complejos y no estructurados.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el extractor de Dockling.
        
        Args:
            api_key: API key para el servicio Dockling (opcional)
        """
        self.processor = DocklingProcessor(api_key=api_key)
    
    def extract(self, file_path: str, interactive: bool = False) -> pd.DataFrame:
        """
        Extrae datos de un documento usando Dockling.
        
        Args:
            file_path: Ruta al archivo
            interactive: Si debe ser interactivo
            
        Returns:
            DataFrame con actividades y precios
        """
        try:
            # Primero intentar extraer datos de precios directamente
            try:
                df = self.processor.extract_price_data(file_path)
                if not df.empty:
                    logger.info("Datos extraídos exitosamente con Dockling price_data")
                    return df
            except Exception as e:
                logger.warning(f"Error en extracción directa de precios: {str(e)}")
            
            # Si no hay datos de precios, intentar con tablas
            try:
                tables = self.processor.extract_tables_from_document(file_path)
                if tables:
                    # Procesar cada tabla encontrada
                    processed_tables = []
                    for idx, table in enumerate(tables):
                        try:
                            if not table.empty and self._is_valid_data(table):
                                processed = self._process_tabular_data(table, interactive)
                                if not processed.empty:
                                    processed_tables.append(processed)
                        except Exception as e:
                            logger.warning(f"Error procesando tabla {idx}: {str(e)}")
                            continue
                    
                    if processed_tables:
                        # Combinar todas las tablas válidas
                        logger.info("Datos extraídos exitosamente con Dockling tables")
                        return pd.concat(processed_tables, ignore_index=True)
            except Exception as e:
                logger.warning(f"Error en extracción de tablas: {str(e)}")
            
            # Si no se encontraron tablas, intentar extraer texto
            try:
                text = self.processor.convert_document_to_text(file_path)
                if text:
                    # Analizar estructura del documento
                    structure = self.processor.analyze_document_structure(file_path)
                    
                    # Usar la estructura para identificar secciones relevantes
                    if structure.get('sections'):
                        relevant_sections = self._find_relevant_sections(structure['sections'])
                        if relevant_sections:
                            # Extraer y procesar datos de las secciones relevantes
                            data = []
                            for section in relevant_sections:
                                section_data = self._extract_price_from_section(section)
                                if section_data:
                                    data.append(section_data)
                            
                            if data:
                                logger.info("Datos extraídos exitosamente con Dockling text analysis")
                                return pd.DataFrame(data)
            except Exception as e:
                logger.warning(f"Error en extracción de texto: {str(e)}")
            
            logger.error(f"No se pudieron extraer datos del documento: {file_path}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.exception(f"Error en extracción con Dockling: {str(e)}")
            return pd.DataFrame()
    
    def _find_relevant_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifica secciones relevantes que pueden contener información de precios.
        
        Args:
            sections: Lista de secciones del documento
            
        Returns:
            Lista de secciones relevantes
        """
        relevant_sections = []
        
        # Palabras clave que indican secciones de precios
        keywords = [
            'precio', 'costo', 'presupuesto', 'cotización', 'total',
            'materiales', 'mano de obra', 'actividades', 'servicios'
        ]
        
        for section in sections:
            # Verificar título de sección
            title = section.get('title', '').lower()
            content = section.get('content', '').lower()
            
            # Buscar palabras clave en título y contenido
            if any(keyword in title for keyword in keywords) or \
               any(keyword in content[:200] for keyword in keywords):  # Solo revisar inicio del contenido
                relevant_sections.append(section)
            
            # Revisar subsecciones recursivamente
            if 'subsections' in section:
                subsections = self._find_relevant_sections(section['subsections'])
                relevant_sections.extend(subsections)
        
        return relevant_sections
    
    def _extract_price_from_section(self, section: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extrae información de precios de una sección de texto.
        
        Args:
            section: Diccionario con información de la sección
            
        Returns:
            Dict con datos extraídos o None si no se encuentran
        """
        import re
        
        content = section.get('content', '')
        if not content:
            return None
        
        # Patrones para encontrar precios
        price_patterns = [
            r'(?:precio|costo|valor)[\s:]*[$€]?\s*([\d,]+\.?\d*)',
            r'[$€]\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d*)\s*(?:USD|MXN|EUR)'
        ]
        
        # Patrones para encontrar cantidades
        quantity_patterns = [
            r'(?:cantidad|qty|unidades)[\s:]*(\d+)',
            r'(\d+)\s*(?:unidades|piezas|m2|m3)'
        ]
        
        # Buscar precio
        price = None
        for pattern in price_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    price = float(self._clean_price_string(match.group(1)))
                    break
                except:
                    continue
        
        if not price:
            return None
        
        # Buscar cantidad
        quantity = 1
        for pattern in quantity_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    quantity = float(match.group(1))
                    break
                except:
                    continue
        
        # Extraer descripción/actividad
        description = section.get('title', '')
        if not description:
            # Intentar extraer primera línea como descripción
            lines = content.split('\n')
            description = next((line.strip() for line in lines if line.strip()), '')
        
        return {
            'actividades': description,
            'cantidad': quantity,
            'costo_unitario': price,
            'costo_total': price * quantity
        }
    
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
