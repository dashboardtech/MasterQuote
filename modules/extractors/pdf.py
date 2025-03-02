import pandas as pd
import logging
import tempfile
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image, ImageEnhance
from .base import BaseExtractor

logger = logging.getLogger(__name__)

class PDFExtractor(BaseExtractor):
    """Extractor para archivos PDF."""
    
    def extract(self, file_path: str, interactive: bool = False) -> pd.DataFrame:
        """
        Extrae datos de un archivo PDF.
        
        Args:
            file_path: Ruta al archivo PDF
            interactive: Si debe ser interactivo
            
        Returns:
            DataFrame con actividades y precios
        """
        try:
            # Importar tabula solo cuando sea necesario
            import tabula
            import PyPDF2
            
            # Verificar si el PDF está protegido
            if self._is_protected(file_path):
                logger.error(f"PDF protegido: {file_path}")
                raise ValueError(f"El PDF está protegido y no se puede procesar: {file_path}")
            
            # Intentar extraer tablas con tabula
            tables = tabula.read_pdf(
                file_path,
                pages='all',
                multiple_tables=True,
                guess=True,
                lattice=True,
                stream=True
            )
            
            if not tables:
                logger.warning("No se encontraron tablas, intentando OCR...")
                return self._extract_with_ocr(file_path)
            
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
                return pd.concat(processed_tables, ignore_index=True)
            else:
                logger.warning("No se encontraron tablas válidas, intentando OCR...")
                return self._extract_with_ocr(file_path)
                
        except Exception as e:
            logger.exception(f"Error al procesar archivo PDF {file_path}: {str(e)}")
            raise
    
    def _is_protected(self, file_path: str) -> bool:
        """
        Verifica si el PDF está protegido contra extracción.
        
        Args:
            file_path: Ruta al archivo PDF
            
        Returns:
            bool: True si está protegido
        """
        try:
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                return pdf.is_encrypted
        except:
            return False
    
    def _extract_with_ocr(self, file_path: str) -> pd.DataFrame:
        """
        Extrae datos usando OCR cuando la extracción directa falla.
        
        Args:
            file_path: Ruta al archivo PDF
            
        Returns:
            DataFrame con datos extraídos
        """
        try:
            import pdf2image
            import pytesseract
            from PIL import Image
            
            logger.info("Iniciando extracción con OCR...")
            
            # Convertir PDF a imágenes
            images = pdf2image.convert_from_path(file_path)
            
            # Procesar cada página
            extracted_text = []
            for idx, image in enumerate(images):
                try:
                    # Mejorar calidad de imagen para OCR
                    image = self._preprocess_image(image)
                    
                    # Extraer texto con OCR
                    text = pytesseract.image_to_string(image, lang='spa')
                    extracted_text.append(text)
                    
                except Exception as e:
                    logger.warning(f"Error en OCR de página {idx}: {str(e)}")
                    continue
            
            # Procesar texto extraído
            if extracted_text:
                return self._process_extracted_text('\n'.join(extracted_text))
            else:
                logger.error("No se pudo extraer texto con OCR")
                return pd.DataFrame()
                
        except Exception as e:
            logger.exception(f"Error en extracción OCR: {str(e)}")
            return pd.DataFrame()
    
    def _preprocess_image(self, image: Image) -> Image:
        """
        Preprocesa una imagen para mejorar resultados del OCR.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            Imagen procesada
        """
        try:
            # Convertir a escala de grises
            image = image.convert('L')
            
            # Aumentar contraste
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Aumentar nitidez
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            return image
        except:
            return image
    
    def _process_extracted_text(self, text: str) -> pd.DataFrame:
        """
        Procesa el texto extraído para encontrar información de precios.
        
        Args:
            text: Texto extraído del PDF
            
        Returns:
            DataFrame con datos procesados
        """
        try:
            # Dividir en líneas
            lines = text.split('\n')
            
            # Buscar patrones de precios y actividades
            data = []
            current_activity = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Buscar precio en la línea
                price_match = self._extract_price(line)
                
                if price_match:
                    if current_activity:
                        data.append({
                            'actividades': current_activity,
                            'costo_unitario': price_match
                        })
                        current_activity = None
                else:
                    # Si no hay precio, podría ser una actividad
                    current_activity = line
            
            if data:
                return pd.DataFrame(data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.exception(f"Error procesando texto extraído: {str(e)}")
            return pd.DataFrame()
    
    def _extract_price(self, text: str) -> Optional[float]:
        """
        Extrae un precio de una línea de texto.
        
        Args:
            text: Línea de texto
            
        Returns:
            float o None si no se encuentra precio
        """
        import re
        
        # Patrones comunes de precios
        patterns = [
            r'\$\s*[\d,]+\.?\d*',  # $123.45 o $1,234.56
            r'[\d,]+\.?\d*\s*(?:USD|MXN|EUR)',  # 123.45 USD
            r'(?:USD|MXN|EUR)\s*[\d,]+\.?\d*',  # USD 123.45
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                price_str = match.group()
                try:
                    # Limpiar y convertir a número
                    clean_price = self._clean_price_string(price_str)
                    return float(clean_price)
                except:
                    continue
        
        return None
    
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
