import pandas as pd
import numpy as np
import os
import re
import logging
import tempfile
import time
from pathlib import Path
import unidecode
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniversalPriceExtractor:
    """
    Extractor universal de precios desde múltiples formatos de archivo.
    Soporta Excel, CSV, PDF, Word, imágenes (con OCR) y utiliza IA para casos complejos.
    """
    
    def __init__(self, api_key=None, dockling_api_key=None, use_cache=True, cache_expiry_days=30):
        """
        Inicializa el extractor.
        
        Args:
            api_key: API key para OpenAI (opcional)
            dockling_api_key: API key para Dockling (opcional)
            use_cache: Si se debe usar caché para archivos procesados
            cache_expiry_days: Días tras los cuales la caché se considera obsoleta
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.dockling_api_key = dockling_api_key or os.environ.get("DOCKLING_API_KEY")
        
        if not self.dockling_api_key:
            logger.warning("No se ha configurado API key para Dockling. Algunas funciones estarán limitadas.")
        else:
            logger.info("API key de Dockling configurada correctamente")
        self.precio_keywords = [
            'precio', 'price', 'valor', 'value', 'costo', 'cost',
            'unitario', 'unit', 'total', 'monto', 'amount', 'tarifa',
            'rate', 'pago', 'payment', 'p.u.', 'p.unit', 'p/u', 'valor_unitario',
            'precio_unitario', 'precio_unit', 'precio unit', 'pu', 'punit', 'p unit',
            'importe', 'precio por', 'coste', '$', 'valor$', 'p.u', 'p_u',
            'precio/u', 'precio/unidad', 'precio_u', 'precio u', 'val unit',
            'valor unit', 'val/u', 'val_u', 'costo unit', 'costo/u', 'costo_u'
        ]
        self.actividad_keywords = [
            'actividad', 'activity', 'item', 'concepto', 'concept',
            'descripcion', 'description', 'servicio', 'service',
            'producto', 'product', 'trabajo', 'work', 'tarea', 'task',
            'detalle', 'detail', 'articulo', 'article', 'material',
            'insumo', 'input', 'elemento', 'element', 'partida', 'item',
            'desc', 'nombre', 'name', 'denominacion', 'designation',
            'especificacion', 'specification', 'caracteristica', 'characteristic'
        ]
        
        # Inicializar el gestor de caché si está habilitado
        self.use_cache = use_cache
        if use_cache:
            from modules.cache_manager import CacheManager
            # Convertir días a segundos para el parámetro expiration_time
            expiration_time = cache_expiry_days * 24 * 60 * 60
            self.cache_manager = CacheManager(expiration_time=expiration_time)
            logger.info("Sistema de caché inicializado")
            
        # Inicializar el gestor de extractores para validación cruzada
        from modules.extractor_manager import ExtractorManager
        self.extractor_manager = ExtractorManager(
            api_key=self.api_key,
            dockling_api_key=self.dockling_api_key,
            num_extractors=3,
            use_parallel=True
        )
        logger.info("Gestor de extractores inicializado para validación cruzada")
            
        # Modo debug para diagnóstico detallado
        self.debug_mode = False
        
    def set_debug_mode(self, enabled: bool = True):
        """Activa o desactiva el modo de depuración."""
        self.debug_mode = enabled
        if enabled:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
            
    def debug_excel_extraction(self, file_path: str) -> bool:
        """Análisis detallado del archivo Excel para debugging."""
        try:
            logger.info(f"=== Iniciando análisis detallado de: {file_path} ===")
            xls = pd.ExcelFile(file_path)
            sheets = xls.sheet_names
            logger.info(f"Hojas detectadas: {sheets}")
            
            for sheet in sheets:
                logger.info(f"\n=== Analizando hoja: {sheet} ===")
                df = pd.read_excel(file_path, sheet_name=sheet)
                logger.info(f"Dimensiones: {df.shape}")
                logger.info(f"Columnas: {df.columns.tolist()}")
                logger.info(f"Tipos de datos:\n{df.dtypes}")
                
                # Análisis de columnas
                for col in df.columns:
                    col_lower = str(col).lower()
                    precio_match = any(keyword in col_lower for keyword in self.precio_keywords)
                    actividad_match = any(keyword in col_lower for keyword in self.actividad_keywords)
                    
                    logger.info(f"\nColumna: {col}")
                    logger.info(f"  - ¿Posible columna de precio? {precio_match}")
                    logger.info(f"  - ¿Posible columna de actividad? {actividad_match}")
                    logger.info(f"  - Primeros valores: {df[col].head(3).tolist()}")
                    
                    # Análisis de contenido numérico
                    if df[col].dtype in [np.float64, np.int64]:
                        logger.info(f"  - Estadísticas numéricas:")
                        logger.info(f"    * Media: {df[col].mean()}")
                        logger.info(f"    * Min: {df[col].min()}")
                        logger.info(f"    * Max: {df[col].max()}")
                    
                    # Análisis de texto
                    if df[col].dtype == object:
                        sample = df[col].dropna().head(3).tolist()
                        logger.info(f"  - Ejemplos de texto: {sample}")
                        
                # Buscar patrones de precio en todas las columnas
                logger.info("\nBúsqueda de patrones de precio en todas las columnas:")
                for col in df.columns:
                    if df[col].dtype == object:
                        precio_pattern = r'\$?\s*\d+([.,]\d{2})?|\d+([.,]\d+)?\s*[$€£¥]'
                        matches = df[col].astype(str).str.extract(precio_pattern, expand=False)
                        if matches.notna().any():
                            logger.info(f"  - Columna {col} contiene posibles precios: {matches.dropna().head(3).tolist()}")
            
            return True
        except Exception as e:
            logger.error(f"Error en debug de Excel: {str(e)}")
            return False
        
    def extract_from_file(self, file_path, interactive=False, column_mapping=None, use_validation=False, min_confidence=0.5):
        """
        Extrae precios de un archivo en cualquier formato soportado.
        
        Args:
            file_path: Ruta al archivo
            interactive: Si debe solicitar ayuda al usuario para mapeo de columnas
            column_mapping: Diccionario con mapeo manual de columnas {'actividades': 'col1', 'precios': 'col2'}
            use_validation: Si debe usar validación cruzada con múltiples extractores
            min_confidence: Confianza mínima requerida para aceptar resultados de validación
            
        Returns:
            DataFrame con actividades y precios normalizados
        """
        start_time = time.time()
        
        # Si file_path es un objeto UploadedFile de Streamlit, guardarlo temporalmente
        if hasattr(file_path, 'getvalue'):
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_path.name)[1]) as tmp:
                tmp.write(file_path.getbuffer())
                temp_file_path = tmp.name
            
            file_path = temp_file_path
        
        # Verificar si los resultados están en caché
        if self.use_cache and hasattr(self, 'cache_manager'):
            cached_df = self.cache_manager.load_from_cache(file_path)
            if cached_df is not None:
                processing_time = time.time() - start_time
                logger.info(f"Resultados cargados desde caché en {processing_time:.2f} segundos")
                
                # Limpiar archivo temporal si existe
                if 'temp_file_path' in locals():
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                
                return cached_df
        
        # Si se solicita validación cruzada, usar el gestor de extractores
        if use_validation:
            try:
                logger.info(f"Usando validación cruzada para procesar {file_path}")
                df, metadata = self.extractor_manager.extract_with_validation(file_path)
                
                # Verificar confianza mínima
                confidence = metadata.get('confidence', 0.0)
                if confidence < min_confidence:
                    logger.warning(f"Confianza insuficiente: {confidence:.2f} < {min_confidence}")
                    
                    # Si la confianza es baja, intentar con el método tradicional
                    logger.info("Intentando con método tradicional debido a baja confianza")
                    traditional_df = self._extract_traditional(file_path, interactive, column_mapping)
                    
                    # Comparar resultados y elegir el mejor
                    if traditional_df is not None and not traditional_df.empty:
                        if df is None or df.empty or len(traditional_df) > len(df):
                            logger.info("Usando resultados del método tradicional")
                            df = traditional_df
                
                processing_time = time.time() - start_time
                logger.info(f"Archivo procesado con validación cruzada en {processing_time:.2f} segundos")
                
                # Guardar resultados en caché
                if self.use_cache and hasattr(self, 'cache_manager') and df is not None and not df.empty:
                    self.cache_manager.save_to_cache(file_path, df)
                
                # Limpiar archivo temporal si existe
                if 'temp_file_path' in locals():
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
                
                return df
                
            except Exception as e:
                logger.exception(f"Error en validación cruzada: {str(e)}")
                logger.info("Fallback a método tradicional")
                # Continuar con el método tradicional
        
        # Método tradicional de extracción
        result_df = self._extract_traditional(file_path, interactive, column_mapping)
        
        # Limpiar archivo temporal si existe
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        return result_df
    
    def _extract_traditional(self, file_path, interactive=False, column_mapping=None):
        """
        Método tradicional de extracción basado en el formato del archivo.
        
        Args:
            file_path: Ruta al archivo
            interactive: Si debe solicitar ayuda al usuario para mapeo de columnas
            column_mapping: Diccionario con mapeo manual de columnas
            
        Returns:
            DataFrame con actividades y precios normalizados
        """
        file_ext = Path(file_path).suffix.lower()
        logger.info(f"Procesando archivo {file_path} con formato {file_ext}")
        
        # Detectar formato y llamar al extractor correspondiente
        try:
            if file_ext in ['.xlsx', '.xls', '.xlsm']:
                result_df = self._extract_from_excel(file_path, interactive, column_mapping)
            elif file_ext == '.csv':
                result_df = self._extract_from_csv(file_path, interactive)
            elif file_ext == '.pdf':
                result_df = self._extract_from_pdf(file_path, interactive)
            elif file_ext in ['.docx', '.doc']:
                result_df = self._extract_from_word(file_path, interactive)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                result_df = self._extract_from_image(file_path, interactive)
            elif file_ext == '.txt':
                result_df = self._extract_from_text(file_path, interactive)
            else:
                logger.error(f"Formato de archivo no soportado: {file_ext}")
                raise ValueError(f"Formato de archivo no soportado: {file_ext}")
            
            # Guardar resultados en caché
            if self.use_cache and hasattr(self, 'cache_manager') and result_df is not None and not result_df.empty:
                self.cache_manager.save_to_cache(file_path, result_df)
            
            return result_df
            
        except Exception as e:
            logger.exception(f"Error al procesar archivo {file_path}: {str(e)}")
            if self.api_key:
                # Intentar con asistencia de IA
                result_df = self._extract_with_ai_assistance(file_path)
                
                # Guardar resultados de IA en caché
                if self.use_cache and hasattr(self, 'cache_manager') and result_df is not None and not result_df.empty:
                    self.cache_manager.save_to_cache(file_path, result_df)
                    
                return result_df
            raise
    
    def _extract_from_excel(self, file_path, interactive=False, column_mapping=None):
        """Extrae datos de archivos Excel."""
        try:
            # Intentar leer todas las hojas
            xls = pd.ExcelFile(file_path)
            sheets = xls.sheet_names
            
            if self.debug_mode:
                logger.debug(f"Hojas encontradas en {file_path}: {sheets}")
            
            if len(sheets) == 1:
                # Si solo hay una hoja, usarla directamente
                if self.debug_mode:
                    logger.debug(f"Usando única hoja: {sheets[0]}")
                df = pd.read_excel(file_path)
                return self._process_tabular_data(df, file_path, interactive, sheet_name=sheets[0], column_mapping=column_mapping)
            else:
                # Si hay múltiples hojas, buscar la que contiene precios
                for sheet in sheets:
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    
                    # Verificar si esta hoja parece contener precios
                    if self._sheet_contains_prices(df):
                        if self.debug_mode:
                            logger.debug(f"Hoja con precios encontrada: {sheet}")
                        return self._process_tabular_data(df, file_path, interactive, sheet_name=sheet, column_mapping=column_mapping)
                
                # Si ninguna hoja es obviamente la correcta y estamos en modo interactivo
                if interactive and len(sheets) > 0:
                    print(f"Múltiples hojas detectadas en {file_path}:")
                    for i, sheet in enumerate(sheets):
                        print(f"{i+1}. {sheet}")
                    
                    sheet_idx = input("Ingrese el número de la hoja a usar: ")
                    try:
                        sheet_idx = int(sheet_idx) - 1
                        if 0 <= sheet_idx < len(sheets):
                            df = pd.read_excel(file_path, sheet_name=sheets[sheet_idx])
                            return self._process_tabular_data(df, file_path, interactive, sheet_name=sheets[sheet_idx], column_mapping=column_mapping)
                    except:
                        pass
                
                # Si todo falla, usar la primera hoja
                if self.debug_mode:
                    logger.debug(f"No se encontró hoja con precios, usando primera hoja: {sheets[0]}")
                df = pd.read_excel(file_path, sheet_name=0)
                return self._process_tabular_data(df, file_path, interactive, sheet_name=sheets[0], column_mapping=column_mapping)
                
        except Exception as e:
            logger.exception(f"Error al procesar archivo Excel {file_path}: {str(e)}")
            if self.api_key:
                # Intentar con asistencia de IA
                return self._extract_with_ai_assistance(file_path)
            raise
    
    def _extract_from_csv(self, file_path, interactive=False):
        """Extrae datos de archivos CSV."""
        try:
            # Intentar diferentes codificaciones y delimitadores
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            delimiters = [',', ';', '\t', '|']
            
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                        if len(df.columns) > 1:  # Verificar que tenga múltiples columnas
                            logger.info(f"CSV leído con encoding={encoding}, delimiter={delimiter}")
                            return self._process_tabular_data(df, file_path, interactive)
                    except:
                        continue
            
            logger.error(f"No se pudo determinar el formato correcto del CSV: {file_path}")
            raise ValueError(f"No se pudo determinar el formato del CSV: {file_path}")
            
        except Exception as e:
            logger.exception(f"Error al procesar archivo CSV {file_path}: {str(e)}")
            if self.api_key:
                return self._extract_with_ai_assistance(file_path)
            raise
    
    def _extract_from_pdf(self, file_path, interactive=False):
        """Extrae datos de archivos PDF."""
        try:
            # Importar tabula solo cuando sea necesario
            import tabula
            
            # Intentar extraer tablas con tabula
            tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            
            if tables and len(tables) > 0:
                # Buscar la tabla que parece contener precios
                for i, table in enumerate(tables):
                    if not table.empty and self._table_contains_prices(table):
                        logger.info(f"Usando tabla {i+1} que contiene datos de precios")
                        return self._process_tabular_data(table, file_path, interactive)
                
                # Si ninguna tabla es obviamente la correcta y estamos en modo interactivo
                if interactive and len(tables) > 0:
                    print(f"Múltiples tablas detectadas en {file_path}:")
                    for i, table in enumerate(tables):
                        print(f"{i+1}. Tabla con {len(table.columns)} columnas y {len(table)} filas")
                    
                    table_idx = input("Ingrese el número de la tabla a usar: ")
                    try:
                        table_idx = int(table_idx) - 1
                        if 0 <= table_idx < len(tables):
                            return self._process_tabular_data(tables[table_idx], file_path, interactive)
                    except:
                        pass
                
                # Si todo falla, concatenar todas las tablas
                logger.warning(f"No se pudo determinar automáticamente la tabla correcta. Combinando todas.")
                combined_df = pd.concat(tables, ignore_index=True)
                return self._process_tabular_data(combined_df, file_path, interactive)
            else:
                # Si no se encontraron tablas, usar OCR
                logger.info(f"No se encontraron tablas en el PDF. Intentando con OCR.")
                return self._extract_with_ai_assistance(file_path)
                
        except Exception as e:
            logger.exception(f"Error al procesar archivo PDF {file_path}: {str(e)}")
            if self.api_key:
                return self._extract_with_ai_assistance(file_path)
            raise
    
    def _extract_from_word(self, file_path, interactive=False):
        """Extrae datos de archivos Word."""
        try:
            # Importar docx2txt solo cuando sea necesario
            import docx2txt
            
            # Extraer el texto
            text = docx2txt.process(file_path)
            
            # Intentar encontrar tablas o estructuras de precio
            # Primero ver si podemos encontrar una tabla delimitada por tabulaciones
            lines = text.split('\n')
            tabular_lines = []
            
            for line in lines:
                if '\t' in line:
                    tabular_lines.append(line)
            
            if tabular_lines:
                # Construir un CSV temporal con los datos tabulares
                with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp:
                    temp.write('\n'.join(tabular_lines))
                    temp_path = temp.name
                
                try:
                    # Intentar leer como CSV
                    result = self._extract_from_csv(temp_path, interactive)
                    os.unlink(temp_path)
                    return result
                except:
                    os.unlink(temp_path)
            
            # Si no funciona, usar asistencia IA
            return self._extract_with_ai_assistance(file_path, text=text)
                
        except Exception as e:
            logger.exception(f"Error al procesar archivo Word {file_path}: {str(e)}")
            if self.api_key:
                return self._extract_with_ai_assistance(file_path)
            raise
    
    def _extract_from_image(self, file_path, interactive=False):
        """Extrae datos de imágenes usando OCR."""
        try:
            # Importar dependencias solo cuando sea necesario
            from PIL import Image
            import pytesseract
            
            # Usar Tesseract OCR para extraer texto
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang='spa+eng')
            
            # Intentar estructurar los datos con IA
            return self._extract_with_ai_assistance(file_path, text=text)
                
        except Exception as e:
            logger.exception(f"Error al procesar imagen {file_path}: {str(e)}")
            if self.api_key:
                return self._extract_with_ai_assistance(file_path)
            raise
    
    def _extract_from_text(self, file_path, interactive=False):
        """Extrae datos de archivos de texto plano."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Intentar estructurar los datos con IA
            return self._extract_with_ai_assistance(file_path, text=text)
                
        except Exception as e:
            logger.exception(f"Error al procesar archivo de texto {file_path}: {str(e)}")
            if self.api_key:
                return self._extract_with_ai_assistance(file_path)
            raise
    
    def _extract_with_ai_assistance(self, file_path, text=None):
        """
        Usa un LLM para extraer información de formatos difíciles.
        
        Args:
            file_path: Ruta al archivo
            text: Texto ya extraído (opcional)
            
        Returns:
            DataFrame con actividades y precios
        """
        if not self.api_key:
            raise ValueError("Se requiere API key para asistencia con IA")
        
        try:
            # Si no tenemos texto, intentar extraerlo
            if text is None:
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext in ['.pdf']:
                    # Usar pdfplumber para PDF
                    import pdfplumber
                    with pdfplumber.open(file_path) as pdf:
                        text = "\n".join([page.extract_text() for page in pdf.pages])
                elif file_ext in ['.docx', '.doc']:
                    import docx2txt
                    text = docx2txt.process(file_path)
                elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                    from PIL import Image
                    import pytesseract
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img, lang='spa+eng')
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
            
            # Si el texto es muy largo, tomar solo las partes relevantes
            if len(text) > 10000:
                # Buscar secciones con palabras clave
                chunks = text.split('\n\n')
                filtered_chunks = []
                
                for chunk in chunks:
                    lower_chunk = chunk.lower()
                    if any(keyword in lower_chunk for keyword in self.precio_keywords + self.actividad_keywords):
                        filtered_chunks.append(chunk)
                
                if filtered_chunks:
                    text = "\n\n".join(filtered_chunks)
                else:
                    # Si no hay chunks relevantes, tomar los primeros 10000 caracteres
                    text = text[:10000]
            
            # Construir el prompt para el LLM
            prompt = f"""
            Necesito extraer información de actividades y precios de este documento. Extrae la siguiente información en formato JSON:
            
            1. Una lista de actividades con sus precios unitarios
            2. La moneda utilizada
            3. Cualquier información relevante sobre impuestos o descuentos
            
            El documento es el siguiente:
            
            ---
            {text}
            ---
            
            Por favor, estructura tu respuesta como un JSON válido con este formato:
            
            ```json
            {{
              "actividades": [
                {{"nombre": "Actividad 1", "precio_unitario": 1000.0}},
                {{"nombre": "Actividad 2", "precio_unitario": 2000.0}},
                ...
              ],
              "moneda": "CLP",
              "notas": "Información adicional sobre impuestos o condiciones de precio"
            }}
            ```
            
            Responde ÚNICAMENTE con JSON válido, sin texto adicional.
            """
            
            # Llamar a la API del LLM
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente especializado en extraer información de documentos financieros."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            # Extraer la respuesta
            result_text = response.choices[0].message.content
            
            # Intentar encontrar el JSON en la respuesta
            json_match = re.search(r'```json\s*(.*?)\s*```', result_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Si no hay delimitadores de código, intentar extraer todo el JSON
                json_str = result_text.strip()
            
            # Limpiar el string para asegurar que es JSON válido
            json_str = re.sub(r'^[^{]*', '', json_str)
            json_str = re.sub(r'[^}]*$', '', json_str)
            
            data = json.loads(json_str)
            
            # Convertir a DataFrame
            df = pd.DataFrame(data['actividades'])
            
            # Renombrar columnas si es necesario
            if 'nombre' in df.columns and 'precio_unitario' in df.columns:
                df = df.rename(columns={'nombre': 'actividades', 'precio_unitario': 'costo_unitario'})
            
            # Agregar columna de cantidad si no existe
            if 'cantidad' not in df.columns:
                df['cantidad'] = 1
            
            # Calcular costo total
            df['costo_total'] = df['cantidad'] * df['costo_unitario']
            
            # Agregar metadatos
            if 'moneda' in data:
                df['moneda'] = data['moneda']
            if 'notas' in data:
                df['notas_extraccion'] = data['notas']
            
            return df
            
        except Exception as e:
            logger.exception(f"Error al procesar con IA: {str(e)}")
            # Crear un DataFrame vacío con las columnas necesarias
            return pd.DataFrame(columns=['actividades', 'cantidad', 'costo_unitario', 'costo_total'])
    
    def _sheet_contains_prices(self, df):
        """Determina si una hoja de Excel parece contener precios."""
        # Verificar nombres de columnas
        col_names = [str(col).lower() for col in df.columns]
        
        # Buscar columnas de precio y actividad
        has_price_col = any(any(price_key in col for price_key in self.precio_keywords) for col in col_names)
        has_activity_col = any(any(act_key in col for act_key in self.actividad_keywords) for col in col_names)
        
        if has_price_col and has_activity_col:
            return True
            
        # Verificar contenido numérico en al menos una columna
        numeric_cols = 0
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols += 1
        
        # Si tiene columnas numéricas y nombre de producto/servicio/actividad
        if numeric_cols > 0 and has_activity_col:
            return True
            
        # Verificar si hay columnas con valores monetarios
        for col in df.columns:
            if df[col].dtype == object:  # Solo para columnas de texto
                # Contar cuántas celdas parecen tener valores monetarios
                monetary_pattern = r'[$€£¥]\s*\d+([.,]\d{2})?|\d+([.,]\d+)?\s*[$€£¥]'
                monetary_cells = df[col].astype(str).str.contains(monetary_pattern, regex=True).sum()
                
                if monetary_cells > len(df) * 0.3:  # Al menos 30% de las celdas
                    return True
        
        return False
    
    def _table_contains_prices(self, df):
        """Determina si una tabla parece contener precios."""
        return self._sheet_contains_prices(df)  # Por ahora, misma lógica que para hojas
    
    def _process_tabular_data(self, df, file_path, interactive=False, sheet_name=None, column_mapping=None):
        """
        Procesa datos tabulares para normalizar y extraer precios.
        
        Args:
            df: DataFrame con datos sin procesar
            file_path: Ruta del archivo de origen
            interactive: Si debe solicitar entrada del usuario
            sheet_name: Nombre de la hoja (solo para Excel)
            
        Returns:
            DataFrame normalizado con actividades y precios
        """
        # Eliminar filas completamente vacías
        df = df.dropna(how='all')
        
        # Eliminar columnas completamente vacías
        df = df.dropna(axis=1, how='all')
        
        # Normalizar nombres de columnas
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        # Intentar identificar columnas clave
        activity_col = None
        price_col = None
        quantity_col = None
        
        if column_mapping:
            # Usar mapeo manual de columnas
            if self.debug_mode:
                logger.debug(f"Usando mapeo manual de columnas: {column_mapping}")
            
            activity_col = column_mapping.get('actividades')
            price_col = column_mapping.get('precios')
            quantity_col = column_mapping.get('cantidad')
            
            # Verificar que las columnas existan
            for col_name, col_value in column_mapping.items():
                if col_value and col_value not in df.columns:
                    raise ValueError(f"Columna {col_value} no encontrada en el archivo para {col_name}")
                    
            if self.debug_mode:
                logger.debug(f"Columnas mapeadas: actividad={activity_col}, precio={price_col}, cantidad={quantity_col}")
        else:
            # Buscar columnas automáticamente
            if self.debug_mode:
                logger.debug("Buscando columnas automáticamente")
            
            # 1. Buscar por nombres de columna
            for col in df.columns:
                col_lower = unidecode.unidecode(str(col)).lower()
                
                # Identificar columna de actividades
                if not activity_col:
                    for keyword in self.actividad_keywords:
                        if keyword in col_lower:
                            activity_col = col
                            if self.debug_mode:
                                logger.debug(f"Columna de actividad encontrada: {col}")
                            break
                
                # Identificar columna de precios
                if not price_col:
                    for keyword in self.precio_keywords:
                        if keyword in col_lower:
                            # Verificar si esta columna parece contener valores numéricos
                            if self._column_contains_numbers(df[col]):
                                price_col = col
                                if self.debug_mode:
                                    logger.debug(f"Columna de precio encontrada: {col}")
                                break
                
                # Identificar columna de cantidad
                if not quantity_col:
                    if any(qty in col_lower for qty in ['cantidad', 'quantity', 'qty', 'cant', 'cant.', 'unidades', 'units']):
                        if self._column_contains_numbers(df[col]):
                            quantity_col = col
                            if self.debug_mode:
                                logger.debug(f"Columna de cantidad encontrada: {col}")
        
        # 2. Si no se encontraron columnas por nombre, intentar por contenido
        if not activity_col or not price_col:
            # Buscar columnas que podrían contener descripciones de actividades
            text_columns = []
            numeric_columns = []
            
            for col in df.columns:
                if self._column_contains_mostly_text(df[col]):
                    text_columns.append(col)
                elif self._column_contains_numbers(df[col]):
                    numeric_columns.append(col)
            
            # Si solo hay una columna de texto, probablemente sea la de actividades
            if not activity_col and len(text_columns) == 1:
                activity_col = text_columns[0]
            
            # Para columnas numéricas, intentar encontrar la que tiene valores que parecen precios
            if not price_col and numeric_columns:
                for col in numeric_columns:
                    if self._column_contains_prices(df[col]):
                        price_col = col
                        break
                
                # Si no se encontró columna de precios, usar la última columna numérica
                if not price_col and numeric_columns:
                    price_col = numeric_columns[-1]
                
                # Si no encontramos columna de cantidad pero hay otra numérica, usarla
                if not quantity_col and len(numeric_columns) > 1:
                    for col in numeric_columns:
                        if col != price_col:
                            quantity_col = col
                            break
        
        # 3. Modo interactivo: preguntar al usuario
        if interactive:
            print("\nColumnas detectadas en el archivo:")
            for i, col in enumerate(df.columns):
                print(f"{i+1}. {col}")
            
            print(f"\nDetección automática: Actividad='{activity_col}', Precio='{price_col}', Cantidad='{quantity_col}'")
            
            # Preguntar si quiere usar la detección automática
            user_input = input("¿Usar estas columnas? (s/n): ").lower()
            
            if user_input != 's' and user_input != 'si' and user_input != 'y' and user_input != 'yes':
                # Pedir mapeo manual
                try:
                    act_idx = int(input("Ingrese número de columna para Actividades: ")) - 1
                    activity_col = df.columns[act_idx] if 0 <= act_idx < len(df.columns) else None
                    
                    price_idx = int(input("Ingrese número de columna para Precio: ")) - 1
                    price_col = df.columns[price_idx] if 0 <= price_idx < len(df.columns) else None
                    
                    qty_input = input("Ingrese número de columna para Cantidad (o Enter si no hay): ")
                    if qty_input:
                        qty_idx = int(qty_input) - 1
                        quantity_col = df.columns[qty_idx] if 0 <= qty_idx < len(df.columns) else None
                except:
                    logger.warning("Entrada inválida, usando detección automática")
        
        # Si no se pudo determinar las columnas, intentar con LLM
        if not activity_col or not price_col:
            if self.api_key:
                logger.info("No se pudieron determinar automáticamente las columnas. Intentando con IA.")
                return self._extract_with_ai_assistance(file_path)
            else:
                raise ValueError("No se pudieron determinar las columnas necesarias y no hay API key para IA")
        
        # Crear DataFrame normalizado
        result_df = pd.DataFrame()
        
        # Copiar columnas identificadas
        result_df['actividades'] = df[activity_col] if activity_col else None
        result_df['costo_unitario'] = self._normalize_price_column(df[price_col]) if price_col else None
        
        # Manejar columna de cantidad
        if quantity_col:
            result_df['cantidad'] = self._normalize_numeric_column(df[quantity_col])
        else:
            result_df['cantidad'] = 1
        
        # Calcular costo total
        result_df['costo_total'] = result_df['cantidad'] * result_df['costo_unitario']
        
        # Limpiar filas inválidas
        result_df = result_df.dropna(subset=['actividades', 'costo_unitario'])
        
        # Filtrar filas vacías o con actividades que parecen encabezados
        result_df = result_df[result_df['actividades'].astype(str).str.len() > 1]
        result_df = result_df[~result_df['actividades'].astype(str).str.lower().isin(
            ['actividad', 'descripcion', 'concepto', 'item', 'producto', 'servicio'])]
        
        return result_df
    
    def _column_contains_numbers(self, series):
        """Verifica si una columna contiene principalmente valores numéricos."""
        try:
            # Convertir a numérico y contar no-nulos
            numeric_count = pd.to_numeric(series, errors='coerce').notna().sum()
            return numeric_count > len(series) * 0.5  # Al menos 50% de valores numéricos
        except:
            return False
    
    def _column_contains_mostly_text(self, series):
        """Verifica si una columna contiene principalmente valores de texto."""
        if series.dtype == object:  # Solo para columnas de tipo objeto
            # Contar valores que no son numéricos
            non_numeric = series.astype(str).str.contains(r'[a-zA-Z]', regex=True).sum()
            return non_numeric > len(series) * 0.5  # Al menos 50% de valores con texto
        return False
    
    def _column_contains_prices(self, series):
        """Determina si una columna parece contener precios."""
        try:
            # Convertir a numérico
            numeric_series = pd.to_numeric(series, errors='coerce')
            
            # Verificar si hay valores no nulos
            if numeric_series.notna().sum() < len(series) * 0.5:
                return False
            
            # Verificar patrón: mayormente valores positivos, con 2 decimales frecuentes
            positive = (numeric_series > 0).sum() > numeric_series.notna().sum() * 0.9
            
            # Verificar si frecuentemente tienen 2 decimales
            decimals = series.astype(str).str.extract(r'\.(\d+)')[0].str.len()
            two_decimals = (decimals == 2).sum() > decimals.notna().sum() * 0.5
            
            # Si son positivos y tienen 2 decimales con frecuencia, probablemente son precios
            return positive and two_decimals
        except:
            return False
    
    def _normalize_price_column(self, series):
        """Normaliza una columna de precios a valores numéricos."""
        try:
            # Manejar el caso de Series de tipo object (texto)
            if series.dtype == object:
                # Limpiar formato de moneda y separadores
                clean_series = series.astype(str).apply(self._clean_price_string)
                return pd.to_numeric(clean_series, errors='coerce')
            else:
                # Ya es numérico
                return series
        except Exception as e:
            logger.warning(f"Error al normalizar columna de precios: {str(e)}")
            return series
    
    def _normalize_numeric_column(self, series):
        """Normaliza una columna numérica."""
        try:
            if series.dtype == object:
                # Limpiar formato y separadores
                clean_series = series.astype(str).apply(lambda x: re.sub(r'[^\d.,]', '', str(x)).replace(',', '.'))
                return pd.to_numeric(clean_series, errors='coerce').fillna(0)
            else:
                # Ya es numérico
                return series
        except Exception as e:
            logger.warning(f"Error al normalizar columna numérica: {str(e)}")
            return series
    
    def _clean_price_string(self, price_str):
        """Limpia un string que representa un precio para convertirlo a número."""
        if not isinstance(price_str, str):
            return price_str
            
        # Eliminar símbolos de moneda y espacios
        clean = re.sub(r'[$€£¥]', '', price_str)
        
        # Manejar diferentes formatos de números
        if ',' in clean and '.' in clean:
            # Determinar cuál es el separador decimal
            if clean.rindex('.') > clean.rindex(','):
                # El punto es el separador decimal (formato USA)
                clean = clean.replace(',', '')
            else:
                # La coma es el separador decimal (formato europeo)
                clean = clean.replace('.', '').replace(',', '.')
        elif ',' in clean:
            # Si solo hay comas, asumir que es separador decimal
            clean = clean.replace(',', '.')
        
        # Eliminar caracteres no numéricos y conservar el signo y punto decimal
        clean = re.sub(r'[^\d.-]', '', clean)
        
        return clean.strip()
