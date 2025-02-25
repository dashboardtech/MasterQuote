import os
import sys
import logging
import re
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Intentar importar DocLing instalado
try:
    import docling
    from docling_core.types.doc import DoclingDocument, TableItem, DocItem
    from docling_core.types.doc.document import ListItem
    from docling_core.types.legacy_doc.base import Table as LegacyTable
    from docling_core.utils.file import resolve_source_to_stream
    from docling.backend.msexcel_backend import MsExcelDocumentBackend
    from docling.pipeline.simple_pipeline import SimplePipeline
    from docling.datamodel.document import InputDocument, ConversionResult
    from docling.datamodel.pipeline_options import PipelineOptions
    from docling.datamodel.base_models import InputFormat
    _has_docling = True
except ImportError:
    # Si falla, intentar agregar la ruta al vendor/docling
    _has_docling = False
    vendor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vendor')
    docling_path = os.path.join(vendor_path, 'docling')
    
    if os.path.exists(docling_path):
        sys.path.append(docling_path)
        try:
            import docling
            from docling_core.types.doc import DoclingDocument, TableItem, DocItem
            from docling_core.types.doc.document import ListItem
            from docling_core.types.legacy_doc.base import Table as LegacyTable
            from docling_core.utils.file import resolve_source_to_stream
            from docling.backend.msexcel_backend import MsExcelDocumentBackend
            from docling.pipeline.simple_pipeline import SimplePipeline
            from docling.datamodel.document import InputDocument, ConversionResult
            from docling.datamodel.pipeline_options import PipelineOptions
            from docling.datamodel.base_models import InputFormat
            _has_docling = True
            logger.info("DocLing importado desde vendor/")
        except ImportError as e:
            logger.warning(f"DocLing no pudo ser importado ni siquiera desde vendor/: {e}")

class DocLingAdapter:
    """Adaptador para integrar DocLing con el sistema de cotizaciones."""
    
    def __init__(self):
        """Inicializa el adaptador de DocLing."""
        if not _has_docling:
            raise ImportError("DocLing no está disponible. Instálalo o clónalo en la carpeta vendor/")
        
        self.pipeline = SimplePipeline(pipeline_options=PipelineOptions())
        
        # Términos para identificar columnas
        self.precio_keywords = [
            'precio', 'price', 'valor', 'value', 'costo', 'cost', 
            'unitario', 'unit', 'total', 'monto', 'amount', 'tarifa', 
            'rate', 'pago', 'payment', 'p.u.', 'p/u', 'pu', '$'
        ]
        self.actividad_keywords = [
            'actividad', 'activity', 'item', 'concepto', 'concept',
            'descripcion', 'description', 'servicio', 'service',
            'producto', 'product', 'trabajo', 'work', 'tarea', 'task'
        ]
        self.cantidad_keywords = [
            'cantidad', 'quantity', 'qty', 'cant', 'cant.', 'unidades', 
            'units', 'vol', 'volume', 'volumen', 'num', 'número', 'number'
        ]
        
        logger.info("DocLing Adapter inicializado correctamente")
    
    def process_document(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Procesa un documento con DocLing y extrae información de precios.
        
        Args:
            file_path: Ruta al archivo a procesar
            
        Returns:
            DataFrame con actividades y precios, o None si falla
        """
        try:
            logger.info(f"Procesando documento con DocLing: {file_path}")
            
            # Cargar y procesar el documento
            path = Path(file_path)
            stream = resolve_source_to_stream(path)
            input_doc = InputDocument(
                path_or_stream=path,
                format=InputFormat.XLSX,
                backend=MsExcelDocumentBackend,
                filename=path.name
            )
            backend = input_doc._backend
            # Crear el resultado de conversión
            conv_result = ConversionResult(input=input_doc)
            
            # Construir el documento
            conv_result = self.pipeline._build_document(conv_result)
            
            # Obtener el documento resultante
            doc = conv_result.document
            
            if not doc:
                logger.error(f"No se pudo cargar el documento: {file_path}")
                return None
            
            # Buscar tablas en el documento
            tables = []
            
            # Acceder a las tablas directamente
            for table in doc.tables:
                logger.info(f"Table type: {type(table)}, attributes: {dir(table)}")
                logger.info(f"Table contents: {table}")
                tables.append(table)
            
            if tables:
                return self._process_tables(tables)
            
            # Si no hay tablas, intentar extraer texto estructurado
            return self._process_text(doc)
            
        except Exception as e:
            logger.warning(f"Error en procesamiento DocLing: {str(e)}")
            return None
    
    def _process_tables(self, tables: List[TableItem]) -> pd.DataFrame:
        """
        Procesa las tablas extraídas para identificar precios y actividades.
        
        Args:
            tables: Lista de tablas de DocLing
            
        Returns:
            DataFrame combinado con la información extraída
        """
        all_dfs = []
        
        for i, table in enumerate(tables):
            logger.info(f"Procesando tabla {i+1} de {len(tables)}")
            
            try:
                # Convertir la tabla a DataFrame
                df = table.export_to_dataframe()
                
                if df.empty:
                    continue
                    
                # Intentar inferir el rol de cada columna
                column_mapping = self._map_columns_heuristic(df)
                
                # Aplicar el mapeo
                if column_mapping and 'actividades' in column_mapping and 'costo_unitario' in column_mapping:
                    processed_df = self._apply_column_mapping(df, column_mapping)
                    all_dfs.append(processed_df)
                    
            except Exception as e:
                logger.warning(f"Error al procesar tabla {i+1}: {str(e)}")
                continue
        
        # Combinar resultados
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        
        return pd.DataFrame()

    def _process_text(self, doc: DoclingDocument) -> pd.DataFrame:
        """
        Procesa el texto del documento cuando no se encuentran tablas.
        
        Args:
            doc: Documento DocLing
            
        Returns:
            DataFrame con la información extraída
        """
        try:
            # Debug: ver atributos del documento
            logger.info(f"Document attributes: {dir(doc)}")
            logger.info(f"Document body: {doc.body}")
            
            # Obtener texto de cada nodo del documento
            lines = []
            for node in doc.body:
                logger.info(f"Node type: {type(node)}, attributes: {dir(node)}")
                if hasattr(node, 'text'):
                    lines.extend(str(node.text).split('\n'))
                elif hasattr(node, 'content'):
                    lines.extend(str(node.content).split('\n'))
            
            # Buscar patrones de precio y descripción
            items = []
            current_item = {}
            
            for line in lines:
                text = line.strip()
                if not text:
                    continue
                
                # Buscar precio
                price_match = re.search(r'\$\s*([\d,\.]+)', text)
                if price_match:
                    if current_item:
                        items.append(current_item)
                    current_item = {'costo_unitario': float(price_match.group(1).replace(',', ''))}
                    # La descripción es el texto antes del precio
                    description = text[:price_match.start()].strip()
                    if description:
                        current_item['actividades'] = description
                elif current_item and 'actividades' in current_item:
                    # Agregar texto adicional a la descripción existente
                    current_item['actividades'] = current_item['actividades'] + ' ' + text
            
            # Agregar el último item si existe
            if current_item:
                items.append(current_item)
            
            return pd.DataFrame(items)
        except Exception as e:
            logger.warning(f"Error al procesar texto: {str(e)}")
            return pd.DataFrame()
    
    def _map_columns_heuristic(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Algoritmo heurístico para mapear columnas.
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Diccionario con mapeo {"columna_original": "columna_destino"}
        """
        mapping = {}
        
        # Normalizar nombres de columnas
        col_names = [str(col).lower() for col in df.columns]
        
        # Buscar columna de actividades
        for keyword in self.actividad_keywords:
            for i, col in enumerate(col_names):
                if keyword in col:
                    mapping['actividades'] = df.columns[i]
                    break
            if 'actividades' in mapping:
                break
        
        # Buscar columna de precios
        for keyword in self.precio_keywords:
            for i, col in enumerate(col_names):
                if keyword in col and self._column_contains_numbers(df[df.columns[i]]):
                    mapping['costo_unitario'] = df.columns[i]
                    break
            if 'costo_unitario' in mapping:
                break
        
        # Buscar columna de cantidad
        for keyword in self.cantidad_keywords:
            for i, col in enumerate(col_names):
                if keyword in col and self._column_contains_numbers(df[df.columns[i]]):
                    mapping['cantidad'] = df.columns[i]
                    break
            if 'cantidad' in mapping:
                break
        
        # Si no encontramos por nombres, intentar por contenido
        if not mapping.get('actividades') or not mapping.get('costo_unitario'):
            # Buscar columna con mayor contenido textual para actividades
            text_columns = []
            for col in df.columns:
                if self._column_contains_mostly_text(df[col]):
                    text_columns.append(col)
            
            if text_columns and not mapping.get('actividades'):
                # Usar la columna de texto más larga promedio
                text_lens = {col: df[col].astype(str).str.len().mean() for col in text_columns}
                mapping['actividades'] = max(text_lens, key=text_lens.get)
            
            # Buscar columnas numéricas para precio/cantidad
            numeric_columns = []
            for col in df.columns:
                if self._column_contains_numbers(df[col]):
                    numeric_columns.append(col)
            
            if numeric_columns and not mapping.get('costo_unitario'):
                # Analizar cuál parece más un precio (valores más altos generalmente)
                if len(numeric_columns) >= 2:
                    # Ordenar por valor promedio descendente
                    numeric_cols_avg = {}
                    for col in numeric_columns:
                        try:
                            numeric_cols_avg[col] = pd.to_numeric(df[col], errors='coerce').mean()
                        except:
                            numeric_cols_avg[col] = 0
                    
                    sorted_cols = sorted(numeric_cols_avg.items(), key=lambda x: x[1], reverse=True)
                    
                    # El de mayor valor suele ser precio, el siguiente cantidad
                    if sorted_cols and not mapping.get('costo_unitario'):
                        mapping['costo_unitario'] = sorted_cols[0][0]
                    
                    if len(sorted_cols) > 1 and not mapping.get('cantidad'):
                        mapping['cantidad'] = sorted_cols[1][0]
                elif numeric_columns:
                    mapping['costo_unitario'] = numeric_columns[0]
        
        return mapping
    
    def _apply_column_mapping(self, df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Aplica el mapeo de columnas al DataFrame.
        
        Args:
            df: DataFrame original
            mapping: Diccionario con mapeo
            
        Returns:
            DataFrame con columnas mapeadas y normalizadas
        """
        result_df = pd.DataFrame()
        
        # Copiar las columnas mapeadas
        if 'actividades' in mapping:
            result_df['actividades'] = df[mapping['actividades']]
        
        if 'costo_unitario' in mapping:
            result_df['costo_unitario'] = self._normalize_price_column(df[mapping['costo_unitario']])
        
        if 'cantidad' in mapping:
            result_df['cantidad'] = self._normalize_numeric_column(df[mapping['cantidad']])
        elif 'cantidad' not in result_df.columns:
            result_df['cantidad'] = 1
        
        # Calcular costo total
        if 'costo_unitario' in result_df.columns and 'cantidad' in result_df.columns:
            result_df['costo_total'] = result_df['cantidad'] * result_df['costo_unitario']
        
        # Filtrar filas vacías o inválidas
        result_df = result_df.dropna(subset=['actividades'])
        
        # Eliminar filas que parecen encabezados
        if 'actividades' in result_df.columns:
            result_df = result_df[~result_df['actividades'].astype(str).str.lower().isin(
                self.actividad_keywords + self.precio_keywords + self.cantidad_keywords
            )]
        
        return result_df
    
    def _process_text(self, doc: DoclingDocument) -> Optional[pd.DataFrame]:
        """
        Intenta extraer precios y actividades del texto cuando no hay tablas.
        
        Args:
            doc: Documento DocLing
            
        Returns:
            DataFrame con información estructurada o None
        """
        try:
            # Extraer texto y dividir en párrafos
            text = doc.text
            paragraphs = text.split('\n\n')
            
            # Buscar patrones de precio y actividad
            items = []
            current_item = {}
            
            for paragraph in paragraphs:
                # Buscar precios con formato monetario
                import re
                price_matches = re.findall(r'\$\s*([\d,]+(?:\.\d{2})?)', paragraph)
                
                # Si encontramos un precio, buscar la descripción cercana
                if price_matches:
                    try:
                        price = float(price_matches[0].replace(',', ''))
                        
                        # La descripción es el texto antes del precio
                        desc_match = re.split(r'\$\s*[\d,]+(?:\.\d{2})?', paragraph)[0]
                        if desc_match:
                            items.append({
                                'actividades': desc_match.strip(),
                                'costo_unitario': price,
                                'cantidad': 1
                            })
                    except:
                        continue
            
            if items:
                df = pd.DataFrame(items)
                df['costo_total'] = df['cantidad'] * df['costo_unitario']
                return df
            
            return None
            
        except Exception as e:
            logger.warning(f"Error al procesar texto: {str(e)}")
            return None
    
    # Métodos auxiliares para análisis de columnas
    def _column_contains_numbers(self, series) -> bool:
        """
        Verifica si una columna contiene principalmente valores numéricos.
        """
        try:
            numeric_count = pd.to_numeric(series, errors='coerce').notna().sum()
            return numeric_count > len(series) * 0.5  # Al menos 50% valores numéricos
        except:
            return False
    
    def _column_contains_mostly_text(self, series) -> bool:
        """
        Verifica si una columna contiene principalmente valores de texto.
        """
        if series.dtype == object:
            non_numeric = series.astype(str).str.contains(r'[a-zA-Z]', regex=True).sum()
            return non_numeric > len(series) * 0.5
        return False
    
    def _normalize_price_column(self, series) -> pd.Series:
        """
        Normaliza una columna de precios a valores numéricos.
        """
        try:
            if series.dtype == object:
                # Limpiar símbolos de moneda y separadores
                clean_series = series.astype(str).apply(self._clean_price_string)
                return pd.to_numeric(clean_series, errors='coerce')
            return series
        except:
            return series
    
    def _normalize_numeric_column(self, series) -> pd.Series:
        """
        Normaliza una columna numérica.
        """
        try:
            if series.dtype == object:
                clean_series = series.astype(str).apply(
                    lambda x: re.sub(r'[^\d.,]', '', str(x)).replace(',', '.')
                )
                return pd.to_numeric(clean_series, errors='coerce').fillna(0)
            return series
        except:
            return series
    
    def _clean_price_string(self, price_str: str) -> str:
        """
        Limpia un string que representa un precio.
        """
        if not isinstance(price_str, str):
            return price_str
            
        # Eliminar símbolos de moneda
        import re
        clean = re.sub(r'[$€£¥]', '', price_str)
        
        # Manejar separadores de miles y decimales
        if ',' in clean and '.' in clean:
            if clean.rindex('.') > clean.rindex(','):
                # Formato USA: 1,234.56
                clean = clean.replace(',', '')
            else:
                # Formato europeo: 1.234,56
                clean = clean.replace('.', '').replace(',', '.')
        elif ',' in clean:
            clean = clean.replace(',', '.')
        
        clean = re.sub(r'[^\d.-]', '', clean)
        return clean.strip()
    
    def analyze_document_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Analiza la estructura del documento para identificar secciones relevantes.
        
        Args:
            file_path: Ruta al archivo a analizar
            
        Returns:
            Diccionario con información sobre la estructura del documento
        """
        try:
            doc = DoclingDocument.from_file(file_path)
            
            # Analizar estructura
            structure = {
                'tables': len(doc.tables) if hasattr(doc, 'tables') else 0,
                'pages': len(doc.pages) if hasattr(doc, 'pages') else 1,
                'sections': [],
                'potential_price_areas': []
            }
            
            # Identificar secciones
            for section in doc.sections if hasattr(doc, 'sections') else []:
                structure['sections'].append({
                    'title': section.title if hasattr(section, 'title') else None,
                    'text_length': len(section.text) if hasattr(section, 'text') else 0,
                    'has_table': bool(section.tables) if hasattr(section, 'tables') else False
                })
            
            return structure
            
        except Exception as e:
            logger.error(f"Error al analizar estructura del documento {file_path}: {e}")
            raise
    
    def extract_table_with_context(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Extrae la tabla más relevante junto con su contexto.
        
        Args:
            file_path: Ruta al archivo a procesar
            
        Returns:
            Tupla de (DataFrame de la tabla más relevante, información de contexto)
        """
        try:
            doc = DoclingDocument.from_file(file_path)
            tables = self.table_extractor.extract(doc)
            
            if not tables:
                return None, {'error': 'No se encontraron tablas'}
            
            # Encontrar la tabla más relevante (la que probablemente contenga precios)
            best_table = None
            best_score = 0
            
            for table in tables:
                score = 0
                headers = [str(h).lower() for h in table.headers]
                
                # Puntuar la tabla basado en sus encabezados
                price_keywords = ['precio', 'costo', 'valor', 'importe', 'total']
                for keyword in price_keywords:
                    if any(keyword in h for h in headers):
                        score += 2
                
                # Puntuar basado en el contenido
                for row in table.data:
                    if any(str(cell).startswith('$') for cell in row):
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_table = table
            
            if best_table:
                df = pd.DataFrame(best_table.data, columns=best_table.headers)
                context = {
                    'confidence_score': best_score,
                    'table_position': 'top' if tables.index(best_table) == 0 else 'middle',
                    'surrounding_text': self._get_surrounding_text(doc, best_table)
                }
                return df, context
            
            return None, {'error': 'No se encontró una tabla relevante'}
            
        except Exception as e:
            logger.error(f"Error al extraer tabla con contexto de {file_path}: {e}")
            raise
    
    def _get_surrounding_text(self, doc: DoclingDocument, table: Any) -> str:
        """Obtiene el texto que rodea a una tabla."""
        try:
            # Este es un método simplificado. La implementación real
            # dependerá de cómo DocLing maneja el contexto de las tablas
            before = ""
            after = ""
            
            if hasattr(table, 'start_index') and hasattr(table, 'end_index'):
                text = doc.text
                before = text[max(0, table.start_index - 200):table.start_index]
                after = text[table.end_index:min(len(text), table.end_index + 200)]
            
            return {
                'before': before.strip(),
                'after': after.strip()
            }
        except Exception as e:
            logger.warning(f"No se pudo obtener el texto circundante: {e}")
            return {'before': '', 'after': ''}
