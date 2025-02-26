"""
Módulo para la gestión avanzada de extractores con validación cruzada.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .extractors.base import BaseExtractor
from .extractors.factory import ExtractorFactory
from .extractors.excel import ExcelExtractor
from .extractors.csv import CSVExtractor
from .extractors.pdf import PDFExtractor
from .extractors.ai_assisted import AIAssistedExtractor
from .extractors.dockling import DocklingExtractor
from .async_processing import AsyncProcessor, ProcessingResult

logger = logging.getLogger(__name__)

class ExtractorManager:
    """
    Gestiona múltiples extractores para proporcionar validación cruzada
    y resultados de mayor confianza mediante comparación y votación.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None, 
                 dockling_api_key: Optional[str] = None,
                 num_extractors: int = 3,
                 use_parallel: bool = True):
        """
        Inicializa el gestor de extractores.
        
        Args:
            api_key: API key para servicios de IA (OpenAI, etc.)
            dockling_api_key: API key para Dockling
            num_extractors: Número máximo de extractores a utilizar (por defecto 3)
            use_parallel: Si debe usar procesamiento paralelo
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.dockling_api_key = dockling_api_key or os.environ.get("DOCKLING_API_KEY")
        self.num_extractors = num_extractors
        self.use_parallel = use_parallel
        self.async_processor = AsyncProcessor() if use_parallel else None
        
    def get_extractors_for_file(self, file_path: str) -> List[BaseExtractor]:
        """
        Obtiene los extractores más adecuados para un archivo específico.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Lista de extractores adecuados para el archivo
        """
        file_ext = Path(file_path).suffix.lower()
        extractors = []
        
        # Siempre incluir el extractor de Dockling como una opción
        extractors.append(DocklingExtractor(api_key=self.dockling_api_key))
        
        # Añadir extractores específicos según tipo de archivo
        if file_ext in ['.xlsx', '.xls', '.xlsm']:
            extractors.append(ExcelExtractor())
        elif file_ext == '.csv':
            extractors.append(CSVExtractor())
        elif file_ext == '.pdf':
            extractors.append(PDFExtractor())
        
        # Siempre intentar incluir el extractor asistido por IA
        if self.api_key:
            extractors.append(AIAssistedExtractor(api_key=self.api_key))
        
        # Limitar el número de extractores si es necesario
        return extractors[:self.num_extractors]
    
    def extract_with_validation(self, file_path: str, min_confidence: float = 0.5, process_all_sheets: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Extrae datos del archivo con validación cruzada entre múltiples extractores.
        
        Args:
            file_path: Ruta al archivo
            min_confidence: Confianza mínima requerida (0.0-1.0)
            process_all_sheets: Si debe procesar todas las hojas en caso de archivos Excel
            
        Returns:
            Tuple con (DataFrame resultado, metadatos)
        """
        start_time = time.time()
        extractors = self.get_extractors_for_file(file_path)
        
        if not extractors:
            logger.error(f"No se encontraron extractores adecuados para {file_path}")
            return pd.DataFrame(), {"error": "No se encontraron extractores adecuados"}
        
        # Limitar al número máximo de extractores configurado
        extractors = extractors[:self.num_extractors]
        
        # Extraer datos con cada extractor
        results = []
        
        if self.use_parallel:
            # Procesamiento paralelo
            futures = []
            
            with ThreadPoolExecutor(max_workers=len(extractors)) as executor:
                for extractor in extractors:
                    extractor_name = extractor.__class__.__name__
                    logger.info(f"Iniciando extracción con {extractor_name}")
                    
                    # Crear una función para este extractor específico
                    future = executor.submit(
                        self._extract_with_extractor, 
                        extractor, 
                        file_path, 
                        extractor_name,
                        process_all_sheets
                    )
                    futures.append(future)
            
            # Recopilar resultados
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error en extracción paralela: {str(e)}")
        else:
            # Procesamiento secuencial
            for extractor in extractors:
                extractor_name = extractor.__class__.__name__
                logger.info(f"Iniciando extracción con {extractor_name}")
                
                result = self._extract_with_extractor(
                    extractor, 
                    file_path, 
                    extractor_name,
                    process_all_sheets
                )
                if result:
                    results.append(result)
        
        # Validar y consolidar resultados
        result_df, metadata = self._validate_and_combine(results)
        
        # Verificar confianza mínima
        if metadata.get('confidence', 0) < min_confidence:
            logger.warning(f"Confianza {metadata.get('confidence'):.2f} por debajo del mínimo requerido {min_confidence}")
            return pd.DataFrame(), {
                **metadata,
                'successful': False,
                'message': f"Confianza insuficiente: {metadata.get('confidence'):.2f} < {min_confidence}"
            }
        
        return result_df, metadata
    
    def _extract_with_extractor(self, extractor: BaseExtractor, file_path: str, extractor_name: str, process_all_sheets: bool) -> Dict[str, Any]:
        """
        Ejecuta la extracción con un extractor específico y maneja errores.
        
        Args:
            extractor: Extractor a utilizar
            file_path: Ruta al archivo
            extractor_name: Nombre del extractor
            process_all_sheets: Si debe procesar todas las hojas en caso de archivos Excel
            
        Returns:
            Diccionario con el resultado de la extracción
        """
        start_time = time.time()
        
        try:
            logger.info(f"Iniciando extracción con {extractor_name}")
            if process_all_sheets and isinstance(extractor, ExcelExtractor):
                dfs = extractor.extract_all_sheets(file_path)
                df = pd.concat(dfs, ignore_index=True)
            else:
                df = extractor.extract(file_path)
            
            processing_time = time.time() - start_time
            
            if df.empty:
                logger.warning(f"Extractor {extractor_name} retornó DataFrame vacío")
                return {
                    'extractor': extractor_name,
                    'data': pd.DataFrame(),
                    'success': False,
                    'error': "DataFrame vacío",
                    'processing_time': processing_time
                }
            
            logger.info(f"Extracción exitosa con {extractor_name} ({len(df)} filas en {processing_time:.2f}s)")
            return {
                'extractor': extractor_name,
                'data': df,
                'success': True,
                'error': None,
                'processing_time': processing_time,
                'rows': len(df)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"Error con extractor {extractor_name}: {str(e)}")
            return {
                'extractor': extractor_name,
                'data': pd.DataFrame(),
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _validate_and_combine(self, extraction_results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Valida y combina resultados de múltiples extractores.
        
        Args:
            extraction_results: Lista de resultados de extracción
            
        Returns:
            Tuple con (DataFrame combinado, metadatos)
        """
        # Filtrar resultados exitosos
        successful_results = [r for r in extraction_results if r['success'] and not r['data'].empty]
        
        if not successful_results:
            logger.warning("Ningún extractor produjo resultados válidos")
            return pd.DataFrame(), {
                'confidence': 0.0,
                'extractors_used': [r['extractor'] for r in extraction_results],
                'errors': {r['extractor']: r['error'] for r in extraction_results if r['error']},
                'successful': False,
                'message': "Ningún extractor produjo resultados válidos"
            }
        
        # Si solo hay un resultado exitoso, usarlo directamente
        if len(successful_results) == 1:
            result = successful_results[0]
            return result['data'], {
                'confidence': 0.7,  # Confianza media-alta por tener solo un extractor exitoso
                'extractor_used': result['extractor'],
                'extractors_tried': [r['extractor'] for r in extraction_results],
                'successful': True,
                'message': f"Resultado único de {result['extractor']}"
            }
        
        # Para múltiples resultados, implementar un sistema de votación/consenso
        # y calcular una puntuación de confianza
        
        # 1. Normalizar los nombres de columnas para comparación
        normalized_results = []
        for result in successful_results:
            df = result['data'].copy()
            # Crear diccionario que mapea nombres originales a normalizados
            col_mapping = {}
            for col in df.columns:
                normalized = self._normalize_column_name(col)
                col_mapping[col] = normalized
            
            # Renombrar columnas con nombres normalizados
            df.rename(columns=col_mapping, inplace=True)
            
            normalized_results.append({
                'extractor': result['extractor'],
                'data': df,
                'col_mapping': col_mapping,
                'orig_data': result['data']
            })
        
        # 2. Identificar columnas comunes (actividad y precio)
        activity_cols = set()
        price_cols = set()
        
        for result in normalized_results:
            # Identificar columnas de actividad y precio basado en nombres normalizados
            for col in result['data'].columns:
                if any(keyword in col.lower() for keyword in ['actividad', 'activity', 'concepto', 'description', 'item']):
                    activity_cols.add(col)
                elif any(keyword in col.lower() for keyword in ['precio', 'price', 'costo', 'cost', 'valor', 'value']):
                    price_cols.add(col)
        
        # 3. Seleccionar el resultado con mayor información
        # y validarlo con los demás resultados
        
        # Ordenar por número de filas, de mayor a menor
        normalized_results.sort(key=lambda x: len(x['data']), reverse=True)
        
        # Seleccionar el dataset principal (el que tiene más filas)
        primary_result = normalized_results[0]
        primary_df = primary_result['orig_data'].copy()
        
        # Calcular confianza basada en similitud con otros resultados
        confidence, metadata = self._calculate_confidence(normalized_results)
        
        # Metadata para devolver junto con los resultados
        metadata = {
            'confidence': confidence,
            'primary_extractor': primary_result['extractor'],
            'supporting_extractors': [r['extractor'] for r in normalized_results[1:]],
            'extractors_tried': [r['extractor'] for r in extraction_results],
            'successful': True,
            'rows_by_extractor': {r['extractor']: len(r['data']) for r in successful_results},
            'message': f"Resultado principal de {primary_result['extractor']} validado con {len(normalized_results)-1} extractores adicionales",
            **metadata
        }
        
        return primary_df, metadata
    
    def _normalize_column_name(self, column_name: str) -> str:
        """
        Normaliza el nombre de una columna para facilitar comparaciones.
        
        Args:
            column_name: Nombre original de la columna
            
        Returns:
            Nombre normalizado
        """
        import re
        import unidecode
        
        # Convertir a minúsculas y eliminar acentos
        normalized = unidecode.unidecode(str(column_name).lower())
        
        # Eliminar caracteres especiales y espacios
        normalized = re.sub(r'[^a-z0-9]', '', normalized)
        
        return normalized
    
    def _calculate_confidence(self, results):
        """
        Calcula la confianza en los resultados basado en la similitud entre extractores
        y los metadatos de validación de cada extractor.
        
        Args:
            results: Lista de resultados de diferentes extractores
            
        Returns:
            Diccionario con confianza y metadatos
        """
        confidence = 0.0
        metadata = {
            "extractors_used": len(results),
            "validation_details": []
        }
        
        # Si solo hay un extractor, usar su confianza interna
        if len(results) == 1:
            # Verificar si el resultado tiene metadatos de validación
            if 'metadata' in results[0] and isinstance(results[0]['metadata'], dict):
                extractor_confidence = results[0]['metadata'].get('confianza', 0.7)
                metadata["validation_details"].append({
                    "extractor": results[0]['extractor'],
                    "confidence": extractor_confidence,
                    "details": results[0]['metadata']
                })
                confidence = extractor_confidence
            else:
                # Confianza predeterminada para un solo extractor sin metadatos
                confidence = 0.7
                metadata["validation_details"].append({
                    "extractor": results[0]['extractor'],
                    "confidence": confidence,
                    "details": {"reason": "No validation metadata available"}
                })
        else:
            # Calcular similitud entre extractores
            similarities = []
            
            for i in range(len(results)):
                for j in range(i+1, len(results)):
                    # Calcular similitud entre resultados[i] y resultados[j]
                    sim = self._calculate_similarity(results[i]['data'], results[j]['data'])
                    similarities.append(sim)
                    
                    # Registrar detalles de similitud
                    metadata["validation_details"].append({
                        "comparison": f"{results[i]['extractor']} vs {results[j]['extractor']}",
                        "similarity": sim
                    })
            
            # Calcular confianza basada en similitud
            if similarities:
                similarity_confidence = sum(similarities) / len(similarities)
            else:
                similarity_confidence = 0.0
                
            # Incorporar confianza interna de cada extractor
            internal_confidences = []
            for result in results:
                if 'metadata' in result and isinstance(result['metadata'], dict):
                    internal_conf = result['metadata'].get('confianza', 0.7)
                    internal_confidences.append(internal_conf)
                    
                    # Registrar detalles de validación interna
                    metadata["validation_details"].append({
                        "extractor": result['extractor'],
                        "internal_confidence": internal_conf,
                        "details": result['metadata']
                    })
            
            # Calcular confianza final combinando similitud y confianza interna
            if internal_confidences:
                internal_confidence = sum(internal_confidences) / len(internal_confidences)
                # Dar más peso a la similitud entre extractores
                confidence = 0.7 * similarity_confidence + 0.3 * internal_confidence
            else:
                confidence = similarity_confidence
        
        # Asegurar que la confianza esté entre 0 y 1
        confidence = max(0.0, min(1.0, confidence))
        
        metadata["final_confidence"] = confidence
        return confidence, metadata
    
    def _calculate_similarity(self, df1: pd.DataFrame, df2: pd.DataFrame) -> float:
        """
        Calcula la similitud entre dos DataFrames.
        
        Args:
            df1: Primer DataFrame
            df2: Segundo DataFrame
            
        Returns:
            Puntuación de similitud entre 0.0 y 1.0
        """
        try:
            # Si alguno está vacío, no hay similitud
            if df1.empty or df2.empty:
                return 0.0
            
            # Normalizar nombres de columnas
            df1_cols = set(df1.columns)
            df2_cols = set(df2.columns)
            
            # Similitud en columnas
            common_cols = df1_cols.intersection(df2_cols)
            col_similarity = len(common_cols) / max(len(df1_cols), len(df2_cols))
            
            # Si no hay columnas comunes, no podemos comparar más
            if not common_cols:
                logger.warning("No hay columnas comunes entre los DataFrames para comparar")
                return col_similarity * 0.5  # Dar algo de crédito por tener columnas
            
            # Comparar el número de filas
            row_count_similarity = min(len(df1), len(df2)) / max(len(df1), len(df2))
            
            # Comparar valores en columnas comunes
            value_similarities = []
            
            # Seleccionar columnas numéricas comunes
            numeric_cols = [col for col in common_cols if col in ['cantidad', 'precio_unitario', 'total']]
            
            if numeric_cols:
                # Para columnas numéricas, comparar estadísticas
                for col in numeric_cols:
                    try:
                        # Crear copias para evitar advertencias de modificación
                        df1_col = pd.to_numeric(df1[col].copy(), errors='coerce')
                        df2_col = pd.to_numeric(df2[col].copy(), errors='coerce')
                        
                        # Filtrar valores NaN para estadísticas más precisas
                        df1_valid = df1_col.dropna()
                        df2_valid = df2_col.dropna()
                        
                        # Si no hay valores válidos, pasar a la siguiente columna
                        if len(df1_valid) == 0 or len(df2_valid) == 0:
                            logger.warning(f"No hay valores numéricos válidos en la columna {col}")
                            continue
                        
                        # Comparar medias
                        mean1 = df1_valid.mean()
                        mean2 = df2_valid.mean()
                        
                        if pd.isna(mean1) or pd.isna(mean2):
                            mean_sim = 0.5  # Neutral si no podemos calcular
                        elif mean1 == 0 and mean2 == 0:
                            mean_sim = 1.0  # Si ambos son 0, son idénticos
                        else:
                            # Evitar división por cero
                            denominator = max(abs(mean1), abs(mean2), 0.001)
                            mean_diff = abs(mean1 - mean2) / denominator
                            mean_sim = 1.0 - min(mean_diff, 1.0)
                        
                        # Comparar desviaciones estándar
                        std1 = df1_valid.std()
                        std2 = df2_valid.std()
                        
                        if pd.isna(std1) or pd.isna(std2):
                            std_sim = 0.5  # Neutral si no podemos calcular
                        elif std1 == 0 and std2 == 0:
                            std_sim = 1.0  # Si ambos son 0, son idénticos
                        else:
                            # Evitar división por cero
                            denominator = max(abs(std1), abs(std2), 0.001)
                            std_diff = abs(std1 - std2) / denominator
                            std_sim = 1.0 - min(std_diff, 1.0)
                        
                        # Combinar similitudes para esta columna
                        col_value_sim = (mean_sim + std_sim) / 2
                        value_similarities.append(col_value_sim)
                    except Exception as e:
                        logger.warning(f"Error al comparar columna {col}: {str(e)}")
            
            # Comparar columnas de texto (como descripción)
            text_cols = [col for col in common_cols if col in ['descripcion', 'codigo', 'unidad']]
            
            if text_cols:
                for col in text_cols:
                    try:
                        # Asegurar que tratamos todos los valores como strings
                        df1[col] = df1[col].astype(str)
                        df2[col] = df2[col].astype(str)
                        
                        # Contar valores únicos y evitar NaN
                        unique1 = df1[col].nunique()
                        unique2 = df2[col].nunique()
                        
                        # Evitar división por cero
                        max_unique = max(unique1, unique2, 1)
                        unique_sim = min(unique1, unique2) / max_unique
                        
                        # Muestrear algunos valores para comparar (hasta 100)
                        sample_size = min(50, min(len(df1), len(df2)))
                        
                        if sample_size > 0:
                            try:
                                # Tomar muestras 
                                sample1 = df1.sample(sample_size, replace=True)[col]
                                sample2 = df2.sample(sample_size, replace=True)[col]
                                
                                # Calcular longitud media de texto, con manejo seguro
                                len1 = sample1.str.len().mean() or 0  # Si es NaN, usar 0
                                len2 = sample2.str.len().mean() or 0  # Si es NaN, usar 0
                                
                                if len1 > 0 and len2 > 0:
                                    # Similitud en longitud de texto
                                    len_sim = min(len1, len2) / max(len1, len2)
                                else:
                                    len_sim = 0.5
                                
                                # Combinar similitudes para esta columna
                                col_value_sim = (unique_sim + len_sim) / 2
                            except Exception as e:
                                logger.warning(f"Error al procesar muestras para {col}: {str(e)}")
                                col_value_sim = unique_sim  # Usar solo la similitud de valores únicos
                        else:
                            col_value_sim = unique_sim
                        
                        value_similarities.append(col_value_sim)
                    except Exception as e:
                        logger.warning(f"Error al comparar columna de texto {col}: {str(e)}")
            
            # Calcular similitud final
            if value_similarities:
                value_similarity = sum(value_similarities) / len(value_similarities)
            else:
                value_similarity = 0.5  # Valor neutral si no pudimos comparar valores
            
            # Combinar todas las similitudes
            final_similarity = (col_similarity * 0.3) + (row_count_similarity * 0.3) + (value_similarity * 0.4)
            
            return final_similarity
            
        except Exception as e:
            logger.exception(f"Error al calcular similitud entre DataFrames: {str(e)}")
            return 0.5  # Valor neutral en caso de error


# Funciones de utilidad para trabajar con el ExtractorManager

def extract_with_confidence(file_path: str, 
                           api_key: Optional[str] = None, 
                           dockling_api_key: Optional[str] = None,
                           min_confidence: float = 0.5,
                           process_all_sheets: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Función de conveniencia para extraer datos con validación cruzada.
    
    Args:
        file_path: Ruta al archivo
        api_key: API key para servicios de IA
        dockling_api_key: API key para Dockling
        min_confidence: Confianza mínima requerida
        process_all_sheets: Si debe procesar todas las hojas en caso de archivos Excel
        
    Returns:
        Tuple con (DataFrame resultado, metadatos)
    """
    manager = ExtractorManager(api_key=api_key, dockling_api_key=dockling_api_key)
    df, metadata = manager.extract_with_validation(file_path, min_confidence, process_all_sheets)
    
    return df, metadata


def process_batch_with_validation(files: List[str], 
                                api_key: Optional[str] = None,
                                dockling_api_key: Optional[str] = None,
                                parallel: bool = True,
                                max_workers: Optional[int] = None,
                                process_all_sheets: bool = False) -> Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Procesa un lote de archivos con validación cruzada.
    
    Args:
        files: Lista de rutas de archivos
        api_key: API key para servicios de IA
        dockling_api_key: API key para Dockling
        parallel: Si debe procesar en paralelo
        max_workers: Número máximo de workers para procesamiento paralelo
        process_all_sheets: Si debe procesar todas las hojas en caso de archivos Excel
        
    Returns:
        Diccionario con {ruta_archivo: (DataFrame, metadatos)}
    """
    results = {}
    
    if parallel:
        from .async_processing import BatchProcessor
        
        processor = BatchProcessor(max_workers=max_workers)
        
        def process_file(file_path):
            try:
                manager = ExtractorManager(
                    api_key=api_key, 
                    dockling_api_key=dockling_api_key,
                    use_parallel=True
                )
                df, metadata = manager.extract_with_validation(file_path, process_all_sheets=process_all_sheets)
                return file_path, (df, metadata)
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {str(e)}")
                return file_path, (pd.DataFrame(), {
                    'error': str(e),
                    'successful': False,
                    'message': f"Error: {str(e)}"
                })
        
        # Procesar archivos en paralelo
        processing_results = processor.process_batch(files, process_file)
        
        # Recopilar resultados
        for pr in processing_results:
            if pr.success and pr.data:
                file_path, result = pr.data
                results[file_path] = result
    else:
        # Procesamiento secuencial
        manager = ExtractorManager(
            api_key=api_key, 
            dockling_api_key=dockling_api_key,
            use_parallel=True  # Paralelo dentro de cada archivo
        )
        
        for file_path in files:
            try:
                df, metadata = manager.extract_with_validation(file_path, process_all_sheets=process_all_sheets)
                results[file_path] = (df, metadata)
            except Exception as e:
                logger.error(f"Error procesando {file_path}: {str(e)}")
                results[file_path] = (pd.DataFrame(), {
                    'error': str(e),
                    'successful': False,
                    'message': f"Error: {str(e)}"
                })
    
    return results
