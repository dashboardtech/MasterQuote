import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import BaseExtractor
import numpy as np
import os

logger = logging.getLogger(__name__)

class ExcelExtractor(BaseExtractor):
    """Extractor para archivos Excel."""
    
    def extract(self, file_path: str, interactive: bool = False, process_all_sheets: bool = False) -> pd.DataFrame:
        """
        Extrae datos de un archivo Excel.
        
        Args:
            file_path: Ruta al archivo Excel
            interactive: Si debe ser interactivo
            process_all_sheets: Si debe procesar todas las hojas y combinar resultados
            
        Returns:
            DataFrame con actividades y precios
        """
        try:
            # Si se solicita procesar todas las hojas, usar el método específico
            if process_all_sheets:
                logger.info("Procesando todas las hojas del workbook")
                return self._process_all_sheets(file_path, interactive)
            
            # Intentar leer todas las hojas
            xls = pd.ExcelFile(file_path)
            sheets = xls.sheet_names
            
            if len(sheets) == 1:
                # Si solo hay una hoja, usarla directamente
                df = pd.read_excel(file_path)
                return self._process_tabular_data(df, interactive)
            else:
                # Si hay múltiples hojas, intentar encontrar las correctas
                logger.info(f"Archivo con múltiples hojas: {sheets}")
                
                # Opción 1: Intentar cada hoja individualmente
                valid_dfs = []
                for sheet in sheets:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet)
                        if not df.empty and self._is_valid_data(df):
                            logger.info(f"Hoja válida encontrada: {sheet}")
                            processed_df = self._process_tabular_data(df, interactive)
                            if not processed_df.empty:
                                valid_dfs.append(processed_df)
                    except Exception as e:
                        logger.warning(f"Error al procesar hoja {sheet}: {str(e)}")
                        continue
                
                # Si encontramos múltiples hojas válidas, combinarlas
                if len(valid_dfs) > 1:
                    logger.info(f"Combinando datos de {len(valid_dfs)} hojas válidas")
                    return pd.concat(valid_dfs, ignore_index=True)
                elif len(valid_dfs) == 1:
                    logger.info("Usando datos de una hoja válida")
                    return valid_dfs[0]
                
                # Opción 2: Si no encontramos hojas válidas individualmente,
                # intentar buscar hojas con nombres comunes de presupuestos
                budget_sheet_patterns = ['presupuesto', 'cotización', 'cotizacion', 
                                         'precios', 'budget', 'quote', 'pricing']
                
                for pattern in budget_sheet_patterns:
                    matching_sheets = [s for s in sheets if pattern.lower() in s.lower()]
                    for sheet in matching_sheets:
                        try:
                            df = pd.read_excel(file_path, sheet_name=sheet)
                            if not df.empty:
                                logger.info(f"Usando hoja con nombre relevante: {sheet}")
                                return self._process_tabular_data(df, interactive)
                        except Exception as e:
                            logger.warning(f"Error al procesar hoja {sheet}: {str(e)}")
                            continue
                
                # Opción 3: Intentar leer todas las hojas y buscar la que tenga más columnas
                # que coincidan con patrones de precios
                sheet_scores = {}
                for sheet in sheets:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet)
                        if df.empty:
                            continue
                            
                        # Calcular puntuación basada en columnas relevantes
                        score = 0
                        for col in df.columns:
                            col_lower = str(col).lower()
                            # Patrones de precios
                            if any(p in col_lower for p in ['precio', 'costo', 'valor', 'total']):
                                score += 2
                            # Patrones de actividades
                            if any(p in col_lower for p in ['actividad', 'descripcion', 'item']):
                                score += 2
                            # Patrones de cantidad
                            if any(p in col_lower for p in ['cantidad', 'qty', 'unidades']):
                                score += 1
                                
                        sheet_scores[sheet] = score
                    except Exception as e:
                        logger.warning(f"Error al evaluar hoja {sheet}: {str(e)}")
                        continue
                
                # Usar la hoja con mayor puntuación
                if sheet_scores:
                    best_sheet = max(sheet_scores.items(), key=lambda x: x[1])
                    if best_sheet[1] > 0:
                        logger.info(f"Usando hoja con mayor puntuación: {best_sheet[0]} (puntuación: {best_sheet[1]})")
                        df = pd.read_excel(file_path, sheet_name=best_sheet[0])
                        return self._process_tabular_data(df, interactive)
                
                # Si llegamos aquí, no encontramos datos válidos
                logger.error(f"No se encontraron datos válidos en ninguna hoja")
                return pd.DataFrame()
                
        except Exception as e:
            logger.exception(f"Error al procesar archivo Excel {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def extract_all_sheets(self, file_path: str, interactive: bool = False) -> List[pd.DataFrame]:
        """
        Extrae datos de todas las hojas de un archivo Excel.
        
        Args:
            file_path: Ruta al archivo Excel
            interactive: Si debe ser interactivo
            
        Returns:
            Lista de DataFrames, uno por cada hoja válida
        """
        return self._process_all_sheets(file_path, interactive)
    
    def extract_construction_budget(self, file_path: str) -> pd.DataFrame:
        """
        Extrae datos específicamente de un archivo de presupuesto de construcción.
        
        Args:
            file_path: Ruta al archivo Excel
            
        Returns:
            DataFrame con los datos extraídos
        """
        try:
            logger.info(f"Extrayendo presupuesto de construcción de {file_path}")
            
            # Importar el validador de presupuestos
            from modules.extractors.budget_validator import BudgetValidator
            
            # Leer todas las hojas del archivo
            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            
            # Filtrar hojas que parecen contener divisiones de presupuesto
            division_sheets = [sheet for sheet in sheet_names if 'div' in sheet.lower()]
            
            if not division_sheets:
                logger.warning("No se encontraron hojas de división en el archivo")
                return pd.DataFrame()
                
            # Inicializar DataFrame para almacenar resultados combinados
            combined_data = []
            
            # Procesar cada hoja de división
            for sheet in division_sheets:
                logger.info(f"Procesando hoja de división: {sheet}")
                
                try:
                    # Leer la hoja
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    
                    # Buscar fila de encabezado (generalmente contiene "Código", "Descripción", etc.)
                    header_row = None
                    for i in range(min(20, len(df))):
                        row_values = [str(val).lower() if pd.notna(val) else "" for val in df.iloc[i]]
                        if any("código" in val for val in row_values) and any("descripción" in val for val in row_values):
                            header_row = i
                            break
                    
                    if header_row is None:
                        logger.warning(f"No se encontró fila de encabezado en hoja {sheet}")
                        continue
                    
                    # Extraer nombre de la división
                    division_name = sheet
                    
                    # Usar la fila de encabezado para renombrar columnas
                    header = df.iloc[header_row]
                    df = df.iloc[header_row+1:].reset_index(drop=True)
                    
                    # Crear nuevos nombres de columnas
                    new_columns = []
                    for i, col in enumerate(header):
                        if pd.isna(col) or str(col).strip() == "":
                            new_columns.append(f"Col_{i}")
                        else:
                            new_columns.append(str(col))
                    
                    df.columns = new_columns
                    
                    # Mapear columnas a nombres estándar
                    column_map = {}
                    for col in df.columns:
                        col_lower = str(col).lower()
                        
                        if "código" in col_lower or "codigo" in col_lower:
                            column_map[col] = "codigo"
                        elif "descripción" in col_lower or "descripcion" in col_lower:
                            column_map[col] = "descripcion"
                        elif "unidad" in col_lower or "u/m" in col_lower or "um" in col_lower:
                            column_map[col] = "unidad"
                        elif "cantidad" in col_lower or "cant" in col_lower or "cant." in col_lower:
                            column_map[col] = "cantidad"
                        elif ("p.u." in col_lower or "precio unitario" in col_lower or "costo unitario" in col_lower or 
                              "precio unit" in col_lower or "p. unitario" in col_lower or "p/u" in col_lower or
                              "valor unitario" in col_lower or "val. unit" in col_lower or "val unit" in col_lower):
                            column_map[col] = "precio_unitario"
                        elif "total" in col_lower or "importe" in col_lower or "subtotal" in col_lower:
                            column_map[col] = "total"
                    
                    # Intentar detectar columnas de precio unitario por análisis de contenido
                    column_map = BudgetValidator.detect_price_unit_column(df, column_map)
                    
                    # Renombrar columnas
                    df = df.rename(columns=column_map)
                    
                    # Seleccionar columnas relevantes
                    relevant_cols = [col for col in ["codigo", "descripcion", "unidad", "cantidad", "precio_unitario", "total"] if col in df.columns]
                    if len(relevant_cols) < 2:
                        logger.warning(f"No se encontraron suficientes columnas relevantes en hoja {sheet}")
                        continue
                        
                    df = df[relevant_cols]
                    
                    # Limpiar datos
                    df = df.dropna(how='all')
                    
                    # Filtrar filas sin descripción o con descripción vacía
                    if "descripcion" in df.columns:
                        df = df.dropna(subset=["descripcion"])
                        df = df[df["descripcion"].astype(str).str.strip() != ""]
                    
                    # Añadir columna de división
                    df["division"] = division_name
                    
                    # Normalizar columnas numéricas
                    df = self.normalize_numeric_columns(df)
                    
                    # Validar y completar datos faltantes
                    df = BudgetValidator.validate_and_complete_budget_data(df)
                    
                    # Añadir al resultado combinado
                    combined_data.append(df)
                    
                except Exception as e:
                    logger.exception(f"Error al procesar hoja {sheet}: {str(e)}")
                    continue
            
            # Combinar todos los DataFrames
            if not combined_data:
                logger.warning("No se encontraron datos válidos en ninguna hoja")
                return pd.DataFrame()
                
            result = pd.concat(combined_data, ignore_index=True)
            
            # Eliminar filas duplicadas
            result = result.drop_duplicates()
            
            # Filtrar filas que parecen ser subtotales o totales
            if "descripcion" in result.columns:
                # Patrones comunes para filas de totales
                total_patterns = ["total", "subtotal", "suma", "importe total"]
                mask = ~result["descripcion"].astype(str).str.lower().str.contains("|".join(total_patterns))
                result = result[mask]
            
            # Validar consistencia del presupuesto completo
            result, validation_metadata = BudgetValidator.validate_budget_consistency(result)
            
            logger.info(f"Extracción completada. Se obtuvieron {len(result)} filas de {len(combined_data)} divisiones")
            logger.info(f"Metadatos de validación: {validation_metadata}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error al extraer presupuesto de construcción: {str(e)}")
            return pd.DataFrame()
    
    def _clean_construction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y filtra datos de construcción para eliminar filas no relevantes.
        
        Args:
            df: DataFrame a limpiar
            
        Returns:
            DataFrame limpio
        """
        try:
            # Hacer una copia para evitar warnings de SettingWithCopyWarning
            df = df.copy()
            
            # Asegurar que no haya etiquetas duplicadas en el índice
            df = df.reset_index(drop=True)
            
            # Filtrar filas que no tienen código ni total (probablemente son encabezados o notas)
            if "codigo" in df.columns and "total" in df.columns:
                codigo_na = df["codigo"].isna().values
                total_na = df["total"].isna().values
                mask = ~(codigo_na & total_na)
                df = df.loc[mask].reset_index(drop=True)
            
            # Filtrar filas donde la descripción parece ser un encabezado o nota
            if "descripcion" in df.columns:
                # Patrones comunes para encabezados o notas
                header_patterns = ["nota:", "observación:", "observacion:", "consideración:", "consideracion:"]
                desc_lower = df["descripcion"].astype(str).str.lower()
                mask = ~desc_lower.str.contains("|".join(header_patterns)).values
                df = df.loc[mask].reset_index(drop=True)
            
            # Filtrar filas donde todos los valores numéricos son 0 o NaN
            numeric_cols = [col for col in ["cantidad", "precio_unitario", "total"] if col in df.columns]
            if numeric_cols:
                # Crear máscaras individuales para cada columna
                masks = []
                for col in numeric_cols:
                    # Filas donde el valor no es NaN y no es 0
                    masks.append((df[col].notna() & (df[col] != 0)).values)
                
                # Combinar las máscaras con OR
                if masks:
                    combined_mask = masks[0]
                    for mask in masks[1:]:
                        combined_mask = combined_mask | mask
                    
                    df = df.loc[combined_mask].reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.exception(f"Error al limpiar datos de construcción: {str(e)}")
            return df
    
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
            
            # 2. Buscar filas de encabezado
            header_row = self._find_header_row(df)
            if header_row is not None:
                # Usar la fila identificada como encabezado
                logger.info(f"Usando fila {header_row} como encabezado")
                # Convertir los valores de la fila de encabezado a strings
                new_columns = [str(val) if pd.notna(val) else f"Col_{i}" for i, val in enumerate(df.iloc[header_row])]
                # Crear un nuevo DataFrame con los nuevos encabezados
                df = df.iloc[header_row+1:].reset_index(drop=True)
                df.columns = new_columns
            
            # 3. Detectar columnas relevantes
            column_map = self._map_columns(df)
            
            # Si no se encontraron columnas suficientes, intentar buscar en las filas
            if len(column_map) < 2:
                logger.info("Pocas columnas mapeadas, intentando buscar en filas")
                # Buscar filas que contengan palabras clave
                keywords = ['código', 'descripción', 'descripcion', 'costo', 'precio', 'total', 
                           'cantidad', 'unidad', 'importe']
                
                # Convertir el DataFrame a string para buscar
                df_str = df.head(20).astype(str).apply(lambda x: x.str.lower())
                
                # Buscar filas con palabras clave
                for idx, row in df_str.iterrows():
                    keyword_count = 0
                    for cell in row:
                        if any(kw in str(cell).lower() for kw in keywords):
                            keyword_count += 1
                    
                    if keyword_count >= 2:
                        # Usar esta fila como encabezado
                        logger.info(f"Usando fila {idx} como encabezado (contiene {keyword_count} palabras clave)")
                        # Convertir los valores de la fila de encabezado a strings
                        new_columns = [str(val) if pd.notna(val) else f"Col_{i}" for i, val in enumerate(df.iloc[idx])]
                        # Crear un nuevo DataFrame con los nuevos encabezados
                        df = df.iloc[idx+1:].reset_index(drop=True)
                        df.columns = new_columns
                        # Volver a mapear columnas
                        column_map = self._map_columns(df)
                        break
            
            # 4. Renombrar columnas
            df = df.rename(columns=column_map)
            
            # 5. Seleccionar solo columnas necesarias
            required_cols = ['actividades', 'costo_unitario', 'cantidad', 'costo_total']
            available_cols = [col for col in required_cols if col in df.columns]
            
            # Si no tenemos suficientes columnas, no podemos procesar
            if len(available_cols) < 2:
                logger.warning(f"No se encontraron suficientes columnas requeridas. Solo disponibles: {available_cols}")
                return pd.DataFrame()
                
            df = df[available_cols]
            
            # 6. Normalizar columnas numéricas
            for col in ['costo_unitario', 'cantidad', 'costo_total']:
                if col in df.columns:
                    df[col] = self._normalize_price_column(df[col])
            
            # 7. Calcular columnas faltantes si es posible
            if 'costo_unitario' in df.columns and 'cantidad' in df.columns and 'costo_total' not in df.columns:
                df['costo_total'] = df['costo_unitario'] * df['cantidad']
                
            if 'costo_total' in df.columns and 'cantidad' in df.columns and 'costo_unitario' not in df.columns:
                # Evitar división por cero
                df['costo_unitario'] = df.apply(
                    lambda row: row['costo_total'] / row['cantidad'] if row['cantidad'] and row['cantidad'] != 0 else 0, 
                    axis=1
                )
                
            # 8. Filtrar filas sin actividad o sin precios
            df = df.dropna(subset=['actividades'])
            df = df[df['actividades'].astype(str).str.strip() != '']
            
            # Filtrar filas donde todos los valores numéricos son cero o nulos
            numeric_cols = ['costo_unitario', 'cantidad', 'costo_total']
            available_numeric = [col for col in numeric_cols if col in df.columns]
            if available_numeric:
                df = df[(df[available_numeric].notna().any(axis=1)) & 
                        (df[available_numeric] != 0).any(axis=1)]
            
            return df
            
        except Exception as e:
            logger.exception(f"Error al procesar datos tabulares: {str(e)}")
            return pd.DataFrame()
            
    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """
        Encuentra la fila que probablemente sea el encabezado.
        
        Args:
            df: DataFrame a analizar
            
        Returns:
            Índice de la fila de encabezado o None si no se encuentra
        """
        # Patrones para identificar encabezados
        header_patterns = [
            'descripcion', 'descripción', 'concepto', 'actividad', 
            'precio', 'costo', 'unitario', 'total', 'cantidad', 'unidad'
        ]
        
        # Convertir a string para buscar
        df_str = df.head(20).astype(str).apply(lambda x: x.str.lower())
        
        best_row = None
        max_matches = 0
        
        # Buscar en cada fila
        for idx, row in df_str.iterrows():
            # Contar coincidencias de patrones en la fila
            matches = 0
            for pattern in header_patterns:
                for cell in row:
                    if isinstance(cell, str) and pattern in cell:
                        matches += 1
            
            if matches > max_matches:
                max_matches = matches
                best_row = idx
        
        # Requerir al menos 2 coincidencias
        if max_matches >= 2:
            return best_row
            
        return None
    
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
            
        # Verificar si hay columnas sin nombre (comunes en archivos de presupuestos)
        unnamed_cols = [col for col in df.columns if 'Unnamed:' in str(col)]
        if len(unnamed_cols) > 5:  # Si hay muchas columnas sin nombre, podría ser un archivo de presupuesto
            # Buscar filas con palabras clave de presupuestos en las primeras 20 filas
            keywords = ['código', 'descripción', 'descripcion', 'costo', 'precio', 'total', 
                       'cantidad', 'unidad', 'importe', 'presupuesto', 'partida']
            
            # Convertir el DataFrame a string para buscar palabras clave
            df_str = df.head(20).astype(str).apply(lambda x: x.str.lower())
            
            # Verificar si alguna celda contiene alguna de las palabras clave
            for keyword in keywords:
                for col in df_str.columns:
                    if df_str[col].str.contains(keyword).any():
                        logger.info(f"Encontrada palabra clave '{keyword}' en columna {col}")
                        return True
        
        # Verificar si hay al menos una columna que parezca de precios
        price_patterns = ['precio', 'costo', 'valor', 'total', 'unitario', 'rate', 'importe']
        has_price_col = any(
            any(pattern in str(col).lower() for pattern in price_patterns)
            for col in df.columns
        )
        
        # Si no hay columnas con nombres de precios, buscar en las primeras filas
        if not has_price_col:
            # Convertir el DataFrame a string para buscar patrones
            df_str = df.head(10).astype(str).apply(lambda x: x.str.lower())
            for pattern in price_patterns:
                for col in df_str.columns:
                    if df_str[col].str.contains(pattern).any():
                        has_price_col = True
                        break
                if has_price_col:
                    break
        
        # Verificar si hay al menos una columna que parezca de actividades
        activity_patterns = ['actividad', 'descripcion', 'descripción', 'item', 'concepto', 'material', 'partida', 'código', 'codigo']
        has_activity_col = any(
            any(pattern in str(col).lower() for pattern in activity_patterns)
            for col in df.columns
        )
        
        # Si no hay columnas con nombres de actividades, buscar en las primeras filas
        if not has_activity_col:
            # Convertir el DataFrame a string para buscar patrones
            df_str = df.head(10).astype(str).apply(lambda x: x.str.lower())
            for pattern in activity_patterns:
                for col in df_str.columns:
                    if df_str[col].str.contains(pattern).any():
                        has_activity_col = True
                        break
                if has_activity_col:
                    break
        
        # Si encontramos tanto columnas de precio como de actividad, es válido
        if has_price_col and has_activity_col:
            return True
            
        # Verificar si hay columnas numéricas (posibles precios)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2 and len(df) > 5:
            # Si hay al menos dos columnas numéricas y más de 5 filas, podría ser válido
            return True
            
        return False
    
    def _map_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Mapea las columnas del DataFrame a nombres estandarizados.
        
        Args:
            df: DataFrame con columnas a mapear
            
        Returns:
            Diccionario con mapeo de nombres originales a estandarizados
        """
        column_map = {}
        
        # Convertir nombres de columnas a minúsculas para comparación
        columns_lower = {col: str(col).lower() for col in df.columns}
        
        # Mapeo para columna de actividades/descripción
        activity_patterns = [
            'descripcion', 'descripción', 'concepto', 'actividad', 'partida', 
            'item', 'material', 'servicio', 'trabajo', 'descripción del concepto'
        ]
        for col, col_lower in columns_lower.items():
            if any(pattern in col_lower for pattern in activity_patterns):
                column_map[col] = 'actividades'
                break
                
        # Si no encontramos columna de actividades por nombre, buscar por contenido
        if 'actividades' not in column_map.values():
            # Buscar columnas que contengan texto largo (posibles descripciones)
            for col in df.columns:
                try:
                    # Verificar si la columna contiene principalmente texto
                    if df[col].dtype == 'object':
                        # Calcular longitud promedio de texto
                        avg_len = df[col].astype(str).str.len().mean()
                        # Si el promedio es mayor a 15 caracteres, probablemente sea descripción
                        if avg_len > 15:
                            column_map[col] = 'actividades'
                            logger.info(f"Columna {col} identificada como 'actividades' por longitud de texto")
                            break
                except Exception as e:
                    logger.debug(f"Error al analizar columna {col}: {str(e)}")
                    continue
        
        # Mapeo para columna de costo unitario
        unit_price_patterns = [
            'precio unitario', 'costo unitario', 'p.u.', 'precio', 'costo', 
            'unit price', 'rate', 'valor unitario', 'precio/unidad'
        ]
        for col, col_lower in columns_lower.items():
            if any(pattern in col_lower for pattern in unit_price_patterns):
                column_map[col] = 'costo_unitario'
                break
        
        # Mapeo para columna de cantidad
        quantity_patterns = [
            'cantidad', 'qty', 'quantity', 'vol', 'volumen', 'unidades', 
            'count', 'cant.', 'cant', 'volume'
        ]
        for col, col_lower in columns_lower.items():
            if any(pattern in col_lower for pattern in quantity_patterns):
                column_map[col] = 'cantidad'
                break
        
        # Mapeo para columna de costo total
        total_price_patterns = [
            'total', 'importe', 'monto', 'precio total', 'costo total', 
            'subtotal', 'amount', 'sum', 'valor total', 'importe total'
        ]
        for col, col_lower in columns_lower.items():
            if any(pattern in col_lower for pattern in total_price_patterns):
                column_map[col] = 'costo_total'
                break
        
        # Si no encontramos columnas por nombre, intentar identificar por tipo de datos
        if 'costo_unitario' not in column_map.values() and 'costo_total' not in column_map.values():
            try:
                # Buscar columnas numéricas que puedan ser precios
                numeric_cols = []
                for col in df.columns:
                    try:
                        # Intentar convertir a numérico
                        if pd.to_numeric(df[col], errors='coerce').notna().any():
                            numeric_cols.append(col)
                    except:
                        pass
                
                if len(numeric_cols) >= 2:
                    # Si hay al menos dos columnas numéricas, la primera podría ser cantidad
                    # y la segunda podría ser precio unitario o total
                    if 'cantidad' not in column_map.values() and len(numeric_cols) > 0:
                        column_map[numeric_cols[0]] = 'cantidad'
                        
                    if 'costo_unitario' not in column_map.values() and len(numeric_cols) > 1:
                        column_map[numeric_cols[1]] = 'costo_unitario'
                        
                    if 'costo_total' not in column_map.values() and len(numeric_cols) > 2:
                        column_map[numeric_cols[2]] = 'costo_total'
            except Exception as e:
                logger.debug(f"Error al identificar columnas numéricas: {str(e)}")
        
        logger.info(f"Mapeo de columnas: {column_map}")
        return column_map
    
    def _normalize_price_column(self, series: pd.Series) -> pd.Series:
        """
        Normaliza una columna de precios para asegurar que todos los valores sean numéricos.
        
        Args:
            series: Serie de pandas con valores a normalizar
            
        Returns:
            Serie normalizada
        """
        try:
            # Verificar que sea una Serie y no un DataFrame
            if isinstance(series, pd.DataFrame):
                logger.warning("Se recibió un DataFrame en lugar de una Serie. Intentando convertir.")
                if len(series.columns) == 1:
                    series = series.iloc[:, 0]
                else:
                    logger.error("No se puede normalizar un DataFrame con múltiples columnas como Serie")
                    return pd.Series([np.nan] * len(series))
            
            # Convertir a string primero para manejar diferentes tipos de datos
            series = series.astype(str)
            
            # Reemplazar comas por puntos y eliminar símbolos de moneda
            series = series.str.replace(',', '.', regex=False)
            series = series.str.replace('$', '', regex=False)
            series = series.str.replace('€', '', regex=False)
            series = series.str.replace('£', '', regex=False)
            series = series.str.replace('¥', '', regex=False)
            
            # Eliminar espacios y otros caracteres no numéricos
            series = series.str.replace(r'[^\d.-]', '', regex=True)
            
            # Convertir a float con manejo de errores
            return pd.to_numeric(series, errors='coerce')
            
        except Exception as e:
            logger.exception(f"Error al normalizar columna de precios: {str(e)}")
            # En caso de error, intentar una conversión directa
            try:
                return pd.to_numeric(series, errors='coerce')
            except:
                # Si todo falla, devolver NaN
                if hasattr(series, '__len__'):
                    return pd.Series([np.nan] * len(series))
                else:
                    return pd.Series([np.nan])
    
    def _process_all_sheets(self, file_path: str, interactive: bool = False) -> pd.DataFrame:
        """
        Procesa todas las hojas de un workbook y combina los resultados.
        
        Args:
            file_path: Ruta al archivo Excel
            interactive: Si debe ser interactivo
            
        Returns:
            DataFrame combinado con datos de todas las hojas válidas
        """
        try:
            xls = pd.ExcelFile(file_path)
            sheets = xls.sheet_names
            
            all_data = []
            for sheet in sheets:
                try:
                    logger.info(f"Procesando hoja: {sheet}")
                    df = pd.read_excel(file_path, sheet_name=sheet)
                    
                    # Verificar si la hoja tiene datos válidos
                    if not df.empty and self._is_valid_data(df):
                        # Añadir columna con nombre de la hoja para referencia
                        processed = self._process_tabular_data(df, interactive)
                        if not processed.empty:
                            processed['hoja_origen'] = sheet
                            all_data.append(processed)
                except Exception as e:
                    logger.warning(f"Error procesando hoja {sheet}: {str(e)}")
                    continue
            
            if all_data:
                # Combinar todos los DataFrames
                result = pd.concat(all_data, ignore_index=True)
                logger.info(f"Combinados datos de {len(all_data)} hojas, total de {len(result)} filas")
                return result
            else:
                logger.warning("No se encontraron datos válidos en ninguna hoja")
                return pd.DataFrame()
                
        except Exception as e:
            logger.exception(f"Error al procesar todas las hojas: {str(e)}")
            return pd.DataFrame()

    def normalize_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza las columnas numéricas de un DataFrame.
        
        Args:
            df: DataFrame con los datos a normalizar
            
        Returns:
            DataFrame con las columnas numéricas normalizadas
        """
        try:
            # Si el DataFrame está vacío, devolverlo sin cambios
            if df.empty:
                return df
                
            # Hacer una copia para evitar modificar el original
            df = df.copy()
            
            # Columnas que deberían ser numéricas
            numeric_columns = ["cantidad", "precio_unitario", "total"]
            
            for col in df.columns:
                # Solo procesar columnas existentes
                if col in numeric_columns:
                    try:
                        # Guardar la columna original para comparar después
                        orig_values = df[col].copy()
                        
                        # Intentar convertir directamente
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Contar valores NaN después de la conversión
                        nan_count = df[col].isna().sum()
                        if nan_count > 0 and len(df) > 0:
                            logger.warning(
                                f"Columna '{col}': {nan_count}/{len(df)} valores no pudieron ser convertidos a numérico"
                            )
                            
                            # Intentar limpiar caracteres especiales comunes
                            str_values = orig_values.astype(str)
                            cleaned = (str_values.str.replace(r'[$€£¥]', '', regex=True)   # Símbolos de moneda
                                                .str.replace(',', '.', regex=False)        # Comas como decimales
                                                .str.replace(r'\s+', '', regex=True)       # Espacios
                                                .str.strip())                              # Espacios al inicio/fin
                            
                            # Intentar convertir de nuevo
                            df[col] = pd.to_numeric(cleaned, errors='coerce')
                            
                            # Verificar si mejoramos
                            new_nan_count = df[col].isna().sum()
                            if new_nan_count < nan_count:
                                logger.info(f"Limpieza de datos mejoró conversión: {nan_count} -> {new_nan_count} valores nulos")
                            else:
                                # No hubo mejora, volver a valores originales
                                df[col] = orig_values
                    except Exception as e:
                        logger.exception(f"Error al normalizar columna '{col}': {str(e)}")
                        # Mantener el valor original en caso de error
                        pass
            
            return df
            
        except Exception as e:
            logger.exception(f"Error al normalizar columnas numéricas: {str(e)}")
            return df  # Devolver df original en caso de error

    def export_to_csv(self, df: pd.DataFrame, output_path: str, encoding: str = 'utf-8') -> bool:
        """
        Exporta un DataFrame a un archivo CSV.
        
        Args:
            df: DataFrame a exportar
            output_path: Ruta donde guardar el archivo CSV
            encoding: Codificación del archivo (por defecto utf-8)
            
        Returns:
            True si la exportación fue exitosa, False en caso contrario
        """
        try:
            if df.empty:
                logger.warning("No hay datos para exportar")
                return False
                
            # Asegurar que el directorio exista
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Exportar a CSV
            df.to_csv(output_path, index=False, encoding=encoding)
            
            logger.info(f"Datos exportados exitosamente a {output_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Error al exportar datos a CSV: {str(e)}")
            return False

    def export_to_excel(self, df: pd.DataFrame, output_path: str) -> bool:
        """
        Exporta un DataFrame a un archivo Excel con formato.
        
        Args:
            df: DataFrame a exportar
            output_path: Ruta donde guardar el archivo Excel
            
        Returns:
            True si la exportación fue exitosa, False en caso contrario
        """
        try:
            if df.empty:
                logger.warning("No hay datos para exportar")
                return False
                
            # Asegurar que el directorio exista
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Crear un escritor de Excel
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
            
            # Convertir a Excel
            df.to_excel(writer, sheet_name='Presupuesto', index=False)
            
            # Obtener el libro y la hoja de trabajo
            workbook = writer.book
            worksheet = writer.sheets['Presupuesto']
            
            # Definir formatos
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })
            
            numeric_format = workbook.add_format({
                'num_format': '#,##0.00',
                'border': 1
            })
            
            text_format = workbook.add_format({
                'text_wrap': True,
                'border': 1
            })
            
            # Aplicar formato a la fila de encabezado
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Establecer el ancho de las columnas
            worksheet.set_column('A:A', 15)  # Código
            worksheet.set_column('B:B', 40)  # Descripción
            worksheet.set_column('C:C', 10)  # Unidad
            worksheet.set_column('D:D', 12, numeric_format)  # Cantidad
            worksheet.set_column('E:E', 15, numeric_format)  # Total
            worksheet.set_column('F:F', 12)  # División
            
            # Aplicar formato a las celdas de texto
            for row_num in range(1, len(df) + 1):
                worksheet.write(row_num, 1, df.iloc[row_num-1]['descripcion'] if 'descripcion' in df.columns else '', text_format)
            
            # Guardar el archivo
            writer.close()
            
            logger.info(f"Datos exportados exitosamente a {output_path}")
            return True
            
        except Exception as e:
            logger.exception(f"Error al exportar datos a Excel: {str(e)}")
            return False

    def _validate_and_complete_budget_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Valida y completa datos faltantes en un presupuesto.
        
        Args:
            df: DataFrame con datos de presupuesto
            
        Returns:
            DataFrame con datos validados y completados
        """
        try:
            # Hacer una copia para evitar modificar el original
            df = df.copy()
            
            # Verificar si tenemos las columnas necesarias
            has_cantidad = "cantidad" in df.columns
            has_precio_unitario = "precio_unitario" in df.columns
            has_total = "total" in df.columns
            
            # Si tenemos cantidad y precio unitario pero no total, calcularlo
            if has_cantidad and has_precio_unitario and not has_total:
                df["total"] = df["cantidad"] * df["precio_unitario"]
                logger.info("Calculados valores de 'total' basados en cantidad * precio_unitario")
                
            # Si tenemos cantidad y total pero no precio unitario, calcularlo
            elif has_cantidad and has_total and not has_precio_unitario:
                # Evitar división por cero
                mask = df["cantidad"] > 0
                df.loc[mask, "precio_unitario"] = df.loc[mask, "total"] / df.loc[mask, "cantidad"]
                logger.info("Calculados valores de 'precio_unitario' basados en total / cantidad")
                
            # Si tenemos precio unitario y total pero no cantidad, calcularlo
            elif has_precio_unitario and has_total and not has_cantidad:
                # Evitar división por cero
                mask = df["precio_unitario"] > 0
                df.loc[mask, "cantidad"] = df.loc[mask, "total"] / df.loc[mask, "precio_unitario"]
                logger.info("Calculados valores de 'cantidad' basados en total / precio_unitario")
            
            return df
            
        except Exception as e:
            logger.exception(f"Error al validar y completar datos de presupuesto: {str(e)}")
            return df
