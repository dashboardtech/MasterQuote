"""
Módulo para validación de datos de presupuestos de construcción.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class BudgetValidator:
    """
    Clase para validar y completar datos de presupuestos de construcción.
    """
    
    @staticmethod
    def validate_and_complete_budget_data(df: pd.DataFrame) -> pd.DataFrame:
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
            
            # Validar consistencia de datos
            if has_cantidad and has_precio_unitario and has_total:
                # Calcular el total esperado
                df["total_calculado"] = df["cantidad"] * df["precio_unitario"]
                
                # Identificar discrepancias significativas (más del 1%)
                df["discrepancia"] = abs(df["total"] - df["total_calculado"]) / df["total"].replace(0, np.nan)
                inconsistencias = df[df["discrepancia"] > 0.01].dropna(subset=["discrepancia"])
                
                if len(inconsistencias) > 0:
                    logger.warning(f"Se encontraron {len(inconsistencias)} filas con inconsistencias entre cantidad, precio unitario y total")
                    
                    # Agregar columna de validación
                    df["validado"] = True
                    df.loc[df["discrepancia"] > 0.01, "validado"] = False
                else:
                    df["validado"] = True
                
                # Eliminar columnas temporales
                df = df.drop(columns=["total_calculado", "discrepancia"], errors="ignore")
            
            return df
            
        except Exception as e:
            logger.exception(f"Error al validar y completar datos de presupuesto: {str(e)}")
            return df
    
    @staticmethod
    def detect_price_unit_column(df: pd.DataFrame, column_map: Dict[str, str]) -> Dict[str, str]:
        """
        Detecta columnas de precio unitario por análisis de contenido.
        
        Args:
            df: DataFrame con datos de presupuesto
            column_map: Mapa de columnas actual
            
        Returns:
            Mapa de columnas actualizado
        """
        try:
            # Si ya tenemos precio unitario, no hacer nada
            if "precio_unitario" in column_map.values():
                return column_map
            
            # Buscar columnas numéricas que podrían ser precios unitarios
            numeric_cols = []
            for col in df.columns:
                if col not in column_map.values():
                    # Verificar si la columna contiene principalmente valores numéricos
                    try:
                        numeric_values = pd.to_numeric(df[col], errors='coerce')
                        # Verificar que no sean todos NaN
                        if numeric_values.notna().sum() > 0:
                            # Si más del 50% son números y no son todos 0 o 1 (posibles flags)
                            if (numeric_values.notna().mean() > 0.5 and 
                                len(numeric_values.unique()) > 2):
                                numeric_cols.append(col)
                    except Exception as e:
                        logger.debug(f"Error al analizar columna {col} como numérica: {str(e)}")
                        continue
            
            # Si hay columnas numéricas y tenemos cantidad y total, buscar precio unitario por relación matemática
            if numeric_cols and "cantidad" in column_map.values() and "total" in column_map.values():
                try:
                    # Buscar la columna que podría ser precio unitario basado en la relación total = cantidad * precio_unitario
                    cantidad_col = [k for k, v in column_map.items() if v == "cantidad"][0]
                    total_col = [k for k, v in column_map.items() if v == "total"][0]
                    
                    best_col = None
                    best_diff = 1.0  # Iniciar con el peor valor posible
                    
                    for num_col in numeric_cols:
                        # Convertir a numérico para hacer cálculos
                        df_temp = df.copy()
                        for c in [cantidad_col, total_col, num_col]:
                            df_temp[c] = pd.to_numeric(df_temp[c], errors='coerce')
                        
                        # Calcular el producto cantidad * precio_unitario
                        df_temp['calc_total'] = df_temp[cantidad_col] * df_temp[num_col]
                        
                        # Comparar con el total real, filtrando NaNs y valores 0 que podrían causar problemas
                        valid_rows = (df_temp[total_col] > 0) & (df_temp['calc_total'] > 0)
                        df_valid = df_temp[valid_rows]
                        
                        if len(df_valid) > 0:
                            # Calcular la diferencia relativa
                            df_valid['diff'] = abs(df_valid[total_col] - df_valid['calc_total']) / df_valid[total_col]
                            mean_diff = df_valid['diff'].mean()
                            
                            # Actualizar si encontramos un mejor candidato
                            if mean_diff < best_diff:
                                best_diff = mean_diff
                                best_col = num_col
                    
                    # Si la mejor diferencia es menor al 10%, es probable que sea precio unitario
                    if best_col is not None and best_diff < 0.1:
                        column_map[best_col] = "precio_unitario"
                        logger.info(f"Detectada columna de precio unitario '{best_col}' por análisis de contenido (diferencia: {best_diff:.4f})")
                except Exception as e:
                    logger.warning(f"Error al analizar relación matemática para precio unitario: {str(e)}")
            
            # Si aún no tenemos precio_unitario pero tenemos columnas numéricas, usar heurísticas
            if "precio_unitario" not in column_map.values() and numeric_cols:
                # Heurística: Si hay una columna con valores numéricos entre cantidad y total, podría ser precio unitario
                if "cantidad" in column_map.values() and "total" in column_map.values():
                    try:
                        cantidad_col = [k for k, v in column_map.items() if v == "cantidad"][0]
                        total_col = [k for k, v in column_map.items() if v == "total"][0]
                        
                        # Calcular medias para comparar magnitudes
                        df_temp = df.copy()
                        for c in [cantidad_col, total_col] + numeric_cols:
                            df_temp[c] = pd.to_numeric(df_temp[c], errors='coerce')
                        
                        cant_mean = df_temp[cantidad_col].mean()
                        total_mean = df_temp[total_col].mean()
                        
                        # El precio unitario debería estar entre cantidad y total en magnitud
                        for num_col in numeric_cols:
                            col_mean = df_temp[num_col].mean()
                            
                            # Si la media está entre cantidad y total (o aproximadamente)
                            if ((cant_mean < col_mean < total_mean) or 
                                (abs(total_mean / (col_mean * cant_mean) - 1) < 0.5)):  # Verificar que total ≈ cantidad * col
                                column_map[num_col] = "precio_unitario"
                                logger.info(f"Detectada columna de precio unitario '{num_col}' por análisis de magnitudes")
                                break
                    except Exception as e:
                        logger.warning(f"Error al utilizar heurística de magnitudes: {str(e)}")
            
            return column_map
            
        except Exception as e:
            logger.exception(f"Error al detectar columna de precio unitario: {str(e)}")
            return column_map
    
    @staticmethod
    def validate_budget_consistency(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Valida la consistencia de un presupuesto y genera metadatos de validación.
        
        Args:
            df: DataFrame con datos de presupuesto
            
        Returns:
            Tuple con (DataFrame validado, metadatos de validación)
        """
        try:
            # Hacer una copia para evitar modificar el original
            df = df.copy()
            
            # Inicializar metadatos
            metadata = {
                "total_filas": len(df),
                "filas_validadas": 0,
                "filas_con_errores": 0,
                "errores_por_tipo": {},
                "confianza": 1.0
            }
            
            # Verificar si el DataFrame está vacío
            if df.empty:
                metadata["errores_por_tipo"]["dataframe_vacio"] = True
                metadata["confianza"] = 0.0
                return df, metadata
            
            # Verificar columnas requeridas
            columnas_requeridas = ["codigo", "descripcion", "unidad", "cantidad", "precio_unitario", "total"]
            columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
            
            if columnas_faltantes:
                metadata["errores_por_tipo"]["columnas_faltantes"] = columnas_faltantes
                # Reducir confianza basado en la cantidad de columnas faltantes
                metadata["confianza"] *= max(0.1, 1.0 - (len(columnas_faltantes) / len(columnas_requeridas) * 0.9))
            
            # Verificar valores nulos
            for col in df.columns:
                nulos = df[col].isna().sum()
                if nulos > 0:
                    if "valores_nulos" not in metadata["errores_por_tipo"]:
                        metadata["errores_por_tipo"]["valores_nulos"] = {}
                    metadata["errores_por_tipo"]["valores_nulos"][col] = nulos
                    
                    # Reducir confianza basado en porcentaje de nulos
                    if len(df) > 0:  # Evitar división por cero
                        metadata["confianza"] *= (1 - (nulos / len(df) * 0.5))
            
            # Verificar consistencia de cálculos
            if all(col in df.columns for col in ["cantidad", "precio_unitario", "total"]):
                try:
                    # Asegurar que las columnas sean numéricas
                    for col in ["cantidad", "precio_unitario", "total"]:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Calcular total esperado
                    df["total_esperado"] = df["cantidad"] * df["precio_unitario"]
                    
                    # Calcular discrepancia (solo para filas con valores válidos)
                    valid_rows = (df["total"].notna() & df["total_esperado"].notna() & 
                                  (df["total"] != 0) & (df["total_esperado"] != 0))
                    
                    if valid_rows.sum() > 0:
                        # Calcular solo para filas con valores válidos
                        df.loc[valid_rows, "discrepancia"] = abs(
                            df.loc[valid_rows, "total"] - df.loc[valid_rows, "total_esperado"]
                        ) / df.loc[valid_rows, "total"]
                        
                        # Marcar filas con discrepancias significativas
                        df["validado"] = False  # Inicializar en False
                        df.loc[valid_rows, "validado"] = df.loc[valid_rows, "discrepancia"] <= 0.01
                        
                        # Para filas sin valores válidos, marcarlas como no validadas
                        df.loc[~valid_rows, "validado"] = False
                        
                        # Contar filas con errores
                        filas_con_errores = (~df["validado"]).sum()
                        metadata["filas_con_errores"] = int(filas_con_errores)
                        metadata["filas_validadas"] = len(df) - filas_con_errores
                        
                        # Reducir confianza basado en porcentaje de errores
                        if len(df) > 0:
                            metadata["confianza"] *= max(0.1, (1 - (filas_con_errores / len(df) * 0.7)))
                        
                        # Estadísticas adicionales para depuración
                        metadata["estadisticas"] = {
                            "discrepancia_media": float(df.loc[valid_rows, "discrepancia"].mean()),
                            "discrepancia_max": float(df.loc[valid_rows, "discrepancia"].max()),
                            "porcentaje_filas_invalidas": float(filas_con_errores / len(df) * 100)
                        }
                    else:
                        # No hay filas válidas para validar
                        metadata["errores_por_tipo"]["sin_datos_para_validar"] = True
                        metadata["confianza"] *= 0.5
                    
                    # Eliminar columnas temporales
                    df = df.drop(columns=["total_esperado", "discrepancia"], errors="ignore")
                    
                except Exception as e:
                    logger.warning(f"Error al validar cálculos de presupuesto: {str(e)}")
                    metadata["errores_por_tipo"]["error_calculo"] = str(e)
                    metadata["confianza"] *= 0.6
            else:
                # No podemos validar los cálculos
                metadata["errores_por_tipo"]["no_se_puede_validar_calculos"] = True
                metadata["confianza"] *= 0.7
            
            # Asegurar que la confianza esté entre 0 y 1
            metadata["confianza"] = max(0.0, min(1.0, metadata["confianza"]))
            
            return df, metadata
            
        except Exception as e:
            logger.exception(f"Error al validar consistencia del presupuesto: {str(e)}")
            return df, {"error": str(e), "confianza": 0.0}
