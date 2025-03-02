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
                # Usar infer_objects para evitar el FutureWarning
                total_replaced = df["total"].replace(0, np.nan).infer_objects(copy=False)
                df["discrepancia"] = abs(df["total"] - df["total_calculado"]) / total_replaced
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
    
    @classmethod
    def validate_budget_consistency(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Valida la consistencia de los datos de presupuesto.
        
        Implementa un sistema de validación parcial que permite:
        1. Validación completa: cantidad * precio_unitario ≈ total
        2. Validación parcial: filas con descripción y al menos dos valores numéricos
        3. Validación de subtotales y agrupaciones
        
        Args:
            df: DataFrame con los datos de presupuesto
            
        Returns:
            Tuple con:
            - DataFrame con columna de validación añadida
            - Diccionario con metadatos de validación
        """
        try:
            # Hacer una copia para no modificar el original
            df = df.copy()
            
            # Inicializar columnas de validación
            df["valid"] = False
            df["validation_type"] = "none"
            df["error_type"] = None
            
            # Inicializar metadatos
            metadata = {
                "total_rows": len(df),
                "valid_rows": 0,
                "valid_percentage": 0.0,
                "validation_types": {},
                "error_types": {}
            }
            
            # Verificar que tenemos las columnas necesarias
            required_cols = ["cantidad", "precio_unitario", "total", "descripcion"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"Faltan columnas requeridas para validación: {missing_cols}")
                metadata["error"] = f"Faltan columnas requeridas: {missing_cols}"
                return df, metadata
            
            # Paso 1: Calcular total esperado (cantidad * precio_unitario)
            df["total_esperado"] = df["cantidad"] * df["precio_unitario"]
            
            # Paso 2: Validación completa (cantidad * precio_unitario ≈ total)
            try:
                # Umbral de discrepancia aceptable (5%)
                threshold = 0.05
                
                # Calcular discrepancia solo para filas con valores numéricos válidos
                mask_full = (
                    df["cantidad"].notna() & 
                    df["precio_unitario"].notna() & 
                    df["total"].notna() &
                    df["total_esperado"].notna() &
                    (df["cantidad"] != 0) & 
                    (df["precio_unitario"] != 0) & 
                    (df["total"] != 0)
                )
                
                if mask_full.any():
                    # Calcular discrepancia como porcentaje del total
                    df.loc[mask_full, "discrepancia"] = (
                        (df.loc[mask_full, "total"] - df.loc[mask_full, "total_esperado"]).abs() / 
                        df.loc[mask_full, "total"]
                    )
                    
                    # Marcar como válidas las filas con discrepancia menor al umbral
                    full_valid = mask_full & (df["discrepancia"] <= threshold)
                    
                    if full_valid.any():
                        df.loc[full_valid, "valid"] = True
                        df.loc[full_valid, "validation_type"] = "full"
                        
                        # Registrar estadísticas
                        full_valid_count = full_valid.sum()
                        metadata["validation_types"]["full"] = full_valid_count
                        
                        logger.info(f"Validación completa: {full_valid_count} filas válidas de {mask_full.sum()} con datos completos")
                    
                    # Marcar error para filas con discrepancia mayor al umbral
                    invalid_discrepancy = mask_full & (df["discrepancia"] > threshold)
                    if invalid_discrepancy.any():
                        df.loc[invalid_discrepancy, "error_type"] = "high_discrepancy"
                        metadata["error_types"]["high_discrepancy"] = invalid_discrepancy.sum()
            except Exception as e:
                logger.exception(f"Error en validación completa: {str(e)}")
            
            # Paso 3: Validación parcial para filas con descripción y al menos dos valores numéricos
            try:
                # Identificar filas no validadas previamente
                not_validated = ~df["valid"]
                
                # Filas con descripción no vacía
                has_description = df["descripcion"].notna() & (df["descripcion"].astype(str).str.strip() != "")
                
                # Contar valores numéricos válidos por fila
                numeric_cols = ["cantidad", "precio_unitario", "total"]
                valid_numeric_count = df[numeric_cols].notna().sum(axis=1)
                
                # Criterios para validación parcial
                partial_criteria = {
                    # Caso 1: Tiene descripción y exactamente dos valores numéricos (el tercero se puede calcular)
                    "calculable": not_validated & has_description & (valid_numeric_count == 2),
                    
                    # Caso 2: Tiene descripción y un valor numérico, pero parece ser un subtotal o agrupación
                    "grouping": not_validated & has_description & (valid_numeric_count == 1) & 
                                df["descripcion"].astype(str).str.lower().str.contains(
                                    r'total|subtotal|suma|capítulo|partida|grupo', regex=True
                                ),
                    
                    # Caso 3: Tiene descripción y tres valores, pero no cumplen la relación matemática
                    # (posiblemente por redondeo o factores adicionales)
                    "inconsistent": not_validated & has_description & (valid_numeric_count == 3) & 
                                   df["total"].notna() & df["total_esperado"].notna()
                }
                
                # Aplicar validación parcial para cada criterio
                for validation_type, mask in partial_criteria.items():
                    if mask.any():
                        df.loc[mask, "valid"] = True
                        df.loc[mask, "validation_type"] = validation_type
                        
                        # Registrar estadísticas
                        count = mask.sum()
                        metadata["validation_types"][validation_type] = count
                        logger.info(f"Validación parcial ({validation_type}): {count} filas")
                
                # Paso 3.1: Para filas calculables, completar el valor faltante
                calculable = partial_criteria["calculable"]
                if calculable.any():
                    # Identificar qué valor falta y calcularlo
                    for _, row in df[calculable].iterrows():
                        idx = row.name
                        if pd.isna(row["cantidad"]) and pd.notna(row["precio_unitario"]) and pd.notna(row["total"]):
                            # Calcular cantidad
                            if row["precio_unitario"] != 0:
                                df.loc[idx, "cantidad"] = row["total"] / row["precio_unitario"]
                                df.loc[idx, "error_type"] = "calculated_quantity"
                        elif pd.isna(row["precio_unitario"]) and pd.notna(row["cantidad"]) and pd.notna(row["total"]):
                            # Calcular precio unitario
                            if row["cantidad"] != 0:
                                df.loc[idx, "precio_unitario"] = row["total"] / row["cantidad"]
                                df.loc[idx, "error_type"] = "calculated_price"
                        elif pd.isna(row["total"]) and pd.notna(row["cantidad"]) and pd.notna(row["precio_unitario"]):
                            # Calcular total
                            df.loc[idx, "total"] = row["cantidad"] * row["precio_unitario"]
                            df.loc[idx, "error_type"] = "calculated_total"
            except Exception as e:
                logger.exception(f"Error en validación parcial: {str(e)}")
            
            # Paso 4: Validación contextual basada en filas vecinas
            try:
                # Identificar filas no validadas previamente
                still_not_validated = ~df["valid"]
                
                if still_not_validated.any():
                    # Buscar patrones de filas similares ya validadas
                    valid_rows = df[df["valid"]].copy()
                    
                    if not valid_rows.empty:
                        # Para cada fila no validada, buscar similitud con filas validadas
                        for idx in df[still_not_validated].index:
                            row = df.loc[idx]
                            
                            # Verificar si tiene descripción
                            if pd.notna(row["descripcion"]) and row["descripcion"].strip() != "":
                                # Buscar filas validadas con descripción similar
                                desc = row["descripcion"].lower()
                                
                                # Calcular similitud de descripción con filas validadas
                                # (implementación simplificada - en producción usar algo como TF-IDF o embeddings)
                                similar_rows = []
                                for valid_idx, valid_row in valid_rows.iterrows():
                                    if pd.notna(valid_row["descripcion"]):
                                        valid_desc = valid_row["descripcion"].lower()
                                        # Calcular similitud basada en palabras compartidas
                                        words1 = set(re.findall(r'\w+', desc))
                                        words2 = set(re.findall(r'\w+', valid_desc))
                                        if words1 and words2:
                                            similarity = len(words1 & words2) / len(words1 | words2)
                                            if similarity > 0.5:  # Umbral de similitud
                                                similar_rows.append((valid_idx, similarity))
                                
                                # Si encontramos filas similares, validar por contexto
                                if similar_rows:
                                    df.loc[idx, "valid"] = True
                                    df.loc[idx, "validation_type"] = "contextual"
                                    df.loc[idx, "error_type"] = "similar_description"
                                    
                                    # Incrementar contador
                                    metadata["validation_types"]["contextual"] = metadata["validation_types"].get("contextual", 0) + 1
            except Exception as e:
                logger.exception(f"Error en validación contextual: {str(e)}")
            
            # Paso 5: Calcular estadísticas finales
            valid_count = df["valid"].sum()
            metadata["valid_rows"] = valid_count
            metadata["valid_percentage"] = (valid_count / len(df)) * 100 if len(df) > 0 else 0
            
            # Contar tipos de error para filas no válidas
            invalid_rows = ~df["valid"]
            if invalid_rows.any():
                error_counts = df.loc[invalid_rows, "error_type"].value_counts().to_dict()
                for error_type, count in error_counts.items():
                    if pd.notna(error_type):
                        metadata["error_types"][error_type] = count
                
                # Filas sin tipo de error específico
                no_error_type = invalid_rows & df["error_type"].isna()
                if no_error_type.any():
                    metadata["error_types"]["unknown"] = no_error_type.sum()
            
            # Registrar estadísticas de validación
            logger.info(f"Validación completada: {valid_count}/{len(df)} filas válidas ({metadata['valid_percentage']:.2f}%)")
            for vtype, count in metadata["validation_types"].items():
                logger.info(f"  - Tipo {vtype}: {count} filas")
            
            return df, metadata
            
        except Exception as e:
            logger.exception(f"Error en validación de presupuesto: {str(e)}")
            return df, {"error": str(e), "total_rows": len(df), "valid_rows": 0, "valid_percentage": 0.0}
    
    @staticmethod
    def detect_price_unit_column(df: pd.DataFrame, column_map: Dict[str, str]) -> Dict[str, str]:
        """
        Detecta la columna de precio unitario en un DataFrame.
        
        Utiliza un algoritmo sofisticado que considera:
        1. Nombres de columnas (coincidencia con patrones conocidos)
        2. Análisis de magnitud (comparación de medias entre columnas)
        3. Relaciones matemáticas (validación mediante cálculos)
        
        Args:
            df: DataFrame con los datos
            column_map: Mapeo de columnas ya identificadas
            
        Returns:
            Mapeo actualizado con la columna de precio unitario identificada
        """
        try:
            # Si ya tenemos identificada la columna de precio unitario, devolver el mapeo actual
            if "precio_unitario" in column_map.values():
                return column_map
                
            # Verificar si tenemos las columnas necesarias para detectar el precio unitario
            if "cantidad" not in column_map.values() or "total" not in column_map.values():
                logger.warning("No se pueden detectar columnas de precio unitario sin columnas de cantidad y total")
                return column_map
                
            # Obtener los nombres de las columnas de cantidad y total
            qty_col = next(col for col, mapped in column_map.items() if mapped == "cantidad")
            total_col = next(col for col, mapped in column_map.items() if mapped == "total")
            
            # Columnas candidatas (excluir columnas ya mapeadas)
            mapped_cols = set(column_map.values())
            candidate_cols = [col for col in df.columns if col not in column_map.keys() and pd.api.types.is_numeric_dtype(df[col])]
            
            # Si no hay columnas candidatas, no podemos detectar el precio unitario
            if not candidate_cols:
                logger.warning("No hay columnas candidatas para precio unitario")
                return column_map
                
            # Inicializar puntuaciones para cada columna candidata
            scores = {col: 0 for col in candidate_cols}
            
            # 1. Análisis de nombres de columnas (30% del peso total)
            name_patterns = {
                'precio_unitario': ['precio_unitario', 'precio', 'unit', 'p.u', 'p/u', 'p.unit', 'costo_unit', 'valor_unit', 'tarifa'],
                'negativo': ['total', 'subtotal', 'cantidad', 'qty', 'volumen', 'area', 'longitud', 'ancho', 'alto']
            }
            
            for col in candidate_cols:
                col_lower = str(col).lower()
                
                # Patrones positivos (indican precio unitario)
                for pattern in name_patterns['precio_unitario']:
                    if pattern in col_lower:
                        scores[col] += 30  # Peso máximo para nombres de columna
                        break
                
                # Patrones negativos (indican que NO es precio unitario)
                for pattern in name_patterns['negativo']:
                    if pattern in col_lower:
                        scores[col] -= 20
                        break
            
            # 2. Análisis de magnitud (30% del peso total)
            # Filtrar filas con valores no nulos en las columnas relevantes
            valid_rows = df[(df[qty_col].notna()) & (df[total_col].notna())]
            
            if not valid_rows.empty:
                qty_mean = valid_rows[qty_col].mean()
                total_mean = valid_rows[total_col].mean()
                
                # Calcular el precio unitario esperado
                expected_unit_price = total_mean / qty_mean if qty_mean != 0 else 0
                
                # Si el precio unitario esperado es muy cercano a 0 o es NaN, usar un valor predeterminado
                if abs(expected_unit_price) < 1e-10 or pd.isna(expected_unit_price):
                    expected_unit_price = total_mean  # Usar el total como referencia
                
                # Comparar cada columna candidata con el precio unitario esperado
                for col in candidate_cols:
                    if col in valid_rows.columns:
                        col_mean = valid_rows[col].mean()
                        
                        # Si la media de la columna es NaN o 0, continuar con la siguiente columna
                        if pd.isna(col_mean) or abs(col_mean) < 1e-10:
                            continue
                        
                        # Calcular la diferencia relativa con el precio unitario esperado
                        rel_diff = abs(col_mean - expected_unit_price) / max(abs(expected_unit_price), 1e-10)
                        
                        # Asignar puntuación basada en la similitud
                        if rel_diff < 0.1:  # Muy similar
                            scores[col] += 30
                        elif rel_diff < 0.5:  # Moderadamente similar
                            scores[col] += 20
                        elif rel_diff < 1.0:  # Algo similar
                            scores[col] += 10
                        elif rel_diff > 100:  # Muy diferente
                            scores[col] -= 10
            
            # 3. Validación mediante relaciones matemáticas (40% del peso total)
            valid_count = {col: 0 for col in candidate_cols}
            total_count = 0
            
            # Calcular cuántas filas cumplen la relación cantidad * precio_unitario ≈ total
            for _, row in df.iterrows():
                if pd.notna(row[qty_col]) and pd.notna(row[total_col]) and row[qty_col] != 0:
                    total_count += 1
                    expected_price = row[total_col] / row[qty_col]
                    
                    for col in candidate_cols:
                        if col in row.index and pd.notna(row[col]):
                            # Calcular error relativo
                            rel_error = abs(row[col] - expected_price) / max(abs(expected_price), 1e-10)
                            
                            # Considerar válido si el error es menor al 10%
                            if rel_error < 0.1:
                                valid_count[col] += 1
            
            # Asignar puntuación basada en la proporción de filas válidas
            if total_count > 0:
                for col in candidate_cols:
                    validation_ratio = valid_count[col] / total_count
                    
                    # Asignar puntuación basada en la proporción
                    if validation_ratio > 0.8:  # Más del 80% de filas válidas
                        scores[col] += 40
                    elif validation_ratio > 0.6:  # Más del 60% de filas válidas
                        scores[col] += 30
                    elif validation_ratio > 0.4:  # Más del 40% de filas válidas
                        scores[col] += 20
                    elif validation_ratio > 0.2:  # Más del 20% de filas válidas
                        scores[col] += 10
            
            # 4. Análisis de distribución de valores (bonus de hasta 20 puntos)
            for col in candidate_cols:
                if col in df.columns:
                    # Verificar si la distribución de valores es consistente con precios unitarios
                    values = df[col].dropna()
                    
                    if not values.empty:
                        # Los precios unitarios suelen tener valores positivos
                        if (values >= 0).mean() > 0.9:  # Más del 90% son positivos
                            scores[col] += 10
                        
                        # Los precios unitarios suelen tener cierta variabilidad pero no extrema
                        cv = values.std() / values.mean() if values.mean() != 0 else float('inf')
                        if 0.1 < cv < 2.0:  # Coeficiente de variación en rango razonable
                            scores[col] += 10
                        elif cv > 10.0:  # Variabilidad extrema, probablemente no es precio unitario
                            scores[col] -= 10
            
            # Seleccionar la columna con mayor puntuación
            if candidate_cols:
                best_col = max(scores.items(), key=lambda x: x[1])
                
                # Solo asignar si la puntuación es positiva
                if best_col[1] > 0:
                    column_map[best_col[0]] = "precio_unitario"
                    logger.info(f"Columna de precio unitario detectada: {best_col[0]} (puntuación: {best_col[1]})")
                    
                    # Registrar puntuaciones para depuración
                    for col, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                        logger.debug(f"Candidato a precio unitario: {col} (puntuación: {score})")
                else:
                    logger.warning("No se pudo detectar columna de precio unitario con suficiente confianza")
            
            return column_map
            
        except Exception as e:
            logger.exception(f"Error al detectar columna de precio unitario: {str(e)}")
            return column_map
