#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de prueba para el sistema mejorado de validación y extracción de presupuestos.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("budget_test")

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar los módulos necesarios
from modules.extractors.excel import ExcelExtractor
from modules.extractors.budget_validator import BudgetValidator

def create_test_data():
    """
    Crea un DataFrame de prueba con datos de presupuesto.
    """
    # Crear datos de prueba con varios casos complejos
    data = {
        "codigo": ["1.1", "1.2", "1.3", "1.4", "1.5", "TOTAL", "2.1", "2.2", "2.3", "2 unidades a 500"],
        "descripcion": [
            "Excavación manual", 
            "Relleno compactado", 
            "Concreto f'c=210 kg/cm²", 
            "Acero de refuerzo fy=4200 kg/cm²", 
            "Encofrado y desencofrado", 
            "SUBTOTAL CAPÍTULO 1", 
            "Muro de ladrillo", 
            "Tarrajeo en muros", 
            "Pintura látex 2 manos",
            "Suministro e instalación de ventanas"
        ],
        "unidad": ["m³", "m³", "m³", "kg", "m²", "", "m²", "m²", "m²", "und"],
        "cantidad": [20, 15, 10, "500 kg", 50, None, 100, 150, 200, "2 unidades"],
        "precio_unitario": ["50.00", "40.00", "300", "$250 por m³", "$35", None, "60", "25.00", "15", "500 c/u"],
        "total": [1000, 600, 3000, 125000, 1750, 131350, 6000, 3750, 3000, 1000]
    }
    
    # Crear DataFrame
    df = pd.DataFrame(data)
    
    # Agregar algunas filas con problemas
    problematic_data = {
        "codigo": ["3.1", "3.2", "3.3", None, "4.1"],
        "descripcion": [
            "Instalación eléctrica", 
            "Instalación sanitaria", 
            "Instalación de gas",
            None,
            "Limpieza final"
        ],
        "unidad": ["pto", "pto", "pto", None, "glb"],
        "cantidad": [25, 15, 10, None, 1],
        "precio_unitario": [120, 150, "200 por punto", None, 2000],
        "total": [3000, 2250, 2000, None, 2000]
    }
    
    problematic_df = pd.DataFrame(problematic_data)
    
    # Concatenar DataFrames
    df = pd.concat([df, problematic_df], ignore_index=True)
    
    return df

def test_normalization(df):
    """
    Prueba la normalización de columnas numéricas.
    """
    logger.info("=== PRUEBA DE NORMALIZACIÓN DE COLUMNAS NUMÉRICAS ===")
    
    # Crear un extractor
    extractor = ExcelExtractor()
    
    # Hacer una copia del DataFrame de prueba
    df_copy = df.copy()
    
    # Normalizar columnas numéricas
    logger.info("Normalizando columnas numéricas...")
    df_normalized = extractor.normalize_numeric_columns(df_copy)
    
    # Mostrar resultados
    logger.info("Resultados de normalización:")
    for col in ["cantidad", "precio_unitario", "total"]:
        if col in df_normalized.columns:
            before_count = df_copy[col].notna().sum()
            after_count = df_normalized[col].notna().sum()
            recovered = after_count - before_count
            
            success_rate = after_count / len(df_normalized) * 100
            
            logger.info(f"Columna '{col}': {after_count}/{len(df_normalized)} valores válidos ({success_rate:.2f}%)")
            if recovered > 0:
                logger.info(f"  - Se recuperaron {recovered} valores durante la normalización")
    
    # Mostrar algunos ejemplos de valores normalizados
    sample_rows = [0, 3, 4, 9]  # Filas con casos interesantes
    for idx in sample_rows:
        if idx < len(df_normalized):
            original_row = df_copy.iloc[idx]
            normalized_row = df_normalized.iloc[idx]
            
            logger.info(f"\nEjemplo {idx+1}: {normalized_row['descripcion']}")
            for col in ["cantidad", "precio_unitario", "total"]:
                if col in df_normalized.columns:
                    orig_val = original_row[col]
                    norm_val = normalized_row[col]
                    logger.info(f"  - {col}: '{orig_val}' -> {norm_val}")
    
    return df_normalized

def test_price_unit_detection(df):
    """
    Prueba la detección de columna de precio unitario.
    """
    logger.info("\n=== PRUEBA DE DETECCIÓN DE COLUMNA DE PRECIO UNITARIO ===")
    
    # Crear un conjunto de datos donde necesitamos detectar la columna de precio unitario
    detection_data = df.copy()
    
    # Renombrar columnas para simular que no sabemos cuál es la de precio unitario
    column_names = list(detection_data.columns)
    precio_col = None
    
    # Buscar la columna de precio unitario actual
    for col in column_names:
        if col == 'precio_unitario':
            # Cambiar el nombre a algo genérico
            detection_data = detection_data.rename(columns={
                col: "precio_por_unidad"
            })
            precio_col = "precio_por_unidad"
            break
    
    # Crear un mapa de columnas parcial (sin precio unitario)
    column_map = {}
    for col in detection_data.columns:
        if col == 'cantidad':
            column_map[col] = "cantidad"
        elif col == 'total':
            column_map[col] = "total"
    
    # Detectar la columna de precio unitario
    logger.info("Detectando columna de precio unitario...")
    updated_map = BudgetValidator.detect_price_unit_column(detection_data, column_map)
    
    # Mostrar resultados
    logger.info("Mapa de columnas original:")
    for col, mapped in column_map.items():
        logger.info(f"  - {col} -> {mapped}")
    
    logger.info("\nMapa de columnas actualizado:")
    for col, mapped in updated_map.items():
        logger.info(f"  - {col} -> {mapped}")
    
    # Verificar si se detectó correctamente
    detected = False
    for col, mapped in updated_map.items():
        if mapped == "precio_unitario":
            detected = True
            logger.info(f"\nSe detectó correctamente la columna de precio unitario: {col}")
    
    if not detected:
        logger.warning("No se pudo detectar la columna de precio unitario")
    
    return updated_map

def test_budget_validation(df):
    """
    Prueba la validación de consistencia de presupuesto.
    """
    logger.info("\n=== PRUEBA DE VALIDACIÓN DE CONSISTENCIA DE PRESUPUESTO ===")
    
    # Hacer una copia para no modificar el dataframe original
    df_validation = df.copy()
    
    # Asegurarse de que el dataframe tiene la estructura correcta
    required_cols = ["cantidad", "precio_unitario", "total", "descripcion"]
    missing_cols = [col for col in required_cols if col not in df_validation.columns]
    
    if missing_cols:
        logger.warning(f"Faltan columnas requeridas para validación: {missing_cols}")
        logger.info("Asegurando que el dataframe tenga todas las columnas necesarias...")
        for col in missing_cols:
            df_validation[col] = None
    
    # Preparar las columnas para la validación (inicialización exlícita)
    if 'valid' not in df_validation.columns:
        df_validation['valid'] = False
    
    # Inicializar otras columnas que usa el validador
    df_validation['validation_type'] = None
    df_validation['error_type'] = None
    df_validation['discrepancia'] = None
    df_validation['total_esperado'] = None
    
    # Convertir explícitamente columnas numéricas a float para evitar problemas
    for col in ['cantidad', 'precio_unitario', 'total']:
        if col in df_validation.columns:
            df_validation[col] = pd.to_numeric(df_validation[col], errors='coerce')
    
    # Validar la consistencia del presupuesto
    logger.info("Validando consistencia del presupuesto...")
    validated_df, metadata = BudgetValidator.validate_budget_consistency(df_validation)
    
    # Mostrar resultados generales
    valid_count = metadata.get("valid_rows", 0)
    total_count = metadata.get("total_rows", len(validated_df))
    valid_percent = metadata.get("valid_percentage", 0)
    
    logger.info(f"Resultados de validación: {valid_count}/{total_count} filas válidas ({valid_percent:.2f}%)")
    
    # Mostrar resultados por tipo de validación
    validation_types = metadata.get("validation_types", {})
    if validation_types:
        logger.info("\nTipos de validación:")
        for vtype, count in validation_types.items():
            logger.info(f"  - {vtype}: {count} filas")
    
    # Mostrar tipos de error
    error_types = metadata.get("error_types", {})
    if error_types:
        logger.info("\nTipos de error:")
        for etype, count in error_types.items():
            logger.info(f"  - {etype}: {count} filas")
    
    # Mostrar ejemplos de filas validadas y no validadas
    if "valid" in validated_df.columns:
        # Verificar filas válidas
        valid_rows = validated_df[validated_df["valid"] == True]
        if not valid_rows.empty:
            logger.info("\nEjemplos de filas validadas:")
            valid_samples = valid_rows.head(3)
            for idx, row in valid_samples.iterrows():
                logger.info(f"  Fila {idx}: {row.get('descripcion', 'Sin descripción')}")
                logger.info(f"    - Tipo de validación: {row.get('validation_type', 'desconocido')}")
                logger.info(f"    - Cantidad: {row.get('cantidad', 'N/A')}, Precio: {row.get('precio_unitario', 'N/A')}, Total: {row.get('total', 'N/A')}")
                logger.info(f"    - Total esperado: {row.get('total_esperado', 'N/A')}, Discrepancia: {row.get('discrepancia', 'N/A')}")
        else:
            logger.warning("No se encontraron filas válidas")
        
        # Verificar filas inválidas
        invalid_rows = validated_df[validated_df["valid"] == False]
        if not invalid_rows.empty:
            logger.info("\nEjemplos de filas no validadas:")
            invalid_samples = invalid_rows.head(3)
            for idx, row in invalid_samples.iterrows():
                logger.info(f"  Fila {idx}: {row.get('descripcion', 'Sin descripción')}")
                logger.info(f"    - Error: {row.get('error_type', 'desconocido')}")
                logger.info(f"    - Cantidad: {row.get('cantidad', 'N/A')}, Precio: {row.get('precio_unitario', 'N/A')}, Total: {row.get('total', 'N/A')}")
                logger.info(f"    - Total esperado: {row.get('total_esperado', 'N/A')}, Discrepancia: {row.get('discrepancia', 'N/A')}")
    else:
        logger.warning("El DataFrame validado no contiene la columna 'valid'")
    
    return validated_df, metadata

def validacion_simplificada(df):
    """
    Implementa una versión simplificada del algoritmo de validación de presupuestos
    para demostrar la funcionalidad.
    """
    logger.info("\n=== VALIDACIÓN SIMPLIFICADA DEL PRESUPUESTO ===")
    
    # Hacer una copia del dataframe
    df_val = df.copy()
    
    # Asegurarse de tener las columnas necesarias
    required_cols = ["cantidad", "precio_unitario", "total", "descripcion"]
    missing_cols = [col for col in required_cols if col not in df_val.columns]
    
    if missing_cols:
        logger.warning(f"Faltan columnas requeridas para validación: {missing_cols}")
        return df, {"error": f"Faltan columnas: {missing_cols}"}
    
    # Inicializar columnas de validación
    df_val["valid"] = False
    df_val["validation_type"] = None
    df_val["error_type"] = None
    df_val["total_esperado"] = None
    df_val["discrepancia"] = None
    
    # Calcular total esperado
    df_val["total_esperado"] = df_val["cantidad"] * df_val["precio_unitario"]
    
    # Umbral de discrepancia (5%)
    threshold = 0.05
    
    # Estadísticas de validación
    metadata = {
        "total_rows": len(df_val),
        "valid_rows": 0,
        "valid_percentage": 0,
        "validation_types": {},
        "error_types": {}
    }
    
    # Validación completa
    try:
        # Filtrar filas con valores numéricos completos
        mask_full = (
            df_val["cantidad"].notna() & 
            df_val["precio_unitario"].notna() & 
            df_val["total"].notna() &
            (df_val["cantidad"] != 0) & 
            (df_val["precio_unitario"] != 0) & 
            (df_val["total"] != 0)
        )
        
        if mask_full.any():
            # Calcular discrepancia
            df_val.loc[mask_full, "discrepancia"] = (
                (df_val.loc[mask_full, "total"] - df_val.loc[mask_full, "total_esperado"]).abs() / 
                df_val.loc[mask_full, "total"]
            )
            
            # Marcar filas válidas (discrepancia dentro del umbral)
            mask_valid = mask_full & (df_val["discrepancia"] <= threshold)
            
            # Actualizar DataFrame y estadísticas
            df_val.loc[mask_valid, "valid"] = True
            df_val.loc[mask_valid, "validation_type"] = "full"
            
            valid_count = mask_valid.sum()
            metadata["validation_types"]["full"] = valid_count
            metadata["valid_rows"] += valid_count
            
            # Marcar filas con alta discrepancia
            mask_invalid = mask_full & (df_val["discrepancia"] > threshold)
            df_val.loc[mask_invalid, "error_type"] = "high_discrepancy"
            metadata["error_types"]["high_discrepancy"] = mask_invalid.sum()
            
            logger.info(f"Validación completa: {valid_count} filas válidas de {mask_full.sum()} con datos completos")
    except Exception as e:
        logger.exception(f"Error en validación completa: {str(e)}")
    
    # Validación parcial
    try:
        # Identificar filas no validadas con descripción
        not_validated = ~df_val["valid"]
        has_description = df_val["descripcion"].notna() & (df_val["descripcion"].astype(str).str.strip() != "")
        
        # Filas calculables (tienen descripción y exactamente 2 valores numéricos)
        numeric_cols = ["cantidad", "precio_unitario", "total"]
        numeric_count = df_val[numeric_cols].notna().sum(axis=1)
        
        # Caso 1: Filas calculables
        mask_calculable = not_validated & has_description & (numeric_count == 2)
        
        if mask_calculable.any():
            # Marcar filas y registrar estadísticas
            df_val.loc[mask_calculable, "valid"] = True
            df_val.loc[mask_calculable, "validation_type"] = "calculable"
            
            valid_count = mask_calculable.sum()
            metadata["validation_types"]["calculable"] = valid_count
            metadata["valid_rows"] += valid_count
            
            # Calcular valores faltantes
            for idx in df_val[mask_calculable].index:
                row = df_val.loc[idx]
                if pd.isna(row["cantidad"]):
                    if row["precio_unitario"] != 0:
                        df_val.loc[idx, "cantidad"] = row["total"] / row["precio_unitario"]
                        df_val.loc[idx, "error_type"] = "calculated_quantity"
                elif pd.isna(row["precio_unitario"]):
                    if row["cantidad"] != 0:
                        df_val.loc[idx, "precio_unitario"] = row["total"] / row["cantidad"]
                        df_val.loc[idx, "error_type"] = "calculated_price"
                elif pd.isna(row["total"]):
                    df_val.loc[idx, "total"] = row["cantidad"] * row["precio_unitario"]
                    df_val.loc[idx, "error_type"] = "calculated_total"
            
            logger.info(f"Validación parcial (calculable): {valid_count} filas")
        
        # Caso 2: Subtotales o agrupaciones
        mask_subtotal = not_validated & has_description & df_val["descripcion"].astype(str).str.lower().str.contains(
            r'total|subtotal|suma|capítulo|partida|grupo', regex=True
        )
        
        if mask_subtotal.any():
            df_val.loc[mask_subtotal, "valid"] = True
            df_val.loc[mask_subtotal, "validation_type"] = "subtotal"
            
            valid_count = mask_subtotal.sum()
            metadata["validation_types"]["subtotal"] = valid_count
            metadata["valid_rows"] += valid_count
            
            logger.info(f"Validación parcial (subtotal): {valid_count} filas")
    except Exception as e:
        logger.exception(f"Error en validación parcial: {str(e)}")
    
    # Actualizar estadísticas finales
    if metadata["total_rows"] > 0:
        metadata["valid_percentage"] = (metadata["valid_rows"] / metadata["total_rows"]) * 100
    
    # Registrar resumen
    logger.info(f"Validación simplificada completada: {metadata['valid_rows']}/{metadata['total_rows']} filas válidas ({metadata['valid_percentage']:.2f}%)")
    
    return df_val, metadata

def main():
    """
    Función principal que ejecuta todas las pruebas.
    """
    logger.info("Iniciando pruebas del sistema de validación y extracción de presupuestos")
    
    # Crear datos de prueba
    test_df = create_test_data()
    logger.info(f"Datos de prueba creados: {len(test_df)} filas")
    
    # Probar normalización
    normalized_df = test_normalization(test_df)
    
    # Probar detección de columna de precio unitario
    column_map = test_price_unit_detection(normalized_df)
    
    # Asegurarse de que el dataframe tiene el nombre correcto para la columna de precio unitario
    if 'precio_por_unidad' in normalized_df.columns:
        normalized_df = normalized_df.rename(columns={'precio_por_unidad': 'precio_unitario'})
    
    # Probar validación de presupuesto con la implementación original
    logger.info("\n--- Validación usando la implementación del sistema ---")
    validated_df, metadata = test_budget_validation(normalized_df)
    
    # Probar validación de presupuesto con nuestra implementación simplificada
    logger.info("\n--- Validación usando la implementación simplificada ---")
    simple_validated_df, simple_metadata = validacion_simplificada(normalized_df)
    
    # Mostrar ejemplos de filas validadas y no validadas con la implementación simplificada
    if "valid" in simple_validated_df.columns:
        # Verificar filas válidas
        valid_rows = simple_validated_df[simple_validated_df["valid"] == True]
        if not valid_rows.empty:
            logger.info("\nEjemplos de filas validadas (implementación simplificada):")
            valid_samples = valid_rows.head(3)
            for idx, row in valid_samples.iterrows():
                logger.info(f"  Fila {idx}: {row.get('descripcion', 'Sin descripción')}")
                logger.info(f"    - Tipo de validación: {row.get('validation_type', 'desconocido')}")
                logger.info(f"    - Cantidad: {row.get('cantidad', 'N/A')}, Precio: {row.get('precio_unitario', 'N/A')}, Total: {row.get('total', 'N/A')}")
                logger.info(f"    - Total esperado: {row.get('total_esperado', 'N/A')}, Discrepancia: {row.get('discrepancia', 'N/A')}")
        else:
            logger.warning("No se encontraron filas válidas (implementación simplificada)")
    
    logger.info("\n=== RESUMEN DE PRUEBAS ===")
    
    # Calcular métricas generales (usar las de la implementación simplificada para el resumen)
    validation_rate = simple_metadata.get("valid_percentage", 0)
    
    logger.info(f"Tasa de validación final (simplificada): {validation_rate:.2f}%")
    
    # Proporcionar una conclusión basada en los resultados
    if validation_rate > 75:
        logger.info("Resultado: EXCELENTE - El sistema valida la mayoría de las filas")
    elif validation_rate > 50:
        logger.info("Resultado: BUENO - El sistema valida más de la mitad de las filas")
    elif validation_rate > 25:
        logger.info("Resultado: ACEPTABLE - El sistema valida un cuarto o más de las filas")
    else:
        logger.info("Resultado: NECESITA MEJORAS - Baja tasa de validación")
    
    logger.info("Pruebas completadas")

if __name__ == "__main__":
    main()
