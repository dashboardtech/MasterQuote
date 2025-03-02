#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para analizar y validar un archivo Excel de presupuesto.
Utiliza todas las mejoras implementadas, incluyendo BudgetValidator
y DocklingExtractor para proporcionar un análisis detallado.
"""

import os
import sys
import pandas as pd
import logging
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("budget_analyzer")

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar los módulos necesarios
from modules.extractors import BudgetValidator
from modules.universal_price_extractor import UniversalPriceExtractor

def analyze_budget_excel(file_path, output_dir=None, use_dockling=True, 
                        api_key=None, dockling_api_key=None, verbose=False):
    """
    Analiza y valida un archivo Excel de presupuesto y genera un informe detallado.
    
    Args:
        file_path: Ruta al archivo de presupuesto
        output_dir: Directorio para guardar resultados (opcional)
        use_dockling: Si debe usar Dockling para mejorar la extracción
        api_key: API key para servicios de IA asistida
        dockling_api_key: API key para Dockling
        verbose: Si debe mostrar información detallada
        
    Returns:
        Dict con estadísticas de análisis y validación
    """
    try:
        logger.info(f"Analizando archivo: {file_path}")
        
        # Crear directorio de salida si no existe
        if output_dir is None:
            file_name = Path(file_path).stem
            output_dir = f"analysis_{file_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Los resultados se guardarán en: {output_dir}")
        
        # Inicializar el extractor universal con caché deshabilitada
        extractor = UniversalPriceExtractor(
            api_key=api_key,
            dockling_api_key=dockling_api_key,
            use_cache=False  # Deshabilitar caché para evitar problemas
        )
        
        # Extraer datos directamente con pandas
        logger.info("Extrayendo datos del archivo con pandas...")
        try:
            # Leer directamente con pandas
            df = pd.read_excel(file_path)
            logger.info(f"Archivo Excel leído correctamente: {len(df)} filas")
            
            # Normalizar nombres de columnas
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Mapear columnas al formato esperado por BudgetValidator
            column_mapping = {
                'división': 'descripcion',
                'division': 'descripcion',
                'div': 'descripcion',
                'monto': 'precio_total',
                'importe': 'precio_total',
                'total': 'precio_total',
                'coste': 'precio_total',
                'costo': 'precio_total',
                'concepto': 'descripcion',
                'item': 'descripcion',
                'actividad': 'descripcion',
                'descripción': 'descripcion'
            }
            
            # Renombrar columnas según mapeo
            renamed_cols = {}
            for col in df.columns:
                if col in column_mapping:
                    renamed_cols[col] = column_mapping[col]
            
            if renamed_cols:
                df = df.rename(columns=renamed_cols)
            
            # Asegurar que las columnas necesarias existan
            required_cols = ['descripcion', 'cantidad', 'precio_unitario', 'precio_total']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'cantidad':
                        df[col] = 1  # Valor por defecto
                    elif col == 'precio_unitario' and 'precio_total' in df.columns:
                        df[col] = df['precio_total'] / df['cantidad']  # Calcular si es posible
                    elif col == 'precio_total' and 'precio_unitario' in df.columns:
                        df[col] = df['precio_unitario'] * df['cantidad']  # Calcular si es posible
                    else:
                        df[col] = None  # Valor nulo para columnas que no se pueden inferir
            
            metadata = {"método": "lectura_directa_pandas", "columnas_originales": list(df.columns)}
            
        except Exception as e:
            logger.error(f"Error leyendo el archivo: {str(e)}")
            return None
            
        if df is None or df.empty:
            logger.error("No se pudieron extraer datos del archivo.")
            return None
            
        logger.info(f"Datos extraídos: {len(df)} filas")
        
        # Guardar el DataFrame extraído
        extracted_file = os.path.join(output_dir, "extracted_data.xlsx")
        df.to_excel(extracted_file, index=False)
        logger.info(f"Datos extraídos guardados en: {extracted_file}")
        
        # Validar la consistencia del presupuesto
        logger.info("Validando consistencia del presupuesto...")
        from modules.extractors.budget_validator import BudgetValidator
        
        # Ajustar nombres de columnas para el validador
        column_map = {
            'descripcion': 'descripcion',
            'cantidad': 'cantidad',
            'precio_unitario': 'precio_unitario',
            'precio_total': 'total'  # El validador espera 'total', no 'precio_total'
        }
        
        # Crear copia del DataFrame con las columnas renombradas para el validador
        validator_df = df.copy()
        renamed_cols = {}
        for orig, expected in column_map.items():
            if orig in validator_df.columns and orig != expected:
                renamed_cols[orig] = expected
        
        if renamed_cols:
            validator_df = validator_df.rename(columns=renamed_cols)
        
        try:
            validator = BudgetValidator()
            validated_df, validation_metadata = validator.validate_budget_consistency(validator_df)
            logger.info(f"Validación completada. Filas validadas: {validation_metadata.get('valid_rows', 0)}/{len(validator_df)}")
            
            # Guardar datos validados
            validated_output = os.path.join(output_dir, "validated_data.xlsx")
            validated_df.to_excel(validated_output, index=False)
            logger.info(f"Datos validados guardados en: {validated_output}")
            
            # Convertir 'total' de nuevo a 'precio_total' para mantener consistencia
            if 'total' in validated_df.columns and 'precio_total' not in validated_df.columns:
                validated_df = validated_df.rename(columns={'total': 'precio_total'})
            
        except Exception as e:
            logger.error(f"Error en la validación: {str(e)}")
            # Si falla la validación, continuamos con los datos originales
            validated_df = df.copy()
        
        logger.info(f"Datos validados: {len(validated_df)} filas")
        
        # Registrar estadísticas de validación
        valid_rows = validation_metadata.get('valid_rows', 0) if 'valid_rows' in validation_metadata else 0
        total_rows = len(df)
        validation_rate = (valid_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Crear informe de análisis
        report_data = {
            "Estadísticas generales": {
                "Filas totales": total_rows,
                "Filas validadas": valid_rows,
                "Porcentaje de validación": f"{validation_rate:.2f}%",
                "Columnas detectadas": list(df.columns)
            }
        }
        
        # Verificar si hay columnas de precios y cantidades
        has_qty = 'cantidad' in validated_df.columns and validated_df['cantidad'].notna().sum() > 0
        has_unit_price = 'precio_unitario' in validated_df.columns and validated_df['precio_unitario'].notna().sum() > 0
        has_total = 'precio_total' in validated_df.columns and validated_df['precio_total'].notna().sum() > 0
        
        report_data["Columnas detectadas"] = {
            "Descripción": 'descripcion' in validated_df.columns,
            "Cantidad": has_qty,
            "Precio unitario": has_unit_price,
            "Precio total": has_total
        }
        
        # Guardar estadísticas en formato JSON
        stats_file = os.path.join(output_dir, "stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            # Función para convertir tipos no serializables por defecto
            def convert_for_json(obj):
                if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
                elif isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.isoformat()
                else:
                    return str(obj)
            
            # Convertir los datos para asegurar que son serializables
            report_data_serializable = json.loads(
                json.dumps(report_data, default=convert_for_json)
            )
            json.dump(report_data_serializable, f, indent=4, ensure_ascii=False)
        logger.info(f"Estadísticas guardadas en: {stats_file}")
        
        # Generar visualizaciones
        logger.info("Generando visualizaciones...")
        plt.figure(figsize=(10, 8))
        
        # Configurar estilo
        sns.set_style("whitegrid")
        
        # Gráfico de barras para tasa de validación
        plt.subplot(2, 2, 1)
        labels = ['Validadas', 'No validadas']
        values = [valid_rows, len(validated_df) - valid_rows]
        plt.bar(labels, values, color=['#4CAF50', '#F44336'])
        plt.title('Tasa de validación de filas')
        plt.ylabel('Número de filas')
        
        # Gráfico de completitud de datos
        plt.subplot(2, 2, 2)
        labels = ['Con descripción', 'Con precio unitario', 'Con cantidad', 'Con precio total']
        descr_count = validated_df['descripcion'].notna().sum() if 'descripcion' in validated_df.columns else 0
        values = [descr_count, has_unit_price, has_qty, has_total]
        plt.bar(labels, values, color=['#2196F3', '#FF9800', '#9C27B0', '#00BCD4'])
        plt.title('Completitud de datos')
        plt.xticks(rotation=45)
        plt.ylabel('Número de filas')
        
        # Histograma de precios unitarios
        plt.subplot(2, 2, 3)
        if has_unit_price > 0:
            unit_prices = validated_df['precio_unitario'].dropna()
            if len(unit_prices) > 0:
                sns.histplot(unit_prices, kde=True)
                plt.title('Distribución de precios unitarios')
                plt.xlabel('Precio unitario')
        else:
            plt.text(0.5, 0.5, 'No hay datos de precios unitarios', ha='center', va='center')
            plt.title('Distribución de precios unitarios')
        
        # Histograma de precios totales
        plt.subplot(2, 2, 4)
        if has_total > 0:
            total_prices = validated_df['precio_total'].dropna()
            if len(total_prices) > 0:
                sns.histplot(total_prices, kde=True)
                plt.title('Distribución de precios totales')
                plt.xlabel('Precio total')
        else:
            plt.text(0.5, 0.5, 'No hay datos de precios totales', ha='center', va='center')
            plt.title('Distribución de precios totales')
        
        plt.tight_layout()
        viz_file = os.path.join(output_dir, "visualizaciones.png")
        plt.savefig(viz_file, dpi=300)
        plt.close()
        logger.info(f"Visualizaciones guardadas en: {viz_file}")
        
        # Imprimir resumen
        logger.info("\n----- RESUMEN DE ANÁLISIS -----")
        logger.info(f"Archivo analizado: {file_path}")
        logger.info(f"Total de filas: {len(validated_df)}")
        logger.info(f"Filas validadas: {valid_rows} ({validation_rate:.2f}%)")
        logger.info(f"Filas con precio unitario: {has_unit_price} ({has_unit_price/len(validated_df)*100:.2f}%)")
        logger.info(f"Filas con cantidad: {has_qty} ({has_qty/len(validated_df)*100:.2f}%)")
        logger.info(f"Filas con precio total: {has_total} ({has_total/len(validated_df)*100:.2f}%)")
        logger.info("--------------------------")
        
        return report_data
        
    except Exception as e:
        logger.exception(f"Error en el análisis del presupuesto: {str(e)}")
        return None
        
def main():
    """Función principal para ejecutar desde línea de comandos"""
    parser = argparse.ArgumentParser(description="Analizar y validar un archivo Excel de presupuesto")
    parser.add_argument("file_path", help="Ruta al archivo de presupuesto a analizar")
    parser.add_argument("--output", help="Directorio para guardar resultados", default=None)
    parser.add_argument("--no-dockling", help="Desactivar Dockling", action="store_true")
    parser.add_argument("--verbose", help="Mostrar información detallada", action="store_true")
    
    args = parser.parse_args()
    
    # Obtener API keys del entorno
    api_key = os.environ.get("OPENAI_API_KEY")
    dockling_api_key = os.environ.get("DOCKLING_API_KEY")
    
    # Analizar archivo
    stats = analyze_budget_excel(
        args.file_path,
        output_dir=args.output,
        use_dockling=not args.no_dockling,
        api_key=api_key,
        dockling_api_key=dockling_api_key,
        verbose=args.verbose
    )
    
    if stats is not None:
        logger.info("Análisis completado exitosamente")
    else:
        logger.error("Error en el análisis del presupuesto")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
