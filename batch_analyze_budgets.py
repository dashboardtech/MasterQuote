#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para procesar y analizar múltiples archivos de presupuestos en lote.
Permite consolidar información de múltiples archivos Excel para crear
un conjunto de datos robusto y optimizar el sistema de validación.
"""

import os
import sys
import pandas as pd
import logging
import argparse
import json
import glob
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from json_utils import dump_json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("batch_analyzer")

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar los módulos necesarios
from modules.extractors import BudgetValidator
from modules.universal_price_extractor import UniversalPriceExtractor
from modules.price_database import PriceDatabase
from analyze_budget_excel import analyze_budget_excel
from import_budget_to_db import process_and_import_budget

def process_file(file_path, output_dir, use_dockling, api_key, dockling_api_key):
    """Procesa un solo archivo y devuelve sus estadísticas"""
    try:
        logger.info(f"Procesando archivo: {file_path}")
        stats = analyze_budget_excel(
            file_path,
            output_dir=os.path.join(output_dir, Path(file_path).stem),
            use_dockling=use_dockling,
            api_key=api_key,
            dockling_api_key=dockling_api_key
        )
        
        return {"path": file_path, "stats": stats, "success": stats is not None}
    except Exception as e:
        logger.error(f"Error procesando {file_path}: {str(e)}")
        return {"path": file_path, "stats": None, "success": False, "error": str(e)}

def batch_analyze_budgets(input_path, output_dir=None, use_dockling=True, 
                         api_key=None, dockling_api_key=None, 
                         import_to_db=False, db_path=None,
                         parallel=True, max_workers=4):
    """
    Analiza múltiples archivos de presupuesto en lote y genera un informe consolidado.
    
    Args:
        input_path: Directorio o patrón glob para encontrar archivos
        output_dir: Directorio para guardar resultados
        use_dockling: Si debe usar Dockling para mejorar la extracción
        api_key: API key para servicios de IA asistida
        dockling_api_key: API key para Dockling
        import_to_db: Si debe importar los resultados a la base de datos
        db_path: Ruta a la base de datos (opcional)
        parallel: Si debe procesar archivos en paralelo
        max_workers: Número máximo de procesos en paralelo
        
    Returns:
        Dict con estadísticas consolidadas de todos los análisis
    """
    try:
        # Crear directorio de salida si no existe
        if output_dir is None:
            output_dir = f"batch_analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Los resultados se guardarán en: {output_dir}")
        
        # Encontrar archivos a procesar
        if os.path.isdir(input_path):
            files = glob.glob(os.path.join(input_path, "*.xls*"))  # Incluye .xls y .xlsx
        else:
            files = glob.glob(input_path)
            
        if not files:
            logger.error(f"No se encontraron archivos en: {input_path}")
            return None
            
        logger.info(f"Se procesarán {len(files)} archivos")
        
        # Procesar archivos (en paralelo o secuencial)
        results = []
        
        if parallel and len(files) > 1:
            logger.info(f"Procesando archivos en paralelo con {max_workers} workers...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        process_file, 
                        file_path, 
                        output_dir, 
                        use_dockling, 
                        api_key, 
                        dockling_api_key
                    ): file_path for file_path in files
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(files)):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error en {file_path}: {str(e)}")
                        results.append({
                            "path": file_path, 
                            "stats": None, 
                            "success": False, 
                            "error": str(e)
                        })
        else:
            logger.info("Procesando archivos secuencialmente...")
            for file_path in tqdm(files):
                result = process_file(file_path, output_dir, use_dockling, api_key, dockling_api_key)
                results.append(result)
        
        # Recopilar estadísticas consolidadas
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        # Si se solicita, importar a la base de datos
        db = None
        if import_to_db:
            logger.info("Importando datos a la base de datos...")
            for result in successful:
                file_path = result["path"]
                try:
                    logger.info(f"Importando {file_path} a la base de datos...")
                    process_and_import_budget(
                        file_path,
                        db_path=db_path,
                        use_dockling=use_dockling,
                        api_key=api_key,
                        dockling_api_key=dockling_api_key,
                        project_name=Path(file_path).stem
                    )
                except Exception as e:
                    logger.error(f"Error al importar {file_path}: {str(e)}")
        
        # Calcular estadísticas globales
        if successful:
            # Adaptar para el nuevo formato de estadísticas
            total_rows = 0
            valid_rows = 0
            validation_rates = []
            
            for r in successful:
                if r["stats"] and "Estadísticas generales" in r["stats"]:
                    stats = r["stats"]["Estadísticas generales"]
                    total_rows += stats.get("Filas totales", 0)
                    valid_rows += stats.get("Filas validadas", 0)
                    
                    # Extraer tasa de validación como número
                    validation_rate_str = stats.get("Porcentaje de validación", "0%")
                    validation_rate = float(validation_rate_str.strip("%") if isinstance(validation_rate_str, str) else validation_rate_str)
                    validation_rates.append(validation_rate)
            
            avg_validation_rate = sum(validation_rates) / len(validation_rates) if validation_rates else 0
            
            # Crear un DataFrame con todas las actividades validadas para análisis
            all_validated_data = []
            for result in successful:
                file_path = result["path"]
                validated_file = os.path.join(output_dir, Path(file_path).stem, "validated_data.xlsx")
                if os.path.exists(validated_file):
                    try:
                        df = pd.read_excel(validated_file)
                        df['source_file'] = file_path
                        all_validated_data.append(df)
                    except Exception as e:
                        logger.error(f"Error al leer {validated_file}: {str(e)}")
            
            # Consolidar todos los datos validados
            if all_validated_data:
                consolidated_df = pd.concat(all_validated_data, ignore_index=True)
                consolidated_file = os.path.join(output_dir, "consolidated_data.xlsx")
                consolidated_df.to_excel(consolidated_file, index=False)
                logger.info(f"Datos consolidados guardados en: {consolidated_file}")
                
                # Análisis de precio unitario por descripción similar
                if 'descripcion' in consolidated_df.columns and 'precio_unitario' in consolidated_df.columns:
                    # Normalizar descripciones
                    consolidated_df['desc_norm'] = consolidated_df['descripcion'].str.lower().str.strip()
                    
                    # Agrupar por descripción normalizada
                    group_columns = {
                        'precio_unitario': ['count', 'mean', 'std', 'min', 'max'],
                        'descripcion': 'first'
                    }
                    
                    # Añadir columna de validación si existe
                    if 'valid' in consolidated_df.columns:
                        group_columns['valid'] = 'sum'
                    elif 'validado' in consolidated_df.columns:
                        group_columns['validado'] = 'sum'
                        
                    price_analysis = consolidated_df.groupby('desc_norm').agg(group_columns).reset_index()
                    
                    # Renombrar columnas
                    new_cols = ['desc_norm']
                    for col, aggs in group_columns.items():
                        if isinstance(aggs, list):
                            for agg in aggs:
                                new_cols.append(f"{col}_{agg}")
                        else:
                            new_cols.append(f"{col}_{aggs}")
                    
                    price_analysis.columns = new_cols
                    
                    # Intentar calcular coeficiente de variación
                    if 'precio_unitario_mean' in price_analysis.columns and 'precio_unitario_std' in price_analysis.columns:
                        price_analysis['cv'] = price_analysis['precio_unitario_std'] / price_analysis['precio_unitario_mean']
                    
                    # Guardar análisis
                    price_analysis_file = os.path.join(output_dir, "price_analysis.xlsx")
                    price_analysis.to_excel(price_analysis_file, index=False)
                    logger.info(f"Análisis de precios guardado en: {price_analysis_file}")
            
            # Crear informe consolidado
            consolidated_stats = {
                "archivos_procesados": len(files),
                "archivos_exitosos": len(successful),
                "archivos_fallidos": len(failed),
                "filas_totales": total_rows,
                "filas_validadas": valid_rows,
                "tasa_validacion_promedio": avg_validation_rate,
                "resultados_por_archivo": [
                    {
                        "archivo": r["path"],
                        "exitoso": r["success"],
                        "filas": r["stats"].get("Estadísticas generales", {}).get("Filas totales", 0) if r["success"] else 0,
                        "validadas": r["stats"].get("Estadísticas generales", {}).get("Filas validadas", 0) if r["success"] else 0,
                        "tasa": r["stats"].get("Estadísticas generales", {}).get("Porcentaje de validación", "0%") if r["success"] else "0%"
                    } for r in results
                ],
                "errores": [{"archivo": r["path"], "error": r.get("error", "")} for r in failed],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Guardar estadísticas en formato JSON
            stats_file = os.path.join(output_dir, "consolidated_stats.json")
            dump_json(consolidated_stats, stats_file)
            logger.info(f"Estadísticas consolidadas guardadas en: {stats_file}")
            
            # Generar visualizaciones consolidadas
            logger.info("Generando visualizaciones consolidadas...")
            plt.figure(figsize=(15, 10))
            
            # Configurar estilo
            sns.set_style("whitegrid")
            
            # Gráfico de tasas de validación por archivo
            plt.subplot(2, 2, 1)
            file_names = [Path(r["path"]).stem for r in successful]
            validation_rates = [r["stats"].get("Estadísticas generales", {}).get("Porcentaje de validación", "0%") for r in successful]
            
            if file_names and validation_rates:
                bars = plt.bar(range(len(file_names)), [float(rate.strip("%")) for rate in validation_rates], color='#2196F3')
                plt.xticks(range(len(file_names)), file_names, rotation=90)
                plt.title('Tasa de validación por archivo')
                plt.ylabel('Tasa de validación (%)')
                plt.axhline(y=avg_validation_rate, color='r', linestyle='-', label=f'Promedio: {avg_validation_rate:.2f}%')
                plt.legend()
            
            # Gráfico de archivos exitosos vs fallidos
            plt.subplot(2, 2, 2)
            labels = ['Exitosos', 'Fallidos']
            values = [len(successful), len(failed)]
            plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
            plt.title('Archivos procesados exitosamente')
            
            # Si hay datos consolidados, mostrar distribución de precios
            if 'consolidated_df' in locals():
                # Histograma de precios unitarios
                plt.subplot(2, 2, 3)
                if 'precio_unitario' in consolidated_df.columns:
                    unit_prices = consolidated_df['precio_unitario'].dropna()
                    if len(unit_prices) > 0:
                        sns.histplot(unit_prices, kde=True)
                        plt.title('Distribución de precios unitarios (todos los archivos)')
                        plt.xlabel('Precio unitario')
                
                # Scatter plot de precio unitario vs cantidad
                plt.subplot(2, 2, 4)
                if 'precio_unitario' in consolidated_df.columns and 'cantidad' in consolidated_df.columns:
                    valid_data = consolidated_df.dropna(subset=['precio_unitario', 'cantidad'])
                    if len(valid_data) > 0:
                        plt.scatter(valid_data['cantidad'], valid_data['precio_unitario'], alpha=0.5)
                        plt.title('Relación cantidad vs precio unitario')
                        plt.xlabel('Cantidad')
                        plt.ylabel('Precio unitario')
                        plt.xscale('log')
                        plt.yscale('log')
            
            plt.tight_layout()
            viz_file = os.path.join(output_dir, "consolidated_visualizaciones.png")
            plt.savefig(viz_file, dpi=300)
            plt.close()
            logger.info(f"Visualizaciones consolidadas guardadas en: {viz_file}")
            
            # Imprimir resumen consolidado
            logger.info("\n----- RESUMEN CONSOLIDADO -----")
            logger.info(f"Archivos procesados: {len(files)}")
            logger.info(f"Archivos exitosos: {len(successful)} ({len(successful)/len(files)*100:.2f}%)")
            logger.info(f"Archivos fallidos: {len(failed)} ({len(failed)/len(files)*100:.2f}%)")
            logger.info(f"Total filas en todos los archivos: {total_rows}")
            logger.info(f"Total filas validadas: {valid_rows} ({valid_rows/total_rows*100:.2f}%)")
            logger.info(f"Tasa de validación promedio: {avg_validation_rate:.2f}%")
            logger.info("------------------------------")
            
            return consolidated_stats
        else:
            logger.error("No se procesó ningún archivo exitosamente")
            return None
        
    except Exception as e:
        logger.exception(f"Error en el procesamiento por lotes: {str(e)}")
        return None
        
def main():
    """Función principal para ejecutar desde línea de comandos"""
    parser = argparse.ArgumentParser(description="Procesar múltiples archivos de presupuesto en lote")
    parser.add_argument("input_path", help="Directorio o patrón glob para encontrar archivos")
    parser.add_argument("--output", help="Directorio para guardar resultados", default=None)
    parser.add_argument("--db", help="Ruta a la base de datos (opcional)", default=None)
    parser.add_argument("--import", dest="import_to_db", help="Importar a la base de datos", action="store_true")
    parser.add_argument("--no-dockling", help="Desactivar Dockling", action="store_true")
    parser.add_argument("--sequential", help="Procesar archivos secuencialmente", action="store_true")
    parser.add_argument("--workers", help="Número máximo de procesos en paralelo", type=int, default=4)
    
    args = parser.parse_args()
    
    # Obtener API keys del entorno
    api_key = os.environ.get("OPENAI_API_KEY")
    dockling_api_key = os.environ.get("DOCKLING_API_KEY")
    
    # Procesar archivos en lote
    stats = batch_analyze_budgets(
        args.input_path,
        output_dir=args.output,
        use_dockling=not args.no_dockling,
        api_key=api_key,
        dockling_api_key=dockling_api_key,
        import_to_db=args.import_to_db,
        db_path=args.db,
        parallel=not args.sequential,
        max_workers=args.workers
    )
    
    if stats is not None:
        logger.info("Procesamiento por lotes completado exitosamente")
    else:
        logger.error("Error en el procesamiento por lotes")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
