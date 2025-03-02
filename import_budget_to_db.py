#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para importar presupuestos validados a la base de datos.
Este script integra todas las mejoras del sistema de validación
y los extractores optimizados, incluyendo Dockling.
"""

import os
import sys
import pandas as pd
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("budget_import")

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar los módulos necesarios
from modules.extractors import BudgetValidator, DocklingExtractor
from modules.universal_price_extractor import UniversalPriceExtractor
from modules.price_database import PriceDatabase

def process_and_import_budget(file_path, db_path=None, use_dockling=True, 
                             api_key=None, dockling_api_key=None, 
                             project_name=None, client_name=None):
    """
    Procesa un archivo de presupuesto, lo valida y lo importa a la base de datos.
    
    Args:
        file_path: Ruta al archivo de presupuesto
        db_path: Ruta a la base de datos (opcional)
        use_dockling: Si debe usar Dockling para mejorar la extracción
        api_key: API key para servicios de IA asistida
        dockling_api_key: API key para Dockling
        project_name: Nombre del proyecto (opcional)
        client_name: Nombre del cliente (opcional)
        
    Returns:
        DataFrame con los datos validados y procesados
    """
    try:
        logger.info(f"Procesando archivo: {file_path}")
        
        # Obtener el nombre del proyecto y cliente si no se especifican
        if project_name is None:
            project_name = Path(file_path).stem
        if client_name is None:
            client_name = "Cliente no especificado"
            
        # Extraer datos con pandas directamente para evitar problemas con APIs
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
                'monto': 'total',
                'importe': 'total',
                'totalim': 'total',
                'coste': 'total',
                'costo': 'total',
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
            required_cols = ['descripcion', 'cantidad', 'precio_unitario', 'total']
            for col in required_cols:
                if col not in df.columns:
                    if col == 'cantidad':
                        df[col] = 1  # Valor por defecto
                    elif col == 'precio_unitario' and 'total' in df.columns:
                        df[col] = df['total'] / df['cantidad']  # Calcular si es posible
                    elif col == 'total' and 'precio_unitario' in df.columns:
                        df[col] = df['precio_unitario'] * df['cantidad']  # Calcular si es posible
                    else:
                        df[col] = None  # Valor nulo para columnas que no se pueden inferir
                        
        except Exception as e:
            logger.error(f"Error leyendo el archivo: {str(e)}")
            return None
        
        if df is None or df.empty:
            logger.error("No se pudieron extraer datos del archivo.")
            return None
            
        logger.info(f"Datos extraídos: {len(df)} filas")
        
        # Validar y completar los datos del presupuesto
        logger.info("Validando consistencia del presupuesto...")
        try:
            validator = BudgetValidator()
            validated_df, validation_metadata = validator.validate_budget_consistency(df)
            
            # Registrar estadísticas de validación
            valid_rows = validation_metadata.get('valid_rows', 0)
            total_rows = len(df)
            validation_rate = (valid_rows / total_rows * 100) if total_rows > 0 else 0
            
            logger.info(f"Validación completada: {valid_rows}/{total_rows} filas ({validation_rate:.2f}%)")
        except Exception as e:
            logger.error(f"Error en la validación: {str(e)}")
            # Si falla la validación, continuamos con los datos originales
            validated_df = df.copy()
        
        # Conectar con la base de datos
        db_path = db_path or "price_history.db"
        db = PriceDatabase(db_path=db_path)
        logger.info(f"Conectado a la base de datos: {db_path}")
        
        # Preparar los datos para importación como cotización
        items = []
        for idx, row in validated_df.iterrows():
            # Solo incluir filas validadas
            if 'validado' in row and row['validado']:
                item = {
                    'actividad': row.get('descripcion', 'Actividad sin descripción'),
                    'cantidad': row.get('cantidad', 1),
                    'precio_unitario': row.get('precio_unitario', 0),
                    'unidad': row.get('unidad', 'unidad')
                }
                items.append(item)
        
        # Si tenemos filas validadas, guardar como cotización
        if items:
            logger.info(f"Guardando cotización con {len(items)} ítems...")
            notes = f"Importado automáticamente de {file_path}. Tasa de validación: {validation_rate:.2f}%"
            
            # Guardar en la base de datos
            cotizacion_id = db.guardar_cotizacion(
                nombre_proyecto=project_name,
                cliente=client_name,
                items=items,
                notas=notes
            )
            
            logger.info(f"Cotización guardada exitosamente con ID: {cotizacion_id}")
            
            # También importar los precios unitarios para futura referencia
            logger.info("Importando precios unitarios a la base de datos...")
            mapping = {
                'actividades': 'descripcion',
                'costo_unitario': 'precio_unitario'
            }
            
            # Guardar DataFrame en archivo temporal para importar
            temp_file = f"temp_import_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
            validated_df.to_excel(temp_file, index=False)
            
            try:
                # Importar precios
                db.importar_excel(temp_file, mapping=mapping)
                logger.info("Precios importados exitosamente")
            except Exception as e:
                logger.error(f"Error al importar precios: {str(e)}")
            finally:
                # Eliminar archivo temporal
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        else:
            logger.warning("No se encontraron filas validadas para importar")
            
        return validated_df
        
    except Exception as e:
        logger.exception(f"Error en el procesamiento e importación: {str(e)}")
        return None
        
def main():
    """Función principal para ejecutar desde línea de comandos"""
    parser = argparse.ArgumentParser(description="Importar presupuestos validados a la base de datos")
    parser.add_argument("file_path", help="Ruta al archivo de presupuesto a procesar")
    parser.add_argument("--db", help="Ruta a la base de datos (opcional)", default=None)
    parser.add_argument("--project", help="Nombre del proyecto", default=None)
    parser.add_argument("--client", help="Nombre del cliente", default=None)
    parser.add_argument("--no-dockling", help="Desactivar Dockling", action="store_true")
    
    args = parser.parse_args()
    
    # Obtener API keys del entorno
    api_key = os.environ.get("OPENAI_API_KEY")
    dockling_api_key = os.environ.get("DOCKLING_API_KEY")
    
    # Procesar archivo
    df = process_and_import_budget(
        args.file_path,
        db_path=args.db,
        use_dockling=not args.no_dockling,
        api_key=api_key,
        dockling_api_key=dockling_api_key,
        project_name=args.project,
        client_name=args.client
    )
    
    if df is not None:
        logger.info("Procesamiento e importación completados exitosamente")
    else:
        logger.error("Error en el procesamiento e importación del presupuesto")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
