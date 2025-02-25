import pandas as pd
import logging
from typing import Dict, Any, Optional, List
import yaml
from datetime import datetime
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def exportar_excel(df: pd.DataFrame, 
                  output_path: str,
                  metadata: Optional[Dict[str, Any]] = None,
                  config_path: str = "config.yaml") -> str:
    """
    Exporta una cotización a formato Excel con formato profesional.
    
    Args:
        df (pd.DataFrame): DataFrame con la cotización
        output_path (str): Ruta donde guardar el archivo Excel
        metadata (Dict[str, Any], optional): Metadatos de la cotización
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        str: Ruta al archivo Excel generado
    """
    try:
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Crear un escritor de Excel
        writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
        
        # Obtener el objeto workbook y worksheet
        workbook = writer.book
        
        # Definir formatos
        formato_titulo = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'align': 'center',
            'valign': 'vcenter',
            'font_name': 'Arial'
        })
        
        formato_encabezado = workbook.add_format({
            'bold': True,
            'font_size': 11,
            'align': 'center',
            'valign': 'vcenter',
            'bg_color': '#D9D9D9',
            'border': 1,
            'font_name': 'Arial'
        })
        
        formato_numero = workbook.add_format({
            'num_format': '$#,##0.00',
            'font_name': 'Arial'
        })
        
        formato_cantidad = workbook.add_format({
            'num_format': '#,##0.00',
            'font_name': 'Arial'
        })
        
        formato_texto = workbook.add_format({
            'font_name': 'Arial',
            'text_wrap': True
        })
        
        # Crear hoja de cotización
        df.to_excel(writer, sheet_name='Cotización', index=False, startrow=4)
        worksheet = writer.sheets['Cotización']
        
        # Configurar ancho de columnas
        worksheet.set_column('A:A', 40)  # Actividades
        worksheet.set_column('B:B', 10)  # Cantidad
        worksheet.set_column('C:C', 15)  # Costo Unitario
        worksheet.set_column('D:D', 15)  # Costo Total
        worksheet.set_column('E:Z', 12)  # Otras columnas
        
        # Agregar encabezado con metadatos
        if metadata:
            empresa = config['export'].get('company_name', 'MasterQuote')
            worksheet.merge_range('A1:D1', empresa, formato_titulo)
            
            fecha = metadata.get('fecha', datetime.now().strftime("%Y-%m-%d"))
            cliente = metadata.get('cliente', '')
            proyecto = metadata.get('proyecto', '')
            
            worksheet.write('A2', 'Fecha:', formato_encabezado)
            worksheet.write('B2', fecha, formato_texto)
            worksheet.write('C2', 'Cliente:', formato_encabezado)
            worksheet.write('D2', cliente, formato_texto)
            worksheet.write('A3', 'Proyecto:', formato_encabezado)
            worksheet.merge_range('B3:D3', proyecto, formato_texto)
        
        # Aplicar formatos a las columnas
        num_filas = len(df) + 5  # +5 por el encabezado
        
        worksheet.set_row(4, 30)  # Altura de la fila de encabezados
        
        # Formato para encabezados de columna
        for col in range(len(df.columns)):
            worksheet.write(4, col, df.columns[col], formato_encabezado)
        
        # Formato para datos
        for row in range(5, num_filas):
            worksheet.write(row, 0, df.iloc[row-5]['actividades'], formato_texto)
            if 'cantidad' in df.columns:
                worksheet.write(row, 1, df.iloc[row-5]['cantidad'], formato_cantidad)
            if 'costo_unitario' in df.columns:
                worksheet.write(row, 2, df.iloc[row-5]['costo_unitario'], formato_numero)
            if 'costo_total' in df.columns:
                worksheet.write(row, 3, df.iloc[row-5]['costo_total'], formato_numero)
        
        # Agregar totales
        if 'costo_total' in df.columns:
            fila_total = num_filas
            total = df['costo_total'].sum()
            
            formato_total = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'num_format': '$#,##0.00',
                'font_name': 'Arial',
                'top': 2
            })
            
            worksheet.write(fila_total, 2, 'Total:', formato_encabezado)
            worksheet.write(fila_total, 3, total, formato_total)
        
        # Guardar el archivo
        writer.close()
        logger.info(f"Excel exportado exitosamente: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error al exportar Excel: {str(e)}")
        raise

def generar_nombre_archivo(metadata: Dict[str, Any], 
                         extension: str = "xlsx") -> str:
    """
    Genera un nombre de archivo para la cotización.
    
    Args:
        metadata (Dict[str, Any]): Metadatos de la cotización
        extension (str): Extensión del archivo
        
    Returns:
        str: Nombre del archivo generado
    """
    try:
        # Obtener elementos para el nombre
        fecha = datetime.now().strftime("%Y%m%d")
        proyecto = metadata.get('proyecto', 'cotizacion').lower()
        cliente = metadata.get('cliente', '').lower()
        
        # Limpiar caracteres no válidos
        proyecto = ''.join(c for c in proyecto if c.isalnum() or c in (' ', '-', '_'))
        cliente = ''.join(c for c in cliente if c.isalnum() or c in (' ', '-', '_'))
        
        # Reemplazar espacios
        proyecto = proyecto.replace(' ', '_')
        cliente = cliente.replace(' ', '_')
        
        # Construir nombre
        nombre = f"{fecha}_{proyecto}"
        if cliente:
            nombre += f"_{cliente}"
        
        return f"{nombre}.{extension}"
        
    except Exception as e:
        logger.error(f"Error al generar nombre de archivo: {str(e)}")
        return f"cotizacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"

def crear_directorio_exportacion(config_path: str = "config.yaml") -> str:
    """
    Crea el directorio para exportación si no existe.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        str: Ruta al directorio de exportación
    """
    try:
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Obtener directorio de exportación
        export_dir = config['export'].get('output_dir', 'exports')
        
        # Crear directorio si no existe
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            logger.info(f"Directorio de exportación creado: {export_dir}")
        
        return export_dir
        
    except Exception as e:
        logger.error(f"Error al crear directorio de exportación: {str(e)}")
        return 'exports'  # Directorio por defecto

def exportar_cotizacion(df: pd.DataFrame,
                       metadata: Dict[str, Any],
                       formato: str = "xlsx",
                       config_path: str = "config.yaml") -> str:
    """
    Exporta una cotización al formato especificado.
    
    Args:
        df (pd.DataFrame): DataFrame con la cotización
        metadata (Dict[str, Any]): Metadatos de la cotización
        formato (str): Formato de exportación ('xlsx', 'pdf', etc.)
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        str: Ruta al archivo generado
    """
    try:
        # Crear directorio de exportación
        export_dir = crear_directorio_exportacion(config_path)
        
        # Generar nombre de archivo
        nombre_archivo = generar_nombre_archivo(metadata, formato)
        ruta_archivo = os.path.join(export_dir, nombre_archivo)
        
        # Exportar según formato
        if formato.lower() == 'xlsx':
            return exportar_excel(df, ruta_archivo, metadata, config_path)
        else:
            raise ValueError(f"Formato de exportación no soportado: {formato}")
            
    except Exception as e:
        logger.error(f"Error al exportar cotización: {str(e)}")
        raise
