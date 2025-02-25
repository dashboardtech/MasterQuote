import pandas as pd
import logging
from typing import Optional, Dict, Any
import yaml

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_excel(excel_path: str, config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Carga y valida un archivo Excel de cotización.
    
    Args:
        excel_path (str): Ruta al archivo Excel
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        pd.DataFrame: DataFrame con los datos validados
    """
    try:
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Cargar Excel
        df = pd.read_excel(excel_path)
        logger.info(f"Excel cargado: {excel_path}")
        
        # Normalizar nombres de columnas
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Validar columnas requeridas
        columnas_requeridas = ['actividades']
        columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
        
        if columnas_faltantes:
            raise ValueError(f"Columnas requeridas faltantes: {', '.join(columnas_faltantes)}")
        
        # Validar y preparar columnas
        if 'cantidad' not in df.columns:
            df['cantidad'] = 1
            logger.info("Columna 'cantidad' no encontrada. Se asigna valor por defecto: 1")
        else:
            # Convertir a numérico y validar
            df['cantidad'] = pd.to_numeric(df['cantidad'], errors='coerce')
            if df['cantidad'].isna().any():
                raise ValueError("La columna 'cantidad' contiene valores no numéricos")
        
        if 'costo_unitario' in df.columns:
            # Convertir a numérico y validar
            df['costo_unitario'] = pd.to_numeric(df['costo_unitario'], errors='coerce')
            # Permitir NaN en costo_unitario ya que pueden ser completados después
        
        if 'costo_total' in df.columns:
            # Convertir a numérico y validar
            df['costo_total'] = pd.to_numeric(df['costo_total'], errors='coerce')
            # Permitir NaN en costo_total ya que pueden ser calculados después
        
        # Validar que no haya filas vacías en actividades
        if df['actividades'].isna().any():
            raise ValueError("La columna 'actividades' contiene valores vacíos")
        
        # Limpiar espacios en blanco en actividades
        df['actividades'] = df['actividades'].str.strip()
        
        # Validar que no haya actividades duplicadas
        duplicados = df['actividades'].duplicated()
        if duplicados.any():
            actividades_duplicadas = df[duplicados]['actividades'].tolist()
            logger.warning(f"Actividades duplicadas encontradas: {', '.join(actividades_duplicadas)}")
        
        # Agregar columnas adicionales si no existen
        columnas_adicionales = ['unidad_medida', 'notas']
        for col in columnas_adicionales:
            if col not in df.columns:
                df[col] = None
        
        return df
        
    except Exception as e:
        logger.error(f"Error al cargar Excel: {str(e)}")
        raise

def validar_formato_excel(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Realiza validaciones adicionales en el DataFrame y retorna un reporte.
    
    Args:
        df (pd.DataFrame): DataFrame a validar
        
    Returns:
        Dict[str, Any]: Reporte de validación
    """
    reporte = {
        'valido': True,
        'errores': [],
        'advertencias': [],
        'estadisticas': {
            'total_actividades': len(df),
            'actividades_con_precio': 0,
            'actividades_sin_precio': 0
        }
    }
    
    try:
        # Validar valores negativos
        if 'cantidad' in df.columns and (df['cantidad'] < 0).any():
            reporte['errores'].append("Existen cantidades negativas")
            reporte['valido'] = False
        
        if 'costo_unitario' in df.columns:
            precios_negativos = df[df['costo_unitario'] < 0]
            if not precios_negativos.empty:
                reporte['errores'].append("Existen precios negativos")
                reporte['valido'] = False
            
            # Contar actividades con y sin precio
            reporte['estadisticas']['actividades_con_precio'] = df['costo_unitario'].notna().sum()
            reporte['estadisticas']['actividades_sin_precio'] = df['costo_unitario'].isna().sum()
        
        # Validar longitud de textos
        if (df['actividades'].str.len() > 500).any():
            reporte['advertencias'].append("Algunas descripciones de actividades son muy largas (>500 caracteres)")
        
        # Validar caracteres especiales
        caracteres_especiales = df['actividades'].str.contains(r'[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ\s\.,;:()\-]')
        if caracteres_especiales.any():
            reporte['advertencias'].append("Algunas actividades contienen caracteres especiales inusuales")
        
        # Validar consistencia en unidades de medida
        if 'unidad_medida' in df.columns:
            unidades_vacias = df['unidad_medida'].isna().sum()
            if unidades_vacias > 0:
                reporte['advertencias'].append(f"{unidades_vacias} actividades no tienen unidad de medida especificada")
        
        return reporte
        
    except Exception as e:
        logger.error(f"Error en validación de Excel: {str(e)}")
        reporte['valido'] = False
        reporte['errores'].append(str(e))
        return reporte

def generar_resumen_excel(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Genera un resumen estadístico del contenido del Excel.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
        
    Returns:
        Dict[str, Any]: Resumen estadístico
    """
    try:
        resumen = {
            'total_actividades': len(df),
            'total_estimado': 0,
            'promedio_precio_unitario': 0,
            'categorias': {},
            'distribucion_precios': {
                'bajo': 0,    # < promedio - 1 std
                'medio': 0,   # entre promedio ± 1 std
                'alto': 0     # > promedio + 1 std
            }
        }
        
        if 'costo_unitario' in df.columns and 'cantidad' in df.columns:
            # Calcular totales
            df['total'] = df['cantidad'] * df['costo_unitario']
            resumen['total_estimado'] = df['total'].sum()
            
            # Estadísticas de precios
            precios = df['costo_unitario'].dropna()
            if not precios.empty:
                resumen['promedio_precio_unitario'] = precios.mean()
                std_precio = precios.std()
                promedio = precios.mean()
                
                # Distribución de precios
                resumen['distribucion_precios']['bajo'] = len(precios[precios < (promedio - std_precio)])
                resumen['distribucion_precios']['alto'] = len(precios[precios > (promedio + std_precio)])
                resumen['distribucion_precios']['medio'] = len(precios) - (
                    resumen['distribucion_precios']['bajo'] + 
                    resumen['distribucion_precios']['alto']
                )
        
        # Análisis por categorías si existe la columna
        if 'categoria' in df.columns:
            categorias = df['categoria'].value_counts()
            for categoria, count in categorias.items():
                resumen['categorias'][categoria] = {
                    'cantidad': count,
                    'porcentaje': (count / len(df)) * 100
                }
        
        return resumen
        
    except Exception as e:
        logger.error(f"Error al generar resumen de Excel: {str(e)}")
        return {
            'error': str(e),
            'total_actividades': len(df)
        }
