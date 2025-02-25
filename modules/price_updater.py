import pandas as pd
import logging
from typing import Dict, Any, Optional
import yaml
from .price_scraper import cargar_cache_precios

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def actualizar_precios(df: pd.DataFrame, 
                      precios_unitarios: Optional[Dict[str, float]] = None,
                      config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Actualiza los precios en el DataFrame usando diferentes fuentes.
    
    Args:
        df (pd.DataFrame): DataFrame con actividades
        precios_unitarios (Dict[str, float], optional): Precios predefinidos
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        pd.DataFrame: DataFrame con precios actualizados
    """
    try:
        # Crear copia del DataFrame para no modificar el original
        df_actualizado = df.copy()
        
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Cargar precios del caché
        precios_cache = cargar_cache_precios()
        
        # Obtener precios fijos de la configuración
        precios_fijos = {}
        if 'precios_unitarios' in config:
            for actividad, detalles in config['precios_unitarios'].items():
                if 'precio_fijo' in detalles:
                    precios_fijos[actividad.lower()] = detalles['precio_fijo']
        
        # Procesar cada fila
        for idx, row in df_actualizado.iterrows():
            actividad = row['actividades'].lower().strip()
            
            # Si ya tiene precio y es válido, continuar
            if pd.notna(row['costo_unitario']) and row['costo_unitario'] > 0:
                continue
            
            precio_asignado = None
            fuente_precio = None
            
            # 1. Intentar usar precios unitarios proporcionados
            if precios_unitarios and actividad in precios_unitarios:
                precio_asignado = precios_unitarios[actividad]
                fuente_precio = 'proporcionado'
            
            # 2. Intentar usar precios del caché
            if precio_asignado is None and actividad in precios_cache:
                precio_asignado = precios_cache[actividad]
                fuente_precio = 'cache'
            
            # 3. Intentar usar precios fijos de configuración
            if precio_asignado is None and actividad in precios_fijos:
                precio_asignado = precios_fijos[actividad]
                fuente_precio = 'configuracion'
            
            # Actualizar precio si se encontró uno
            if precio_asignado is not None:
                df_actualizado.at[idx, 'costo_unitario'] = precio_asignado
                
                # Agregar fuente del precio si existe la columna
                if 'fuente_precio' not in df_actualizado.columns:
                    df_actualizado['fuente_precio'] = None
                df_actualizado.at[idx, 'fuente_precio'] = fuente_precio
                
                # Calcular costo total si existe cantidad
                if 'cantidad' in df_actualizado.columns:
                    df_actualizado.at[idx, 'costo_total'] = (
                        precio_asignado * df_actualizado.at[idx, 'cantidad']
                    )
        
        # Registrar estadísticas de actualización
        total_actualizados = df_actualizado['costo_unitario'].notna().sum()
        total_pendientes = df_actualizado['costo_unitario'].isna().sum()
        
        logger.info(f"Precios actualizados: {total_actualizados}")
        logger.info(f"Precios pendientes: {total_pendientes}")
        
        return df_actualizado
        
    except Exception as e:
        logger.error(f"Error al actualizar precios: {str(e)}")
        return df

def aplicar_ajustes(df: pd.DataFrame, 
                    ajustes: Dict[str, Any],
                    config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Aplica ajustes a los precios según reglas configuradas.
    
    Args:
        df (pd.DataFrame): DataFrame con precios
        ajustes (Dict[str, Any]): Reglas de ajuste
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        pd.DataFrame: DataFrame con precios ajustados
    """
    try:
        df_ajustado = df.copy()
        
        # Cargar configuración
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Aplicar ajustes globales
        if 'ajuste_global' in ajustes:
            porcentaje = ajustes['ajuste_global']
            mask = df_ajustado['costo_unitario'].notna()
            df_ajustado.loc[mask, 'costo_unitario'] *= (1 + porcentaje/100)
        
        # Aplicar ajustes por categoría
        if 'ajustes_categoria' in ajustes:
            for categoria, porcentaje in ajustes['ajustes_categoria'].items():
                mask = (df_ajustado['categoria'] == categoria) & df_ajustado['costo_unitario'].notna()
                df_ajustado.loc[mask, 'costo_unitario'] *= (1 + porcentaje/100)
        
        # Aplicar ajustes específicos por actividad
        if 'ajustes_actividad' in ajustes:
            for actividad, porcentaje in ajustes['ajustes_actividad'].items():
                mask = (df_ajustado['actividades'].str.lower() == actividad.lower()) & df_ajustado['costo_unitario'].notna()
                df_ajustado.loc[mask, 'costo_unitario'] *= (1 + porcentaje/100)
        
        # Aplicar reglas de redondeo
        if 'redondeo' in ajustes:
            decimales = ajustes['redondeo']
            mask = df_ajustado['costo_unitario'].notna()
            df_ajustado.loc[mask, 'costo_unitario'] = df_ajustado.loc[mask, 'costo_unitario'].round(decimales)
        
        # Recalcular costos totales
        if 'cantidad' in df_ajustado.columns:
            mask = df_ajustado['costo_unitario'].notna()
            df_ajustado.loc[mask, 'costo_total'] = (
                df_ajustado.loc[mask, 'costo_unitario'] * df_ajustado.loc[mask, 'cantidad']
            )
        
        # Registrar ajustes aplicados
        logger.info("Ajustes aplicados:")
        for tipo_ajuste, valor in ajustes.items():
            logger.info(f"- {tipo_ajuste}: {valor}")
        
        return df_ajustado
        
    except Exception as e:
        logger.error(f"Error al aplicar ajustes: {str(e)}")
        return df

def validar_precios(df: pd.DataFrame, 
                    limites: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Valida los precios según límites establecidos.
    
    Args:
        df (pd.DataFrame): DataFrame con precios
        limites (Dict[str, Dict[str, float]], optional): Límites de precios por categoría
        
    Returns:
        Dict[str, Any]: Reporte de validación
    """
    try:
        reporte = {
            'valido': True,
            'errores': [],
            'advertencias': [],
            'estadisticas': {
                'total_items': len(df),
                'items_validados': 0,
                'items_con_errores': 0
            }
        }
        
        # Validar que haya precios
        if 'costo_unitario' not in df.columns:
            reporte['errores'].append("No se encontró la columna de precios unitarios")
            reporte['valido'] = False
            return reporte
        
        # Contar items validados
        items_con_precio = df['costo_unitario'].notna()
        reporte['estadisticas']['items_validados'] = items_con_precio.sum()
        
        # Validar precios negativos o cero
        precios_invalidos = df[items_con_precio & (df['costo_unitario'] <= 0)]
        if not precios_invalidos.empty:
            reporte['errores'].append(f"Se encontraron {len(precios_invalidos)} precios negativos o cero")
            reporte['valido'] = False
            reporte['estadisticas']['items_con_errores'] += len(precios_invalidos)
        
        # Validar límites por categoría si se proporcionan
        if limites and 'categoria' in df.columns:
            for categoria, rango in limites.items():
                mask_categoria = (df['categoria'] == categoria) & items_con_precio
                if mask_categoria.any():
                    precios_categoria = df.loc[mask_categoria, 'costo_unitario']
                    
                    # Verificar mínimo
                    if 'min' in rango:
                        bajo_minimo = precios_categoria < rango['min']
                        if bajo_minimo.any():
                            reporte['advertencias'].append(
                                f"Categoría {categoria}: {bajo_minimo.sum()} precios bajo el mínimo ({rango['min']})"
                            )
                    
                    # Verificar máximo
                    if 'max' in rango:
                        sobre_maximo = precios_categoria > rango['max']
                        if sobre_maximo.any():
                            reporte['advertencias'].append(
                                f"Categoría {categoria}: {sobre_maximo.sum()} precios sobre el máximo ({rango['max']})"
                            )
        
        # Validar consistencia en precios similares
        precios_agrupados = df[items_con_precio].groupby('actividades')['costo_unitario'].agg(['mean', 'std'])
        variaciones_altas = precios_agrupados[precios_agrupados['std'] > precios_agrupados['mean'] * 0.2]
        
        if not variaciones_altas.empty:
            reporte['advertencias'].append(
                f"Se encontraron {len(variaciones_altas)} actividades con variación alta en precios"
            )
        
        return reporte
        
    except Exception as e:
        logger.error(f"Error en validación de precios: {str(e)}")
        return {
            'valido': False,
            'errores': [str(e)],
            'advertencias': [],
            'estadisticas': {
                'total_items': len(df),
                'items_validados': 0,
                'items_con_errores': 0
            }
        }
