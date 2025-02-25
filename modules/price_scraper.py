import aiohttp
import asyncio
from bs4 import BeautifulSoup
import logging
from typing import Optional, Dict, Any
import yaml
import json
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def obtener_precio(url: str, selector: str, precio_default: Optional[float] = None) -> float:
    """
    Obtiene el precio de una actividad desde una fuente web.
    
    Args:
        url (str): URL de la página web
        selector (str): Selector CSS para encontrar el precio
        precio_default (float, optional): Precio por defecto si falla el scraping
        
    Returns:
        float: Precio obtenido o precio por defecto
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Error al acceder a {url}: {response.status}")
                    return precio_default if precio_default is not None else 0.0
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                # Buscar el elemento con el precio
                elemento_precio = soup.select_one(selector)
                if not elemento_precio:
                    logger.warning(f"No se encontró el elemento con selector: {selector}")
                    return precio_default if precio_default is not None else 0.0
                
                # Extraer y limpiar el precio
                texto_precio = elemento_precio.get_text().strip()
                # Eliminar símbolos de moneda y caracteres no numéricos
                precio_limpio = ''.join(c for c in texto_precio if c.isdigit() or c == '.' or c == ',')
                precio_limpio = precio_limpio.replace(',', '.')
                
                try:
                    precio = float(precio_limpio)
                    logger.info(f"Precio obtenido de {url}: {precio}")
                    return precio
                except ValueError:
                    logger.error(f"No se pudo convertir el precio: {texto_precio}")
                    return precio_default if precio_default is not None else 0.0
                
    except Exception as e:
        logger.error(f"Error al obtener precio de {url}: {str(e)}")
        return precio_default if precio_default is not None else 0.0

async def obtener_precios_batch(configuracion: Dict[str, Any]) -> Dict[str, float]:
    """
    Obtiene precios para múltiples actividades en paralelo.
    
    Args:
        configuracion (Dict[str, Any]): Diccionario con la configuración de scraping
        
    Returns:
        Dict[str, float]: Diccionario con los precios obtenidos
    """
    try:
        tareas = []
        actividades = []
        
        for actividad, config in configuracion.items():
            if 'url' in config and 'selector' in config:
                tareas.append(
                    obtener_precio(
                        config['url'],
                        config['selector'],
                        config.get('default_price', 0.0)
                    )
                )
                actividades.append(actividad)
        
        if not tareas:
            return {}
        
        # Ejecutar todas las tareas en paralelo
        resultados = await asyncio.gather(*tareas, return_exceptions=True)
        
        # Procesar resultados
        precios = {}
        for actividad, resultado in zip(actividades, resultados):
            if isinstance(resultado, Exception):
                logger.error(f"Error al obtener precio para {actividad}: {str(resultado)}")
                precios[actividad] = configuracion[actividad].get('default_price', 0.0)
            else:
                precios[actividad] = resultado
        
        return precios
        
    except Exception as e:
        logger.error(f"Error al obtener precios en batch: {str(e)}")
        return {}

def guardar_cache_precios(precios: Dict[str, float], archivo: str = "cache_precios.json"):
    """
    Guarda los precios obtenidos en un archivo de caché.
    
    Args:
        precios (Dict[str, float]): Diccionario con los precios a guardar
        archivo (str): Ruta al archivo de caché
    """
    try:
        cache = {
            'ultima_actualizacion': datetime.now().isoformat(),
            'precios': precios
        }
        
        with open(archivo, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Caché de precios guardado en: {archivo}")
        
    except Exception as e:
        logger.error(f"Error al guardar caché de precios: {str(e)}")

def cargar_cache_precios(archivo: str = "cache_precios.json") -> Dict[str, float]:
    """
    Carga los precios desde el archivo de caché.
    
    Args:
        archivo (str): Ruta al archivo de caché
        
    Returns:
        Dict[str, float]: Diccionario con los precios cacheados
    """
    try:
        with open(archivo, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            
        # Verificar si el caché está vigente (menos de 24 horas)
        ultima_actualizacion = datetime.fromisoformat(cache['ultima_actualizacion'])
        tiempo_transcurrido = datetime.now() - ultima_actualizacion
        
        if tiempo_transcurrido.total_seconds() > 86400:  # 24 horas
            logger.warning("Caché de precios expirado")
            return {}
            
        return cache['precios']
        
    except FileNotFoundError:
        logger.info("No se encontró archivo de caché")
        return {}
    except Exception as e:
        logger.error(f"Error al cargar caché de precios: {str(e)}")
        return {}

async def actualizar_precios_automaticamente(intervalo: int = 3600):
    """
    Actualiza los precios automáticamente cada cierto intervalo.
    
    Args:
        intervalo (int): Intervalo en segundos entre actualizaciones
    """
    try:
        while True:
            logger.info("Iniciando actualización automática de precios")
            
            # Cargar configuración
            with open("config.yaml", 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'precios_unitarios' in config:
                # Obtener precios actualizados
                precios = await obtener_precios_batch(config['precios_unitarios'])
                
                # Guardar en caché
                if precios:
                    guardar_cache_precios(precios)
            
            # Esperar hasta la próxima actualización
            await asyncio.sleep(intervalo)
            
    except Exception as e:
        logger.error(f"Error en actualización automática: {str(e)}")
        # Reintentar después de un error
        await asyncio.sleep(60)
        await actualizar_precios_automaticamente(intervalo)

async def iniciar_actualizacion_automatica():
    """Inicia el proceso de actualización automática de precios."""
    try:
        # Crear tarea de actualización
        tarea = asyncio.create_task(actualizar_precios_automaticamente())
        logger.info("Actualización automática de precios iniciada")
        
        # Mantener la tarea corriendo
        await tarea
        
    except Exception as e:
        logger.error(f"Error al iniciar actualización automática: {str(e)}")
