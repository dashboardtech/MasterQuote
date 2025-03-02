"""
Módulo para gestión avanzada de caché con invalidación inteligente.
"""
import os
import json
import hashlib
import time
from typing import Any, Dict, Optional, List, Callable
import logging
import pandas as pd
import pickle
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Gestor de caché avanzado con soporte para diferentes estrategias de almacenamiento
    e invalidación inteligente basada en tiempos y cambios en archivos.
    """
    
    def __init__(self, 
                 cache_dir: str = ".cache", 
                 expiration_time: int = 3600,
                 max_size_mb: int = 500,
                 storage_type: str = "disk"):
        """
        Inicializa el gestor de caché.
        
        Args:
            cache_dir: Directorio para almacenar los archivos de caché
            expiration_time: Tiempo de expiración en segundos
            max_size_mb: Tamaño máximo del caché en MB
            storage_type: Tipo de almacenamiento ('memory', 'disk', 'hybrid')
        """
        self.cache_dir = cache_dir
        self.expiration_time = expiration_time
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.storage_type = storage_type
        
        # Caché en memoria
        self.memory_cache = {}
        
        # Crear directorio de caché si no existe
        if not os.path.exists(cache_dir) and storage_type in ['disk', 'hybrid']:
            os.makedirs(cache_dir)
            
        # Lock para operaciones thread-safe
        self.lock = threading.Lock()
        
        # Limpiar caché expirado al iniciar
        self.cleanup_expired()
        
    def generate_key(self, key_data: Any, prefix: str = "") -> str:
        """
        Genera una clave única para el caché basada en los datos proporcionados.
        
        Args:
            key_data: Datos para generar la clave (puede ser string, dict, etc.)
            prefix: Prefijo opcional para la clave
            
        Returns:
            Clave como string (hash)
        """
        if isinstance(key_data, str):
            data_str = key_data
        elif isinstance(key_data, (dict, list, tuple)):
            try:
                data_str = json.dumps(key_data, sort_keys=True)
            except TypeError:
                # Si no es serializable directamente a JSON
                data_str = str(key_data)
        else:
            data_str = str(key_data)
            
        key_hash = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}_{key_hash}" if prefix else key_hash
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor del caché.
        
        Args:
            key: Clave del valor a obtener
            default: Valor por defecto si no existe la clave
            
        Returns:
            Valor almacenado o default si no existe
        """
        with self.lock:
            # Verificar en memoria primero
            if self.storage_type in ['memory', 'hybrid']:
                if key in self.memory_cache:
                    cache_entry = self.memory_cache[key]
                    
                    # Verificar expiración
                    if 'expiration' in cache_entry and time.time() > cache_entry['expiration']:
                        del self.memory_cache[key]
                        return default
                    
                    # Verificar si hay información de archivo y si ha cambiado
                    if 'file_info' in cache_entry:
                        file_path = cache_entry['file_info']['path']
                        last_modified = cache_entry['file_info']['last_modified']
                        
                        if os.path.exists(file_path):
                            current_mtime = os.path.getmtime(file_path)
                            if current_mtime > last_modified:
                                # Archivo modificado, invalidar caché
                                del self.memory_cache[key]
                                return default
                    
                    logger.debug(f"Caché hit (memoria): {key}")
                    return cache_entry['data']
            
            # Verificar en disco si es necesario
            if self.storage_type in ['disk', 'hybrid']:
                cache_file = os.path.join(self.cache_dir, f"{key}.cache")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            cache_entry = pickle.load(f)
                        
                        # Verificar expiración
                        if 'expiration' in cache_entry and time.time() > cache_entry['expiration']:
                            os.remove(cache_file)
                            return default
                        
                        # Verificar si hay información de archivo y si ha cambiado
                        if 'file_info' in cache_entry:
                            file_path = cache_entry['file_info']['path']
                            last_modified = cache_entry['file_info']['last_modified']
                            
                            if os.path.exists(file_path):
                                current_mtime = os.path.getmtime(file_path)
                                if current_mtime > last_modified:
                                    # Archivo modificado, invalidar caché
                                    os.remove(cache_file)
                                    return default
                        
                        # Si estamos en modo híbrido, cargar en memoria
                        if self.storage_type == 'hybrid':
                            self.memory_cache[key] = cache_entry
                        
                        logger.debug(f"Caché hit (disco): {key}")
                        return cache_entry['data']
                    
                    except (pickle.PickleError, EOFError, IOError) as e:
                        logger.warning(f"Error al leer caché {key}: {str(e)}")
                        if os.path.exists(cache_file):
                            os.remove(cache_file)
        
        return default
    
    def set(self, 
            key: str, 
            value: Any, 
            file_path: Optional[str] = None,
            expiration: Optional[int] = None) -> None:
        """
        Almacena un valor en el caché.
        
        Args:
            key: Clave para el valor
            value: Valor a almacenar
            file_path: Ruta al archivo asociado (para invalidación basada en cambios)
            expiration: Tiempo de expiración personalizado en segundos
        """
        with self.lock:
            # Verificar espacio disponible
            if self.storage_type in ['disk', 'hybrid']:
                self._check_and_enforce_size_limit()
            
            # Crear entrada de caché
            cache_entry = {
                'data': value,
                'timestamp': time.time(),
                'expiration': time.time() + (expiration or self.expiration_time)
            }
            
            # Añadir información del archivo si se proporciona
            if file_path and os.path.exists(file_path):
                cache_entry['file_info'] = {
                    'path': file_path,
                    'last_modified': os.path.getmtime(file_path),
                    'size': os.path.getsize(file_path)
                }
            
            # Guardar en memoria si corresponde
            if self.storage_type in ['memory', 'hybrid']:
                self.memory_cache[key] = cache_entry
            
            # Guardar en disco si corresponde
            if self.storage_type in ['disk', 'hybrid']:
                cache_file = os.path.join(self.cache_dir, f"{key}.cache")
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(cache_entry, f, protocol=pickle.HIGHEST_PROTOCOL)
                except (pickle.PickleError, IOError) as e:
                    logger.warning(f"Error al escribir caché {key}: {str(e)}")
    
    def delete(self, key: str) -> bool:
        """
        Elimina una entrada del caché.
        
        Args:
            key: Clave a eliminar
            
        Returns:
            True si se eliminó correctamente, False en caso contrario
        """
        with self.lock:
            deleted = False
            
            # Eliminar de memoria
            if self.storage_type in ['memory', 'hybrid'] and key in self.memory_cache:
                del self.memory_cache[key]
                deleted = True
            
            # Eliminar de disco
            if self.storage_type in ['disk', 'hybrid']:
                cache_file = os.path.join(self.cache_dir, f"{key}.cache")
                if os.path.exists(cache_file):
                    try:
                        os.remove(cache_file)
                        deleted = True
                    except IOError as e:
                        logger.warning(f"Error al eliminar caché {key}: {str(e)}")
            
            return deleted
    
    def clear(self) -> None:
        """Limpia todo el caché."""
        with self.lock:
            # Limpiar memoria
            if self.storage_type in ['memory', 'hybrid']:
                self.memory_cache.clear()
            
            # Limpiar disco
            if self.storage_type in ['disk', 'hybrid'] and os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        try:
                            os.remove(os.path.join(self.cache_dir, filename))
                        except IOError as e:
                            logger.warning(f"Error al eliminar archivo de caché {filename}: {str(e)}")
    
    def cleanup_expired(self) -> int:
        """
        Limpia entradas expiradas del caché.
        
        Returns:
            Número de entradas eliminadas
        """
        count = 0
        current_time = time.time()
        
        with self.lock:
            # Limpiar memoria
            if self.storage_type in ['memory', 'hybrid']:
                expired_keys = [
                    k for k, v in self.memory_cache.items() 
                    if 'expiration' in v and current_time > v['expiration']
                ]
                for key in expired_keys:
                    del self.memory_cache[key]
                    count += 1
            
            # Limpiar disco
            if self.storage_type in ['disk', 'hybrid'] and os.path.exists(self.cache_dir):
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        cache_file = os.path.join(self.cache_dir, filename)
                        try:
                            with open(cache_file, 'rb') as f:
                                cache_entry = pickle.load(f)
                            
                            if 'expiration' in cache_entry and current_time > cache_entry['expiration']:
                                os.remove(cache_file)
                                count += 1
                        except (pickle.PickleError, EOFError, IOError):
                            # Si hay error al leer, eliminar el archivo
                            try:
                                os.remove(cache_file)
                                count += 1
                            except IOError:
                                pass
        
        logger.info(f"Limpiadas {count} entradas de caché expiradas")
        return count
    
    def _check_and_enforce_size_limit(self) -> None:
        """
        Verifica y aplica el límite de tamaño del caché en disco.
        Elimina las entradas más antiguas si se supera el límite.
        """
        if self.storage_type not in ['disk', 'hybrid'] or not os.path.exists(self.cache_dir):
            return
        
        # Obtener tamaño actual
        total_size = 0
        cache_files = []
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.cache'):
                file_path = os.path.join(self.cache_dir, filename)
                file_size = os.path.getsize(file_path)
                file_mtime = os.path.getmtime(file_path)
                
                total_size += file_size
                cache_files.append({
                    'path': file_path,
                    'size': file_size,
                    'mtime': file_mtime
                })
        
        # Si se supera el límite, eliminar los más antiguos
        if total_size > self.max_size_bytes and cache_files:
            # Ordenar por tiempo de modificación (más antiguos primero)
            cache_files.sort(key=lambda x: x['mtime'])
            
            # Eliminar hasta estar por debajo del límite
            for file_info in cache_files:
                if total_size <= self.max_size_bytes:
                    break
                
                try:
                    os.remove(file_info['path'])
                    total_size -= file_info['size']
                    logger.debug(f"Eliminado caché antiguo: {os.path.basename(file_info['path'])}")
                except IOError as e:
                    logger.warning(f"Error al eliminar caché antiguo: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché.
        
        Returns:
            Diccionario con estadísticas
        """
        stats = {
            'storage_type': self.storage_type,
            'expiration_time': self.expiration_time,
            'max_size_mb': self.max_size_bytes / (1024 * 1024)
        }
        
        # Estadísticas de memoria
        if self.storage_type in ['memory', 'hybrid']:
            stats['memory_entries'] = len(self.memory_cache)
        
        # Estadísticas de disco
        if self.storage_type in ['disk', 'hybrid'] and os.path.exists(self.cache_dir):
            disk_entries = 0
            disk_size = 0
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    disk_entries += 1
                    disk_size += os.path.getsize(os.path.join(self.cache_dir, filename))
            
            stats['disk_entries'] = disk_entries
            stats['disk_size_mb'] = disk_size / (1024 * 1024)
        
        return stats


def cached(cache_manager: CacheManager, 
           key_prefix: str = "", 
           expire_time: Optional[int] = None):
    """
    Decorador para cachear el resultado de una función.
    
    Args:
        cache_manager: Instancia de CacheManager
        key_prefix: Prefijo para la clave de caché
        expire_time: Tiempo de expiración personalizado
        
    Returns:
        Decorador configurado
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generar clave basada en función y argumentos
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = cache_manager.generate_key(key_data, prefix=key_prefix)
            
            # Intentar obtener del caché
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en caché
            cache_manager.set(cache_key, result, expiration=expire_time)
            
            return result
        return wrapper
    return decorator


def cached_df(cache_manager: CacheManager, 
              key_prefix: str = "", 
              expire_time: Optional[int] = None,
              file_arg_name: Optional[str] = None):
    """
    Decorador especializado para cachear DataFrames de pandas.
    
    Args:
        cache_manager: Instancia de CacheManager
        key_prefix: Prefijo para la clave de caché
        expire_time: Tiempo de expiración personalizado
        file_arg_name: Nombre del argumento que contiene la ruta del archivo
        
    Returns:
        Decorador configurado
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generar clave basada en función y argumentos
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = cache_manager.generate_key(key_data, prefix=key_prefix)
            
            # Obtener ruta de archivo si se especificó
            file_path = None
            if file_arg_name:
                if file_arg_name in kwargs:
                    file_path = kwargs[file_arg_name]
                else:
                    # Verificar en args basado en la firma de la función
                    import inspect
                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    
                    if file_arg_name in params:
                        idx = params.index(file_arg_name)
                        if idx < len(args):
                            file_path = args[idx]
            
            # Intentar obtener del caché
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                # Verificar que sea un DataFrame
                if isinstance(cached_result, pd.DataFrame):
                    return cached_result
            
            # Ejecutar función
            result = func(*args, **kwargs)
            
            # Guardar en caché solo si es un DataFrame
            if isinstance(result, pd.DataFrame):
                cache_manager.set(cache_key, result, file_path=file_path, expiration=expire_time)
            
            return result
        return wrapper
    return decorator
