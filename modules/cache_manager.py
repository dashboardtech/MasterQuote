import os
import json
import hashlib
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Gestiona un sistema de caché para evitar reprocesar archivos idénticos.
    Calcula un hash único para cada archivo y guarda/recupera resultados procesados.
    """
    
    def __init__(self, cache_dir=None, expiry_days=30):
        """
        Inicializa el gestor de caché.
        
        Args:
            cache_dir: Directorio para almacenar archivos de caché
            expiry_days: Días tras los cuales la caché se considera obsoleta
        """
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), ".price_extractor_cache")
        self.expiry_days = expiry_days
        
        # Crear directorio de caché si no existe
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Directorio de caché creado: {self.cache_dir}")
        
        # Estadísticas de uso
        self.cache_hits = 0
        self.cache_misses = 0
    
    def calculate_file_hash(self, filepath):
        """
        Calcula un hash MD5 único para el archivo.
        
        Args:
            filepath: Ruta al archivo
            
        Returns:
            str: Hash MD5 del archivo
        """
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read(65536)  # Leer en bloques de 64k
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    
    def get_cache_path(self, filepath):
        """
        Obtiene la ruta al archivo de caché.
        
        Args:
            filepath: Ruta al archivo original
            
        Returns:
            str: Ruta al archivo de caché
        """
        file_hash = self.calculate_file_hash(filepath)
        return os.path.join(self.cache_dir, f"{file_hash}.json")
    
    def is_cache_valid(self, cache_path):
        """
        Verifica si la caché es válida y no ha expirado.
        
        Args:
            cache_path: Ruta al archivo de caché
            
        Returns:
            bool: True si la caché es válida, False en caso contrario
        """
        if not os.path.exists(cache_path):
            return False
        
        # Verificar fecha de caché
        if self.expiry_days > 0:
            file_time = os.path.getmtime(cache_path)
            file_date = datetime.fromtimestamp(file_time)
            days_old = (datetime.now() - file_date).days
            
            if days_old > self.expiry_days:
                logger.info(f"Caché expirada ({days_old} días): {cache_path}")
                return False
        
        return True
    
    def save_to_cache(self, filepath, dataframe):
        """
        Guarda un DataFrame procesado en la caché.
        
        Args:
            filepath: Ruta al archivo original
            dataframe: DataFrame a guardar
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            cache_path = self.get_cache_path(filepath)
            
            # Preparar metadatos
            metadata = {
                "original_file": os.path.basename(filepath),
                "cache_date": datetime.now().isoformat(),
                "file_size": os.path.getsize(filepath),
                "file_modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
            }
            
            # Combinar datos y metadatos
            cache_data = {
                "metadata": metadata,
                "data": dataframe.to_dict(orient="records")
            }
            
            # Guardar a archivo JSON
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Datos guardados en caché: {cache_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Error al guardar en caché: {str(e)}")
            return False
    
    def load_from_cache(self, filepath):
        """
        Carga datos procesados desde la caché.
        
        Args:
            filepath: Ruta al archivo original
            
        Returns:
            pd.DataFrame: DataFrame cargado desde caché, o None si no existe
        """
        try:
            cache_path = self.get_cache_path(filepath)
            
            if not self.is_cache_valid(cache_path):
                self.cache_misses += 1
                return None
            
            # Cargar datos desde JSON
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Extraer metadatos
            metadata = cache_data.get("metadata", {})
            cache_date = metadata.get("cache_date", "desconocido")
            
            # Convertir a DataFrame
            df = pd.DataFrame(cache_data.get("data", []))
            
            if not df.empty:
                logger.info(f"Datos cargados desde caché ({cache_date}): {cache_path}")
                self.cache_hits += 1
                return df
            else:
                logger.warning(f"Caché vacía: {cache_path}")
                self.cache_misses += 1
                return None
                
        except Exception as e:
            logger.warning(f"Error al cargar desde caché: {str(e)}")
            self.cache_misses += 1
            return None
    
    def clear_expired_cache(self):
        """
        Elimina archivos de caché expirados.
        
        Returns:
            int: Número de archivos eliminados
        """
        if self.expiry_days <= 0:
            return 0
        
        deleted_count = 0
        now = datetime.now()
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                file_time = os.path.getmtime(file_path)
                file_date = datetime.fromtimestamp(file_time)
                days_old = (now - file_date).days
                
                if days_old > self.expiry_days:
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"Caché eliminada ({days_old} días): {filename}")
        
        return deleted_count
    
    def clear_all_cache(self):
        """
        Elimina todos los archivos de caché.
        
        Returns:
            int: Número de archivos eliminados
        """
        deleted_count = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                os.remove(file_path)
                deleted_count += 1
        
        logger.info(f"Caché limpiada: {deleted_count} archivos eliminados")
        return deleted_count
    
    def get_cache_stats(self):
        """
        Obtiene estadísticas de uso de la caché.
        
        Returns:
            dict: Estadísticas de caché
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Contar y calcular tamaño total de archivos en caché
        file_count = 0
        total_size = 0
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.cache_dir, filename)
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cached_files": file_count,
            "cache_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_directory": self.cache_dir
        }
