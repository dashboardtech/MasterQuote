from abc import ABC, abstractmethod
import pandas as pd
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """Clase base abstracta para todos los extractores de precios."""
    
    @abstractmethod
    def extract(self, file_path: str, interactive: bool = False, process_all_sheets: bool = False) -> pd.DataFrame:
        """
        Extrae datos de un archivo.
        
        Args:
            file_path: Ruta al archivo
            interactive: Si debe ser interactivo
            process_all_sheets: Si debe procesar todas las hojas en caso de archivos con múltiples hojas
            
        Returns:
            DataFrame con actividades y precios
        """
        pass
    
    def _normalize_price_column(self, series):
        """Normaliza una columna de precios a valores numéricos."""
        import re
        try:
            # Manejar el caso de Series de tipo object (texto)
            if series.dtype == object:
                # Limpiar formato de moneda y separadores
                clean_series = series.astype(str).apply(self._clean_price_string)
                return pd.to_numeric(clean_series, errors='coerce')
            else:
                # Ya es numérico
                return series
        except Exception as e:
            logger.warning(f"Error al normalizar columna de precios: {str(e)}")
            return series
            
    def _clean_price_string(self, price_str):
        """Limpia un string que representa un precio para convertirlo a número."""
        import re
        if not isinstance(price_str, str):
            return price_str
            
        # Eliminar símbolos de moneda y espacios
        clean = re.sub(r'[$€£¥]', '', price_str)
        
        # Manejar diferentes formatos de números
        if ',' in clean and '.' in clean:
            # Determinar cuál es el separador decimal
            if clean.rindex('.') > clean.rindex(','):
                # El punto es el separador decimal (formato USA)
                clean = clean.replace(',', '')
            else:
                # La coma es el separador decimal (formato europeo)
                clean = clean.replace('.', '').replace(',', '.')
        elif ',' in clean:
            # Si solo hay comas, asumir que es separador decimal
            clean = clean.replace(',', '.')
        
        # Eliminar caracteres no numéricos y conservar el signo y punto decimal
        clean = re.sub(r'[^\d.-]', '', clean)
        
        return clean.strip()
