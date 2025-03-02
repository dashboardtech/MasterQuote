import logging
from typing import Optional
from pathlib import Path
from .base import BaseExtractor
from .excel import ExcelExtractor
from .csv import CSVExtractor
from .pdf import PDFExtractor
from .ai_assisted import AIAssistedExtractor
from .dockling import DocklingExtractor

logger = logging.getLogger(__name__)

class ExtractorFactory:
    """Fábrica para crear el extractor adecuado según el tipo de archivo."""
    
    @staticmethod
    def create_extractor(file_path: str, api_key: Optional[str] = None, use_dockling: bool = False) -> BaseExtractor:
        """
        Crea el extractor adecuado para el tipo de archivo.
        
        Args:
            file_path: Ruta al archivo
            api_key: API key para servicios de IA (opcional)
            use_dockling: Si debe usar el extractor de Dockling en lugar de los específicos
            
        Returns:
            Un extractor apropiado para el tipo de archivo
            
        Raises:
            ValueError: Si el formato no es soportado
        """
        # Si se especifica usar Dockling, retornar ese extractor
        if use_dockling:
            return DocklingExtractor(api_key=api_key)
        
        file_ext = Path(file_path).suffix.lower()
        
        # Mapeo de extensiones a extractores
        extractors = {
            '.xlsx': ExcelExtractor,
            '.xls': ExcelExtractor,
            '.xlsm': ExcelExtractor,
            '.csv': CSVExtractor,
            '.pdf': PDFExtractor,
            '.docx': lambda: AIAssistedExtractor(api_key),
            '.doc': lambda: AIAssistedExtractor(api_key),
            '.jpg': lambda: AIAssistedExtractor(api_key),
            '.jpeg': lambda: AIAssistedExtractor(api_key),
            '.png': lambda: AIAssistedExtractor(api_key),
            '.tiff': lambda: AIAssistedExtractor(api_key),
            '.bmp': lambda: AIAssistedExtractor(api_key),
            '.txt': lambda: AIAssistedExtractor(api_key)
        }
        
        # Obtener el constructor del extractor
        extractor_class = extractors.get(file_ext)
        
        if extractor_class is None:
            # Si el formato no es soportado por extractores específicos,
            # intentar con Dockling
            logger.info(f"Formato {file_ext} no soportado por extractores específicos, usando Dockling")
            try:
                return DocklingExtractor(api_key=api_key)
            except Exception as e:
                logger.error(f"Error al crear extractor Dockling: {str(e)}")
                # Fallback a AIAssistedExtractor como último recurso
                logger.info("Usando AIAssistedExtractor como fallback")
                return AIAssistedExtractor(api_key=api_key)
        
        try:
            # Si es una función lambda (para AIAssistedExtractor), llamarla
            if callable(extractor_class) and not isinstance(extractor_class, type):
                return extractor_class()
            
            # Para otros extractores, crear una instancia directamente
            return extractor_class()
        except Exception as e:
            logger.error(f"Error al crear extractor para {file_ext}: {str(e)}")
            # Fallback a Dockling como último recurso
            logger.info("Usando Dockling como fallback")
            try:
                return DocklingExtractor(api_key=api_key)
            except Exception as e2:
                logger.error(f"Error al crear extractor Dockling de fallback: {str(e2)}")
                raise ValueError(f"No se pudo crear un extractor para el archivo {file_path}: {str(e)}")
