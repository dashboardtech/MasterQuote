import os
import sys
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_docling():
    """Prueba la instalación y funcionalidad básica de DocLing."""
    
    # Verificar si DocLing está instalado
    try:
        import docling
        logger.info(f"DocLing está instalado en: {os.path.dirname(docling.__file__)}")
        logger.info(f"Ruta de instalación: {os.path.dirname(docling.__file__)}")
    except ImportError as e:
        logger.error(f"No se pudo importar DocLing: {e}")
        return False
    
    # Probar funcionalidad básica
    try:
        from docling.backend.document import Document
        from docling.backend.extractors import TableExtractor
        logger.info("Clases básicas disponibles")
        
        # Crear un documento de prueba
        logger.info("Intentando crear un documento de prueba...")
        doc = Document.from_text("Este es un texto de prueba con un precio de $1000 por servicio de instalación.")
        logger.info("Documento creado exitosamente")
        
        # Probar extractor de tablas
        extractor = TableExtractor()
        logger.info("Extractor de tablas creado exitosamente")
        
        return True
        
    except Exception as e:
        logger.error(f"Error al probar funcionalidad básica: {e}")
        return False

def test_docling_adapter():
    """Prueba el adaptador de DocLing."""
    try:
        # Importar el adaptador
        sys.path.append(str(Path(__file__).parent.parent))
        from modules.docling_adapter import DocLingAdapter
        
        # Crear instancia del adaptador
        adapter = DocLingAdapter()
        logger.info("Adaptador creado exitosamente")
        
        # Probar extracción de entidades de precio
        text = "El servicio cuesta $1500 más IVA. La instalación tiene un costo adicional de 500 pesos."
        entities = adapter.extract_price_entities(text)
        logger.info(f"Entidades de precio encontradas: {entities}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error al probar el adaptador: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando pruebas de DocLing...")
    
    if test_docling():
        logger.info("✅ Prueba de DocLing exitosa")
    else:
        logger.error("❌ Prueba de DocLing falló")
    
    if test_docling_adapter():
        logger.info("✅ Prueba del adaptador exitosa")
    else:
        logger.error("❌ Prueba del adaptador falló")
