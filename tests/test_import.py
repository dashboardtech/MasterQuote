import os
import sys
import logging

# Configurar logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Añadir el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.universal_price_extractor import UniversalPriceExtractor

def test_excel_import(file_path):
    """Prueba la importación de un archivo Excel."""
    logger.info(f"\nProbando importación de archivo: {file_path}")
    
    # Crear extractor
    extractor = UniversalPriceExtractor(use_cache=True)
    
    try:
        # Intentar extraer datos
        start_time = time.time()
        df = extractor.extract_from_file(file_path)
        elapsed = time.time() - start_time
        
        # Mostrar información del DataFrame
        logger.info("\nInformación del DataFrame:")
        logger.info(f"Tiempo de procesamiento: {elapsed:.2f} segundos")
        logger.info(f"Columnas: {df.columns.tolist()}")
        logger.info(f"Número de filas: {len(df)}")
        logger.info("\nPrimeras filas:")
        print(df.head())
        print("\n" + "-"*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error al importar archivo: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import time
    
    # Probar archivo original (debe estar en caché)
    file1 = os.path.join(os.path.dirname(__file__), 'ejemplo_precios.xlsx')
    test_excel_import(file1)
    
    # Probar nuevo archivo (sin caché)
    file2 = os.path.join(os.path.dirname(__file__), 'ejemplo_precios_2.xlsx')
    test_excel_import(file2)
    
    # Probar nuevo archivo otra vez (ahora debe estar en caché)
    test_excel_import(file2)
