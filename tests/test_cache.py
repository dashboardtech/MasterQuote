import os
import sys
import time

# Añadir el directorio raíz al path para poder importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.universal_price_extractor import UniversalPriceExtractor

def test_cache_performance(file_path):
    """
    Prueba el rendimiento del sistema de caché.
    
    Args:
        file_path: Ruta a un archivo de ejemplo para procesar
    """
    print(f"Probando rendimiento de caché con archivo: {file_path}")
    
    # Verificar que el archivo existe
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe")
        return
    
    # Inicializar extractor con caché
    extractor = UniversalPriceExtractor(use_cache=True, cache_expiry_days=30)
    
    # Primera extracción (debería ser lenta)
    print("\nPrimera extracción (sin caché)...")
    start_time = time.time()
    df1 = extractor.extract_from_file(file_path)
    elapsed1 = time.time() - start_time
    print(f"Tiempo: {elapsed1:.2f} segundos")
    print(f"Filas extraídas: {len(df1)}")
    
    # Segunda extracción (debería ser mucho más rápida)
    print("\nSegunda extracción (con caché)...")
    start_time = time.time()
    df2 = extractor.extract_from_file(file_path)
    elapsed2 = time.time() - start_time
    print(f"Tiempo: {elapsed2:.2f} segundos")
    print(f"Filas extraídas: {len(df2)}")
    
    # Mostrar mejora
    if elapsed2 > 0:
        speedup = elapsed1 / elapsed2
        print(f"\nAceleración: {speedup:.1f}x más rápido")
    else:
        print("\nAceleración: Instantáneo")
    
    # Mostrar estadísticas de caché
    if hasattr(extractor, 'cache_manager'):
        stats = extractor.cache_manager.get_cache_stats()
        print("\nEstadísticas de caché:")
        for key, value in stats.items():
            print(f"- {key}: {value}")

if __name__ == "__main__":
    # Verificar si se proporcionó un archivo como argumento
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Buscar un archivo Excel en el directorio actual
        excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
        if excel_files:
            file_path = excel_files[0]
            print(f"Usando archivo encontrado: {file_path}")
        else:
            print("Error: No se proporcionó un archivo y no se encontraron archivos Excel en el directorio actual")
            print("Uso: python test_cache.py <ruta_al_archivo>")
            sys.exit(1)
    
    test_cache_performance(file_path)
