import logging
from modules.docling_adapter import DocLingAdapter
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO)

def main():
    # Crear instancia del adaptador
    adapter = DocLingAdapter()
    
    # Procesar el archivo de ejemplo
    file_path = "tests/ejemplo_precios.xlsx"
    
    try:
        # Procesar el documento
        df = adapter.process_document(file_path)
        
        if df is not None:
            print("\nResultados del procesamiento:")
            print("----------------------------")
            print("\nColumnas encontradas:", df.columns.tolist())
            print("\nPrimeras 5 filas:")
            print(df.head())
            print("\nInformación del DataFrame:")
            print(df.info())
        else:
            print("No se pudo extraer información del documento")
            
    except Exception as e:
        print(f"Error al procesar el documento: {e}")

if __name__ == "__main__":
    main()
