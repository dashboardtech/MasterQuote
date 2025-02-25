# Sistema Inteligente de Cotizaciones con IA

Sistema avanzado para la gestión de cotizaciones con asistencia de inteligencia artificial para sugerencia de precios y extracción automática de datos desde múltiples formatos.

## Características Principales

- **Sugerencia Inteligente de Precios**: Utiliza IA para sugerir precios basados en históricos y contexto.
- **Extractor Universal de Precios**: Procesa archivos en múltiples formatos (Excel, CSV, PDF, Word, imágenes).
- **Base de Datos Histórica**: Almacena y aprende de cotizaciones anteriores.
- **Interfaz Intuitiva**: Interfaz web fácil de usar con Streamlit.
- **Exportación Profesional**: Genera cotizaciones en formato Excel con diseño profesional.

## Requisitos

- Python 3.8 o superior
- Dependencias listadas en `requirements.txt`
- Tesseract OCR (para procesamiento de imágenes y PDFs)
- API key de OpenAI (opcional, para funcionalidades avanzadas de IA)

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/MasterQuote.git
   cd MasterQuote
   ```

2. Crea un entorno virtual e instala las dependencias:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Instala Tesseract OCR:
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr tesseract-ocr-spa`
   - **Windows**: Descarga e instala desde [aquí](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`

4. Configura las variables de entorno:
   ```bash
   cp .env.example .env
   # Edita el archivo .env y agrega tu API key de OpenAI
   ```

## Uso

1. Inicia la aplicación:
   ```bash
   streamlit run app.py
   ```

2. Accede a la interfaz web en tu navegador (por defecto en http://localhost:8501)

3. Utiliza las diferentes pestañas para:
   - **Nueva Cotización**: Crear cotizaciones con sugerencias de precios
   - **Importar Precios**: Extraer precios de diversos formatos de archivo
   - **Histórico de Precios**: Consultar precios históricos
   - **Administrar Datos**: Gestionar la base de datos y categorías

## Estructura del Proyecto

```
MasterQuote/
├── app.py                  # Aplicación principal de Streamlit
├── config.yaml             # Configuración del sistema
├── requirements.txt        # Dependencias del proyecto
├── price_history.db        # Base de datos SQLite
├── .env                    # Variables de entorno (API keys)
├── modules/
│   ├── __init__.py
│   ├── price_database.py   # Gestión de base de datos
│   ├── cotizacion_llm.py   # Integración con IA
│   ├── data_loader.py      # Carga de datos de Excel
│   ├── price_scraper.py    # Scraping de precios
│   ├── price_updater.py    # Actualización de precios
│   ├── exporter.py         # Exportación de resultados
│   └── universal_price_extractor.py  # Extractor universal
├── exports/                # Directorio para archivos exportados
└── templates/              # Plantillas para exportación
```

## Funcionalidades Detalladas

### Extractor Universal de Precios

El sistema puede procesar y extraer información de precios de:

- **Excel/CSV**: Detección automática de columnas relevantes
- **PDF**: Extracción de tablas y procesamiento OCR
- **Word**: Análisis de contenido tabular
- **Imágenes**: OCR para reconocimiento de texto y tablas
- **Texto plano**: Análisis estructurado

Para formatos complejos, el sistema utiliza IA para interpretar y estructurar la información.

### Sugerencia de Precios con IA

El sistema utiliza varias fuentes para sugerir precios:

1. **Base de datos histórica**: Precios de cotizaciones anteriores
2. **Análisis de similitud**: Identificación de actividades similares
3. **Modelo de IA**: Sugerencias basadas en contexto cuando no hay datos históricos

### Base de Datos

El sistema utiliza SQLite para almacenar:

- Historial de precios por actividad
- Relaciones entre actividades similares
- Cotizaciones completas
- Categorías y metadatos

## Contribución

Si deseas contribuir al proyecto, por favor:

1. Haz un fork del repositorio
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## Licencia

Este proyecto está licenciado bajo [MIT License](LICENSE).
