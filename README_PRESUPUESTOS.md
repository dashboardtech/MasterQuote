# Sistema Optimizado de Validación de Presupuestos

Este sistema proporciona herramientas avanzadas para la extracción, validación y análisis de presupuestos de construcción, integrando todas las mejoras que hemos implementado, incluyendo Dockling para el procesamiento avanzado de documentos y el sistema de validación optimizado.

## Herramientas incluidas

### 1. Analizador de Excel de Presupuestos (`analyze_budget_excel.py`)

Esta herramienta analiza un archivo Excel de presupuesto, extrae los datos utilizando extractores optimizados, valida la consistencia de los datos y genera un informe detallado con visualizaciones.

**Uso:**
```bash
python analyze_budget_excel.py ruta/al/presupuesto.xlsx --output directorio/salida
```

**Opciones:**
- `--output`: Directorio para guardar los resultados (opcional)
- `--no-dockling`: Desactivar el uso de Dockling para la extracción
- `--verbose`: Mostrar información detallada durante el proceso

### 2. Importador de Presupuestos a BD (`import_budget_to_db.py`)

Esta herramienta procesa un archivo de presupuesto, valida sus datos y los importa a la base de datos para su uso posterior.

**Uso:**
```bash
python import_budget_to_db.py ruta/al/presupuesto.xlsx --db ruta/a/bd.db --project "Nombre del Proyecto" --client "Nombre del Cliente"
```

**Opciones:**
- `--db`: Ruta a la base de datos (opcional, por defecto usa "price_history.db")
- `--project`: Nombre del proyecto (opcional, por defecto usa el nombre del archivo)
- `--client`: Nombre del cliente (opcional)
- `--no-dockling`: Desactivar Dockling

### 3. Procesador por Lotes (`batch_analyze_budgets.py`)

Esta herramienta permite procesar múltiples archivos de presupuesto en lote, generando estadísticas consolidadas y análisis comparativos.

**Uso:**
```bash
python batch_analyze_budgets.py "directorio/presupuestos/*.xlsx" --output directorio/salida --import --db ruta/a/bd.db
```

**Opciones:**
- `--output`: Directorio para guardar los resultados (opcional)
- `--db`: Ruta a la base de datos (opcional)
- `--import`: Importar los presupuestos validados a la base de datos
- `--no-dockling`: Desactivar Dockling
- `--sequential`: Procesar archivos secuencialmente (sin paralelismo)
- `--workers`: Número máximo de procesos en paralelo (por defecto: 4)

## Requisitos

Para utilizar estas herramientas se necesita:

1. Todas las dependencias del proyecto MasterQuote instaladas
2. Configuración de API keys:
   - `DOCKLING_API_KEY`: API key para Dockling (opcional, mejora la extracción)
   - `OPENAI_API_KEY`: API key para OpenAI (opcional, mejora la clasificación)

## Ejemplos de uso

### Analizar un único presupuesto
```bash
python analyze_budget_excel.py datos/presupuesto_obra.xlsx
```

### Importar un presupuesto a la base de datos
```bash
python import_budget_to_db.py datos/presupuesto_obra.xlsx --project "Remodelación Casa" --client "Juan Pérez"
```

### Procesar múltiples presupuestos y consolidar información
```bash
python batch_analyze_budgets.py "datos/presupuestos/*.xlsx" --output analisis_consolidado --import
```

## Ventajas del sistema optimizado

1. **Mayor precisión en la extracción**: Utilizando la validación cruzada con múltiples extractores y Dockling para documentos complejos.
2. **Validación robusta de presupuestos**: Sistema avanzado para validar la consistencia matemática de los datos y detectar errores.
3. **Análisis visual detallado**: Generación de gráficos y estadísticas para comprender mejor los datos.
4. **Procesamiento en lote**: Capacidad para procesar múltiples archivos en paralelo para mayor eficiencia.
5. **Integración con base de datos**: Almacenamiento estructurado de los datos validados para su uso en cotizaciones futuras.

## Estructura de la base de datos

La base de datos almacena:
- **Actividades**: Nombres normalizados de las actividades de construcción
- **Precios históricos**: Registro de precios con fechas de actualización
- **Cotizaciones**: Presupuestos completos con todos sus ítems
- **Metadatos de validación**: Información sobre la calidad de los datos y su origen

## Flujo de trabajo recomendado

1. **Analizar** los presupuestos individuales para entender su estructura y calidad
2. **Procesar en lote** múltiples presupuestos para obtener estadísticas consolidadas
3. **Importar** a la base de datos los presupuestos validados para construir un repositorio de precios históricos
4. **Consultar** la base de datos para obtener sugerencias de precios en nuevas cotizaciones

Este flujo permite ir mejorando progresivamente la calidad de las sugerencias de precios a medida que se incorporan más datos validados al sistema.
