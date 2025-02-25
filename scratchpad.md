# Optimización del Sistema de Cotizaciones

## Objetivo
Mejorar la arquitectura y rendimiento del sistema de cotizaciones, integrando procesamiento asíncrono y mejorando la seguridad.

## Plan de Implementación

### Fase 1: Arquitectura Base
[X] 1. Implementar DocLing Adapter
    - ✅ Crear adaptador básico
    - ✅ Integrar extracción de tablas
    - ✅ Agregar tests unitarios
    - ✅ Integrar con Streamlit UI

### Fase 2: Optimización de Arquitectura
[X] 1. Implementar Extractores Especializados
    - ✅ BaseExtractor abstracto
    - ✅ ExcelExtractor
    - ✅ CSVExtractor
    - ✅ PDFExtractor
    - ✅ AIAssistedExtractor
    - ✅ DocklingExtractor

[ ] 2. Procesamiento Asíncrono
    - [ ] AsyncProcessor para tareas en segundo plano
    - [ ] BatchProcessor para múltiples archivos
    - [ ] UI de progreso en Streamlit

[ ] 3. Seguridad
    - [ ] SecretManager para API keys
    - [ ] APIKeyManager para OpenAI
    - [ ] UI de gestión de secretos

## Estado Actual

1. **Extractores Implementados**:
   - ✅ BaseExtractor con normalización de precios
   - ✅ ExcelExtractor con soporte multi-hoja
   - ✅ CSVExtractor con detección de formato
   - ✅ PDFExtractor con OCR fallback
   - ✅ AIAssistedExtractor para casos complejos
   - ✅ DocklingExtractor para procesamiento avanzado

2. **Próximos Pasos**:
   - Implementar procesamiento asíncrono
   - Agregar gestión segura de API keys
   - Actualizar UI de Streamlit

## Lecciones Aprendidas

1. **Gestión de Dependencias**:
   - Usar versiones flexibles (>= en lugar de ==) para evitar conflictos de dependencias
   - Verificar compatibilidad entre paquetes antes de fijar versiones
   - Para openpyxl, usar versión >=3.1.5 cuando se usa con docling

2. **Streamlit Cache**:
   - Para argumentos no hasheables en funciones cacheadas, usar prefijo '_' (e.g., _secret_manager)
   - Esto es especialmente importante para objetos complejos como gestores de API y bases de datos
   - El caché mejora significativamente el rendimiento al evitar reinicializaciones

3. **Extracción de Datos**:
   - Usar múltiples estrategias de extracción mejora la robustez
   - La normalización de precios debe manejar diversos formatos
   - El OCR es útil como fallback para PDFs complejos

2. **Arquitectura**:
   - El patrón Factory simplifica la selección de extractores
   - La herencia común reduce la duplicación de código
   - Los extractores especializados mejoran la precisión

3. **APIs Externas**:
   - Dockling proporciona capacidades avanzadas de extracción
   - La gestión de API keys debe ser segura y flexible
   - El manejo de errores de API es crucial

