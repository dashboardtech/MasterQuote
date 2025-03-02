"""
Módulo para componentes modernos de UI en Streamlit.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import base64
from pathlib import Path

class ModernUIComponents:
    """Componentes modernos de UI para Streamlit."""
    
    @staticmethod
    def custom_header(title: str, subtitle: str, icon: str = None):
        """
        Crea un encabezado personalizado con estilo moderno.
        
        Args:
            title: Título principal
            subtitle: Subtítulo o descripción
            icon: Emoji o ícono (opcional)
        """
        if icon:
            st.markdown(f"# {icon} {title}")
        else:
            st.markdown(f"# {title}")
        
        st.markdown(
            f"<p style='font-size: 1.2em; opacity: 0.7; margin-top: -10px;'>{subtitle}</p>",
            unsafe_allow_html=True
        )
        st.markdown("---")
    
    @staticmethod
    def info_card(title: str, value: Any, delta: Optional[float] = None, prefix: str = "", suffix: str = ""):
        """
        Crea una tarjeta de información con valor y cambio.
        
        Args:
            title: Título de la métrica
            value: Valor principal
            delta: Cambio porcentual (opcional)
            prefix: Prefijo para el valor (e.g., "$")
            suffix: Sufijo para el valor (e.g., "%")
        """
        st.metric(
            label=title,
            value=f"{prefix}{value}{suffix}",
            delta=f"{delta:+.1f}%" if delta is not None else None
        )
    
    @staticmethod
    def file_upload_area(accept_multiple_files: bool = False,
                        file_types: List[str] = None,
                        key: Optional[str] = None) -> Any:
        """
        Crea un área mejorada para carga de archivos.
        
        Args:
            accept_multiple_files: Si permite múltiples archivos
            file_types: Lista de extensiones permitidas
            key: Clave única para el componente
        
        Returns:
            Archivo(s) cargado(s)
        """
        # Preparar tipos de archivo
        if file_types:
            file_types = [f".{ft.lower().strip('.')}" for ft in file_types]
        
        # Crear área de carga
        st.markdown(
            """
            <style>
            .uploadedFile {
                border: 2px dashed #4CAF50;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                background-color: #f8f9fa;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "Arrastra tus archivos aquí o haz clic para seleccionar",
            accept_multiple_files=accept_multiple_files,
            type=file_types,
            key=key
        )
        
        return uploaded_file
    
    @staticmethod
    def create_activity_table(df: pd.DataFrame,
                            editable: bool = False,
                            with_confidence: bool = False,
                            height: int = 400):
        """
        Crea una tabla de actividades con formato mejorado.
        
        Args:
            df: DataFrame con los datos
            editable: Si la tabla es editable
            with_confidence: Si muestra nivel de confianza
            height: Altura de la tabla en píxeles
        """
        # Aplicar formato condicional
        def color_confidence(val):
            if 'confianza' in df.columns:
                colors = {
                    'alta': '#c6efce',
                    'media': '#ffeb9c',
                    'baja': '#ffc7ce',
                    'media-baja': '#ffc7ce'
                }
                return f'background-color: {colors.get(val, "")}'
            return ''
        
        # Configurar columnas editables
        if editable:
            for col in df.columns:
                if col in ['costo_unitario', 'cantidad']:
                    df[col] = df[col].astype(float)
        
        # Mostrar tabla
        st.dataframe(
            df.style.applymap(color_confidence, subset=['confianza'] if with_confidence else None),
            height=height,
            use_container_width=True
        )
    
    @staticmethod
    def create_price_histogram(df: pd.DataFrame,
                             price_column: str = 'costo_unitario',
                             bins: int = 20):
        """
        Crea un histograma de precios interactivo.
        
        Args:
            df: DataFrame con los datos
            price_column: Nombre de la columna de precios
            bins: Número de bins para el histograma
        """
        fig = px.histogram(
            df,
            x=price_column,
            nbins=bins,
            title="Distribución de Precios",
            labels={price_column: "Precio"},
            color_discrete_sequence=['#4CAF50']
        )
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(t=50, l=0, r=0, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_price_trend_chart(df: pd.DataFrame,
                               date_column: str,
                               price_column: str,
                               item_column: str):
        """
        Crea un gráfico de tendencia de precios.
        
        Args:
            df: DataFrame con los datos
            date_column: Nombre de la columna de fechas
            price_column: Nombre de la columna de precios
            item_column: Nombre de la columna de items
        """
        fig = px.line(
            df,
            x=date_column,
            y=price_column,
            color=item_column,
            title="Tendencia de Precios",
            labels={
                date_column: "Fecha",
                price_column: "Precio",
                item_column: "Actividad"
            }
        )
        
        fig.update_layout(
            plot_bgcolor='white',
            margin=dict(t=50, l=0, r=0, b=0),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_dashboard_page(construction_config=None):
    """
    Crea la página de dashboard.
    
    Args:
        construction_config: Configuración específica para el sector de construcción
    """
    ModernUIComponents.custom_header(
        "Dashboard de Costos",
        "Análisis y visualización de costos de construcción",
        "📊"
    )
    
    st.markdown("""
    Este dashboard muestra información sobre costos y tendencias de los proyectos de construcción.
    """)
    
    # Selector de categoría
    if construction_config:
        st.subheader("Filtros")
        selected_category = st.selectbox(
            "Categoría de Construcción",
            ["Todas"] + construction_config["categories"]
        )
    
    # Gráficos de ejemplo
    st.subheader("Tendencias de Costos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Costos por Categoría")
        
        # Datos de ejemplo
        if construction_config:
            chart_data = pd.DataFrame({
                'categoria': construction_config["categories"][:5],
                'costo_promedio': [random.randint(1000, 10000) for _ in range(5)]
            })
            
            st.bar_chart(chart_data.set_index('categoria'))
    
    with col2:
        st.markdown("### Evolución de Precios")
        
        # Datos de ejemplo
        months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun"]
        chart_data = pd.DataFrame({
            'mes': months,
            'indice_costos': [random.randint(95, 110) for _ in range(6)]
        })
        
        st.line_chart(chart_data.set_index('mes'))
    
    # Tabla de costos recientes
    st.subheader("Últimos Proyectos")
    
    # Datos de ejemplo
    project_data = pd.DataFrame({
        'proyecto': [f"Proyecto {i}" for i in range(1, 6)],
        'categoria': [random.choice(construction_config["categories"] if construction_config else ["Vivienda", "Comercial", "Industrial"]) for _ in range(5)],
        'total': [random.randint(50000, 500000) for _ in range(5)],
        'fecha': [(datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%d/%m/%Y') for _ in range(5)]
    })
    
    st.dataframe(project_data)

def create_settings_page(api_key_manager, construction_config=None):
    """
    Crea la página de configuración.
    
    Args:
        api_key_manager: Gestor de claves API
        construction_config: Configuración específica para el sector de construcción
    """
    ModernUIComponents.custom_header(
        "Configuración",
        "Personaliza la configuración del sistema",
        "⚙️"
    )
    
    tabs = st.tabs(["API Keys", "Construcción", "Sistema", "Avanzado"])
    
    with tabs[0]:
        st.subheader("Configuración de APIs")
        
        # OpenAI API Key
        openai_api_key = api_key_manager.get_openai_api_key()
        new_openai_key = st.text_input(
            "API Key de OpenAI",
            value=openai_api_key if openai_api_key else "",
            placeholder="Ingrese su API Key",
            type="password"
        )
        
        if new_openai_key and new_openai_key != openai_api_key:
            if st.button("Guardar API Key de OpenAI"):
                api_key_manager.set_api_key("openai", new_openai_key)
                st.success("API Key de OpenAI guardada correctamente")
        
        # Dockling API Key
        dockling_api_key = api_key_manager.get_dockling_api_key()
        new_dockling_key = st.text_input(
            "API Key de Dockling",
            value=dockling_api_key if dockling_api_key else "",
            placeholder="Ingrese su API Key",
            type="password"
        )
        
        if new_dockling_key and new_dockling_key != dockling_api_key:
            if st.button("Guardar API Key de Dockling"):
                api_key_manager.set_api_key("dockling", new_dockling_key)
                st.success("API Key de Dockling guardada correctamente")
    
    with tabs[1]:
        if construction_config:
            st.subheader("Configuración de Construcción")
            
            # Mostrar categorías
            st.markdown("### Categorías de Construcción")
            
            for category in construction_config["categories"]:
                st.write(f"- {category}")
            
            # Mostrar unidades
            st.markdown("### Unidades de Medida")
            
            col1, col2, col3 = st.columns(3)
            categories_per_column = len(construction_config["units"]) // 3 + 1
            
            for i, unit in enumerate(construction_config["units"]):
                with [col1, col2, col3][i // categories_per_column]:
                    st.write(f"- {unit}")
            
            # Ejemplo de edición de configuración
            st.markdown("### Editar Configuración")
            st.warning("Esta funcionalidad estará disponible en la próxima versión.")
    
    with tabs[2]:
        st.subheader("Configuración del Sistema")
        
        # Opciones de caché
        st.markdown("### Caché")
        use_cache = st.checkbox("Habilitar caché", value=True)
        cache_days = st.slider("Días para expiración de caché", 1, 90, 30)
        
        # Extractores
        st.markdown("### Extractores")
        use_validation = st.checkbox("Validación cruzada por defecto", value=True)
        min_confidence = st.slider("Confianza mínima predeterminada", 0.0, 1.0, 0.6, 0.1)
        
        if st.button("Guardar configuración del sistema"):
            st.success("Configuración guardada. (Función simulada)")
    
    with tabs[3]:
        st.subheader("Configuración Avanzada")
        
        # Directorio de datos
        data_dir = st.text_input("Directorio de datos", value="./data")
        
        # Opciones de procesamiento
        use_parallel = st.checkbox("Utilizar procesamiento paralelo", value=True)
        num_processes = st.slider("Número de procesos", 1, 8, 4)
        
        st.warning("Cambiar la configuración avanzada puede afectar el rendimiento del sistema.")
        
        if st.button("Aplicar configuración avanzada"):
            st.success("Configuración avanzada aplicada. (Función simulada)")


def setup_new_quotation_page(extractor, db, llm, extractor_manager=None, construction_config=None, default_min_confidence=0.5, validation_enabled_by_default=True):
    """
    Configura la página de nueva cotización.
    
    Args:
        extractor: Extractor universal de precios
        db: Conexión a la base de datos
        llm: Instancia de CotizacionLLM
        extractor_manager: Gestor de extractores para validación cruzada (opcional)
        construction_config: Configuración específica para el sector de construcción
        default_min_confidence: Valor predeterminado para la confianza mínima
        validation_enabled_by_default: Si la validación cruzada está habilitada por defecto
    """
    ModernUIComponents.custom_header(
        "Nueva Cotización para Construcción",
        "Crea una nueva cotización con asistencia de IA especializada en construcción",
        "🏗️"
    )
    
    # Panel de información
    with st.expander("ℹ️ Información sobre esta herramienta"):
        st.markdown("""
        **Cómo usar la herramienta de cotización para proyectos de construcción:**
        
        1. Sube un archivo con tu lista de actividades de construcción (Excel, CSV, PDF o Word)
        2. El sistema analizará las actividades y sugerirá precios basados en:
           - Historial de precios anterior
           - Análisis de similitud con proyectos de construcción
           - Inteligencia artificial para casos sin precedentes
        3. Revisa y ajusta las sugerencias según sea necesario
        4. Exporta la cotización en formato profesional para proyectos de construcción
        """)
    
    # Verificar si hay una plantilla seleccionada previamente
    if 'new_project_type' in st.session_state and 'new_project_sections' in st.session_state:
        st.success(f"Usando plantilla de proyecto: {st.session_state.new_project_type}")
        project_type = st.session_state.new_project_type
        sections = st.session_state.new_project_sections
        
        # Mostrar secciones del proyecto
        with st.expander("Secciones del Proyecto", expanded=True):
            st.markdown("### Secciones incluidas en esta plantilla:")
            for section in sections:
                st.markdown(f"- {section}")
            
            if st.button("Limpiar plantilla"):
                del st.session_state.new_project_type
                del st.session_state.new_project_sections
                st.experimental_rerun()
                
    # Información del proyecto
    st.subheader("Información del Proyecto")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        project_name = st.text_input("Nombre del Proyecto", "")
    
    with col2:
        if construction_config:
            project_category = st.selectbox("Categoría", construction_config["categories"])
        else:
            project_category = st.text_input("Categoría", "")
    
    with col3:
        project_date = st.date_input("Fecha", datetime.now())
    
    # Opciones de extracción
    st.subheader("Configuración de Extracción")
    col1, col2 = st.columns(2)
    with col1:
        use_validation = st.checkbox("Usar validación cruzada", value=validation_enabled_by_default, 
                                    help="Utiliza múltiples extractores para validar los resultados")
    
    with col2:
        min_confidence = st.slider("Confianza mínima", min_value=0.0, max_value=1.0, value=default_min_confidence, step=0.1,
                                 help="Nivel mínimo de confianza requerido para aceptar resultados")
    
    # Selección de unidades de medida
    if construction_config:
        with st.expander("Unidades de Medida Comunes en Construcción"):
            st.markdown("### Unidades Disponibles")
            units_columns = st.columns(4)
            for i, unit in enumerate(construction_config["units"]):
                with units_columns[i % 4]:
                    st.markdown(f"- {unit}")
    
    # Área de carga de archivos
    st.subheader("Carga de Archivo")
    uploaded_file = ModernUIComponents.file_upload_area(
        accept_multiple_files=False,
        file_types=["xlsx", "xls", "csv", "pdf", "docx"],
        key="quotation_uploader"
    )
    
    if uploaded_file:
        st.success(f"Archivo subido: {uploaded_file.name}")
        
        # Procesar archivo
        with st.spinner("Procesando archivo..."):
            try:
                # Guardar el archivo temporalmente
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_file_path = tmp.name
                
                if use_validation and extractor_manager:
                    # Usar validación cruzada con múltiples extractores
                    df, metadata = extractor_manager.extract_with_validation(temp_file_path)
                    
                    # Mostrar información de confianza
                    confidence = metadata.get('confidence', 0.0)
                    confidence_color = "green" if confidence >= 0.7 else "orange" if confidence >= 0.4 else "red"
                    
                    st.write("### Información de Extracción")
                    st.markdown(f"""
                    - **Confianza**: <span style='color:{confidence_color}'>{confidence:.2f}</span>
                    - **Extractor principal**: {metadata.get('primary_extractor', 'N/A')}
                    - **Extractores de apoyo**: {', '.join(metadata.get('supporting_extractors', ['N/A']))}
                    """, unsafe_allow_html=True)
                    
                    # Si la confianza es muy baja, mostrar advertencia
                    if confidence < min_confidence:
                        st.warning(f"La confianza en los resultados ({confidence:.2f}) está por debajo del mínimo configurado ({min_confidence}). Considere revisar manualmente los datos o probar con otro formato de archivo.")
                else:
                    # Usar el extractor universal tradicional
                    df = extractor.extract_from_file(temp_file_path, use_validation=False)
                
                # Limpiar archivo temporal
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                
                if df is not None and not df.empty:
                    # Preparar DataFrame para el sector de construcción
                    if construction_config and 'categoria' not in df.columns:
                        df['categoria'] = project_category
                    
                    if 'unidad' not in df.columns:
                        df['unidad'] = 'global'  # Valor predeterminado
                    
                    # Procesar con LLM
                    df = llm.procesar_dataframe(df)
                    
                    # Añadir información del proyecto
                    st.write("### Información del Proyecto")
                    st.markdown(f"""
                    - **Proyecto**: {project_name}
                    - **Categoría**: {project_category}
                    - **Fecha**: {project_date.strftime('%d/%m/%Y')}
                    """)
                    
                    # Mostrar resultados
                    st.write("### Sugerencias de Precios")
                    
                    # Calcular totales
                    if 'precio' in df.columns and 'cantidad' in df.columns:
                        if 'total' not in df.columns:
                            df['total'] = df['precio'] * df['cantidad']
                    
                    ModernUIComponents.create_activity_table(
                        df,
                        editable=True,
                        with_confidence=True
                    )
                    
                    # Mostrar totales
                    if 'total' in df.columns:
                        total_presupuesto = df['total'].sum()
                        st.metric("Total Presupuesto", f"${total_presupuesto:,.2f}")
                    
                    # Botones de acción
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Exportar a Excel"):
                            # Lógica para exportar a Excel
                            st.success("Exportación a Excel completada. (Función simulada)")
                    
                    with col2:
                        if st.button("Guardar Cotización"):
                            # Lógica para guardar en la base de datos
                            st.success("Cotización guardada. (Función simulada)")
                    
                    with col3:
                        if st.button("Generar PDF"):
                            # Lógica para generar PDF
                            st.success("PDF generado. (Función simulada)")
                    
                else:
                    st.error("No se pudieron extraer datos del archivo. Por favor intente con otro formato o archivo.")
            
            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    else:
        # Si no hay archivo cargado, mostrar plantilla vacía
        if 'new_project_type' in st.session_state and 'new_project_sections' in st.session_state:
            st.subheader("Plantilla de Proyecto")
            
            # Crear DataFrame con las secciones de la plantilla
            sections = st.session_state.new_project_sections
            template_df = pd.DataFrame({
                'actividad': sections,
                'descripcion': [f"Descripción para {s}" for s in sections],
                'unidad': ['global'] * len(sections),
                'cantidad': [1] * len(sections),
                'precio': [0] * len(sections),
                'total': [0] * len(sections)
            })
            
            ModernUIComponents.create_activity_table(
                template_df,
                editable=True,
                with_confidence=False
            )
            
            st.info("Complete la plantilla manual o suba un archivo para análisis automático")

def create_budget_analysis_page():
    """
    Crea la página de análisis de presupuestos.
    
    Esta página permite cargar archivos de presupuesto para su análisis y validación,
    utilizando las funcionalidades de batch_analyze_budgets.py y analyze_budget_excel.py.
    """
    ModernUIComponents.custom_header(
        "Análisis de Presupuestos",
        "Analiza y valida presupuestos de construcción con asistencia de IA",
        "📈"
    )
    
    # Panel de información
    with st.expander("ℹ️ Información sobre esta herramienta"):
        st.markdown("""
        **Cómo usar la herramienta de análisis de presupuestos:**
        
        1. Sube uno o varios archivos de presupuesto (Excel)
        2. El sistema analizará los presupuestos y validará su consistencia
        3. Se generarán estadísticas detalladas y visualizaciones
        4. Podrás exportar los resultados en diferentes formatos
        
        Esta herramienta utiliza algoritmos avanzados para:
        - Detectar columnas de precios unitarios
        - Validar la consistencia matemática (cantidad × precio = total)
        - Identificar patrones y anomalías en los datos
        - Generar informes detallados con estadísticas
        """)
    
    # Opciones de análisis
    st.subheader("Configuración de Análisis")
    col1, col2 = st.columns(2)
    
    with col1:
        validation_enabled = st.checkbox("Habilitar validación", value=True, 
                                        help="Valida la consistencia matemática de los presupuestos")
    
    with col2:
        parallel_processing = st.checkbox("Procesamiento en paralelo", value=True,
                                        help="Procesa múltiples archivos simultáneamente")
    
    # Área de carga de archivos
    st.subheader("Carga de Archivos")
    uploaded_files = ModernUIComponents.file_upload_area(
        accept_multiple_files=True,
        file_types=["xlsx", "xls"],
        key="budget_uploader"
    )
    
    if uploaded_files:
        st.success(f"Archivos subidos: {len(uploaded_files)}")
        
        # Mostrar archivos subidos
        for file in uploaded_files:
            st.write(f"- {file.name}")
        
        # Botón para iniciar el análisis
        if st.button("Iniciar Análisis"):
            with st.spinner("Analizando presupuestos..."):
                try:
                    import tempfile
                    import os
                    import json
                    from datetime import datetime
                    import analyze_budget_excel
                    import batch_analyze_budgets
                    from json_utils import CustomJSONEncoder
                    
                    # Crear directorio temporal para los archivos
                    temp_dir = tempfile.mkdtemp()
                    file_paths = []
                    
                    # Guardar archivos temporalmente
                    for file in uploaded_files:
                        temp_file_path = os.path.join(temp_dir, file.name)
                        with open(temp_file_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(temp_file_path)
                    
                    # Crear directorio para resultados
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    output_dir = f"web_analysis_{timestamp}"
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Ejecutar análisis por lotes
                    stats = batch_analyze_budgets.batch_analyze_budgets(
                        file_paths,
                        output_dir,
                        parallel=parallel_processing,
                        validation_enabled=validation_enabled
                    )
                    
                    # Mostrar resultados
                    st.subheader("Resultados del Análisis")
                    
                    # Estadísticas generales
                    st.write("### Estadísticas Generales")
                    
                    # Crear métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Archivos Procesados", len(stats))
                    
                    with col2:
                        total_rows = sum(stat.get("total_rows", 0) for stat in stats)
                        st.metric("Filas Totales", total_rows)
                    
                    with col3:
                        valid_rows = sum(stat.get("valid_rows", 0) for stat in stats)
                        validation_rate = (valid_rows / total_rows * 100) if total_rows > 0 else 0
                        st.metric("Tasa de Validación", f"{validation_rate:.2f}%")
                    
                    # Mostrar visualizaciones
                    st.write("### Visualizaciones")
                    if os.path.exists(os.path.join(output_dir, "consolidated_visualizaciones.png")):
                        st.image(os.path.join(output_dir, "consolidated_visualizaciones.png"))
                    
                    # Mostrar detalles por archivo
                    st.write("### Detalles por Archivo")
                    for i, stat in enumerate(stats):
                        with st.expander(f"Archivo {i+1}: {os.path.basename(stat.get('file_path', 'Desconocido'))}"):
                            st.write(f"- Filas totales: {stat.get('total_rows', 0)}")
                            st.write(f"- Filas válidas: {stat.get('valid_rows', 0)}")
                            validation_rate = (stat.get('valid_rows', 0) / stat.get('total_rows', 1) * 100) if stat.get('total_rows', 0) > 0 else 0
                            st.write(f"- Tasa de validación: {validation_rate:.2f}%")
                            
                            # Mostrar más detalles si están disponibles
                            if "column_scores" in stat:
                                st.write("#### Puntuaciones de Columnas")
                                for col, score in stat["column_scores"].items():
                                    st.write(f"- {col}: {score:.2f}")
                    
                    # Botones para descargar resultados
                    st.subheader("Descargar Resultados")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if os.path.exists(os.path.join(output_dir, "consolidated_data.xlsx")):
                            with open(os.path.join(output_dir, "consolidated_data.xlsx"), "rb") as f:
                                btn = st.download_button(
                                    label="Descargar Datos Consolidados (Excel)",
                                    data=f,
                                    file_name="datos_consolidados.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    
                    with col2:
                        if os.path.exists(os.path.join(output_dir, "price_analysis.xlsx")):
                            with open(os.path.join(output_dir, "price_analysis.xlsx"), "rb") as f:
                                btn = st.download_button(
                                    label="Descargar Análisis de Precios (Excel)",
                                    data=f,
                                    file_name="analisis_precios.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    
                except Exception as e:
                    st.error(f"Error al procesar los archivos: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
