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


def create_dashboard_page():
    """Crea la página de dashboard."""
    ModernUIComponents.custom_header(
        "Dashboard",
        "Análisis y métricas de cotizaciones",
        "📊"
    )
    
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ModernUIComponents.info_card(
            "Cotizaciones este mes",
            15,
            delta=8.5
        )
    
    with col2:
        ModernUIComponents.info_card(
            "Monto promedio",
            25000,
            delta=12.3,
            prefix="$"
        )
    
    with col3:
        ModernUIComponents.info_card(
            "Tasa de conversión",
            65,
            delta=-2.1,
            suffix="%"
        )
    
    # Datos de ejemplo para gráficos
    df_ejemplo = pd.DataFrame({
        'fecha': pd.date_range(end=datetime.now(), periods=30),
        'precio': [1000 + i * 50 + (i % 3) * 200 for i in range(30)],
        'actividad': ['Pintura'] * 10 + ['Electricidad'] * 10 + ['Plomería'] * 10
    })
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        ModernUIComponents.create_price_histogram(
            df_ejemplo,
            price_column='precio'
        )
    
    with col2:
        ModernUIComponents.create_price_trend_chart(
            df_ejemplo,
            date_column='fecha',
            price_column='precio',
            item_column='actividad'
        )


def create_settings_page(api_key_manager):
    """
    Crea la página de configuración.
    
    Args:
        api_key_manager: Instancia de APIKeyManager
    """
    ModernUIComponents.custom_header(
        "Configuración",
        "Gestiona las configuraciones del sistema",
        "⚙️"
    )
    
    # API Keys
    st.write("### 🔑 API Keys")
    
    # OpenAI API Key
    openai_key = api_key_manager.get_openai_api_key()
    openai_key_input = st.text_input(
        "OpenAI API Key",
        value=openai_key if openai_key else "",
        type="password"
    )
    
    if st.button("Guardar OpenAI API Key"):
        if openai_key_input:
            api_key_manager.set_api_key("openai", openai_key_input)
            st.success("API Key de OpenAI guardada correctamente")
        else:
            st.error("Por favor ingresa una API Key válida")
    
    # Dockling API Key
    st.write("### 📄 Dockling API Key")
    dockling_key = api_key_manager.get_dockling_api_key()
    dockling_key_input = st.text_input(
        "Dockling API Key",
        value=dockling_key if dockling_key else "",
        type="password"
    )
    
    if st.button("Guardar Dockling API Key"):
        if dockling_key_input:
            api_key_manager.set_api_key("dockling", dockling_key_input)
            st.success("API Key de Dockling guardada correctamente")
        else:
            st.error("Por favor ingresa una API Key válida")
    
    # Configuración de exportación
    st.write("### 📤 Configuración de Exportación")
    
    export_format = st.selectbox(
        "Formato de exportación predeterminado",
        ["Excel (.xlsx)", "CSV (.csv)", "PDF (.pdf)"]
    )
    
    company_name = st.text_input(
        "Nombre de la empresa",
        value="MasterQuote"
    )
    
    company_logo = st.file_uploader(
        "Logo de la empresa",
        type=["png", "jpg", "jpeg"]
    )
    
    if st.button("Guardar configuración"):
        # TODO: Implementar guardado de configuración
        st.success("Configuración guardada correctamente")


def setup_new_quotation_page(extractor, db, llm):
    """
    Configura la página de nueva cotización.
    
    Args:
        extractor: Extractor universal de precios
        db: Conexión a la base de datos
        llm: Instancia de CotizacionLLM
    """
    ModernUIComponents.custom_header(
        "Nueva Cotización",
        "Crea una nueva cotización con asistencia de IA",
        "📝"
    )
    
    # Panel de información
    with st.expander("ℹ️ Información sobre esta herramienta"):
        st.markdown("""
        **Cómo usar la herramienta de cotización:**
        
        1. Sube un archivo con tu lista de actividades (Excel, CSV, PDF o Word)
        2. El sistema analizará las actividades y sugerirá precios basados en:
           - Historial de precios anterior
           - Análisis de similitud
           - Inteligencia artificial para casos sin precedentes
        3. Revisa y ajusta las sugerencias según sea necesario
        4. Exporta la cotización en formato profesional
        """)
    
    # Área de carga de archivos
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
                df = extractor.extract_from_file(uploaded_file)
                
                if df is not None and not df.empty:
                    # Procesar con LLM
                    df = llm.procesar_dataframe(df)
                    
                    # Mostrar resultados
                    st.write("### Sugerencias de Precios")
                    ModernUIComponents.create_activity_table(
                        df,
                        editable=True,
                        with_confidence=True
                    )
                    
                    # Botones de acción
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Exportar Excel"):
                            # TODO: Implementar exportación
                            st.success("Cotización exportada correctamente")
                    
                    with col2:
                        if st.button("Guardar en Base de Datos"):
                            # TODO: Implementar guardado
                            st.success("Cotización guardada en base de datos")
                else:
                    st.warning("No se pudieron extraer datos del archivo")
            
            except Exception as e:
                st.error(f"Error procesando archivo: {str(e)}")
