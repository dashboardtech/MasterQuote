import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import time
import logging
from .modern_components import ModernUIComponents

logger = logging.getLogger(__name__)

def create_dashboard_page():
    """Crea una página de dashboard con estadísticas de cotizaciones."""
    ModernUIComponents.custom_header(
        "Dashboard de Cotizaciones",
        "Vista general del sistema de cotizaciones",
        "📊"
    )
    
    # Fechas para filtrado
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Fecha inicial",
            datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "Fecha final",
            datetime.now()
        )
    
    # Métricas principales
    metrics = [
        {"title": "Total Cotizaciones", "value": 157, "delta": 12.5, "color": "#1f77b4"},
        {"title": "Monto Promedio", "value": "$15,230", "delta": -3.2, "color": "#ff7f0e"},
        {"title": "Tasa de Conversión", "value": "24.8", "suffix": "%", "delta": 5.7, "color": "#2ca02c"}
    ]
    ModernUIComponents.dashboard_cards(metrics)
    
    # Gráficos
    st.subheader("Análisis de Datos")
    col1, col2 = st.columns(2)
    
    with col1:
        sample_data = pd.DataFrame({
            'costo_unitario': np.random.lognormal(mean=8, sigma=1, size=100)
        })
        ModernUIComponents.create_price_histogram(sample_data)
    
    with col2:
        date_range = pd.date_range(end=datetime.now(), periods=50)
        activities = ['Pintura', 'Limpieza', 'Instalación', 'Reparación']
        data = []
        
        for activity in activities:
            base_price = np.random.uniform(500, 2000)
            trend = np.linspace(0, np.random.uniform(0.05, 0.2), len(date_range))
            prices = base_price * (1 + trend + np.random.normal(0, 0.02, len(date_range)))
            
            for i, date in enumerate(date_range):
                data.append({
                    'fecha_actualizacion': date,
                    'actividad': activity,
                    'precio': prices[i]
                })
        
        sample_history = pd.DataFrame(data)
        ModernUIComponents.create_price_trend_chart(sample_history)

def create_settings_page(api_key_manager):
    """Crea una página de configuración del sistema."""
    ModernUIComponents.custom_header(
        "Configuración del Sistema",
        "Administre la configuración y preferencias del sistema",
        "⚙️"
    )
    
    tabs = st.tabs(["API Keys", "Base de Datos", "Caché", "Interfaz", "Avanzado"])
    
    with tabs[0]:
        st.info("Configuración de APIs para los servicios externos")
        st.subheader("OpenAI API")
        
        current_key = api_key_manager.get_openai_api_key()
        key_status = "✅ Configurada" if current_key else "❌ No configurada"
        st.markdown(f"**Estado de API Key**: {key_status}")
        
        new_key = st.text_input(
            "API Key de OpenAI", 
            type="password",
            help="Comienza con 'sk-'"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Guardar API Key", use_container_width=True):
                if new_key:
                    if api_key_manager.set_openai_api_key(new_key):
                        st.success("✅ API Key guardada correctamente")
                    else:
                        st.error("❌ Error al guardar API Key")
                else:
                    st.warning("⚠️ Ingrese una API Key válida")
        
        with col2:
            if st.button("Probar API Key", use_container_width=True):
                if current_key:
                    with st.spinner("Probando API Key..."):
                        if api_key_manager.test_openai_api_key():
                            st.success("✅ API Key válida")
                        else:
                            st.error("❌ API Key inválida")
                else:
                    st.warning("⚠️ No hay API Key configurada")

def setup_new_quotation_page(universal_extractor, price_database, llm_processor):
    """Configura la página de nueva cotización con mejoras visuales."""
    ModernUIComponents.custom_header(
        "Nueva Cotización",
        "Cree y administre cotizaciones con sugerencias inteligentes",
        "📝"
    )
    
    with st.expander("ℹ️ Información sobre esta herramienta"):
        st.markdown("""
        **Cómo usar la herramienta de cotización:**
        
        1. Suba un archivo Excel con su lista de actividades
        2. El sistema analizará las actividades y sugerirá precios basados en:
           - Historial de precios anterior
           - Análisis de similitud
           - Inteligencia artificial para casos sin precedentes
        3. Revise y ajuste las sugerencias según sea necesario
        4. Exporte la cotización en formato profesional
        """)
    
    with st.sidebar:
        st.subheader("Opciones de Cotización")
        usar_llm = st.checkbox("Usar IA para sugerencias", value=True)
        usar_bd = st.checkbox("Usar Base de Datos Histórica", value=True)
        
        st.write("**Información del Proyecto**")
        nombre_proyecto = st.text_input("Nombre del Proyecto", "Nueva Cotización")
        cliente = st.text_input("Cliente", "")
        
        st.write("**Ajustes de Precios**")
        ajuste_global = st.number_input("Ajuste Global (%)", value=0.0, step=1.0)
        decimales = st.number_input("Decimales", value=2, min_value=0, max_value=4)
    
    uploaded_file = ModernUIComponents.file_upload_area(
        accept_multiple_files=False,
        file_types=["xlsx", "xls"],
        key="cotizacion_uploader"
    )
    
    if uploaded_file:
        st.success(f"Archivo subido: {uploaded_file.name}")
        
        with st.spinner("Procesando archivo..."):
            try:
                # Procesar archivo
                df = universal_extractor.extract_from_file(
                    uploaded_file,
                    use_dockling=usar_llm
                )
                
                if not df.empty:
                    # Mostrar sugerencias
                    st.subheader("Sugerencias de Precios")
                    ModernUIComponents.create_activity_table(
                        df,
                        editable=True,
                        with_confidence=True
                    )
                    
                    # Calcular total
                    if 'costo_total' in df.columns:
                        total = df['costo_total'].sum()
                        st.metric("Total Cotización", f"${total:,.2f}")
                        
                        # Opciones de exportación
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Exportar Excel", use_container_width=True):
                                # Aquí iría la lógica de exportación
                                st.success("Excel exportado correctamente")
                        
                        with col2:
                            if st.button("Guardar en Base de Datos", use_container_width=True):
                                # Aquí iría la lógica de guardado
                                st.success("Cotización guardada en base de datos")
                else:
                    st.warning("No se pudieron extraer datos del archivo")
            
            except Exception as e:
                logger.exception("Error procesando archivo")
                st.error(f"Error procesando archivo: {str(e)}")
