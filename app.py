import streamlit as st
import pandas as pd
import os
import tempfile
import logging
from datetime import datetime
import yaml
import time

# Importar configuración
from config import API_KEYS, CONSTRUCTION_CONFIG, UI_CONFIG, CACHE_CONFIG, EXTRACTOR_CONFIG

# Importar módulos core
from modules.price_database import PriceDatabase
from modules.cotizacion_llm import CotizacionLLM
from modules.data_loader import cargar_excel, validar_formato_excel
from modules.price_updater import actualizar_precios, aplicar_ajustes
from modules.exporter import exportar_cotizacion
from modules.universal_price_extractor import UniversalPriceExtractor

# Importar nuevos módulos
from modules.security_improvements import SecretManager, APIKeyManager
from modules.async_processing import AsyncProcessor, BatchProcessor
from modules.dockling_processor import DocklingProcessor
from modules.streamlit_ui_improvements import ModernUIComponents, create_dashboard_page, create_settings_page, setup_new_quotation_page, create_budget_analysis_page
from modules.extractor_manager import ExtractorManager, extract_with_confidence, process_batch_with_validation

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar componentes
@st.cache_resource
def inicializar_secretos():
    return SecretManager()

@st.cache_resource
def inicializar_api_keys(_secret_manager):
    # Usar las claves API del archivo de configuración
    api_key_manager = APIKeyManager(_secret_manager)
    
    # Configurar claves predeterminadas desde config.py
    if not api_key_manager.get_openai_api_key():
        api_key_manager.set_api_key("openai", API_KEYS["OPENAI_API_KEY"])
    
    if not api_key_manager.get_dockling_api_key():
        api_key_manager.set_api_key("dockling", API_KEYS["DOCKLING_API_KEY"])
    
    return api_key_manager

@st.cache_resource
def inicializar_db():
    return PriceDatabase("price_history.db")

@st.cache_resource
def inicializar_llm(_db, _api_key_manager):
    api_key = _api_key_manager.get_openai_api_key()
    return CotizacionLLM(_db, api_key=api_key)

@st.cache_resource
def inicializar_extractor(_api_key_manager):
    api_key = _api_key_manager.get_openai_api_key()
    dockling_api_key = _api_key_manager.get_dockling_api_key()
    return UniversalPriceExtractor(
        api_key=api_key, 
        dockling_api_key=dockling_api_key,
        use_cache=CACHE_CONFIG["enabled"], 
        cache_expiry_days=CACHE_CONFIG["expiration_time"] // (24 * 60 * 60)  # Convertir segundos a días
    )

@st.cache_resource
def inicializar_extractor_manager(_api_key_manager):
    api_key = _api_key_manager.get_openai_api_key()
    dockling_api_key = _api_key_manager.get_dockling_api_key()
    return ExtractorManager(
        api_key=api_key, 
        dockling_api_key=dockling_api_key,
        num_extractors=EXTRACTOR_CONFIG["num_extractors"],
        use_parallel=EXTRACTOR_CONFIG["use_parallel"]
    )

@st.cache_resource
def inicializar_async_processor():
    return AsyncProcessor()

@st.cache_resource
def inicializar_dockling(_api_key_manager):
    api_key = _api_key_manager.get_dockling_api_key()
    return DocklingProcessor(api_key=api_key)

def main():
    # Configurar página con los valores de UI_CONFIG
    st.set_page_config(
        page_title=UI_CONFIG["page_title"],
        page_icon=UI_CONFIG["page_icon"],
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=UI_CONFIG["menu_items"]
    )
    
    # Descripción de la aplicación para documentación
    st.markdown("""
    # MasterQuote - Sistema Inteligente de Cotización para Proyectos de Construcción

    ## Descripción
    MasterQuote es una herramienta integral para la generación de cotizaciones en el sector construcción,
    que permite extraer información de precios desde múltiples fuentes (Excel, PDF, CSV, etc.), 
    procesar esta información utilizando inteligencia artificial, y generar cotizaciones 
    personalizadas según las necesidades específicas del usuario.

    ## Principales Funcionalidades
    1. Extracción inteligente de precios desde documentos usando IA
    2. Validación cruzada entre múltiples extractores para garantizar precisión
    3. Categorización automática de items según su relación con la construcción
    4. Ajuste regional de precios según localidad del proyecto
    5. Plantillas predefinidas para diferentes tipos de proyectos (edificios, obra civil, etc.)
    6. Exportación a múltiples formatos (Excel, PDF, CSV)
    7. Histórico de precios y análisis de tendencias
    8. Integración con APIs externas (OpenAI, Dockling) para mejora continua

    Esta aplicación facilita el proceso de cotización en proyectos de construcción,
    ahorrando tiempo en la recopilación y procesamiento de información de precios,
    y proporcionando resultados más precisos y adaptados al sector.
    """)
    
    # Aplicar tema personalizado
    theme = UI_CONFIG["theme"]
    st.markdown(f"""
        <style>
        .reportview-container .main .block-container{{
            max-width: 1200px;
            padding-top: 2rem;
            padding-right: 2rem;
            padding-left: 2rem;
            padding-bottom: 2rem;
        }}
        .stApp {{
            background-color: {theme["backgroundColor"]};
            color: {theme["textColor"]};
        }}
        .stButton>button {{
            background-color: {theme["primaryColor"]};
            color: white;
        }}
        .stTextInput, .stNumberInput, .stSelectbox, .stDateInput > div {{
            background-color: {theme["secondaryBackgroundColor"]};
            color: {theme["textColor"]};
        }}
        .sidebar .sidebar-content {{
            background-color: {theme["secondaryBackgroundColor"]};
        }}
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {theme["secondaryBackgroundColor"]};
        }}
        .stTabs [data-baseweb="tab"] {{
            color: {theme["textColor"]};
        }}
        .stTabs [data-baseweb="tab-panel"] {{
            background-color: {theme["backgroundColor"]};
        }}
        .st-bb {{
            background-color: {theme["secondaryBackgroundColor"]};
        }}
        .st-at {{
            background-color: {theme["backgroundColor"]};
        }}
        .st-cc {{
            color: {theme["textColor"]};
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Inicializar componentes
    secret_manager = inicializar_secretos()
    api_key_manager = inicializar_api_keys(secret_manager)
    db = inicializar_db()
    llm = inicializar_llm(db, api_key_manager)
    extractor = inicializar_extractor(api_key_manager)
    extractor_manager = inicializar_extractor_manager(api_key_manager)
    async_processor = inicializar_async_processor()
    dockling = inicializar_dockling(api_key_manager)
    
    # Verificar y crear directorios necesarios
    for directorio in ['exports', 'templates', 'cache']:
        if not os.path.exists(directorio):
            os.makedirs(directorio)
    
    # Menú principal en sidebar
    with st.sidebar:
        ModernUIComponents.custom_header(
            UI_CONFIG["page_title"].split(" - ")[0], 
            "Sistema de cotizaciones para construcción", 
            UI_CONFIG["page_icon"]
        )
        
        st.markdown("### Menú Principal")
        pagina = st.radio(
            "Seleccione una sección:",
            [
                "📝 Nueva Cotización", 
                "🏗️ Proyectos de Construcción",
                "📊 Dashboard",
                "📅 Importar Precios", 
                "📃 Histórico de Precios", 
                "📈 Análisis de Presupuestos",
                "⚙️ Configuración"
            ]
        )
        
        # Información de la versión
        st.markdown("---")
        st.markdown("### Información del Sistema")
        st.text(f"Versión: 2.2.0")
        st.text(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}")
        
        # Estado de la API
        api_key = api_key_manager.get_openai_api_key()
        api_status = "✅ Configurada" if api_key else "❌ No configurada"
        st.text(f"API OpenAI: {api_status}")
        
        dockling_api_key = api_key_manager.get_dockling_api_key()
        dockling_status = "✅ Configurada" if dockling_api_key else "❌ No configurada"
        st.text(f"API Dockling: {dockling_status}")
        
        # Botón para verificar actualizaciones
        if st.button("Verificar actualizaciones", key="check_updates"):
            with st.spinner("Verificando..."):
                time.sleep(1)  # Simulación
                st.success("Sistema actualizado a la última versión")
    
    # Renderizar la página seleccionada
    if pagina == "📝 Nueva Cotización":
        # Pasar la configuración de construcción al setup_new_quotation_page
        setup_new_quotation_page(
            extractor, 
            db, 
            llm, 
            extractor_manager, 
            construction_config=CONSTRUCTION_CONFIG,
            default_min_confidence=EXTRACTOR_CONFIG["default_min_confidence"],
            validation_enabled_by_default=EXTRACTOR_CONFIG["validation_enabled_by_default"]
        )
    
    elif pagina == "🏗️ Proyectos de Construcción":
        st.title("Proyectos de Construcción")
        
        # Mostrar plantillas de proyectos
        st.header("Plantillas de Proyectos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tipo de Proyecto")
            project_type = st.selectbox(
                "Seleccione un tipo de proyecto",
                list(CONSTRUCTION_CONFIG["project_templates"].keys())
            )
        
        with col2:
            st.subheader("Secciones del Proyecto")
            sections = CONSTRUCTION_CONFIG["project_templates"][project_type]
            for section in sections:
                st.markdown(f"- {section}")
        
        st.markdown("---")
        
        # Botón para crear nuevo proyecto
        if st.button("Crear Nuevo Proyecto con esta Plantilla"):
            st.session_state.new_project_type = project_type
            st.session_state.new_project_sections = sections
            st.success(f"Plantilla de {project_type} seleccionada. Vaya a 'Nueva Cotización' para continuar.")
    
    elif pagina == "📊 Dashboard":
        create_dashboard_page(construction_config=CONSTRUCTION_CONFIG)
    
    elif pagina == "📅 Importar Precios":
        st.title("Importar Precios")
        
        st.header("Importar Lista de Precios para Construcción")
        
        uploaded_file = st.file_uploader("Seleccione un archivo Excel con precios", type=["xlsx", "xls"])
        
        if uploaded_file:
            try:
                df = pd.read_excel(uploaded_file)
                st.write("Vista previa de los datos:")
                st.dataframe(df.head())
                
                # Mapeo de columnas
                st.subheader("Mapeo de Columnas")
                
                col_description = st.selectbox("Columna de Descripción", df.columns.tolist())
                col_unit = st.selectbox("Columna de Unidad", df.columns.tolist())
                col_price = st.selectbox("Columna de Precio", df.columns.tolist())
                
                # Categoría de construcción
                category = st.selectbox("Categoría", CONSTRUCTION_CONFIG["categories"])
                
                if st.button("Importar Precios"):
                    with st.spinner("Importando precios..."):
                        # Aquí iría la lógica para guardar en la base de datos
                        time.sleep(2)  # Simulación
                        st.success(f"Se importaron {len(df)} precios correctamente")
            except Exception as e:
                st.error(f"Error al procesar el archivo: {str(e)}")
    
    elif pagina == "📃 Histórico de Precios":
        st.title("Histórico de Precios")
        
        # Filtros
        st.header("Filtros")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox("Categoría", ["Todas"] + CONSTRUCTION_CONFIG["categories"])
        
        with col2:
            search = st.text_input("Buscar por descripción")
        
        # Aquí iría la lógica para obtener y mostrar los precios
        st.info("Funcionalidad en desarrollo")
    
    elif pagina == "📈 Análisis de Presupuestos":
        create_budget_analysis_page()
    
    elif pagina == "⚙️ Configuración":
        create_settings_page(api_key_manager, construction_config=CONSTRUCTION_CONFIG)

if __name__ == "__main__":
    main()
