import streamlit as st
import pandas as pd
import os
import tempfile
import logging
from datetime import datetime
import yaml
import time

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
from modules.streamlit_ui_improvements import ModernUIComponents, create_dashboard_page, create_settings_page, setup_new_quotation_page

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar componentes
@st.cache_resource
def inicializar_secretos():
    return SecretManager()

@st.cache_resource
def inicializar_api_keys(_secret_manager):
    return APIKeyManager(_secret_manager)

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
    cache_expiry_days = 30
    return UniversalPriceExtractor(
        api_key=api_key, 
        dockling_api_key=dockling_api_key,
        use_cache=True, 
        cache_expiry_days=cache_expiry_days
    )

@st.cache_resource
def inicializar_async_processor():
    return AsyncProcessor()

@st.cache_resource
def inicializar_dockling(_api_key_manager):
    api_key = _api_key_manager.get_dockling_api_key()
    return DocklingProcessor(api_key=api_key)

def main():
    # Configurar página
    st.set_page_config(
        page_title="MasterQuote - Sistema Inteligente de Cotizaciones",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inicializar componentes
    secret_manager = inicializar_secretos()
    api_key_manager = inicializar_api_keys(secret_manager)
    db = inicializar_db()
    llm = inicializar_llm(db, api_key_manager)
    extractor = inicializar_extractor(api_key_manager)
    async_processor = inicializar_async_processor()
    dockling = inicializar_dockling(api_key_manager)
    
    # Verificar y crear directorios necesarios
    for directorio in ['exports', 'templates', 'cache']:
        if not os.path.exists(directorio):
            os.makedirs(directorio)
    
    # Menú principal en sidebar
    with st.sidebar:
        ModernUIComponents.custom_header(
            "MasterQuote", 
            "Sistema de cotizaciones con IA", 
            "💼"
        )
        
        st.markdown("### Menú Principal")
        pagina = st.radio(
            "Seleccione una sección:",
            [
                "📝 Nueva Cotización", 
                "📊 Dashboard",
                "📅 Importar Precios", 
                "📃 Histórico de Precios", 
                "⚙️ Configuración"
            ]
        )
        
        # Información de la versión
        st.markdown("---")
        st.markdown("### Información del Sistema")
        st.text(f"Versión: 2.0.0")
        st.text(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}")
        
        # Estado de la API
        api_key = api_key_manager.get_openai_api_key()
        api_status = "✅ Configurada" if api_key else "❌ No configurada"
        st.text(f"API OpenAI: {api_status}")
        
        # Botón para verificar actualizaciones
        if st.button("Verificar actualizaciones", key="check_updates"):
            with st.spinner("Verificando..."):
                time.sleep(1)  # Simulación
                st.success("Sistema actualizado a la última versión")
    
    # Renderizar la página seleccionada
    if pagina == "📝 Nueva Cotización":
        setup_new_quotation_page(extractor, db, llm)
    
    elif pagina == "📊 Dashboard":
        create_dashboard_page()
    
    elif pagina == "📅 Importar Precios":
        st.title("Importar Precios")
        # TODO: Implementar página de importación
    
    elif pagina == "📃 Histórico de Precios":
        st.title("Histórico de Precios")
        # TODO: Implementar página de histórico
    
    elif pagina == "⚙️ Configuración":
        create_settings_page(api_key_manager)

if __name__ == "__main__":
    main()
