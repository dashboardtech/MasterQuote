import streamlit as st
import pandas as pd
import os
import tempfile
import logging
from datetime import datetime
import yaml
from modules.price_database import PriceDatabase
from modules.cotizacion_llm import CotizacionLLM
from modules.data_loader import cargar_excel, validar_formato_excel, generar_resumen_excel
from modules.price_updater import actualizar_precios, aplicar_ajustes, validar_precios
from modules.exporter import exportar_cotizacion
from modules.universal_price_extractor import UniversalPriceExtractor

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializar base de datos
@st.cache_resource
def inicializar_db():
    return PriceDatabase("price_history.db")

# Inicializar LLM
@st.cache_resource
def inicializar_llm(_db):
    return CotizacionLLM(_db)

# Inicializar extractor universal
@st.cache_resource
def inicializar_extractor():
    api_key = os.environ.get("OPENAI_API_KEY", None)
    cache_expiry_days = 30  # Días para expiración de caché
    return UniversalPriceExtractor(api_key=api_key, use_cache=True, cache_expiry_days=cache_expiry_days)

# Función para aplicar estilos condicionales
def highlight_sugerencias(df):
    df_styled = pd.DataFrame('', index=df.index, columns=df.columns)
    
    if 'confianza' in df.columns:
        for i, v in enumerate(df['confianza']):
            if v == 'alta':
                df_styled.iloc[i, df.columns.get_loc('precio_sugerido')] = 'background-color: #c6efce'
            elif v == 'media':
                df_styled.iloc[i, df.columns.get_loc('precio_sugerido')] = 'background-color: #ffeb9c'
            elif v == 'baja' or v == 'media-baja':
                df_styled.iloc[i, df.columns.get_loc('precio_sugerido')] = 'background-color: #ffc7ce'
    
    return df_styled

# Configurar página
st.set_page_config(
    page_title="Sistema Inteligente de Cotizaciones",
    page_icon="💰",
    layout="wide"
)

# Título principal
st.title("Sistema Inteligente de Cotizaciones")
st.write("Sistema de cotizaciones con inteligencia artificial para sugerencia de precios")

# Inicializar componentes
db = inicializar_db()
llm = inicializar_llm(db)
extractor = inicializar_extractor()

# Crear sidebar para opciones
with st.sidebar:
    st.title("Opciones")
    
    # Opciones de procesamiento
    st.subheader("Procesamiento")
    usar_llm = st.checkbox("Usar IA para sugerencias", value=True)
    usar_bd = st.checkbox("Usar Base de Datos Histórica", value=True)
    
    # Información del proyecto
    st.subheader("Información del Proyecto")
    nombre_proyecto = st.text_input("Nombre del Proyecto", "Nueva Cotización")
    cliente = st.text_input("Cliente", "")
    
    # Ajustes de precios
    st.subheader("Ajustes de Precios")
    ajuste_global = st.number_input("Ajuste Global (%)", value=0.0, step=1.0)
    decimales = st.number_input("Decimales", value=2, min_value=0, max_value=4)
    
    # Acciones adicionales
    st.subheader("Acciones")
    btn_ver_historico = st.button("Ver Histórico de Precios")
    btn_importar_bd = st.button("Importar Excel a Base de Datos")

# Crear pestañas principales
tab1, tab2, tab3, tab4 = st.tabs(["Nueva Cotización", "Importar Precios", "Histórico de Precios", "Administrar Datos"])

with tab1:
    uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            # Guardar archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name
            
            # Cargar y validar Excel
            with st.spinner("Procesando archivo..."):
                # Cargar datos básicos
                df = cargar_excel(temp_path)
                
                # Validar formato
                reporte_validacion = validar_formato_excel(df)
                if not reporte_validacion['valido']:
                    st.error("Error en el formato del archivo:")
                    for error in reporte_validacion['errores']:
                        st.error(f"- {error}")
                    for advertencia in reporte_validacion['advertencias']:
                        st.warning(f"- {advertencia}")
                else:
                    # Procesar con LLM si está activado
                    if usar_llm:
                        df_procesado = llm.procesar_excel(temp_path)
                        
                        # Mostrar sugerencias
                        st.subheader("Sugerencias de Precios")
                        st.dataframe(
                            df_procesado[['actividades', 'cantidad', 'precio_sugerido', 'confianza', 'fuente', 'notas']].style.apply(highlight_sugerencias, axis=None),
                            height=300
                        )
                        
                        # Botón para aplicar sugerencias
                        if st.button("Aplicar Sugerencias"):
                            df = df_procesado.copy()
                            st.success("Sugerencias aplicadas")
                    
                    # Aplicar ajustes si hay
                    if ajuste_global != 0:
                        ajustes = {
                            'ajuste_global': ajuste_global,
                            'redondeo': decimales
                        }
                        df = aplicar_ajustes(df, ajustes)
                    
                    # Mostrar cotización editable
                    st.subheader("Cotización Final (Editable)")
                    df_editado = st.data_editor(
                        df,
                        num_rows="dynamic",
                        column_config={
                            "actividades": st.column_config.TextColumn("Actividades"),
                            "cantidad": st.column_config.NumberColumn("Cantidad", format="%.2f"),
                            "costo_unitario": st.column_config.NumberColumn("Costo Unitario", format="$%.2f"),
                            "costo_total": st.column_config.NumberColumn("Costo Total", format="$%.2f"),
                        },
                        disabled=["costo_total"],
                        hide_index=True,
                    )
                    
                    # Calcular y mostrar total
                    if 'costo_total' in df_editado.columns:
                        total = df_editado['costo_total'].sum()
                        st.metric("Total Cotización", f"${total:,.2f}")
                    
                    # Opciones de exportación
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Exportar Excel"):
                            try:
                                metadata = {
                                    'proyecto': nombre_proyecto,
                                    'cliente': cliente,
                                    'fecha': datetime.now().strftime("%Y-%m-%d")
                                }
                                
                                ruta_excel = exportar_cotizacion(
                                    df_editado,
                                    metadata=metadata,
                                    formato='xlsx'
                                )
                                
                                # Ofrecer descarga
                                with open(ruta_excel, "rb") as file:
                                    st.download_button(
                                        "Descargar Excel",
                                        file.read(),
                                        file_name=os.path.basename(ruta_excel),
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
                            except Exception as e:
                                st.error(f"Error al exportar: {str(e)}")
                    
                    with col2:
                        if st.button("Guardar en Base de Datos"):
                            try:
                                # Preparar items para BD
                                items = []
                                for _, row in df_editado.iterrows():
                                    items.append({
                                        'actividad': row['actividades'],
                                        'cantidad': row['cantidad'],
                                        'precio_unitario': row['costo_unitario']
                                    })
                                
                                # Guardar cotización
                                cotizacion_id = db.guardar_cotizacion(
                                    nombre_proyecto,
                                    cliente,
                                    items,
                                    notas="Cotización creada con IA"
                                )
                                
                                st.success(f"Cotización guardada con ID: {cotizacion_id}")
                                
                                # Aprender de la cotización
                                if usar_llm:
                                    llm.aprender_de_cotizacion(df_editado)
                                    
                            except Exception as e:
                                st.error(f"Error al guardar: {str(e)}")
        
        except Exception as e:
            st.error(f"Error al procesar archivo: {str(e)}")
            logger.exception("Error en procesamiento")
        
        finally:
            # Limpiar archivo temporal
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)

with tab2:
    st.subheader("Importar Precios")
    st.write("""
    Sube archivos con precios en cualquier formato (Excel, CSV, PDF, Word, imagen) 
    y el sistema los procesará automáticamente.
    """)
    
    # Opciones adicionales
    col1, col2 = st.columns(2)
    with col1:
        usar_ia = st.checkbox("Usar IA para procesamiento avanzado", value=True)
    with col2:
        modo_interactivo = st.checkbox("Modo interactivo (para formatos complejos)", value=False)
    
    # Uploader para múltiples formatos
    uploaded_files = st.file_uploader(
        "Arrastra aquí los archivos con precios", 
        type=["xlsx", "xls", "csv", "pdf", "docx", "doc", "jpg", "jpeg", "png", "txt"],
        accept_multiple_files=True,
        key="precio_uploader"
    )
    
    if uploaded_files:
        st.info(f"Se procesarán {len(uploaded_files)} archivos...")
        
        for uploaded_file in uploaded_files:
            # Guardar el archivo temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_path = tmp.name
            
            with st.spinner(f"Procesando {uploaded_file.name}..."):
                try:
                    # Configurar el extractor
                    if not usar_ia:
                        extractor.api_key = None
                    
                    # Procesar el archivo
                    df_extraido = extractor.extract_from_file(temp_path, interactive=modo_interactivo)
                    
                    # Mostrar resultados
                    st.subheader(f"Precios extraídos de {uploaded_file.name}")
                    st.dataframe(df_extraido)
                    
                    # Botón para importar a la base de datos
                    if st.button(f"Importar precios de {uploaded_file.name} a la BD"):
                        # Mapeo básico
                        mapping = {
                            'actividades': 'actividades',
                            'costo_unitario': 'costo_unitario'
                        }
                        db.importar_excel_a_db(temp_path, mapping)
                        st.success(f"Precios de {uploaded_file.name} importados a la base de datos")
                
                except Exception as e:
                    st.error(f"Error al procesar {uploaded_file.name}: {str(e)}")
                    logger.exception(f"Error en procesamiento de {uploaded_file.name}")
                
                finally:
                    # Limpiar archivos temporales
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

with tab3:
    st.subheader("Histórico de Precios")
    
    # Filtros
    col1, col2 = st.columns(2)
    with col1:
        categoria = st.selectbox("Filtrar por Categoría", ["Todas", "Construcción", "Instalación", "Acabados"])
    with col2:
        busqueda = st.text_input("Buscar Actividad")
    
    # Mostrar histórico
    if btn_ver_historico or busqueda:
        try:
            df_historico = db.obtener_historial_precios(
                categoria if categoria != "Todas" else None,
                busqueda
            )
            
            if not df_historico.empty:
                st.dataframe(df_historico)
            else:
                st.info("No se encontraron registros")
                
        except Exception as e:
            st.error(f"Error al cargar histórico: {str(e)}")

with tab4:
    # Opciones de administración
    opciones_admin = st.radio(
        "Opciones de Administración",
        ["Base de Datos", "Categorías", "Caché", "Configuración del Sistema"]
    )
    
    if opciones_admin == "Base de Datos":
        st.subheader("Administrar Base de Datos")
        
        # Importar Excel a BD
        admin_file = st.file_uploader("Importar Excel a Base de Datos", type=["xlsx"], key="admin_uploader")
        
        if admin_file is not None:
            try:
                # Guardar archivo temporal
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                    tmp.write(admin_file.getvalue())
                    admin_path = tmp.name
                
                # Opciones de mapeo
                st.write("Mapeo de Columnas")
                col1, col2 = st.columns(2)
                with col1:
                    col_actividad = st.text_input("Columna de Actividades", "actividades")
                with col2:
                    col_precio = st.text_input("Columna de Precios", "costo_unitario")
                
                # Botón para importar
                if st.button("Importar a Base de Datos"):
                    try:
                        mapping = {
                            'actividades': col_actividad,
                            'costo_unitario': col_precio
                        }
                        db.importar_excel_a_db(admin_path, mapping)
                        st.success("Datos importados correctamente")
                    except Exception as e:
                        st.error(f"Error al importar: {str(e)}")
                    finally:
                        if os.path.exists(admin_path):
                            os.remove(admin_path)
                            
            except Exception as e:
                st.error(f"Error al procesar archivo: {str(e)}")
    
    elif opciones_admin == "Categorías":
        st.subheader("Gestionar Categorías")
        col1, col2 = st.columns(2)
        
        with col1:
            nueva_categoria = st.text_input("Nueva Categoría")
            descripcion = st.text_area("Descripción", height=100)
            if st.button("Agregar Categoría"):
                try:
                    # Aquí iría la lógica para agregar categoría
                    st.success(f"Categoría '{nueva_categoria}' agregada")
                except Exception as e:
                    st.error(f"Error al agregar categoría: {str(e)}")
        
        with col2:
            # Lista de categorías (simulada por ahora)
            categorias = ["Construcción", "Instalación", "Acabados"]
            categoria_sel = st.selectbox("Seleccionar Categoría", categorias)
            if st.button("Eliminar Categoría"):
                try:
                    # Aquí iría la lógica para eliminar categoría
                    st.warning(f"¿Seguro de eliminar '{categoria_sel}'?")
                    confirmar_eliminar = st.button("Confirmar Eliminación")
                    if confirmar_eliminar:
                        st.success(f"Categoría '{categoria_sel}' eliminada")
                except Exception as e:
                    st.error(f"Error al eliminar categoría: {str(e)}")
    
    elif opciones_admin == "Caché":
        st.subheader("Administración de Caché")
        
        # Obtener estadísticas de caché
        if hasattr(extractor, 'cache_manager'):
            cache_stats = extractor.cache_manager.get_cache_stats()
            
            # Mostrar información de caché
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Archivos en caché", cache_stats["cached_files"])
            with col2:
                st.metric("Tasa de aciertos", f"{cache_stats['hit_rate_percent']}%")
            with col3:
                st.metric("Tamaño de caché", f"{cache_stats['cache_size_mb']} MB")
            
            # Mostrar directorio de caché
            st.info(f"Directorio de caché: {cache_stats['cache_directory']}")
            
            # Opciones de administración
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Limpiar caché expirada"):
                    deleted = extractor.cache_manager.clear_expired_cache()
                    st.success(f"Se eliminaron {deleted} archivos de caché expirados")
            
            with col2:
                if st.button("Limpiar toda la caché"):
                    st.warning("¿Estás seguro de que deseas eliminar toda la caché?")
                    confirmar_limpiar = st.button("Confirmar eliminación")
                    if confirmar_limpiar:
                        deleted = extractor.cache_manager.clear_all_cache()
                        st.success(f"Se eliminaron {deleted} archivos de caché")
            
            # Configuración de caché
            st.subheader("Configuración de caché")
            dias_expiracion = st.slider(
                "Días para expirar caché", 
                min_value=1, 
                max_value=90, 
                value=30,
                help="Número de días después de los cuales la caché se considera obsoleta"
            )
            
            if st.button("Aplicar configuración"):
                try:
                    extractor.cache_manager.expiry_days = dias_expiracion
                    st.success(f"Se configuró la expiración de caché a {dias_expiracion} días")
                except Exception as e:
                    st.error(f"Error al configurar caché: {str(e)}")
        else:
            st.warning("El sistema de caché no está habilitado en el extractor.")
            
            if st.button("Habilitar caché"):
                st.info("Para habilitar la caché, reinicia la aplicación.")
    
    elif opciones_admin == "Configuración del Sistema":
        st.subheader("Configuración del Sistema")
        
        with st.expander("Configuración de API"):
            col1, col2 = st.columns(2)
            with col1:
                api_key = st.text_input("API Key para OpenAI", type="password",
                                        value=os.environ.get("OPENAI_API_KEY", ""))
            with col2:
                modelo_llm = st.selectbox("Modelo LLM", ["gpt-3.5-turbo", "gpt-4"])
            
            if st.button("Guardar API Key"):
                try:
                    # Guardar temporalmente (solo para esta sesión)
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.success("API Key guardada para esta sesión")
                except Exception as e:
                    st.error(f"Error al guardar API Key: {str(e)}")

if __name__ == "__main__":
    # Verificar y crear directorios necesarios
    for directorio in ['exports', 'templates']:
        if not os.path.exists(directorio):
            os.makedirs(directorio)
