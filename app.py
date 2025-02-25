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
    Importa precios desde archivos o ingresa directamente el texto con los precios.
    El sistema procesará automáticamente la información en cualquier formato.
    """)
    
    # Crear pestañas para los diferentes métodos de entrada
    import_tab1, import_tab2 = st.tabs(["Subir Archivo", "Ingresar Texto"])
    
    with import_tab1:
        st.write("Sube archivos con precios en cualquier formato (Excel, CSV, PDF, Word, imagen)")

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
        
    with import_tab2:
        st.write("""
        Pega el texto con los precios aquí. El sistema procesará:
        1. Texto simple con precios
        2. Datos copiados de Excel (formato tabular)
        3. Listas de precios en cualquier formato
        """)
        
        # Opciones de formato
        formato_entrada = st.radio(
            "Formato de entrada",
            ["Texto Simple", "Datos de Excel (Tabular)"],
            help="Selecciona 'Datos de Excel' si copiaste directamente desde Excel"
        )
        
        # Área de texto para entrada directa
        texto_precios = st.text_area(
            "Texto con Precios",
            height=300,
            placeholder="Ejemplo para texto simple:\nLaptop Dell XPS 13 - $1,299.99\nMonitor LG 27\" - $399.99 x 2\n\nO pega datos copiados directamente de Excel"
        )
        
        # Columnas para mapeo si es formato tabular
        if formato_entrada == "Datos de Excel (Tabular)":
            st.write("Configuración de Columnas:")
            
            tipo_tabla = st.radio(
                "Tipo de Tabla",
                ["Tabla Simple", "Tabla de Construcción (Mano de obra + Material)"],
                help="Selecciona 'Tabla de Construcción' si tu tabla tiene columnas separadas para mano de obra y material"
            )
            
            if tipo_tabla == "Tabla Simple":
                col1, col2, col3 = st.columns(3)
                with col1:
                    col_descripcion = st.number_input(
                        "Columna de Descripción (1-based)",
                        min_value=1,
                        value=1,
                        help="Número de columna que contiene la descripción"
                    )
                with col2:
                    col_precio = st.number_input(
                        "Columna de Precio (1-based)",
                        min_value=1,
                        value=2,
                        help="Número de columna que contiene el precio"
                    )
                with col3:
                    col_cantidad = st.number_input(
                        "Columna de Cantidad (1-based)",
                        min_value=1,
                        value=2,
                        help="Número de columna que contiene la cantidad"
                    )
            else:
                col1, col2 = st.columns(2)
                with col1:
                    col_descripcion = st.number_input(
                        "Columna de Descripción",
                        min_value=1,
                        value=1,
                        help="Número de columna que contiene la descripción"
                    )
                    col_cantidad = st.number_input(
                        "Columna de Cantidad",
                        min_value=1,
                        value=2,
                        help="Número de columna que contiene la cantidad"
                    )
                with col2:
                    col_mano_obra = st.number_input(
                        "Columna P.U. Mano de Obra",
                        min_value=1,
                        value=4,
                        help="Número de columna que contiene el precio unitario de mano de obra"
                    )
                    col_material = st.number_input(
                        "Columna P.U. Material",
                        min_value=1,
                        value=6,
                        help="Número de columna que contiene el precio unitario de material"
                    )
        
        # Botón para procesar texto
        if st.button("Procesar Texto") and texto_precios:
            try:
                # Procesar según el formato seleccionado
                if formato_entrada == "Datos de Excel (Tabular)":
                    # Convertir texto tabular a DataFrame
                    try:
                        # Dividir el texto en líneas y procesar como CSV
                        import io
                        import csv
                        
                        # Limpiar el texto y convertir a CSV
                        lines = texto_precios.strip().split('\n')
                        processed_lines = []
                        for line in lines:
                            # Dividir por tabulaciones o espacios múltiples
                            cells = [cell.strip() for cell in line.split('\t')]
                            if len(cells) == 1:  # Si no hay tabs, dividir por espacios
                                cells = [cell.strip() for cell in line.split('  ') if cell.strip()]
                            processed_lines.append(cells)
                        
                        # Crear DataFrame
                        df = pd.DataFrame(processed_lines)
                        
                        # Extraer columnas según la configuración del usuario
                        df_final = pd.DataFrame()
                        df_final['actividades'] = df[col_descripcion - 1]
                        
                        if tipo_tabla == "Tabla Simple":
                            df_final['costo_unitario'] = df[col_precio - 1]
                            if col_cantidad:
                                df_final['cantidad'] = df[col_cantidad - 1]
                        else:
                            # Procesar tabla de construcción
                            df_final['cantidad'] = df[col_cantidad - 1]
                            df_final['mano_obra_unitario'] = df[col_mano_obra - 1]
                            df_final['material_unitario'] = df[col_material - 1]
                            
                            # Convertir y limpiar datos
                            df_final['mano_obra_unitario'] = df_final['mano_obra_unitario'].apply(lambda x: str(x).replace('$', '').replace(',', ''))
                            df_final['material_unitario'] = df_final['material_unitario'].apply(lambda x: str(x).replace('$', '').replace(',', ''))
                            
                            # Convertir a números
                            df_final['mano_obra_unitario'] = pd.to_numeric(df_final['mano_obra_unitario'], errors='coerce')
                            df_final['material_unitario'] = pd.to_numeric(df_final['material_unitario'], errors='coerce')
                            
                            # Calcular costo unitario total
                            df_final['costo_unitario'] = df_final['mano_obra_unitario'].fillna(0) + df_final['material_unitario'].fillna(0)
                        
                        # Limpiar y convertir datos comunes
                        if 'costo_unitario' in df_final.columns:
                            df_final['costo_unitario'] = df_final['costo_unitario'].apply(lambda x: str(x).replace('$', '').replace(',', '') if isinstance(x, str) else x)
                            df_final['costo_unitario'] = pd.to_numeric(df_final['costo_unitario'], errors='coerce')
                        
                        if 'cantidad' in df_final.columns:
                            df_final['cantidad'] = pd.to_numeric(df_final['cantidad'], errors='coerce')
                            df_final['cantidad'] = df_final['cantidad'].fillna(1)
                            df_final['costo_total'] = df_final['costo_unitario'] * df_final['cantidad']
                        
                        # Eliminar filas con precios nulos o 0
                        df_final = df_final[df_final['costo_unitario'].notna() & (df_final['costo_unitario'] > 0)]
                        
                    except Exception as e:
                        st.error(f"Error al procesar datos tabulares: {str(e)}")
                        logger.exception("Error procesando datos tabulares")
                        df_final = None
                        
                else:
                    # Crear un DataFrame simple para el texto
                    df_final = pd.DataFrame()
                
                try:
                    with st.spinner("Procesando texto..."):
                        if formato_entrada == "Datos de Excel (Tabular)":
                            # Usar directamente el DataFrame procesado
                            df = df_final
                        else:
                            # Procesar texto simple con el extractor universal
                            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
                                tmp.write(texto_precios)
                                temp_path = tmp.name
                                # Procesar con el extractor universal
                                df = extractor.extract_from_file(temp_path)
                                # Limpiar archivo temporal
                                os.unlink(temp_path)
                        
                        if df is not None and not df.empty:
                            st.success("¡Texto procesado exitosamente!")
                            
                            # Mostrar resultados
                            st.subheader("Resultados Extraídos")
                            st.dataframe(df)
                            
                            # Mostrar estadísticas
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total de Items", len(df))
                            with col2:
                                total = df['costo_total'].sum() if 'costo_total' in df.columns else (df['costo_unitario'] * df.get('cantidad', 1)).sum()
                                st.metric("Costo Total", f"${total:,.2f}")
                            with col3:
                                promedio = df['costo_unitario'].mean()
                                st.metric("Precio Promedio", f"${promedio:,.2f}")
                            
                            # Opción para guardar en la base de datos
                            if st.button("Guardar en Base de Datos", key="save_text_to_db"):
                                items = []
                                for _, row in df.iterrows():
                                    item = {
                                        'actividad': row['actividades'],
                                        'precio': float(row['costo_unitario']),
                                        'fecha': datetime.now(),
                                        'fuente': 'entrada_texto',
                                        'proyecto': nombre_proyecto,
                                        'cliente': cliente
                                    }
                                    items.append(item)
                                
                                db.agregar_items(items)
                                st.success("¡Datos guardados en la base de datos!")
                        else:
                            st.warning("No se pudieron extraer datos del texto proporcionado.")
                            
                finally:
                    # Limpiar archivo temporal
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
            except Exception as e:
                st.error(f"Error al procesar el texto: {str(e)}")
                logger.exception("Error al procesar texto ingresado")
    
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
