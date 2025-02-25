import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, List, Any, Optional
from datetime import datetime

class ModernUIComponents:
    """
    Componentes modernos de UI para Streamlit con mejor estética y funcionalidad.
    """
    
    @staticmethod
    def custom_header(title: str, subtitle: Optional[str] = None, icon: Optional[str] = None):
        """
        Crea un encabezado personalizado con estilo mejorado.
        """
        header_html = f"""
        <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:10px;">
            <h1 style="color:#2e4057;margin-bottom:0px;">
                {icon + " " if icon else ""}{title}
            </h1>
            {f'<p style="color:#555555;">{subtitle}</p>' if subtitle else ''}
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    @staticmethod
    def info_card(title: str, value: Any, delta: Optional[float] = None, 
                  suffix: Optional[str] = None, color: str = "#2e4057"):
        """
        Crea una tarjeta de información estilo dashboard.
        """
        delta_html = ""
        if delta is not None:
            delta_color = "green" if delta >= 0 else "red"
            delta_symbol = "▲" if delta >= 0 else "▼"
            delta_html = f'<span style="color:{delta_color};font-size:1rem;">{delta_symbol} {abs(delta):.1f}%</span>'
        
        value_display = f"{value}{suffix if suffix else ''}"
        
        card_html = f"""
        <div style="background-color:white;padding:15px;border-radius:5px;box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            <h3 style="margin:0;color:#555555;font-size:0.8rem;">{title}</h3>
            <p style="margin:0;color:{color};font-size:1.8rem;font-weight:bold;">{value_display} {delta_html}</p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
    
    @staticmethod
    def dashboard_cards(metrics: List[Dict[str, Any]], columns: int = 3):
        """
        Crea una fila de tarjetas para dashboard.
        """
        cols = st.columns(columns)
        for i, metric in enumerate(metrics):
            with cols[i % columns]:
                ModernUIComponents.info_card(
                    title=metric.get("title", ""),
                    value=metric.get("value", ""),
                    delta=metric.get("delta"),
                    suffix=metric.get("suffix"),
                    color=metric.get("color", "#2e4057")
                )
    
    @staticmethod
    def create_price_histogram(df: pd.DataFrame, price_column: str = 'costo_unitario',
                              title: str = "Distribución de Precios"):
        """
        Crea un histograma de precios con Plotly.
        """
        if df.empty or price_column not in df.columns:
            st.warning(f"No hay datos disponibles para mostrar el histograma de {price_column}")
            return
        
        fig = px.histogram(
            df, 
            x=price_column,
            title=title,
            labels={price_column: "Precio"},
            color_discrete_sequence=['#3366CC'],
            opacity=0.8
        )
        
        fig.update_layout(
            xaxis_title="Precio",
            yaxis_title="Frecuencia",
            bargap=0.05,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_price_trend_chart(price_history: pd.DataFrame, date_column: str = 'fecha_actualizacion',
                                price_column: str = 'precio', item_column: str = 'actividad',
                                title: str = "Tendencia de Precios"):
        """
        Crea un gráfico de tendencia de precios en el tiempo.
        """
        if price_history.empty:
            st.warning("No hay datos históricos para mostrar tendencias")
            return
        
        if price_history[date_column].dtype == 'object':
            price_history[date_column] = pd.to_datetime(price_history[date_column])
        
        price_history = price_history.sort_values(date_column)
        top_items = price_history[item_column].value_counts().head(5).index.tolist()
        filtered_data = price_history[price_history[item_column].isin(top_items)]
        
        fig = px.line(
            filtered_data,
            x=date_column,
            y=price_column,
            color=item_column,
            title=title,
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Precio",
            legend_title="Actividad",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def file_upload_area(accept_multiple_files: bool = True, 
                        file_types: List[str] = None, 
                        key: str = "file_uploader"):
        """
        Crea un área mejorada para subir archivos.
        """
        file_types = file_types or ["xlsx", "xls", "csv", "pdf", "docx", "doc", "jpg", "jpeg", "png", "txt"]
        
        st.markdown("""
        <style>
        .stFileUploader > div > input[type="file"] {
            display: none;
        }
        .stFileUploader > div > button {
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            border: none;
            transition: background-color 0.3s;
        }
        .stFileUploader > div > button:hover {
            background-color: #45a049;
        }
        .drag-zone {
            border: 2px dashed #cccccc;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            background-color: #f8f9fa;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="drag-zone">
            <p>Arrastra aquí tus archivos o haz clic en 'Browse files'</p>
            <p style="font-size:12px;color:#666666;">Formatos aceptados: Excel, CSV, PDF, Word, imágenes, texto</p>
        </div>
        """, unsafe_allow_html=True)
        
        return st.file_uploader(
            "Selecciona los archivos",
            type=file_types,
            accept_multiple_files=accept_multiple_files,
            key=key
        )
    
    @staticmethod
    def create_activity_table(df: pd.DataFrame, 
                             editable: bool = True,
                             with_confidence: bool = False,
                             height: int = 300):
        """
        Crea una tabla de actividades con formato mejorado.
        """
        if with_confidence and 'confianza' in df.columns:
            def highlight_confianza(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                for i, conf in enumerate(df['confianza']):
                    if conf == 'alta':
                        styles.iloc[i, df.columns.get_loc('precio_sugerido')] = 'background-color: #c6efce'
                    elif conf == 'media':
                        styles.iloc[i, df.columns.get_loc('precio_sugerido')] = 'background-color: #ffeb9c'
                    elif conf in ['baja', 'media-baja']:
                        styles.iloc[i, df.columns.get_loc('precio_sugerido')] = 'background-color: #ffc7ce'
                return styles
            
            if editable:
                return st.data_editor(
                    df.style.apply(highlight_confianza, axis=None),
                    height=height,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config={
                        "confianza": st.column_config.SelectboxColumn(
                            "Confianza",
                            options=["alta", "media", "baja", "media-baja"],
                            width="small",
                        )
                    }
                )
            else:
                return st.dataframe(
                    df.style.apply(highlight_confianza, axis=None),
                    height=height,
                    use_container_width=True
                )
        else:
            if editable:
                return st.data_editor(
                    df,
                    height=height,
                    use_container_width=True,
                    num_rows="dynamic"
                )
            else:
                return st.dataframe(
                    df,
                    height=height,
                    use_container_width=True
                )
