import os
import logging
import tempfile
import pandas as pd
import requests
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class DocklingProcessor:
    """
    Integración con Dockling para procesamiento avanzado de documentos.
    Proporciona extracción inteligente de contenido estructurado desde
    diversos formatos de documentos.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.dockling.com/v1"):
        """
        Inicializa el procesador de Dockling.
        
        Args:
            api_key: API key para el servicio Dockling (opcional)
            base_url: URL base del API de Dockling
        """
        self.api_key = api_key or os.environ.get("DOCKLING_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            logger.warning("API key de Dockling no configurada. Algunas funcionalidades estarán limitadas.")
    
    def extract_tables_from_document(self, file_path: str) -> List[pd.DataFrame]:
        """
        Extrae tablas desde un documento usando Dockling API.
        
        Args:
            file_path: Ruta al archivo a procesar
            
        Returns:
            Lista de DataFrames, cada uno representando una tabla extraída
        """
        if not self.api_key:
            logger.warning("No se ha configurado API key de Dockling para extraer tablas")
            return []
        
        try:
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                logger.error(f"El archivo no existe: {file_path}")
                return []
                
            # Preparar archivo para envío
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                
                # Configurar headers de autenticación
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                # Realizar petición al API
                try:
                    response = requests.post(
                        f"{self.base_url}/extract/tables",
                        files=files,
                        headers=headers,
                        timeout=30  # Añadir timeout para evitar bloqueos
                    )
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error de conexión con API Dockling: {str(e)}")
                    return []
                
                if response.status_code != 200:
                    error_msg = f"Error en API Dockling: {response.status_code}"
                    try:
                        error_detail = response.json().get('error', 'Sin detalles')
                        error_msg += f", {error_detail}"
                    except:
                        error_msg += f", {response.text[:100]}"
                    
                    logger.error(error_msg)
                    return []
                
                # Procesar respuesta
                result = response.json()
                
                # Convertir tablas a DataFrames
                tables = []
                for table_data in result.get('tables', []):
                    if 'data' in table_data and table_data['data']:
                        # Convertir a DataFrame
                        df = pd.DataFrame(table_data['data'])
                        
                        # Si la primera fila parece un encabezado, usarla como tal
                        if table_data.get('has_header', True):
                            df.columns = df.iloc[0]
                            df = df.iloc[1:].reset_index(drop=True)
                        
                        tables.append(df)
                
                return tables
                
        except Exception as e:
            logger.exception(f"Error al extraer tablas con Dockling: {str(e)}")
            raise
    
    def extract_price_data(self, file_path: str) -> pd.DataFrame:
        """
        Extrae específicamente datos de precios desde un documento usando Dockling.
        
        Args:
            file_path: Ruta al archivo a procesar
            
        Returns:
            DataFrame con actividades y precios normalizados
        """
        if not self.api_key:
            logger.warning("No se ha configurado API key de Dockling para extraer datos de precios")
            return pd.DataFrame()
        
        try:
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                logger.error(f"El archivo no existe: {file_path}")
                return pd.DataFrame()
                
            # Preparar archivo para envío
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                
                # Configurar headers de autenticación
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                # Realizar petición al API
                try:
                    response = requests.post(
                        f"{self.base_url}/extract/prices",
                        files=files,
                        headers=headers,
                        timeout=30  # Añadir timeout para evitar bloqueos
                    )
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error de conexión con API Dockling (precios): {str(e)}")
                    return pd.DataFrame()
                
                if response.status_code != 200:
                    error_msg = f"Error en API Dockling (precios): {response.status_code}"
                    try:
                        error_detail = response.json().get('error', 'Sin detalles')
                        error_msg += f", {error_detail}"
                    except:
                        error_msg += f", {response.text[:100]}"
                    
                    logger.error(error_msg)
                    return pd.DataFrame()
                
                # Procesar respuesta
                result = response.json()
                
                # Verificar si hay datos extraídos
                if 'data' not in result or not result['data']:
                    logger.warning(f"No se encontraron datos de precios en el documento {file_path}")
                    return pd.DataFrame()
                
                # Convertir a DataFrame
                df = pd.DataFrame(result['data'])
                
                # Mapear columnas al formato esperado por el sistema
                column_mapping = {
                    'activity': 'actividades',
                    'description': 'descripcion',
                    'quantity': 'cantidad',
                    'unit_price': 'costo_unitario',
                    'total_price': 'costo_total'
                }
                
                # Renombrar columnas según mapeo
                df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
                
                # Asegurar que existen las columnas necesarias
                required_columns = ['actividades', 'cantidad', 'costo_unitario']
                for col in required_columns:
                    if col not in df.columns:
                        if col == 'cantidad':
                            df[col] = 1
                        else:
                            df[col] = None
                
                # Calcular costo total si no existe
                if 'costo_total' not in df.columns:
                    df['costo_total'] = df['cantidad'] * df['costo_unitario']
                
                return df
                
        except Exception as e:
            logger.exception(f"Error al extraer datos de precios con Dockling: {str(e)}")
            return pd.DataFrame()
    
    def convert_document_to_text(self, file_path: str) -> str:
        """
        Convierte un documento a texto plano usando Dockling.
        
        Args:
            file_path: Ruta al archivo a procesar
            
        Returns:
            Texto extraído del documento
        """
        if not self.api_key:
            logger.warning("No se ha configurado API key de Dockling para convertir documento a texto")
            return ""
        
        try:
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                logger.error(f"El archivo no existe: {file_path}")
                return ""
                
            # Preparar archivo para envío
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                
                # Configurar headers de autenticación
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                # Realizar petición al API
                try:
                    response = requests.post(
                        f"{self.base_url}/convert/text",
                        files=files,
                        headers=headers,
                        timeout=30
                    )
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error de conexión con API Dockling (texto): {str(e)}")
                    return ""
                
                if response.status_code != 200:
                    error_msg = f"Error en API Dockling (texto): {response.status_code}"
                    try:
                        error_detail = response.json().get('error', 'Sin detalles')
                        error_msg += f", {error_detail}"
                    except:
                        error_msg += f", {response.text[:100]}"
                    
                    logger.error(error_msg)
                    return ""
                
                # Obtener texto extraído
                return response.text
                
        except Exception as e:
            logger.exception(f"Error al convertir documento a texto con Dockling: {str(e)}")
            return ""
    
    def analyze_document_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Analiza la estructura de un documento usando Dockling.
        
        Args:
            file_path: Ruta al archivo a procesar
            
        Returns:
            Diccionario con información sobre la estructura del documento
        """
        if not self.api_key:
            logger.warning("No se ha configurado API key de Dockling para analizar estructura")
            return {}
        
        try:
            # Verificar que el archivo existe
            if not os.path.exists(file_path):
                logger.error(f"El archivo no existe: {file_path}")
                return {}
                
            # Preparar archivo para envío
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f)}
                
                # Configurar headers de autenticación
                headers = {
                    'Authorization': f'Bearer {self.api_key}'
                }
                
                # Realizar petición al API
                try:
                    response = requests.post(
                        f"{self.base_url}/analyze/structure",
                        files=files,
                        headers=headers,
                        timeout=30
                    )
                except requests.exceptions.RequestException as e:
                    logger.error(f"Error de conexión con API Dockling (estructura): {str(e)}")
                    return {}
                
                if response.status_code != 200:
                    error_msg = f"Error en API Dockling (estructura): {response.status_code}"
                    try:
                        error_detail = response.json().get('error', 'Sin detalles')
                        error_msg += f", {error_detail}"
                    except:
                        error_msg += f", {response.text[:100]}"
                    
                    logger.error(error_msg)
                    return {}
                
                # Procesar respuesta
                return response.json()
                
        except Exception as e:
            logger.exception(f"Error al analizar estructura con Dockling: {str(e)}")
            return {}
