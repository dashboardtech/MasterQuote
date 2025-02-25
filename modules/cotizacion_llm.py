import pandas as pd
import json
import os
from datetime import datetime
import unidecode
from typing import List, Dict, Any, Optional, Tuple
import logging
import yaml
from .price_database import PriceDatabase

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CotizacionLLM:
    def __init__(self, db_connector: PriceDatabase, api_key: Optional[str] = None):
        """
        Inicializa el procesador de cotizaciones con LLM.
        
        Args:
            db_connector: Conector a la base de datos de precios
            api_key: API key para OpenAI (opcional)
        """
        self.db = db_connector
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No se ha configurado API key para OpenAI. Algunas funciones estarán limitadas.")
        else:
            logger.info("API key de OpenAI configurada correctamente")
        
        logger.info("Inicializado procesador de cotizaciones con LLM")
        
        # Cargar configuración
        try:
            with open("config.yaml", 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
                logger.info("Configuración cargada exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar configuración: {str(e)}")
            self.config = {}
    
    def procesar_excel(self, excel_path: str) -> pd.DataFrame:
        """
        Procesa un Excel de cotización y sugiere precios usando LLM.
        
        Args:
            excel_path (str): Ruta al archivo Excel con actividades
            
        Returns:
            pd.DataFrame: DataFrame con sugerencias de precios
        """
        try:
            # Cargar el Excel
            df = pd.read_excel(excel_path)
            
            # Normalizar columnas
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Verificar columna de actividades
            if 'actividades' not in df.columns:
                raise ValueError("El Excel debe contener una columna 'actividades'")
            
            # Verificar columna de cantidad
            if 'cantidad' not in df.columns:
                df['cantidad'] = 1
                logger.info("Columna 'cantidad' no encontrada. Se asigna valor por defecto: 1")
            
            # Preparar columnas para los precios
            if 'costo_unitario' not in df.columns:
                df['costo_unitario'] = None
            if 'costo_total' not in df.columns:
                df['costo_total'] = None
            
            # Columnas para sugerencias
            df['precio_sugerido'] = None
            df['confianza'] = None
            df['fuente'] = None
            df['notas'] = None
            
            # Procesar cada actividad
            actividades_sin_precio = []
            
            for idx, row in df.iterrows():
                actividad = row['actividades']
                
                # Si ya tiene precio, no sugerir
                if pd.notna(row['costo_unitario']) and row['costo_unitario'] > 0:
                    continue
                    
                # Consultar precio en base de datos
                sugerencia = self.db.obtener_precio_sugerido(actividad)
                
                if sugerencia['precio'] is not None:
                    df.at[idx, 'precio_sugerido'] = sugerencia['precio']
                    df.at[idx, 'confianza'] = sugerencia['confianza']
                    df.at[idx, 'fuente'] = 'base de datos'
                    
                    if sugerencia['tipo_coincidencia'] == 'similar':
                        df.at[idx, 'notas'] = f"Basado en actividad similar: {sugerencia['actividad_similar']}"
                else:
                    actividades_sin_precio.append(actividad)
            
            # Usar LLM para actividades sin precio en BD
            if actividades_sin_precio:
                sugerencias_llm = self._obtener_sugerencias_llm(actividades_sin_precio)
                
                for idx, row in df.iterrows():
                    actividad = row['actividades']
                    if actividad in sugerencias_llm and pd.isna(df.at[idx, 'precio_sugerido']):
                        sugerencia = sugerencias_llm[actividad]
                        df.at[idx, 'precio_sugerido'] = sugerencia['precio']
                        df.at[idx, 'confianza'] = 'media-baja'
                        df.at[idx, 'fuente'] = 'LLM'
                        df.at[idx, 'notas'] = sugerencia.get('razonamiento', '')
            
            # Calcular valores totales
            for idx, row in df.iterrows():
                if pd.isna(row['costo_unitario']) and pd.notna(row['precio_sugerido']):
                    df.at[idx, 'costo_unitario'] = row['precio_sugerido']
                
                if pd.notna(row['costo_unitario']):
                    df.at[idx, 'costo_total'] = row['cantidad'] * row['costo_unitario']
            
            logger.info(f"Excel procesado exitosamente: {excel_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error al procesar Excel: {str(e)}")
            raise
    
    def _obtener_sugerencias_llm(self, actividades: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Consulta al LLM para obtener sugerencias de precios para actividades desconocidas.
        
        Args:
            actividades (List[str]): Lista de nombres de actividades
            
        Returns:
            Dict: Diccionario con sugerencias por actividad
        """
        try:
            # Obtener contexto de la base de datos
            actividades_conocidas = self._obtener_actividades_frecuentes(10)
            
            # Construir prompt para el LLM
            prompt = self._construir_prompt(actividades, actividades_conocidas)
            
            # Simular respuesta del LLM (aquí se integraría con Claude)
            # Por ahora retornamos sugerencias basadas en promedios y heurísticas simples
            sugerencias = {}
            precios_base = {
                'limpieza': 1000,
                'pintura': 1200,
                'instalacion': 1500,
                'reparacion': 800,
                'mantenimiento': 1200
            }
            
            for actividad in actividades:
                # Análisis simple basado en palabras clave
                precio_base = 1000  # valor por defecto
                for keyword, precio in precios_base.items():
                    if keyword in actividad.lower():
                        precio_base = precio
                        break
                
                # Ajustar precio según complejidad aparente
                complejidad = 1.0
                if 'complejo' in actividad.lower() or 'especializado' in actividad.lower():
                    complejidad = 1.5
                elif 'simple' in actividad.lower() or 'básico' in actividad.lower():
                    complejidad = 0.8
                
                precio_sugerido = precio_base * complejidad
                
                sugerencias[actividad] = {
                    'precio': precio_sugerido,
                    'razonamiento': f"Precio base ajustado por complejidad y tipo de actividad",
                    'similitud': None
                }
            
            logger.info(f"Sugerencias generadas para {len(actividades)} actividades")
            return sugerencias
            
        except Exception as e:
            logger.error(f"Error al obtener sugerencias del LLM: {str(e)}")
            return {}
    
    def _obtener_actividades_frecuentes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Obtiene las actividades más frecuentes de la base de datos como contexto.
        
        Args:
            limit (int): Número máximo de actividades a obtener
            
        Returns:
            List[Dict]: Lista de actividades con sus precios
        """
        try:
            # Consulta para obtener actividades frecuentes con sus últimos precios
            query = """
                SELECT 
                    a.nombre,
                    p.precio,
                    COUNT(*) as frecuencia
                FROM actividades a
                JOIN precios p ON a.id = p.actividad_id
                GROUP BY a.id
                ORDER BY frecuencia DESC, p.fecha_actualizacion DESC
                LIMIT ?
            """
            
            self.db.cursor.execute(query, (limit,))
            resultados = self.db.cursor.fetchall()
            
            actividades = []
            for resultado in resultados:
                actividades.append({
                    "nombre": resultado[0],
                    "precio": resultado[1],
                    "frecuencia": resultado[2]
                })
            
            return actividades
            
        except Exception as e:
            logger.error(f"Error al obtener actividades frecuentes: {str(e)}")
            return []
    
    def _construir_prompt(self, 
                         actividades: List[str], 
                         contexto: List[Dict[str, Any]]) -> str:
        """
        Construye el prompt para enviar al LLM.
        
        Args:
            actividades (List[str]): Actividades sin precio
            contexto (List[Dict]): Actividades conocidas con precios
            
        Returns:
            str: Prompt formateado
        """
        # Formatear el contexto
        contexto_str = "\n".join([
            f"- {item['nombre']}: ${item['precio']} (usado {item['frecuencia']} veces)" 
            for item in contexto
        ])
        
        # Actividades a evaluar
        actividades_str = "\n".join([f"- {act}" for act in actividades])
        
        prompt = f"""Eres un asistente especializado en cotizaciones de construcción y servicios.
        
CONTEXTO:
Las siguientes son actividades conocidas con sus precios de referencia:
{contexto_str}

INSTRUCCIONES:
Necesito estimar precios para las siguientes actividades:
{actividades_str}

Por cada actividad, proporciona:
1. Un precio estimado basado en las actividades de referencia
2. Un breve razonamiento de por qué sugieres ese precio
3. Una indicación de similitud con alguna actividad de referencia, si aplica

Formato de respuesta requerido (JSON):
{{
  "actividad1": {{
    "precio": 1000,
    "razonamiento": "Explicación concisa",
    "similitud": "actividad similar (si aplica)"
  }},
  "actividad2": {{
    "precio": 800,
    "razonamiento": "Explicación concisa",
    "similitud": "actividad similar (si aplica)"
  }}
}}

Responde ÚNICAMENTE con el JSON, sin texto adicional.
"""
        return prompt
    
    def aprender_de_cotizacion(self, cotizacion_df: pd.DataFrame) -> None:
        """
        Aprende de una cotización completada para mejorar futuras sugerencias.
        
        Args:
            cotizacion_df (pd.DataFrame): DataFrame con la cotización finalizada
        """
        try:
            for idx, row in cotizacion_df.iterrows():
                if pd.notna(row['costo_unitario']) and pd.notna(row['actividades']):
                    # Guardar el precio aprobado en la base de datos
                    nombre_norm = unidecode.unidecode(row['actividades'].lower().strip())
                    
                    # Buscar la actividad
                    self.db.cursor.execute(
                        "SELECT id FROM actividades WHERE nombre_normalizado = ?",
                        (nombre_norm,)
                    )
                    resultado = self.db.cursor.fetchone()
                    
                    if resultado:
                        actividad_id = resultado[0]
                    else:
                        # Crear nueva actividad
                        self.db.cursor.execute(
                            "INSERT INTO actividades (nombre, nombre_normalizado) VALUES (?, ?)",
                            (row['actividades'], nombre_norm)
                        )
                        actividad_id = self.db.cursor.lastrowid
                    
                    # Registrar el precio aprobado
                    self.db.cursor.execute(
                        """INSERT INTO precios 
                           (actividad_id, precio, fecha_actualizacion, fuente, notas) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (
                            actividad_id,
                            row['costo_unitario'],
                            datetime.now().strftime("%Y-%m-%d"),
                            "cotizacion_aprobada",
                            "Precio aprobado en cotización"
                        )
                    )
            
            self.db.conn.commit()
            logger.info(f"Aprendizaje completado: {len(cotizacion_df)} actividades procesadas")
            
        except Exception as e:
            logger.error(f"Error en aprendizaje de cotización: {str(e)}")
            self.db.conn.rollback()
            raise
