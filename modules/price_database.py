import sqlite3
import pandas as pd
from datetime import datetime
import unidecode
import logging
from typing import List, Dict, Any, Optional, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PriceDatabase:
    def __init__(self, db_path: str = "price_history.db"):
        """Inicializa la conexión a la base de datos.
        
        Args:
            db_path (str): Ruta al archivo de la base de datos SQLite
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.initialize_db()
        logger.info(f"Base de datos inicializada en: {db_path}")
        
    def initialize_db(self):
        """Crea las tablas necesarias si no existen."""
        try:
            # Tabla de categorías de actividades
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS categorias (
                id INTEGER PRIMARY KEY,
                nombre TEXT UNIQUE,
                descripcion TEXT
            )
            ''')
            
            # Tabla de actividades
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS actividades (
                id INTEGER PRIMARY KEY,
                nombre TEXT,
                nombre_normalizado TEXT,
                categoria_id INTEGER,
                unidad_medida TEXT,
                descripcion TEXT,
                FOREIGN KEY (categoria_id) REFERENCES categorias (id),
                UNIQUE (nombre_normalizado)
            )
            ''')
            
            # Tabla de historial de precios
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS precios (
                id INTEGER PRIMARY KEY,
                actividad_id INTEGER,
                precio REAL,
                fecha_actualizacion TEXT,
                fuente TEXT,
                region TEXT,
                notas TEXT,
                FOREIGN KEY (actividad_id) REFERENCES actividades (id)
            )
            ''')
            
            # Tabla de relaciones entre actividades similares
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS actividades_similares (
                actividad1_id INTEGER,
                actividad2_id INTEGER,
                similitud REAL,
                PRIMARY KEY (actividad1_id, actividad2_id),
                FOREIGN KEY (actividad1_id) REFERENCES actividades (id),
                FOREIGN KEY (actividad2_id) REFERENCES actividades (id)
            )
            ''')
            
            # Tabla para cotizaciones completas
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cotizaciones (
                id INTEGER PRIMARY KEY,
                nombre_proyecto TEXT,
                fecha TEXT,
                cliente TEXT,
                total REAL,
                notas TEXT
            )
            ''')
            
            # Tabla para detalles de cotizaciones
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cotizacion_detalles (
                id INTEGER PRIMARY KEY,
                cotizacion_id INTEGER,
                actividad_id INTEGER,
                cantidad REAL,
                precio_unitario REAL,
                precio_total REAL,
                FOREIGN KEY (cotizacion_id) REFERENCES cotizaciones (id),
                FOREIGN KEY (actividad_id) REFERENCES actividades (id)
            )
            ''')
            
            self.conn.commit()
            logger.info("Tablas creadas/verificadas exitosamente")
            
        except sqlite3.Error as e:
            logger.error(f"Error al inicializar la base de datos: {str(e)}")
            raise
    
    def importar_excel_a_db(self, excel_path: str, mapping: Optional[Dict[str, str]] = None) -> None:
        """Importa datos de un Excel procesado a la base de datos.
        
        Args:
            excel_path (str): Ruta al archivo Excel con precios actualizados
            mapping (Dict[str, str], optional): Mapeo de columnas Excel a columnas BD
        """
        try:
            df = pd.read_excel(excel_path)
            fecha_actual = datetime.now().strftime("%Y-%m-%d")
            
            # Mapeo predeterminado si no se proporciona
            if mapping is None:
                mapping = {
                    'actividades': 'nombre',
                    'costo_unitario': 'precio'
                }
            
            for index, row in df.iterrows():
                nombre = row[mapping['actividades']]
                nombre_norm = unidecode.unidecode(nombre.lower().strip())
                precio = row[mapping['costo_unitario']]
                
                # Verificar si la actividad ya existe
                self.cursor.execute(
                    "SELECT id FROM actividades WHERE nombre_normalizado = ?", 
                    (nombre_norm,)
                )
                resultado = self.cursor.fetchone()
                
                if resultado:
                    actividad_id = resultado[0]
                else:
                    # Insertar nueva actividad
                    self.cursor.execute(
                        "INSERT INTO actividades (nombre, nombre_normalizado, unidad_medida) VALUES (?, ?, ?)",
                        (nombre, nombre_norm, "unidad")
                    )
                    actividad_id = self.cursor.lastrowid
                
                # Registrar el precio
                self.cursor.execute(
                    "INSERT INTO precios (actividad_id, precio, fecha_actualizacion, fuente) VALUES (?, ?, ?, ?)",
                    (actividad_id, precio, fecha_actual, excel_path)
                )
            
            self.conn.commit()
            logger.info(f"Datos importados exitosamente desde: {excel_path}")
            
        except Exception as e:
            logger.error(f"Error al importar Excel: {str(e)}")
            self.conn.rollback()
            raise
    
    def obtener_precio_sugerido(self, actividad: str) -> Dict[str, Any]:
        """Obtiene el precio sugerido para una actividad basado en datos históricos.
        
        Args:
            actividad (str): Nombre de la actividad
            
        Returns:
            Dict[str, Any]: Información de precio y confianza
        """
        try:
            nombre_norm = unidecode.unidecode(actividad.lower().strip())
            
            # Buscar coincidencia exacta
            self.cursor.execute('''
                SELECT a.id, a.nombre, p.precio, p.fecha_actualizacion 
                FROM actividades a
                JOIN precios p ON a.id = p.actividad_id
                WHERE a.nombre_normalizado = ?
                ORDER BY p.fecha_actualizacion DESC
                LIMIT 1
            ''', (nombre_norm,))
            
            resultado = self.cursor.fetchone()
            if resultado:
                return {
                    'actividad_id': resultado[0],
                    'nombre': resultado[1],
                    'precio': resultado[2],
                    'ultima_actualizacion': resultado[3],
                    'confianza': 'alta',
                    'tipo_coincidencia': 'exacta'
                }
            
            # Buscar coincidencia por similitud
            self.cursor.execute('''
                SELECT a.id, a.nombre, p.precio, p.fecha_actualizacion, 
                       as.similitud, a2.nombre as nombre_similar
                FROM actividades a
                JOIN actividades_similares as ON a.id = as.actividad2_id
                JOIN actividades a2 ON as.actividad1_id = a2.id
                JOIN precios p ON a2.id = p.actividad_id
                WHERE a.nombre_normalizado LIKE ?
                ORDER BY as.similitud DESC, p.fecha_actualizacion DESC
                LIMIT 1
            ''', (f'%{nombre_norm}%',))
            
            resultado = self.cursor.fetchone()
            if resultado:
                return {
                    'actividad_id': resultado[0],
                    'nombre': resultado[1],
                    'precio': resultado[2],
                    'ultima_actualizacion': resultado[3],
                    'confianza': 'media' if resultado[4] > 0.7 else 'baja',
                    'tipo_coincidencia': 'similar',
                    'actividad_similar': resultado[5],
                    'similitud': resultado[4]
                }
            
            return {
                'precio': None,
                'confianza': 'nula',
                'tipo_coincidencia': 'ninguna'
            }
            
        except sqlite3.Error as e:
            logger.error(f"Error al obtener precio sugerido: {str(e)}")
            raise
    
    def guardar_cotizacion(self, 
                          nombre_proyecto: str, 
                          cliente: str, 
                          items: List[Dict[str, Any]], 
                          notas: str = "") -> int:
        """Guarda una cotización completa en la base de datos.
        
        Args:
            nombre_proyecto (str): Nombre del proyecto
            cliente (str): Nombre del cliente
            items (List[Dict]): Lista de diccionarios con actividad, cantidad y precio
            notas (str): Notas adicionales
            
        Returns:
            int: ID de la cotización creada
        """
        try:
            fecha_actual = datetime.now().strftime("%Y-%m-%d")
            total = sum(item['cantidad'] * item['precio_unitario'] for item in items)
            
            self.cursor.execute(
                "INSERT INTO cotizaciones (nombre_proyecto, fecha, cliente, total, notas) VALUES (?, ?, ?, ?, ?)",
                (nombre_proyecto, fecha_actual, cliente, total, notas)
            )
            cotizacion_id = self.cursor.lastrowid
            
            for item in items:
                actividad_nombre = item['actividad']
                nombre_norm = unidecode.unidecode(actividad_nombre.lower().strip())
                
                # Buscar o crear la actividad
                self.cursor.execute(
                    "SELECT id FROM actividades WHERE nombre_normalizado = ?", 
                    (nombre_norm,)
                )
                resultado = self.cursor.fetchone()
                
                if resultado:
                    actividad_id = resultado[0]
                else:
                    self.cursor.execute(
                        "INSERT INTO actividades (nombre, nombre_normalizado) VALUES (?, ?)",
                        (actividad_nombre, nombre_norm)
                    )
                    actividad_id = self.cursor.lastrowid
                
                # Guardar detalle de cotización
                self.cursor.execute(
                    """INSERT INTO cotizacion_detalles 
                       (cotizacion_id, actividad_id, cantidad, precio_unitario, precio_total) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        cotizacion_id, 
                        actividad_id, 
                        item['cantidad'], 
                        item['precio_unitario'], 
                        item['cantidad'] * item['precio_unitario']
                    )
                )
            
            self.conn.commit()
            logger.info(f"Cotización guardada exitosamente. ID: {cotizacion_id}")
            return cotizacion_id
            
        except Exception as e:
            logger.error(f"Error al guardar cotización: {str(e)}")
            self.conn.rollback()
            raise
    
    def obtener_historial_precios(self, 
                                categoria: Optional[str] = None, 
                                busqueda: Optional[str] = None) -> pd.DataFrame:
        """Obtiene el historial de precios con opciones de filtrado.
        
        Args:
            categoria (str, optional): Filtrar por categoría
            busqueda (str, optional): Texto para buscar en nombres de actividades
            
        Returns:
            pd.DataFrame: DataFrame con el historial de precios
        """
        try:
            query = """
                SELECT 
                    a.nombre as actividad,
                    c.nombre as categoria,
                    p.precio,
                    p.fecha_actualizacion,
                    p.fuente,
                    p.region
                FROM actividades a
                LEFT JOIN categorias c ON a.categoria_id = c.id
                JOIN precios p ON a.id = p.actividad_id
                WHERE 1=1
            """
            params = []
            
            if categoria and categoria.lower() != "todas":
                query += " AND c.nombre = ?"
                params.append(categoria)
            
            if busqueda:
                query += " AND a.nombre_normalizado LIKE ?"
                params.append(f"%{unidecode.unidecode(busqueda.lower().strip())}%")
            
            query += " ORDER BY p.fecha_actualizacion DESC"
            
            return pd.read_sql_query(query, self.conn, params=params)
            
        except Exception as e:
            logger.error(f"Error al obtener historial de precios: {str(e)}")
            raise
    
    def cerrar(self):
        """Cierra la conexión a la base de datos."""
        try:
            self.conn.close()
            logger.info("Conexión a la base de datos cerrada")
        except Exception as e:
            logger.error(f"Error al cerrar la base de datos: {str(e)}")
            raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cerrar()
