import pandas as pd
import logging
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import BaseExtractor

logger = logging.getLogger(__name__)

class AIAssistedExtractor(BaseExtractor):
    """Extractor asistido por IA para formatos complejos o no estructurados."""
    
    def __init__(self, api_key: str):
        """
        Inicializa el extractor asistido por IA.
        
        Args:
            api_key: API key para el servicio de IA (OpenAI)
        """
        self.api_key = api_key
        self._init_openai()
    
    def _init_openai(self):
        """Inicializa el cliente de OpenAI."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Error inicializando OpenAI: {str(e)}")
            raise
    
    def extract(self, file_path: str, interactive: bool = False, text: Optional[str] = None) -> pd.DataFrame:
        """
        Extrae datos usando asistencia de IA.
        
        Args:
            file_path: Ruta al archivo
            interactive: Si debe ser interactivo
            text: Texto ya extraído (opcional)
            
        Returns:
            DataFrame con actividades y precios
        """
        if not self.api_key:
            raise ValueError("Se requiere API key para asistencia con IA")
        
        try:
            # Si no se proporciona texto, extraerlo del archivo
            if text is None:
                text = self._extract_text(file_path)
            
            if not text:
                logger.error(f"No se pudo extraer texto del archivo: {file_path}")
                return pd.DataFrame()
            
            # Usar IA para estructurar los datos
            structured_data = self._process_with_ai(text)
            
            if structured_data:
                return pd.DataFrame(structured_data)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.exception(f"Error en extracción asistida por IA: {str(e)}")
            return pd.DataFrame()
    
    def _extract_text(self, file_path: str) -> Optional[str]:
        """
        Extrae texto del archivo según su tipo.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            str con texto extraído o None si falla
        """
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext in ['.docx', '.doc']:
                return self._extract_from_word(file_path)
            elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                return self._extract_from_image(file_path)
            elif file_ext == '.txt':
                return self._extract_from_text(file_path)
            else:
                raise ValueError(f"Formato no soportado: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error extrayendo texto de {file_path}: {str(e)}")
            return None
    
    def _extract_from_word(self, file_path: str) -> str:
        """
        Extrae texto de un documento Word.
        
        Args:
            file_path: Ruta al archivo Word
            
        Returns:
            str con texto extraído
        """
        try:
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            logger.error(f"Error extrayendo texto de Word: {str(e)}")
            return ''
    
    def _extract_from_image(self, file_path: str) -> str:
        """
        Extrae texto de una imagen usando OCR.
        
        Args:
            file_path: Ruta a la imagen
            
        Returns:
            str con texto extraído
        """
        try:
            import pytesseract
            from PIL import Image
            
            # Abrir y preprocesar imagen
            image = Image.open(file_path)
            image = self._preprocess_image(image)
            
            # Extraer texto
            text = pytesseract.image_to_string(image, lang='spa')
            return text
            
        except Exception as e:
            logger.error(f"Error en OCR: {str(e)}")
            return ''
    
    def _extract_from_text(self, file_path: str) -> str:
        """
        Lee texto de un archivo de texto plano.
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            str con texto extraído
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Intentar con otra codificación
            try:
                with open(file_path, 'r', encoding='latin1') as f:
                    return f.read()
            except:
                return ''
    
    def _preprocess_image(self, image):
        """
        Preprocesa una imagen para mejorar resultados del OCR.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            Imagen procesada
        """
        try:
            # Convertir a escala de grises
            image = image.convert('L')
            
            # Aumentar contraste
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Aumentar nitidez
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            return image
        except:
            return image
    
    def _process_with_ai(self, text: str) -> List[Dict[str, Any]]:
        """
        Procesa el texto usando IA para extraer información estructurada.
        
        Args:
            text: Texto a procesar
            
        Returns:
            Lista de diccionarios con datos estructurados
        """
        try:
            # Prompt para extraer información
            prompt = f"""
            Analiza el siguiente texto y extrae información sobre actividades y precios.
            Estructura la información en un formato JSON con los siguientes campos:
            - actividades: descripción de la actividad o material
            - costo_unitario: precio por unidad
            - cantidad: cantidad requerida
            - costo_total: costo total

            Si algún campo no está disponible, déjalo como null.
            Texto a analizar:
            {text}
            """
            
            # Llamar a la API de OpenAI
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en extraer información estructurada de textos sobre cotizaciones y precios."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            # Procesar respuesta
            try:
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Asegurar que es una lista de diccionarios
                if isinstance(data, dict):
                    data = [data]
                
                return data
                
            except json.JSONDecodeError:
                logger.error("Error decodificando respuesta de IA")
                return []
                
        except Exception as e:
            logger.exception(f"Error procesando con IA: {str(e)}")
            return []
