"""
Módulo para gestión segura de secretos y API keys.
"""
import os
import json
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)

class SecretManager:
    """
    Gestiona el almacenamiento seguro de secretos usando encriptación.
    """
    def __init__(self, storage_path: str = ".secrets"):
        """
        Inicializa el gestor de secretos.
        
        Args:
            storage_path: Ruta donde se almacenarán los secretos encriptados
        """
        self.storage_path = storage_path
        self._fernet = None
        self._secrets = {}
        self._initialize_encryption()
        self._load_secrets()
    
    def _initialize_encryption(self):
        """Inicializa el sistema de encriptación."""
        try:
            # Obtener o generar salt
            salt = os.environ.get("APP_SALT", None)
            if not salt:
                salt = base64.b64encode(os.urandom(16)).decode()
                logger.warning("APP_SALT no encontrado, generando uno nuevo")
            
            # Derivar clave de encriptación
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt.encode(),
                iterations=100000,
            )
            key = base64.b64encode(kdf.derive(salt.encode()))
            self._fernet = Fernet(key)
            
        except Exception as e:
            logger.error(f"Error inicializando encriptación: {str(e)}")
            raise
    
    def _load_secrets(self):
        """Carga los secretos desde el almacenamiento."""
        try:
            if not os.path.exists(self.storage_path):
                os.makedirs(self.storage_path)
                return
            
            secrets_file = os.path.join(self.storage_path, "secrets.enc")
            if os.path.exists(secrets_file):
                with open(secrets_file, "rb") as f:
                    encrypted_data = f.read()
                    decrypted_data = self._fernet.decrypt(encrypted_data)
                    self._secrets = json.loads(decrypted_data)
        
        except Exception as e:
            logger.error(f"Error cargando secretos: {str(e)}")
            self._secrets = {}
    
    def _save_secrets(self):
        """Guarda los secretos en el almacenamiento."""
        try:
            secrets_file = os.path.join(self.storage_path, "secrets.enc")
            encrypted_data = self._fernet.encrypt(json.dumps(self._secrets).encode())
            
            with open(secrets_file, "wb") as f:
                f.write(encrypted_data)
        
        except Exception as e:
            logger.error(f"Error guardando secretos: {str(e)}")
            raise
    
    def set_secret(self, key: str, value: str, expires_in: Optional[int] = None):
        """
        Almacena un secreto de forma segura.
        
        Args:
            key: Identificador del secreto
            value: Valor del secreto
            expires_in: Tiempo en segundos hasta que expire (opcional)
        """
        try:
            expiry = None
            if expires_in:
                expiry = (datetime.now() + timedelta(seconds=expires_in)).isoformat()
            
            self._secrets[key] = {
                "value": value,
                "expires": expiry
            }
            self._save_secrets()
        
        except Exception as e:
            logger.error(f"Error estableciendo secreto: {str(e)}")
            raise
    
    def get_secret(self, key: str) -> Optional[str]:
        """
        Recupera un secreto.
        
        Args:
            key: Identificador del secreto
        
        Returns:
            El valor del secreto o None si no existe o expiró
        """
        try:
            if key not in self._secrets:
                return None
            
            secret = self._secrets[key]
            if secret["expires"]:
                expiry = datetime.fromisoformat(secret["expires"])
                if datetime.now() > expiry:
                    del self._secrets[key]
                    self._save_secrets()
                    return None
            
            return secret["value"]
        
        except Exception as e:
            logger.error(f"Error obteniendo secreto: {str(e)}")
            return None
    
    def delete_secret(self, key: str):
        """
        Elimina un secreto.
        
        Args:
            key: Identificador del secreto
        """
        try:
            if key in self._secrets:
                del self._secrets[key]
                self._save_secrets()
        
        except Exception as e:
            logger.error(f"Error eliminando secreto: {str(e)}")
            raise


class APIKeyManager:
    """
    Gestiona las API keys de forma segura.
    """
    def __init__(self, secret_manager: SecretManager):
        """
        Inicializa el gestor de API keys.
        
        Args:
            secret_manager: Instancia de SecretManager para almacenamiento seguro
        """
        self.secret_manager = secret_manager
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Carga las API keys desde variables de entorno."""
        # OpenAI API Key
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self.secret_manager.set_secret("OPENAI_API_KEY", openai_key)
        
        # Dockling API Key
        dockling_key = os.environ.get("DOCKLING_API_KEY")
        if dockling_key:
            self.secret_manager.set_secret("DOCKLING_API_KEY", dockling_key)
    
    def set_api_key(self, service: str, key: str, expires_in: Optional[int] = None):
        """
        Establece una API key.
        
        Args:
            service: Nombre del servicio (e.g., 'openai', 'dockling')
            key: API key
            expires_in: Tiempo en segundos hasta que expire (opcional)
        """
        service_key = f"{service.upper()}_API_KEY"
        self.secret_manager.set_secret(service_key, key, expires_in)
    
    def get_openai_api_key(self) -> Optional[str]:
        """Obtiene la API key de OpenAI."""
        return self.secret_manager.get_secret("OPENAI_API_KEY")
    
    def get_dockling_api_key(self) -> Optional[str]:
        """Obtiene la API key de Dockling."""
        return self.secret_manager.get_secret("DOCKLING_API_KEY")
    
    def validate_api_key(self, service: str, key: str) -> bool:
        """
        Valida una API key.
        
        Args:
            service: Nombre del servicio
            key: API key a validar
        
        Returns:
            True si la key es válida, False en caso contrario
        """
        # TODO: Implementar validación real con los servicios
        return bool(key and len(key) > 10)
