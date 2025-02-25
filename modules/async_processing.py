"""
Módulo para procesamiento asíncrono y por lotes.
"""
import asyncio
import concurrent.futures
import os
from typing import List, Callable, Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Resultado de una operación de procesamiento."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime = datetime.now()


class AsyncProcessor:
    """
    Gestiona el procesamiento asíncrono de tareas.
    """
    def __init__(self, max_workers: int = None):
        """
        Inicializa el procesador asíncrono.
        
        Args:
            max_workers: Número máximo de workers para el ThreadPoolExecutor
        """
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self._tasks = {}
    
    async def process_async(self, task_id: str, func: Callable, *args, **kwargs) -> ProcessingResult:
        """
        Procesa una tarea de forma asíncrona.
        
        Args:
            task_id: Identificador único de la tarea
            func: Función a ejecutar
            *args: Argumentos posicionales para la función
            **kwargs: Argumentos nombrados para la función
        
        Returns:
            ProcessingResult con el resultado de la operación
        """
        start_time = datetime.now()
        
        try:
            # Ejecutar la función en el ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(self._executor, func, *args, **kwargs)
            self._tasks[task_id] = future
            
            # Esperar resultado
            result = await future
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=result,
                processing_time=processing_time
            )
        
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.exception(f"Error en tarea {task_id}")
            
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time=processing_time
            )
        
        finally:
            if task_id in self._tasks:
                del self._tasks[task_id]
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado de una tarea.
        
        Args:
            task_id: Identificador de la tarea
        
        Returns:
            Diccionario con el estado de la tarea
        """
        if task_id not in self._tasks:
            return {
                "status": "not_found",
                "message": "Tarea no encontrada"
            }
        
        future = self._tasks[task_id]
        if future.done():
            if future.exception():
                return {
                    "status": "error",
                    "message": str(future.exception())
                }
            return {
                "status": "completed",
                "message": "Tarea completada"
            }
        
        return {
            "status": "running",
            "message": "Tarea en proceso"
        }
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancela una tarea en ejecución.
        
        Args:
            task_id: Identificador de la tarea
        
        Returns:
            True si la tarea fue cancelada, False en caso contrario
        """
        if task_id in self._tasks:
            future = self._tasks[task_id]
            cancelled = future.cancel()
            if cancelled:
                del self._tasks[task_id]
            return cancelled
        return False


class BatchProcessor:
    """
    Gestiona el procesamiento por lotes de archivos o datos.
    """
    def __init__(self, 
                 batch_size: int = 5,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Inicializa el procesador por lotes.
        
        Args:
            batch_size: Tamaño del lote para procesamiento
            max_retries: Número máximo de reintentos por item
            retry_delay: Tiempo de espera entre reintentos (segundos)
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.async_processor = AsyncProcessor()
    
    async def process_batch(self, 
                          items: List[Any],
                          process_func: Callable[[Any], Any]) -> List[ProcessingResult]:
        """
        Procesa un lote de items.
        
        Args:
            items: Lista de items a procesar
            process_func: Función para procesar cada item
        
        Returns:
            Lista de ProcessingResult con los resultados
        """
        results = []
        
        # Procesar en lotes
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await asyncio.gather(*[
                self._process_with_retry(f"task_{i}_{j}", item, process_func)
                for j, item in enumerate(batch)
            ])
            results.extend(batch_results)
        
        return results
    
    async def _process_with_retry(self,
                                task_id: str,
                                item: Any,
                                process_func: Callable) -> ProcessingResult:
        """
        Procesa un item con reintentos.
        
        Args:
            task_id: Identificador de la tarea
            item: Item a procesar
            process_func: Función de procesamiento
        
        Returns:
            ProcessingResult con el resultado
        """
        for attempt in range(self.max_retries):
            result = await self.async_processor.process_async(
                task_id,
                process_func,
                item
            )
            
            if result.success:
                return result
            
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        return result  # Retornar último resultado fallido
    
    async def process_dataframe(self,
                              df: pd.DataFrame,
                              process_func: Callable[[pd.Series], Any]) -> pd.DataFrame:
        """
        Procesa un DataFrame por lotes.
        
        Args:
            df: DataFrame a procesar
            process_func: Función para procesar cada fila
        
        Returns:
            DataFrame con los resultados
        """
        # Convertir DataFrame a lista de Series
        rows = [row for _, row in df.iterrows()]
        
        # Procesar por lotes
        results = await self.process_batch(rows, process_func)
        
        # Construir DataFrame de resultados
        processed_data = []
        for result in results:
            if result.success and result.data is not None:
                processed_data.append(result.data)
        
        return pd.DataFrame(processed_data)
