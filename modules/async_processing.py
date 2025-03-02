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
import multiprocessing
import time

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
        self._tasks[task_id] = {"status": "running", "start_time": start_time}
        
        try:
            # Ejecutar la función en un thread separado
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: func(*args, **kwargs)
            )
            
            # Calcular tiempo de procesamiento
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Actualizar estado de la tarea
            self._tasks[task_id] = {
                "status": "completed",
                "start_time": start_time,
                "end_time": end_time,
                "processing_time": processing_time
            }
            
            return ProcessingResult(
                success=True,
                data=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Manejar error
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            error_msg = str(e)
            
            logger.error(f"Error en tarea {task_id}: {error_msg}")
            
            # Actualizar estado de la tarea
            self._tasks[task_id] = {
                "status": "failed",
                "start_time": start_time,
                "end_time": end_time,
                "processing_time": processing_time,
                "error": error_msg
            }
            
            return ProcessingResult(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado de una tarea.
        
        Args:
            task_id: Identificador de la tarea
            
        Returns:
            Diccionario con el estado de la tarea
        """
        return self._tasks.get(task_id, {"status": "not_found"})
    
    def process_batch_parallel(self, files: List[str], process_func: Callable, max_workers: Optional[int] = None) -> List[ProcessingResult]:
        """
        Procesa múltiples archivos en paralelo usando multiprocessing para un rendimiento óptimo.
        
        Args:
            files: Lista de rutas de archivos a procesar
            process_func: Función para procesar cada archivo
            max_workers: Número máximo de procesos (default: número de CPUs)
            
        Returns:
            Lista de ProcessingResult con los resultados
        """
        max_workers = max_workers or min(multiprocessing.cpu_count(), len(files))
        results = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Preparar las tareas
            future_to_file = {executor.submit(self._process_single_file, process_func, file): file for file in files}
            
            # Recoger resultados a medida que se completan
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Procesado correctamente: {file}")
                except Exception as e:
                    logger.error(f"Error procesando {file}: {str(e)}")
                    results.append(ProcessingResult(
                        success=False,
                        error=str(e),
                        timestamp=datetime.now()
                    ))
        
        return results
    
    def _process_single_file(self, process_func: Callable, file_path: str) -> ProcessingResult:
        """
        Procesa un solo archivo y maneja errores.
        
        Args:
            process_func: Función para procesar el archivo
            file_path: Ruta al archivo
            
        Returns:
            ProcessingResult con el resultado
        """
        start_time = datetime.now()
        
        try:
            result = process_func(file_path)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=result,
                processing_time=processing_time,
                timestamp=end_time
            )
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            error_msg = str(e)
            
            logger.error(f"Error procesando archivo {file_path}: {error_msg}")
            
            return ProcessingResult(
                success=False,
                error=error_msg,
                processing_time=processing_time,
                timestamp=end_time
            )


class BatchProcessor:
    """
    Gestiona el procesamiento por lotes de archivos o datos.
    """
    def __init__(self, 
                 batch_size: int = 5,
                 max_workers: int = None,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        Inicializa el procesador por lotes.
        
        Args:
            batch_size: Tamaño del lote para procesamiento
            max_workers: Número máximo de workers para procesamiento paralelo
            max_retries: Número máximo de reintentos por item
            retry_delay: Tiempo de espera entre reintentos (segundos)
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.async_processor = AsyncProcessor(max_workers=max_workers)
        
    def process_batch(self, 
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
        # Dividir en lotes según batch_size
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        results = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Procesando lote {i+1}/{len(batches)} ({len(batch)} items)")
            
            # Para archivos, usar process_batch_parallel del AsyncProcessor
            if all(isinstance(item, str) and os.path.isfile(item) for item in batch):
                batch_results = self.async_processor.process_batch_parallel(
                    batch, 
                    process_func, 
                    max_workers=self.max_workers
                )
                results.extend(batch_results)
            else:
                # Para otros tipos de datos, procesar secuencialmente con reintentos
                for item in batch:
                    task_id = f"task_{i}_{len(results)}"
                    result = self._process_with_retry(task_id, item, process_func)
                    results.append(result)
        
        return results
                
    def _process_with_retry(self,
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
        for attempt in range(1, self.max_retries + 1):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self.async_processor.process_async(task_id, process_func, item)
                )
                loop.close()
                
                if result.success:
                    return result
                
                logger.warning(f"Intento {attempt}/{self.max_retries} falló: {result.error}")
                
                # Si no es el último intento, esperar antes de reintentar
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Error en intento {attempt}: {str(e)}")
                
                # Si no es el último intento, esperar antes de reintentar
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
        
        # Si llegamos aquí, todos los intentos fallaron
        return ProcessingResult(
            success=False,
            error=f"Fallaron todos los intentos (máximo: {self.max_retries})"
        )
        
    def process_directory(self, directory_path: str, process_func: Callable, file_pattern: str = "*.*") -> List[ProcessingResult]:
        """
        Procesa todos los archivos en un directorio que coincidan con el patrón.
        
        Args:
            directory_path: Ruta al directorio
            process_func: Función para procesar cada archivo
            file_pattern: Patrón de archivos (glob)
            
        Returns:
            Lista de ProcessingResult con los resultados
        """
        import glob
        
        # Encontrar todos los archivos que coincidan con el patrón
        files = glob.glob(os.path.join(directory_path, file_pattern))
        
        if not files:
            logger.warning(f"No se encontraron archivos con el patrón {file_pattern} en {directory_path}")
            return []
        
        logger.info(f"Procesando {len(files)} archivos de {directory_path}")
        return self.process_batch(files, process_func)
    
    def process_dataframe(self,
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
        results = []
        total_rows = len(df)
        
        # Dividir en lotes
        for i in range(0, total_rows, self.batch_size):
            end_idx = min(i + self.batch_size, total_rows)
            batch = df.iloc[i:end_idx]
            
            logger.info(f"Procesando lote de filas {i+1}-{end_idx} de {total_rows}")
            
            # Procesar cada fila del lote
            batch_results = []
            for _, row in batch.iterrows():
                task_id = f"row_{i}_{len(batch_results)}"
                result = self._process_with_retry(task_id, row, process_func)
                
                if result.success:
                    batch_results.append(result.data)
                else:
                    # Si falla, añadir None o un valor por defecto
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        # Crear un nuevo DataFrame con los resultados
        return pd.DataFrame(results)
