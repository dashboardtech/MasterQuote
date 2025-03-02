# -*- coding: utf-8 -*-

"""
Utilidades para manejar la serialización JSON de tipos de datos especiales,
especialmente los que vienen de numpy y pandas.
"""

import json
import numpy as np
from datetime import datetime

def numpy_serializer(obj):
    """JSON serializer para objetos no serializables por defecto como numpy arrays"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    raise TypeError(f"Tipo no serializable: {type(obj)}")

def dump_json(obj, file_path, indent=4, ensure_ascii=False):
    """Guarda un objeto en formato JSON, manejando tipos de datos de numpy"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii, default=numpy_serializer)
        
def loads_json(json_str):
    """Carga un string JSON"""
    return json.loads(json_str)

def dumps_json(obj, indent=4, ensure_ascii=False):
    """Convierte un objeto a string JSON, manejando tipos de datos de numpy"""
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, default=numpy_serializer)
