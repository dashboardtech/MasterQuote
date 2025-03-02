"""
Configuración global para MasterQuote.
"""
import os
from typing import Dict, Any, Optional, List

# Configuración de API Keys
# Para uso en desarrollo, puedes definir las claves aquí
# Para producción, usa variables de entorno
API_KEYS = {
    # Clave para OpenAI (GPT)
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
    
    # Clave para Dockling - Sistema interno
    # Para sistemas internos, puedes usar una clave fija o un token de servicio
    "DOCKLING_API_KEY": os.environ.get("DOCKLING_API_KEY", "dockling_internal_service_token"),
}

# Configuración específica para el sector de construcción
CONSTRUCTION_CONFIG = {
    # Categorías de productos/servicios de construcción
    "categories": [
        "Materiales de construcción",
        "Mano de obra",
        "Equipos y maquinaria",
        "Acabados",
        "Instalaciones eléctricas",
        "Instalaciones sanitarias",
        "Estructuras",
        "Cimentaciones",
        "Cubiertas",
        "Carpintería",
        "Pintura",
        "Vidrios y aluminio",
        "Pisos y revestimientos",
        "Demolición",
        "Excavaciones",
        "Impermeabilización",
        "Andamios y apuntalamiento",
        "Prefabricados",
        "Urbanización",
        "Decoración"
    ],
    
    # Unidades de medida comunes en construcción
    "units": [
        "m²", "m³", "ml", "kg", "ton", "unidad", "global", "hora", "día", "semana",
        "mes", "pie²", "pie³", "galón", "litro", "bolsa", "rollo", "plancha", "juego",
        "pieza", "caja", "viaje", "jornada", "lote", "obra", "servicio", "punto",
        "equipo", "cuadrilla", "quintal", "paquete", "par", "bobina", "millar"
    ],
    
    # Términos específicos para identificar columnas en cotizaciones de construcción
    "column_keywords": {
        "description": [
            "descripción", "concepto", "partida", "actividad", "trabajo", "ítem", 
            "detalle", "especificación", "material", "insumo", "elemento"
        ],
        "unit": [
            "unidad", "ud", "u.m.", "u/m", "medida", "un"
        ],
        "quantity": [
            "cantidad", "cant", "vol", "volumen", "área", "area", "longitud", "peso", "qt"
        ],
        "unit_price": [
            "precio unitario", "p.u.", "valor unitario", "costo unitario", "precio/u", 
            "$/u", "precio/unidad", "tarifa"
        ],
        "total_price": [
            "importe", "total", "subtotal", "valor total", "costo total", "monto"
        ]
    },
    
    # Plantillas predefinidas para diferentes tipos de proyectos de construcción
    "project_templates": {
        "vivienda": [
            "Preliminares", "Cimentación", "Estructura", "Mampostería", "Instalaciones Eléctricas",
            "Instalaciones Hidráulicas", "Acabados", "Carpintería", "Pintura", "Limpieza"
        ],
        "edificio_comercial": [
            "Preliminares", "Cimentación", "Estructura", "Fachada", "Instalaciones Eléctricas",
            "Instalaciones Hidráulicas", "Instalaciones Especiales", "Acabados", "Equipamiento", 
            "Seguridad", "Limpieza"
        ],
        "obra_civil": [
            "Preliminares", "Movimiento de tierras", "Estructuras", "Drenaje",
            "Pavimentos", "Señalización", "Obras complementarias"
        ],
        "remodelacion": [
            "Demolición", "Desmantelamiento", "Estructura", "Instalaciones", 
            "Acabados", "Pintura", "Limpieza"
        ],
        "instalaciones_industriales": [
            "Preliminares", "Cimentación", "Estructura Metálica", "Cerramientos", 
            "Instalación Eléctrica Industrial", "Instalación Hidráulica", "Equipamiento Industrial",
            "Sistemas de Seguridad", "Acabados Industriales"
        ]
    },
    
    # Factores de ajuste por región
    "regional_factors": {
        "Norte": 1.05,
        "Centro": 1.0,
        "Sur": 0.95,
        "Metropolitana": 1.15,
        "Costa": 1.08,
        "Montaña": 1.12
    },
    
    # Precios de referencia para materiales comunes (en unidades locales)
    "reference_prices": {
        "Cemento (bolsa 50kg)": 120.00,
        "Arena (m³)": 350.00,
        "Grava (m³)": 380.00,
        "Varilla 3/8\" (ton)": 22000.00,
        "Ladrillo (millar)": 2300.00,
        "Bloque 15x20x40 (pieza)": 12.50,
        "Cable eléctrico THW cal.12 (ml)": 9.80,
        "Tubo PVC sanitario 4\" (tramo 6m)": 210.00,
        "Pintura vinílica (cubeta 19l)": 850.00
    },
    
    # Costos hora-persona de mano de obra por especialidad
    "labor_prices": {
        "Peón": 80.00,
        "Albañil": 120.00,
        "Oficial": 150.00,
        "Electricista": 180.00,
        "Plomero": 180.00,
        "Carpintero": 160.00,
        "Herrero": 170.00,
        "Maestro de obra": 250.00,
        "Ingeniero residente": 350.00
    },
    
    # Tiempos estimados por unidad para actividades comunes
    "estimated_times": {
        "Excavación manual (m³)": 1.5,  # horas-hombre por m³
        "Armado de acero (ton)": 8.0,   # horas-hombre por tonelada
        "Colado de concreto (m³)": 2.0,  # horas-hombre por m³
        "Levantamiento de muro (m²)": 1.0  # horas-hombre por m²
    },
    
    # Valores predeterminados para cálculos
    "defaults": {
        "overhead_percentage": 15.0,  # Porcentaje de indirectos
        "profit_percentage": 20.0,    # Porcentaje de utilidad
        "tax_rate": 16.0,             # Tasa de impuestos
        "contingency": 5.0            # Contingencia
    }
}

# Configuración de la interfaz de usuario
UI_CONFIG = {
    "theme": {
        "primaryColor": "#4D8AF0",
        "backgroundColor": "#121212",
        "secondaryBackgroundColor": "#1E1E1E",
        "textColor": "#E0E0E0",
        "font": "sans serif"
    },
    "page_title": "MasterQuote - Cotizador para Construcción",
    "page_icon": "🏗️",
    "menu_items": {
        "Get Help": "https://www.example.com/help",
        "Report a bug": "https://www.example.com/bug",
        "About": "# MasterQuote\nSistema inteligente de cotización para proyectos de construcción."
    }
}

# Configuración del sistema de caché
CACHE_CONFIG = {
    "enabled": True,
    "expiration_time": 30 * 24 * 60 * 60,  # 30 días en segundos
    "max_size_mb": 500,
    "storage_type": "disk",
    "cache_dir": ".cache"
}

# Configuración del extractor
EXTRACTOR_CONFIG = {
    "num_extractors": 3,
    "use_parallel": True,
    "default_min_confidence": 0.6,
    "validation_enabled_by_default": True
}
