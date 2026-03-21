import json
import os
from typing import Any, Dict, List

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")

# Traducción de errores ya detectados por tu modelo
DETECTED_ERROR_MAP = {
    "codo_separado_del_torso": "codo separado del torso durante la rotación",
    "inclinacion_excesiva_tronco": "compensación con inclinación excesiva del tronco",
    "elevacion_escapular_excesiva": "elevación escapular excesiva durante el movimiento",
    "descenso_de_muneca_en_rotacion": "descenso de la muñeca durante la rotación",
    "asimetria_derecha_izquierda": "asimetría entre lado derecho e izquierdo",
    "velocidad_angular_irregular": "velocidad angular irregular o movimiento brusco",
}

SYSTEM_PROMPT = """
Eres un asistente de rehabilitación física.

Tu tarea es transformar resultados biomecánicos estructurados en retroalimentación breve, clara y segura para el usuario.

Reglas:
- Usa solo la información proporcionada.
- No inventes errores, métricas o diagnósticos.
- No menciones lesiones o enfermedades.
- No agregues recomendaciones médicas.
- Prioriza el error más importante.
- Usa español claro y natural.
- Si la ejecución es incorrecta, indica primero la corrección más importante.
- Responde únicamente en el formato solicitado.
"""

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "priority_correction": {"type": "string"},
        "feedback_short": {"type": "string"},
        "feedback_detailed": {"type": "string"}
    },
    "required": [
        "priority_correction",
        "feedback_short",
        "feedback_detailed"
    ]
}


def normalize_classification(classification: str) -> str:
    value = (classification or "").strip().lower()
    if value in {"incorrecta", "incorrecto", "incorrect"}:
        return "incorrect"
    if value in {"correcta", "correcto", "correct"}:
        return "correct"
    return value or "unknown"


def map_detected_errors(errors_detected: List[str]) -> List[Dict[str, Any]]:
    enriched_errors = []

    for idx, error_code in enumerate(errors_detected):
        enriched_errors.append({
            "code": error_code,
            "description": DETECTED_ERROR_MAP.get(error_code, error_code.replace("_", " ")),
            "priority_rank": idx + 1
        })

    return enriched_errors


def prepare_diagnosis(raw_prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapta la salida real de tu modelo al formato que consumirá Ollama.

    Formato esperado:
    {
        'exercise': 'Standing Shoulder Internal/External Rotation',
        'movement_id': 7,
        'label': 1,
        'classification': 'incorrecta',
        'probability_incorrect': 1.0,
        'threshold': 0.8317,
        'severity': 'severa',
        'errors_detected': [
            'codo_separado_del_torso',
            'inclinacion_excesiva_tronco'
        ],
        'biomechanical_summary': {...}
    }
    """
    classification = normalize_classification(raw_prediction.get("classification", ""))
    errors_detected = raw_prediction.get("errors_detected", [])
    enriched_errors = map_detected_errors(errors_detected)

    diagnosis = {
        "exercise": raw_prediction.get("exercise", "unknown_exercise"),
        "movement_id": raw_prediction.get("movement_id"),
        "overall_status": classification,
        "severity": raw_prediction.get("severity", "desconocida"),
        "confidence": raw_prediction.get("probability_incorrect", 0.0),
        "threshold": raw_prediction.get("threshold"),
        "primary_error": enriched_errors[0] if enriched_errors else None,
        "errors": enriched_errors,
        "metrics": raw_prediction.get("biomechanical_summary", {})
    }

    return diagnosis


def build_user_prompt(diagnosis: Dict[str, Any]) -> str:
    return f"""
Genera retroalimentación para este resultado biomecánico:

{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

Tarea:
- Indica la corrección prioritaria.
- Da una retroalimentación corta.
- Da una explicación breve para el paciente.

Restricciones:
- No inventes errores nuevos.
- No agregues recomendaciones médicas.
- Basa la respuesta solo en "overall_status", "severity", "errors" y "metrics".
- Sé claro, amable y breve.
"""


def ask_ollama_from_diagnosis(diagnosis: Dict[str, Any]) -> Dict[str, str]:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(diagnosis)}
        ],
        "format": FEEDBACK_SCHEMA,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=90)
    response.raise_for_status()

    data = response.json()
    content = data["message"]["content"]
    return json.loads(content)


def ask_ollama(raw_prediction: Dict[str, Any]) -> Dict[str, str]:
    diagnosis = prepare_diagnosis(raw_prediction)
    return ask_ollama_from_diagnosis(diagnosis)
