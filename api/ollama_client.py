from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "priority_correction": {"type": "string"},
        "feedback_short": {"type": "string"},
        "feedback_detailed": {"type": "string"},
    },
    "required": [
        "priority_correction",
        "feedback_short",
        "feedback_detailed",
    ],
}

SYSTEM_PROMPT = """
Eres un asistente de rehabilitacion fisica.

Tu tarea es transformar resultados biomecanicos estructurados en retroalimentacion breve, clara y segura para el usuario.

Reglas:
- Usa solo la informacion proporcionada.
- No inventes errores, metricas o diagnosticos.
- No menciones lesiones o enfermedades.
- No agregues recomendaciones medicas.
- Prioriza el error mas importante.
- Usa espanol claro y natural.
- Si la ejecucion es incorrecta, indica primero la correccion mas importante.
- Responde unicamente en el formato solicitado.
"""

EXERCISE_CONFIG: dict[int, dict[str, Any]] = {
    7: {
        "model_env": "OLLAMA_MODEL_NAME_M07",
        "default_model_name": DEFAULT_MODEL_NAME,
        "error_map": {
            "codo_separado_del_torso": "codo separado del torso durante la rotacion",
            "inclinacion_excesiva_tronco": "compensacion con inclinacion excesiva del tronco",
            "elevacion_escapular_excesiva": "elevacion escapular excesiva durante el movimiento",
            "descenso_de_muneca_en_rotacion": "descenso de la muneca durante la rotacion",
            "asimetria_derecha_izquierda": "asimetria entre lado derecho e izquierdo",
            "velocidad_angular_irregular": "velocidad angular irregular o movimiento brusco",
        },
    },
    9: {
        "model_env": "OLLAMA_MODEL_NAME_M09",
        "default_model_name": DEFAULT_MODEL_NAME,
        "error_map": {
            "inclinacion_excesiva_tronco": "compensacion con inclinacion excesiva del tronco",
            "descenso_de_codo_o_muneca": "descenso del codo o de la muneca durante el movimiento",
            "elevacion_compensatoria_hombro": "elevacion compensatoria del hombro",
            "rango_de_movimiento_insuficiente": "rango de movimiento insuficiente",
            "asimetria_en_el_movimiento": "asimetria en el movimiento",
            "falta_de_control_del_movimiento": "falta de control del movimiento",
        },
    },
}


def _get_exercise_config(exercise_id: int) -> dict[str, Any]:
    config = EXERCISE_CONFIG.get(exercise_id)
    if config is None:
        raise ValueError(f"Ejercicio no soportado para Ollama: {exercise_id}")
    return config


def _get_model_name(exercise_id: int) -> str:
    config = _get_exercise_config(exercise_id)
    model_env = config["model_env"]
    return os.getenv(model_env, config["default_model_name"])


def normalize_classification(classification: str) -> str:
    value = (classification or "").strip().lower()
    if value in {"incorrecta", "incorrecto", "incorrect"}:
        return "incorrect"
    if value in {"correcta", "correcto", "correct"}:
        return "correct"
    return value or "unknown"


def map_detected_errors(errors_detected: List[str], exercise_id: int) -> List[Dict[str, Any]]:
    error_map = _get_exercise_config(exercise_id)["error_map"]
    enriched_errors = []

    for idx, error_code in enumerate(errors_detected):
        enriched_errors.append(
            {
                "code": error_code,
                "description": error_map.get(error_code, error_code.replace("_", " ")),
                "priority_rank": idx + 1,
            }
        )

    return enriched_errors


def prepare_diagnosis(raw_prediction: Dict[str, Any], exercise_id: int) -> Dict[str, Any]:
    classification = normalize_classification(raw_prediction.get("classification", ""))
    errors_detected = raw_prediction.get("errors_detected", [])
    enriched_errors = map_detected_errors(errors_detected, exercise_id)

    return {
        "exercise": raw_prediction.get("exercise", "unknown_exercise"),
        "movement_id": raw_prediction.get("movement_id", exercise_id),
        "overall_status": classification,
        "severity": raw_prediction.get("severity", "desconocida"),
        "confidence": raw_prediction.get("probability_incorrect", 0.0),
        "threshold": raw_prediction.get("threshold"),
        "primary_error": enriched_errors[0] if enriched_errors else None,
        "errors": enriched_errors,
        "metrics": raw_prediction.get("biomechanical_summary", {}),
    }


def build_user_prompt(diagnosis: Dict[str, Any]) -> str:
    return f"""
Genera retroalimentacion para este resultado biomecanico:

{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

Tarea:
- Indica la correccion prioritaria.
- Da una retroalimentacion corta.
- Da una explicacion breve para el paciente.

Restricciones:
- No inventes errores nuevos.
- No agregues recomendaciones medicas.
- Basa la respuesta solo en \"overall_status\", \"severity\", \"errors\" y \"metrics\".
- Se claro, amable y breve.
"""


def ask_ollama_from_diagnosis(diagnosis: Dict[str, Any], exercise_id: int) -> Dict[str, str]:
    model_name = _get_model_name(exercise_id)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(diagnosis)},
        ],
        "format": FEEDBACK_SCHEMA,
        "stream": False,
    }

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            content = data["message"]["content"]
            return json.loads(content)
        except requests.HTTPError as exc:
            last_error = exc
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text if exc.response is not None else ""
            if status == 500 and attempt == 0:
                # Some runners fail on first generation right after model load.
                time.sleep(1)
                continue
            raise RuntimeError(
                f"Ollama HTTP {status} (model={model_name}, url={OLLAMA_URL}). Body: {body}"
            ) from exc
        except (requests.RequestException, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(1)
                continue
            raise RuntimeError(
                f"Fallo parseando/respondiendo Ollama (model={model_name}, url={OLLAMA_URL}): {exc}"
            ) from exc

    raise RuntimeError(f"Fallo consultando Ollama (model={model_name}): {last_error}")


def ask_ollama(raw_prediction: Dict[str, Any], exercise_id: int) -> Dict[str, str]:
    diagnosis = prepare_diagnosis(raw_prediction, exercise_id)
    return ask_ollama_from_diagnosis(diagnosis, exercise_id)