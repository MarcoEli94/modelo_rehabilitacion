import json

ERROR_MAP = {
    "max_shoulder_angle_deg": "rango de movimiento insuficiente",
    "trunk_tilt_p95_deg": "compensación con inclinación del tronco",
    "shoulder_elevation_p95": "elevación compensatoria del hombro",
    "wrist_above_elbow_mean": "mecánica incorrecta del brazo",
    "symmetry_p95_deg": "asimetría en el movimiento",
    "velocity_std_deg": "falta de control del movimiento"
}

SYSTEM_PROMPT = """
Eres un asistente de rehabilitación física.

Tu tarea es transformar resultados biomecánicos estructurados en retroalimentación breve, clara y segura para el usuario.

Reglas:
- Usa solo la información proporcionada.
- No inventes errores, métricas o diagnósticos.
- No menciones lesiones o enfermedades.
- Prioriza el error más importante.
- Usa español claro y natural.
- Responde únicamente en el formato solicitado.
"""

def prepare_diagnosis(raw_prediction: dict) -> dict:
    errors = raw_prediction.get("errors", [])
    enriched_errors = []

    for error in errors:
        code = error["code"]
        enriched_errors.append({
            "code": code,
            "description": ERROR_MAP.get(code, code),
            "score": error["score"]
        })

    enriched_errors.sort(key=lambda x: x["score"], reverse=True)

    return {
        "exercise": raw_prediction["exercise"],
        "overall_status": raw_prediction["overall_status"],
        "severity": raw_prediction["severity"],
        "confidence": raw_prediction["confidence"],
        "primary_error": enriched_errors[0] if enriched_errors else None,
        "errors": enriched_errors,
        "metrics": raw_prediction.get("metrics", {})
    }

def build_user_prompt(diagnosis: dict) -> str:
    return f"""
Genera retroalimentación para este resultado:

{json.dumps(diagnosis, ensure_ascii=False, indent=2)}

Tarea:
- Indica la corrección prioritaria.
- Da una retroalimentación corta.
- Da una explicación breve para el paciente.

Restricciones:
- No inventes errores nuevos.
- No agregues recomendaciones médicas.
- Sé claro, amable y breve.
"""