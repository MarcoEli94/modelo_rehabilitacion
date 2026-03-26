from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException

from api import ollama_client

app = FastAPI(
    title="API Rehabilitacion",
    version="1.0.0",
    description="Pipeline API -> Ollama -> Modelos (m07/m09).",
)

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent


def _load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar modulo: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODELS_CONFIG: dict[int, dict[str, Path]] = {
    7: {
        "module": PROJECT_ROOT
        / "standing_shoulder_internal_external_rotation"
        / "Standing_Shoulder_Internal_External_Rotation.py",
        "bundle": PROJECT_ROOT
        / "standing_shoulder_internal_external_rotation"
        / "standing_shoulder_internal_external_rotation_artifacts"
        / "standing_shoulder_internal_external_rotation_bundle.joblib",
        "model": PROJECT_ROOT
        / "standing_shoulder_internal_external_rotation"
        / "standing_shoulder_internal_external_rotation_artifacts"
        / "standing_shoulder_internal_external_rotation_tcn.keras",
    },
    9: {
        "module": PROJECT_ROOT / "standing_shoulder_abduction" / "Standing_Shoulder_Abduction.py",
        "bundle": PROJECT_ROOT
        / "standing_shoulder_abduction"
        / "standing_shoulder_abduction_artifacts"
        / "standing_shoulder_abduction_bundle.joblib",
        "model": PROJECT_ROOT
        / "standing_shoulder_abduction"
        / "standing_shoulder_abduction_artifacts"
        / "standing_shoulder_abduction_tcn.keras",
    },
}


#adición para aumentar velocidad
RUNTIME_CACHE: dict[int, tuple[Any, Any, dict[str, Any]]] = {}



def _parse_exercise_id(payload: dict[str, Any]) -> int:
    raw = payload.get("ejercicio_id")
    if raw is None:
        raw = payload.get("movement_id")

    # if raw in ("", None):
    #     # Compatibilidad con jsonEjemplo actual donde ejercicio_id puede venir vacio.
    #     return 7

    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        val = raw.strip().lower().replace("m", "")
        if val.isdigit():
            return int(val)

    raise HTTPException(
        status_code=400,
        detail="No se pudo determinar el ejercicio. Envia 'ejercicio_id' (ej. 'm07'/'m09' o 7/9).",
    )


#def _load_runtime(exercise_id: int) -> tuple[Any, Any, dict[str, Any]]:
#    conf = MODELS_CONFIG.get(exercise_id)
#    if conf is None:
#        raise HTTPException(status_code=400, detail="Ejercicio no soportado. Usa m07 o m09.")
#
#    module = _load_module(conf["module"], f"movement_module_{exercise_id}")
#
#    bundle_path = conf["bundle"]
#    model_path = conf["model"]
#    if not bundle_path.exists() or not model_path.exists():
#        raise HTTPException(
#            status_code=500,
#            detail=(
#                "No se encontraron artefactos del modelo entrenado. "
#                f"Esperados: {bundle_path.name} y {model_path.name}"
#            ),
#        )
#
#    try:
#        tf = module.require_tensorflow()
#    except RuntimeError as exc:
#        raise HTTPException(
#            status_code=500,
#            detail=(
#                "TensorFlow no esta disponible en este entorno. "
#                "Ejecuta la API con Python 3.11/3.12 e instala tensorflow, numpy, pandas, scikit-learn y joblib."
#            ),
#        ) from exc
#
#    bundle = joblib.load(bundle_path)
#    model = tf.keras.models.load_model(model_path)
#    return module, model, bundle

def _load_runtime(exercise_id: int) -> tuple[Any, Any, dict[str, Any]]:
    if exercise_id in RUNTIME_CACHE:
        return RUNTIME_CACHE[exercise_id]

    conf = MODELS_CONFIG.get(exercise_id)
    if conf is None:
        raise HTTPException(status_code=400, detail="Ejercicio no soportado. Usa m07 o m09.")

    module = _load_module(conf["module"], f"movement_module_{exercise_id}")

    bundle_path = conf["bundle"]
    model_path = conf["model"]
    if not bundle_path.exists() or not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "No se encontraron artefactos del modelo entrenado. "
                f"Esperados: {bundle_path.name} y {model_path.name}"
            ),
        )

    try:
        tf = module.require_tensorflow()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "TensorFlow no esta disponible en este entorno. "
                "Ejecuta la API con Python 3.11/3.12 e instala tensorflow, numpy, pandas, scikit-learn y joblib."
            ),
        ) from exc

    bundle = joblib.load(bundle_path)
    model = tf.keras.models.load_model(model_path)

    runtime = (module, model, bundle)
    RUNTIME_CACHE[exercise_id] = runtime
    return runtime



@app.get("/")
def healthcheck() -> dict[str, str]:
    return {"status": "ok", "message": "Usa POST /ollama-response para pipeline API -> Ollama -> modelos"}


#@app.post("/get-feedback")
#def ollama_response(payload: dict[str, Any]) -> dict[str, Any]:
#    if not isinstance(payload, dict):
#        raise HTTPException(status_code=400, detail="El body debe ser un objeto JSON.")
#
#    if not any(k in payload for k in ("secuencia", "frames", "ventana", "window")):
#        raise HTTPException(
#            status_code=400,
#            detail="Se esperaba secuencia en una de estas claves: secuencia, frames, ventana o window.",
#        )
#
#    exercise_id = _parse_exercise_id(payload)
#    module, model, bundle = _load_runtime(exercise_id)
#
#    frames = module.coerce_frames(payload)
#    prediction = module.infer_from_window(frames_json=frames, model=model, bundle=bundle)
#
#    try:
#        ollama_feedback = ollama_client.ask_ollama(prediction, exercise_id)
#
#
#
#    except Exception as exc:
#        raise HTTPException(status_code=502, detail=f"Error consultando Ollama: {exc}") from exc
#
#    return ollama_feedback


@app.post("/get-feedback")
def ollama_response(payload: dict[str, Any]) -> dict[str, Any]:
    import time
    # --- AÑADE ESTAS LÍNEAS AQUÍ ---
    print("--- NUEVA PETICIÓN RECIBIDA ---")
    print(f"Llaves recibidas: {list(payload.keys())}")
    # Solo imprimimos los primeros 500 caracteres para no saturar la consola
    print(f"Cuerpo truncado: {str(payload)[:500]}...") 
    # -------------------------------

    t0 = time.perf_counter()

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="El body debe ser un objeto JSON.")

    if not any(k in payload for k in ("secuencia", "frames", "ventana", "window")):
        raise HTTPException(
            status_code=400,
            detail="Se esperaba secuencia en una de estas claves: secuencia, frames, ventana o window.",
        )

    exercise_id = _parse_exercise_id(payload)

    t1 = time.perf_counter()
    module, model, bundle = _load_runtime(exercise_id)
    t2 = time.perf_counter()

    frames = module.coerce_frames(payload)
    prediction = module.infer_from_window(frames_json=frames, model=model, bundle=bundle)
    t3 = time.perf_counter()

    try:
        ollama_feedback = ollama_client.ask_ollama(prediction, exercise_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error consultando Ollama: {exc}") from exc
    t4 = time.perf_counter()

    print(f"[TIMING] validacion={(t1-t0):.3f}s")
    print(f"[TIMING] carga_runtime={(t2-t1):.3f}s")
    print(f"[TIMING] inferencia={(t3-t2):.3f}s")
    print(f"[TIMING] ollama={(t4-t3):.3f}s")
    print(f"[TIMING] total={(t4-t0):.3f}s")

    return ollama_feedback

@app.on_event("startup")
def preload_models() -> None:
    for exercise_id in MODELS_CONFIG.keys():
        try:
            _load_runtime(exercise_id)
            print(f"Modelo precargado: m{exercise_id:02d}")
        except Exception as exc:
            print(f"No se pudo precargar m{exercise_id:02d}: {exc}")