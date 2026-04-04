from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_DATA_DIR = ROOT_DIR / "json_examples"
MODULES = {
    7: ROOT_DIR / "standing_shoulder_internal_external_rotation" / "Standing_Shoulder_Internal_External_Rotation.py",
    9: ROOT_DIR / "standing_shoulder_abduction" / "Standing_Shoulder_Abduction.py",
}

sys.path.insert(0, str(ROOT_DIR))

from main import _parse_exercise_id  # noqa: E402


def load_module(path: Path, name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el módulo {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_json_examples_exist() -> None:
    assert TEST_DATA_DIR.exists(), f"No existe {TEST_DATA_DIR}"
    example_files = sorted(TEST_DATA_DIR.rglob("*.json"))
    assert len(example_files) >= 60, f"Se esperaban al menos 60 ejemplos, se encontraron {len(example_files)}"


def test_payloads_are_parsable_and_valid() -> None:
    example_files = sorted(TEST_DATA_DIR.rglob("*.json"))
    assert example_files, "No se encontraron archivos JSON de ejemplo"

    for path in example_files:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        assert isinstance(payload, dict), f"Payload inválido en {path}"
        assert "ejercicio_id" in payload, f"Falta ejercicio_id en {path}"
        assert any(key in payload for key in ("secuencia", "frames", "ventana", "window")), f"Falta secuencia en {path}"

        exercise_id = _parse_exercise_id(payload)
        assert exercise_id in MODULES, f"Ejercicio no soportado en {path}: {exercise_id}"

        module = load_module(MODULES[exercise_id], f"validate_m{exercise_id}")
        frames = module.coerce_frames(payload)
        assert frames, f"No se pudieron normalizar frames en {path}"
        assert all("puntos_clave" in frame for frame in frames), f"Frame no tiene puntos_clave en {path}"
        assert frames[0].get("frame_index", 0) == 0

        window_matrix, summary = module.extract_biomechanical_window(frames)
        assert isinstance(window_matrix, np.ndarray)
        assert window_matrix.shape[0] == module.WINDOW_SIZE
        assert window_matrix.shape[1] == len(module.FEATURE_NAMES)
        assert isinstance(summary, dict)
        assert all(name in summary for name in summary)

if __name__ == "__main__":
    test_json_examples_exist()
    test_payloads_are_parsable_and_valid()
    print("Todas las pruebas de payload JSON pasaron correctamente.")