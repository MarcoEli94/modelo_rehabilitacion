from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
EXAMPLE_PATH = ROOT_DIR / "jsonEjemplo.json"
OUTPUT_DIR = ROOT_DIR / "json_examples"
MODELS = {
    7: ROOT_DIR / "standing_shoulder_internal_external_rotation" / "Standing_Shoulder_Internal_External_Rotation.py",
    9: ROOT_DIR / "standing_shoulder_abduction" / "Standing_Shoulder_Abduction.py",
}


def load_example() -> dict[str, Any]:
    with EXAMPLE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_module(path: Path, name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el módulo {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_payloads() -> None:
    example = load_example()
    OUTPUT_DIR.mkdir(exist_ok=True)

    for exercise_id, module_path in MODELS.items():
        module = load_module(module_path, f"m{exercise_id}_generator")
        exercise_dir = OUTPUT_DIR / f"m{exercise_id:02d}"
        exercise_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42 + exercise_id)

        for idx in range(100):
            incorrect = idx < 50
            frames, errors = module.generate_temporal_window_from_example(
                payload=example,
                rng=rng,
                incorrect=incorrect,
                n_frames=int(rng.integers(36, 60)),
            )
            payload = {
                "ejercicio_id": f"m{exercise_id:02d}",
                "secuencia": frames,
            }
            if idx % 2 == 0:
                payload["movement_id"] = exercise_id

            output_path = exercise_dir / f"example_{idx+1:03d}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)

        print(f"Generados 100 ejemplos para m{exercise_id:02d}: {exercise_dir}")


if __name__ == "__main__":
    make_payloads()
