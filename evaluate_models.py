from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT_DIR = Path(__file__).resolve().parent

MODELS_CONFIG: dict[int, dict[str, Path]] = {
    7: {
        "module": ROOT_DIR
        / "standing_shoulder_internal_external_rotation"
        / "Standing_Shoulder_Internal_External_Rotation.py",
        "bundle": ROOT_DIR
        / "standing_shoulder_internal_external_rotation"
        / "standing_shoulder_internal_external_rotation_artifacts"
        / "standing_shoulder_internal_external_rotation_bundle.joblib",
        "model": ROOT_DIR
        / "standing_shoulder_internal_external_rotation"
        / "standing_shoulder_internal_external_rotation_artifacts"
        / "standing_shoulder_internal_external_rotation_tcn.keras",
    },
    9: {
        "module": ROOT_DIR / "standing_shoulder_abduction" / "Standing_Shoulder_Abduction.py",
        "bundle": ROOT_DIR
        / "standing_shoulder_abduction"
        / "standing_shoulder_abduction_artifacts"
        / "standing_shoulder_abduction_bundle.joblib",
        "model": ROOT_DIR
        / "standing_shoulder_abduction"
        / "standing_shoulder_abduction_artifacts"
        / "standing_shoulder_abduction_tcn.keras",
    },
}


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar el módulo {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate_model(exercise_id: int) -> dict[str, Any]:
    config = MODELS_CONFIG.get(exercise_id)
    if config is None:
        raise ValueError(f"Ejercicio no soportado: {exercise_id}. Usa 7 o 9.")

    module = load_module(config["module"], f"model_eval_m{exercise_id}")
    bundle = joblib.load(config["bundle"])
    model = tf.keras.models.load_model(config["model"])

    X, y, meta, source = module.get_dataset()
    X_scaled = module.apply_scaler(X, bundle["scaler_mean"], bundle["scaler_std"])
    y_proba = model.predict(X_scaled, verbose=0).reshape(-1)
    threshold = float(bundle["threshold"])
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall_inc = recall_score(y, y_pred, pos_label=1, zero_division=0)
    precision_inc = precision_score(y, y_pred, pos_label=1, zero_division=0)
    f1_inc = f1_score(y, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y, y_proba)

    return {
        "exercise_id": exercise_id,
        "exercise_name": module.EXERCISE_NAME,
        "dataset_source": source,
        "n_samples": int(len(y)),
        "threshold": threshold,
        "recall_incorrect": float(recall_inc),
        "precision_incorrect": float(precision_inc),
        "f1_incorrect": float(f1_inc),
        "roc_auc": float(roc_auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": classification_report(
            y, y_pred, target_names=["correcta", "incorrecta"], digits=4
        ),
    }


def print_report(report: dict[str, Any]) -> None:
    print("\n=== Evaluación modelo m{exercise_id:02d} ===".format(**report))
    print(f"Ejercicio           : {report['exercise_name']}")
    print(f"Dataset             : {report['dataset_source']}")
    print(f"Muestras            : {report['n_samples']}")
    print(f"Threshold usado     : {report['threshold']:.4f}")
    print(f"Recall incorrecto   : {report['recall_incorrect']:.4f}")
    print(f"Precision incorrecto: {report['precision_incorrect']:.4f}")
    print(f"F1 incorrecto       : {report['f1_incorrect']:.4f}")
    print(f"ROC-AUC             : {report['roc_auc']:.4f}")
    print(f"TN={report['tn']}  FP={report['fp']}  FN={report['fn']}  TP={report['tp']}")
    print("\nMatriz de clasificación:")
    print(report["classification_report"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluar modelos de rehabilitación y verificar recall / falsos negativos."
    )
    parser.add_argument(
        "--exercise",
        choices=["7", "9", "all"],
        default="all",
        help="Ejercicio a evaluar: 7, 9 o all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ids = [7, 9] if args.exercise == "all" else [int(args.exercise)]
    for exercise_id in ids:
        report = evaluate_model(exercise_id)
        print_report(report)


if __name__ == "__main__":
    main()
