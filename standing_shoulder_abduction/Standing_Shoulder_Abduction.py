from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


EXERCISE_ID = 9
EXERCISE_NAME = "Standing Shoulder Abduction"
WINDOW_SIZE = 48
RANDOM_STATE = 42
TARGET_RECALL = 0.90

PROJECT_DIR = Path(__file__).resolve().parent.parent
JSON_EXAMPLE_PATH = PROJECT_DIR / "jsonEjemplo.json"
DATASET_ROOT = PROJECT_DIR / "dataset"
ARTIFACTS_DIR = PROJECT_DIR / "standing_shoulder_abduction/standing_shoulder_abduction_artifacts"

UI_PRMD_SPLITS = [
    ("Segmented Movements", 0, True),
    ("Incorrect Segmented Movements", 1, True),
    ("Movements", 0, False),
    ("Incorrect Movements", 1, False),
]

# Indices de joints tipo Kinect (3 columnas por joint) para archivos Positions (66 cols = 22*3).
KINECT_JOINT_IDX = {
    "hip_center": 0,
    "shoulder_center": 2,
    "head": 3,
    "shoulder_l": 4,
    "elbow_l": 5,
    "wrist_l": 6,
    "shoulder_r": 8,
    "elbow_r": 9,
    "wrist_r": 10,
}

JOINT_ALIASES = {
    "nose": ["nariz", "nose"],
    "neck": ["cuello", "neck"],
    "shoulder_r": ["hombro_d", "shoulder_r", "right_shoulder"],
    "elbow_r": ["codo_d", "elbow_r", "right_elbow"],
    "wrist_r": ["muneca_d", "muneca_d", "wrist_r", "right_wrist"],
    "shoulder_l": ["hombro_i", "shoulder_l", "left_shoulder"],
    "elbow_l": ["codo_i", "elbow_l", "left_elbow"],
    "wrist_l": ["muneca_i", "muneca_i", "wrist_l", "left_wrist"],
    "hip_r": ["cadera_d", "hip_r", "right_hip"],
    "hip_l": ["cadera_i", "hip_l", "left_hip"],
}

FEATURE_NAMES = [
    "shoulder_angle_deg",
    "wrist_above_elbow_norm",
    "elbow_above_shoulder_norm",
    "shoulder_elevation_norm",
    "trunk_tilt_deg",
    "symmetry_diff_deg",
    "symmetry_available",
    "angular_velocity_deg",
    "range_progress_norm",
    "valid_points_ratio",
]

ERROR_LABELS = {
    "low_range": "rango_abduccion_insuficiente",
    "trunk_lean": "inclinacion_excesiva_tronco",
    "shrug": "elevacion_escapular_excesiva",
    "wrist_drop": "descenso_de_codo_o_muneca",
    "asymmetry": "asimetria_derecha_izquierda",
    "jerky": "velocidad_angular_irregular",
}


def seed_everything(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    try:
        tf = require_tensorflow()
        tf.random.set_seed(seed)
    except RuntimeError:
        pass


def require_tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow no está disponible en este entorno. "
            "Para entrenar el TCN usa Python 3.11/3.12 y luego instala tensorflow, numpy, pandas, scikit-learn y joblib."
        ) from exc
    return tf


def _parse_subject_id(value: str) -> int:
    match = re.search(r"s(\d{2})", value)
    return int(match.group(1)) if match else -1


def _parse_movement_id(value: str) -> int:
    match = re.search(r"m(\d{2})", value)
    return int(match.group(1)) if match else -1


def _load_txt(filepath: Path) -> np.ndarray | None:
    try:
        for sep in (",", None):
            try:
                arr = np.loadtxt(filepath, delimiter=sep)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[0] > 0 and arr.shape[1] > 0:
                    return arr.astype(np.float32)
            except ValueError:
                continue
        return None
    except Exception:
        return None


def _safe_div(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray | float:
    return numerator / (denominator + 1e-6)


def _rotation_2d(angle_deg: float) -> np.ndarray:
    angle_rad = math.radians(angle_deg)
    return np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad)],
            [math.sin(angle_rad), math.cos(angle_rad)],
        ],
        dtype=np.float32,
    )


def _point_from_frame(frame_points: dict[str, Any], canonical_name: str) -> tuple[np.ndarray, bool]:
    for alias in JOINT_ALIASES[canonical_name]:
        if alias in frame_points:
            joint = frame_points[alias] or {}
            return np.array(
                [
                    float(joint.get("x", np.nan)),
                    float(joint.get("y", np.nan)),
                    float(joint.get("z", 0.0)),
                ],
                dtype=np.float32,
            ), True
    return np.array([np.nan, np.nan, np.nan], dtype=np.float32), False


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def _norm_2d(vector: np.ndarray) -> float:
    return float(np.linalg.norm(vector[:2]))


def _angle_between_2d(a: np.ndarray, b: np.ndarray) -> float:
    a2 = a[:2].astype(np.float64)
    b2 = b[:2].astype(np.float64)
    norm_a = np.linalg.norm(a2)
    norm_b = np.linalg.norm(b2)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    cosine = np.clip(np.dot(a2, b2) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _resample_series(values: np.ndarray, target_len: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if values.size == 1:
        return np.repeat(values, target_len).astype(np.float32)
    old_idx = np.linspace(0.0, 1.0, num=values.size)
    new_idx = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(new_idx, old_idx, values).astype(np.float32)


def _smooth_monotonic_progress(progress: np.ndarray) -> np.ndarray:
    progress = np.clip(progress, 0.0, 1.0)
    progress = np.maximum.accumulate(progress)
    max_value = float(progress.max())
    if max_value < 1e-6:
        return np.linspace(0.0, 1.0, num=progress.size, dtype=np.float32)
    progress = progress / max_value
    return (0.5 - 0.5 * np.cos(progress * np.pi)).astype(np.float32)


def load_json_payload(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_frame(frame: Any, default_index: int = 0) -> dict[str, Any]:
    if not isinstance(frame, dict):
        raise TypeError("Cada frame debe ser un objeto JSON.")

    if "puntos_clave" in frame and isinstance(frame["puntos_clave"], dict):
        points = frame["puntos_clave"]
    elif "puntos" in frame and isinstance(frame["puntos"], dict):
        points = frame["puntos"]
    else:
        # Formato legado: el frame ya es el diccionario de joints.
        points = frame

    normalized: dict[str, Any] = {
        "frame_index": int(frame.get("frame_index", default_index)),
        "puntos_clave": points,
    }

    if "t" in frame:
        normalized["t"] = frame["t"]

    return normalized


def coerce_frames(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_normalize_frame(frame, idx) for idx, frame in enumerate(payload)]
    if not isinstance(payload, dict):
        raise TypeError("El JSON debe ser una lista de frames o un objeto con puntos_clave/puntos/frames/secuencia.")
    for key in ("frames", "ventana", "secuencia", "window"):
        if key in payload:
            frames = payload[key]
            if not isinstance(frames, list):
                raise TypeError(f"'{key}' debe ser una lista de frames.")
            return [_normalize_frame(frame, idx) for idx, frame in enumerate(frames)]
    if "puntos_clave" in payload or "puntos" in payload:
        return [_normalize_frame(payload, 0)]
    raise KeyError("No se encontró una secuencia temporal en el JSON.")


def _default_left_side(points: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    shoulder_r = points[JOINT_ALIASES["shoulder_r"][0]]
    elbow_r = points[JOINT_ALIASES["elbow_r"][0]]
    wrist_r = points[JOINT_ALIASES["wrist_r"][0]]
    center_x = float(points.get(JOINT_ALIASES["nose"][0], {"x": 0.5}).get("x", 0.5))
    left_shoulder = {"x": center_x + (center_x - shoulder_r["x"]), "y": shoulder_r["y"], "z": shoulder_r.get("z", 0.0)}
    left_elbow = {"x": center_x + (center_x - elbow_r["x"]), "y": elbow_r["y"], "z": elbow_r.get("z", 0.0)}
    left_wrist = {"x": center_x + (center_x - wrist_r["x"]), "y": wrist_r["y"], "z": wrist_r.get("z", 0.0)}
    return {
        JOINT_ALIASES["shoulder_l"][0]: left_shoulder,
        JOINT_ALIASES["elbow_l"][0]: left_elbow,
        JOINT_ALIASES["wrist_l"][0]: left_wrist,
    }


def generate_temporal_window_from_example(
    payload: dict[str, Any],
    rng: np.random.Generator,
    incorrect: bool,
    n_frames: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    n_frames = n_frames or int(rng.integers(36, 60))
    seed_frames = coerce_frames(payload)
    if not seed_frames:
        raise ValueError("El payload de ejemplo no contiene frames.")

    base_points = dict(seed_frames[0]["puntos_clave"])
    base_points.update(_default_left_side(base_points))

    nose = base_points[JOINT_ALIASES["nose"][0]]
    shoulder_r = base_points[JOINT_ALIASES["shoulder_r"][0]]
    elbow_r = base_points[JOINT_ALIASES["elbow_r"][0]]
    wrist_r = base_points[JOINT_ALIASES["wrist_r"][0]]
    shoulder_l = base_points[JOINT_ALIASES["shoulder_l"][0]]
    elbow_l = base_points[JOINT_ALIASES["elbow_l"][0]]
    wrist_l = base_points[JOINT_ALIASES["wrist_l"][0]]

    upper_r = np.array([elbow_r["x"] - shoulder_r["x"], elbow_r["y"] - shoulder_r["y"]], dtype=np.float32)
    fore_r = np.array([wrist_r["x"] - elbow_r["x"], wrist_r["y"] - elbow_r["y"]], dtype=np.float32)
    upper_l = np.array([elbow_l["x"] - shoulder_l["x"], elbow_l["y"] - shoulder_l["y"]], dtype=np.float32)
    fore_l = np.array([wrist_l["x"] - elbow_l["x"], wrist_l["y"] - elbow_l["y"]], dtype=np.float32)

    progress = np.linspace(0.0, 1.0, num=n_frames, dtype=np.float32)
    if incorrect and rng.random() < 0.35:
        progress = progress + rng.normal(0.0, 0.08, size=n_frames).astype(np.float32)
    progress = _smooth_monotonic_progress(progress)

    active_errors: list[str] = []
    if incorrect:
        error_pool = ["low_range", "trunk_lean", "shrug", "wrist_drop", "asymmetry", "jerky"]
        n_errors = int(rng.integers(1, 4))
        active_errors = sorted(set(rng.choice(error_pool, size=n_errors, replace=False).tolist()))

    target_angle = float(rng.uniform(82.0, 100.0))
    if "low_range" in active_errors:
        target_angle = float(rng.uniform(35.0, 68.0))

    trunk_lean = float(rng.uniform(0.0, 0.018))
    if "trunk_lean" in active_errors:
        trunk_lean = float(rng.uniform(0.045, 0.10))

    shrug_gain = float(rng.uniform(0.0, 0.012))
    if "shrug" in active_errors:
        shrug_gain = float(rng.uniform(0.035, 0.075))

    wrist_drop_gain = float(rng.uniform(-0.005, 0.008))
    if "wrist_drop" in active_errors:
        wrist_drop_gain = float(rng.uniform(0.03, 0.08))

    left_scale = 1.0
    if "asymmetry" in active_errors:
        left_scale = float(rng.uniform(0.45, 0.8))

    noise_std = 0.005 if not incorrect else 0.012

    frames: list[dict[str, Any]] = []
    for idx, step in enumerate(progress):
        current_angle = float(step * target_angle)
        right_rot = _rotation_2d(-current_angle)
        left_rot = _rotation_2d(current_angle * left_scale)

        shoulder_r_xy = np.array([shoulder_r["x"], shoulder_r["y"]], dtype=np.float32)
        shoulder_r_xy[1] -= shrug_gain * float(step)

        shoulder_l_xy = np.array([shoulder_l["x"], shoulder_l["y"]], dtype=np.float32)
        shoulder_l_xy[1] -= shrug_gain * float(step) * 0.35

        elbow_r_xy = shoulder_r_xy + right_rot @ upper_r
        wrist_r_xy = elbow_r_xy + right_rot @ fore_r
        wrist_r_xy[1] += wrist_drop_gain * float(step)

        elbow_l_xy = shoulder_l_xy + left_rot @ upper_l
        wrist_l_xy = elbow_l_xy + left_rot @ fore_l

        nose_xy = np.array([nose["x"], nose["y"]], dtype=np.float32)
        nose_xy[0] += trunk_lean * float(step)

        if "jerky" in active_errors and idx % 5 == 0:
            elbow_r_xy += rng.normal(0.0, 0.01, size=2)
            wrist_r_xy += rng.normal(0.0, 0.015, size=2)

        frame_points = {
            JOINT_ALIASES["nose"][0]: {
                "x": float(nose_xy[0] + rng.normal(0.0, noise_std)),
                "y": float(nose_xy[1] + rng.normal(0.0, noise_std)),
                "z": float(nose.get("z", 0.0)),
            },
            JOINT_ALIASES["shoulder_r"][0]: {
                "x": float(shoulder_r_xy[0] + rng.normal(0.0, noise_std)),
                "y": float(shoulder_r_xy[1] + rng.normal(0.0, noise_std)),
                "z": float(shoulder_r.get("z", 0.0)),
            },
            JOINT_ALIASES["elbow_r"][0]: {
                "x": float(elbow_r_xy[0] + rng.normal(0.0, noise_std)),
                "y": float(elbow_r_xy[1] + rng.normal(0.0, noise_std)),
                "z": float(elbow_r.get("z", 0.0)),
            },
            JOINT_ALIASES["wrist_r"][0]: {
                "x": float(wrist_r_xy[0] + rng.normal(0.0, noise_std)),
                "y": float(wrist_r_xy[1] + rng.normal(0.0, noise_std)),
                "z": float(wrist_r.get("z", 0.0)),
            },
            JOINT_ALIASES["shoulder_l"][0]: {
                "x": float(shoulder_l_xy[0] + rng.normal(0.0, noise_std)),
                "y": float(shoulder_l_xy[1] + rng.normal(0.0, noise_std)),
                "z": float(shoulder_l.get("z", 0.0)),
            },
            JOINT_ALIASES["elbow_l"][0]: {
                "x": float(elbow_l_xy[0] + rng.normal(0.0, noise_std)),
                "y": float(elbow_l_xy[1] + rng.normal(0.0, noise_std)),
                "z": float(elbow_l.get("z", 0.0)),
            },
            JOINT_ALIASES["wrist_l"][0]: {
                "x": float(wrist_l_xy[0] + rng.normal(0.0, noise_std)),
                "y": float(wrist_l_xy[1] + rng.normal(0.0, noise_std)),
                "z": float(wrist_l.get("z", 0.0)),
            },
        }
        frames.append(
            {
                "frame_index": idx,
                "puntos_clave": frame_points,
            }
        )

    return frames, active_errors


def extract_biomechanical_window(
    frames_json: list[dict[str, Any]],
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, dict[str, float]]:
    if not frames_json:
        raise ValueError("La ventana temporal no puede estar vacía.")

    right_angles: list[float] = []
    wrist_minus_elbow: list[float] = []
    elbow_minus_shoulder: list[float] = []
    shoulder_head_gap: list[float] = []
    trunk_tilt: list[float] = []
    symmetry_diff: list[float] = []
    symmetry_available: list[float] = []
    valid_ratio: list[float] = []

    for frame in frames_json:
        points = frame.get("puntos_clave", frame)
        nose, nose_ok = _point_from_frame(points, "nose")
        shoulder_r, shoulder_r_ok = _point_from_frame(points, "shoulder_r")
        elbow_r, elbow_r_ok = _point_from_frame(points, "elbow_r")
        wrist_r, wrist_r_ok = _point_from_frame(points, "wrist_r")
        shoulder_l, shoulder_l_ok = _point_from_frame(points, "shoulder_l")
        elbow_l, elbow_l_ok = _point_from_frame(points, "elbow_l")

        available = [nose_ok, shoulder_r_ok, elbow_r_ok, wrist_r_ok, shoulder_l_ok, elbow_l_ok]
        valid_ratio.append(float(np.mean(available)))

        torso_ref = None
        hip_l, hip_l_ok = _point_from_frame(points, "hip_l")
        hip_r, hip_r_ok = _point_from_frame(points, "hip_r")
        if hip_l_ok and hip_r_ok and shoulder_r_ok:
            torso_ref = shoulder_r - _midpoint(hip_l, hip_r)
        elif nose_ok and shoulder_r_ok:
            torso_ref = nose - shoulder_r
        else:
            torso_ref = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        upper_arm_r = elbow_r - shoulder_r if shoulder_r_ok and elbow_r_ok else np.array([0.0, -1.0, 0.0], dtype=np.float32)
        right_angle = _angle_between_2d(upper_arm_r, torso_ref)
        right_angles.append(right_angle)

        scale = _norm_2d(elbow_r - shoulder_r) + _norm_2d(wrist_r - elbow_r) if shoulder_r_ok and elbow_r_ok and wrist_r_ok else 1.0
        wrist_minus_elbow.append(float(_safe_div(elbow_r[1] - wrist_r[1], scale)))
        elbow_minus_shoulder.append(float(_safe_div(shoulder_r[1] - elbow_r[1], scale)))
        shoulder_head_gap.append(float(_safe_div(shoulder_r[1] - nose[1], scale))) if nose_ok and shoulder_r_ok else shoulder_head_gap.append(0.0)

        trunk_vector = nose - shoulder_r if nose_ok and shoulder_r_ok else np.array([0.0, -1.0, 0.0], dtype=np.float32)
        trunk_tilt.append(_angle_between_2d(trunk_vector, np.array([0.0, -1.0, 0.0], dtype=np.float32)))

        if shoulder_l_ok and elbow_l_ok:
            upper_arm_l = elbow_l - shoulder_l
            left_torso_ref = nose - shoulder_l if nose_ok else torso_ref
            symmetry_diff.append(abs(right_angle - _angle_between_2d(upper_arm_l, left_torso_ref)))
            symmetry_available.append(1.0)
        else:
            symmetry_diff.append(0.0)
            symmetry_available.append(0.0)

    right_angles_arr = _resample_series(np.array(right_angles), window_size)
    wrist_minus_elbow_arr = _resample_series(np.array(wrist_minus_elbow), window_size)
    elbow_minus_shoulder_arr = _resample_series(np.array(elbow_minus_shoulder), window_size)
    shoulder_head_gap_arr = _resample_series(np.array(shoulder_head_gap), window_size)
    trunk_tilt_arr = _resample_series(np.array(trunk_tilt), window_size)
    symmetry_diff_arr = _resample_series(np.array(symmetry_diff), window_size)
    symmetry_available_arr = _resample_series(np.array(symmetry_available), window_size)
    valid_ratio_arr = _resample_series(np.array(valid_ratio), window_size)

    baseline_gap = float(np.median(shoulder_head_gap_arr[: max(3, window_size // 8)]))
    shoulder_elevation_arr = np.clip(
        _safe_div(baseline_gap - shoulder_head_gap_arr, abs(baseline_gap) + 1e-6),
        -2.0,
        2.0,
    ).astype(np.float32)

    angular_velocity_arr = np.gradient(right_angles_arr).astype(np.float32)
    range_progress_arr = np.clip(np.maximum.accumulate(right_angles_arr) / 120.0, 0.0, 1.5).astype(np.float32)

    window_matrix = np.stack(
        [
            right_angles_arr,
            wrist_minus_elbow_arr,
            elbow_minus_shoulder_arr,
            shoulder_elevation_arr,
            trunk_tilt_arr,
            symmetry_diff_arr,
            symmetry_available_arr,
            angular_velocity_arr,
            range_progress_arr,
            valid_ratio_arr,
        ],
        axis=1,
    ).astype(np.float32)

    summary = {
        "max_shoulder_angle_deg": float(right_angles_arr.max()),
        "mean_shoulder_angle_deg": float(right_angles_arr.mean()),
        "trunk_tilt_p95_deg": float(np.percentile(trunk_tilt_arr, 95)),
        "shoulder_elevation_p95": float(np.percentile(shoulder_elevation_arr, 95)),
        "symmetry_p95_deg": float(np.percentile(symmetry_diff_arr, 95)) if float(symmetry_available_arr.max()) > 0 else 0.0,
        "velocity_std_deg": float(angular_velocity_arr.std()),
        "wrist_above_elbow_mean": float(wrist_minus_elbow_arr.mean()),
        "valid_points_ratio_mean": float(valid_ratio_arr.mean()),
    }
    return window_matrix, summary


def _joint_series_from_positions(arr: np.ndarray, joint_idx: int) -> np.ndarray:
    start = joint_idx * 3
    end = start + 3
    if arr.shape[1] < end:
        return np.zeros((arr.shape[0], 3), dtype=np.float32)
    return arr[:, start:end].astype(np.float32)


def extract_biomechanical_window_from_positions(
    arr: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> tuple[np.ndarray, dict[str, float]]:
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("La secuencia de posiciones debe tener al menos 2 frames.")

    shoulder_r = _joint_series_from_positions(arr, KINECT_JOINT_IDX["shoulder_r"])
    elbow_r = _joint_series_from_positions(arr, KINECT_JOINT_IDX["elbow_r"])
    wrist_r = _joint_series_from_positions(arr, KINECT_JOINT_IDX["wrist_r"])
    shoulder_l = _joint_series_from_positions(arr, KINECT_JOINT_IDX["shoulder_l"])
    elbow_l = _joint_series_from_positions(arr, KINECT_JOINT_IDX["elbow_l"])
    shoulder_center = _joint_series_from_positions(arr, KINECT_JOINT_IDX["shoulder_center"])
    hip_center = _joint_series_from_positions(arr, KINECT_JOINT_IDX["hip_center"])
    head = _joint_series_from_positions(arr, KINECT_JOINT_IDX["head"])

    right_angles: list[float] = []
    wrist_minus_elbow: list[float] = []
    elbow_minus_shoulder: list[float] = []
    shoulder_center_gap: list[float] = []
    trunk_tilt: list[float] = []
    symmetry_diff: list[float] = []
    symmetry_available: list[float] = []
    valid_ratio: list[float] = []

    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    for idx in range(arr.shape[0]):
        torso_ref = shoulder_center[idx] - hip_center[idx]
        if _norm_2d(torso_ref) < 1e-6:
            torso_ref = head[idx] - shoulder_center[idx]
        if _norm_2d(torso_ref) < 1e-6:
            torso_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        upper_r = elbow_r[idx] - shoulder_r[idx]
        fore_r = wrist_r[idx] - elbow_r[idx]
        upper_l = elbow_l[idx] - shoulder_l[idx]

        right_angle = _angle_between_2d(upper_r, torso_ref)
        left_angle = _angle_between_2d(upper_l, torso_ref)
        right_angles.append(right_angle)

        arm_scale = _norm_2d(upper_r) + _norm_2d(fore_r)
        if arm_scale < 1e-6:
            arm_scale = 1.0

        wrist_minus_elbow.append(float((elbow_r[idx, 1] - wrist_r[idx, 1]) / arm_scale))
        elbow_minus_shoulder.append(float((shoulder_r[idx, 1] - elbow_r[idx, 1]) / arm_scale))
        shoulder_center_gap.append(float((shoulder_r[idx, 1] - shoulder_center[idx, 1]) / arm_scale))
        trunk_tilt.append(_angle_between_2d(torso_ref, vertical))

        symmetry_diff.append(abs(right_angle - left_angle))
        symmetry_available.append(1.0)
        valid_ratio.append(1.0)

    right_angles_arr = _resample_series(np.array(right_angles), window_size)
    wrist_minus_elbow_arr = _resample_series(np.array(wrist_minus_elbow), window_size)
    elbow_minus_shoulder_arr = _resample_series(np.array(elbow_minus_shoulder), window_size)
    shoulder_center_gap_arr = _resample_series(np.array(shoulder_center_gap), window_size)
    trunk_tilt_arr = _resample_series(np.array(trunk_tilt), window_size)
    symmetry_diff_arr = _resample_series(np.array(symmetry_diff), window_size)
    symmetry_available_arr = _resample_series(np.array(symmetry_available), window_size)
    valid_ratio_arr = _resample_series(np.array(valid_ratio), window_size)

    baseline_gap = float(np.median(shoulder_center_gap_arr[: max(3, window_size // 8)]))
    shoulder_elevation_arr = np.clip(
        _safe_div(shoulder_center_gap_arr - baseline_gap, abs(baseline_gap) + 1e-6),
        -2.0,
        2.0,
    ).astype(np.float32)

    angular_velocity_arr = np.gradient(right_angles_arr).astype(np.float32)
    range_progress_arr = np.clip(np.maximum.accumulate(right_angles_arr) / 120.0, 0.0, 1.5).astype(np.float32)

    window_matrix = np.stack(
        [
            right_angles_arr,
            wrist_minus_elbow_arr,
            elbow_minus_shoulder_arr,
            shoulder_elevation_arr,
            trunk_tilt_arr,
            symmetry_diff_arr,
            symmetry_available_arr,
            angular_velocity_arr,
            range_progress_arr,
            valid_ratio_arr,
        ],
        axis=1,
    ).astype(np.float32)

    summary = {
        "max_shoulder_angle_deg": float(right_angles_arr.max()),
        "mean_shoulder_angle_deg": float(right_angles_arr.mean()),
        "trunk_tilt_p95_deg": float(np.percentile(trunk_tilt_arr, 95)),
        "shoulder_elevation_p95": float(np.percentile(shoulder_elevation_arr, 95)),
        "symmetry_p95_deg": float(np.percentile(symmetry_diff_arr, 95)),
        "velocity_std_deg": float(angular_velocity_arr.std()),
        "wrist_above_elbow_mean": float(wrist_minus_elbow_arr.mean()),
        "valid_points_ratio_mean": float(valid_ratio_arr.mean()),
    }
    return window_matrix, summary


def load_ui_prmd_dataset(
    base_dir: Path = DATASET_ROOT,
    movement_id: int = EXERCISE_ID,
    use_segmented: bool = True,
    include_non_segmented: bool = True,
    strict_all: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    candidate_files = 0
    skipped_load = 0
    skipped_shape = 0
    skipped_features = 0

    for folder_name, label, is_segmented in UI_PRMD_SPLITS:
        if is_segmented and not use_segmented:
            continue
        if (not is_segmented) and not include_non_segmented:
            continue

        data_dir = base_dir / folder_name / folder_name / "Kinect" / "Positions"
        if not data_dir.exists():
            continue

        for fp in sorted(data_dir.glob("*.txt")):
            if _parse_movement_id(fp.name) != movement_id:
                continue
            candidate_files += 1

            arr = _load_txt(fp)
            if arr is None:
                skipped_load += 1
                continue
            if arr.shape[0] < 2 or arr.shape[1] < 33:
                skipped_shape += 1
                continue

            try:
                window_matrix, summary = extract_biomechanical_window_from_positions(arr, window_size=WINDOW_SIZE)
            except Exception:
                skipped_features += 1
                continue

            records.append(
                {
                    "sequence_id": fp.stem,
                    "subject_id": _parse_subject_id(fp.name),
                    "label": label,
                    "file": fp.name,
                    "folder_split": folder_name,
                    "is_segmented": is_segmented,
                    "n_frames_raw": int(arr.shape[0]),
                    "window": window_matrix,
                    **summary,
                }
            )

    if not records:
        raise FileNotFoundError("No se encontraron secuencias m09 en las carpetas UI-PRMD esperadas.")

    if strict_all and len(records) != candidate_files:
        raise RuntimeError(
            "No se pudieron usar todas las secuencias m09 detectadas. "
            f"detectadas={candidate_files}, usadas={len(records)}, "
            f"fallo_carga={skipped_load}, formato_invalido={skipped_shape}, fallo_features={skipped_features}."
        )

    meta = pd.DataFrame(records)
    meta.attrs["coverage"] = {
        "candidate_files": int(candidate_files),
        "used_sequences": int(len(records)),
        "skipped_load": int(skipped_load),
        "skipped_shape": int(skipped_shape),
        "skipped_features": int(skipped_features),
    }
    X = np.stack(meta.pop("window").to_list()).astype(np.float32)
    y = meta["label"].to_numpy(dtype=np.int32)
    return X, y, meta


def get_dataset() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, str]:
    try:
        X, y, meta = load_ui_prmd_dataset(
            base_dir=DATASET_ROOT,
            movement_id=EXERCISE_ID,
            use_segmented=True,
            include_non_segmented=True,
            strict_all=True,
        )
        return X, y, meta, "ui_prmd_kinect_positions"
    except RuntimeError:
        # Si la cobertura total falla, prioriza seguir con UI-PRMD en modo relajado.
        X, y, meta = load_ui_prmd_dataset(
            base_dir=DATASET_ROOT,
            movement_id=EXERCISE_ID,
            use_segmented=True,
            include_non_segmented=True,
            strict_all=False,
        )
        return X, y, meta, "ui_prmd_kinect_positions_relaxed"
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No se encontraron secuencias UI-PRMD m09 en dataset/. "
            "Verifica la estructura: dataset/<split>/<split>/Kinect/Positions/*.txt"
        ) from exc


def fit_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = X_train.std(axis=(0, 1), keepdims=True).astype(np.float32) + 1e-6
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def tcn_residual_block(inputs: Any, filters: int, dilation_rate: int, dropout_rate: float) -> Any:
    tf = require_tensorflow()
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding="causal",
        dilation_rate=dilation_rate,
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    residual = inputs
    if inputs.shape[-1] != filters:
        residual = tf.keras.layers.Conv1D(filters=filters, kernel_size=1, padding="same")(inputs)

    x = tf.keras.layers.Add()([x, residual])
    return tf.keras.layers.Activation("relu")(x)


def build_tcn_model(input_shape: tuple[int, int]) -> Any:
    tf = require_tensorflow()
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for filters, dilation in ((32, 1), (32, 2), (64, 4), (64, 8)):
        x = tcn_residual_block(x, filters=filters, dilation_rate=dilation, dropout_rate=0.15)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="standing_shoulder_abduction_tcn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray, target_recall: float = TARGET_RECALL) -> tuple[float, pd.DataFrame]:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba, pos_label=1)
    threshold_values = np.concatenate(([0.0], thresholds))

    rows: list[dict[str, float]] = []
    for threshold in threshold_values:
        y_pred = (y_proba >= threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = 0.0 if (4 * precision + recall) == 0 else (5 * precision * recall) / (4 * precision + recall)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "f2": float(f2),
            }
        )

    grid = pd.DataFrame(rows).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    candidates = grid[grid["recall"] >= target_recall]
    if not candidates.empty:
        best = candidates.sort_values(["precision", "f2", "threshold"], ascending=[False, False, True]).iloc[0]
    else:
        best = grid.sort_values(["f2", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), grid


def detect_errors(summary: dict[str, float]) -> list[str]:
    errors: list[str] = []
    if summary["max_shoulder_angle_deg"] < 75.0:
        errors.append(ERROR_LABELS["low_range"])
    if summary["trunk_tilt_p95_deg"] > 12.0:
        errors.append(ERROR_LABELS["trunk_lean"])
    if summary["shoulder_elevation_p95"] > 0.18:
        errors.append(ERROR_LABELS["shrug"])
    if summary["wrist_above_elbow_mean"] < -0.02:
        errors.append(ERROR_LABELS["wrist_drop"])
    if summary["symmetry_p95_deg"] > 18.0:
        errors.append(ERROR_LABELS["asymmetry"])
    if summary["velocity_std_deg"] > 4.5:
        errors.append(ERROR_LABELS["jerky"])
    return errors


def severity_from_probability(probability_incorrect: float, errors: list[str]) -> str:
    severity_score = max(probability_incorrect, min(1.0, 0.2 * len(errors) + probability_incorrect * 0.8))
    if severity_score < 0.45:
        return "leve"
    if severity_score < 0.75:
        return "moderada"
    return "severa"


def export_artifacts(
    model: Any,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    threshold: float,
    dataset_source: str,
    metrics: dict[str, float],
    output_dir: Path = ARTIFACTS_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "standing_shoulder_abduction_tcn.keras"
    bundle_path = output_dir / "standing_shoulder_abduction_bundle.joblib"

    model.save(model_path)
    joblib.dump(
        {
            "model_path": str(model_path),
            "threshold": threshold,
            "window_size": WINDOW_SIZE,
            "feature_names": FEATURE_NAMES,
            "exercise_id": EXERCISE_ID,
            "exercise_name": EXERCISE_NAME,
            "dataset_source": dataset_source,
            "target_recall": TARGET_RECALL,
            "scaler_mean": scaler_mean,
            "scaler_std": scaler_std,
            "metrics": metrics,
            "outputs": {
                0: "correcta",
                1: "incorrecta",
            },
        },
        bundle_path,
    )
    return {"model": model_path, "bundle": bundle_path}


def infer_from_window(
    frames_json: list[dict[str, Any]],
    model: Any,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    window_matrix, summary = extract_biomechanical_window(frames_json, window_size=bundle["window_size"])
    X = apply_scaler(window_matrix[None, ...], bundle["scaler_mean"], bundle["scaler_std"])
    probability_incorrect = float(model.predict(X, verbose=0).reshape(-1)[0])
    label = int(probability_incorrect >= bundle["threshold"])
    errors = detect_errors(summary)
    if label == 0:
        errors = []
    return {
        "exercise": bundle["exercise_name"],
        "movement_id": bundle["exercise_id"],
        "label": label,
        "classification": "incorrecta" if label else "correcta",
        "probability_incorrect": round(probability_incorrect, 4),
        "threshold": round(float(bundle["threshold"]), 4),
        "severity": severity_from_probability(probability_incorrect, errors),
        "errors_detected": errors,
        "biomechanical_summary": {k: round(v, 4) for k, v in summary.items()},
    }


def train_and_export() -> dict[str, Any]:
    seed_everything(RANDOM_STATE)
    tf = require_tensorflow()
    X, y, meta, dataset_source = get_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler_mean, scaler_std = fit_scaler(X_train)
    X_train_scaled = apply_scaler(X_train, scaler_mean, scaler_std)
    X_val_scaled = apply_scaler(X_val, scaler_mean, scaler_std)

    model = build_tcn_model(input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]))

    class_counts = np.bincount(y_train)
    positive_weight = float(max(1.0, (class_counts[0] / max(class_counts[1], 1)) * 1.35))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_recall",
            mode="max",
            patience=12,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=80,
        batch_size=32,
        class_weight={0: 1.0, 1: positive_weight},
        callbacks=callbacks,
        verbose=0,
    )

    y_val_proba = model.predict(X_val_scaled, verbose=0).reshape(-1)
    threshold, threshold_grid = tune_threshold(y_val, y_val_proba, target_recall=TARGET_RECALL)
    y_val_pred = (y_val_proba >= threshold).astype(int)

    # Fine-tune with all available sequences so every movement contributes to final weights.
    X_all_scaled = apply_scaler(X, scaler_mean, scaler_std)
    class_counts_full = np.bincount(y)
    positive_weight_full = float(max(1.0, (class_counts_full[0] / max(class_counts_full[1], 1)) * 1.35))
    val_recall_history = history.history.get("val_recall", [])
    if val_recall_history:
        best_epoch = int(np.argmax(val_recall_history)) + 1
    else:
        best_epoch = max(1, len(history.history.get("loss", [])))
    full_fit_epochs = max(1, min(12, best_epoch))
    model.fit(
        X_all_scaled,
        y,
        epochs=full_fit_epochs,
        batch_size=32,
        class_weight={0: 1.0, 1: positive_weight_full},
        verbose=0,
    )

    y_all_proba = model.predict(X_all_scaled, verbose=0).reshape(-1)
    threshold, _ = tune_threshold(y, y_all_proba, target_recall=TARGET_RECALL)
    y_val_proba = model.predict(X_val_scaled, verbose=0).reshape(-1)
    y_val_pred = (y_val_proba >= threshold).astype(int)
    cm = confusion_matrix(y_val, y_val_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "dataset_source": dataset_source,
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "positive_weight": round(positive_weight, 4),
        "positive_weight_full_fit": round(positive_weight_full, 4),
        "full_fit_epochs": int(full_fit_epochs),
        "best_threshold": round(threshold, 4),
        "precision_incorrect": round(precision_score(y_val, y_val_pred, zero_division=0), 4),
        "recall_incorrect": round(recall_score(y_val, y_val_pred, zero_division=0), 4),
        "f1_incorrect": round(f1_score(y_val, y_val_pred, zero_division=0), 4),
        "true_negatives_incorrect": int(tn),
        "false_positives_incorrect": int(fp),
        "false_negatives_incorrect": int(fn),
        "true_positives_incorrect": int(tp),
        "roc_auc": round(roc_auc_score(y_val, y_val_proba), 4),
        "val_loss_min": round(float(np.min(history.history["val_loss"])), 4),
    }

    exported = export_artifacts(
        model=model,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        threshold=threshold,
        dataset_source=dataset_source,
        metrics=metrics,
    )

    bundle = joblib.load(exported["bundle"])
    demo_payload = load_json_payload(JSON_EXAMPLE_PATH)
    demo_frames, _ = generate_temporal_window_from_example(
        payload=demo_payload,
        rng=np.random.default_rng(RANDOM_STATE + 7),
        incorrect=True,
        n_frames=WINDOW_SIZE,
    )
    demo_prediction = infer_from_window(demo_frames, model=model, bundle=bundle)

    print("=== Standing Shoulder Abduction / TCN ===")
    print(f"Dataset usado        : {dataset_source}")
    print(f"Muestras totales     : {len(y)}")
    print(f"Threshold optimizado : {threshold:.4f}")
    print(f"Recall incorrecto    : {metrics['recall_incorrect']:.4f}")
    print(f"Precision incorrecto : {metrics['precision_incorrect']:.4f}")
    print(f"F1 incorrecto        : {metrics['f1_incorrect']:.4f}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"ROC-AUC              : {metrics['roc_auc']:.4f}")
    print("Matriz de confusión:")
    print(cm)
    print("Reporte de clasificación:")
    print(classification_report(y_val, y_val_pred, target_names=["correcta", "incorrecta"], digits=4))
    print("Demo de inferencia:")
    print(json.dumps(demo_prediction, ensure_ascii=False, indent=2))

    return {
        "meta": meta,
        "threshold_grid": threshold_grid,
        "metrics": metrics,
        "artifacts": exported,
        "demo_prediction": demo_prediction,
    }


if __name__ == "__main__":
    train_and_export()