"""Score Flight vision backends against Unreal reference-dataset truth.

The evaluator deliberately uses the live detector and publisher objects while
keeping truth loading, matching, rendering, and metrics outside Flight.  Each
run receives fresh backend instances so temporal state cannot cross clips.

Coordinate normalization is explicit:

* Unreal world metres ``(X, Y, Z-up)`` -> evaluation local NED ``(X, Y, -Z)``.
* Dataset camera ``(right, up, forward)`` -> OpenCV ``(right, -up, forward)``.
* The synthetic VehicleState attitude removes the live camera mount extrinsic,
  so ``rotation_world_from_camera`` reconstructs the dataset camera pose.

Every processed frame verifies that transform against truth gate positions and
projected corners before pose errors are accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLIGHT_SRC = PROJECT_ROOT / "Flight" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FLIGHT_SRC) not in sys.path:
    sys.path.insert(0, str(FLIGHT_SRC))

from core.schema import VehicleState, VisionFrame, VisionObservation
from sensing.vision.models.deterministic_v3 import DeterministicVision as DeterministicVisionV3
from sensing.vision.models.deterministic_v3_2 import DeterministicVision as DeterministicVisionV3_2
from sensing.vision.models.deterministic_v3.src import void_detection as detection_v3
from sensing.vision.models.deterministic_v3_2.src import void_detection as detection_v3_2
from sensing.vision.models.deterministic_v3_2.src.void_ned import BODY_FROM_CAMERA


DEFAULT_DATASET_ROOT = Path(os.environ.get(
    "VISION_EVALUATION_DATASET_ROOT",
    r"C:\Users\brend\Projects\UE_5\TestProject\ml\datasets\evaluation",
))
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Viewer" / "logs" / "evaluation" / "runs"
BACKEND_FACTORIES = {
    "deterministic_v3": DeterministicVisionV3,
    "deterministic_v3_2": DeterministicVisionV3_2,
}
BACKEND_DETECTION_MODULES = {
    "deterministic_v3": detection_v3,
    "deterministic_v3_2": detection_v3_2,
}
UNREAL_TO_NED = np.diag((1.0, 1.0, -1.0))
CORNER_ROLE_MAP = {
    "upper_left": "tl",
    "upper_right": "tr",
    "lower_left": "bl",
    "lower_right": "br",
}
CORNER_ORDER = ("tl", "tr", "br", "bl")
MATCH_DISTANCE_FLOOR_PX = 20.0
MATCH_DISTANCE_CEILING_PX = 80.0
MATCH_BBOX_DIAGONAL_FRACTION = 0.65
COORDINATE_POSITION_TOLERANCE_M = 1e-5
COORDINATE_PROJECTION_TOLERANCE_PX = 1e-4


class ReferenceEvaluationError(RuntimeError):
    """Raised when reference truth cannot be evaluated safely."""


@dataclass(frozen=True)
class CameraNormalization:
    position_local_ned_m: tuple[float, float, float]
    rotation_ned_from_camera_cv: np.ndarray
    attitude_body_to_ned_wxyz: tuple[float, float, float, float]


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(payload), separators=(",", ":")) + "\n")


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceEvaluationError(f"unable to read JSON: {path}") from exc


def _rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized WXYZ quaternion."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((w, x, y, z), dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _unreal_camera_axes(rotation_deg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Unreal world vectors for camera right, up, and forward."""
    pitch = math.radians(float(rotation_deg["pitch"]))
    yaw = math.radians(float(rotation_deg["yaw"]))
    roll = math.radians(float(rotation_deg["roll"]))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.asarray((cp * cy, cp * sy, sp), dtype=float)
    right = np.asarray((
        sr * sp * cy - cr * sy,
        sr * sp * sy + cr * cy,
        -sr * cp,
    ), dtype=float)
    up = np.asarray((
        -(cr * sp * cy + sr * sy),
        cy * sr - cr * sp * sy,
        cr * cp,
    ), dtype=float)
    return right, up, forward


def normalize_camera(frame_truth: dict[str, Any]) -> CameraNormalization:
    """Build the pose consumed by Flight from one Unreal camera truth record."""
    camera = frame_truth["camera"]
    location = camera["world_location_m"]
    position_unreal = np.asarray(
        (location["x"], location["y"], location["z"]), dtype=float
    )
    right, up, forward = _unreal_camera_axes(camera["world_rotation_deg"])
    unreal_from_camera_cv = np.column_stack((right, -up, forward))
    ned_from_camera_cv = UNREAL_TO_NED @ unreal_from_camera_cv
    if not np.allclose(ned_from_camera_cv.T @ ned_from_camera_cv, np.eye(3), atol=1e-9):
        raise ReferenceEvaluationError("dataset camera rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(ned_from_camera_cv)), 1.0, abs_tol=1e-9):
        raise ReferenceEvaluationError("dataset camera normalization is not a proper rotation")
    ned_from_body = ned_from_camera_cv @ BODY_FROM_CAMERA.T
    return CameraNormalization(
        position_local_ned_m=tuple(float(value) for value in UNREAL_TO_NED @ position_unreal),
        rotation_ned_from_camera_cv=ned_from_camera_cv,
        attitude_body_to_ned_wxyz=_rotation_matrix_to_quaternion_wxyz(ned_from_body),
    )


def _frame_time_ns(frame_truth: dict[str, Any]) -> int:
    flight = frame_truth.get("flight")
    if isinstance(flight, dict) and flight.get("time_seconds") is not None:
        return int(round(float(flight["time_seconds"]) * 1_000_000_000.0))
    return max(0, int(frame_truth["frame_number"]) - 1) * 33_333_333


def vehicle_state_from_truth(frame_truth: dict[str, Any]) -> VehicleState:
    normalized = normalize_camera(frame_truth)
    flight = frame_truth.get("flight") if isinstance(frame_truth.get("flight"), dict) else {}
    velocity = flight.get("velocity_m_per_second")
    if isinstance(velocity, dict):
        velocity_unreal = np.asarray(
            (velocity["x"], velocity["y"], velocity["z"]), dtype=float
        )
        velocity_ned = tuple(float(value) for value in UNREAL_TO_NED @ velocity_unreal)
    else:
        velocity_ned = (0.0, 0.0, 0.0)
    sim_time_ns = _frame_time_ns(frame_truth)
    return VehicleState(
        sim_time_ns=sim_time_ns,
        position_local_ned_m=normalized.position_local_ned_m,
        velocity_local_ned_mps=velocity_ned,
        attitude_quaternion=normalized.attitude_body_to_ned_wxyz,
        body_rates_frd_rps=(0.0, 0.0, 0.0),
        acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
        elapsed_time_ns=sim_time_ns,
    )


def _xyz(payload: dict[str, Any]) -> np.ndarray:
    return np.asarray((payload["x"], payload["y"], payload["z"]), dtype=float)


def _truth_camera_cv(gate: dict[str, Any]) -> np.ndarray:
    relative = gate["relative_position_camera_frame_m"]
    return np.asarray(
        (relative["right"], -relative["up"], relative["forward"]), dtype=float
    )


def _truth_gate_ned(gate: dict[str, Any]) -> np.ndarray:
    return UNREAL_TO_NED @ _xyz(gate["world_location_m"])


def validate_coordinate_normalization(
    frame_truth: dict[str, Any],
    intrinsics: dict[str, Any],
) -> dict[str, float]:
    """Verify normalized positions and pinhole projection against dataset truth."""
    normalized = normalize_camera(frame_truth)
    camera_position = np.asarray(normalized.position_local_ned_m)
    position_residuals = []
    projection_residuals = []
    fx = float(intrinsics["focal_x_px"])
    fy = float(intrinsics["focal_y_px"])
    cx = float(intrinsics["principal_x_px"])
    cy = float(intrinsics["principal_y_px"])
    for gate in frame_truth.get("gates", ()):
        camera_cv = _truth_camera_cv(gate)
        predicted_ned = camera_position + normalized.rotation_ned_from_camera_cv @ camera_cv
        position_residuals.append(float(np.linalg.norm(predicted_ned - _truth_gate_ned(gate))))
        for family in ("outer", "inner"):
            corners = gate.get("corners", {}).get(family, {})
            for corner in corners.values():
                camera = corner.get("camera_frame_m")
                pixel = corner.get("pixel_px")
                if not isinstance(camera, dict) or not isinstance(pixel, dict):
                    continue
                forward = float(camera["x"])
                if forward <= 0.0:
                    continue
                projected = np.asarray((
                    cx + fx * float(camera["y"]) / forward,
                    cy - fy * float(camera["z"]) / forward,
                ))
                expected = np.asarray((pixel["x"], pixel["y"]), dtype=float)
                projection_residuals.append(float(np.linalg.norm(projected - expected)))
    result = {
        "max_position_residual_m": max(position_residuals, default=0.0),
        "max_projection_residual_px": max(projection_residuals, default=0.0),
    }
    if result["max_position_residual_m"] > COORDINATE_POSITION_TOLERANCE_M:
        raise ReferenceEvaluationError(
            "Unreal-to-NED position validation failed: "
            f"{result['max_position_residual_m']:.9f} m"
        )
    if result["max_projection_residual_px"] > COORDINATE_PROJECTION_TOLERANCE_PX:
        raise ReferenceEvaluationError(
            "camera projection validation failed: "
            f"{result['max_projection_residual_px']:.9f} px"
        )
    return result


def _resolve_dataset(dataset: str | Path) -> Path:
    value = Path(dataset)
    candidates = [value]
    if not value.is_absolute():
        candidates.append(DEFAULT_DATASET_ROOT / value)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "camera_intrinsics.json").is_file():
            return candidate.resolve()
    raise ReferenceEvaluationError(f"reference dataset not found: {dataset}")


def _resolve_runs(dataset_dir: Path, requested: Iterable[str] | None) -> list[Path]:
    runs_root = dataset_dir / "runs"
    if requested:
        runs = []
        for run_id in requested:
            run = (runs_root / run_id).resolve()
            if not run.is_relative_to(runs_root.resolve()) or not run.is_dir():
                raise ReferenceEvaluationError(f"dataset run not found: {run_id}")
            runs.append(run)
        return runs
    return sorted(path for path in runs_root.glob("run_*") if path.is_dir())


def _iter_run_truth(run_dir: Path) -> Iterable[dict[str, Any]]:
    metadata_path = run_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise ReferenceEvaluationError(f"run has no metadata.jsonl: {run_dir}")
    previous_frame = 0
    with metadata_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
                frame_number = int(frame["frame_number"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ReferenceEvaluationError(
                    f"invalid truth record at {metadata_path}:{line_number}"
                ) from exc
            if frame_number <= previous_frame:
                raise ReferenceEvaluationError(
                    f"non-monotonic frame number at {metadata_path}:{line_number}"
                )
            previous_frame = frame_number
            yield frame


def _confined_file(run_dir: Path, relative_value: Any, field: str) -> Path:
    relative = Path(str(relative_value))
    if relative.is_absolute() or not relative.parts:
        raise ReferenceEvaluationError(f"{field} must be relative: {relative}")
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()):
        raise ReferenceEvaluationError(f"{field} escapes run: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _observation_payload(observation: VisionObservation) -> dict[str, Any]:
    payload = asdict(observation)
    payload["controller_payload"] = observation.to_controller_payload()
    return payload


def _detection_payload(detection: Any, observation: Any | None) -> dict[str, Any]:
    geometry = detection.geometry
    return {
        "track_id": detection.track.track_id,
        "ellipse_id": geometry.ellipse.ellipse_id,
        "consecutive_frame_count": detection.track.consecutive_frame_count,
        "iou_score": detection.track.iou_score,
        "center_px": list(geometry.ellipse.center_px),
        "semi_axes_px": list(geometry.ellipse.semi_axes_px),
        "angle_degrees": geometry.ellipse.angle_degrees,
        "outer_corners_px": {
            corner.role.removeprefix("outer_"): list(corner.point_px)
            for corner in geometry.outer_corners
        },
        "inner_corners_px": {
            corner.role.removeprefix("inner_"): list(corner.point_px)
            for corner in geometry.inner_corners
        },
        "published_gate_id": None if observation is None else observation.gate_id,
        "observation": None if observation is None else asdict(observation),
    }


def _run_backend(
    backend: Any,
    frame: VisionFrame,
    vehicle_state: VehicleState,
) -> tuple[dict[str, Any], Any | None]:
    started = time.perf_counter_ns()
    try:
        detection_frame = backend._detector.process_frame(frame, vehicle_state)
        observation = backend._publisher.observe(detection_frame, vehicle_state)
    except Exception as exc:  # noqa: BLE001 - evaluation continues per backend/frame.
        return {
            "status": "error",
            "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "detections": [],
            "observation": None,
        }, None
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    observation_by_track = {
        gate.trace.get("track_id"): gate
        for gate in observation.gates
        if isinstance(gate.trace, dict) and gate.trace.get("track_id")
    }
    detections = [
        _detection_payload(item, observation_by_track.get(item.track.track_id))
        for item in detection_frame.detections
    ]
    return {
        "status": "ok",
        "elapsed_ms": elapsed_ms,
        "detections": detections,
        "observation": _observation_payload(observation),
    }, detection_frame


def _is_visible_truth(gate: dict[str, Any]) -> bool:
    pixels = gate.get("visible_mask_pixels")
    return (
        gate.get("visible_in_frame") is True
        and isinstance(pixels, dict)
        and int(pixels.get("pixel_count", 0)) > 0
    )


def _truth_center_px(gate: dict[str, Any]) -> np.ndarray:
    inner = gate.get("corners", {}).get("inner", {})
    points = [
        (float(corner["pixel_px"]["x"]), float(corner["pixel_px"]["y"]))
        for corner in inner.values()
        if isinstance(corner.get("pixel_px"), dict)
    ]
    if points:
        return np.mean(np.asarray(points, dtype=float), axis=0)
    bbox = gate["bbox_2d_px"]
    return np.asarray((
        0.5 * (float(bbox["x_min_px"]) + float(bbox["x_max_px"])),
        0.5 * (float(bbox["y_min_px"]) + float(bbox["y_max_px"])),
    ))


def _truth_bbox(gate: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = gate["bbox_2d_px"]
    return tuple(float(bbox[key]) for key in (
        "x_min_px", "y_min_px", "x_max_px", "y_max_px"
    ))


def _match_threshold_px(gate: dict[str, Any]) -> float:
    x0, y0, x1, y1 = _truth_bbox(gate)
    diagonal = math.hypot(x1 - x0, y1 - y0)
    return min(
        MATCH_DISTANCE_CEILING_PX,
        max(MATCH_DISTANCE_FLOOR_PX, MATCH_BBOX_DIAGONAL_FRACTION * diagonal),
    )


def match_detections(
    gates: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> tuple[dict[int, int], list[int], list[int]]:
    """Globally match detection centers to visible truth with bounded distance."""
    if not gates or not detections:
        return {}, list(range(len(gates))), list(range(len(detections)))
    costs = np.full((len(gates), len(detections)), 1e9, dtype=float)
    for gate_index, gate in enumerate(gates):
        center = _truth_center_px(gate)
        threshold = _match_threshold_px(gate)
        for detection_index, detection in enumerate(detections):
            distance = float(np.linalg.norm(
                center - np.asarray(detection["center_px"], dtype=float)
            ))
            if distance <= threshold:
                costs[gate_index, detection_index] = distance
    rows, columns = linear_sum_assignment(costs)
    matches = {
        int(gate_index): int(detection_index)
        for gate_index, detection_index in zip(rows, columns)
        if costs[gate_index, detection_index] < 1e8
    }
    unmatched_gates = [index for index in range(len(gates)) if index not in matches]
    matched_detections = set(matches.values())
    unmatched_detections = [
        index for index in range(len(detections))
        if index not in matched_detections
    ]
    return matches, unmatched_gates, unmatched_detections


def _visibility_bucket(gate: dict[str, Any]) -> str:
    fraction = float(gate.get("visibility_fraction", 0.0))
    if fraction < 0.25:
        return "lt_0.25"
    if fraction < 0.50:
        return "0.25_to_0.50"
    if fraction < 0.75:
        return "0.50_to_0.75"
    return "gte_0.75"


def _pixel_bucket(gate: dict[str, Any]) -> str:
    pixels = int(gate.get("visible_mask_pixels", {}).get("pixel_count", 0))
    if pixels < 100:
        return "lt_100"
    if pixels < 500:
        return "100_to_499"
    if pixels < 2000:
        return "500_to_1999"
    return "gte_2000"


def _occlusion_bucket(gate: dict[str, Any]) -> str:
    occlusion_type = str(gate.get("occlusion_type") or "none")
    if occlusion_type not in {"none", "edge_clipped"}:
        return f"occluded:{occlusion_type}"
    if occlusion_type == "edge_clipped":
        return "edge_clipped"
    return "none"


def _view_angle_bucket(gate: dict[str, Any]) -> str:
    orientation = gate.get("relative_orientation_euler_deg")
    if not isinstance(orientation, dict):
        return "unknown"
    angle = max(
        abs(float(orientation.get("yaw_deg", 0.0))),
        abs(float(orientation.get("pitch_deg", 0.0))),
    )
    if angle < 30.0:
        return "frontal_lt_30_deg"
    if angle < 60.0:
        return "oblique_30_to_60_deg"
    return "high_angle_gte_60_deg"


def mask_iou_for_gate(
    prediction_mask: np.ndarray,
    truth_mask: np.ndarray,
    gate: dict[str, Any],
) -> float:
    """Score binary gate segmentation inside the gate's projected evaluation box."""
    if prediction_mask.shape != truth_mask.shape:
        raise ValueError("prediction and truth masks must have equal shape")
    height, width = truth_mask.shape
    x0, y0, x1, y1 = _truth_bbox(gate)
    left = max(0, int(math.floor(x0)) - 2)
    top = max(0, int(math.floor(y0)) - 2)
    right = min(width, int(math.ceil(x1)) + 3)
    bottom = min(height, int(math.ceil(y1)) + 3)
    if left >= right or top >= bottom:
        return 0.0
    stencil = int(gate["custom_depth_stencil_value"])
    truth = truth_mask[top:bottom, left:right] == stencil
    prediction = prediction_mask[top:bottom, left:right] > 0
    union = int(np.count_nonzero(truth | prediction))
    return float(np.count_nonzero(truth & prediction) / union) if union else 0.0


def _corner_error(
    gate: dict[str, Any],
    detection: dict[str, Any],
    family: str,
) -> float | None:
    predicted = detection[f"{family}_corners_px"]
    truth = gate.get("corners", {}).get(family, {})
    errors = []
    for predicted_role, truth_role in CORNER_ROLE_MAP.items():
        point = predicted.get(predicted_role)
        expected = truth.get(truth_role, {}).get("pixel_px")
        if point is None or not isinstance(expected, dict):
            return None
        errors.append(math.hypot(
            float(point[0]) - float(expected["x"]),
            float(point[1]) - float(expected["y"]),
        ))
    return float(np.mean(errors)) if errors else None


def _pose_errors(
    gate: dict[str, Any],
    detection: dict[str, Any],
) -> dict[str, float] | None:
    observation = detection.get("observation")
    if not isinstance(observation, dict):
        return None
    trace = observation.get("trace")
    camera_position = trace.get("position_camera_cv_m") if isinstance(trace, dict) else None
    local_position = observation.get("position_local_ned")
    if (
        not isinstance(camera_position, (list, tuple))
        or len(camera_position) != 3
        or not isinstance(local_position, (list, tuple))
        or len(local_position) != 3
    ):
        return None
    predicted_camera = np.asarray(camera_position, dtype=float)
    truth_camera = _truth_camera_cv(gate)
    predicted_local = np.asarray(local_position, dtype=float)
    truth_local = _truth_gate_ned(gate)
    return {
        "camera_position_error_m": float(np.linalg.norm(predicted_camera - truth_camera)),
        "local_position_error_m": float(np.linalg.norm(predicted_local - truth_local)),
        "depth_error_m": abs(float(predicted_camera[2] - truth_camera[2])),
    }


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _new_backend_accumulator() -> dict[str, Any]:
    return {
        "ok": 0,
        "error": 0,
        "elapsed_ms": [],
        "truth_visible": 0,
        "detected": 0,
        "published": 0,
        "false_positive_detections": 0,
        "false_positive_publications": 0,
        "by_gate": defaultdict(lambda: {"truth": 0, "detected": 0, "published": 0}),
        "by_visibility": defaultdict(lambda: {"truth": 0, "detected": 0, "published": 0}),
        "by_pixels": defaultdict(lambda: {"truth": 0, "detected": 0, "published": 0}),
        "by_occlusion": defaultdict(lambda: {"truth": 0, "detected": 0, "published": 0}),
        "by_view_angle": defaultdict(lambda: {"truth": 0, "detected": 0, "published": 0}),
        "mask_iou": [],
        "mask_iou_by_gate": defaultdict(list),
        "outer_corner_error_px": [],
        "inner_corner_error_px": [],
        "camera_position_error_m": [],
        "local_position_error_m": [],
        "depth_error_m": [],
        "pose_by_occlusion": defaultdict(lambda: {
            "camera_position_error_m": [],
            "local_position_error_m": [],
            "depth_error_m": [],
        }),
        "zero_visible_frames": 0,
        "zero_visible_frame_false_positives": 0,
        "track_history": defaultdict(list),
    }


def _recall_group(payload: dict[str, int]) -> dict[str, int | float | None]:
    truth = int(payload["truth"])
    return {
        **payload,
        "detection_recall": None if not truth else float(payload["detected"] / truth),
        "publication_recall": None if not truth else float(payload["published"] / truth),
    }


def _tracking_summary(history: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, Any]:
    per_gate = []
    promotion_visible_frames = []
    promotion_detection_frames = []
    censored_segments = 0
    for (run_id, label), records in sorted(history.items()):
        detector_tracks = [record["track_id"] for record in records if record["track_id"]]
        published_tracks = [
            record["published_gate_id"]
            for record in records
            if record["published_gate_id"]
        ]
        detector_switches = sum(
            left != right
            for left, right in zip(detector_tracks, detector_tracks[1:])
        )
        published_switches = sum(
            left != right
            for left, right in zip(published_tracks, published_tracks[1:])
        )
        unique_detector_tracks = sorted(set(detector_tracks))
        unique_published_tracks = sorted(set(published_tracks))
        segments = []
        current = []
        for record in records:
            if current and record["frame_index"] != current[-1]["frame_index"] + 1:
                segments.append(current)
                current = []
            current.append(record)
        if current:
            segments.append(current)
        for segment in segments:
            published = next(
                (item for item in segment if item["published_gate_id"]), None
            )
            first_detected = next(
                (item for item in segment if item["track_id"]), None
            )
            if published is None:
                censored_segments += 1
                continue
            promotion_visible_frames.append(
                published["frame_index"] - segment[0]["frame_index"]
            )
            if first_detected is not None:
                promotion_detection_frames.append(
                    published["frame_index"] - first_detected["frame_index"]
                )
        per_gate.append({
            "run_id": run_id,
            "gate_label": label,
            "visible_frames": len(records),
            "detector_matched_frames": len(detector_tracks),
            "publisher_matched_frames": len(published_tracks),
            "unique_detector_track_ids": unique_detector_tracks,
            "unique_published_gate_ids": unique_published_tracks,
            "detector_id_switches": detector_switches,
            "publisher_id_switches": published_switches,
            "detector_fragmentation_count": max(0, len(unique_detector_tracks) - 1),
            "publisher_fragmentation_count": max(0, len(unique_published_tracks) - 1),
        })
    return {
        "gate_sequences": len(per_gate),
        "publisher_id_switches": sum(
            item["publisher_id_switches"] for item in per_gate
        ),
        "publisher_fragmentation_count": sum(
            item["publisher_fragmentation_count"] for item in per_gate
        ),
        "detector_id_switches": sum(
            item["detector_id_switches"] for item in per_gate
        ),
        "detector_fragmentation_count": sum(
            item["detector_fragmentation_count"] for item in per_gate
        ),
        "publication_latency_from_first_visible_frames": _distribution(
            promotion_visible_frames
        ),
        "publication_latency_from_first_detection_frames": _distribution(
            promotion_detection_frames
        ),
        "censored_visibility_segments": censored_segments,
        "latency_scope": (
            "Vision backend publication latency; the global GateMap was not "
            "part of this dataset evaluation."
        ),
        "per_gate": per_gate,
    }


def _finalize_backend(accumulator: dict[str, Any], processed_frames: int) -> dict[str, Any]:
    truth = int(accumulator["truth_visible"])
    elapsed = _distribution(accumulator["elapsed_ms"])
    mean_ms = elapsed["mean"]
    return {
        "ok": accumulator["ok"],
        "error": accumulator["error"],
        "truth_visible_gate_instances": truth,
        "detected_gate_instances": accumulator["detected"],
        "published_gate_instances": accumulator["published"],
        "detection_recall": None if not truth else accumulator["detected"] / truth,
        "publication_recall": None if not truth else accumulator["published"] / truth,
        "false_positive_detections": accumulator["false_positive_detections"],
        "false_positive_detections_per_frame": (
            accumulator["false_positive_detections"] / max(1, processed_frames)
        ),
        "false_positive_publications": accumulator["false_positive_publications"],
        "false_positive_publications_per_frame": (
            accumulator["false_positive_publications"] / max(1, processed_frames)
        ),
        "zero_visible_gate_frames": accumulator["zero_visible_frames"],
        "zero_visible_frame_false_positives": (
            accumulator["zero_visible_frame_false_positives"]
        ),
        "zero_visible_frame_false_positives_per_frame": (
            accumulator["zero_visible_frame_false_positives"]
            / max(1, accumulator["zero_visible_frames"])
        ),
        "runtime_ms": elapsed,
        "throughput_hz_from_mean_latency": (
            None if mean_ms in (None, 0.0) else 1000.0 / float(mean_ms)
        ),
        "mask_iou": _distribution(accumulator["mask_iou"]),
        "mask_iou_by_gate": {
            key: _distribution(values)
            for key, values in sorted(accumulator["mask_iou_by_gate"].items())
        },
        "outer_corner_error_px": _distribution(accumulator["outer_corner_error_px"]),
        "inner_corner_error_px": _distribution(accumulator["inner_corner_error_px"]),
        "camera_position_error_m": _distribution(accumulator["camera_position_error_m"]),
        "local_position_error_m": _distribution(accumulator["local_position_error_m"]),
        "depth_error_m": _distribution(accumulator["depth_error_m"]),
        "orientation": {
            "published_count": 0,
            "note": "The evaluated backends do not publish gate orientation.",
        },
        "recall_by_gate": {
            key: _recall_group(dict(value))
            for key, value in sorted(accumulator["by_gate"].items())
        },
        "recall_by_visibility_fraction": {
            key: _recall_group(dict(value))
            for key, value in sorted(accumulator["by_visibility"].items())
        },
        "recall_by_visible_pixels": {
            key: _recall_group(dict(value))
            for key, value in sorted(accumulator["by_pixels"].items())
        },
        "recall_by_occlusion": {
            key: _recall_group(dict(value))
            for key, value in sorted(accumulator["by_occlusion"].items())
        },
        "recall_by_view_angle": {
            key: _recall_group(dict(value))
            for key, value in sorted(accumulator["by_view_angle"].items())
        },
        "pose_error_by_occlusion": {
            key: {
                metric: _distribution(values)
                for metric, values in payload.items()
            }
            for key, payload in sorted(accumulator["pose_by_occlusion"].items())
        },
        "tracking": _tracking_summary(accumulator["track_history"]),
    }


def _update_group(group: dict[str, int], *, detected: bool, published: bool) -> None:
    group["truth"] += 1
    group["detected"] += int(detected)
    group["published"] += int(published)


def _score_backend_frame(
    *,
    run_id: str,
    frame_index: int,
    visible_gates: list[dict[str, Any]],
    result: dict[str, Any],
    prediction_mask: np.ndarray | None,
    truth_mask: np.ndarray,
    accumulator: dict[str, Any],
) -> dict[str, Any]:
    if result.get("elapsed_ms") is not None:
        accumulator["elapsed_ms"].append(float(result["elapsed_ms"]))
    if result["status"] != "ok":
        accumulator["error"] += 1
        accumulator["truth_visible"] += len(visible_gates)
        if not visible_gates:
            accumulator["zero_visible_frames"] += 1
        for gate in visible_gates:
            _update_group(
                accumulator["by_gate"][str(gate["label"])],
                detected=False,
                published=False,
            )
            _update_group(
                accumulator["by_visibility"][_visibility_bucket(gate)],
                detected=False,
                published=False,
            )
            _update_group(
                accumulator["by_pixels"][_pixel_bucket(gate)],
                detected=False,
                published=False,
            )
            _update_group(
                accumulator["by_occlusion"][_occlusion_bucket(gate)],
                detected=False,
                published=False,
            )
            _update_group(
                accumulator["by_view_angle"][_view_angle_bucket(gate)],
                detected=False,
                published=False,
            )
            accumulator["track_history"][(run_id, str(gate["label"]))].append({
                "frame_index": frame_index,
                "track_id": None,
                "published_gate_id": None,
            })
        return {
            "status": "error",
            "matches": [],
            "missed_truth_labels": [gate["label"] for gate in visible_gates],
            "false_positive_detection_indices": [],
            "false_positive_publication_count": 0,
        }
    accumulator["ok"] += 1
    detections = result["detections"]
    matches, missed, false_positive_indices = match_detections(visible_gates, detections)
    published_indices = {
        index for index, item in enumerate(detections)
        if item.get("published_gate_id") is not None
    }
    matched_detection_indices = set(matches.values())
    false_positive_publications = len(published_indices - matched_detection_indices)
    accumulator["truth_visible"] += len(visible_gates)
    accumulator["detected"] += len(matches)
    accumulator["published"] += sum(
        detections[detection_index].get("published_gate_id") is not None
        for detection_index in matches.values()
    )
    accumulator["false_positive_detections"] += len(false_positive_indices)
    accumulator["false_positive_publications"] += false_positive_publications
    if not visible_gates:
        accumulator["zero_visible_frames"] += 1
        accumulator["zero_visible_frame_false_positives"] += len(
            false_positive_indices)
    match_records = []
    for gate_index, gate in enumerate(visible_gates):
        detection_index = matches.get(gate_index)
        detection = None if detection_index is None else detections[detection_index]
        detected = detection is not None
        published = detected and detection.get("published_gate_id") is not None
        _update_group(
            accumulator["by_gate"][str(gate["label"])],
            detected=detected,
            published=published,
        )
        _update_group(
            accumulator["by_visibility"][_visibility_bucket(gate)],
            detected=detected,
            published=published,
        )
        _update_group(
            accumulator["by_pixels"][_pixel_bucket(gate)],
            detected=detected,
            published=published,
        )
        _update_group(
            accumulator["by_occlusion"][_occlusion_bucket(gate)],
            detected=detected,
            published=published,
        )
        _update_group(
            accumulator["by_view_angle"][_view_angle_bucket(gate)],
            detected=detected,
            published=published,
        )
        iou = (
            None if prediction_mask is None
            else mask_iou_for_gate(prediction_mask, truth_mask, gate)
        )
        if iou is not None:
            accumulator["mask_iou"].append(iou)
            accumulator["mask_iou_by_gate"][str(gate["label"])].append(iou)
        outer_error = None if detection is None else _corner_error(gate, detection, "outer")
        inner_error = None if detection is None else _corner_error(gate, detection, "inner")
        if outer_error is not None:
            accumulator["outer_corner_error_px"].append(outer_error)
        if inner_error is not None:
            accumulator["inner_corner_error_px"].append(inner_error)
        pose = None if detection is None else _pose_errors(gate, detection)
        if pose is not None:
            for key, value in pose.items():
                accumulator[key].append(value)
                accumulator["pose_by_occlusion"][_occlusion_bucket(gate)][key].append(
                    value)
        accumulator["track_history"][(run_id, str(gate["label"]))].append({
            "frame_index": frame_index,
            "track_id": None if detection is None else detection["track_id"],
            "published_gate_id": (
                None if detection is None else detection.get("published_gate_id")
            ),
        })
        match_records.append({
            "truth_index": gate_index,
            "truth_label": gate["label"],
            "detection_index": detection_index,
            "track_id": None if detection is None else detection["track_id"],
            "published_gate_id": (
                None if detection is None else detection.get("published_gate_id")
            ),
            "mask_iou": iou,
            "outer_corner_error_px": outer_error,
            "inner_corner_error_px": inner_error,
            "pose_errors": pose,
        })
    return {
        "status": "ok",
        "matches": match_records,
        "missed_truth_labels": [visible_gates[index]["label"] for index in missed],
        "false_positive_detection_indices": false_positive_indices,
        "false_positive_publication_count": false_positive_publications,
    }


def _truth_summary(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": gate["label"],
        "stencil_value": gate["custom_depth_stencil_value"],
        "visible_in_frame": gate.get("visible_in_frame"),
        "geometric_visibility": gate.get("geometric_visibility"),
        "visible_mask_pixels": gate.get("visible_mask_pixels", {}).get("pixel_count"),
        "visibility_fraction": gate.get("visibility_fraction"),
        "occlusion_fraction": gate.get("occlusion_fraction"),
        "occlusion_type": gate.get("occlusion_type"),
        "occluder_label": gate.get("occluder_label"),
        "relative_orientation_euler_deg": gate.get(
            "relative_orientation_euler_deg"),
        "center_px": _truth_center_px(gate),
        "bbox_2d_px": gate.get("bbox_2d_px"),
        "position_camera_cv_m": _truth_camera_cv(gate),
        "position_local_ned_m": _truth_gate_ned(gate),
        "corners": gate.get("corners"),
    }


def _draw_polyline(image: np.ndarray, points: list[list[float]], color: tuple[int, int, int]) -> None:
    if len(points) != 4:
        return
    polygon = np.asarray(points, dtype=float).round().astype(np.int32)
    cv2.polylines(image, [polygon], True, color, 2, cv2.LINE_AA)


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.38,
) -> None:
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, 1, cv2.LINE_AA,
    )


def _render_overlay(
    image: np.ndarray,
    visible_gates: list[dict[str, Any]],
    result: dict[str, Any],
    score: dict[str, Any],
) -> np.ndarray:
    overlay = image.copy()
    matches = {
        int(record["truth_index"]): record
        for record in score.get("matches", ())
        if record.get("detection_index") is not None
    }
    for gate_index, gate in enumerate(visible_gates):
        points = [
            [
                gate["corners"]["outer"][role]["pixel_px"]["x"],
                gate["corners"]["outer"][role]["pixel_px"]["y"],
            ]
            for role in CORNER_ORDER
        ]
        matched = gate_index in matches
        color = (40, 210, 40) if matched else (30, 30, 235)
        _draw_polyline(overlay, points, color)
        _draw_text(
            overlay,
            f"T {gate['label']} {'detected' if matched else 'missed'}",
            (8, 16 + 14 * gate_index),
            color,
        )
    for index, detection in enumerate(result.get("detections", ())):
        color = (235, 180, 35)
        corners = detection.get("outer_corners_px", {})
        points = [
            corners[role]
            for role in ("upper_left", "upper_right", "lower_right", "lower_left")
            if role in corners
        ]
        _draw_polyline(overlay, points, color)
        center = tuple(round(float(value)) for value in detection["center_px"])
        cv2.drawMarker(overlay, center, color, cv2.MARKER_CROSS, 10, 2)
        _draw_text(overlay, f"P{index}", (center[0] + 4, center[1] + 12), color)
    return overlay


def _overlay_filename(frame_id: int, sim_time_ns: int) -> str:
    return f"frame-{frame_id:08d}-{sim_time_ns}.png"


def evaluate_dataset(
    dataset: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    run_ids: Iterable[str] | None = None,
    start: int = 0,
    limit: int | None = None,
    max_runs: int | None = None,
    render_overlays: bool = False,
    verbose: bool = True,
) -> Path:
    if start < 0:
        raise ValueError("start must be zero or greater")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")

    def status(message: str) -> None:
        if verbose:
            print(f"[reference-eval] {message}", flush=True)

    dataset_dir = _resolve_dataset(dataset)
    intrinsics_payload = _read_json(dataset_dir / "camera_intrinsics.json")
    intrinsics = intrinsics_payload["camera"]["intrinsics"]
    runs = _resolve_runs(dataset_dir, run_ids)
    if max_runs is not None:
        runs = runs[:max_runs]
    if not runs:
        raise ReferenceEvaluationError("no dataset runs selected")
    dataset_id = str(intrinsics_payload.get("dataset_id", dataset_dir.name))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    selection_id = runs[0].name if len(runs) == 1 else f"{len(runs)}-runs"
    output_dir = Path(output_root) / (
        f"evaluation-{timestamp}-{dataset_id}-{selection_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    frames_path = output_dir / "frames.jsonl"
    accumulators = {
        name: _new_backend_accumulator()
        for name in BACKEND_FACTORIES
    }
    processed_frames = 0
    frames_changed = 0
    frames_with_errors = 0
    zero_visible_frames = 0
    multi_visible_frames = 0
    coordinate_position_residuals = []
    coordinate_projection_residuals = []
    run_summaries = []

    _write_json(output_dir / "metadata.json", {
        "evaluation_format_version": 2,
        "evaluation_kind": "unreal_reference_dataset",
        "created_utc": timestamp,
        "source_dataset": {
            "dataset_id": dataset_id,
            "path": str(dataset_dir),
            "camera_intrinsics": intrinsics_payload,
            "run_ids": [run.name for run in runs],
        },
        "source_run": {
            "run_id": selection_id,
            "path": str(dataset_dir / "runs"),
        },
        "selection": {
            "run_count": len(runs),
            "start_per_run": start,
            "limit_per_run": limit,
            "selected_frame_count": None,
        },
        "coordinate_normalization": {
            "unreal_world_xyz_to_local_ned": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            "dataset_camera_to_opencv": {
                "input_axes": ["right", "up", "forward"],
                "output_axes": ["right", "down=-up", "forward"],
            },
            "position_validation_tolerance_m": COORDINATE_POSITION_TOLERANCE_M,
            "projection_validation_tolerance_px": COORDINATE_PROJECTION_TOLERANCE_PX,
        },
        "backends": list(BACKEND_FACTORIES),
        "overlays": {
            "enabled": render_overlays,
            "layers": ["truth_prediction", "prediction_mask"] if render_overlays else [],
            "root": "overlays" if render_overlays else None,
        },
        "output_files": {
            "frames": "frames.jsonl",
            "summary": "summary.json",
        },
    })

    status(f"dataset={dataset_dir} runs={len(runs)}")
    for run_number, run_dir in enumerate(runs, 1):
        truths = list(_iter_run_truth(run_dir))
        selected = truths[start:start + limit if limit is not None else None]
        backends = {
            name: factory(run_id=f"evaluation-{run_dir.name}")
            for name, factory in BACKEND_FACTORIES.items()
        }
        run_processed = 0
        status(
            f"run {run_number}/{len(runs)} {run_dir.name}: "
            f"{len(selected)}/{len(truths)} frames"
        )
        for offset, frame_truth in enumerate(selected):
            frame_index = start + offset
            frame_id = int(frame_truth["frame_number"])
            sim_time_ns = _frame_time_ns(frame_truth)
            frame_path = _confined_file(
                run_dir, frame_truth["frame_path_relative_to_run"], "frame path"
            )
            mask_path = _confined_file(
                run_dir, frame_truth["mask_path_relative_to_run"], "mask path"
            )
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            truth_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if image is None or truth_mask is None:
                raise ReferenceEvaluationError(f"unable to decode frame assets: {frame_path}")
            if image.shape[:2] != truth_mask.shape:
                raise ReferenceEvaluationError(f"frame/mask shape mismatch: {frame_path}")
            validation = validate_coordinate_normalization(frame_truth, intrinsics)
            coordinate_position_residuals.append(validation["max_position_residual_m"])
            coordinate_projection_residuals.append(validation["max_projection_residual_px"])
            vehicle_state = vehicle_state_from_truth(frame_truth)
            frame = VisionFrame(
                frame_id=frame_id,
                sim_time_ns=sim_time_ns,
                jpeg_bytes=b"",
                image=image,
                saved_path=frame_truth["frame_path_relative_to_run"],
                elapsed_time_ns=sim_time_ns,
            )
            visible_gates = [
                gate for gate in frame_truth.get("gates", ())
                if _is_visible_truth(gate)
            ]
            zero_visible_frames += int(not visible_gates)
            multi_visible_frames += int(len(visible_gates) > 1)
            backend_results = {}
            backend_scores = {}
            prediction_masks = {}
            for name, backend in backends.items():
                result, _ = _run_backend(backend, frame, vehicle_state)
                backend_results[name] = result
                prediction_mask = None
                if result["status"] == "ok":
                    module = BACKEND_DETECTION_MODULES[name]
                    prediction_mask = module.image_to_mask(image, backend._detector.lut)
                prediction_masks[name] = prediction_mask
                backend_scores[name] = _score_backend_frame(
                    run_id=run_dir.name,
                    frame_index=frame_index,
                    visible_gates=visible_gates,
                    result=result,
                    prediction_mask=prediction_mask,
                    truth_mask=truth_mask,
                    accumulator=accumulators[name],
                )
            left_score = backend_scores["deterministic_v3"]
            right_score = backend_scores["deterministic_v3_2"]
            left_detected = sum(
                item["detection_index"] is not None
                for item in left_score.get("matches", ())
            )
            right_detected = sum(
                item["detection_index"] is not None
                for item in right_score.get("matches", ())
            )
            changed = (
                left_detected != right_detected
                or len(left_score.get("missed_truth_labels", ()))
                != len(right_score.get("missed_truth_labels", ()))
                or len(left_score.get("false_positive_detection_indices", ()))
                != len(right_score.get("false_positive_detection_indices", ()))
                or any(
                    backend_results[name]["status"] != "ok"
                    for name in BACKEND_FACTORIES
                )
            )
            frames_changed += int(changed)
            frame_error = any(
                backend_results[name]["status"] != "ok"
                for name in BACKEND_FACTORIES
            )
            frames_with_errors += int(frame_error)
            source = {
                "dataset_id": dataset_id,
                "run_id": run_dir.name,
                "frame_index": frame_index,
                "frame_id": frame_id,
                "sim_time_ns": sim_time_ns,
                "relative_path": frame_truth["frame_path_relative_to_run"],
                "mask_relative_path": frame_truth["mask_path_relative_to_run"],
                "vehicle_state": vehicle_state,
            }
            _append_jsonl(frames_path, {
                "source": source,
                "truth": {
                    "known_gate_count": len(frame_truth.get("gates", ())),
                    "visible_gate_count": len(visible_gates),
                    "visible_gates": [_truth_summary(gate) for gate in visible_gates],
                    "coordinate_validation": validation,
                },
                "backends": {
                    name: {
                        **backend_results[name],
                        "score": backend_scores[name],
                    }
                    for name in BACKEND_FACTORIES
                },
                "comparison": {
                    "changed": changed,
                    "detected_gate_count_delta": right_detected - left_detected,
                    "gate_count_delta": (
                        len(backend_results["deterministic_v3_2"].get("detections", ()))
                        - len(backend_results["deterministic_v3"].get("detections", ()))
                    ),
                    "published_gate_count_delta": (
                        sum(
                            item.get("published_gate_id") is not None
                            for item in backend_results["deterministic_v3_2"].get(
                                "detections", ()
                            )
                        )
                        - sum(
                            item.get("published_gate_id") is not None
                            for item in backend_results["deterministic_v3"].get(
                                "detections", ()
                            )
                        )
                    ),
                },
            })
            if render_overlays:
                filename = _overlay_filename(frame_id, sim_time_ns)
                for name in BACKEND_FACTORIES:
                    overlay = _render_overlay(
                        image, visible_gates, backend_results[name], backend_scores[name]
                    )
                    overlay_path = (
                        output_dir / "overlays" / name / "truth_prediction" / filename
                    )
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(overlay_path), overlay):
                        raise ReferenceEvaluationError(f"unable to write {overlay_path}")
                    mask = prediction_masks[name]
                    if mask is not None:
                        mask_path_out = (
                            output_dir / "overlays" / name / "prediction_mask" / filename
                        )
                        mask_path_out.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(mask_path_out), mask):
                            raise ReferenceEvaluationError(
                                f"unable to write {mask_path_out}"
                            )
            processed_frames += 1
            run_processed += 1
            if verbose and (
                run_processed == 1
                or run_processed == len(selected)
                or run_processed % 25 == 0
            ):
                status(
                    f"{run_dir.name} frame {run_processed}/{len(selected)} "
                    f"truth={len(visible_gates)} v3={left_detected} v3_2={right_detected}"
                )
        run_summaries.append({
            "run_id": run_dir.name,
            "available_frame_count": len(truths),
            "processed_frame_count": run_processed,
        })

    backend_summaries = {
        name: _finalize_backend(accumulator, processed_frames)
        for name, accumulator in accumulators.items()
    }
    left_detection_recall = backend_summaries["deterministic_v3"]["detection_recall"]
    right_detection_recall = backend_summaries["deterministic_v3_2"]["detection_recall"]
    left_publication_recall = backend_summaries["deterministic_v3"]["publication_recall"]
    right_publication_recall = backend_summaries["deterministic_v3_2"]["publication_recall"]
    left_mean_latency = backend_summaries["deterministic_v3"]["runtime_ms"]["mean"]
    right_mean_latency = backend_summaries["deterministic_v3_2"]["runtime_ms"]["mean"]
    summary = {
        "dataset_id": dataset_id,
        "processed_run_count": len(runs),
        "processed_frame_count": processed_frames,
        "frames_changed": frames_changed,
        "frames_with_errors": frames_with_errors,
        "frames_with_vehicle_state": processed_frames,
        "zero_visible_gate_frames": zero_visible_frames,
        "multi_visible_gate_frames": multi_visible_frames,
        "left_backend": "deterministic_v3",
        "right_backend": "deterministic_v3_2",
        "backend_totals": backend_summaries,
        "coordinate_validation": {
            "position_residual_m": _distribution(coordinate_position_residuals),
            "projection_residual_px": _distribution(coordinate_projection_residuals),
        },
        "run_summaries": run_summaries,
        "comparison": {
            "detection_recall_delta": (
                right_detection_recall - left_detection_recall
            ) if (
                right_detection_recall is not None
                and left_detection_recall is not None
            ) else None,
            "publication_recall_delta": (
                right_publication_recall - left_publication_recall
            ) if (
                right_publication_recall is not None
                and left_publication_recall is not None
            ) else None,
            "mean_latency_delta_ms": (
                right_mean_latency - left_mean_latency
            ) if (
                right_mean_latency is not None
                and left_mean_latency is not None
            ) else None,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    metadata = _read_json(output_dir / "metadata.json")
    metadata["selection"]["selected_frame_count"] = processed_frames
    _write_json(output_dir / "metadata.json", metadata)
    status(f"wrote evaluation: {output_dir}")
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic_v3 and deterministic_v3_2 over an Unreal "
            "reference dataset and score detections, geometry, pose, tracking, "
            "and runtime against truth."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="dataset_001",
        help=(
            "Dataset directory or id under VISION_EVALUATION_DATASET_ROOT "
            "(default: dataset_001)."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run",
        action="append",
        dest="run_ids",
        help="Run id to evaluate; repeat to select multiple runs. Defaults to all runs.",
    )
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--render-overlays",
        action="store_true",
        help="Write truth/prediction and segmentation overlays for the evaluation UI.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = evaluate_dataset(
        args.dataset,
        args.output_root,
        run_ids=args.run_ids,
        start=args.start,
        limit=args.limit,
        max_runs=args.max_runs,
        render_overlays=args.render_overlays,
        verbose=not args.quiet,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
