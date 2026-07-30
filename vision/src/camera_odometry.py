from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

FX, FY, CX, CY = 320.0, 320.0, 320.0, 180.0
MAX_TELEMETRY_GAP_S = 0.25
CAMERA_TILT_DEG = 20.0  # camera tilted upward from body; sign verified empirically against real optical flow


def _pitch_matrix(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# Camera CV convention (X=right, Y=down, Z=forward/depth) <-> body FRD (X=forward, Y=right, Z=down),
# composed with the fixed 20-degree upward tilt (a rotation about the body-right/pitch axis).
_BASE_BODY_FROM_CAM = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_BODY_FROM_CAM = _pitch_matrix(CAMERA_TILT_DEG) @ _BASE_BODY_FROM_CAM
_CAM_FROM_BODY = _BODY_FROM_CAM.T  # orthogonal -> transpose is inverse


@dataclass
class TelemetryTrack:
    epochs: np.ndarray
    quats: np.ndarray
    positions: np.ndarray


def load_telemetry(path: str | Path) -> TelemetryTrack:
    data = json.loads(Path(path).read_text())
    epochs, quats, positions = [], [], []
    for sample in data["samples"]:
        vehicle_state = sample["telemetry"]["vehicle_state"]
        epochs.append(datetime.fromisoformat(sample["wall_time_utc"]).timestamp())
        quats.append(vehicle_state["attitude_quaternion"])
        positions.append(vehicle_state["position_local_ned_m"])
    return TelemetryTrack(np.array(epochs), np.array(quats), np.array(positions))


def find_telemetry_for_frame(frame_path: str | Path) -> Path | None:
    candidate = Path(frame_path).parent.parent / "telemetry.json"
    return candidate if candidate.is_file() else None


def frame_epoch_seconds(frame_path: str | Path) -> float:
    stem = Path(frame_path).stem
    return int(stem.rsplit("-", 1)[-1]) / 1e9


def _nearest_index(epochs: np.ndarray, t: float, max_gap_s: float = MAX_TELEMETRY_GAP_S) -> int | None:
    i = bisect.bisect_left(epochs, t)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(epochs)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(epochs[j] - t))
    return best if abs(epochs[best] - t) <= max_gap_s else None


def sample_for_frame(track: TelemetryTrack, frame_path: str | Path, max_gap_s: float = MAX_TELEMETRY_GAP_S):
    index = _nearest_index(track.epochs, frame_epoch_seconds(frame_path), max_gap_s)
    if index is None:
        return None
    return track.quats[index], track.positions[index]


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def relative_camera_rotation(r_bw_a: np.ndarray, r_bw_b: np.ndarray) -> np.ndarray:
    r_rel_body = r_bw_b.T @ r_bw_a
    return _CAM_FROM_BODY @ r_rel_body @ _BODY_FROM_CAM


def predict_pixel(r_cam_rel: np.ndarray, pixel: tuple[float, float]) -> tuple[float, float] | None:
    u, v = pixel
    ray_a = np.array([(u - CX) / FX, (v - CY) / FY, 1.0])
    ray_b = r_cam_rel @ ray_a
    if ray_b[2] <= 1e-6:
        return None
    return CX + FX * ray_b[0] / ray_b[2], CY + FY * ray_b[1] / ray_b[2]


def predict_center(track: TelemetryTrack, frame_a, center_a, frame_b) -> tuple[float, float] | None:
    sample_a = sample_for_frame(track, frame_a)
    sample_b = sample_for_frame(track, frame_b)
    if sample_a is None or sample_b is None:
        return None
    r_rel = relative_camera_rotation(_quat_to_matrix(sample_a[0]), _quat_to_matrix(sample_b[0]))
    return predict_pixel(r_rel, center_a)


def compensated_match(track: TelemetryTrack, frame_a, center_a, frame_b, center_b,
                       gain: float = 0.03, max_scale: float = 2.5) -> tuple[float, tuple[float, float], float] | None:
    sample_a = sample_for_frame(track, frame_a)
    sample_b = sample_for_frame(track, frame_b)
    if sample_a is None or sample_b is None:
        return None
    r_rel = relative_camera_rotation(_quat_to_matrix(sample_a[0]), _quat_to_matrix(sample_b[0]))
    predicted = predict_pixel(r_rel, center_a)
    if predicted is None:
        return None
    distance = math.hypot(predicted[0] - center_b[0], predicted[1] - center_b[1])
    scale = min(max_scale, 1.0 + gain * float(np.linalg.norm(sample_b[1] - sample_a[1])))
    return distance, predicted, scale
