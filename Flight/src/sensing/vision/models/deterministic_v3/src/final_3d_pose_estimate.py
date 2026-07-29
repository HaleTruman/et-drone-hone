"""Fuses each tracked 2D void (analysis.final_void_estimates) with camera rays over time into
one persistent 3D position per track_id, tracked internally in local NED. Independent of
instance_tracking.py and final_2d_void_estimate.py; duplicates camera_odometry.py's rotation
math to stay self-contained -- camera_odometry.py is only ever used here for its public
telemetry-loading functions. Published output is the frozen VisionObservation/
VisionGateObservation contract (camera-optical frame, per gate/frame), rebuilt from the internal
NED state at publish time every frame. See v0_3d_pose_plan.md for the full design."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import camera_odometry as codo
from camera_odometry import MAX_TELEMETRY_GAP_S
from schema import VehicleState, VisionGateObservation, VisionObservation, VoidAnalysis

FX, FY, CX, CY = 320.0, 320.0, 320.0, 180.0  # duplicated from camera_odometry.py
CAMERA_TILT_DEG = 20.0  # duplicated from camera_odometry.py

ANCHOR_PX, ANCHOR_M = 4000.0, 10.0
_OTHER_PX, _OTHER_M = 40.0, 40.0
# Power law solved from both given anchor points (4000px=10m, 40px=40m) so both are satisfied
# exactly -- a plain inverse-square model does not fit both at once.
DEPTH_PRIOR_POWER = math.log(_OTHER_M / ANCHOR_M) / math.log(_OTHER_PX / ANCHOR_PX)

PRIOR_WEIGHT = 0.05  # low relative to real-ray weights; not precision-tuned (v0)
SPREAD_SATURATION_DEG = 15.0  # angle at which parallax spread maxes out the confidence factor; not tuned (v0)
RAY_COUNT_TIERS = ((8, 0.9), (3, 0.6), (1, 0.3))  # (min ray_count, base confidence), highest first; not tuned (v0)
MIN_VOID_OBSERVATION_CONFIDENCE = 0.6


def _pitch_matrix(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# Camera CV convention (X=right, Y=down, Z=forward/depth) <-> body FRD, composed with the fixed
# 20-degree upward tilt -- identical to camera_odometry.py's _BODY_FROM_CAM, duplicated here.
_BASE_BODY_FROM_CAM = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_BODY_FROM_CAM = _pitch_matrix(CAMERA_TILT_DEG) @ _BASE_BODY_FROM_CAM


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def _camera_ray(center: tuple[float, float]) -> np.ndarray:
    u, v = center
    return _normalize(np.array([(u - CX) / FX, (v - CY) / FY, 1.0]))


def _expected_distance(pixel_count: float) -> float:
    return ANCHOR_M * (max(pixel_count, 1.0) / ANCHOR_PX) ** DEPTH_PRIOR_POWER


def _frame_identity(frame_path: str) -> tuple[int, int]:
    # frame-<frame_id>-<sim_time_ns>.ext -- matches frames.jsonl exactly, no extra file I/O needed.
    parts = Path(frame_path).stem.split("-")
    return int(parts[1]), int(parts[-1])


def _new_entry() -> dict:
    return {
        "A": np.zeros((3, 3)), "b": np.zeros(3), "direction_sum": np.zeros(3), "min_dot": 1.0,
        "ray_count": 0, "last_pixel_count": None, "position": None, "passed": False,
    }


class Final3DPoseEstimator:
    def __init__(self) -> None:
        self._telemetry_checked = False
        self._telemetry = None
        self._pool: dict[str, dict] = {}

    def _ensure_telemetry(self, frame_path: str) -> None:
        if self._telemetry_checked:
            return
        self._telemetry_checked = True
        path = codo.find_telemetry_for_frame(frame_path)
        if path is not None:
            self._telemetry = codo.load_telemetry(path)

    def _pose(self, frame_path: str):
        if self._telemetry is None:
            return None
        sample = codo.sample_for_frame(self._telemetry, frame_path)
        if sample is None:
            return None
        quat, position = sample
        r_wc = _quat_to_matrix(np.array(quat)) @ _BODY_FROM_CAM
        return r_wc, np.array(position, dtype=float)

    def _pose_from_vehicle_state(self, vehicle_state: VehicleState | None, frame_sim_time_ns: int):
        if vehicle_state is None:
            return None
        if abs(frame_sim_time_ns - vehicle_state.sim_time_ns) > MAX_TELEMETRY_GAP_S * 1e9:
            return None
        r_wc = _quat_to_matrix(np.array(vehicle_state.attitude_quaternion)) @ _BODY_FROM_CAM
        return r_wc, np.array(vehicle_state.position_local_ned_m, dtype=float)

    def _fold_ray(self, entry: dict, center: tuple[float, float], weight: float,
                  r_wc: np.ndarray, c_t: np.ndarray) -> None:
        d_world = _normalize(r_wc @ _camera_ray(center))
        p_t = np.eye(3) - np.outer(d_world, d_world)
        entry["A"] += weight * p_t
        entry["b"] += weight * (p_t @ c_t)
        if entry["ray_count"] > 0:
            # compare the new ray against the mean of every *prior* ray -- a cheap running
            # parallax-spread proxy that doesn't require keeping the raw ray history around.
            mean_direction = _normalize(entry["direction_sum"])
            entry["min_dot"] = min(entry["min_dot"], float(np.dot(d_world, mean_direction)))
        entry["direction_sum"] += d_world
        entry["ray_count"] += 1

    def _solve(self, entry: dict, c_t: np.ndarray) -> np.ndarray:
        # Depth prior applied at solve time only (never baked into the stored accumulators) as
        # an isotropic point term, not a same-direction "extra ray" -- a ray sharing the real
        # observations' direction would contribute the identical projector and add zero rank,
        # leaving a single-ray estimate singular. See v0_3d_pose_plan.md.
        mean_direction = _normalize(entry["direction_sum"])
        x_prior = c_t + _expected_distance(entry["last_pixel_count"] or ANCHOR_PX) * mean_direction
        a_total = entry["A"] + PRIOR_WEIGHT * np.eye(3)
        b_total = entry["b"] + PRIOR_WEIGHT * x_prior
        return np.linalg.pinv(a_total) @ b_total

    def _confidence(self, entry: dict) -> float:
        base = next(value for min_count, value in RAY_COUNT_TIERS if entry["ray_count"] >= min_count)
        if entry["ray_count"] < 2:
            return base * 0.5  # spread is undefined with a single ray; the prior carries it
        spread_deg = math.degrees(math.acos(max(-1.0, min(1.0, entry["min_dot"]))))
        spread_factor = max(0.0, min(1.0, spread_deg / SPREAD_SATURATION_DEG))
        return max(0.0, min(1.0, base * spread_factor))

    def _check_passed(self, entry: dict, r_wc: np.ndarray, c_t: np.ndarray) -> None:
        if entry["passed"] or entry["position"] is None:
            return
        x_cam = r_wc.T @ (entry["position"] - c_t)
        if x_cam[2] <= 0:
            entry["passed"] = True

    def _gate_observation(self, track_id: str, entry: dict, r_wc: np.ndarray, c_t: np.ndarray) -> VisionGateObservation:
        # Camera-relative NED: translate to the camera's current position only -- no rotation
        # into the camera's own optical attitude, so axes stay north/east/down.
        position_local_ned = tuple(float(v) for v in (entry["position"] - c_t))
        trace = {
            "track_id": track_id,
            "position_ned_m": tuple(float(v) for v in entry["position"]),
            "ray_count": entry["ray_count"],
        }
        return VisionGateObservation(
            gate_id=track_id,
            position_local_ned=position_local_ned,
            position_confidence=self._confidence(entry),
            trace=trace,
        )

    def _advance(self, analysis: VoidAnalysis, frame_path: str, pose) -> list[VisionGateObservation]:
        gates: list[VisionGateObservation] = []
        if pose is None:
            # no usable pose this frame -> can't compute a camera-relative position for anyone;
            # gates=[] rather than a stale/fabricated one (io_specification.md: a valid
            # observation with gates=[], never None).
            return gates

        r_wc, c_t = pose
        pixel_count_by_track = {
            void.track_id: void.pixel_count for region in analysis.regions for void in region.voids
        }
        touched = set()
        for estimate in analysis.final_void_estimates:
            if not math.isfinite(estimate.confidence) or estimate.confidence < MIN_VOID_OBSERVATION_CONFIDENCE:
                continue
            entry = self._pool.setdefault(estimate.track_id, _new_entry())
            touched.add(estimate.track_id)
            if entry["passed"]:
                continue
            if estimate.track_id in pixel_count_by_track:
                entry["last_pixel_count"] = pixel_count_by_track[estimate.track_id]
            self._fold_ray(entry, estimate.center, estimate.confidence, r_wc, c_t)
            entry["position"] = self._solve(entry, c_t)
            self._check_passed(entry, r_wc, c_t)

        # a landmark can be passed by the camera's own motion alone, without a fresh 2D
        # detection this frame -- still worth checking every not-yet-passed, untouched entry.
        for track_id, entry in self._pool.items():
            if track_id in touched or entry["passed"]:
                continue
            self._check_passed(entry, r_wc, c_t)

        # position_local_ned is pose-dependent -- rebuilt fresh every frame for every still-
        # active landmark, even ones with no new ray this frame, since the camera moved.
        for track_id, entry in self._pool.items():
            if entry["passed"] or entry["position"] is None:
                continue
            gates.append(self._gate_observation(track_id, entry, r_wc, c_t))
        return gates

    def update(self, analysis: VoidAnalysis, frame_path: str) -> VoidAnalysis:
        self._ensure_telemetry(frame_path)
        gates = self._advance(analysis, frame_path, self._pose(frame_path))
        frame_id, sim_time_ns = _frame_identity(frame_path)
        analysis.vision_observation = VisionObservation(frame_id, sim_time_ns, gates, source="vision")
        return analysis

    def update_live(self, analysis: VoidAnalysis, frame_path: str, vehicle_state: VehicleState | None) -> VoidAnalysis:
        frame_id, sim_time_ns = _frame_identity(frame_path)
        pose = self._pose_from_vehicle_state(vehicle_state, sim_time_ns)
        gates = self._advance(analysis, frame_path, pose)
        analysis.vision_observation = VisionObservation(frame_id, sim_time_ns, gates, source="vision")
        return analysis
