"""Build an auditable LOCAL_NED replay from logs and camera-relative PnP.

This module is UI-only. It joins exact recorded frame/cycle identities, then
reuses the production camera-to-NED transform. Derived display evidence stays
under ``ui_projection`` and never feeds production inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping

import cv2
import numpy as np

from ...src.void_ned import (
    CAMERA_OFFSET_BODY_M,
    CAMERA_TILT_DEG,
    camera_position_ned,
    rotation_world_from_body,
    rotation_world_from_camera,
)
from .pnp_world_replay_configuration import (
    DEFAULT_PNP_WORLD_REPLAY_CONFIGURATION,
    PnpWorldReplayConfiguration,
)


WORLD_REPLAY_VERSION = "deterministic-v3.pnp-world-replay.v2"


@dataclass(frozen=True, slots=True)
class VehicleSample:
    sample_index: int
    inner_cycle: int
    sim_time_ns: int
    position_local_ned_m: tuple[float, float, float]
    attitude_quaternion: tuple[float, float, float, float]
    monotonic_s: float | None

    def json_value(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "inner_cycle": self.inner_cycle,
            "sim_time_ns": self.sim_time_ns,
            "position_local_ned_m": list(self.position_local_ned_m),
            "attitude_quaternion": list(self.attitude_quaternion),
        }


@dataclass(frozen=True, slots=True)
class FrameVehicleAssociation:
    frame_id: int
    frame_sim_time_ns: int
    sample: VehicleSample
    alignment_error_ms: float
    alignment_error_source: str


def _json_document(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _json_lines(path: Path) -> Iterable[dict]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path.name}:{line_number}") from error
            if isinstance(value, dict):
                yield value


def _finite_vector(value, length: int) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _vehicle_samples(run_dir: Path) -> tuple[VehicleSample, ...]:
    samples = []
    seen_cycles = set()
    path = run_dir / "lists" / "telemetry.jsonl"
    for record in _json_lines(path):
        telemetry = record.get("telemetry")
        vehicle = telemetry.get("vehicle_state") if isinstance(
            telemetry, dict) else None
        position = _finite_vector(
            vehicle.get("position_local_ned_m") if isinstance(vehicle, dict)
            else None, 3)
        attitude = _finite_vector(
            vehicle.get("attitude_quaternion") if isinstance(vehicle, dict)
            else None, 4)
        try:
            inner_cycle = int(record["inner_cycle"])
            sim_time_ns = int(vehicle["sim_time_ns"])
        except (KeyError, TypeError, ValueError):
            continue
        if position is None or attitude is None or inner_cycle in seen_cycles:
            continue
        monotonic = record.get("monotonic_s")
        monotonic_s = (
            float(monotonic) if isinstance(monotonic, (int, float)) and
            math.isfinite(float(monotonic)) else None)
        seen_cycles.add(inner_cycle)
        samples.append(VehicleSample(
            len(samples), inner_cycle, sim_time_ns, position, attitude,
            monotonic_s))
    return tuple(samples)


def _alignment_error_ms(
    record: Mapping, frame: Mapping, sample: VehicleSample
) -> tuple[float, str] | None:
    frame_elapsed = frame.get("frame_elapsed_ns")
    vehicle_elapsed = frame.get("vehicle_state_elapsed_ns")
    if isinstance(frame_elapsed, int) and isinstance(vehicle_elapsed, int):
        return (
            abs(frame_elapsed - vehicle_elapsed) / 1_000_000,
            "frame_vehicle_state_elapsed_ns",
        )
    monotonic = record.get("monotonic_s")
    if (isinstance(monotonic, (int, float)) and sample.monotonic_s is not None
            and math.isfinite(float(monotonic))):
        return (
            abs(float(monotonic) - sample.monotonic_s) * 1000,
            "record_monotonic_s",
        )
    return None


def _frame_associations(
    run_dir: Path, samples: tuple[VehicleSample, ...]
) -> dict[tuple[int, int], FrameVehicleAssociation]:
    by_cycle = {sample.inner_cycle: sample for sample in samples}
    associations = {}
    path = run_dir / "lists" / "vision_frames.jsonl"
    for record in _json_lines(path):
        frame = record.get("frame")
        if not isinstance(frame, dict):
            continue
        try:
            frame_id = int(frame["frame_id"])
            frame_time = int(frame["sim_time_ns"])
            inner_cycle = int(frame["inner_cycle"])
        except (KeyError, TypeError, ValueError):
            continue
        sample = by_cycle.get(inner_cycle)
        error = (
            _alignment_error_ms(record, frame, sample)
            if sample is not None else None)
        if sample is None or error is None:
            continue
        key = (frame_id, frame_time)
        if key in associations:
            raise ValueError(f"duplicate vision frame association: {key}")
        associations[key] = FrameVehicleAssociation(
            frame_id, frame_time, sample, error[0], error[1])
    return associations


def _initialization_constants(run_dir: Path) -> Mapping:
    metadata = _json_document(run_dir / "metadata.json")
    constants = metadata.get("initialization_constants")
    if isinstance(constants, dict):
        return constants
    run = _json_document(run_dir / "run.json")
    nested = run.get("metadata")
    constants = nested.get("initialization_constants") if isinstance(
        nested, dict) else None
    return constants if isinstance(constants, dict) else {}


def _camera_mount(
    run_dir: Path, configuration: PnpWorldReplayConfiguration
) -> dict[str, object]:
    constants = _initialization_constants(run_dir)
    tilt = constants.get("VIO_CAMERA_TILT_DEG")
    translation = _finite_vector(
        constants.get("VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M"), 3)
    logged = (
        isinstance(tilt, (int, float)) and math.isfinite(float(tilt)) and
        translation is not None)
    if logged:
        matches = (
            math.isclose(float(tilt), CAMERA_TILT_DEG, abs_tol=1e-9) and
            np.allclose(translation, CAMERA_OFFSET_BODY_M, atol=1e-9))
        reason = None if matches else "logged_mount_differs_from_production"
    elif configuration.require_logged_camera_mount:
        matches = False
        reason = "logged_camera_mount_unavailable"
    else:
        tilt = CAMERA_TILT_DEG
        translation = tuple(map(float, CAMERA_OFFSET_BODY_M))
        matches = True
        reason = None
    return {
        "source": (
            "metadata.initialization_constants" if logged
            else "production_transform"),
        "VIO_CAMERA_TILT_DEG": float(tilt) if tilt is not None else None,
        "VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M": (
            list(translation) if translation is not None else None),
        "production_module": (
            "sensing.vision.models.deterministic_v3.src.void_ned"),
        "matches_production_transform": bool(matches),
        "usable": bool(matches),
        "unavailable_reason": reason,
    }


def _matrix_json(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in matrix]


def _point_json(point: np.ndarray) -> list[float]:
    return [float(value) for value in point]


def _project_transform(
    transform: Mapping,
    *,
    source_field: str,
    pose_index: int,
    component_id: object,
    gate_index: object,
    route: object,
    camera_origin_ned: np.ndarray,
    rotation_ned_from_camera_cv: np.ndarray,
    configuration: PnpWorldReplayConfiguration,
    object_points: tuple[np.ndarray, ...] | None,
) -> dict[str, object] | None:
    position = _finite_vector(transform.get("position_camera_m"), 3)
    rotation_vector = _finite_vector(
        transform.get("rotation_vector_model_to_camera"), 3)
    if position is None or rotation_vector is None:
        return None
    center = camera_origin_ned + rotation_ned_from_camera_cv @ position
    corners = None
    orientation_status = "unavailable"
    if configuration.render_provisional_orientation and object_points is not None:
        rotation_camera_from_model = cv2.Rodrigues(
            np.asarray(rotation_vector, np.float64))[0]
        rotation_ned_from_model = (
            rotation_ned_from_camera_cv @ rotation_camera_from_model)
        corners = [
            _point_json(center + rotation_ned_from_model @ point)
            for point in object_points
        ]
        orientation_status = "provisional"
    return {
        "source_field": source_field,
        "pose_index": pose_index,
        "component_id": component_id,
        "gate_index": gate_index,
        "route": route,
        "position_local_ned_m": _point_json(center),
        "corners_local_ned_m": corners,
        "orientation_status": orientation_status,
        "camera_depth_m": float(position[2]),
    }


def _runtime_pose_context(runtime: Mapping) -> tuple[object, object, object]:
    model = runtime.get("gate_model")
    calibration = runtime.get("camera_calibration")
    calibration_id = calibration.get("calibration_id") if isinstance(
        calibration, dict) else None
    model_id = model.get("model_id") if isinstance(model, dict) else None
    object_points = _finite_object_points(
        model.get("object_points_m") if isinstance(model, dict) else None)
    return calibration_id, model_id, object_points


def _project_poses(
    runtime: Mapping,
    field_name: str,
    camera_origin_ned: np.ndarray,
    rotation_ned_from_camera_cv: np.ndarray,
    configuration: PnpWorldReplayConfiguration,
) -> list[dict[str, object]]:
    poses = runtime.get(field_name)
    calibration_id, model_id, object_points = _runtime_pose_context(runtime)
    if not isinstance(poses, list):
        return []
    projected = []
    for pose_index, pose in enumerate(poses):
        if (not isinstance(pose, dict) or pose.get("accepted") is not True or
                pose.get("camera_calibration_id") != calibration_id or
                pose.get("gate_model_id") != model_id):
            continue
        value = _project_transform(
            pose,
            source_field=f"GeometryFrameResult.{field_name}",
            pose_index=pose_index,
            component_id=pose.get("component_id"),
            gate_index=pose.get("gate_index"),
            route=pose.get("route"),
            camera_origin_ned=camera_origin_ned,
            rotation_ned_from_camera_cv=rotation_ned_from_camera_cv,
            configuration=configuration,
            object_points=object_points,
        )
        if value is not None:
            projected.append(value)
    return projected


def _project_pnp_estimates(
    runtime: Mapping,
    camera_origin_ned: np.ndarray,
    rotation_ned_from_camera_cv: np.ndarray,
    configuration: PnpWorldReplayConfiguration,
) -> list[dict[str, object]]:
    estimates = runtime.get("pnp_relative_pose_estimates")
    calibration_id, model_id, object_points = _runtime_pose_context(runtime)
    if not isinstance(estimates, list):
        return []
    projected = []
    for pose_index, estimate in enumerate(estimates):
        if (not isinstance(estimate, dict) or
                estimate.get("accepted") is not True or
                estimate.get("camera_calibration_id") != calibration_id or
                estimate.get("gate_model_id") != model_id or
                not isinstance(estimate.get("candidates"), list)):
            continue
        try:
            selected_rank = int(estimate["selected_candidate_rank"])
        except (KeyError, TypeError, ValueError):
            continue
        selected = next((
            candidate for candidate in estimate["candidates"]
            if isinstance(candidate, dict) and
            candidate.get("candidate_rank") == selected_rank), None)
        secondary = next((
            candidate for candidate in estimate["candidates"]
            if isinstance(candidate, dict) and
            candidate.get("candidate_rank") != selected_rank), None)
        common = {
            "pose_index": pose_index,
            "component_id": estimate.get("component_id"),
            "gate_index": estimate.get("gate_index"),
            "route": estimate.get("route"),
            "camera_origin_ned": camera_origin_ned,
            "rotation_ned_from_camera_cv": rotation_ned_from_camera_cv,
            "configuration": configuration,
            "object_points": object_points,
        }
        selected_projection = _project_transform(
            selected or {},
            source_field=("GeometryFrameResult.pnp_relative_pose_estimates."
                          "candidates[selected]"),
            **common,
        )
        if selected_projection is None:
            continue
        selected_projection["candidate_rank"] = selected_rank
        secondary_projection = _project_transform(
            secondary or {},
            source_field=("GeometryFrameResult.pnp_relative_pose_estimates."
                          "candidates[secondary]"),
            **common,
        )
        if secondary_projection is not None:
            secondary_projection["candidate_rank"] = int(
                secondary["candidate_rank"])
        projected.append({
            "source_field": "GeometryFrameResult.pnp_relative_pose_estimates",
            "pose_index": pose_index,
            "component_id": estimate.get("component_id"),
            "gate_index": estimate.get("gate_index"),
            "route": estimate.get("route"),
            "selected_candidate": selected_projection,
            "secondary_candidate": secondary_projection,
        })
    return projected


def _finite_object_points(value) -> tuple[np.ndarray, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    points = []
    for point in value:
        vector = _finite_vector(point, 3)
        if vector is None:
            return None
        points.append(np.asarray(vector, np.float64))
    return tuple(points)


def _world_frame(
    item: Mapping,
    association: FrameVehicleAssociation | None,
    mount: Mapping,
    configuration: PnpWorldReplayConfiguration,
) -> dict[str, object]:
    runtime = item["runtime_result"]
    frame_id = int(runtime["frame_id"])
    frame_time = int(runtime["sim_time_ns"])
    within_tolerance = bool(
        association is not None and
        association.alignment_error_ms <=
        configuration.maximum_alignment_error_ms)
    sync = {
        "match_source": "frame_cycle" if association else "none",
        "frame_sim_time_ns": frame_time,
        "vehicle_state_sim_time_ns": (
            association.sample.sim_time_ns if association else None),
        "alignment_error_ms": (
            association.alignment_error_ms if association else None),
        "alignment_error_source": (
            association.alignment_error_source if association else None),
        "within_tolerance": within_tolerance,
    }
    vehicle = association.sample.json_value() if association else None
    projection: dict[str, object] = {
        "available": False,
        "unavailable_reason": None,
        "camera_position_local_ned_m": None,
        "rotation_local_ned_from_body_frd": None,
        "rotation_local_ned_from_camera_cv": None,
        "pnp_relative_pose_estimates": [],
        "camera_pose_estimates": [],
    }
    if association is None:
        projection["unavailable_reason"] = "exact_frame_cycle_unavailable"
    elif not within_tolerance:
        projection["unavailable_reason"] = "alignment_tolerance_exceeded"
    elif not mount.get("usable"):
        projection["unavailable_reason"] = mount.get("unavailable_reason")
    else:
        pose = SimpleNamespace(
            position_local_ned_m=association.sample.position_local_ned_m,
            attitude_quaternion=association.sample.attitude_quaternion,
            sim_time_ns=association.sample.sim_time_ns,
        )
        camera_origin = camera_position_ned(pose)
        rotation_ned_from_body = rotation_world_from_body(
            pose.attitude_quaternion)
        rotation_ned_from_camera_cv = rotation_world_from_camera(
            pose.attitude_quaternion)
        projection.update({
            "available": True,
            "camera_position_local_ned_m": _point_json(camera_origin),
            "rotation_local_ned_from_body_frd": _matrix_json(
                rotation_ned_from_body),
            "rotation_local_ned_from_camera_cv": _matrix_json(
                rotation_ned_from_camera_cv),
            "pnp_relative_pose_estimates": _project_pnp_estimates(
                runtime, camera_origin, rotation_ned_from_camera_cv,
                configuration),
            "camera_pose_estimates": _project_poses(
                runtime, "camera_pose_estimates", camera_origin,
                rotation_ned_from_camera_cv, configuration),
        })
    return {
        "frame": {"frame_id": frame_id, "sim_time_ns": frame_time},
        "sync": sync,
        "vehicle_state": vehicle,
        "runtime_result": runtime,
        "ui_projection": projection,
    }


def _trajectory_sample(
    sample: VehicleSample, mount: Mapping
) -> dict[str, object]:
    value = sample.json_value()
    projection: dict[str, object] = {
        "available": False,
        "camera_position_local_ned_m": None,
        "unavailable_reason": mount.get("unavailable_reason"),
    }
    if mount.get("usable"):
        pose = SimpleNamespace(
            position_local_ned_m=sample.position_local_ned_m,
            attitude_quaternion=sample.attitude_quaternion,
            sim_time_ns=sample.sim_time_ns,
        )
        projection.update({
            "available": True,
            "camera_position_local_ned_m": _point_json(
                camera_position_ned(pose)),
            "unavailable_reason": None,
        })
    value["ui_projection"] = projection
    return value


def build_pnp_world_replay(
    *,
    run_id: str,
    run_dir: str | Path,
    geometry_items: Iterable[Mapping],
    configuration: PnpWorldReplayConfiguration =
        DEFAULT_PNP_WORLD_REPLAY_CONFIGURATION,
) -> dict[str, object]:
    """Build one immutable-in-practice world replay document for the UI."""
    if not configuration.require_exact_frame_cycle:
        raise ValueError("only exact frame/cycle world replay is implemented")
    run_path = Path(run_dir).resolve()
    samples = _vehicle_samples(run_path)
    associations = _frame_associations(run_path, samples)
    mount = _camera_mount(run_path, configuration)
    frames = []
    seen = set()
    for item in geometry_items:
        if not isinstance(item, Mapping):
            continue
        runtime = item.get("runtime_result")
        if (item.get("run_id") != run_id or not isinstance(runtime, Mapping)):
            continue
        try:
            identity = (int(runtime["frame_id"]), int(runtime["sim_time_ns"]))
        except (KeyError, TypeError, ValueError):
            continue
        if identity in seen:
            raise ValueError(f"duplicate GeometryFrameResult identity: {identity}")
        seen.add(identity)
        frames.append(_world_frame(
            item, associations.get(identity), mount, configuration))
    frames.sort(key=lambda value: (
        value["frame"]["sim_time_ns"], value["frame"]["frame_id"]))
    return {
        "version": WORLD_REPLAY_VERSION,
        "run_id": run_id,
        "configuration": configuration.json_value(),
        "coordinate_system": {
            "world": "local_ned",
            "body": "frd",
            "camera": "opencv_cv",
            "three_display": "north_up_east",
        },
        "camera_mount": mount,
        "trajectory": [_trajectory_sample(sample, mount) for sample in samples],
        "frames": frames,
    }
