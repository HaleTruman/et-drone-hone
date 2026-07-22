"""Visual-inertial odometry measurement helpers.

This module keeps VIO inputs in the same LOCAL_NED / BODY_FRD conventions used
by VehicleStateEstimator. It intentionally does not own a VIO backend yet; it
provides the data contract and frame conversion utilities needed to fuse one.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from core.coordinates import (
    camera_optical_to_body_frd,
    normalize_quaternion,
    quat_wxyz,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion,
    vec3,
)
from core.schemas import QuatWxyz, Vec3, VehicleState


class VioProvider(Protocol):
    """Interface for future VIO backends."""

    def get_latest_measurement(self) -> "VioMeasurement | None":
        ...


@dataclass(frozen=True)
class VioCorrectionConfig:
    """Blend factors for the initial complementary-filter VIO correction."""

    position_alpha: float = 0.05
    velocity_alpha: float = 0.10
    attitude_alpha: float = 0.03
    max_measurement_age_s: float = 0.25
    max_position_residual_m: float | None = 10.0
    max_velocity_residual_mps: float | None = 20.0


@dataclass(frozen=True)
class VioMeasurement:
    """Pose/velocity measurement expressed in LOCAL_NED."""

    sim_time_ns: int
    position_local_ned_m: Vec3
    attitude_quaternion: QuatWxyz
    velocity_local_ned_mps: Vec3 | None = None
    position_std_m: Vec3 | None = None
    attitude_std_rad: Vec3 | None = None
    velocity_std_mps: Vec3 | None = None
    confidence: float = 1.0
    source: str = "vio"
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sim_time_ns", int(self.sim_time_ns))
        object.__setattr__(self, "position_local_ned_m", vec3(self.position_local_ned_m))
        object.__setattr__(self, "attitude_quaternion", quat_wxyz(self.attitude_quaternion))
        if self.velocity_local_ned_mps is not None:
            object.__setattr__(self, "velocity_local_ned_mps", vec3(self.velocity_local_ned_mps))
        if self.position_std_m is not None:
            object.__setattr__(self, "position_std_m", vec3(self.position_std_m))
        if self.attitude_std_rad is not None:
            object.__setattr__(self, "attitude_std_rad", vec3(self.attitude_std_rad))
        if self.velocity_std_mps is not None:
            object.__setattr__(self, "velocity_std_mps", vec3(self.velocity_std_mps))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, float(self.confidence))))
        object.__setattr__(self, "raw", dict(self.raw or {}))

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "sim_time_ns": self.sim_time_ns,
            "position_local_ned_m": list(self.position_local_ned_m),
            "velocity_local_ned_mps": None
            if self.velocity_local_ned_mps is None
            else list(self.velocity_local_ned_mps),
            "attitude_quaternion": list(self.attitude_quaternion),
            "position_std_m": None if self.position_std_m is None else list(self.position_std_m),
            "attitude_std_rad": None if self.attitude_std_rad is None else list(self.attitude_std_rad),
            "velocity_std_mps": None if self.velocity_std_mps is None else list(self.velocity_std_mps),
            "confidence": self.confidence,
            "source": self.source,
            "raw": self.raw,
        }

    @classmethod
    def from_camera_optical_pose(
        cls,
        *,
        sim_time_ns: int,
        camera_position_local_ned_m: Vec3,
        camera_attitude_quaternion: QuatWxyz,
        body_to_camera_translation_body_frd_m: Vec3 = (0.0, 0.0, 0.0),
        camera_tilt_deg: float = 20.0,
        velocity_local_ned_mps: Vec3 | None = None,
        confidence: float = 1.0,
        source: str = "vio_camera_optical",
        raw: dict[str, Any] | None = None,
    ) -> "VioMeasurement":
        """Convert a LOCAL_NED camera pose into the vehicle body pose.

        `camera_attitude_quaternion` is interpreted as camera optical axes in
        LOCAL_NED. `body_to_camera_translation_body_frd_m` is the camera origin
        offset from the body origin, expressed in BODY_FRD. The returned
        measurement estimates the body FRD origin pose.
        """

        rotation_camera_to_local = rotation_matrix_from_quaternion(camera_attitude_quaternion)
        rotation_optical_to_body = camera_optical_to_body_frd(camera_tilt_deg)
        rotation_body_to_camera = rotation_optical_to_body.T
        rotation_body_to_local = rotation_camera_to_local @ rotation_body_to_camera

        body_offset_local = rotation_body_to_local @ np.asarray(body_to_camera_translation_body_frd_m, dtype=float)
        body_position_local = np.asarray(camera_position_local_ned_m, dtype=float) - body_offset_local

        return cls(
            sim_time_ns=sim_time_ns,
            position_local_ned_m=vec3(body_position_local),
            velocity_local_ned_mps=velocity_local_ned_mps,
            attitude_quaternion=quaternion_from_rotation_matrix(rotation_body_to_local),
            confidence=confidence,
            source=source,
            raw=raw,
        )

    def residuals(self, state: VehicleState) -> dict[str, Any]:
        position_residual = np.asarray(self.position_local_ned_m, dtype=float) - np.asarray(
            state.position_local_ned_m,
            dtype=float,
        )
        velocity_residual = None
        if self.velocity_local_ned_mps is not None:
            velocity_residual = np.asarray(self.velocity_local_ned_mps, dtype=float) - np.asarray(
                state.velocity_local_ned_mps,
                dtype=float,
            )
        return {
            "position_local_ned_m": position_residual.tolist(),
            "position_norm_m": float(np.linalg.norm(position_residual)),
            "velocity_local_ned_mps": None if velocity_residual is None else velocity_residual.tolist(),
            "velocity_norm_mps": None if velocity_residual is None else float(np.linalg.norm(velocity_residual)),
        }


def should_apply_vio_measurement(
    state: VehicleState,
    measurement: VioMeasurement,
    *,
    config: VioCorrectionConfig,
) -> tuple[bool, str]:
    age_s = abs(float(state.sim_time_ns - measurement.sim_time_ns)) / 1_000_000_000.0
    if age_s > config.max_measurement_age_s:
        return False, "stale_vio_measurement"

    residuals = measurement.residuals(state)
    position_norm = float(residuals["position_norm_m"])
    if config.max_position_residual_m is not None and position_norm > config.max_position_residual_m:
        return False, "position_residual_too_large"

    velocity_norm = residuals["velocity_norm_mps"]
    if (
        config.max_velocity_residual_mps is not None
        and velocity_norm is not None
        and float(velocity_norm) > config.max_velocity_residual_mps
    ):
        return False, "velocity_residual_too_large"

    if measurement.confidence <= 0.0:
        return False, "zero_vio_confidence"

    return True, "accepted"


def blend_vio_state(
    state: VehicleState,
    measurement: VioMeasurement,
    *,
    config: VioCorrectionConfig,
) -> VehicleState:
    confidence = measurement.confidence
    position_alpha = _clamped_alpha(config.position_alpha * confidence)
    velocity_alpha = _clamped_alpha(config.velocity_alpha * confidence)
    attitude_alpha = _clamped_alpha(config.attitude_alpha * confidence)

    position = _lerp_vec3(state.position_local_ned_m, measurement.position_local_ned_m, position_alpha)
    velocity = state.velocity_local_ned_mps
    if measurement.velocity_local_ned_mps is not None:
        velocity = _lerp_vec3(state.velocity_local_ned_mps, measurement.velocity_local_ned_mps, velocity_alpha)

    return VehicleState(
        sim_time_ns=state.sim_time_ns,
        position_local_ned_m=position,
        velocity_local_ned_mps=velocity,
        attitude_quaternion=slerp_quaternion(
            state.attitude_quaternion,
            measurement.attitude_quaternion,
            attitude_alpha,
        ),
        body_rates_frd_rps=state.body_rates_frd_rps,
        acceleration_local_ned_mps2=state.acceleration_local_ned_mps2,
    )


def slerp_quaternion(start: QuatWxyz, end: QuatWxyz, alpha: float) -> QuatWxyz:
    t = _clamped_alpha(alpha)
    q0 = normalize_quaternion(start)
    q1 = normalize_quaternion(end)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        return quat_wxyz(q0 + t * (q1 - q0))

    theta_0 = float(np.arccos(max(-1.0, min(1.0, dot))))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * t
    sin_theta = float(np.sin(theta))

    scale0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    scale1 = sin_theta / sin_theta_0
    return quat_wxyz(scale0 * q0 + scale1 * q1)


def _lerp_vec3(start: Vec3, end: Vec3, alpha: float) -> Vec3:
    return vec3(np.asarray(start, dtype=float) + _clamped_alpha(alpha) * (np.asarray(end, dtype=float) - np.asarray(start, dtype=float)))


def _clamped_alpha(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "VioCorrectionConfig",
    "VioMeasurement",
    "VioProvider",
    "blend_vio_state",
    "should_apply_vio_measurement",
    "slerp_quaternion",
]
