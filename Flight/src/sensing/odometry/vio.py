"""Visual-inertial odometry measurement helpers.

This module keeps VIO inputs in the same LOCAL_NED / BODY_FRD conventions used
by VehicleStateEstimator. The OpenCV provider is intentionally lightweight: it
tracks visual features, estimates frame-to-frame monocular motion, and uses
recent IMU samples only for metric scale hints.
"""

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from core.coordinates import (
    camera_optical_to_body_frd,
    normalize_quaternion,
    quat_wxyz,
    quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion,
    vec3,
)
from core.schemas import MavlinkHighresImu, QuatWxyz, Vec3, VehicleState

if TYPE_CHECKING:
    from sensing.vision import VisionFrame


class VioProvider(Protocol):
    """Interface for future VIO backends."""

    def get_latest_measurement(self) -> "VioMeasurement | None":
        ...


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera calibration used by the OpenCV monocular frontend."""

    width_px: int
    height_px: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float
    distortion: tuple[float, ...] = ()

    @classmethod
    def from_image_size(
        cls,
        *,
        width_px: int,
        height_px: int,
        horizontal_fov_deg: float = 90.0,
        distortion: tuple[float, ...] = (),
    ) -> "CameraIntrinsics":
        half_fov_rad = np.deg2rad(float(horizontal_fov_deg)) * 0.5
        fx_px = (float(width_px) * 0.5) / max(float(np.tan(half_fov_rad)), 1e-9)
        fy_px = fx_px
        return cls(
            width_px=int(width_px),
            height_px=int(height_px),
            fx_px=fx_px,
            fy_px=fy_px,
            cx_px=(float(width_px) - 1.0) * 0.5,
            cy_px=(float(height_px) - 1.0) * 0.5,
            distortion=distortion,
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx_px, 0.0, self.cx_px],
                [0.0, self.fy_px, self.cy_px],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    @property
    def distortion_array(self) -> np.ndarray | None:
        if not self.distortion:
            return None
        return np.asarray(self.distortion, dtype=float)


@dataclass(frozen=True)
class VioFrontendConfig:
    """Feature tracking and relative-pose thresholds for OpenCV VIO."""

    max_corners: int = 250
    quality_level: float = 0.01
    min_distance_px: float = 8.0
    block_size_px: int = 7
    lk_window_px: int = 21
    lk_max_level: int = 3
    lk_max_iterations: int = 30
    lk_epsilon: float = 0.01
    ransac_prob: float = 0.999
    ransac_threshold_px: float = 1.5
    min_tracked_points: int = 40
    min_inliers: int = 30
    horizontal_fov_deg: float = 90.0
    fallback_translation_scale_m: float = 0.03
    max_imu_buffer_samples: int = 400
    max_frame_gap_s: float = 0.25
    min_confidence: float = 0.15


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


@dataclass(frozen=True)
class _TrackedFrame:
    frame_id: int
    sim_time_ns: int
    gray: np.ndarray
    keypoints_px: np.ndarray
    camera_position_local_ned_m: Vec3
    camera_attitude_quaternion: QuatWxyz


class OpenCvMonocularVioProvider:
    """Monocular visual odometry frontend with IMU-assisted scale hints.

    This is a pragmatic bootstrap VIO backend for the current Python flight
    loop. Monocular essential-matrix pose is only scale-observable through the
    IMU displacement estimate, so position corrections should stay lightly
    weighted until a metric landmark, depth, or stereo source is added.
    """

    def __init__(
        self,
        *,
        camera: CameraIntrinsics | None = None,
        frontend_config: VioFrontendConfig | None = None,
        body_to_camera_translation_body_frd_m: Vec3 = (0.0, 0.0, 0.0),
        camera_tilt_deg: float = 20.0,
    ):
        self.camera = camera
        self.config = frontend_config or VioFrontendConfig()
        self.body_to_camera_translation_body_frd_m = vec3(body_to_camera_translation_body_frd_m)
        self.camera_tilt_deg = float(camera_tilt_deg)
        self._imu_samples: deque[MavlinkHighresImu] = deque(maxlen=self.config.max_imu_buffer_samples)
        self._previous_frame: _TrackedFrame | None = None
        self._latest_measurement: VioMeasurement | None = None
        self._last_status = "not_initialized"
        self._last_raw: dict[str, Any] = {}

    def reset(self) -> None:
        self._imu_samples.clear()
        self._previous_frame = None
        self._latest_measurement = None
        self._last_status = "reset"
        self._last_raw = {}

    def add_imu_sample(self, imu: MavlinkHighresImu) -> None:
        if self._imu_samples and int(self._imu_samples[-1].time_boot_us) == int(imu.time_boot_us):
            return
        self._imu_samples.append(imu)

    def get_latest_measurement(self) -> VioMeasurement | None:
        return self._latest_measurement

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "opencv_monocular",
            "status": self._last_status,
            "imu_buffer_count": len(self._imu_samples),
            "has_camera_intrinsics": self.camera is not None,
            "has_previous_frame": self._previous_frame is not None,
            "latest_measurement": None if self._latest_measurement is None else self._latest_measurement.to_log_dict(),
            "last_raw": self._last_raw,
        }

    def process_frame(self, frame: "VisionFrame") -> VioMeasurement | None:
        gray = self._decode_gray(frame.jpeg_bytes)
        if gray is None:
            self._set_status("decode_failed", frame_id=frame.frame_id)
            return None

        if self.camera is None:
            height_px, width_px = gray.shape[:2]
            self.camera = CameraIntrinsics.from_image_size(
                width_px=width_px,
                height_px=height_px,
                horizontal_fov_deg=self.config.horizontal_fov_deg,
            )

        if self._previous_frame is None:
            keypoints = self._detect_features(gray)
            camera_attitude = quaternion_from_rotation_matrix(camera_optical_to_body_frd(self.camera_tilt_deg))
            self._previous_frame = _TrackedFrame(
                frame_id=frame.frame_id,
                sim_time_ns=frame.sim_time_ns,
                gray=gray,
                keypoints_px=keypoints,
                camera_position_local_ned_m=(0.0, 0.0, 0.0),
                camera_attitude_quaternion=camera_attitude,
            )
            self._set_status("initialized", frame_id=frame.frame_id, feature_count=len(keypoints))
            return None

        dt_s = (int(frame.sim_time_ns) - int(self._previous_frame.sim_time_ns)) / 1_000_000_000.0
        if dt_s <= 0.0 or dt_s > self.config.max_frame_gap_s:
            self._replace_previous_frame(frame, gray, status="frame_gap_too_large")
            return None

        previous_points, current_points = self._track_features(
            self._previous_frame.gray,
            gray,
            self._previous_frame.keypoints_px,
        )
        if len(current_points) < self.config.min_tracked_points:
            self._replace_previous_frame(frame, gray, status="too_few_tracked_points", tracked_count=len(current_points))
            return None

        relative_pose = self._estimate_relative_camera_pose(previous_points, current_points)
        if relative_pose is None:
            self._replace_previous_frame(frame, gray, status="relative_pose_failed", tracked_count=len(current_points))
            return None

        rotation_21, translation_21, inlier_count = relative_pose
        if inlier_count < self.config.min_inliers:
            self._replace_previous_frame(
                frame,
                gray,
                status="too_few_pose_inliers",
                tracked_count=len(current_points),
                inlier_count=inlier_count,
            )
            return None

        scale_m = self._imu_translation_scale_m(self._previous_frame.sim_time_ns, frame.sim_time_ns)
        if scale_m is None:
            scale_m = self.config.fallback_translation_scale_m

        previous_rotation_camera_to_local = rotation_matrix_from_quaternion(
            self._previous_frame.camera_attitude_quaternion
        )
        current_rotation_camera_to_local = previous_rotation_camera_to_local @ rotation_21.T
        translation_unit = translation_21.reshape(3)
        current_position = (
            np.asarray(self._previous_frame.camera_position_local_ned_m, dtype=float)
            - current_rotation_camera_to_local @ translation_unit * float(scale_m)
        )

        velocity = vec3(
            (
                current_position - np.asarray(self._previous_frame.camera_position_local_ned_m, dtype=float)
            )
            / max(dt_s, 1e-6)
        )
        confidence = self._confidence(tracked_count=len(current_points), inlier_count=inlier_count)
        camera_attitude = quaternion_from_rotation_matrix(current_rotation_camera_to_local)
        measurement = VioMeasurement.from_camera_optical_pose(
            sim_time_ns=frame.sim_time_ns,
            camera_position_local_ned_m=vec3(current_position),
            camera_attitude_quaternion=camera_attitude,
            body_to_camera_translation_body_frd_m=self.body_to_camera_translation_body_frd_m,
            camera_tilt_deg=self.camera_tilt_deg,
            velocity_local_ned_mps=velocity,
            confidence=confidence,
            source="opencv_monocular_vio",
            raw={
                "frame_id": int(frame.frame_id),
                "previous_frame_id": int(self._previous_frame.frame_id),
                "dt_s": dt_s,
                "tracked_count": int(len(current_points)),
                "inlier_count": int(inlier_count),
                "translation_scale_m": float(scale_m),
            },
        )

        keypoints = self._detect_features(gray)
        if len(keypoints) < self.config.min_tracked_points:
            keypoints = current_points.reshape(-1, 1, 2).astype(np.float32)

        self._previous_frame = _TrackedFrame(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            gray=gray,
            keypoints_px=keypoints,
            camera_position_local_ned_m=vec3(current_position),
            camera_attitude_quaternion=camera_attitude,
        )
        self._latest_measurement = measurement
        self._set_status("measurement_ready", **measurement.raw)
        return measurement

    def _decode_gray(self, jpeg_bytes: bytes) -> np.ndarray | None:
        try:
            import cv2
        except ImportError:
            self._set_status("opencv_unavailable")
            return None

        image = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            return None
        return image

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        import cv2

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(self.config.max_corners),
            qualityLevel=float(self.config.quality_level),
            minDistance=float(self.config.min_distance_px),
            blockSize=int(self.config.block_size_px),
        )
        if points is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        return points.astype(np.float32)

    def _track_features(
        self,
        previous_gray: np.ndarray,
        gray: np.ndarray,
        previous_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        if previous_points.size == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            previous_points,
            None,
            winSize=(int(self.config.lk_window_px), int(self.config.lk_window_px)),
            maxLevel=int(self.config.lk_max_level),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(self.config.lk_max_iterations),
                float(self.config.lk_epsilon),
            ),
        )
        if current_points is None or status is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
        valid = status.reshape(-1).astype(bool)
        return previous_points.reshape(-1, 2)[valid], current_points.reshape(-1, 2)[valid]

    def _estimate_relative_camera_pose(
        self,
        previous_points: np.ndarray,
        current_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        import cv2

        assert self.camera is not None
        essential, mask = cv2.findEssentialMat(
            previous_points,
            current_points,
            self.camera.matrix,
            method=cv2.RANSAC,
            prob=float(self.config.ransac_prob),
            threshold=float(self.config.ransac_threshold_px),
        )
        if essential is None:
            return None
        if essential.shape[0] > 3:
            essential = essential[:3, :]
        recovered_count, rotation_21, translation_21, pose_mask = cv2.recoverPose(
            essential,
            previous_points,
            current_points,
            self.camera.matrix,
            mask=mask,
        )
        inlier_count = int(recovered_count)
        if pose_mask is not None:
            inlier_count = int(np.count_nonzero(pose_mask))
        return rotation_21, translation_21, inlier_count

    def _imu_translation_scale_m(self, start_sim_time_ns: int, end_sim_time_ns: int) -> float | None:
        samples = [
            sample
            for sample in self._imu_samples
            if int(start_sim_time_ns) <= int(sample.time_boot_us) * 1_000 <= int(end_sim_time_ns)
        ]
        if len(samples) < 2:
            return None

        velocity = np.zeros(3, dtype=float)
        displacement = np.zeros(3, dtype=float)
        previous = samples[0]
        for sample in samples[1:]:
            dt_s = (int(sample.time_boot_us) - int(previous.time_boot_us)) / 1_000_000.0
            if dt_s <= 0.0:
                previous = sample
                continue
            acceleration = np.asarray(sample.acceleration_body_frd_mps2, dtype=float)
            displacement += velocity * dt_s + 0.5 * acceleration * dt_s * dt_s
            velocity += acceleration * dt_s
            previous = sample

        displacement_norm = float(np.linalg.norm(displacement))
        if not np.isfinite(displacement_norm) or displacement_norm <= 1e-4:
            return None
        return displacement_norm

    def _replace_previous_frame(self, frame: "VisionFrame", gray: np.ndarray, *, status: str, **raw: Any) -> None:
        keypoints = self._detect_features(gray)
        previous_position = (0.0, 0.0, 0.0)
        previous_attitude = quaternion_from_rotation_matrix(camera_optical_to_body_frd(self.camera_tilt_deg))
        if self._previous_frame is not None:
            previous_position = self._previous_frame.camera_position_local_ned_m
            previous_attitude = self._previous_frame.camera_attitude_quaternion
        self._previous_frame = _TrackedFrame(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            gray=gray,
            keypoints_px=keypoints,
            camera_position_local_ned_m=previous_position,
            camera_attitude_quaternion=previous_attitude,
        )
        self._set_status(status, frame_id=frame.frame_id, feature_count=len(keypoints), **raw)

    def _confidence(self, *, tracked_count: int, inlier_count: int) -> float:
        tracked_score = min(1.0, tracked_count / max(float(self.config.max_corners), 1.0))
        inlier_score = min(1.0, inlier_count / max(float(self.config.min_inliers * 2), 1.0))
        return max(float(self.config.min_confidence), min(1.0, 0.4 * tracked_score + 0.6 * inlier_score))

    def _set_status(self, status: str, **raw: Any) -> None:
        self._last_status = status
        self._last_raw = dict(raw)


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
    "CameraIntrinsics",
    "OpenCvMonocularVioProvider",
    "VioCorrectionConfig",
    "VioFrontendConfig",
    "VioMeasurement",
    "VioProvider",
    "blend_vio_state",
    "should_apply_vio_measurement",
    "slerp_quaternion",
]
