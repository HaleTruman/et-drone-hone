"""Stateful post-PnP gate pose regression and association orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from ..configurations import DEFAULT_GATE_REGRESSION_CONFIGURATION
from ..schema import (
    GateRegressionConfiguration,
    GeometryFrameResult,
    PnPCandidateEstimate,
    PnPRelativePoseEstimate,
    QuadrilateralEstimate,
)
from ..void_ned import camera_position_ned, rotation_world_from_camera
from .fusion import (
    FusedPose,
    decay,
    initial_pose,
    reframe,
    robust_fuse,
    rotation_matrix,
    rotation_residual_degrees,
    rotation_vector,
)
from .history import GateObservation, GateTrack
from .publication import publish_pose


REGRESSOR_NAME = "post-pnp-gate-regression-v2"


@dataclass(frozen=True, slots=True)
class _PoseInput:
    raw_pose: PnPRelativePoseEstimate
    quadrilateral: QuadrilateralEstimate | None
    candidates: tuple[PnPCandidateEstimate, ...]
    association_candidate: PnPCandidateEstimate
    association_covariance: np.ndarray
    center_uv: np.ndarray
    image_scale_px: float
    binding_key: tuple[str, int, int]


def _project_camera(
    position: np.ndarray,
    camera_matrix: np.ndarray,
    minimum_depth_m: float,
) -> np.ndarray | None:
    if position[2] <= minimum_depth_m:
        return None
    projected = camera_matrix @ position
    return projected[:2] / projected[2]


def _paired_quadrilaterals(
    frame: GeometryFrameResult,
) -> dict[tuple[int, str, int], QuadrilateralEstimate]:
    return {
        (item.component_id, item.route, item.gate_index): item
        for item in frame.quadrilateral_estimates
        if item.frame_id == frame.frame_id and item.sim_time_ns == frame.sim_time_ns
    }


def _candidate_by_rank(
    raw_pose: PnPRelativePoseEstimate,
) -> PnPCandidateEstimate | None:
    if not raw_pose.candidates:
        return None
    requested = raw_pose.selected_candidate_rank
    if requested is not None:
        for candidate in raw_pose.candidates:
            if candidate.candidate_rank == requested:
                return candidate
    return min(raw_pose.candidates, key=lambda item: item.candidate_rank)


def _metric_covariance(
    position: np.ndarray,
    rmse_px: float,
    quadrilateral: QuadrilateralEstimate | None,
    camera_matrix: np.ndarray,
    pixel_noise_px: float,
) -> np.ndarray:
    span = 10.0
    if quadrilateral is not None and quadrilateral.corners_uv is not None:
        points = np.asarray(quadrilateral.corners_uv, np.float64)
        if points.shape == (4, 2) and np.all(np.isfinite(points)):
            span = max(10.0, float(np.max(np.linalg.norm(
                points - np.mean(points, axis=0), axis=1
            ))))
    depth = float(position[2])
    direction = position / max(float(np.linalg.norm(position)), 1e-12)
    focal = math.sqrt(float(camera_matrix[0, 0] * camera_matrix[1, 1]))
    sigma_depth = max(0.035 * depth, depth * max(pixel_noise_px, rmse_px) / span)
    bearing_sigma = depth * pixel_noise_px / max(focal, 1e-6)
    return (
        sigma_depth * sigma_depth * np.outer(direction, direction)
        + bearing_sigma * bearing_sigma
        * (np.eye(3) - np.outer(direction, direction))
    )


def _pose_inputs(
    frame: GeometryFrameResult,
    configuration: GateRegressionConfiguration,
) -> tuple[_PoseInput, ...]:
    camera_matrix = np.asarray(frame.camera_calibration.camera_matrix, np.float64)
    quadrilaterals = _paired_quadrilaterals(frame)
    inputs: list[_PoseInput] = []
    for raw_pose in frame.pnp_relative_pose_estimates:
        if raw_pose.frame_id != frame.frame_id or raw_pose.sim_time_ns != frame.sim_time_ns:
            raise ValueError("PnP pose identity does not match geometry frame")
        if not raw_pose.accepted:
            continue
        association_candidate = _candidate_by_rank(raw_pose)
        if association_candidate is None:
            continue
        valid_candidates = tuple(
            candidate for candidate in raw_pose.candidates
            if (
                len(candidate.position_camera_m) == 3
                and len(candidate.rotation_vector_model_to_camera) == 3
                and np.all(np.isfinite(candidate.position_camera_m))
                and np.all(np.isfinite(candidate.rotation_vector_model_to_camera))
                and candidate.position_camera_m[2] > configuration.minimum_camera_depth_m
                and math.isfinite(candidate.reprojection_rmse_px)
            )
        )
        if not valid_candidates or association_candidate not in valid_candidates:
            continue
        quadrilateral = quadrilaterals.get(
            (raw_pose.component_id, raw_pose.route, raw_pose.gate_index)
        )
        position = np.asarray(association_candidate.position_camera_m, np.float64)
        projected = _project_camera(
            position, camera_matrix, configuration.minimum_camera_depth_m
        )
        if projected is None:
            continue
        center_uv = projected
        image_scale_px = configuration.association_floor_px / 0.30
        if quadrilateral is not None and quadrilateral.corners_uv is not None:
            points = np.asarray(quadrilateral.corners_uv, np.float64)
            if points.shape == (4, 2) and np.all(np.isfinite(points)):
                center_uv = np.mean(points, axis=0)
                image_scale_px = max(
                    float(np.ptp(points[:, 0])),
                    float(np.ptp(points[:, 1])),
                    image_scale_px,
                )
        inputs.append(_PoseInput(
            raw_pose=raw_pose,
            quadrilateral=quadrilateral,
            candidates=valid_candidates,
            association_candidate=association_candidate,
            association_covariance=_metric_covariance(
                position,
                association_candidate.reprojection_rmse_px,
                quadrilateral,
                camera_matrix,
                configuration.pixel_noise_px,
            ),
            center_uv=center_uv,
            image_scale_px=image_scale_px,
            binding_key=(raw_pose.route, raw_pose.component_id, raw_pose.gate_index),
        ))
    return tuple(inputs)


class GatePoseRegressor:
    """Publish authoritative full poses from consecutive raw PnP evidence."""

    def __init__(
        self,
        configuration: GateRegressionConfiguration = DEFAULT_GATE_REGRESSION_CONFIGURATION,
    ) -> None:
        self.configuration = configuration
        self._states: dict[str, GateTrack] = {}
        self._bindings: dict[tuple[str, int, int], str] = {}
        self._tick = 0
        self._sequence = 0
        self._last_frame_key: tuple[int, int] | None = None
        self._last_result: GeometryFrameResult | None = None

    @property
    def active_track_count(self) -> int:
        return len(self._states)

    @property
    def active_observation_count(self) -> int:
        return sum(state.observation_count for state in self._states.values())

    def _fresh(self, binding_key: tuple[str, int, int]) -> GateTrack:
        track_id = f"pnp-gate-{self._sequence}"
        self._sequence += 1
        state = GateTrack(track_id=track_id, last_seen_tick=self._tick)
        self._states[track_id] = state
        self._bindings[binding_key] = track_id
        return state

    def _association(
        self,
        state: GateTrack,
        item: _PoseInput,
        origin: np.ndarray,
        world_from_camera: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> float | None:
        if state.posterior is None:
            return 0.0
        prior = reframe(
            state.posterior,
            state.anchor_origin,
            state.anchor_world_from_camera,
            origin,
            world_from_camera,
        )
        predicted = None if prior is None else _project_camera(
            prior.position,
            camera_matrix,
            self.configuration.minimum_camera_depth_m,
        )
        if prior is None or predicted is None:
            return None
        current_position = np.asarray(
            item.association_candidate.position_camera_m, np.float64
        )
        pixel_error = float(np.linalg.norm(predicted - item.center_uv))
        pixel_limit = max(
            self.configuration.association_floor_px, 0.30 * item.image_scale_px
        )
        position_error = float(np.linalg.norm(prior.position - current_position))
        position_limit = max(0.75, 0.12 * float(np.linalg.norm(current_position)))
        if pixel_error > pixel_limit or position_error > position_limit:
            return None
        return (
            0.65 * pixel_error / pixel_limit
            + 0.35 * position_error / position_limit
        )

    def _assign(
        self,
        inputs: tuple[_PoseInput, ...],
        origin: np.ndarray,
        world_from_camera: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> dict[int, GateTrack]:
        assigned: set[str] = set()
        result: dict[int, GateTrack] = {}
        pending: list[_PoseInput] = []
        for item in inputs:
            track_id = self._bindings.get(item.binding_key)
            state = self._states.get(track_id) if track_id else None
            score = (
                self._association(state, item, origin, world_from_camera, camera_matrix)
                if state is not None else None
            )
            if state is not None and score is not None and state.track_id not in assigned:
                result[id(item.raw_pose)] = state
                assigned.add(state.track_id)
            else:
                if track_id is not None:
                    self._bindings.pop(item.binding_key, None)
                pending.append(item)

        choices = []
        for item in pending:
            for state in self._states.values():
                if (
                    state.track_id in assigned
                    or self._tick - state.last_seen_tick
                    > self.configuration.maximum_missed_frames
                ):
                    continue
                score = self._association(
                    state, item, origin, world_from_camera, camera_matrix
                )
                if score is not None:
                    choices.append((score, id(item.raw_pose), item, state))
        used: set[int] = set()
        for _, identity, item, state in sorted(choices, key=lambda value: value[0]):
            if identity in used or state.track_id in assigned:
                continue
            result[identity] = state
            self._bindings[item.binding_key] = state.track_id
            used.add(identity)
            assigned.add(state.track_id)
        for item in pending:
            if id(item.raw_pose) not in result:
                result[id(item.raw_pose)] = self._fresh(item.binding_key)
        return result

    def _select_candidate(
        self,
        state: GateTrack,
        item: _PoseInput,
        prior: FusedPose | None,
    ) -> tuple[PnPCandidateEstimate, float | None]:
        if prior is None:
            candidate = _candidate_by_rank(item.raw_pose)
            assert candidate is not None
            return candidate, None
        ranked = []
        for candidate in item.candidates:
            candidate_rotation = rotation_matrix(
                candidate.rotation_vector_model_to_camera
            )
            residual = rotation_residual_degrees(
                candidate_rotation, prior.rotation_model_to_camera
            )
            ranked.append((
                residual,
                candidate.reprojection_rmse_px,
                candidate.candidate_rank,
                candidate,
            ))
        residual, _, _, candidate = min(ranked, key=lambda value: value[:3])
        return candidate, float(residual)

    def _update_state(
        self,
        state: GateTrack,
        item: _PoseInput,
        origin: np.ndarray,
        world_from_camera: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> tuple[FusedPose, PnPCandidateEstimate, float | None]:
        prior = reframe(
            state.posterior,
            state.anchor_origin,
            state.anchor_world_from_camera,
            origin,
            world_from_camera,
        )
        if prior is not None:
            prior = decay(
                prior,
                self._tick - state.last_seen_tick,
                self.configuration.history_half_life_frames,
            )
        candidate, candidate_residual = self._select_candidate(state, item, prior)
        position = np.asarray(candidate.position_camera_m, np.float64)
        covariance = _metric_covariance(
            position,
            candidate.reprojection_rmse_px,
            item.quadrilateral,
            camera_matrix,
            self.configuration.pixel_noise_px,
        )
        rotation = rotation_matrix(candidate.rotation_vector_model_to_camera)
        fit_quality = 1.0 / (1.0 + (candidate.reprojection_rmse_px / 3.0) ** 2)
        quality = float(np.clip(
            item.raw_pose.position_confidence
            if candidate.candidate_rank == item.raw_pose.selected_candidate_rank
            else fit_quality,
            1e-3,
            1.0,
        ))
        if prior is None:
            posterior = initial_pose(
                position=position,
                covariance=covariance,
                rotation=rotation,
                reprojection_rmse_px=candidate.reprojection_rmse_px,
                quality=quality,
            )
            fused_residual = None
        else:
            posterior, fused_residual = robust_fuse(
                prior,
                position=position,
                covariance=covariance,
                rotation=rotation,
                reprojection_rmse_px=candidate.reprojection_rmse_px,
                quality=quality,
                observation_count=state.observation_count + 1,
            )
        state.append(GateObservation(
            frame_id=item.raw_pose.frame_id,
            sim_time_ns=item.raw_pose.sim_time_ns,
            candidate_rank=candidate.candidate_rank,
            position_camera_m=tuple(map(float, candidate.position_camera_m)),
            rotation_vector_model_to_camera=tuple(map(
                float, candidate.rotation_vector_model_to_camera
            )),
            reprojection_rmse_px=float(candidate.reprojection_rmse_px),
            evidence_weight=quality,
        ))
        state.posterior = posterior
        state.anchor_origin = origin.copy()
        state.anchor_world_from_camera = world_from_camera.copy()
        state.last_seen_tick = self._tick
        return posterior, candidate, (
            fused_residual if fused_residual is not None else candidate_residual
        )

    def refine(
        self,
        frame: GeometryFrameResult,
        vehicle_state: Any | None = None,
    ) -> GeometryFrameResult:
        """Return raw PnP plus authoritative regressed poses for this frame."""
        frame_key = (int(frame.frame_id), int(frame.sim_time_ns))
        if frame_key == self._last_frame_key and self._last_result is not None:
            return self._last_result
        self._tick += 1
        stale = {
            track_id for track_id, state in self._states.items()
            if self._tick - state.last_seen_tick
            > self.configuration.maximum_missed_frames
        }
        for track_id in stale:
            self._states.pop(track_id, None)
        self._bindings = {
            key: track_id for key, track_id in self._bindings.items()
            if track_id in self._states
        }

        result = replace(
            frame,
            gate_regression_configuration=self.configuration,
            camera_pose_estimates=(),
        )
        inputs = _pose_inputs(frame, self.configuration)
        if vehicle_state is not None and inputs:
            origin = camera_position_ned(vehicle_state)
            world_from_camera = rotation_world_from_camera(
                vehicle_state.attitude_quaternion
            )
            camera_matrix = np.asarray(
                frame.camera_calibration.camera_matrix, np.float64
            )
            assignments = self._assign(
                inputs, origin, world_from_camera, camera_matrix
            )
            final_poses = []
            for item in inputs:
                state = assignments[id(item.raw_pose)]
                posterior, candidate, residual = self._update_state(
                    state, item, origin, world_from_camera, camera_matrix
                )
                final_poses.append(publish_pose(
                    raw_pose=item.raw_pose,
                    posterior=posterior,
                    quadrilateral=item.quadrilateral,
                    calibration=frame.camera_calibration,
                    gate_model=frame.gate_model,
                    configuration=self.configuration,
                    track_id=state.track_id,
                    selected_candidate_rank=candidate.candidate_rank,
                    observation_count=state.observation_count,
                    orientation_residual_deg=residual,
                ))
            result = replace(result, camera_pose_estimates=tuple(final_poses))

        self._last_frame_key = frame_key
        self._last_result = result
        return result


# Compatibility name for callers of the original single-file implementation.
PostPnPGateRegressor = GatePoseRegressor
