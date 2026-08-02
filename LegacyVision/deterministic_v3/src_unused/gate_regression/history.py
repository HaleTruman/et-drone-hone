"""Retained observation history for active gate-regression tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fusion import FusedPose


@dataclass(frozen=True, slots=True)
class GateObservation:
    """One immutable accepted PnP observation retained for audit/replay."""

    frame_id: int
    sim_time_ns: int
    candidate_rank: int
    position_camera_m: tuple[float, float, float]
    rotation_vector_model_to_camera: tuple[float, float, float]
    reprojection_rmse_px: float
    evidence_weight: float


@dataclass(slots=True)
class GateTrack:
    """All observations plus the bounded-cost posterior for one active gate."""

    track_id: str
    last_seen_tick: int
    posterior: FusedPose | None = None
    anchor_origin: np.ndarray | None = None
    anchor_world_from_camera: np.ndarray | None = None
    observations: list[GateObservation] = field(default_factory=list)

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    def append(self, observation: GateObservation) -> None:
        self.observations.append(observation)
