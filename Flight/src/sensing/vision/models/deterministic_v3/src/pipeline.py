"""Topology-first deterministic-v3 gate-geometry pipeline scaffold.

This module owns orchestration only.  Data contracts belong in ``schema.py``;
image and component construction belong in ``preprocessing.py``; density
calculation and caching belong in ``density_bank.py``; and topology-dependent
fitting belongs in ``standard_gate.py``, ``c_shape/``, or ``multi_gate/``.

`StandardGatePipeline` is the implemented JPEG-to-camera-PnP production slice.
The dependency-injected scaffold remains below it for adding the C-shape and
multi-gate fitters without importing legacy detectors or review code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Protocol

from .density_bank import DensityBank
from .gate_centerline_pnp import (
    CAMERA_CALIBRATION, GATE_MODEL, solve_gate_pose)
from .preprocessing import (
    DEFAULT_CONFIG, PreprocessingConfig, decode_jpeg, load_lut,
    preprocess_frame)
from .schema import (C_SHAPE_ROUTE, MULTI_GATE_ROUTE, STANDARD_ROUTE,
                     GeometryFrameResult, TopologyDecision)
from .standard_gate import fit_standard_gate
from .topology import assess_frame

if TYPE_CHECKING:
    from core.schema import VehicleState, VisionObservation


SUPPORTED_ROUTES = frozenset((STANDARD_ROUTE, C_SHAPE_ROUTE, MULTI_GATE_ROUTE))

PIPELINE_STAGES = (
    "decode_jpeg",
    "preprocess_frame",
    "analyze_topology",
    "select_density_profile",
    "fit_specialized_geometry",
    "normalize_quadrilateral",
    "validate_quadrilateral",
    "solve_camera_pnp",
    "publish_observation",
)

# Target modules from AGENTS.md.  These are documentation, not dynamic imports.
TARGET_IMPLEMENTATION_MODULES = {
    "preprocessing": ".preprocessing",
    "topology": ".topology",
    "density_bank": ".density_bank",
    STANDARD_ROUTE: ".standard_gate",
    C_SHAPE_ROUTE: ".c_shape.process",
    MULTI_GATE_ROUTE: ".multi_gate.process",
}


@dataclass(slots=True)
class StandardGatePipeline:
    """Implemented production slice from JPEG through standard-gate PnP."""

    lut: Any
    config: PreprocessingConfig = DEFAULT_CONFIG

    @classmethod
    def from_default_lut(cls) -> "StandardGatePipeline":
        return cls(load_lut())

    def process_frame(
        self, *, frame_id: int, sim_time_ns: int, jpeg_bytes: bytes
    ) -> GeometryFrameResult:
        image = decode_jpeg(jpeg_bytes)
        frame = preprocess_frame(
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            image=image,
            lut=self.lut,
            config=self.config,
        )
        bank = DensityBank(frame)
        decisions = assess_frame(frame)
        quadrilaterals = []
        poses = []
        for component, decision in zip(frame.components, decisions):
            if not decision.accepted or decision.route != STANDARD_ROUTE:
                continue
            profile = bank.recommended_standard_profile(component)
            density = bank.get(component, profile.profile_id)
            quadrilateral = fit_standard_gate(
                frame, component, decision, density)
            quadrilaterals.append(quadrilateral)
            poses.append(solve_gate_pose(quadrilateral))
        return GeometryFrameResult(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            preprocessing_version=frame.preprocessing_version,
            density_configuration=bank.configuration(),
            camera_calibration=CAMERA_CALIBRATION,
            gate_model=GATE_MODEL,
            processed_routes=(STANDARD_ROUTE,),
            topology_decisions=decisions,
            quadrilateral_estimates=tuple(quadrilaterals),
            camera_pose_estimates=tuple(poses),
        )


class FitOutcome(Protocol):
    """Schema-owned result from exactly one topology-assigned fitter."""

    accepted: bool
    geometry: Any
    rejection_reason: str | None


class ValidationOutcome(Protocol):
    """Schema-owned result from common quadrilateral validation."""

    accepted: bool
    rejection_reason: str | None


Fitter = Callable[[Any, Any], FitOutcome]


class PipelinePorts(Protocol):
    """Shared operations supplied as the new implementation is extracted.

    Production implementations of these methods must return the explicit data
    contracts defined in ``schema.py``.  ``vehicle_state`` is intentionally
    absent until ``publish`` so it cannot influence image-space geometry.
    """

    def decode_jpeg(self, jpeg_bytes: bytes) -> Any: ...

    def preprocess_frame(
        self, *, frame_id: int, sim_time_ns: int, image: Any
    ) -> Any: ...

    def components(self, frame: Any) -> Iterable[Any]: ...

    def classify_topology(
        self, frame: Any, component: Any
    ) -> TopologyDecision: ...

    def select_density(self, component: Any, route: str) -> Any: ...

    def normalize_quadrilaterals(
        self, component: Any, route: str, geometry: Any
    ) -> Any: ...

    def validate_quadrilaterals(
        self, component: Any, route: str, normalized: Any
    ) -> ValidationOutcome: ...

    def solve_camera_pnp(self, component: Any, normalized: Any) -> Any: ...

    def accepted_candidates(
        self,
        component: Any,
        route: str,
        density: Any,
        normalized: Any,
        validation: ValidationOutcome,
        camera_pose: Any,
    ) -> Iterable[Any]: ...

    def rejection(
        self, component: Any, *, stage: str, reason: str, route: str | None
    ) -> Any: ...

    def publish(
        self,
        *,
        frame: Any,
        accepted: tuple[Any, ...],
        rejected: tuple[Any, ...],
        vehicle_state: VehicleState | None,
    ) -> VisionObservation: ...


@dataclass(slots=True)
class GeometryPipeline:
    """Run shared work once and dispatch each component exactly once."""

    ports: PipelinePorts
    fitters: Mapping[str, Fitter]

    def process_frame(
        self,
        *,
        frame_id: int,
        sim_time_ns: int,
        jpeg_bytes: bytes,
        vehicle_state: VehicleState | None = None,
    ) -> VisionObservation:
        """Return one observation while preserving authoritative ingress IDs.

        A rejected fitter never falls through to another topology route.
        Camera-to-NED conversion, when supported by valid vehicle state and
        calibrated extrinsics, occurs only inside the final publisher.
        """
        if not isinstance(jpeg_bytes, bytes):
            raise TypeError("jpeg_bytes must be bytes")

        image = self.ports.decode_jpeg(jpeg_bytes)
        frame = self.ports.preprocess_frame(
            frame_id=int(frame_id), sim_time_ns=int(sim_time_ns), image=image
        )
        accepted: list[Any] = []
        rejected: list[Any] = []

        for component in self.ports.components(frame):
            decision = self.ports.classify_topology(frame, component)
            route = decision.route
            if not decision.accepted:
                rejected.append(self.ports.rejection(
                    component,
                    stage="analyze_topology",
                    reason=decision.rejection_reason or "topology_rejected",
                    route=route or None,
                ))
                continue
            if route not in SUPPORTED_ROUTES:
                rejected.append(self.ports.rejection(
                    component,
                    stage="analyze_topology",
                    reason="unsupported_topology_route",
                    route=route,
                ))
                continue

            fitter = self.fitters.get(route)
            if fitter is None:
                rejected.append(self.ports.rejection(
                    component,
                    stage="fit_specialized_geometry",
                    reason="fitter_unavailable",
                    route=route,
                ))
                continue

            density = self.ports.select_density(component, route)
            fit = fitter(component, density)
            if not fit.accepted:
                rejected.append(self.ports.rejection(
                    component,
                    stage="fit_specialized_geometry",
                    reason=fit.rejection_reason or "geometry_fit_rejected",
                    route=route,
                ))
                continue

            normalized = self.ports.normalize_quadrilaterals(
                component, route, fit.geometry
            )
            validation = self.ports.validate_quadrilaterals(
                component, route, normalized
            )
            if not validation.accepted:
                rejected.append(self.ports.rejection(
                    component,
                    stage="validate_quadrilateral",
                    reason=(validation.rejection_reason
                            or "quadrilateral_validation_rejected"),
                    route=route,
                ))
                continue

            camera_pose = self.ports.solve_camera_pnp(component, normalized)
            if camera_pose is None:
                rejected.append(self.ports.rejection(
                    component,
                    stage="solve_camera_pnp",
                    reason="camera_pnp_rejected",
                    route=route,
                ))
                continue
            accepted.extend(self.ports.accepted_candidates(
                component, route, density, normalized, validation, camera_pose
            ))

        observation = self.ports.publish(
            frame=frame,
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            vehicle_state=vehicle_state,
        )
        if (observation.frame_id != int(frame_id)
                or observation.sim_time_ns != int(sim_time_ns)):
            raise ValueError("publisher changed authoritative frame identity")
        return observation
