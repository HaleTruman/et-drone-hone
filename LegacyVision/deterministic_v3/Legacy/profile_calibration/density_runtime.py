"""Exact production-density adapter and review-only visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

import cv2
import numpy as np

from ..density_bank import DensityBank
from ..schema import (
    ComponentObservation,
    ContourEvidence,
    DensityBankConfiguration,
    DensityEvidence,
    DensityProfile,
    FrameObservation,
    TopologyEvidence,
)


@dataclass(frozen=True, slots=True)
class CandidateSettings:
    density_radius_px: int
    ridge_radius_px: int
    relative_cap: float
    inverse_gamma: float
    ridge_gamma: float

    @classmethod
    def from_profile(cls, profile: DensityProfile) -> "CandidateSettings":
        return cls(
            profile.density_radius_px,
            profile.ridge_radius_px,
            profile.relative_cap,
            profile.inverse_gamma,
            profile.ridge_gamma,
        )

    def validate(self) -> "CandidateSettings":
        if self.density_radius_px < 1 or self.ridge_radius_px < 1:
            raise ValueError("density and ridge radii must be positive integers")
        values = (self.relative_cap, self.inverse_gamma, self.ridge_gamma)
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("relative cap and gamma values must be finite and positive")
        return self


def _synthetic_frame(mask: np.ndarray) -> tuple[FrameObservation, ComponentObservation]:
    """Wrap one saved production-local mask for an exact DensityBank call."""
    mask = (np.asarray(mask) != 0).astype(np.uint8)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("component mask must be a non-empty two-dimensional array")
    height, width = mask.shape
    if height != width:
        raise ValueError("production component analysis masks must be square")
    labels = mask.astype(np.int32)
    topology = TopologyEvidence((), (), (), ())
    contours = ContourEvidence(None, None, None, None, None, (), ())
    component = ComponentObservation(
        component_id=1,
        bbox_xywh=(0, 0, width, height),
        image_origin_uv=(0, 0),
        analysis_shape=(height, width),
        touches_frame=False,
        area_px=int(np.count_nonzero(mask)),
        fill_ratio=float(np.count_nonzero(mask) / mask.size),
        solidity=1.0,
        topology=topology,
        contours=contours,
        distance_transform=cv2.distanceTransform(mask, cv2.DIST_L2, 5),
    )
    frame = FrameObservation(
        frame_id=0,
        sim_time_ns=0,
        image_shape=(height, width),
        preprocessing_version="profile-calibration-saved-closed-mask-v1",
        base_mask=mask,
        size_filtered_mask=mask,
        closed_mask=mask,
        component_labels=labels,
        closed_contour_nodes=(),
        components=(component,),
        input_component_count=1,
        gated=False,
        rejection_reason=None,
    )
    return frame, component


def compute_density_evidence(
    mask: np.ndarray,
    profiles: tuple[DensityProfile, ...],
    selected_profile_id: str,
    settings: CandidateSettings | None = None,
) -> DensityEvidence:
    """Call the production DensityBank with a baseline or candidate deck."""
    selected = next(
        (profile for profile in profiles
         if profile.profile_id == selected_profile_id),
        None,
    )
    if selected is None:
        raise KeyError(f"unknown density profile: {selected_profile_id}")
    if settings is None:
        configuration_version = selected.calibration_version
        resolved_profiles = profiles
    else:
        settings.validate()
        configuration_version = (
            f"profile-calibration-candidate:{selected_profile_id}:"
            f"dr{settings.density_radius_px}:rr{settings.ridge_radius_px}:"
            f"cap{settings.relative_cap:g}:ig{settings.inverse_gamma:g}:"
            f"rg{settings.ridge_gamma:g}"
        )
        resolved_profiles = tuple(
            replace(
                profile,
                calibration_version=configuration_version,
                **({
                    "density_radius_px": settings.density_radius_px,
                    "ridge_radius_px": settings.ridge_radius_px,
                    "relative_cap": settings.relative_cap,
                    "inverse_gamma": settings.inverse_gamma,
                    "ridge_gamma": settings.ridge_gamma,
                } if profile.profile_id == selected_profile_id else {}),
            )
            for profile in profiles
        )
    configuration = DensityBankConfiguration(
        calibration_version=configuration_version,
        ignore_frame_edge_clipped=False,
        profiles=resolved_profiles,
    )
    frame, component = _synthetic_frame(mask)
    return DensityBank(frame, configuration).get(component, selected_profile_id)


def _fit_panel(image: np.ndarray, side: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(side / max(width, 1), side / max(height, 1))
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_NEAREST,
    )
    panel = np.zeros((side, side, 3), np.uint8)
    top = (side - resized.shape[0]) // 2
    left = (side - resized.shape[1]) // 2
    panel[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return panel


def _heat(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.rint(np.clip(field, 0, 1) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    heat[np.asarray(mask) == 0] = 0
    return heat


def _binary(mask: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.asarray(mask) != 0).astype(np.uint8) * 255,
                        cv2.COLOR_GRAY2BGR)


def _labeled_panel(image: np.ndarray, label: str, side: int = 180) -> np.ndarray:
    header = 24
    panel = np.zeros((header + side, side, 3), np.uint8)
    panel[header:] = _fit_panel(image, side)
    cv2.putText(panel, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (225, 225, 225), 1, cv2.LINE_AA)
    return panel


def comparison_image(
    source_crop: np.ndarray,
    mask: np.ndarray,
    baseline: DensityEvidence | None,
    candidate: DensityEvidence | None,
) -> np.ndarray:
    """Render immutable baseline beside the live candidate evidence."""
    if baseline is None or candidate is None:
        excluded = np.zeros((*mask.shape, 3), np.uint8)
        cv2.putText(excluded, "density excluded", (4, max(16, mask.shape[0] // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1,
                    cv2.LINE_AA)
        baseline_field = baseline_p90 = candidate_field = excluded
        candidate_p70 = candidate_p80 = candidate_p90 = excluded
    else:
        baseline_field = _heat(baseline.final_field, mask)
        baseline_p90 = _binary(baseline.p90_mask)
        candidate_field = _heat(candidate.final_field, mask)
        candidate_p70 = _binary(candidate.p70_mask)
        candidate_p80 = _binary(candidate.p80_mask)
        candidate_p90 = _binary(candidate.p90_mask)
    panels = (
        _labeled_panel(source_crop, "source bbox"),
        _labeled_panel(_binary(mask), "closed component mask"),
        _labeled_panel(baseline_field, "baseline final_field"),
        _labeled_panel(baseline_p90, "baseline p90_mask"),
        _labeled_panel(candidate_field, "candidate final_field"),
        _labeled_panel(candidate_p70, "candidate p70_mask"),
        _labeled_panel(candidate_p80, "candidate p80_mask"),
        _labeled_panel(candidate_p90, "candidate p90_mask"),
    )
    return np.vstack((np.hstack(panels[:4]), np.hstack(panels[4:])))

