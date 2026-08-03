"""Frame-scoped cache of named component-local inverse-density variants."""

from __future__ import annotations

import cv2
import numpy as np

from .configurations import DEFAULT_DENSITY_CONFIGURATION
from .preprocessing import component_mask
from .schema import (ComponentObservation, DensityBankConfiguration,
                     DensityEvidence, DensityProfile, FrameObservation)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _box_sum(values: np.ndarray, radius: int, ddepth: int = -1) -> np.ndarray:
    window = 2 * radius + 1
    return cv2.boxFilter(
        values, ddepth, (window, window), normalize=False,
        borderType=cv2.BORDER_CONSTANT)


class DensityBank:
    """Compute each ``(component, profile)`` field at most once per frame.

    All profiles remain available regardless of the component's ordinary size
    bucket.  Selection policy belongs to the standard, C-shape, or multi-gate
    profile selector and must record the chosen profile in its result.
    """

    def __init__(
        self,
        frame: FrameObservation,
        configuration: DensityBankConfiguration =
            DEFAULT_DENSITY_CONFIGURATION,
    ) -> None:
        profiles = configuration.profiles
        if not profiles:
            raise ValueError("density bank requires at least one profile")
        profile_map = {profile.profile_id: profile for profile in profiles}
        if len(profile_map) != len(profiles):
            raise ValueError("density profile IDs must be unique")
        for profile in profiles:
            if profile.density_radius_px < 1 or profile.ridge_radius_px < 1:
                raise ValueError("density and ridge radii must be positive")
            calibration_values = (
                profile.relative_cap,
                profile.inverse_gamma,
                profile.ridge_gamma,
            )
            if any(not np.isfinite(value) or value <= 0
                   for value in calibration_values):
                raise ValueError(
                    "density cap and gamma values must be finite and positive")
        area_limits = tuple(
            profile.maximum_component_area_px for profile in profiles)
        if (any(limit is not None and limit < 1 for limit in area_limits) or
                area_limits[-1] is not None or
                any(limit is None for limit in area_limits[:-1]) or
                any(left >= right for left, right in zip(
                    area_limits[:-2], area_limits[1:-1]))):
            raise ValueError(
                "density profile area limits must increase and end unbounded")
        if any(profile.calibration_version != configuration.calibration_version
               for profile in profiles):
            raise ValueError(
                "density profiles must share the configuration version")
        self._configuration = configuration
        self.frame = frame
        self.profiles = profile_map
        self.ignore_frame_edge_clipped = configuration.ignore_frame_edge_clipped
        self._components = {
            component.component_id: component for component in frame.components}
        self._cache: dict[tuple[int, str], DensityEvidence] = {}
        self._coordinate_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._denominator_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def profile_ids(self) -> tuple[str, ...]:
        return tuple(self.profiles)

    def configuration(self) -> DensityBankConfiguration:
        """Return the schema-owned configuration represented by this bank."""
        return self._configuration

    def recommended_standard_profile(
        self, component: ComponentObservation
    ) -> DensityProfile:
        """Return the aggregate non-clipped-area-decile recommendation."""
        for profile in self.profiles.values():
            if (profile.maximum_component_area_px is None or
                    component.area_px <= profile.maximum_component_area_px):
                return profile
        raise ValueError("density bank requires one unbounded profile")

    def includes_component(self, component: ComponentObservation) -> bool:
        """Return whether this component is eligible for density computation."""
        return not (
            self.ignore_frame_edge_clipped and component.touches_frame)

    def get(
        self, component: ComponentObservation, profile_id: str
    ) -> DensityEvidence:
        """Return a cached analytical variant, computing it on first use."""
        owned = self._components.get(component.component_id)
        if owned is None or owned is not component:
            raise ValueError("component does not belong to this frame density bank")
        if not self.includes_component(component):
            raise ValueError(
                "frame_edge_clipped component is ignored for density")
        try:
            profile = self.profiles[profile_id]
        except KeyError as error:
            raise KeyError(f"unknown density profile: {profile_id}") from error
        key = (component.component_id, profile_id)
        if key not in self._cache:
            self._cache[key] = self._compute(component, profile)
        return self._cache[key]

    def precompute(
        self, component: ComponentObservation
    ) -> tuple[DensityEvidence, ...]:
        """Materialize every variant, or none when density policy ignores it."""
        if not self.includes_component(component):
            return ()
        return tuple(self.get(component, profile_id)
                     for profile_id in self.profile_ids())

    def cached_profile_ids(
        self, component: ComponentObservation
    ) -> tuple[str, ...]:
        return tuple(profile_id for profile_id in self.profile_ids()
                     if (component.component_id, profile_id) in self._cache)

    def _coordinates(self, shape: tuple[int, int]):
        if shape not in self._coordinate_cache:
            # Preserve the accepted production implementation's float64
            # coordinate and moment precision.
            yy, xx = np.indices(shape, dtype=np.float64)
            self._coordinate_cache[shape] = yy, xx
        return self._coordinate_cache[shape]

    def _denominator(self, shape: tuple[int, int], radius: int):
        key = (*shape, radius)
        if key not in self._denominator_cache:
            self._denominator_cache[key] = _box_sum(
                np.ones(shape, np.float32), radius, cv2.CV_32F)
        return self._denominator_cache[key]

    def _compute(
        self, component: ComponentObservation, profile: DensityProfile
    ) -> DensityEvidence:
        mask = component_mask(self.frame, component, stage="closed")
        inside = mask != 0
        foreground = inside.astype(np.float32)
        density = _box_sum(
            foreground, profile.density_radius_px, cv2.CV_32F)
        denominator = self._denominator(
            mask.shape, profile.density_radius_px)
        density = np.divide(
            density,
            denominator,
            out=np.zeros_like(density),
            where=denominator > 0,
        )
        density[~inside] = 0

        normalized = np.zeros_like(density)
        mean_density = float(density[inside].mean()) if np.any(inside) else 0.0
        if mean_density > 0:
            normalized[inside] = np.clip(
                density[inside] / (mean_density * profile.relative_cap), 0, 1)
        inverse = np.zeros_like(density)
        inverse[inside] = np.power(
            1.0 - normalized[inside], 1.0 / profile.inverse_gamma)

        yy, xx = self._coordinates(mask.shape)
        moments = np.stack((inverse, inverse * xx, inverse * yy), axis=-1)
        sums = _box_sum(moments, profile.ridge_radius_px)
        mass, weighted_x, weighted_y = cv2.split(sums)
        centroid_x = np.divide(
            weighted_x, mass, out=np.zeros_like(mass), where=mass > 0)
        centroid_y = np.divide(
            weighted_y, mass, out=np.zeros_like(mass), where=mass > 0)
        distance = cv2.magnitude(xx - centroid_x, yy - centroid_y)
        ridge = np.power(
            1.0 - np.clip(distance / profile.ridge_radius_px, 0, 1),
            profile.ridge_gamma,
        )
        ridge[(~inside) | (mass <= 0)] = 0
        field = inverse * ridge

        positive_selection = inside & (field > 0)
        yy_positive, xx_positive = np.nonzero(positive_selection)
        points = np.column_stack((xx_positive, yy_positive)).astype(np.int32)
        weights = field[yy_positive, xx_positive].astype(np.float32)

        thresholds = {}
        masks = {}
        for percentile in (70, 80, 90):
            threshold = (float(np.percentile(weights, percentile))
                         if weights.size else 0.0)
            evidence_mask = (
                positive_selection & (field >= threshold)).astype(np.uint8)
            thresholds[percentile] = threshold
            masks[percentile] = _readonly(evidence_mask)

        return DensityEvidence(
            component_id=component.component_id,
            profile=profile,
            final_field=_readonly(field),
            positive_points_xy=_readonly(points),
            positive_weights=_readonly(weights),
            p70_threshold=thresholds[70],
            p70_mask=masks[70],
            p80_threshold=thresholds[80],
            p80_mask=masks[80],
            p90_threshold=thresholds[90],
            p90_mask=masks[90],
        )
