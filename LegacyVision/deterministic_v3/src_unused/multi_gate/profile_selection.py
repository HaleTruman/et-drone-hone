"""Named density-profile selection shared by multi-gate stages."""

from __future__ import annotations

from ..configurations import DEFAULT_MULTI_GATE_CONFIGURATION
from ..schema import ComponentObservation, MultiGateConfiguration


def select_multi_gate_density_profile_id(
    component: ComponentObservation,
    configuration: MultiGateConfiguration = DEFAULT_MULTI_GATE_CONFIGURATION,
) -> str:
    """Map the historical dimension bands onto shared density-bank IDs."""
    rules = configuration.density_profile_rules
    if not rules:
        raise ValueError("multi-gate configuration requires profile rules")
    limits = tuple(rule.maximum_component_dimension_px for rule in rules)
    if (
        limits[-1] is not None
        or any(limit is None for limit in limits[:-1])
        or any(limit is not None and limit < 1 for limit in limits)
        or any(
            left >= right
            for left, right in zip(limits[:-2], limits[1:-1])
        )
    ):
        raise ValueError(
            "multi-gate profile limits must increase and end unbounded"
        )
    maximum_dimension = max(component.bbox_xywh[2:])
    for rule in rules:
        if (
            rule.maximum_component_dimension_px is None
            or maximum_dimension <= rule.maximum_component_dimension_px
        ):
            return rule.density_profile_id
    raise AssertionError("the final multi-gate profile rule must be unbounded")
