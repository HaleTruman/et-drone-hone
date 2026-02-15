from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opt_engine.types import Scenario


@dataclass(frozen=True)
class UnrealAdapterConfig:
    cm_to_m: float = 0.01


def scenario_waypoints_to_internal_m(
    scenario: Scenario, *, unreal: UnrealAdapterConfig | None = None
) -> np.ndarray:
    unreal = unreal or UnrealAdapterConfig()

    pts = np.array([[w.x, w.y, w.z] for w in scenario.waypoints], dtype=float)

    if scenario.units == "cm":
        pts *= unreal.cm_to_m
    elif scenario.units == "m":
        pass
    else:
        raise ValueError(f"Unsupported scenario units: {scenario.units}")

    if scenario.frame == "internal":
        return pts
    if scenario.frame == "unreal":
        # MVP: keep axis mapping identity and focus on consistent units.
        # This module exists so we can later apply any required axis/handedness remaps here.
        return pts

    raise ValueError(f"Unsupported scenario frame: {scenario.frame}")
