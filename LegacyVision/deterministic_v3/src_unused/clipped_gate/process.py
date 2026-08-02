"""Minimal prediction over one trusted clipped component mask.

The process does not classify clipping and does not infer a gate pose.  It
accepts a component whose historic ``TopologyDecision`` already says
``clipped``, isolates its raw and post-close masks, and describes the observed
frame-boundary contact for offline review.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


PROCESS_VERSION = "clipped-mask-status-v1"
SIDES = ("left", "right", "top", "bottom")
_DIRECTIONS = {
    "left": (-1, 0),
    "right": (1, 0),
    "top": (0, -1),
    "bottom": (0, 1),
}
_ADJACENT_PAIRS = {
    frozenset(("left", "top")),
    frozenset(("left", "bottom")),
    frozenset(("right", "top")),
    frozenset(("right", "bottom")),
}


def _integer_tuple(value: Any, length: int, field: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain {length} integers") from error
    if len(result) != length:
        raise ValueError(f"{field} must contain {length} integers")
    return result


def _identity(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["frame_id"]),
        int(record["sim_time_ns"]),
        int(record["component_id"]),
    )


def _intervals(values: np.ndarray) -> list[list[int]]:
    """Return sorted half-open intervals for one set of integer positions."""
    positions = np.unique(np.asarray(values, dtype=np.int64))
    if positions.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(positions) > 1)
    starts = np.concatenate((positions[:1], positions[breaks + 1]))
    ends = np.concatenate((positions[breaks] + 1, positions[-1:] + 1))
    return [[int(start), int(end)] for start, end in zip(starts, ends)]


def _row_runs(mask: np.ndarray) -> list[list[int]]:
    """Encode a full-frame boolean mask as ``[v, u0, u1)`` row runs."""
    inside = np.asarray(mask, dtype=bool)
    runs: list[list[int]] = []
    for row in np.flatnonzero(np.any(inside, axis=1)):
        for start, end in _intervals(np.flatnonzero(inside[row])):
            runs.append([int(row), start, end])
    return runs


def _contact_positions(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return np.flatnonzero(mask[:, 0])
    if side == "right":
        return np.flatnonzero(mask[:, -1])
    if side == "top":
        return np.flatnonzero(mask[0, :])
    if side == "bottom":
        return np.flatnonzero(mask[-1, :])
    raise ValueError(f"unknown frame side: {side}")


def _touched_sides(
    bbox_xywh: tuple[int, int, int, int], image_shape: tuple[int, int]
) -> tuple[str, ...]:
    x, y, width, height = bbox_xywh
    frame_height, frame_width = image_shape
    sides = []
    if x <= 0:
        sides.append("left")
    if x + width >= frame_width:
        sides.append("right")
    if y <= 0:
        sides.append("top")
    if y + height >= frame_height:
        sides.append("bottom")
    return tuple(sides)


def _pattern(sides: tuple[str, ...]) -> str:
    if len(sides) == 1:
        return "single_edge"
    if len(sides) == 2:
        return (
            "adjacent_corner"
            if frozenset(sides) in _ADJACENT_PAIRS
            else "opposite_edges"
        )
    if len(sides) > 2:
        return "multiple_edges"
    return "edge_unspecified"


def _side_evidence(
    raw_mask: np.ndarray,
    closed_mask: np.ndarray,
    bbox_sides: tuple[str, ...],
) -> list[dict[str, Any]]:
    evidence = []
    for side in SIDES:
        raw_positions = _contact_positions(raw_mask, side)
        closed_positions = _contact_positions(closed_mask, side)
        if raw_positions.size:
            contact_source = "raw_and_closed"
        elif closed_positions.size:
            contact_source = "close_induced"
        else:
            contact_source = "none"
        evidence.append({
            "side": side,
            "bbox_touches": side in bbox_sides,
            "raw_contact_pixels": int(raw_positions.size),
            "closed_contact_pixels": int(closed_positions.size),
            "raw_contact_runs": _intervals(raw_positions),
            "closed_contact_runs": _intervals(closed_positions),
            "contact_source": contact_source,
        })
    return evidence


def process_clipped_mask(
    *,
    source: Mapping[str, Any],
    frame: Mapping[str, Any],
    component: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce one review prediction from an authoritative clipped decision.

    ``decision.topology_label`` is trusted input.  The mask analysis only
    describes which recorded frame sides are involved; it never decides whether
    the component is clipped.
    """
    if decision.get("topology_label") != "clipped":
        raise ValueError("process_clipped_mask requires a trusted clipped decision")

    frame_identity = (int(frame["frame_id"]), int(frame["sim_time_ns"]))
    component_identity = (
        frame_identity[0], frame_identity[1], int(component["component_id"])
    )
    decision_identity = _identity(decision)
    if component_identity != decision_identity:
        raise ValueError("component and topology decision identity differ")
    if component_identity[:2] != frame_identity:
        raise ValueError("component identity differs from FrameObservation")

    image_shape = _integer_tuple(frame["image_shape"], 2, "image_shape")
    bbox = _integer_tuple(component["bbox_xywh"], 4, "bbox_xywh")
    frame_height, frame_width = image_shape
    labels = np.asarray(frame["component_labels"])
    closed_plane = np.asarray(frame["closed_mask"]) != 0
    raw_plane = np.asarray(frame["size_filtered_mask"]) != 0
    expected_shape = (frame_height, frame_width)
    if (labels.shape != expected_shape or closed_plane.shape != expected_shape
            or raw_plane.shape != expected_shape):
        raise ValueError("serialized frame mask shapes do not match image_shape")

    component_id = component_identity[2]
    owned = labels == component_id
    closed_component = owned & closed_plane
    raw_component = owned & raw_plane
    close_added = closed_component & ~raw_component
    sides = _touched_sides(bbox, image_shape)
    side_evidence = _side_evidence(raw_component, closed_component, sides)

    return {
        "process_version": PROCESS_VERSION,
        "source": {
            "run_id": str(source.get("run_id", "")),
            "relative_path": source.get("relative_path"),
            "review_json_path": source.get("review_json_path"),
        },
        "frame_id": component_identity[0],
        "sim_time_ns": component_identity[1],
        "component_id": component_id,
        "image_shape": list(image_shape),
        "bbox_xywh": list(bbox),
        "image_origin_uv": list(_integer_tuple(
            component["image_origin_uv"], 2, "image_origin_uv")),
        "analysis_shape": list(_integer_tuple(
            component["analysis_shape"], 2, "analysis_shape")),
        "prediction": {
            "status": "clipped",
            "pattern": _pattern(sides),
            "clipped_sides": list(sides),
            "out_of_frame_directions": [list(_DIRECTIONS[side]) for side in sides],
            "authority": "trusted_review_topology_decision",
            "gate_geometry_status": "not_attempted",
            "routing_eligible": False,
            "fitter_ready": False,
        },
        "topology_evidence": {
            "classification_rule": decision.get("classification_rule"),
            "rejection_reason": decision.get("rejection_reason"),
            "raw_significant_holes": int(
                decision.get("raw_significant_holes", 0)),
            "closed_significant_holes": int(
                decision.get("closed_significant_holes", 0)),
            "significant_parent_child_contour_ids": [
                list(pair) for pair in decision.get(
                    "significant_parent_child_contour_ids", ())
            ],
            "foreground_distance_max_px": float(
                decision.get("foreground_distance_max_px", 0.0)),
        },
        "mask_evidence": {
            "encoding": "full-frame-row-runs-v1",
            "closed_foreground_px": int(np.count_nonzero(closed_component)),
            "raw_foreground_px": int(np.count_nonzero(raw_component)),
            "close_added_px": int(np.count_nonzero(close_added)),
            "closed_runs": _row_runs(closed_component),
            "raw_runs": _row_runs(raw_component),
            "close_added_runs": _row_runs(close_added),
            "boundary_sides": side_evidence,
        },
    }
