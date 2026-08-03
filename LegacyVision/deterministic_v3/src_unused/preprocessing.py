"""Shared JPEG, mask, component, contour, and topology-evidence preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schema import (ComponentObservation, ContourEvidence,
                     ContourNodeEvidence, FrameObservation, TopologyEvidence)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LUT_PATH = ROOT / "assets" / "color_lut_v1.npz"

# High-leverage production dial.  Zero disables the initial frame-level
# component-count safety gate; a positive value rejects at ``>=`` that count.
DEFAULT_MAXIMUM_INPUT_COMPONENTS = 0


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Versioned settings applied before every topology-specific operation."""

    version: str = "deterministic-v3-mask-v3:a100:c0-disabled:k5:conn8:tree"
    minimum_component_area_px: int = 100
    maximum_input_components: int = DEFAULT_MAXIMUM_INPUT_COMPONENTS
    close_kernel_px: int = 5
    connectivity: int = 8
    contour_simplify_epsilon_px: float = 3.0
    distance_mask_size: int = 5

    def __post_init__(self) -> None:
        if self.maximum_input_components < 0:
            raise ValueError(
                "maximum_input_components must be zero or positive")


DEFAULT_CONFIG = PreprocessingConfig()


def _readonly(array: np.ndarray | None):
    if array is not None:
        array.setflags(write=False)
    return array


def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode one required JPEG byte payload into a BGR image."""
    if not isinstance(jpeg_bytes, bytes) or not jpeg_bytes:
        raise ValueError("jpeg_bytes must contain an encoded image")
    image = cv2.imdecode(
        np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("unable to decode jpeg_bytes")
    return image


def load_lut(path: str | Path = DEFAULT_LUT_PATH) -> np.ndarray:
    """Load and validate the existing 24-bit BGR lookup table."""
    with np.load(Path(path)) as archive:
        lut = archive["lut"]
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError(f"unexpected LUT shape or dtype: {lut.shape} {lut.dtype}")
    return lut


def image_to_mask(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Return a zero/one LUT mask without modifying the decoded image."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be an HxWx3 uint8 BGR array")
    if lut.shape != (1 << 24,) or lut.dtype != np.uint8:
        raise ValueError("lut must be a uint8 array with 2^24 entries")
    bgr = image.astype(np.uint32)
    keys = (bgr[:, :, 2] << 16) | (bgr[:, :, 1] << 8) | bgr[:, :, 0]
    return (lut[keys] != 0).astype(np.uint8)


def _contour_evidence(mask: np.ndarray):
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return (), (), None, None
    hierarchy = hierarchy[0].astype(np.int32, copy=True)
    hole_indices = [index for index, node in enumerate(hierarchy)
                    if int(node[3]) >= 0]
    holes = tuple(_readonly(contours[index].copy()) for index in hole_indices)
    areas = tuple(float(cv2.contourArea(contours[index]))
                  for index in hole_indices)
    outer_indices = [index for index, node in enumerate(hierarchy)
                     if int(node[3]) < 0]
    outer = (None if not outer_indices else contours[max(
        outer_indices, key=lambda index: cv2.contourArea(contours[index]))].copy())
    return holes, areas, _readonly(outer), _readonly(hierarchy)


def _closed_contour_tree(
    mask: np.ndarray, component_labels: np.ndarray
) -> tuple[ContourNodeEvidence, ...]:
    """Catalog the final closed-mask hierarchy once in full-frame UV space."""
    contours, hierarchy = cv2.findContours(
        mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return ()
    hierarchy = hierarchy[0]
    depths = []
    for contour_id in range(len(contours)):
        depth = 0
        parent_id = int(hierarchy[contour_id, 3])
        while parent_id >= 0:
            depth += 1
            parent_id = int(hierarchy[parent_id, 3])
        depths.append(depth)

    child_ids: list[list[int]] = [[] for _ in contours]
    for child_id, node in enumerate(hierarchy):
        parent_id = int(node[3])
        if parent_id >= 0:
            child_ids[parent_id].append(child_id)

    component_ids = [0] * len(contours)
    for contour_id, (contour, depth) in enumerate(zip(contours, depths)):
        if depth % 2:
            continue
        points = contour[:, 0, :]
        labels = component_labels[points[:, 1], points[:, 0]]
        labels = labels[labels > 0]
        if labels.size == 0:
            raise ValueError("foreground contour has no component label")
        component_ids[contour_id] = int(np.bincount(labels).argmax())

    for contour_id, depth in enumerate(depths):
        if not depth % 2:
            continue
        parent_id = int(hierarchy[contour_id, 3])
        if parent_id < 0 or component_ids[parent_id] <= 0:
            raise ValueError("hole contour has no foreground parent")
        component_ids[contour_id] = component_ids[parent_id]

    return tuple(ContourNodeEvidence(
        contour_id=contour_id,
        component_id=component_ids[contour_id],
        parent_contour_id=(None if int(hierarchy[contour_id, 3]) < 0
                           else int(hierarchy[contour_id, 3])),
        child_contour_ids=tuple(child_ids[contour_id]),
        depth=depths[contour_id],
        is_hole=bool(depths[contour_id] % 2),
        area_px2=float(cv2.contourArea(contour)),
        points_uv=_readonly(contour.copy()),
    ) for contour_id, contour in enumerate(contours))


def _local_masks(
    size_filtered_mask: np.ndarray,
    component_labels: np.ndarray,
    component_id: int,
    bbox_xywh: tuple[int, int, int, int],
    image_origin_uv: tuple[int, int],
    analysis_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    x, y, width, height = bbox_xywh
    origin_x, origin_y = image_origin_uv
    offset_x, offset_y = x - origin_x, y - origin_y
    selected = component_labels[y:y + height, x:x + width] == component_id
    raw_crop = selected & (
        size_filtered_mask[y:y + height, x:x + width] != 0)
    closed = np.zeros(analysis_shape, np.uint8)
    raw = np.zeros(analysis_shape, np.uint8)
    closed[offset_y:offset_y + height, offset_x:offset_x + width] = selected
    raw[offset_y:offset_y + height, offset_x:offset_x + width] = raw_crop
    return raw, closed


def component_mask(
    frame: FrameObservation,
    component: ComponentObservation,
    *,
    stage: str = "closed",
) -> np.ndarray:
    """Reconstruct one component-local mask from authoritative frame planes."""
    if stage not in {"raw", "closed"}:
        raise ValueError("stage must be 'raw' or 'closed'")
    raw, closed = _local_masks(
        frame.size_filtered_mask,
        frame.component_labels,
        component.component_id,
        component.bbox_xywh,
        component.image_origin_uv,
        component.analysis_shape,
    )
    return raw if stage == "raw" else closed


def preprocess_frame(
    *,
    frame_id: int,
    sim_time_ns: int,
    image: np.ndarray,
    lut: np.ndarray,
    config: PreprocessingConfig = DEFAULT_CONFIG,
) -> FrameObservation:
    """Compute shared frame and component evidence exactly once."""
    base = image_to_mask(image, lut)
    input_count, input_labels, input_stats, _ = \
        cv2.connectedComponentsWithStats(base, connectivity=config.connectivity)
    input_components = int(input_count - 1)
    height, width = base.shape

    if (config.maximum_input_components > 0 and
            input_components >= config.maximum_input_components):
        empty_mask = np.zeros(base.shape, np.uint8)
        empty_labels = np.zeros(base.shape, np.int32)
        return FrameObservation(
            frame_id=int(frame_id),
            sim_time_ns=int(sim_time_ns),
            image_shape=(height, width),
            preprocessing_version=config.version,
            base_mask=_readonly(base),
            size_filtered_mask=_readonly(empty_mask.copy()),
            closed_mask=_readonly(empty_mask),
            component_labels=_readonly(empty_labels),
            closed_contour_nodes=(),
            components=(),
            input_component_count=input_components,
            gated=True,
            rejection_reason="frame_component_gate",
        )

    keep = input_stats[:, cv2.CC_STAT_AREA] >= \
        config.minimum_component_area_px
    keep[0] = False
    size_filtered = keep[input_labels].astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (config.close_kernel_px, config.close_kernel_px))
    closed = cv2.morphologyEx(size_filtered, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed, connectivity=config.connectivity)
    closed_nodes = _closed_contour_tree(closed, labels)

    components = []
    for component_id in range(1, count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[component_id])
        side = max(component_width, component_height)
        offset_x = (side - component_width) // 2
        offset_y = (side - component_height) // 2
        bbox = (x, y, component_width, component_height)
        origin = (x - offset_x, y - offset_y)
        analysis_shape = (side, side)
        raw_local, closed_local = _local_masks(
            size_filtered, labels, component_id, bbox, origin, analysis_shape)
        raw_holes, raw_areas, _, raw_hierarchy = _contour_evidence(raw_local)
        closed_holes, closed_areas, outer, closed_hierarchy = \
            _contour_evidence(closed_local)
        parent_nodes = tuple(
            node for node in closed_nodes
            if node.component_id == component_id and not node.is_hole)
        parent_ids = tuple(node.contour_id for node in parent_nodes)
        parent_child_ids = tuple(
            (node.contour_id, child_id)
            for node in parent_nodes
            for child_id in node.child_contour_ids
            if closed_nodes[child_id].is_hole
            and closed_nodes[child_id].component_id == component_id
        )
        outer_contour_id = (None if not parent_nodes else max(
            parent_nodes, key=lambda node: node.area_px2).contour_id)
        simplified = (None if outer is None else cv2.approxPolyDP(
            outer.copy(), config.contour_simplify_epsilon_px, True))
        simplified = _readonly(simplified)
        if outer is None:
            solidity = 0.0
        else:
            hull_area = float(cv2.contourArea(cv2.convexHull(outer.copy())))
            solidity = float(cv2.contourArea(outer.copy())) / max(hull_area, 1e-9)
        distance = cv2.distanceTransform(
            closed_local, cv2.DIST_L2, config.distance_mask_size)
        components.append(ComponentObservation(
            component_id=component_id,
            bbox_xywh=bbox,
            image_origin_uv=origin,
            analysis_shape=analysis_shape,
            touches_frame=(x == 0 or y == 0 or
                           x + component_width >= width or
                           y + component_height >= height),
            area_px=area,
            fill_ratio=float(area) / max(component_width * component_height, 1),
            solidity=solidity,
            topology=TopologyEvidence(
                raw_holes, raw_areas, closed_holes, closed_areas),
            contours=ContourEvidence(
                raw_hierarchy=raw_hierarchy,
                closed_hierarchy=closed_hierarchy,
                outer_raw=outer,
                outer_simplified=simplified,
                outer_contour_id=outer_contour_id,
                parent_contour_ids=parent_ids,
                parent_child_contour_ids=parent_child_ids,
            ),
            distance_transform=_readonly(distance),
        ))

    return FrameObservation(
        frame_id=int(frame_id),
        sim_time_ns=int(sim_time_ns),
        image_shape=(height, width),
        preprocessing_version=config.version,
        base_mask=_readonly(base),
        size_filtered_mask=_readonly(size_filtered),
        closed_mask=_readonly(closed),
        component_labels=_readonly(labels),
        closed_contour_nodes=closed_nodes,
        components=tuple(components),
        input_component_count=input_components,
        gated=False,
        rejection_reason=None,
    )
