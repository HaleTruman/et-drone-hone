"""Review-only rendering for one spectrum-calibration component.

The seven panels deliberately contain no density math.  Their arrays come from
the production ``DensityBank`` adapter owned by the parent calibration package.
"""

from __future__ import annotations

import cv2
import numpy as np

from ...schema import DensityEvidence


PANEL_SIDE_PX = 150
PANEL_HEADER_PX = 22
PANEL_LABELS = (
    "source analysis crop",
    "persisted closed component mask",
    "candidate DensityBank input mask",
    "DensityEvidence.final_field",
    "DensityEvidence.p70_mask",
    "DensityEvidence.p80_mask",
    "DensityEvidence.p90_mask",
)


def _bgr(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.astype(np.uint8, copy=False)
    raise ValueError("render panels require a two-dimensional or BGR image")


def _fit(image: np.ndarray, *, nearest: bool) -> np.ndarray:
    image = _bgr(image)
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("render panels cannot contain an empty image")
    scale = min(PANEL_SIDE_PX / width, PANEL_SIDE_PX / height)
    size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    interpolation = cv2.INTER_NEAREST if nearest else (
        cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    resized = cv2.resize(image, size, interpolation=interpolation)
    result = np.zeros((PANEL_SIDE_PX, PANEL_SIDE_PX, 3), np.uint8)
    top = (PANEL_SIDE_PX - resized.shape[0]) // 2
    left = (PANEL_SIDE_PX - resized.shape[1]) // 2
    result[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return result


def _binary(mask: np.ndarray) -> np.ndarray:
    return ((np.asarray(mask) != 0).astype(np.uint8) * 255)


def _field(field: np.ndarray, component_mask: np.ndarray) -> np.ndarray:
    values = np.rint(np.clip(np.asarray(field), 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(values, cv2.COLORMAP_TURBO)
    heat[np.asarray(component_mask) == 0] = 0
    return heat


def _panel(image: np.ndarray, label: str, *, nearest: bool) -> np.ndarray:
    panel = np.zeros(
        (PANEL_HEADER_PX + PANEL_SIDE_PX, PANEL_SIDE_PX, 3), np.uint8)
    panel[PANEL_HEADER_PX:] = _fit(image, nearest=nearest)
    cv2.putText(
        panel,
        label,
        (4, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.31,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return panel


def vertical_evidence_sheet(
    source_analysis_crop: np.ndarray,
    persisted_closed_component_mask: np.ndarray,
    candidate_input_mask: np.ndarray,
    evidence: DensityEvidence,
) -> np.ndarray:
    """Return the two masks and refreshed density evidence as seven panels."""
    persisted_mask = (
        np.asarray(persisted_closed_component_mask) != 0).astype(np.uint8)
    candidate_mask = (np.asarray(candidate_input_mask) != 0).astype(np.uint8)
    if persisted_mask.ndim != 2 or not np.any(persisted_mask):
        raise ValueError("persisted closed component mask must be non-empty and 2D")
    if candidate_mask.ndim != 2 or not np.any(candidate_mask):
        raise ValueError("candidate DensityBank input mask must be non-empty and 2D")
    if candidate_mask.shape != persisted_mask.shape:
        raise ValueError("persisted and candidate masks must share an analysis shape")
    expected_shape = candidate_mask.shape
    for name in ("final_field", "p70_mask", "p80_mask", "p90_mask"):
        if np.asarray(getattr(evidence, name)).shape != expected_shape:
            raise ValueError(f"{name} does not match the component analysis shape")

    images = (
        np.asarray(source_analysis_crop),
        _binary(persisted_mask),
        _binary(candidate_mask),
        _field(evidence.final_field, candidate_mask),
        _binary(evidence.p70_mask),
        _binary(evidence.p80_mask),
        _binary(evidence.p90_mask),
    )
    nearest = (False, True, True, True, True, True, True)
    return np.vstack(tuple(
        _panel(image, label, nearest=use_nearest)
        for image, label, use_nearest in zip(images, PANEL_LABELS, nearest)
    ))


def encode_png(image: np.ndarray) -> bytes:
    """Encode one completed evidence sheet without writing a review artifact."""
    ok, encoded = cv2.imencode(
        ".png", np.asarray(image), (cv2.IMWRITE_PNG_COMPRESSION, 2))
    if not ok:
        raise OSError("unable to encode spectrum-calibration evidence sheet")
    return encoded.tobytes()
