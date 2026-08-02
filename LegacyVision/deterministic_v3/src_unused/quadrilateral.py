"""Shared quadrilateral compatibility surface during fitter specialization.

Specialized construction belongs to its owning process folder; this module
retains common ordering and validation. See the directory ``AGENTS.md``.
"""

from __future__ import annotations

import cv2
import numpy as np


def ordered_quadrilateral_corners(points: np.ndarray) -> np.ndarray:
    """Return upper-left-first clockwise image corners."""
    points = np.asarray(points, np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1],
                        points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    ordered = np.roll(ordered, -int(np.argmin(ordered.sum(axis=1))), axis=0)
    contour = ordered.astype(np.float32).reshape(-1, 1, 2)
    if cv2.contourArea(contour, oriented=True) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def valid_quadrilateral_corners(
    corners: np.ndarray | None,
    image_shape: tuple[int, int],
) -> bool:
    """Apply the common convex, area, finite, and in-frame checks."""
    if corners is None or corners.shape != (4, 2) or \
            not np.all(np.isfinite(corners)):
        return False
    contour = corners.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 4.0:
        return False
    height, width = image_shape
    return bool(
        np.all((0 <= corners[:, 0]) & (corners[:, 0] < width)) and
        np.all((0 <= corners[:, 1]) & (corners[:, 1] < height)))


def normalize_quadrilateral(
    points,
    image_shape: tuple[int, int],
) -> tuple[tuple[tuple[float, float], ...] | None, float | None]:
    """Normalize valid points to the schema corner order and return area."""
    if points is None:
        return None, None
    try:
        corners = ordered_quadrilateral_corners(np.asarray(points, np.float64))
    except ValueError:
        return None, None
    if not valid_quadrilateral_corners(corners, image_shape):
        return None, None
    area = float(cv2.contourArea(
        corners.astype(np.float32).reshape(-1, 1, 2)))
    return tuple((float(x), float(y)) for x, y in corners), area
