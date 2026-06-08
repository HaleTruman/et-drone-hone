"""Contour quadrilateral fitting for predicted gate mask components."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


MODEL_WIDTH = 160
MODEL_HEIGHT = 90


def model_to_source_xy(x_px: float, y_px: float, source_width: float, source_height: float) -> tuple[float, float]:
    return (
        float(x_px) * (max(float(source_width), 1.0) / float(MODEL_WIDTH)),
        float(y_px) * (max(float(source_height), 1.0) / float(MODEL_HEIGHT)),
    )


def _ordered_clockwise(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))))


def _line_intersection(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> tuple[float, float] | None:
    x1, y1 = float(a[0]), float(a[1])
    x2, y2 = float(b[0]), float(b[1])
    x3, y3 = float(c[0]), float(c[1])
    x4, y4 = float(d[0]), float(d[1])
    denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if abs(denominator) <= 1.0e-6:
        return None
    px = (((x1 * y2) - (y1 * x2)) * (x3 - x4) - (x1 - x2) * ((x3 * y4) - (y3 * x4))) / denominator
    py = (((x1 * y2) - (y1 * x2)) * (y3 - y4) - (y1 - y2) * ((x3 * y4) - (y3 * x4))) / denominator
    if not math.isfinite(px) or not math.isfinite(py):
        return None
    return float(px), float(py)


def _quad_center(points: np.ndarray) -> tuple[tuple[float, float], str]:
    ordered = _ordered_clockwise(points)
    intersection = _line_intersection(ordered[0], ordered[2], ordered[1], ordered[3])
    if intersection is not None:
        return intersection, "diagonal_intersection"
    moments = cv2.moments(ordered.reshape(-1, 1, 2))
    if abs(float(moments["m00"])) > 1.0e-6:
        return float(moments["m10"] / moments["m00"]), float(moments["m01"] / moments["m00"]), "polygon_centroid_fallback"
    center = ordered.mean(axis=0)
    return (float(center[0]), float(center[1])), "corner_average_fallback"


def _is_valid_quad(points: np.ndarray) -> bool:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape != (4, 2):
        return False
    if _polygon_area(pts) <= 1.0:
        return False
    rounded = {(round(float(x), 3), round(float(y), 3)) for x, y in pts}
    if len(rounded) != 4:
        return False
    return bool(cv2.isContourConvex(_ordered_clockwise(pts).reshape(-1, 1, 2)))


def _fallback_min_area_rect(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if xs.size >= 3:
        points = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
        rect = cv2.minAreaRect(points)
        corners = cv2.boxPoints(rect).astype(np.float32)
        (_cx, _cy), (box_w, box_h), angle_deg = rect
    else:
        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = float(ys.min())
        y_max = float(ys.max())
        corners = np.asarray([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
        box_w = max(1.0, x_max - x_min + 1.0)
        box_h = max(1.0, y_max - y_min + 1.0)
        angle_deg = 0.0
    return _ordered_clockwise(corners), {
        "quad_fit_source": "min_area_rect_fallback",
        "quad_is_fallback": True,
        "quad_fit_epsilon_frac": None,
        "quad_size_model_px": [float(box_w), float(box_h)],
        "quad_angle_deg": float(angle_deg),
    }


def fit_contour_quad_from_pixels(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    x_values = np.asarray(xs, dtype=np.float32).reshape(-1)
    y_values = np.asarray(ys, dtype=np.float32).reshape(-1)
    if x_values.size <= 0 or y_values.size <= 0:
        raise ValueError("Cannot fit a quadrilateral without component pixels.")

    quad: np.ndarray | None = None
    metadata: dict[str, Any] | None = None
    if x_values.size >= 4:
        x_min = int(math.floor(float(x_values.min())))
        y_min = int(math.floor(float(y_values.min())))
        x_max = int(math.ceil(float(x_values.max())))
        y_max = int(math.ceil(float(y_values.max())))
        roi = np.zeros((max(1, y_max - y_min + 1), max(1, x_max - x_min + 1)), dtype=np.uint8)
        local_x = np.clip(np.rint(x_values).astype(np.int32) - x_min, 0, roi.shape[1] - 1)
        local_y = np.clip(np.rint(y_values).astype(np.int32) - y_min, 0, roi.shape[0] - 1)
        roi[local_y, local_x] = 255
        contours, _hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = max(contours, key=cv2.contourArea).astype(np.float32)
            contour[:, 0, 0] += float(x_min)
            contour[:, 0, 1] += float(y_min)
            hull = cv2.convexHull(contour)
            perimeter = float(cv2.arcLength(hull, True))
            if perimeter > 1.0e-6:
                for epsilon_frac in np.linspace(0.01, 0.12, 23):
                    approx = cv2.approxPolyDP(hull, float(epsilon_frac) * perimeter, True).reshape(-1, 2)
                    if _is_valid_quad(approx):
                        quad = _ordered_clockwise(approx)
                        metadata = {
                            "quad_fit_source": "contour_hull_approx_poly_dp",
                            "quad_is_fallback": False,
                            "quad_fit_epsilon_frac": float(epsilon_frac),
                            "quad_size_model_px": None,
                            "quad_angle_deg": None,
                        }
                        break

    if quad is None or metadata is None:
        quad, metadata = _fallback_min_area_rect(x_values, y_values)

    (center_x, center_y), center_source = _quad_center(quad)
    source_center = model_to_source_xy(center_x, center_y, source_width, source_height)
    quad_px = [
        [float(value) for value in model_to_source_xy(float(point[0]), float(point[1]), source_width, source_height)]
        for point in quad
    ]
    return {
        "quad_model_px": [[float(point[0]), float(point[1])] for point in quad],
        "quad_px": quad_px,
        "quad_center_model_px": [float(center_x), float(center_y)],
        "quad_center_px": [float(source_center[0]), float(source_center[1])],
        "quad_center_source": center_source,
        "quad_area_model_px": _polygon_area(quad),
        **metadata,
    }


__all__ = ["MODEL_HEIGHT", "MODEL_WIDTH", "fit_contour_quad_from_pixels", "model_to_source_xy"]
