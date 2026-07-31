"""Metric camera-frame position of a void from its published geometry.

Bearing comes from the ellipse centre through the intrinsics; range from the
known gate dimensions under a pinhole model. The outer and inner corners sit on
different contours, so they give two independent range estimates whose
disagreement is a free per-detection quality signal needing no ground truth.

Camera frame only. Mapping into vehicle or world coordinates needs the camera
tilt and a vehicle pose, which are a separate concern with a separate dependency.
"""
import math
from .schema import CORNER_ROLES, VoidPositionEstimate

FOCAL_PX = 320.0
PRINCIPAL_PX = (320.0, 180.0)
OUTER_WIDTH_M = 2.7
INNER_WIDTH_M = 1.5
CENTROID_JITTER_PX = 0.41
MIN_SPAN_PX = 4.0
MIN_RANGE_ERROR = 0.05

def corner_quad(corners, prefix):
    points = {corner.role: corner.point_px for corner in corners}
    roles = [f"{prefix}_{role}" for role in CORNER_ROLES]
    return [points[role] for role in roles] \
        if all(role in points for role in roles) else None

def quad_span(quad):
    """Longer of the mean horizontal and mean vertical side, in pixels."""
    upper_left, upper_right, lower_left, lower_right = quad
    width = math.dist(upper_left, upper_right) + math.dist(lower_left, lower_right)
    height = math.dist(upper_left, lower_left) + math.dist(upper_right, lower_right)
    return max(width, height) / 2

def channel_range(corners, prefix, width_m):
    quad = corner_quad(corners, prefix)
    span = quad_span(quad) if quad else 0.0
    return FOCAL_PX * width_m / span if span >= MIN_SPAN_PX else None

def bearing_unit(center_px):
    x = (center_px[0] - PRINCIPAL_PX[0]) / FOCAL_PX
    y = (center_px[1] - PRINCIPAL_PX[1]) / FOCAL_PX
    norm = math.sqrt(x * x + y * y + 1.0)
    return (x / norm, y / norm, 1.0 / norm)

def estimate_position(record):
    """One VoidDetectionGeometry to a VoidPositionEstimate, or None if too small."""
    outer = channel_range(record.outer_corners, "outer", OUTER_WIDTH_M)
    if outer is None:
        return None
    inner = channel_range(record.inner_corners, "inner", INNER_WIDTH_M)
    channels = (outer, inner) if inner else (outer,)
    range_m = sum(channels) / len(channels)
    residual = (max(channels) - min(channels)) / range_m
    bearing = bearing_unit(record.ellipse.center_px)
    return VoidPositionEstimate(
        record.source, bearing, range_m,
        tuple(range_m * axis for axis in bearing),
        range_m * CENTROID_JITTER_PX / FOCAL_PX,
        range_m * max(MIN_RANGE_ERROR, residual), residual, channels)

def estimate_positions(records):
    estimates = (estimate_position(record) for record in records)
    return [estimate for estimate in estimates if estimate is not None]
