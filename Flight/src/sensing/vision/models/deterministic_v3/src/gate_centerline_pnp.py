"""Minimal camera-frame PnP for a gate-border centerline quadrilateral."""
from dataclasses import dataclass
import cv2
import numpy as np

K = np.array(((320.0, 0.0, 320.0), (0.0, 320.0, 180.0),
              (0.0, 0.0, 1.0)), np.float64)
HALF_SIDE_M = 0.5 * ((2.70 + 1.50) / 2.0)
OBJECT_POINTS_M = np.array(((-HALF_SIDE_M, HALF_SIDE_M, 0.0),
                            (HALF_SIDE_M, HALF_SIDE_M, 0.0),
                            (HALF_SIDE_M, -HALF_SIDE_M, 0.0),
                            (-HALF_SIDE_M, -HALF_SIDE_M, 0.0)), np.float64)


@dataclass(frozen=True, slots=True)
class GateCenterlineQuad:
    """Corners are TL, TR, BR, BL pixel coordinates."""
    instance_id: str
    corners_uv: tuple[tuple[float, float], ...]

@dataclass(frozen=True, slots=True)
class GatePose:
    instance_id: str
    rvec_camera: tuple[float, float, float] | None
    tvec_camera_m: tuple[float, float, float] | None
    confidence: float
    reprojection_rmse_px: float | None


def solve_gate_pose(quad: GateCenterlineQuad) -> GatePose:
    """Solve the 2.10 m centerline square; confidence measures pixel fit only."""
    image_points = np.asarray(quad.corners_uv, np.float64)
    if image_points.shape != (4, 2) or not np.all(np.isfinite(image_points)):
        return GatePose(quad.instance_id, None, None, 0.0, None)
    solved = cv2.solvePnPGeneric(
        OBJECT_POINTS_M, image_points, K, None, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for rvec, tvec in zip(solved[1], solved[2]) if solved[0] else ():
        if float(tvec[2, 0]) <= 0.0:
            continue
        projected, _ = cv2.projectPoints(OBJECT_POINTS_M, rvec, tvec, K, None)
        rmse = float(np.sqrt(np.mean(np.sum(
            (projected.reshape(4, 2) - image_points) ** 2, axis=1))))
        candidates.append((rmse, rvec.reshape(3), tvec.reshape(3)))
    if not candidates:
        return GatePose(quad.instance_id, None, None, 0.0, None)
    rmse, rvec, tvec = min(candidates, key=lambda item: item[0])
    confidence = 1.0 / (1.0 + (rmse / 3.0) ** 2)
    return GatePose(quad.instance_id, tuple(map(float, rvec)),
                    tuple(map(float, tvec)), confidence, rmse)
