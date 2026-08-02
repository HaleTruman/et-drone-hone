"""Independent three-visible-line pose experiment for one C-shaped gate."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .gate_centerline_pnp import K, OBJECT_POINTS_M
from .inverse_density_quadrilateral_production import fit_p90_quadrilateral


@dataclass(frozen=True, slots=True)
class CShapeDensityInput:
    instance_id: str
    mask: np.ndarray
    density: np.ndarray
    image_origin_uv: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class VisibleLine:
    side_index: int
    coefficients: tuple[float, float, float]
    support_points: int


@dataclass(frozen=True, slots=True)
class CShapeLineFit:
    instance_id: str
    threshold: float
    scaffold_uv: tuple[tuple[float, float], ...]
    missing_side_index: int
    visible_lines: tuple[VisibleLine, ...]


@dataclass(frozen=True, slots=True)
class CShapePoseCandidate:
    seed_index: int
    rvec_camera: tuple[float, float, float]
    tvec_camera_m: tuple[float, float, float]
    quadrilateral_uv: tuple[tuple[float, float], ...]
    line_rmse_px: float


@dataclass(frozen=True, slots=True)
class CShapePoseResult:
    instance_id: str
    threshold: float
    scaffold_uv: tuple[tuple[float, float], ...]
    missing_side_index: int
    visible_lines: tuple[VisibleLine, ...]
    candidates: tuple[CShapePoseCandidate, ...]


def _segment_distances(points, start, end):
    edge = end - start
    position = np.clip(
        ((points - start) @ edge) / max(float(edge @ edge), 1e-9), 0, 1)
    return np.linalg.norm(points - (start + position[:, None] * edge), axis=1)


def _fit_line(points):
    vx, vy, x, y = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    line = np.array((-vy, vx, vy * x - vx * y), np.float64)
    return line / np.linalg.norm(line[:2])


def fit_c_shape_lines(source: CShapeDensityInput):
    """Return three supported infinite lines without computing a pose."""
    if source.mask.shape != source.density.shape:
        raise ValueError("mask and density shapes must match")
    fit = fit_p90_quadrilateral(source.mask, source.density)
    if fit is None:
        return None
    origin = np.asarray(source.image_origin_uv, np.float64)
    yy, xx = np.nonzero(
        (source.mask != 0) & (source.density >= fit.threshold))
    evidence = np.column_stack((xx, yy)).astype(np.float64) + origin
    scaffold = fit.corners.astype(np.float64) + origin
    distances = np.column_stack([
        _segment_distances(evidence, scaffold[i], scaffold[(i + 1) % 4])
        for i in range(4)])
    assignments = np.argmin(distances, axis=1)
    counts = np.bincount(assignments, minlength=4)
    missing = int(np.argmin(counts))
    if any(counts[i] < 2 for i in range(4) if i != missing):
        return None
    line_arrays = tuple(
        None if i == missing else _fit_line(evidence[assignments == i])
        for i in range(4))
    lines = tuple(VisibleLine(i, tuple(map(float, line_arrays[i])), int(counts[i]))
                  for i in range(4) if i != missing)
    return CShapeLineFit(
        source.instance_id, float(fit.threshold), tuple(map(tuple, scaffold)),
        missing, lines)


def solve_c_shape_pose(source: CShapeDensityInput):
    """Fit three supported sides and return every unranked positive-depth pose."""
    line_fit = fit_c_shape_lines(source)
    if line_fit is None:
        return None
    scaffold = np.asarray(line_fit.scaffold_uv, np.float64)
    lines = line_fit.visible_lines

    def residual(parameters):
        values = []
        for line in lines:
            endpoints = OBJECT_POINTS_M[
                [line.side_index, (line.side_index + 1) % 4]]
            projected, _ = cv2.projectPoints(
                endpoints, parameters[:3], parameters[3:], K, None)
            homogeneous = np.column_stack(
                (projected.reshape(2, 2), np.ones(2)))
            values.extend(homogeneous @ np.asarray(line.coefficients))
        return np.asarray(values)

    solved = cv2.solvePnPGeneric(
        OBJECT_POINTS_M, scaffold, K, None, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for seed_index, (rvec, tvec) in enumerate(
            zip(solved[1], solved[2]) if solved[0] else ()):
        optimized = least_squares(
            residual, np.concatenate((rvec.ravel(), tvec.ravel())),
            method="lm", max_nfev=200)
        if not optimized.success or optimized.x[5] <= 0 or \
                not np.all(np.isfinite(optimized.x)):
            continue
        projected, _ = cv2.projectPoints(
            OBJECT_POINTS_M, optimized.x[:3], optimized.x[3:], K, None)
        error = residual(optimized.x)
        candidates.append(CShapePoseCandidate(
            seed_index, tuple(map(float, optimized.x[:3])),
            tuple(map(float, optimized.x[3:])),
            tuple(map(tuple, projected.reshape(4, 2).astype(float))),
            float(np.sqrt(np.mean(error ** 2)))))
    return CShapePoseResult(
        source.instance_id, line_fit.threshold, line_fit.scaffold_uv,
        line_fit.missing_side_index, lines, tuple(candidates))
