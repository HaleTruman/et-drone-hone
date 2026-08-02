"""Evidence-only multi-view gate localization, published in run-local NED.

All depth is solved in the current camera frame from cached 2-D projections and
relative camera motion.  NED is used once, after inference, to publish a point.
"""
from dataclasses import dataclass, field, replace
from itertools import combinations
import math
import cv2
import numpy as np

from core.schema import VisionGateObservation, VisionObservation
from .void_ned import camera_position_ned, rotation_world_from_camera
from .void_position import FOCAL_PX, INNER_WIDTH_M, OUTER_WIDTH_M, PRINCIPAL_PX
SOURCE, MAX_VIEWS = "detection_v3_projective_bundle", 30
MAX_MISSED_FRAMES = 15         # short occlusion bridge, not a global map
KEYFRAME_BASELINE_M, STATIONARY_BASELINE_M = 0.10, 0.01
MIN_BASELINE_M, MIN_TRANSVERSE_BASELINE_M, MIN_PARALLAX_DEG = 0.50, 0.15, 0.50
MIN_PLANE_BASELINE_M, PLANE_CUTOFF_PX = 0.60, 12.0
MIN_CAMERA_DEPTH_M, MAX_RANGE_M = 0.20, 200.0
PIXEL_NOISE_PX, TUKEY_CUTOFF_PX = 1.5, 10.0
ASSOCIATION_FLOOR_PX, MIN_PUBLISH_CONFIDENCE, MIN_PUBLISH_VIEWS = 18.0, 0.05, 3
_ROLES = ("upper_left", "upper_right", "lower_left", "lower_right")
@dataclass(slots=True)
class _View:
    key: tuple[int, int]; origin: np.ndarray
    world_from_camera: np.ndarray; pixel: np.ndarray
    features: dict[str, np.ndarray] | None = None
@dataclass(slots=True)
class _FeatureSolution:
    position: np.ndarray; covariance: np.ndarray
    rmse_px: float; inliers: int; views: int
    parallax_deg: float; quality: float
@dataclass(slots=True)
class _GateSolution:
    position: np.ndarray; covariance: np.ndarray
    rmse_px: float; inliers: int; views: int
    parallax_deg: float; quality: float
    groups: tuple[str, ...]
@dataclass(slots=True)
class _Landmark:
    gate_id: str; last_seen: int = 0; observed_frames: int = 0
    histories: dict[str, list[_View]] = field(default_factory=dict)
    plane_history: list[_View] = field(default_factory=list)
    feature_solutions: dict[str, _FeatureSolution] = field(default_factory=dict)
    solution: _GateSolution | None = None
    anchor_origin: np.ndarray | None = None
    anchor_rotation: np.ndarray | None = None

@dataclass(slots=True)
class _PlaneMatch:
    system: np.ndarray; target: np.ndarray
    rotation: np.ndarray; translation: np.ndarray
    current_ray: np.ndarray; prior_ray: np.ndarray
    name: str; view_index: int; baseline: float; reliability: float

def _unit(vector):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector
def _ray(pixel, world_from_camera):
    u, v = pixel
    camera = np.array(((u - PRINCIPAL_PX[0]) / FOCAL_PX,
                       (v - PRINCIPAL_PX[1]) / FOCAL_PX, 1.0))
    return _unit(world_from_camera @ _unit(camera))
def _camera_ray(pixel):
    return np.array(((pixel[0] - PRINCIPAL_PX[0]) / FOCAL_PX,
                     (pixel[1] - PRINCIPAL_PX[1]) / FOCAL_PX, 1.0))
def _project_camera(camera):
    if camera[2] <= MIN_CAMERA_DEPTH_M:
        return None
    return np.array((PRINCIPAL_PX[0] + FOCAL_PX * camera[0] / camera[2],
                     PRINCIPAL_PX[1] + FOCAL_PX * camera[1] / camera[2]))
def _parallax(directions):
    if len(directions) < 2:
        return 0.0
    dots = np.clip(np.asarray(directions) @ np.asarray(directions).T, -1.0, 1.0)
    return math.degrees(math.acos(float(np.min(dots))))
def _bundle_terms(position, origins, rotations, pixels):
    camera = np.einsum("nij,nj->ni", rotations, position - origins)
    x, y, z = camera.T
    safe_z = np.maximum(z, 1e-9)
    predicted = np.column_stack((PRINCIPAL_PX[0] + FOCAL_PX * x / safe_z,
                                 PRINCIPAL_PX[1] + FOCAL_PX * y / safe_z))
    projection = np.zeros((len(origins), 2, 3))
    projection[:, 0, 0], projection[:, 1, 1] = FOCAL_PX / safe_z, FOCAL_PX / safe_z
    projection[:, 0, 2] = -FOCAL_PX * x / safe_z ** 2
    projection[:, 1, 2] = -FOCAL_PX * y / safe_z ** 2
    residuals = predicted - pixels
    jacobians = np.einsum("nij,njk->nik", projection, rotations)
    invalid = z <= MIN_CAMERA_DEPTH_M
    residuals[invalid], jacobians[invalid] = 1e6, 0.0
    return residuals, jacobians, np.linalg.norm(residuals, axis=1)

def _linear_intersection(origins, directions, indices):
    selected_directions = directions[indices]
    projectors = np.eye(3) - np.einsum("ni,nj->nij", selected_directions,
                                       selected_directions)
    system = np.sum(projectors, axis=0)
    if np.linalg.eigvalsh(system)[0] <= 1e-9:
        return None
    target = np.einsum("nij,nj->i", projectors, origins[indices])
    return np.linalg.pinv(system, rcond=1e-10) @ target

def _robust_initial_position(origins, directions, rotations, pixels):
    latest, candidates = len(origins) - 1, []
    for index in range(latest):
        position = _linear_intersection(origins, directions, np.array((index, latest)))
        if position is None:
            continue
        _, _, errors = _bundle_terms(position, origins, rotations, pixels)
        inliers = errors < TUKEY_CUTOFF_PX
        if np.count_nonzero(inliers) >= 2:
            candidates.append((int(np.count_nonzero(inliers)),
                               -float(np.median(errors[inliers])), position))
    if candidates:
        return max(candidates, key=lambda item: item[:2])[2]
    return _linear_intersection(origins, directions, np.arange(len(origins)))

def _current_views(history, view):
    """Cache spaced keyframes but always solve in the supplied current frame."""
    if not history:
        history.append(view)
        return list(history)
    if history[-1].key == view.key:
        return list(history)
    baseline = float(np.linalg.norm(view.origin - history[-1].origin))
    if baseline >= KEYFRAME_BASELINE_M:
        history.append(view)
        del history[:-MAX_VIEWS]
        return list(history)
    effective = view
    if baseline < STATIONARY_BASELINE_M:
        history[-1].pixel = 0.9 * history[-1].pixel + 0.1 * view.pixel
        features = view.features
        if history[-1].features is not None and features is not None:
            averaged = {}
            for name, value in features.items():
                old = history[-1].features.get(name)
                averaged[name] = value if old is None else 0.9 * old + 0.1 * value
                history[-1].features[name] = averaged[name]
            features = averaged
        effective = _View(view.key, view.origin, view.world_from_camera,
                          history[-1].pixel.copy(), features)
    return [*history[-(MAX_VIEWS - 1):], effective]

def _solve_feature(views):
    """Solve in the latest camera frame, never in absolute NED."""
    if len(views) < 2:
        return None
    reference = views[-1]
    current_from_ned = reference.world_from_camera.T
    views = [_View(view.key, current_from_ned @ (view.origin - reference.origin),
                   current_from_ned @ view.world_from_camera, view.pixel)
             for view in views]
    origins = np.asarray([view.origin for view in views])
    rotations = np.asarray([view.world_from_camera.T for view in views])
    pixels = np.asarray([view.pixel for view in views])
    baseline = float(np.max(np.linalg.norm(origins - origins[-1], axis=1)))
    directions = np.asarray([_ray(view.pixel, view.world_from_camera) for view in views])
    parallax = _parallax(directions)
    transverse = float(np.max(np.linalg.norm(
        np.cross(origins - origins[-1], directions[-1]), axis=1)))
    if (baseline < MIN_BASELINE_M or transverse < MIN_TRANSVERSE_BASELINE_M
            or parallax < MIN_PARALLAX_DEG):
        return None
    position = _robust_initial_position(origins, directions, rotations, pixels)
    if position is None:
        return None

    weights = np.ones(len(views))
    for _ in range(8):
        residuals, jacobians, errors = _bundle_terms(position, origins, rotations, pixels)
        ratio = errors / TUKEY_CUTOFF_PX
        weights = np.where(ratio < 1.0, (1.0 - ratio ** 2) ** 2, 0.0)
        if np.count_nonzero(weights > 0.05) < 2:
            return None
        hessian = np.einsum("n,nai,naj->ij", weights, jacobians, jacobians)
        gradient = np.einsum("n,nai,na->i", weights, jacobians, residuals)
        step = -np.linalg.pinv(hessian, rcond=1e-10) @ gradient
        position += step
        if float(np.linalg.norm(step)) < 1e-5:
            break

    _, jacobians, errors = _bundle_terms(position, origins, rotations, pixels)
    inlier_mask = (errors < TUKEY_CUTOFF_PX) & (weights > 0.05)
    if np.count_nonzero(inlier_mask) < 2:
        return None
    current_camera = views[-1].world_from_camera.T @ (position - views[-1].origin)
    distance = float(np.linalg.norm(position - views[-1].origin))
    if current_camera[2] <= MIN_CAMERA_DEPTH_M or not MIN_CAMERA_DEPTH_M < distance < MAX_RANGE_M:
        return None

    inlier_weights = weights[inlier_mask]
    hessian = np.einsum("n,nai,naj->ij", inlier_weights, jacobians[inlier_mask],
                        jacobians[inlier_mask])
    dof = max(1, 2 * int(np.count_nonzero(inlier_mask)) - 3)
    variance = max(PIXEL_NOISE_PX ** 2,
                   float(np.sum(inlier_weights * errors[inlier_mask] ** 2)) / dof)
    if np.linalg.eigvalsh(hessian)[0] <= 1e-12:
        return None
    covariance = variance * np.linalg.pinv(hessian, rcond=1e-10)
    if not np.all(np.isfinite(covariance)):
        return None
    sigma_major = math.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(covariance)))))
    rmse = math.sqrt(float(np.mean(errors[inlier_mask] ** 2)))
    inliers = int(np.count_nonzero(inlier_mask))
    support = min(1.0, (inliers - 1) / 5.0)
    geometry = min(1.0, parallax / 3.0)
    fit = 1.0 / (1.0 + (rmse / 3.0) ** 2)
    precision = 1.0 / (1.0 + sigma_major / max(0.25, 0.10 * distance))
    quality = float(np.clip(support * geometry * fit * precision, 0.0, 1.0))
    return _FeatureSolution(position, covariance, rmse, inliers, len(views),
                            parallax, quality)

def _plane_matches(views):
    """Linear constraints for H = R + t v^T, with plane v^T X = 1."""
    reference = views[-1]
    if reference.features is None:
        return []
    matches = []
    reliability = {"center": 0.75}
    for view_index, prior in enumerate(views[:-1]):
        if prior.features is None:
            continue
        baseline = float(np.linalg.norm(reference.origin - prior.origin))
        if baseline < MIN_PLANE_BASELINE_M:
            continue
        rotation = prior.world_from_camera.T @ reference.world_from_camera
        translation = prior.world_from_camera.T @ (reference.origin - prior.origin)
        shared = reference.features.keys() & prior.features.keys()
        if len(shared) < 4:
            continue
        for name in shared:
            current_ray = _camera_ray(reference.features[name])
            prior_ray = _camera_ray(prior.features[name])
            rotated = rotation @ current_ray
            u, v = prior_ray[:2]
            system = np.vstack(((translation[0] - u * translation[2]) * current_ray,
                                (translation[1] - v * translation[2]) * current_ray))
            target = np.array((u * rotated[2] - rotated[0],
                               v * rotated[2] - rotated[1]))
            weight = reliability.get(name, 0.70 if name.startswith("inner_") else 1.0)
            matches.append(_PlaneMatch(system, target, rotation, translation,
                                       current_ray, prior_ray, name, view_index,
                                       baseline, weight))
    return matches

def _fit_plane(matches, indices, weights=None):
    if len(indices) < 2:
        return None
    if weights is None:
        weights = np.asarray([matches[index].reliability for index in indices])
    matrix = np.vstack([math.sqrt(max(0.0, weight)) * matches[index].system
                        for index, weight in zip(indices, weights)])
    target = np.hstack([math.sqrt(max(0.0, weight)) * matches[index].target
                        for index, weight in zip(indices, weights)])
    if np.linalg.matrix_rank(matrix, tol=1e-9) < 3:
        return None
    return np.linalg.lstsq(matrix, target, rcond=1e-10)[0]

def _plane_errors(vector, matches):
    errors = []
    for match in matches:
        projected = ((match.rotation + np.outer(match.translation, vector))
                     @ match.current_ray)
        if projected[2] <= 1e-9:
            errors.append(1e6)
        else:
            errors.append(FOCAL_PX * float(np.linalg.norm(
                projected[:2] / projected[2] - match.prior_ray[:2])))
    return np.asarray(errors)

def _valid_plane(vector, center_ray, current_features):
    denominator = float(vector @ center_ray)
    if denominator <= 1.0 / MAX_RANGE_M:
        return False
    position = center_ray / denominator
    if not MIN_CAMERA_DEPTH_M < float(np.linalg.norm(position)) < MAX_RANGE_M:
        return False
    normal_cosine = abs(denominator) / max(1e-12, np.linalg.norm(vector) *
                                          np.linalg.norm(center_ray))
    if normal_cosine < 0.08:
        return False
    depths = [1.0 / float(vector @ _camera_ray(pixel))
              for pixel in current_features.values()
              if float(vector @ _camera_ray(pixel)) > 1.0 / MAX_RANGE_M]
    return len(depths) >= 4 and max(depths) < MAX_RANGE_M

def _solve_plane(views):
    """Estimate metric gate-plane depth from correspondences and relative pose."""
    if len(views) < 2 or views[-1].features is None:
        return None
    matches = _plane_matches(views)
    if len(matches) < 4:
        return None
    center_ray = _camera_ray(views[-1].pixel)
    candidates = []
    all_indices = list(range(len(matches)))
    candidate = _fit_plane(matches, all_indices)
    if candidate is not None:
        candidates.append(candidate)
    by_view = {}
    for index, match in enumerate(matches):
        by_view.setdefault(match.view_index, []).append(index)
    for indices in by_view.values():
        candidate = _fit_plane(matches, indices)
        if candidate is not None:
            candidates.append(candidate)
        for prefix in ("outer_", "inner_"):
            family = [index for index in indices if matches[index].name.startswith(prefix)]
            for subset in combinations(family, 3):
                candidate = _fit_plane(matches, list(subset))
                if candidate is not None:
                    candidates.append(candidate)
    candidates = [item for item in candidates
                  if _valid_plane(item, center_ray, views[-1].features)]
    if not candidates:
        return None
    reliabilities = np.asarray([match.reliability for match in matches])
    def score(vector):
        errors = _plane_errors(vector, matches)
        inliers = errors < PLANE_CUTOFF_PX
        return (float(np.sum(reliabilities[inliers])),
                -float(np.median(errors[inliers])) if np.any(inliers) else -1e6)
    vector = max(candidates, key=score)
    weights = reliabilities.copy()
    for _ in range(8):
        errors = _plane_errors(vector, matches)
        ratio = errors / PLANE_CUTOFF_PX
        robust = np.where(ratio < 1.0, (1.0 - ratio ** 2) ** 2, 0.0)
        weights = reliabilities * robust
        active = np.flatnonzero(weights > 0.03)
        if len(active) < 4:
            return None
        updated = _fit_plane(matches, active.tolist(), weights[active])
        if updated is None or not _valid_plane(updated, center_ray, views[-1].features):
            return None
        if float(np.linalg.norm(updated - vector)) < 1e-7:
            vector = updated
            break
        vector = updated
    errors = _plane_errors(vector, matches)
    inliers = (errors < PLANE_CUTOFF_PX) & (weights > 0.03)
    active_views = {match.view_index for match, active in zip(matches, inliers) if active}
    active_names = {match.name for match, active in zip(matches, inliers) if active}
    if np.count_nonzero(inliers) < 4 or not active_views:
        return None
    hessian = np.zeros((3, 3))
    for match, weight, active in zip(matches, weights, inliers):
        if not active:
            continue
        projected = ((match.rotation + np.outer(match.translation, vector))
                     @ match.current_ray)
        z = projected[2]
        jacobian = FOCAL_PX * np.vstack((
            (match.translation[0] * z - projected[0] * match.translation[2])
            * match.current_ray / z ** 2,
            (match.translation[1] * z - projected[1] * match.translation[2])
            * match.current_ray / z ** 2))
        hessian += weight * jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= max(1e-10, eigenvalues[-1] * 1e-10):
        return None
    dof = max(1, 2 * int(np.count_nonzero(inliers)) - 3)
    variance = max(PIXEL_NOISE_PX ** 2,
                   float(np.sum(weights[inliers] * errors[inliers] ** 2)) / dof)
    covariance_v = variance * np.linalg.pinv(hessian, rcond=1e-10)
    inverse_depth = float(vector @ center_ray)
    depth = 1.0 / inverse_depth
    position = center_ray * depth
    position_jacobian = -np.outer(center_ray, center_ray) / inverse_depth ** 2
    direction = _unit(center_ray)
    bearing_sigma = depth * PIXEL_NOISE_PX / FOCAL_PX
    covariance = position_jacobian @ covariance_v @ position_jacobian.T
    covariance += bearing_sigma ** 2 * (np.eye(3) - np.outer(direction, direction))
    if not np.all(np.isfinite(covariance)):
        return None
    sigma = math.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(covariance)))))
    rmse = math.sqrt(float(np.average(errors[inliers] ** 2, weights=weights[inliers])))
    baseline = max(match.baseline for match, active in zip(matches, inliers) if active)
    support = min(1.0, np.count_nonzero(inliers) / 12.0)
    geometry = min(1.0, baseline / max(MIN_PLANE_BASELINE_M, 0.10 * depth))
    fit = 1.0 / (1.0 + (rmse / 4.0) ** 2)
    precision = 1.0 / (1.0 + sigma / max(0.35, 0.12 * np.linalg.norm(position)))
    quality = float(np.clip(support * geometry * fit * precision, 0.0, 1.0))
    families = tuple(name for name, present in (
        ("outer_corners", any(name.startswith("outer_") for name in active_names)),
        ("inner_corners", any(name.startswith("inner_") for name in active_names)),
        ("ellipse_center", "center" in active_names)) if present)
    return _GateSolution(position, covariance, rmse, int(np.count_nonzero(inliers)),
                         len(active_views) + 1,
                         math.degrees(math.atan2(baseline, depth)), quality,
                         ("projective_plane", *families))

def _pnp_family(geometry, prefix, width_m):
    corners = geometry.outer_corners if prefix == "outer" else geometry.inner_corners
    pixels = {corner.role: corner.point_px for corner in corners}
    roles = tuple(f"{prefix}_{role}" for role in _ROLES)
    if not all(role in pixels for role in roles):
        return None
    half = 0.5 * width_m
    coordinates = ((-half, -half, 0.0), (half, -half, 0.0),
                   (-half, half, 0.0), (half, half, 0.0))
    object_points = np.asarray(coordinates, dtype=float)
    image_points = np.asarray([pixels[role] for role in roles], dtype=float)
    camera_matrix = np.array(((FOCAL_PX, 0.0, PRINCIPAL_PX[0]),
                              (0.0, FOCAL_PX, PRINCIPAL_PX[1]),
                              (0.0, 0.0, 1.0)))
    result = cv2.solvePnPGeneric(object_points, image_points, camera_matrix, None,
                                 flags=cv2.SOLVEPNP_IPPE)
    if not result[0]:
        return None
    choices = []
    for rotation, translation in zip(result[1], result[2]):
        translation = translation.reshape(3)
        if translation[2] <= MIN_CAMERA_DEPTH_M:
            continue
        projected, _ = cv2.projectPoints(object_points, rotation, translation,
                                         camera_matrix, None)
        rmse = math.sqrt(float(np.mean(np.sum(
            (projected.reshape(-1, 2) - image_points) ** 2, axis=1))))
        projected_center = _project_camera(translation)
        center_error = (1e6 if projected_center is None else float(np.linalg.norm(
            projected_center - np.asarray(geometry.ellipse.center_px))))
        choices.append((rmse + 0.25 * center_error, translation[2], rmse, center_error))
    if not choices:
        return None
    _, depth, rmse, center_error = min(choices, key=lambda item: item[0])
    ellipse_scale = max(geometry.ellipse.semi_axes_px)
    if rmse > 8.0 or center_error > max(12.0, 0.20 * ellipse_scale):
        return None
    span = max(10.0, float(np.max(np.linalg.norm(
        image_points - np.mean(image_points, axis=0), axis=1))))
    sigma = max(0.04 * depth, depth * max(PIXEL_NOISE_PX, rmse) / span)
    quality = 1.0 / (1.0 + (rmse / 3.0) ** 2 + (center_error / 6.0) ** 2)
    return float(depth), float(sigma), float(rmse), float(quality), prefix

def _solve_metric_geometry(geometry):
    """Planar PnP from physical gate geometry; no range value or offset is used."""
    candidates = [item for item in (
        _pnp_family(geometry, "outer", OUTER_WIDTH_M),
        _pnp_family(geometry, "inner", INNER_WIDTH_M)) if item is not None]
    if not candidates:
        return None
    depths = np.asarray([item[0] for item in candidates])
    sigmas = np.asarray([item[1] for item in candidates])
    qualities = np.asarray([item[3] for item in candidates])
    base = qualities / np.maximum(sigmas ** 2, 1e-6)
    depth = float(np.sum(base * depths) / np.sum(base))
    radius = max(0.40, 0.10 * depth)
    robust = np.minimum(1.0, radius / np.maximum(np.abs(depths - depth), 1e-9))
    weights = base * robust
    depth = float(np.sum(weights * depths) / np.sum(weights))
    normalized = weights / np.sum(weights)
    scatter = float(np.sum(normalized * (depths - depth) ** 2))
    sigma_depth = math.sqrt(max(1.0 / float(np.sum(1.0 / sigmas ** 2)), scatter,
                                (0.035 * depth) ** 2))
    direction = _camera_ray(geometry.ellipse.center_px)
    position = direction * depth
    bearing_sigma = depth * PIXEL_NOISE_PX / FOCAL_PX
    covariance = sigma_depth ** 2 * np.outer(direction, direction)
    unit = _unit(direction)
    covariance += bearing_sigma ** 2 * (np.eye(3) - np.outer(unit, unit))
    agreement = math.exp(-0.5 * scatter / max(0.04, (0.08 * depth) ** 2))
    quality = float(np.clip(np.sum(normalized * qualities) * agreement, 0.0, 1.0))
    rmse = float(np.sum(normalized * [item[2] for item in candidates]))
    return _GateSolution(position, covariance, rmse, 4 * len(candidates), 1, 0.0,
                         quality, ("metric_planar_pnp",
                                   *(f"{item[4]}_geometry" for item in candidates)))

def _merge_current_evidence(metric, temporal):
    if metric is None:
        return temporal
    if temporal is None:
        return metric
    delta = temporal.position - metric.position
    combined = metric.covariance + temporal.covariance
    mahalanobis = float(delta @ np.linalg.pinv(combined, rcond=1e-10) @ delta)
    if mahalanobis <= 16.0 or np.linalg.norm(delta) <= max(0.60, 0.08 *
                                                           np.linalg.norm(metric.position)):
        return _covariance_intersection(metric, temporal)
    return replace(metric, quality=0.9 * metric.quality,
                   covariance=metric.covariance + 0.05 * np.outer(delta, delta),
                   groups=tuple(dict.fromkeys((*metric.groups, "rejected_temporal"))))

def _on_current_bearing(solution, pixel):
    if solution is None or solution.position[2] <= MIN_CAMERA_DEPTH_M:
        return None
    return replace(solution, position=_camera_ray(pixel) * solution.position[2])

def _to_ned(solution, origin, world_from_camera):
    """The sole camera-solution -> run-local NED conversion."""
    return replace(solution, position=origin + world_from_camera @ solution.position,
                   covariance=world_from_camera @ solution.covariance @ world_from_camera.T)
def _rectangle_center(prefix, solved):
    names = {role: f"{prefix}_{role}" for role in _ROLES}
    diagonals = [(solved[names[a]], solved[names[b]]) for a, b in
                 (("upper_left", "lower_right"), ("upper_right", "lower_left"))
                 if names[a] in solved and names[b] in solved]
    if not diagonals:
        return None
    midpoints = [(first.position + second.position) / 2.0
                 for first, second in diagonals]
    spans = [np.linalg.norm(first.position - second.position)
             for first, second in diagonals]
    closure = (float(np.linalg.norm(midpoints[0] - midpoints[1]))
               if len(midpoints) == 2 else 0.0)
    span = max(0.25, float(np.median(spans)))
    if closure > max(1.0, 0.75 * span):
        return None
    items = tuple(item for diagonal in diagonals for item in diagonal)
    position = sum(midpoints) / len(midpoints)
    covariance = (sum((first.covariance + second.covariance) / 4.0
                      for first, second in diagonals) / len(diagonals) ** 2)
    closure_quality = math.exp(-0.5 * (closure / max(0.15, 0.25 * span)) ** 2)
    completeness = 1.0 if len(diagonals) == 2 else 0.7
    return _GateSolution(
        position, covariance, float(np.median([item.rmse_px for item in items])),
        sum(item.inliers for item in items), max(item.views for item in items),
        float(np.median([item.parallax_deg for item in items])),
        float(np.median([item.quality for item in items])) * closure_quality * completeness,
        (prefix,))

def _fuse_gate(solved, plane_candidate=None):
    candidates = []
    center = solved.get("center")
    if center is not None:
        candidates.append(_GateSolution(center.position, center.covariance, center.rmse_px,
                                        center.inliers, center.views, center.parallax_deg,
                                        center.quality, ("ellipse_center",)))
    for prefix in ("outer", "inner"):
        candidate = _rectangle_center(prefix, solved)
        if candidate is not None:
            candidates.append(candidate)
    if plane_candidate is None and len(candidates) < 2:
        return None
    if plane_candidate is not None:
        consistent = []
        for candidate in candidates:
            combined = candidate.covariance + plane_candidate.covariance
            sigma = math.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(combined)))))
            if np.linalg.norm(candidate.position - plane_candidate.position) <= max(0.75, 3.0 * sigma):
                consistent.append(candidate)
        candidates = [plane_candidate, *consistent]
    if not candidates:
        return None

    base = np.asarray([max(1e-6, item.quality) /
                       max(1e-5, float(np.trace(item.covariance))) for item in candidates])
    position = sum(weight * item.position for weight, item in zip(base, candidates)) / np.sum(base)
    robust = np.ones(len(candidates))
    for _ in range(3):
        radius = max(0.35, 0.05 * float(np.linalg.norm(position)))
        distances = np.asarray([np.linalg.norm(item.position - position) for item in candidates])
        robust = np.minimum(1.0, radius / np.maximum(distances, 1e-9))
        weights = base * robust
        position = sum(weight * item.position for weight, item in zip(weights, candidates)) / np.sum(weights)
    weights = base * robust
    normalized = weights / np.sum(weights)
    covariance = sum(weight ** 2 * item.covariance
                     for weight, item in zip(normalized, candidates))
    covariance += sum(weight * np.outer(item.position - position, item.position - position)
                      for weight, item in zip(normalized, candidates))
    agreement = float(np.sum(normalized * robust))
    return _GateSolution(
        position, covariance,
        float(np.sum(normalized * [item.rmse_px for item in candidates])),
        sum(item.inliers for item in candidates), max(item.views for item in candidates),
        float(np.sum(normalized * [item.parallax_deg for item in candidates])),
        float(np.clip(np.sum(normalized * [item.quality for item in candidates]) * agreement,
                      0.0, 1.0)),
        tuple(group for item in candidates for group in item.groups))

def _feature_pixels(geometry):
    pixels = {"center": np.asarray(geometry.ellipse.center_px, dtype=float)}
    for corner in (*geometry.outer_corners, *geometry.inner_corners):
        pixels[corner.role] = np.asarray(corner.point_px, dtype=float)
    return pixels

def _reframe(solution, old_origin, old_rotation, new_origin, new_rotation):
    """Move a cached camera-frame result using odometry deltas, not a NED ray."""
    if solution is None or old_origin is None or old_rotation is None:
        return None
    rotation = new_rotation.T @ old_rotation
    translation = new_rotation.T @ (old_origin - new_origin)
    return replace(solution,
                   position=translation + rotation @ solution.position,
                   covariance=rotation @ solution.covariance @ rotation.T)

def _covariance_intersection(prior, evidence):
    """Fuse correlated cached/current evidence without double-counting it."""
    prior_information = np.linalg.pinv(prior.covariance, rcond=1e-10)
    evidence_information = np.linalg.pinv(evidence.covariance, rcond=1e-10)
    choices = []
    for weight in np.linspace(0.05, 0.95, 19):
        information = weight * prior_information + (1.0 - weight) * evidence_information
        covariance = np.linalg.pinv(information, rcond=1e-10)
        position = covariance @ (weight * prior_information @ prior.position
                                 + (1.0 - weight) * evidence_information @ evidence.position)
        choices.append((float(np.trace(covariance)), weight, position, covariance))
    _, weight, position, covariance = min(
        choices, key=lambda item: (round(item[0], 12), abs(item[1] - 0.5)))
    quality = float(np.clip(weight * prior.quality + (1.0 - weight) * evidence.quality,
                            0.0, 1.0))
    groups = tuple(dict.fromkeys((*evidence.groups, *prior.groups, "temporal_posterior")))
    return _GateSolution(
        position, covariance,
        weight * prior.rmse_px + (1.0 - weight) * evidence.rmse_px,
        max(prior.inliers, evidence.inliers), max(prior.views, evidence.views),
        max(prior.parallax_deg, evidence.parallax_deg), quality, groups)

def _independent_fusion(prior, evidence):
    prior_information = np.linalg.pinv(prior.covariance, rcond=1e-10)
    evidence_information = np.linalg.pinv(evidence.covariance, rcond=1e-10)
    covariance = np.linalg.pinv(prior_information + evidence_information, rcond=1e-10)
    position = covariance @ (prior_information @ prior.position
                             + evidence_information @ evidence.position)
    groups = tuple(dict.fromkeys((*evidence.groups, *prior.groups, "temporal_filter")))
    return _GateSolution(
        position, covariance, min(prior.rmse_px, evidence.rmse_px),
        prior.inliers + evidence.inliers, max(prior.views, evidence.views),
        max(prior.parallax_deg, evidence.parallax_deg),
        float(np.clip(0.5 * (prior.quality + evidence.quality), 0.0, 1.0)), groups)

def _posterior(prior, evidence):
    if prior is None:
        return evidence
    if evidence is None:
        return replace(prior, covariance=prior.covariance + np.eye(3) * 1e-4,
                       quality=0.995 * prior.quality,
                       groups=tuple(dict.fromkeys((*prior.groups, "temporal_prediction"))))
    delta = evidence.position - prior.position
    combined = prior.covariance + evidence.covariance
    mahalanobis = float(delta @ np.linalg.pinv(combined, rcond=1e-10) @ delta)
    distance_gate = max(0.50, 0.08 * float(np.linalg.norm(evidence.position)))
    if mahalanobis <= 16.0 or float(np.linalg.norm(delta)) <= distance_gate:
        if ("metric_planar_pnp" in prior.groups and
                "metric_planar_pnp" in evidence.groups):
            return _independent_fusion(prior, evidence)
        return _covariance_intersection(prior, evidence)
    prior_score = prior.quality / max(1e-6, float(np.trace(prior.covariance)))
    evidence_score = evidence.quality / max(1e-6, float(np.trace(evidence.covariance)))
    stronger = (evidence.views > prior.views and
                (evidence.rmse_px < 1.25 * prior.rmse_px or evidence_score > 1.5 * prior_score))
    if stronger:
        return replace(evidence,
                       groups=tuple(dict.fromkeys((*evidence.groups, "posterior_replaced"))))
    return replace(prior, covariance=prior.covariance + np.eye(3) * 0.01,
                   quality=0.90 * prior.quality,
                   groups=tuple(dict.fromkeys((*prior.groups, "rejected_current"))))

class GatePublisher:
    """Consumes one frame, caches its projections, and emits current gates only."""

    def __init__(self, source=SOURCE):
        self.source = str(source)
        self._states: dict[str, _Landmark] = {}
        self._bindings: dict[str, str] = {}
        self._tick = 0
        self._last_frame_key = None
        self._last_observation = None
        self._segment = 0

    def _fresh(self, track_id):
        gate_id = track_id if track_id not in self._states else f"{track_id}-segment-{self._segment}"
        self._segment += gate_id != track_id
        state = _Landmark(gate_id, last_seen=self._tick)
        self._states[gate_id] = state
        self._bindings[track_id] = gate_id
        return state

    @staticmethod
    def _association(state, pixels, origin, world_from_camera, ellipse_scale):
        if state.solution is None:
            return None
        prior = _reframe(state.solution, state.anchor_origin, state.anchor_rotation,
                         origin, world_from_camera)
        predicted = None if prior is None else _project_camera(prior.position)
        if predicted is None:
            return None
        center_error = float(np.linalg.norm(predicted - pixels["center"]))
        feature_errors = []
        for name, pixel in pixels.items():
            solution = state.feature_solutions.get(name)
            if solution is None:
                continue
            current = _reframe(solution, state.anchor_origin, state.anchor_rotation,
                               origin, world_from_camera)
            projected = None if current is None else _project_camera(current.position)
            if projected is not None:
                feature_errors.append(float(np.linalg.norm(projected - pixel)))
        feature_error = float(np.median(feature_errors)) if len(feature_errors) >= 3 else center_error
        threshold = max(ASSOCIATION_FLOOR_PX, 0.30 * ellipse_scale)
        if center_error > threshold or feature_error > 1.5 * threshold:
            return None
        return (0.65 * center_error + 0.35 * feature_error) / threshold

    def _assign(self, detections, origin, world_from_camera):
        assigned, result, pending = set(), {}, []
        for detection in detections:
            track_id = detection.track.track_id
            pixels = _feature_pixels(detection.geometry)
            scale = max(detection.geometry.ellipse.semi_axes_px)
            gate_id = self._bindings.get(track_id)
            state = self._states.get(gate_id) if gate_id else None
            score = self._association(state, pixels, origin, world_from_camera,
                                      scale) if state else None
            if state is not None and state.solution is None:
                score = 0.0
            if state is not None and score is not None and state.gate_id not in assigned:
                result[id(detection)] = state
                assigned.add(state.gate_id)
            else:
                if gate_id:
                    self._bindings.pop(track_id, None)
                pending.append((detection, pixels, scale))

        choices = []
        for detection, pixels, scale in pending:
            for state in self._states.values():
                if state.gate_id in assigned or self._tick - state.last_seen > MAX_MISSED_FRAMES:
                    continue
                score = self._association(state, pixels, origin, world_from_camera, scale)
                if score is not None:
                    choices.append((score, id(detection), detection, state))
        used = set()
        for _, identity, detection, state in sorted(choices, key=lambda item: item[0]):
            if identity in used or state.gate_id in assigned:
                continue
            result[identity] = state
            self._bindings[detection.track.track_id] = state.gate_id
            used.add(identity)
            assigned.add(state.gate_id)
        for detection, _, _ in pending:
            if id(detection) not in result:
                result[id(detection)] = self._fresh(detection.track.track_id)
        return result

    def _update_state(self, state, detection, frame_key, origin, world_from_camera):
        pixels = _feature_pixels(detection.geometry)
        state.observed_frames += 1
        camera_features = {}
        for name, pixel in pixels.items():
            history = state.histories.setdefault(name, [])
            views = _current_views(history, _View(
                frame_key, origin.copy(), world_from_camera.copy(), pixel.copy()))
            solution = _solve_feature(views)
            if solution is not None:
                camera_features[name] = solution
        plane_view = _View(
            frame_key, origin.copy(), world_from_camera.copy(), pixels["center"].copy(),
            {name: pixel.copy() for name, pixel in pixels.items()})
        plane_views = _current_views(state.plane_history, plane_view)
        plane_gate = _solve_plane(plane_views)
        temporal = _fuse_gate(camera_features, plane_gate)
        metric = _solve_metric_geometry(detection.geometry)
        evidence = _merge_current_evidence(metric, temporal)
        if evidence is not None:
            evidence = replace(evidence, views=max(
                evidence.views, min(MAX_VIEWS, state.observed_frames)))
        prior = _reframe(state.solution, state.anchor_origin, state.anchor_rotation,
                         origin, world_from_camera)
        state.solution = _on_current_bearing(
            _posterior(prior, evidence), pixels["center"])
        state.feature_solutions = camera_features
        state.anchor_origin = origin.copy()
        state.anchor_rotation = world_from_camera.copy()
        state.last_seen = self._tick
        return state.solution

    @staticmethod
    def _observation(state, detection, origin, world_from_camera):
        solved = state.solution
        if solved is None:
            return None
        camera = solved.position
        sigma = math.sqrt(max(0.0, float(np.max(np.linalg.eigvalsh(solved.covariance)))))
        if (camera[2] <= MIN_CAMERA_DEPTH_M or solved.quality < MIN_PUBLISH_CONFIDENCE or solved.views < MIN_PUBLISH_VIEWS
                or sigma > max(2.0, 0.35 * float(np.linalg.norm(camera)))):
            return None
        ned = _to_ned(solved, origin, world_from_camera)
        relative = world_from_camera @ camera
        trace = {
            "estimator": "robust_projective_multiview",
            "plane_depth_used": "projective_plane" in solved.groups,
            "metric_geometry_used": "metric_planar_pnp" in solved.groups,
            "scale_depth_used": False,
            "metric_depth_prior_used": False,
            "ray_cast_origin": "current_camera_optical_center",
            "ned_conversion_stage": "publication_only",
            "keyframe_baseline_m": KEYFRAME_BASELINE_M,
            "track_id": detection.track.track_id,
            "camera_position_ned_m": tuple(float(value) for value in origin),
            "position_camera_cv_m": tuple(float(value) for value in camera),
            "position_relative_ned_m": tuple(float(value) for value in relative),
            "camera_depth_m": round(float(camera[2]), 4),
            "range_m": round(float(np.linalg.norm(relative)), 4),
            "sigma_position_m": round(sigma, 4),
            "reprojection_rmse_px": round(solved.rmse_px, 3),
            "parallax_deg": round(solved.parallax_deg, 4),
            "inlier_correspondences": solved.inliers, "history_frames": solved.views,
            "center_sources": solved.groups,
        }
        return VisionGateObservation(
            gate_id=state.gate_id,
            position_local_ned=tuple(float(value) for value in ned.position),
            position_confidence=round(solved.quality, 6), trace=trace)

    def observe(self, detection_frame, vehicle_state=None):
        frame_key = (int(detection_frame.frame_id), int(detection_frame.sim_time_ns))
        if frame_key == self._last_frame_key and self._last_observation is not None:
            return self._last_observation
        self._tick += 1
        self._last_frame_key = frame_key
        stale = {gate_id for gate_id, state in self._states.items()
                 if self._tick - state.last_seen > MAX_MISSED_FRAMES}
        for gate_id in stale:
            self._states.pop(gate_id, None)
        self._bindings = {track: gate for track, gate in self._bindings.items()
                          if gate in self._states}

        gates = []
        if vehicle_state is not None and detection_frame.detections:
            origin = camera_position_ned(vehicle_state)
            world_from_camera = rotation_world_from_camera(vehicle_state.attitude_quaternion)
            assignments = self._assign(detection_frame.detections, origin, world_from_camera)
            for detection in detection_frame.detections:
                state = assignments[id(detection)]
                self._update_state(state, detection, frame_key, origin, world_from_camera)
                observation = self._observation(
                    state, detection, origin, world_from_camera)
                if observation is not None:
                    gates.append(observation)
        self._last_observation = VisionObservation(
            frame_id=detection_frame.frame_id, sim_time_ns=detection_frame.sim_time_ns,
            gates=gates, source=self.source,
            trace={"coordinate_frame": "local_ned", "position_semantics": "absolute_landmark",
                   "estimator": "projective_plane_multiview",
                   "detections": len(detection_frame.detections),
                   "published": len(gates),
                   "unobservable": len(detection_frame.detections) - len(gates),
                   "cached_tracks": len(self._states),
                   "pose_available": vehicle_state is not None})
        return self._last_observation
