#!/usr/bin/env python3
"""Build 2.7m square pose candidates from bbox-local outer contours."""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import math
from pathlib import Path
from typing import Callable

import numpy as np

from build_mask_bboxes import DEFAULT_OUTPUT_ROOT as DEFAULT_BBOX_ROOT, bbox_run_root
from build_mask_corners import DEFAULT_OUTPUT_ROOT as DEFAULT_CORNER_ROOT, corner_run_root
from build_mask_contours import DEFAULT_OUTPUT_ROOT as DEFAULT_CONTOUR_ROOT, contour_run_root
from build_vision_memory import APP_DIR, REPO_ROOT, atomic_write_json, read_json, read_json_optional, selected_precompute_manifest, stable_hash


DEFAULT_OUTPUT_ROOT = APP_DIR / "assets" / "square_pose"
DEFAULT_SQUARE_SIZE_M = 2.7
DEFAULT_EDGE_TOLERANCE_PX = 4.0
DEFAULT_MIN_EDGE_COVERAGE = 0.45
DEFAULT_MAX_REPROJECTION_ERROR_PX = 8.0
DEFAULT_MIN_CONTOUR_POINTS = 4
DEFAULT_CLIP_POSE_GUARD_ENABLED = True
DEFAULT_CLIP_INVALIDATION_THRESHOLD = 1.0
DEFAULT_CORNER_AWARE_FIT_ENABLED = True
DEFAULT_MIN_CORNER_AGREEMENT_LINKS = 2
DEFAULT_COMPLETE_CORNER_AGREEMENT_BONUS = True
DEFAULT_MULTI_GATE_EXTRA_CORNER_LIMIT = 4
DEFAULT_MAX_POSE_CANDIDATES = 12
DEFAULT_CORNER_AGREEMENT_MAX_DISTANCE_PX = 96.0
DEFAULT_CORNER_AGREEMENT_WHITE_TOLERANCE_DEG = 35.0
DEFAULT_CORNER_AGREEMENT_ANGLE_TOLERANCE_DEG = 25.0
DEFAULT_CORNER_AGREEMENT_MIN_SCORE = 0.55
CAMERA = {
    "widthPx": 640,
    "heightPx": 360,
    "fx": 320.0,
    "fy": 320.0,
    "cx": 320.0,
    "cy": 180.0,
    "distortion": "none",
}
SIDE_NAMES = ("top", "right", "bottom", "left")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def repo_url(path: Path) -> str:
    resolved = path.resolve()
    try:
        return "/" + resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def clean_path_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(value or "").strip())[:160]


def square_pose_run_root(run_key: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    clean = clean_path_component(run_key)
    if not clean:
        raise ValueError("cannot build square pose artifacts without an active run key")
    return output_root / clean


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def numeric_setting(raw: dict, key: str, default: float, low: float, high: float, integer: bool = False):
    value = raw.get(key, default)
    if value in {None, ""}:
        value = default
    value = clamp(float(value), low, high)
    return int(round(value)) if integer else value


def boolean_setting(raw: dict, key: str, default: bool) -> bool:
    if key not in raw:
        return bool(default)
    return raw.get(key) is not False


def normalized_square_pose_settings(raw: dict | None) -> dict:
    source = raw if isinstance(raw, dict) else {}
    return {
        "squareSizeM": numeric_setting(source, "squareSizeM", DEFAULT_SQUARE_SIZE_M, 0.1, 20.0),
        "edgeTolerancePx": numeric_setting(source, "edgeTolerancePx", DEFAULT_EDGE_TOLERANCE_PX, 0.25, 64.0),
        "minEdgeCoverage": numeric_setting(source, "minEdgeCoverage", DEFAULT_MIN_EDGE_COVERAGE, 0.0, 1.0),
        "maxReprojectionErrorPx": numeric_setting(source, "maxReprojectionErrorPx", DEFAULT_MAX_REPROJECTION_ERROR_PX, 0.1, 200.0),
        "minContourPoints": numeric_setting(source, "minContourPoints", DEFAULT_MIN_CONTOUR_POINTS, 4, 5000, integer=True),
        "clipPoseGuardEnabled": boolean_setting(source, "clipPoseGuardEnabled", DEFAULT_CLIP_POSE_GUARD_ENABLED),
        "clipInvalidationThreshold": numeric_setting(source, "clipInvalidationThreshold", DEFAULT_CLIP_INVALIDATION_THRESHOLD, 0.0, 1.0),
        "cornerAwareFitEnabled": boolean_setting(source, "cornerAwareFitEnabled", DEFAULT_CORNER_AWARE_FIT_ENABLED),
        "minCornerAgreementLinks": numeric_setting(source, "minCornerAgreementLinks", DEFAULT_MIN_CORNER_AGREEMENT_LINKS, 0, 12, integer=True),
        "completeCornerAgreementBonus": boolean_setting(source, "completeCornerAgreementBonus", DEFAULT_COMPLETE_CORNER_AGREEMENT_BONUS),
        "multiGateExtraCornerLimit": numeric_setting(source, "multiGateExtraCornerLimit", DEFAULT_MULTI_GATE_EXTRA_CORNER_LIMIT, 4, 24, integer=True),
        "maxPoseCandidates": numeric_setting(source, "maxPoseCandidates", DEFAULT_MAX_POSE_CANDIDATES, 1, 64, integer=True),
    }


def square_pose_settings_from_review(review: dict) -> dict:
    raw = review.get("squarePose") if isinstance(review.get("squarePose"), dict) else {}
    return normalized_square_pose_settings(raw)


def square_pose_source_signature(
    manifest: dict,
    manifest_path: Path,
    settings: dict,
    contour_signature: str | None,
    bbox_signature: str | None = None,
    corner_signature: str | None = None,
) -> str:
    return stable_hash(
        {
            "kind": "0720-square-pose-2p7m-from-contours-and-corners-v2",
            "runKey": manifest.get("runKey"),
            "precomputeManifest": repo_url(manifest_path),
            "frameCount": int(manifest.get("frameCount") or len(manifest.get("frames") or [])),
            "width": int(manifest.get("width") or CAMERA["widthPx"]),
            "height": int(manifest.get("height") or CAMERA["heightPx"]),
            "camera": CAMERA,
            "settings": normalized_square_pose_settings(settings),
            "contourSourceSignature": contour_signature,
            "bboxSourceSignature": bbox_signature,
            "cornerSourceSignature": corner_signature,
        }
    )


def polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return 0.0
    x_values = pts[:, 0]
    y_values = pts[:, 1]
    return float(0.5 * abs(np.dot(x_values, np.roll(y_values, -1)) - np.dot(y_values, np.roll(x_values, -1))))


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("quad requires four points")
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]
    ordered = np.array(
        [
            pts[int(np.argmin(sums))],
            pts[int(np.argmax(diffs))],
            pts[int(np.argmax(sums))],
            pts[int(np.argmin(diffs))],
        ],
        dtype=np.float32,
    )
    if np.unique(np.round(ordered, 3), axis=0).shape[0] == 4 and polygon_area(ordered) > 0.1:
        return ordered
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0).astype(np.float32)
    if ordered[1][1] > ordered[-1][1]:
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)
    return ordered


def point_list(points: np.ndarray, digits: int = 4) -> list[list[float]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return [[clean_float(x, digits), clean_float(y, digits)] for x, y in pts]


def vector_list(points: np.ndarray, digits: int = 6) -> list[list[float]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1)
    return [[clean_float(value, digits)] for value in pts[:, 0]]


def camera_matrix() -> np.ndarray:
    return np.array(
        [
            [CAMERA["fx"], 0.0, CAMERA["cx"]],
            [0.0, CAMERA["fy"], CAMERA["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def square_object_points(square_size_m: float) -> np.ndarray:
    half = float(square_size_m) * 0.5
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )


def initial_quad(cv2, points: np.ndarray) -> tuple[np.ndarray | None, str]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] < 4:
        return None, "not-enough-contour-points"
    contour = pts.reshape(-1, 1, 2)
    hull = cv2.convexHull(contour)
    perimeter = float(cv2.arcLength(hull, True))
    if perimeter > 0:
        for epsilon_ratio in np.linspace(0.01, 0.16, 16):
            approx = cv2.approxPolyDP(hull, epsilon_ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidate = order_quad_points(approx.reshape(4, 2))
                if polygon_area(candidate) > 0.1:
                    return candidate, "convex-hull-quad"
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    if polygon_area(box) > 0.1:
        return order_quad_points(box), "min-area-rect"
    return None, "degenerate-initial-quad"


def point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    vector = end - start
    length_sq = float(np.dot(vector, vector))
    if length_sq <= 1e-9:
        return np.linalg.norm(points - start, axis=1)
    t = ((points - start) @ vector) / length_sq
    t = np.clip(t, 0.0, 1.0)
    projection = start + t[:, None] * vector
    return np.linalg.norm(points - projection, axis=1)


def line_from_points(start: np.ndarray, end: np.ndarray) -> dict | None:
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    norm = math.hypot(dx, dy)
    if norm <= 1e-9:
        return None
    a = dy / norm
    b = -dx / norm
    c = -(a * float(start[0]) + b * float(start[1]))
    return {"a": a, "b": b, "c": c, "point": [float(start[0]), float(start[1])], "direction": [dx / norm, dy / norm]}


def fit_line(cv2, points: np.ndarray, fallback_start: np.ndarray, fallback_end: np.ndarray) -> dict:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] >= 2:
        vx, vy, x0, y0 = [float(value) for value in cv2.fitLine(pts.reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01).reshape(4)]
        norm = math.hypot(vx, vy)
        if norm > 1e-9:
            vx /= norm
            vy /= norm
            a = vy
            b = -vx
            c = -(a * x0 + b * y0)
            return {"a": a, "b": b, "c": c, "point": [x0, y0], "direction": [vx, vy], "fallback": False}
    line = line_from_points(fallback_start, fallback_end)
    if line is None:
        return {"a": 0.0, "b": -1.0, "c": float(fallback_start[1]), "point": fallback_start.tolist(), "direction": [1.0, 0.0], "fallback": True}
    line["fallback"] = True
    return line


def line_intersection(first: dict, second: dict) -> np.ndarray | None:
    a1, b1, c1 = float(first["a"]), float(first["b"]), float(first["c"])
    a2, b2, c2 = float(second["a"]), float(second["b"]), float(second["c"])
    det = a1 * b2 - a2 * b1
    if abs(det) <= 1e-8:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return np.array([x, y], dtype=np.float32)


def refine_quad_from_contour(cv2, points: np.ndarray, seed_quad: np.ndarray) -> tuple[np.ndarray, dict]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    side_distances = np.column_stack(
        [
            point_segment_distance(pts, seed_quad[index], seed_quad[(index + 1) % 4])
            for index in range(4)
        ]
    )
    nearest = np.argmin(side_distances, axis=1)
    lines = []
    assignments = {}
    for index, side in enumerate(SIDE_NAMES):
        assigned = pts[nearest == index]
        assignments[side] = assigned
        lines.append(fit_line(cv2, assigned, seed_quad[index], seed_quad[(index + 1) % 4]))

    corners = []
    for index in range(4):
        previous_line = lines[(index - 1) % 4]
        current_line = lines[index]
        point = line_intersection(previous_line, current_line)
        if point is None:
            return seed_quad, {"lines": lines, "assignments": assignments, "fallbackReason": "parallel-refined-lines"}
        corners.append(point)
    refined = order_quad_points(np.asarray(corners, dtype=np.float32))
    if polygon_area(refined) <= 0.1:
        return seed_quad, {"lines": lines, "assignments": assignments, "fallbackReason": "degenerate-refined-quad"}
    return refined, {"lines": lines, "assignments": assignments, "fallbackReason": None}


def side_metrics(points: np.ndarray, quad: np.ndarray, side_index: int, tolerance_px: float) -> dict:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    start = quad[side_index]
    end = quad[(side_index + 1) % 4]
    vector = end - start
    length = float(np.linalg.norm(vector))
    if length <= 1e-6:
        return {"coverage": 0.0, "supportPixels": 0, "residualPx": None, "lengthPx": 0.0, "score": 0.0}
    line_distance = point_segment_distance(pts, start, end)
    projection = ((pts - start) @ vector) / max(length * length, 1e-9)
    supported = (line_distance <= float(tolerance_px)) & (projection >= 0.0) & (projection <= 1.0)
    support_points = pts[supported]
    if support_points.size:
        bin_count = max(1, int(math.ceil(length)))
        bins = np.unique(np.clip(np.floor(projection[supported] * bin_count).astype(np.int32), 0, bin_count - 1))
        coverage = float(len(bins)) / float(bin_count)
        residual = float(np.median(line_distance[supported]))
    else:
        coverage = 0.0
        residual = None
    score = coverage
    if residual is not None:
        score *= max(0.0, 1.0 - residual / max(float(tolerance_px) * 2.0, 1e-6))
    return {
        "coverage": clean_float(coverage, 4),
        "supportPixels": int(support_points.shape[0]),
        "residualPx": clean_float(residual, 4) if residual is not None else None,
        "lengthPx": clean_float(length, 4),
        "score": clean_float(score, 4),
    }


def solve_square_pose(cv2, corners_px: np.ndarray, square_size_m: float) -> dict | None:
    image_points = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
    object_points = square_object_points(square_size_m)
    matrix = camera_matrix()
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    candidates = []
    flags = []
    if hasattr(cv2, "SOLVEPNP_IPPE"):
        flags.append(cv2.SOLVEPNP_IPPE)
    flags.append(cv2.SOLVEPNP_ITERATIVE)
    for flag in flags:
        try:
            if hasattr(cv2, "solvePnPGeneric"):
                result = cv2.solvePnPGeneric(object_points, image_points, matrix, dist_coeffs, flags=flag)
                if result and result[0]:
                    rvecs = result[1]
                    tvecs = result[2]
                    for rvec, tvec in zip(rvecs, tvecs):
                        candidates.append((np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1), int(flag)))
                    continue
            ok, rvec, tvec = cv2.solvePnP(object_points, image_points, matrix, dist_coeffs, flags=flag)
            if ok:
                candidates.append((np.asarray(rvec, dtype=np.float64).reshape(3, 1), np.asarray(tvec, dtype=np.float64).reshape(3, 1), int(flag)))
        except Exception:
            continue
    best = None
    for rvec, tvec, flag in candidates:
        if float(tvec[2, 0]) <= 0:
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points, axis=1)
        mean_error = float(np.mean(errors))
        candidate = {
            "rvec": [clean_float(value, 8) for value in rvec.reshape(3)],
            "tvec": [clean_float(value, 8) for value in tvec.reshape(3)],
            "depthM": clean_float(float(tvec[2, 0]), 6),
            "reprojectionErrorPx": clean_float(mean_error, 4),
            "cornerErrorsPx": [clean_float(value, 4) for value in errors],
            "cornersPx": point_list(image_points),
            "reprojectedCornersPx": point_list(projected),
            "solverFlag": flag,
        }
        if best is None or candidate["reprojectionErrorPx"] < best["reprojectionErrorPx"]:
            best = candidate
    return best


def normalized_angle_deg(value: float) -> float:
    return (float(value) % 360.0 + 360.0) % 360.0


def angle_delta_deg(first: float, second: float) -> float:
    diff = abs(normalized_angle_deg(first) - normalized_angle_deg(second)) % 360.0
    return 360.0 - diff if diff > 180.0 else diff


def angle_between_points_deg(start: dict, end: dict) -> float:
    return normalized_angle_deg(math.degrees(math.atan2(float(end["y"]) - float(start["y"]), float(end["x"]) - float(start["x"]))))


def corner_point(corner: dict) -> dict | None:
    point = corner.get("pointPx") if isinstance(corner, dict) else None
    if not isinstance(point, list) or len(point) < 2:
        return None
    x = float(point[0])
    y = float(point[1])
    return {"x": x, "y": y} if math.isfinite(x) and math.isfinite(y) else None


def corner_groups(corner_obj: dict | None) -> tuple[list[dict], list[dict]]:
    if not isinstance(corner_obj, dict):
        return [], []
    whites = [corner for corner in corner_obj.get("hullCorners") or [] if str(corner.get("cornerType") or "") == "white"]
    greens = []
    for void_entry in corner_obj.get("voids") or []:
        greens.extend([corner for corner in void_entry.get("voidCorners") or [] if str(corner.get("cornerType") or "") == "green"])
    return whites, greens


def corner_agreement_links(corner_obj: dict | None) -> list[dict]:
    whites, greens = corner_groups(corner_obj)
    candidates = []
    for white in whites:
        white_point = corner_point(white)
        if not white_point:
            continue
        for green in greens:
            if str(green.get("bboxId") or "") != str(white.get("bboxId") or ""):
                continue
            green_point = corner_point(green)
            if not green_point:
                continue
            distance = math.hypot(green_point["x"] - white_point["x"], green_point["y"] - white_point["y"])
            if distance <= 0.5 or distance > DEFAULT_CORNER_AGREEMENT_MAX_DISTANCE_PX:
                continue
            vector_angle = angle_between_points_deg(white_point, green_point)
            white_delta = angle_delta_deg(float(white.get("directionDeg") or 0.0), vector_angle)
            angle_delta = angle_delta_deg(float(white.get("angleDeg") or 0.0), float(green.get("angleDeg") or 0.0))
            if white_delta > DEFAULT_CORNER_AGREEMENT_WHITE_TOLERANCE_DEG or angle_delta > DEFAULT_CORNER_AGREEMENT_ANGLE_TOLERANCE_DEG:
                continue
            direction_score = 1.0 - white_delta / max(1.0, DEFAULT_CORNER_AGREEMENT_WHITE_TOLERANCE_DEG)
            corner_angle_score = 1.0 - angle_delta / max(1.0, DEFAULT_CORNER_AGREEMENT_ANGLE_TOLERANCE_DEG)
            distance_score = 1.0 - distance / max(1.0, DEFAULT_CORNER_AGREEMENT_MAX_DISTANCE_PX)
            score = (direction_score * 0.45) + (corner_angle_score * 0.35) + (distance_score * 0.20)
            if score < DEFAULT_CORNER_AGREEMENT_MIN_SCORE:
                continue
            candidates.append(
                {
                    "whiteCornerId": white.get("cornerId"),
                    "greenCornerId": green.get("cornerId"),
                    "whitePointPx": [clean_float(white_point["x"], 3), clean_float(white_point["y"], 3)],
                    "greenPointPx": [clean_float(green_point["x"], 3), clean_float(green_point["y"], 3)],
                    "voidId": green.get("voidId"),
                    "distancePx": clean_float(distance, 3),
                    "vectorAngleDeg": clean_float(vector_angle, 3),
                    "whiteDirectionDeltaDeg": clean_float(white_delta, 3),
                    "cornerAngleDeltaDeg": clean_float(angle_delta, 3),
                    "score": clean_float(score, 5),
                    "_whiteScore": clean_float(float(white.get("score") or 0.0), 5),
                }
            )
    candidates.sort(key=lambda item: (-float(item["score"]), float(item["distancePx"]), str(item.get("whiteCornerId") or "")))
    used_white = set()
    used_green = set()
    links = []
    for candidate in candidates:
        white_id = str(candidate.get("whiteCornerId") or "")
        green_id = str(candidate.get("greenCornerId") or "")
        if white_id in used_white or green_id in used_green:
            continue
        used_white.add(white_id)
        used_green.add(green_id)
        stored = {key: value for key, value in candidate.items() if key != "_whiteScore"}
        stored["whiteScore"] = candidate["_whiteScore"]
        links.append(stored)
    return links


def complete_corner_agreement(links: list[dict], settings: dict) -> bool:
    if not bool(settings.get("completeCornerAgreementBonus")) or len(links) < 4:
        return False
    opposite_pairs = 0
    for first, second in itertools.combinations(links, 2):
        delta = angle_delta_deg(float(first.get("vectorAngleDeg") or 0.0), float(second.get("vectorAngleDeg") or 0.0))
        if abs(180.0 - delta) <= 25.0:
            opposite_pairs += 1
    return opposite_pairs >= 2


def quad_key(points: np.ndarray) -> tuple[tuple[int, int], ...]:
    ordered = order_quad_points(points)
    return tuple((int(round(float(x) * 2.0)), int(round(float(y) * 2.0))) for x, y in ordered)


def corner_candidate_quads(corner_obj: dict | None, settings: dict) -> list[tuple[str, np.ndarray]]:
    if not bool(settings.get("cornerAwareFitEnabled")):
        return []
    whites, _greens = corner_groups(corner_obj)
    links = corner_agreement_links(corner_obj)
    point_by_id = {}
    for corner in whites:
        point = corner_point(corner)
        if point:
            point_by_id[str(corner.get("cornerId") or f"{point['x']},{point['y']}")] = (
                float(corner.get("score") or 0.0),
                np.array([point["x"], point["y"]], dtype=np.float32),
            )
    linked_ids = [str(link.get("whiteCornerId") or "") for link in links if str(link.get("whiteCornerId") or "") in point_by_id]
    all_ids = [item[0] for item in sorted(point_by_id.items(), key=lambda item: (-item[1][0], item[0]))]
    seeds = []
    if len(linked_ids) >= 4:
        seeds.extend(list(itertools.combinations(linked_ids[:8], 4)))
    if len(all_ids) >= 4:
        seeds.extend(list(itertools.combinations(all_ids[:8], 4)))
    quads = []
    seen = set()
    for ids in seeds:
        points = np.asarray([point_by_id[item][1] for item in ids], dtype=np.float32)
        if polygon_area(points) <= 4.0:
            continue
        ordered = order_quad_points(points)
        key = quad_key(ordered)
        if key in seen:
            continue
        seen.add(key)
        source = "corner-agreement" if set(ids).issubset(set(linked_ids)) else "corner-white"
        quads.append((source, ordered))
        if len(quads) >= int(settings["maxPoseCandidates"]):
            break
    return quads


def candidate_corner_support(quad: np.ndarray, links: list[dict], settings: dict) -> dict:
    if not links:
        return {"linkCount": 0, "matchedCornerCount": 0, "score": 0.0, "complete": False, "links": []}
    corners = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    match_radius = max(float(settings["edgeTolerancePx"]) * 4.0, 12.0)
    matched = set()
    used_links = []
    for link in links:
        point = np.asarray(link.get("whitePointPx") or [math.nan, math.nan], dtype=np.float32)
        if not np.all(np.isfinite(point)):
            continue
        distances = np.linalg.norm(corners - point, axis=1)
        nearest = int(np.argmin(distances))
        if float(distances[nearest]) <= match_radius:
            matched.add(nearest)
            used_links.append(link)
    link_score = min(1.0, len(used_links) / max(1.0, float(settings["minCornerAgreementLinks"])))
    corner_coverage = len(matched) / 4.0
    complete = complete_corner_agreement(used_links, settings)
    score = min(1.0, (link_score * 0.55) + (corner_coverage * 0.35) + (0.10 if complete else 0.0))
    return {
        "linkCount": len(used_links),
        "matchedCornerCount": len(matched),
        "score": clean_float(score, 5),
        "complete": bool(complete),
        "links": used_links[: int(settings["maxPoseCandidates"])],
    }


def pose_candidate(cv2, source: str, quad: np.ndarray, points: np.ndarray, settings: dict, links: list[dict]) -> dict | None:
    ordered = order_quad_points(quad)
    pose = solve_square_pose(cv2, ordered, float(settings["squareSizeM"]))
    if not pose:
        return None
    edge_signals = {
        side: side_metrics(points, ordered, index, float(settings["edgeTolerancePx"]))
        for index, side in enumerate(SIDE_NAMES)
    }
    edge_score = float(np.mean([float(item.get("score") or 0.0) for item in edge_signals.values()]))
    edge_coverage = float(np.mean([float(item.get("coverage") or 0.0) for item in edge_signals.values()]))
    reprojection_score = max(0.0, 1.0 - float(pose["reprojectionErrorPx"]) / max(0.1, float(settings["maxReprojectionErrorPx"])))
    corner_support = candidate_corner_support(ordered, links, settings)
    return {
        "candidateId": "",
        "source": source,
        "cornersPx": point_list(ordered),
        "areaPx": clean_float(polygon_area(ordered), 3),
        "pose": pose,
        "edgeSignals": edge_signals,
        "cornerAgreement": corner_support,
        "fitQuality": {
            "reprojection": clean_float(reprojection_score, 5),
            "edgeSupport": clean_float(edge_score, 5),
            "edgeCoverage": clean_float(edge_coverage, 5),
            "cornerAgreement": clean_float(float(corner_support["score"]), 5),
            "maskFit": clean_float(edge_score, 5),
            "ambiguityPenalty": 0.0,
            "tightness": 0.0,
            "overall": 0.0,
        },
        "reasonCodes": [],
    }


def score_pose_candidates(candidates: list[dict], ambiguity_flags: dict, settings: dict) -> list[dict]:
    if not candidates:
        return []
    areas = [float(item.get("areaPx") or 0.0) for item in candidates]
    min_area = min(areas)
    max_area = max(areas)
    for candidate in candidates:
        quality = candidate["fitQuality"]
        if max_area > min_area and (ambiguity_flags.get("ambiguousOverlap") or ambiguity_flags.get("multiGateEvidence")):
            tightness = 1.0 - (float(candidate.get("areaPx") or min_area) - min_area) / max(1e-6, max_area - min_area)
        else:
            tightness = 0.5
        ambiguity_penalty = 0.0
        if ambiguity_flags.get("multiGateEvidence"):
            ambiguity_penalty += 0.12
        if ambiguity_flags.get("ambiguousOverlap"):
            ambiguity_penalty += 0.08
        if float(quality["cornerAgreement"]) >= 0.9:
            ambiguity_penalty *= 0.45
        overall = (
            float(quality["reprojection"]) * 0.25
            + float(quality["edgeSupport"]) * 0.30
            + float(quality["cornerAgreement"]) * 0.30
            + tightness * 0.15
            - ambiguity_penalty
        )
        quality["tightness"] = clean_float(tightness, 5)
        quality["ambiguityPenalty"] = clean_float(ambiguity_penalty, 5)
        quality["overall"] = clean_float(max(0.0, min(1.0, overall)), 5)
        if any(float(side.get("coverage") or 0.0) < float(settings["minEdgeCoverage"]) for side in candidate.get("edgeSignals", {}).values()):
            candidate["reasonCodes"].append("low-edge-coverage")
        if float(candidate["pose"]["reprojectionErrorPx"]) > float(settings["maxReprojectionErrorPx"]):
            candidate["reasonCodes"].append("high-reprojection-error")
    candidates.sort(key=lambda item: (-float(item["fitQuality"]["overall"]), len(item["reasonCodes"]), float(item["pose"]["reprojectionErrorPx"])))
    for index, candidate in enumerate(candidates, start=1):
        candidate["candidateId"] = f"pose-candidate-{index:03d}"
    return candidates[: int(settings["maxPoseCandidates"])]


def fov_measurement_validity(bbox_obj: dict | None, settings: dict) -> dict:
    fov = bbox_obj.get("fovClip") if isinstance(bbox_obj, dict) and isinstance(bbox_obj.get("fovClip"), dict) else {}
    status = str(fov.get("status") or "unknown")
    severity = float(fov.get("severity") or 0.0)
    clipped = bool(settings.get("clipPoseGuardEnabled")) and status == "clipped" and severity >= float(settings["clipInvalidationThreshold"])
    return {
        "valid": not clipped,
        "reasonCodes": ["fov-clipped"] if clipped else [],
        "clipped": bool(clipped),
        "clippedSides": list(fov.get("sides") or []),
        "nearSides": list(fov.get("nearSides") or []),
        "clipSeverity": clean_float(severity, 4),
        "clipStatus": status,
    }


def object_pose(cv2, obj: dict, settings: dict, bbox_obj: dict | None = None, corner_obj: dict | None = None) -> dict:
    bbox_id = str(obj.get("bboxId") or "")
    outer = obj.get("outer") if isinstance(obj.get("outer"), dict) else None
    source_points = []
    if outer:
        source_points = outer.get("rawPointsPx") or outer.get("pointsPx") or []
    points = np.asarray(source_points, dtype=np.float32).reshape(-1, 2)
    base = {
        "bboxId": bbox_id,
        "frameOrdinal": obj.get("frameOrdinal"),
        "bboxPx": obj.get("bboxPx"),
        "bboxUv": obj.get("bboxUv"),
        "pixelCount": obj.get("pixelCount"),
        "accepted": False,
        "status": "rejected",
        "reasonCodes": [],
        "initialMethod": None,
        "initialCornersPx": [],
        "refinedCornersPx": [],
        "edgeSignals": {},
        "bestPose": None,
        "sideCandidates": {},
        "measurementValidity": fov_measurement_validity(bbox_obj, settings),
        "ambiguityFlags": {"ambiguousOverlap": False, "multiGateEvidence": False},
        "poseCandidates": [],
        "selectedCandidateId": None,
        "fitQuality": {},
        "cornerAgreement": {"linkCount": 0, "complete": False},
    }
    if points.shape[0] < int(settings["minContourPoints"]):
        base["reasonCodes"].append("not-enough-contour-points")
        return base
    seed, method = initial_quad(cv2, points)
    base["initialMethod"] = method
    if seed is None:
        base["reasonCodes"].append(method)
        return base
    refined, detail = refine_quad_from_contour(cv2, points, seed)
    base["initialCornersPx"] = point_list(seed)
    base["refinedCornersPx"] = point_list(refined)
    if detail.get("fallbackReason"):
        base["reasonCodes"].append(detail["fallbackReason"])
    links = corner_agreement_links(corner_obj) if bool(settings.get("cornerAwareFitEnabled")) else []
    whites, greens = corner_groups(corner_obj)
    extra_limit = int(settings["multiGateExtraCornerLimit"])
    ambiguity_flags = {
        "ambiguousOverlap": bool(len(whites) > extra_limit or len(greens) > extra_limit),
        "multiGateEvidence": bool(int((corner_obj or {}).get("voidCount") or 0) > 1),
    }
    candidate_sources = [("contour-refined", refined)]
    candidate_sources.extend(corner_candidate_quads(corner_obj, settings))
    candidates = []
    seen_quads = set()
    for source, quad in candidate_sources:
        try:
            key = quad_key(quad)
        except Exception:
            continue
        if key in seen_quads:
            continue
        seen_quads.add(key)
        candidate = pose_candidate(cv2, source, quad, points, settings, links)
        if candidate:
            candidates.append(candidate)
    candidates = score_pose_candidates(candidates, ambiguity_flags, settings)
    if len(candidates) > 1:
        best = candidates[0]
        best_area = float(best.get("areaPx") or 0.0)
        best_score = float(best.get("fitQuality", {}).get("overall") or 0.0)
        for alternate in candidates[1:4]:
            area = float(alternate.get("areaPx") or 0.0)
            score = float(alternate.get("fitQuality", {}).get("overall") or 0.0)
            area_delta = abs(area - best_area) / max(1.0, best_area)
            if best_score - score <= 0.04 and area_delta >= 0.18:
                ambiguity_flags["ambiguousOverlap"] = True
                candidates = score_pose_candidates(candidates, ambiguity_flags, settings)
                break
    base["ambiguityFlags"] = ambiguity_flags
    base["poseCandidates"] = candidates
    if not candidates:
        base["reasonCodes"].append("pose-solve-failed")
        return base
    selected = candidates[0]
    pose = selected["pose"]
    edge_signals = selected["edgeSignals"]
    base["refinedCornersPx"] = selected["cornersPx"]
    base["edgeSignals"] = edge_signals
    base["bestPose"] = pose
    base["selectedCandidateId"] = selected["candidateId"]
    base["fitQuality"] = selected["fitQuality"]
    base["cornerAgreement"] = selected["cornerAgreement"]
    for side in SIDE_NAMES:
        base["sideCandidates"][side] = {
            "primarySide": side,
            "score": edge_signals[side]["score"],
            "coverage": edge_signals[side]["coverage"],
            "pose": pose,
        }
    low_coverage = [side for side, metrics in edge_signals.items() if float(metrics["coverage"]) < float(settings["minEdgeCoverage"])]
    if low_coverage:
        base["reasonCodes"].append("low-edge-coverage:" + ",".join(low_coverage))
    if float(pose["reprojectionErrorPx"]) > float(settings["maxReprojectionErrorPx"]):
        base["reasonCodes"].append("high-reprojection-error")
    if not base["measurementValidity"]["valid"]:
        base["reasonCodes"].extend([f"pose-measurement-invalid:{reason}" for reason in base["measurementValidity"].get("reasonCodes") or []])
    if ambiguity_flags["ambiguousOverlap"]:
        base["reasonCodes"].append("ambiguous-overlap")
    if ambiguity_flags["multiGateEvidence"]:
        base["reasonCodes"].append("multi-gate-evidence")
    base["accepted"] = not base["reasonCodes"]
    base["status"] = "accepted" if base["accepted"] else "rejected"
    return base


def load_contour_manifest(run_key: str, output_root: Path) -> tuple[dict, Path]:
    path = contour_run_root(run_key, output_root) / "contours_manifest.json"
    if not path.exists():
        raise ValueError(f"contour manifest is missing for run {run_key}")
    return read_json(path), path


def optional_manifest_lookup(manifest: dict | None, object_key: str) -> dict[tuple[int, str], dict]:
    lookup = {}
    if not manifest:
        return lookup
    for frame in manifest.get("frames") or []:
        frame_ordinal = int(frame.get("frameOrdinal") or 0)
        for obj in frame.get(object_key) or []:
            bbox_id = str(obj.get("bboxId") or "")
            if bbox_id:
                lookup[(frame_ordinal, bbox_id)] = obj
    return lookup


def optional_corner_lookup(manifest: dict | None) -> dict[tuple[int, str], dict]:
    return optional_manifest_lookup(manifest, "bboxCorners")


def build_square_pose(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
    try:
        import cv2  # type: ignore
    except Exception as error:  # pragma: no cover - depends on local environment
        raise RuntimeError(f"OpenCV is required for square pose fitting: {error}") from error

    raw_settings = getattr(args, "square_pose_settings", None)
    settings = normalized_square_pose_settings(raw_settings)
    manifest_path = args.precompute_manifest or selected_precompute_manifest()
    manifest = read_json(manifest_path)
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or CAMERA["widthPx"])
    height = int(manifest.get("height") or CAMERA["heightPx"])
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))
    if width != CAMERA["widthPx"] or height != CAMERA["heightPx"]:
        raise ValueError(f"square pose v1 expects {CAMERA['widthPx']}x{CAMERA['heightPx']} frames, got {width}x{height}")
    contour_manifest, contour_manifest_path = load_contour_manifest(run_key, getattr(args, "contour_output_root", DEFAULT_CONTOUR_ROOT))
    contour_signature = str(contour_manifest.get("sourceSignature") or "")
    bbox_manifest_path = bbox_run_root(run_key, getattr(args, "bbox_output_root", DEFAULT_BBOX_ROOT)) / "bbox_manifest.json"
    bbox_manifest = read_json_optional(bbox_manifest_path)
    bbox_signature = str(bbox_manifest.get("sourceSignature") or "") if bbox_manifest else None
    corner_manifest_path = corner_run_root(run_key, getattr(args, "corner_output_root", DEFAULT_CORNER_ROOT)) / "corners_manifest.json"
    corner_manifest = read_json_optional(corner_manifest_path)
    corner_signature = str(corner_manifest.get("sourceSignature") or "") if corner_manifest else None
    source_signature = square_pose_source_signature(
        manifest,
        manifest_path,
        settings,
        contour_signature,
        bbox_signature if bool(settings.get("clipPoseGuardEnabled")) else None,
        corner_signature if bool(settings.get("cornerAwareFitEnabled")) else None,
    )
    if str(contour_manifest.get("runKey") or "") != run_key:
        raise ValueError("contour manifest run key does not match active run")
    if int(contour_manifest.get("sourceFrameCount") or 0) != frame_count:
        raise ValueError("contour manifest frame count does not match active run")
    if not contour_manifest.get("complete"):
        raise ValueError("square pose requires a complete contour manifest")
    bbox_lookup = optional_manifest_lookup(bbox_manifest, "bboxes")
    corner_lookup = optional_corner_lookup(corner_manifest)

    start = max(0, int(args.start_frame or 0))
    end = frame_count
    if args.max_frames is not None:
        end = min(end, start + max(0, int(args.max_frames)))
    contour_frames = list(contour_manifest.get("frames") or [])
    selected_frames = list(enumerate(contour_frames[start:end], start=start))
    if not selected_frames:
        raise ValueError("selected square-pose frame range is empty")

    output_root = square_pose_run_root(run_key, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    frames_out = []
    totals = {
        "objectCount": 0,
        "acceptedCount": 0,
        "rejectedCount": 0,
        "poseCount": 0,
        "usablePoseCount": 0,
        "invalidPoseCount": 0,
        "clippedIgnoredCount": 0,
        "ambiguousOverlapCount": 0,
        "multiGateEvidenceCount": 0,
    }
    error_values = []
    depth_values = []
    for local_index, (frame_ordinal, frame_entry) in enumerate(selected_frames):
        objects = []
        for obj in list(frame_entry.get("objects") or []):
            bbox_id = str(obj.get("bboxId") or "")
            lookup_key = (int(frame_entry.get("frameOrdinal", frame_ordinal) or frame_ordinal), bbox_id)
            pose = object_pose(cv2, obj, settings, bbox_lookup.get(lookup_key), corner_lookup.get(lookup_key))
            objects.append(pose)
            totals["objectCount"] += 1
            if pose.get("accepted"):
                totals["acceptedCount"] += 1
            else:
                totals["rejectedCount"] += 1
            if pose.get("bestPose"):
                totals["poseCount"] += 1
                error_values.append(float(pose["bestPose"]["reprojectionErrorPx"]))
                depth_values.append(float(pose["bestPose"]["depthM"]))
            validity = pose.get("measurementValidity") if isinstance(pose.get("measurementValidity"), dict) else {}
            ambiguity = pose.get("ambiguityFlags") if isinstance(pose.get("ambiguityFlags"), dict) else {}
            if pose.get("bestPose") and validity.get("valid") is not False:
                totals["usablePoseCount"] += 1
            if pose.get("bestPose") and validity.get("valid") is False:
                totals["invalidPoseCount"] += 1
            if validity.get("clipped"):
                totals["clippedIgnoredCount"] += 1
            if ambiguity.get("ambiguousOverlap"):
                totals["ambiguousOverlapCount"] += 1
            if ambiguity.get("multiGateEvidence"):
                totals["multiGateEvidenceCount"] += 1
        frame_summary = {
            "objectCount": len(objects),
            "acceptedCount": sum(1 for item in objects if item.get("accepted")),
            "rejectedCount": sum(1 for item in objects if not item.get("accepted")),
            "poseCount": sum(1 for item in objects if item.get("bestPose")),
            "usablePoseCount": sum(1 for item in objects if item.get("bestPose") and (item.get("measurementValidity") or {}).get("valid") is not False),
            "invalidPoseCount": sum(1 for item in objects if item.get("bestPose") and (item.get("measurementValidity") or {}).get("valid") is False),
            "ambiguousOverlapCount": sum(1 for item in objects if (item.get("ambiguityFlags") or {}).get("ambiguousOverlap")),
            "multiGateEvidenceCount": sum(1 for item in objects if (item.get("ambiguityFlags") or {}).get("multiGateEvidence")),
        }
        frames_out.append(
            {
                "frameOrdinal": frame_entry.get("frameOrdinal", frame_ordinal),
                "frameId": frame_entry.get("frameId"),
                "path": frame_entry.get("path"),
                "cachePath": frame_entry.get("cachePath"),
                **frame_summary,
                "objects": objects,
            }
        )
        if progress_callback:
            progress_callback(
                {
                    "phase": "building-square-pose",
                    "index": local_index + 1,
                    "total": len(selected_frames),
                    "frameOrdinal": frame_ordinal,
                    **frame_summary,
                }
            )

    summary = {
        **totals,
        "meanReprojectionErrorPx": clean_float(float(np.mean(error_values)), 4) if error_values else None,
        "maxReprojectionErrorPx": clean_float(float(np.max(error_values)), 4) if error_values else None,
        "meanDepthM": clean_float(float(np.mean(depth_values)), 6) if depth_values else None,
        "minDepthM": clean_float(float(np.min(depth_values)), 6) if depth_values else None,
        "maxDepthM": clean_float(float(np.max(depth_values)), 6) if depth_values else None,
    }
    output = {
        "version": 1,
        "app": "vision_passthrough_review",
        "kind": "square-pose-2p7m-from-contours",
        "createdAt": utc_now(),
        "runKey": run_key,
        "precomputeManifest": repo_url(manifest_path),
        "sourceSignature": source_signature,
        "complete": start == 0 and len(frames_out) >= frame_count,
        "frameStart": start,
        "frameCount": len(frames_out),
        "sourceFrameCount": frame_count,
        "image": {"width": width, "height": height},
        "camera": CAMERA,
        "square": {"widthM": clean_float(settings["squareSizeM"], 6), "heightM": clean_float(settings["squareSizeM"], 6)},
        "settings": settings,
        "input": {
            "contourManifest": repo_url(contour_manifest_path),
            "contourSourceSignature": contour_signature,
            "bboxManifest": repo_url(bbox_manifest_path) if bbox_manifest_path.exists() else None,
            "bboxSourceSignature": bbox_signature,
            "cornerManifest": repo_url(corner_manifest_path) if corner_manifest_path.exists() else None,
            "cornerSourceSignature": corner_signature,
            "clipPoseGuardAvailable": bool(bbox_manifest),
            "cornerAwareAvailable": bool(corner_manifest),
        },
        "summary": summary,
        "frames": frames_out,
    }
    atomic_write_json(output_root / "square_pose_manifest.json", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precompute-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--contour-output-root", type=Path, default=DEFAULT_CONTOUR_ROOT)
    parser.add_argument("--bbox-output-root", type=Path, default=DEFAULT_BBOX_ROOT)
    parser.add_argument("--corner-output-root", type=Path, default=DEFAULT_CORNER_ROOT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--square-size-m", type=float, default=DEFAULT_SQUARE_SIZE_M)
    parser.add_argument("--edge-tolerance-px", type=float, default=DEFAULT_EDGE_TOLERANCE_PX)
    parser.add_argument("--min-edge-coverage", type=float, default=DEFAULT_MIN_EDGE_COVERAGE)
    parser.add_argument("--max-reprojection-error-px", type=float, default=DEFAULT_MAX_REPROJECTION_ERROR_PX)
    parser.add_argument("--min-contour-points", type=int, default=DEFAULT_MIN_CONTOUR_POINTS)
    parser.add_argument("--clip-pose-guard-enabled", action=argparse.BooleanOptionalAction, default=DEFAULT_CLIP_POSE_GUARD_ENABLED)
    parser.add_argument("--clip-invalidation-threshold", type=float, default=DEFAULT_CLIP_INVALIDATION_THRESHOLD)
    parser.add_argument("--corner-aware-fit-enabled", action=argparse.BooleanOptionalAction, default=DEFAULT_CORNER_AWARE_FIT_ENABLED)
    parser.add_argument("--min-corner-agreement-links", type=int, default=DEFAULT_MIN_CORNER_AGREEMENT_LINKS)
    parser.add_argument("--complete-corner-agreement-bonus", action=argparse.BooleanOptionalAction, default=DEFAULT_COMPLETE_CORNER_AGREEMENT_BONUS)
    parser.add_argument("--multi-gate-extra-corner-limit", type=int, default=DEFAULT_MULTI_GATE_EXTRA_CORNER_LIMIT)
    parser.add_argument("--max-pose-candidates", type=int, default=DEFAULT_MAX_POSE_CANDIDATES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.square_pose_settings = {
        "squareSizeM": args.square_size_m,
        "edgeTolerancePx": args.edge_tolerance_px,
        "minEdgeCoverage": args.min_edge_coverage,
        "maxReprojectionErrorPx": args.max_reprojection_error_px,
        "minContourPoints": args.min_contour_points,
        "clipPoseGuardEnabled": args.clip_pose_guard_enabled,
        "clipInvalidationThreshold": args.clip_invalidation_threshold,
        "cornerAwareFitEnabled": args.corner_aware_fit_enabled,
        "minCornerAgreementLinks": args.min_corner_agreement_links,
        "completeCornerAgreementBonus": args.complete_corner_agreement_bonus,
        "multiGateExtraCornerLimit": args.multi_gate_extra_corner_limit,
        "maxPoseCandidates": args.max_pose_candidates,
    }
    result = build_square_pose(args)
    print(
        f"wrote {result['frameCount']} square-pose frames, "
        f"{result['summary']['poseCount']} poses, "
        f"{result['summary']['acceptedCount']} accepted"
    )


if __name__ == "__main__":
    main()
