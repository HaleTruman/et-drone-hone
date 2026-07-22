#!/usr/bin/env python3
"""Build compact frame-to-frame instance mappings from bboxes and pose fits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Callable


APP_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_DIR.parent
TOOLS_DIR = APP_DIR / "tools"
SRC_DIR = APP_DIR / "src"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_vision_memory import atomic_write_json, read_json, read_json_optional, selected_precompute_manifest, stable_hash  # noqa: E402
from mask_bbox.build_from_maskbits import DEFAULT_OUTPUT_ROOT as DEFAULT_MASKBITS_BBOX_ROOT, maskbits_bbox_run_root  # noqa: E402
from pose_estimation.pose_fit import DEFAULT_OUTPUT_ROOT as DEFAULT_POSE_ROOT, pose_run_root  # noqa: E402


DEFAULT_OUTPUT_ROOT = APP_DIR / "assets" / "instance_tracking"
DEFAULT_ENABLED = False
DEFAULT_SHOW_OVERLAY = True
DEFAULT_SHOW_LABELS = True
DEFAULT_SHOW_LINKS = True
DEFAULT_SHOW_CANDIDATES = False
DEFAULT_SHOW_READOUT = True
DEFAULT_MAX_FRAME_GAP = 5
DEFAULT_MIN_ASSOCIATION_SCORE = 0.45
DEFAULT_MAX_2D_DISTANCE_PX = 90.0
DEFAULT_MIN_IOU = 0.02
DEFAULT_MAX_3D_DISTANCE_M = 3.0
DEFAULT_MAX_RPY_DELTA_DEG = 45.0
DEFAULT_SPLIT_CANDIDATE_SCORE = 0.35
DEFAULT_DEBUG_CANDIDATE_LIMIT = 5
DEFAULT_TRAIL_LENGTH_FRAMES = 12
DEFAULT_POSE_SOURCE = "poseFit"
POSE_SOURCES = {"poseFit", "squarePose", "none"}

SIGNAL_WEIGHTS = {
    "center2d": 0.25,
    "bboxIou": 0.15,
    "sizeSimilarity": 0.10,
    "layerMix": 0.15,
    "pose3d": 0.20,
    "rpy": 0.10,
    "velocity2d": 0.05,
}

CONTRACT = {
    "system": "instance_tracking",
    "status": "active",
    "inputs": [
        {"name": "maskbits_bboxes", "path": "assets/mask_bboxes_maskbits/<run_key>/bbox_manifest.json", "required": True},
        {"name": "pose_fit", "path": "assets/pose_estimation/<run_key>/3d_pose_fit.json", "required": True},
    ],
    "output": {"name": "instance_mapping", "path": "assets/instance_tracking/<run_key>/instance_mapping.json"},
    "behavior": [
        "read bbox observations in frame order",
        "read per-bbox pose fit records",
        "assign stable instance IDs across frames",
        "preserve bbox observations when pose measurements are invalid",
    ],
    "non_goals": ["no corner input", "no semantic/template fit", "no legacy instance_provenance output writes"],
}


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


def instance_mapping_run_root(run_key: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path:
    clean = clean_path_component(run_key)
    if not clean:
        raise ValueError("cannot build instance tracking artifacts without an active run key")
    return output_root / clean


def clean_float(value: float, digits: int = 4) -> float:
    return float(round(float(value), digits))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def bool_setting(value, default: bool) -> bool:
    if value in {None, ""}:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def numeric_setting(raw: dict, key: str, default: float, low: float, high: float, integer: bool = False):
    value = raw.get(key, default)
    if value in {None, ""}:
        value = default
    value = clamp(float(value), low, high)
    return int(round(value)) if integer else value


def normalized_instance_tracking_settings(raw: dict | None) -> dict:
    source = raw if isinstance(raw, dict) else {}
    pose_source = str(source.get("poseSource") or DEFAULT_POSE_SOURCE)
    if pose_source == "squarePose":
        pose_source = "poseFit"
    if pose_source not in POSE_SOURCES:
        pose_source = DEFAULT_POSE_SOURCE
    return {
        "enabled": bool_setting(source.get("enabled"), DEFAULT_ENABLED),
        "showOverlay": bool_setting(source.get("showOverlay"), DEFAULT_SHOW_OVERLAY),
        "showLabels": bool_setting(source.get("showLabels"), DEFAULT_SHOW_LABELS),
        "showLinks": bool_setting(source.get("showLinks"), DEFAULT_SHOW_LINKS),
        "showCandidates": bool_setting(source.get("showCandidates"), DEFAULT_SHOW_CANDIDATES),
        "showReadout": bool_setting(source.get("showReadout"), DEFAULT_SHOW_READOUT),
        "maxFrameGap": numeric_setting(source, "maxFrameGap", DEFAULT_MAX_FRAME_GAP, 1, 120, integer=True),
        "minAssociationScore": numeric_setting(source, "minAssociationScore", DEFAULT_MIN_ASSOCIATION_SCORE, 0.0, 1.0),
        "max2dDistancePx": numeric_setting(source, "max2dDistancePx", DEFAULT_MAX_2D_DISTANCE_PX, 1.0, 1000.0),
        "minIou": numeric_setting(source, "minIou", DEFAULT_MIN_IOU, 0.0, 1.0),
        "max3dDistanceM": numeric_setting(source, "max3dDistanceM", DEFAULT_MAX_3D_DISTANCE_M, 0.01, 100.0),
        "maxRpyDeltaDeg": numeric_setting(source, "maxRpyDeltaDeg", DEFAULT_MAX_RPY_DELTA_DEG, 1.0, 180.0),
        "splitCandidateScore": numeric_setting(source, "splitCandidateScore", DEFAULT_SPLIT_CANDIDATE_SCORE, 0.0, 1.0),
        "debugCandidateLimit": numeric_setting(source, "debugCandidateLimit", DEFAULT_DEBUG_CANDIDATE_LIMIT, 1, 50, integer=True),
        "trailLengthFrames": numeric_setting(source, "trailLengthFrames", DEFAULT_TRAIL_LENGTH_FRAMES, 1, 240, integer=True),
        "poseSource": pose_source,
    }


def instance_tracking_settings_from_review(review: dict) -> dict:
    raw = review.get("instanceTracking") if isinstance(review.get("instanceTracking"), dict) else {}
    return normalized_instance_tracking_settings(raw)


def instance_tracking_signature_settings(settings: dict) -> dict:
    clean = dict(normalized_instance_tracking_settings(settings))
    for visual_key in ("enabled", "showOverlay", "showLabels", "showLinks", "showCandidates", "showReadout", "trailLengthFrames"):
        clean.pop(visual_key, None)
    return clean


def instance_tracking_source_signature(
    manifest: dict,
    manifest_path: Path,
    settings: dict,
    bbox_signature: str | None,
    pose_signature: str | None,
) -> str:
    return stable_hash(
        {
            "kind": "0721-instance-tracking-pose-estimation-v1",
            "runKey": manifest.get("runKey"),
            "precomputeManifest": repo_url(manifest_path),
            "frameCount": int(manifest.get("frameCount") or len(manifest.get("frames") or [])),
            "width": int(manifest.get("width") or 640),
            "height": int(manifest.get("height") or 360),
            "settings": instance_tracking_signature_settings(settings),
            "bboxSourceSignature": bbox_signature,
            "poseSourceSignature": pose_signature,
        }
    )


def rodrigues_to_matrix(rvec: list[float]) -> list[list[float]]:
    rx, ry, rz = [float(value) for value in (rvec or [0, 0, 0])[:3]]
    theta = math.sqrt(rx * rx + ry * ry + rz * rz)
    if not math.isfinite(theta) or theta <= 1e-12:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x, y, z = rx / theta, ry / theta, rz / theta
    c, s, t = math.cos(theta), math.sin(theta), 1 - math.cos(theta)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def euler_xyz_deg_from_matrix(r: list[list[float]]) -> dict:
    value = max(-1.0, min(1.0, float(r[0][2])))
    pitch = math.asin(value)
    if abs(value) < 0.9999999:
        roll = math.atan2(-float(r[1][2]), float(r[2][2]))
        yaw = math.atan2(-float(r[0][1]), float(r[0][0]))
    else:
        roll = math.atan2(float(r[2][1]), float(r[1][1]))
        yaw = 0.0
    to_deg = 180.0 / math.pi
    return {"roll": clean_float(roll * to_deg, 4), "pitch": clean_float(pitch * to_deg, 4), "yaw": clean_float(yaw * to_deg, 4), "order": "XYZ"}


def pose_from_pose_object(pose_obj: dict | None) -> dict:
    pose = pose_obj.get("bestPose") if isinstance(pose_obj, dict) and isinstance(pose_obj.get("bestPose"), dict) else None
    validity = pose_obj.get("measurementValidity") if isinstance(pose_obj, dict) and isinstance(pose_obj.get("measurementValidity"), dict) else {}
    if pose and validity.get("valid") is False:
        return {
            "available": False,
            "ignored": True,
            "reasonCodes": list(validity.get("reasonCodes") or []),
            "measurementValidity": validity,
            "selectedCandidateId": pose_obj.get("selectedCandidateId") if isinstance(pose_obj, dict) else None,
            "fitQuality": pose_obj.get("fitQuality") if isinstance(pose_obj, dict) else None,
        }
    if not pose:
        return {"available": False}
    tvec = [float(value) for value in (pose.get("tvec") or [])[:3]]
    rvec = [float(value) for value in (pose.get("rvec") or [])[:3]]
    if len(tvec) != 3 or len(rvec) != 3:
        return {"available": False}
    rpy = euler_xyz_deg_from_matrix(rodrigues_to_matrix(rvec))
    return {
        "available": True,
        "accepted": bool(pose_obj.get("accepted")),
        "measurementValidity": validity or {"valid": True},
        "selectedCandidateId": pose_obj.get("selectedCandidateId"),
        "fitQuality": pose_obj.get("fitQuality"),
        "ambiguityFlags": pose_obj.get("ambiguityFlags"),
        "xyzCameraM": [clean_float(value, 6) for value in tvec],
        "rpyCameraDeg": [rpy["roll"], rpy["pitch"], rpy["yaw"]],
        "rpyOrder": rpy["order"],
        "rvec": [clean_float(value, 8) for value in rvec],
        "tvec": [clean_float(value, 8) for value in tvec],
        "reprojectionErrorPx": clean_float(float(pose.get("reprojectionErrorPx") or 0.0), 4),
        "depthM": clean_float(float(pose.get("depthM") or tvec[2]), 6),
    }


def rect_iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0 + 1.0), max(0.0, iy1 - iy0 + 1.0)
    intersection = iw * ih
    area_a = max(0.0, ax1 - ax0 + 1.0) * max(0.0, ay1 - ay0 + 1.0)
    area_b = max(0.0, bx1 - bx0 + 1.0) * max(0.0, by1 - by0 + 1.0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def cosine_similarity(first: dict[str, float], second: dict[str, float]) -> float:
    keys = set(first) | set(second)
    if not keys:
        return 0.0
    dot = sum(float(first.get(key, 0.0)) * float(second.get(key, 0.0)) for key in keys)
    norm_a = math.sqrt(sum(float(first.get(key, 0.0)) ** 2 for key in keys))
    norm_b = math.sqrt(sum(float(second.get(key, 0.0)) ** 2 for key in keys))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


def layer_mix(layer_counts: list[dict]) -> dict[str, float]:
    raw = {str(item.get("prefix")): float(item.get("pixelCount") or 0) for item in layer_counts or []}
    total = sum(raw.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in sorted(raw.items())}


def pose_lookup(pose_manifest: dict | None) -> dict[tuple[int, str], dict]:
    lookup = {}
    if not pose_manifest:
        return lookup
    for frame in pose_manifest.get("frames") or []:
        frame_ordinal = int(frame.get("frameOrdinal") or 0)
        for obj in frame.get("objects") or []:
            bbox_id = str(obj.get("bboxId") or "")
            if bbox_id:
                lookup[(frame_ordinal, bbox_id)] = obj
    return lookup


def build_observation(frame_ordinal: int, bbox_index: int, bbox: dict, pose_by_key: dict[tuple[int, str], dict], pose_source: str) -> dict:
    bbox_id = str(bbox.get("bboxId") or f"bbox-{frame_ordinal:06d}-{bbox_index + 1:03d}")
    bbox_px = [float(value) for value in (bbox.get("bboxPx") or [0, 0, 0, 0])[:4]]
    centroid = [float(value) for value in (bbox.get("centroidPx") or [(bbox_px[0] + bbox_px[2]) / 2.0, (bbox_px[1] + bbox_px[3]) / 2.0])[:2]]
    area = max(1.0, (bbox_px[2] - bbox_px[0] + 1.0) * (bbox_px[3] - bbox_px[1] + 1.0))
    pose = pose_from_pose_object(pose_by_key.get((frame_ordinal, bbox_id))) if pose_source == "poseFit" else {"available": False}
    pose["source"] = pose_source
    fov_clip = bbox.get("fovClip") if isinstance(bbox.get("fovClip"), dict) else {}
    clip_severity = float(fov_clip.get("severity") or 0.0)
    pose_quality = max(0.0, 1.0 - float(pose.get("reprojectionErrorPx") or 0.0) / 20.0) if pose.get("available") else 0.5
    observation_quality = max(0.0, min(1.0, (0.45 * min(1.0, float(bbox.get("pixelCount") or 0) / 1000.0)) + (0.35 * pose_quality) + (0.20 * (1.0 - clip_severity))))
    return {
        "observationId": f"obs-{frame_ordinal:06d}-{bbox_index + 1:03d}",
        "frameOrdinal": frame_ordinal,
        "bboxId": bbox_id,
        "bboxPx": [clean_float(value, 3) for value in bbox_px],
        "bboxUv": bbox.get("bboxUv"),
        "centroidPx": [clean_float(value, 4) for value in centroid],
        "pixelCount": int(bbox.get("pixelCount") or 0),
        "areaPx": clean_float(area, 3),
        "layerMix": layer_mix(bbox.get("layerCounts") or []),
        "fovClip": fov_clip,
        "pose": pose,
        "observationQuality": clean_float(observation_quality, 4),
        "source": {"bbox": bbox},
    }


def predicted_center(track: dict, frame_ordinal: int) -> list[float] | None:
    last = track.get("lastObservation")
    prev = track.get("previousObservation")
    if not last or not prev:
        return None
    gap = int(frame_ordinal) - int(last["frameOrdinal"])
    previous_gap = max(1, int(last["frameOrdinal"]) - int(prev["frameOrdinal"]))
    vx = (float(last["centroidPx"][0]) - float(prev["centroidPx"][0])) / previous_gap
    vy = (float(last["centroidPx"][1]) - float(prev["centroidPx"][1])) / previous_gap
    return [float(last["centroidPx"][0]) + vx * gap, float(last["centroidPx"][1]) + vy * gap]


def pose_distance(first: dict, second: dict) -> float | None:
    if not first.get("available") or not second.get("available"):
        return None
    a = first.get("xyzCameraM") or []
    b = second.get("xyzCameraM") or []
    if len(a) != 3 or len(b) != 3:
        return None
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def rpy_delta(first: dict, second: dict) -> float | None:
    if not first.get("available") or not second.get("available"):
        return None
    a = first.get("rpyCameraDeg") or []
    b = second.get("rpyCameraDeg") or []
    if len(a) != 3 or len(b) != 3:
        return None
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def score_candidate(observation: dict, track: dict, settings: dict) -> dict:
    last = track["lastObservation"]
    frame_gap = max(1, int(observation["frameOrdinal"]) - int(last["frameOrdinal"]))
    center_distance = math.dist(observation["centroidPx"], last["centroidPx"])
    iou = rect_iou(observation["bboxPx"], last["bboxPx"])
    size_similarity = min(observation["areaPx"], last["areaPx"]) / max(observation["areaPx"], last["areaPx"], 1.0)
    layer_score = cosine_similarity(observation.get("layerMix") or {}, last.get("layerMix") or {})
    pose_m = pose_distance(observation["pose"], last["pose"])
    rpy_deg = rpy_delta(observation["pose"], last["pose"])
    predicted = predicted_center(track, int(observation["frameOrdinal"]))
    velocity_distance = math.dist(observation["centroidPx"], predicted) if predicted else None
    signals = {
        "center2d": max(0.0, 1.0 - center_distance / max(1.0, float(settings["max2dDistancePx"]))),
        "bboxIou": iou,
        "sizeSimilarity": max(0.0, min(1.0, size_similarity)),
        "layerMix": max(0.0, min(1.0, layer_score)),
        "clipReliability": max(0.0, 1.0 - float((observation.get("fovClip") or {}).get("severity") or 0.0)),
    }
    if pose_m is not None:
        signals["pose3d"] = max(0.0, 1.0 - pose_m / max(0.01, float(settings["max3dDistanceM"])))
    if rpy_deg is not None:
        signals["rpy"] = max(0.0, 1.0 - rpy_deg / max(1.0, float(settings["maxRpyDeltaDeg"])))
    if velocity_distance is not None:
        signals["velocity2d"] = max(0.0, 1.0 - velocity_distance / max(1.0, float(settings["max2dDistancePx"])))
    weighted_keys = [key for key in SIGNAL_WEIGHTS if key in signals]
    weight_total = sum(SIGNAL_WEIGHTS[key] for key in weighted_keys)
    score = sum(signals[key] * SIGNAL_WEIGHTS[key] for key in weighted_keys) / weight_total if weight_total > 0 else 0.0
    spatial_gate = center_distance <= float(settings["max2dDistancePx"]) or iou >= float(settings["minIou"]) or (pose_m is not None and pose_m <= float(settings["max3dDistanceM"]))
    return {
        "instanceId": track["instanceId"],
        "score": clean_float(score, 4),
        "accepted": bool(spatial_gate and score >= float(settings["minAssociationScore"])),
        "signals": {key: clean_float(value, 4) for key, value in signals.items()},
        "raw": {
            "centerDistancePx": clean_float(center_distance, 4),
            "iou": clean_float(iou, 6),
            "poseDistanceM": clean_float(pose_m, 6) if pose_m is not None else None,
            "rpyDeltaDeg": clean_float(rpy_deg, 4) if rpy_deg is not None else None,
            "velocityDistancePx": clean_float(velocity_distance, 4) if velocity_distance is not None else None,
            "frameGap": frame_gap,
        },
    }


def instance_color(instance_ordinal: int) -> str:
    hue = (int(instance_ordinal) * 137) % 360
    return f"hsl({hue} 88% 58%)"


def finalize_instance(instance: dict) -> dict:
    observations = instance.pop("_observations", [])
    scores = [float(item.get("associationScore") or 0.0) for item in observations if item.get("status") != "new"]
    instance["observationCount"] = len(observations)
    instance["meanAssociationScore"] = clean_float(sum(scores) / len(scores), 4) if scores else None
    instance["frameOrdinals"] = [int(item["frameOrdinal"]) for item in observations]
    return instance


def snapshot_instance(instance: dict) -> dict:
    """Return a non-mutating instance summary for live/pipeline publishing."""
    observations = list(instance.get("_observations") or [])
    scores = [float(item.get("associationScore") or 0.0) for item in observations if item.get("status") != "new"]
    output = {key: value for key, value in instance.items() if key != "_observations"}
    output["observationCount"] = len(observations)
    output["meanAssociationScore"] = clean_float(sum(scores) / len(scores), 4) if scores else None
    output["frameOrdinals"] = [int(item["frameOrdinal"]) for item in observations]
    return output


def pose_lookup_for_frame(pose_frame: dict | None) -> dict[tuple[int, str], dict]:
    if not isinstance(pose_frame, dict):
        return {}
    frame_ordinal = int(pose_frame.get("frameOrdinal") or 0)
    return {
        (frame_ordinal, str(obj.get("bboxId") or "")): obj
        for obj in pose_frame.get("objects") or []
        if str(obj.get("bboxId") or "")
    }


class InstanceTrackingRuntime:
    """Stateful frame-by-frame instance tracker used by the live pipeline."""

    def __init__(self, settings: dict | None = None):
        clean_settings = normalized_instance_tracking_settings(settings)
        if clean_settings.get("poseSource") == "squarePose":
            clean_settings["poseSource"] = "poseFit"
        self.settings = clean_settings
        self.active_tracks: list[dict] = []
        self.all_instances: list[dict] = []
        self.instance_counter = 0
        self.frames_out: list[dict] = []
        self.matched_count = 0
        self.new_count = 0
        self.split_count = 0
        self.merge_count = 0
        self.association_scores: list[float] = []

    def process_frame(self, frame: dict, pose_frame: dict | None = None) -> dict:
        frame_ordinal = int(frame.get("frameOrdinal") or 0)
        max_gap = int(self.settings["maxFrameGap"])
        expired = [track for track in self.active_tracks if frame_ordinal - int(track["lastFrame"]) > max_gap]
        for track in expired:
            track["status"] = "ended"
        self.active_tracks = [track for track in self.active_tracks if frame_ordinal - int(track["lastFrame"]) <= max_gap]
        pose_by_key = pose_lookup_for_frame(pose_frame) if self.settings["poseSource"] == "poseFit" else {}
        observations = [
            build_observation(frame_ordinal, index, bbox, pose_by_key, self.settings["poseSource"])
            for index, bbox in enumerate(frame.get("bboxes") or [])
        ]
        candidate_pairs = []
        candidate_lists: dict[str, list[dict]] = {}
        candidate_limit = int(self.settings["debugCandidateLimit"])
        for observation in observations:
            candidates = [score_candidate(observation, track, self.settings) for track in self.active_tracks]
            candidates.sort(key=lambda item: (-float(item["score"]), item["instanceId"]))
            candidate_lists[observation["observationId"]] = candidates[:candidate_limit]
            for candidate in candidates:
                if candidate["accepted"]:
                    candidate_pairs.append((float(candidate["score"]), observation["observationId"], candidate["instanceId"], candidate))
        candidate_pairs.sort(key=lambda item: (-item[0], item[2], item[1]))
        track_by_id = {item["instanceId"]: item for item in self.active_tracks}
        assigned_observations: set[str] = set()
        assigned_tracks: set[str] = set()
        assignments: dict[str, tuple[dict, dict]] = {}
        for _, observation_id, instance_id, candidate in candidate_pairs:
            if observation_id in assigned_observations or instance_id in assigned_tracks:
                continue
            assignments[observation_id] = (track_by_id[instance_id], candidate)
            assigned_observations.add(observation_id)
            assigned_tracks.add(instance_id)

        strong_candidate_usage: dict[str, list[str]] = {}
        for observation in observations:
            for candidate in candidate_lists.get(observation["observationId"], []):
                if float(candidate["score"]) >= float(self.settings["splitCandidateScore"]):
                    strong_candidate_usage.setdefault(candidate["instanceId"], []).append(observation["observationId"])

        frame_observations = []
        for observation in observations:
            top_candidates = candidate_lists.get(observation["observationId"], [])
            merge_candidates = [candidate["instanceId"] for candidate in top_candidates if float(candidate["score"]) >= float(self.settings["splitCandidateScore"])]
            if observation["observationId"] in assignments:
                track, candidate = assignments[observation["observationId"]]
                self.matched_count += 1
                association_score = float(candidate["score"])
                observation["instanceId"] = track["instanceId"]
                observation["instanceOrdinal"] = track["ordinal"]
                observation["instanceColor"] = track["color"]
                observation["associationScore"] = clean_float(association_score, 4)
                observation["status"] = "matched"
                observation["candidates"] = top_candidates
                if observation.get("pose", {}).get("available"):
                    track["lastValidPose"] = observation["pose"]
                    observation["lastValidPose"] = observation["pose"]
                else:
                    observation["lastValidPose"] = track.get("lastValidPose")
                if len(merge_candidates) > 1:
                    self.merge_count += 1
                    observation["mergeCandidateFrom"] = merge_candidates
                    track["events"].append({"type": "merge-candidate", "frameOrdinal": frame_ordinal, "sourceInstanceIds": merge_candidates})
                if observation.get("pose", {}).get("ignored"):
                    track["events"].append({"type": "pose-ignored", "frameOrdinal": frame_ordinal, "bboxId": observation["bboxId"], "reasonCodes": observation["pose"].get("reasonCodes") or []})
                track["previousObservation"] = track.get("lastObservation")
                track["lastObservation"] = {key: observation[key] for key in ("frameOrdinal", "bboxId", "bboxPx", "centroidPx", "areaPx", "layerMix", "pose")}
                track["lastFrame"] = frame_ordinal
                track["_observations"].append(observation)
                self.association_scores.append(association_score)
            else:
                parent_candidates = [
                    candidate
                    for candidate in top_candidates
                    if float(candidate["score"]) >= float(self.settings["splitCandidateScore"]) and len(strong_candidate_usage.get(candidate["instanceId"], [])) > 1
                ]
                self.instance_counter += 1
                instance_id = f"instance-{self.instance_counter:06d}"
                color = instance_color(self.instance_counter)
                status = "split-candidate" if parent_candidates else "new"
                self.split_count += 1 if status == "split-candidate" else 0
                self.new_count += 1 if status == "new" else 0
                observation["instanceId"] = instance_id
                observation["instanceOrdinal"] = self.instance_counter
                observation["instanceColor"] = color
                observation["associationScore"] = 0.0
                observation["status"] = status
                observation["candidates"] = top_candidates
                observation["lastValidPose"] = observation["pose"] if observation.get("pose", {}).get("available") else None
                events = []
                if observation.get("pose", {}).get("ignored"):
                    events.append({"type": "pose-ignored", "frameOrdinal": frame_ordinal, "bboxId": observation["bboxId"], "reasonCodes": observation["pose"].get("reasonCodes") or []})
                if parent_candidates:
                    parent_id = parent_candidates[0]["instanceId"]
                    observation["createdFrom"] = {"type": "split-candidate", "parentInstanceId": parent_id, "frameOrdinal": frame_ordinal}
                    events.append({"type": "split-candidate", "frameOrdinal": frame_ordinal, "parentInstanceId": parent_id})
                    parent = track_by_id.get(parent_id)
                    if parent:
                        parent["events"].append({"type": "split-candidate", "frameOrdinal": frame_ordinal, "childInstanceIds": [instance_id]})
                track = {
                    "instanceId": instance_id,
                    "ordinal": self.instance_counter,
                    "color": color,
                    "createdFrame": frame_ordinal,
                    "lastFrame": frame_ordinal,
                    "status": "active",
                    "events": events,
                    "previousObservation": None,
                    "lastObservation": {key: observation[key] for key in ("frameOrdinal", "bboxId", "bboxPx", "centroidPx", "areaPx", "layerMix", "pose")},
                    "lastValidPose": observation["lastValidPose"],
                    "_observations": [observation],
                }
                self.active_tracks.append(track)
                self.all_instances.append(track)
            frame_observations.append(dict(observation))

        frame_out = {
            "frameOrdinal": frame_ordinal,
            "frameId": frame.get("frameId"),
            "path": frame.get("path"),
            "observationCount": len(frame_observations),
            "observations": frame_observations,
        }
        self.frames_out.append(frame_out)
        return frame_out

    def summary(self, pose_signals_available: bool) -> dict:
        observation_count = sum(int(frame.get("observationCount") or 0) for frame in self.frames_out)
        return {
            "instanceCount": len(self.all_instances),
            "observationCount": observation_count,
            "matchedCount": self.matched_count,
            "newCount": self.new_count,
            "splitCandidateCount": self.split_count,
            "mergeCandidateCount": self.merge_count,
            "meanAssociationScore": clean_float(sum(self.association_scores) / len(self.association_scores), 4) if self.association_scores else 0.0,
            "poseSignalsAvailable": bool(pose_signals_available),
            "poseSource": self.settings["poseSource"],
        }

    def instances(self) -> list[dict]:
        return [snapshot_instance(track) for track in self.all_instances]


def build_instance_tracking(args: argparse.Namespace, progress_callback: Callable[[dict], None] | None = None) -> dict:
    settings = normalized_instance_tracking_settings(getattr(args, "instance_settings", None))
    manifest_path = args.precompute_manifest or selected_precompute_manifest()
    manifest = read_json(manifest_path)
    run_key = str(manifest.get("runKey") or "")
    width = int(manifest.get("width") or 640)
    height = int(manifest.get("height") or 360)
    frame_count = int(manifest.get("frameCount") or len(manifest.get("frames") or []))

    bbox_manifest_path = maskbits_bbox_run_root(run_key, getattr(args, "bbox_output_root", DEFAULT_MASKBITS_BBOX_ROOT)) / "bbox_manifest.json"
    bbox_manifest = read_json(bbox_manifest_path)
    if str(bbox_manifest.get("runKey") or "") != run_key:
        raise ValueError("maskbits bbox manifest run key does not match active run")
    if not bbox_manifest.get("complete"):
        raise ValueError("instance tracking requires a complete maskbits bbox manifest")
    bbox_signature = str(bbox_manifest.get("sourceSignature") or "")

    pose_manifest_path = pose_run_root(run_key, getattr(args, "pose_output_root", DEFAULT_POSE_ROOT)) / "3d_pose_fit.json"
    pose_manifest = read_json_optional(pose_manifest_path)
    if pose_manifest and (str(pose_manifest.get("runKey") or "") != run_key or not pose_manifest.get("complete")):
        pose_manifest = None
    pose_signature = str(pose_manifest.get("sourceSignature") or "") if pose_manifest else None
    active_pose_manifest = pose_manifest if settings["poseSource"] == "poseFit" else None
    active_pose_signature = pose_signature if settings["poseSource"] == "poseFit" else None
    source_signature = instance_tracking_source_signature(manifest, manifest_path, settings, bbox_signature, active_pose_signature)
    pose_by_key = pose_lookup(active_pose_manifest)

    start = max(0, int(args.start_frame or 0))
    end = frame_count
    if args.max_frames is not None:
        end = min(end, start + max(0, int(args.max_frames)))
    selected_frames = [frame for frame in list(bbox_manifest.get("frames") or []) if start <= int(frame.get("frameOrdinal") or 0) < end]
    if not selected_frames:
        raise ValueError("selected instance tracking frame range is empty")

    active_tracks: list[dict] = []
    all_instances: list[dict] = []
    frames_out = []
    instance_counter = 0
    matched_count = 0
    new_count = 0
    split_count = 0
    merge_count = 0
    association_scores: list[float] = []
    max_gap = int(settings["maxFrameGap"])
    candidate_limit = int(settings["debugCandidateLimit"])

    for local_index, frame in enumerate(selected_frames):
        frame_ordinal = int(frame.get("frameOrdinal") or 0)
        active_tracks = [track for track in active_tracks if frame_ordinal - int(track["lastFrame"]) <= max_gap]
        observations = [build_observation(frame_ordinal, index, bbox, pose_by_key, settings["poseSource"]) for index, bbox in enumerate(frame.get("bboxes") or [])]
        candidate_pairs = []
        candidate_lists: dict[str, list[dict]] = {}
        for observation in observations:
            candidates = [score_candidate(observation, track, settings) for track in active_tracks]
            candidates.sort(key=lambda item: (-float(item["score"]), item["instanceId"]))
            candidate_lists[observation["observationId"]] = candidates[:candidate_limit]
            for candidate in candidates:
                if candidate["accepted"]:
                    candidate_pairs.append((float(candidate["score"]), observation["observationId"], candidate["instanceId"], candidate))
        candidate_pairs.sort(key=lambda item: (-item[0], item[2], item[1]))
        track_by_id = {item["instanceId"]: item for item in active_tracks}
        assigned_observations: set[str] = set()
        assigned_tracks: set[str] = set()
        assignments: dict[str, tuple[dict, dict]] = {}
        for _, observation_id, instance_id, candidate in candidate_pairs:
            if observation_id in assigned_observations or instance_id in assigned_tracks:
                continue
            assignments[observation_id] = (track_by_id[instance_id], candidate)
            assigned_observations.add(observation_id)
            assigned_tracks.add(instance_id)

        frame_observations = []
        strong_candidate_usage: dict[str, list[str]] = {}
        for observation in observations:
            for candidate in candidate_lists.get(observation["observationId"], []):
                if float(candidate["score"]) >= float(settings["splitCandidateScore"]):
                    strong_candidate_usage.setdefault(candidate["instanceId"], []).append(observation["observationId"])

        for observation in observations:
            top_candidates = candidate_lists.get(observation["observationId"], [])
            merge_candidates = [candidate["instanceId"] for candidate in top_candidates if float(candidate["score"]) >= float(settings["splitCandidateScore"])]
            if observation["observationId"] in assignments:
                track, candidate = assignments[observation["observationId"]]
                matched_count += 1
                association_score = float(candidate["score"])
                observation["instanceId"] = track["instanceId"]
                observation["instanceOrdinal"] = track["ordinal"]
                observation["instanceColor"] = track["color"]
                observation["associationScore"] = clean_float(association_score, 4)
                observation["status"] = "matched"
                observation["candidates"] = top_candidates
                if observation.get("pose", {}).get("available"):
                    track["lastValidPose"] = observation["pose"]
                    observation["lastValidPose"] = observation["pose"]
                else:
                    observation["lastValidPose"] = track.get("lastValidPose")
                if len(merge_candidates) > 1:
                    merge_count += 1
                    observation["mergeCandidateFrom"] = merge_candidates
                    track["events"].append({"type": "merge-candidate", "frameOrdinal": frame_ordinal, "sourceInstanceIds": merge_candidates})
                if observation.get("pose", {}).get("ignored"):
                    track["events"].append({"type": "pose-ignored", "frameOrdinal": frame_ordinal, "bboxId": observation["bboxId"], "reasonCodes": observation["pose"].get("reasonCodes") or []})
                track["previousObservation"] = track.get("lastObservation")
                track["lastObservation"] = {key: observation[key] for key in ("frameOrdinal", "bboxId", "bboxPx", "centroidPx", "areaPx", "layerMix", "pose")}
                track["lastFrame"] = frame_ordinal
                track["_observations"].append(observation)
                association_scores.append(association_score)
            else:
                parent_candidates = [
                    candidate
                    for candidate in top_candidates
                    if float(candidate["score"]) >= float(settings["splitCandidateScore"]) and len(strong_candidate_usage.get(candidate["instanceId"], [])) > 1
                ]
                instance_counter += 1
                instance_id = f"instance-{instance_counter:06d}"
                color = instance_color(instance_counter)
                status = "split-candidate" if parent_candidates else "new"
                split_count += 1 if status == "split-candidate" else 0
                new_count += 1 if status == "new" else 0
                observation["instanceId"] = instance_id
                observation["instanceOrdinal"] = instance_counter
                observation["instanceColor"] = color
                observation["associationScore"] = 0.0
                observation["status"] = status
                observation["candidates"] = top_candidates
                observation["lastValidPose"] = observation["pose"] if observation.get("pose", {}).get("available") else None
                events = []
                if observation.get("pose", {}).get("ignored"):
                    events.append({"type": "pose-ignored", "frameOrdinal": frame_ordinal, "bboxId": observation["bboxId"], "reasonCodes": observation["pose"].get("reasonCodes") or []})
                if parent_candidates:
                    parent_id = parent_candidates[0]["instanceId"]
                    observation["createdFrom"] = {"type": "split-candidate", "parentInstanceId": parent_id, "frameOrdinal": frame_ordinal}
                    events.append({"type": "split-candidate", "frameOrdinal": frame_ordinal, "parentInstanceId": parent_id})
                    parent = track_by_id.get(parent_id)
                    if parent:
                        parent["events"].append({"type": "split-candidate", "frameOrdinal": frame_ordinal, "childInstanceIds": [instance_id]})
                track = {
                    "instanceId": instance_id,
                    "ordinal": instance_counter,
                    "color": color,
                    "createdFrame": frame_ordinal,
                    "lastFrame": frame_ordinal,
                    "status": "active",
                    "events": events,
                    "previousObservation": None,
                    "lastObservation": {key: observation[key] for key in ("frameOrdinal", "bboxId", "bboxPx", "centroidPx", "areaPx", "layerMix", "pose")},
                    "lastValidPose": observation["lastValidPose"],
                    "_observations": [observation],
                }
                active_tracks.append(track)
                all_instances.append(track)
            frame_observations.append(dict(observation))

        frames_out.append(
            {
                "frameOrdinal": frame_ordinal,
                "frameId": frame.get("frameId"),
                "path": frame.get("path"),
                "observationCount": len(frame_observations),
                "observations": frame_observations,
            }
        )
        if progress_callback:
            progress_callback({"phase": "building-instance-tracking", "index": local_index + 1, "total": len(selected_frames), "frameOrdinal": frame_ordinal, "observationCount": len(frame_observations), "activeInstanceCount": len(active_tracks)})

    for track in active_tracks:
        if int(selected_frames[-1].get("frameOrdinal") or 0) - int(track["lastFrame"]) > max_gap:
            track["status"] = "ended"

    instances_out = [finalize_instance(track) for track in all_instances]
    output_root = instance_mapping_run_root(run_key, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "instanceCount": len(instances_out),
        "observationCount": sum(int(frame.get("observationCount") or 0) for frame in frames_out),
        "matchedCount": matched_count,
        "newCount": new_count,
        "splitCandidateCount": split_count,
        "mergeCandidateCount": merge_count,
        "meanAssociationScore": clean_float(sum(association_scores) / len(association_scores), 4) if association_scores else 0.0,
        "poseSignalsAvailable": bool(active_pose_manifest),
        "poseSource": settings["poseSource"],
    }
    output = {
        "version": 1,
        "app": "vision_passthrough_review",
        "kind": "instance-mapping-pose-estimation-v1",
        "createdAt": utc_now(),
        "runKey": run_key,
        "precomputeManifest": repo_url(manifest_path),
        "sourceSignature": source_signature,
        "complete": start == 0 and len(frames_out) >= frame_count,
        "frameStart": start,
        "frameCount": len(frames_out),
        "sourceFrameCount": frame_count,
        "image": {"width": width, "height": height},
        "settings": settings,
        "input": {
            "bboxManifest": repo_url(bbox_manifest_path),
            "bboxSourceSignature": bbox_signature,
            "poseManifest": repo_url(pose_manifest_path) if pose_manifest else None,
            "poseSourceSignature": pose_signature,
            "activePoseSource": settings["poseSource"],
            "activePoseSourceSignature": active_pose_signature,
            "poseSignalsAvailable": bool(active_pose_manifest),
        },
        "summary": summary,
        "instances": instances_out,
        "frames": frames_out,
    }
    atomic_write_json(output_root / "instance_mapping.json", output)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="print the instance-tracking contract as JSON")
    parser.add_argument("--precompute-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bbox-output-root", type=Path, default=DEFAULT_MASKBITS_BBOX_ROOT)
    parser.add_argument("--pose-output-root", type=Path, default=DEFAULT_POSE_ROOT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--pose-source", choices=sorted(POSE_SOURCES), default=DEFAULT_POSE_SOURCE)
    parser.add_argument("--max-frame-gap", type=int, default=DEFAULT_MAX_FRAME_GAP)
    parser.add_argument("--min-association-score", type=float, default=DEFAULT_MIN_ASSOCIATION_SCORE)
    parser.add_argument("--max-2d-distance-px", type=float, default=DEFAULT_MAX_2D_DISTANCE_PX)
    parser.add_argument("--min-iou", type=float, default=DEFAULT_MIN_IOU)
    parser.add_argument("--max-3d-distance-m", type=float, default=DEFAULT_MAX_3D_DISTANCE_M)
    parser.add_argument("--max-rpy-delta-deg", type=float, default=DEFAULT_MAX_RPY_DELTA_DEG)
    parser.add_argument("--split-candidate-score", type=float, default=DEFAULT_SPLIT_CANDIDATE_SCORE)
    parser.add_argument("--debug-candidate-limit", type=int, default=DEFAULT_DEBUG_CANDIDATE_LIMIT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.describe:
        print(json.dumps(CONTRACT, indent=2))
        return
    args.instance_settings = {
        "maxFrameGap": args.max_frame_gap,
        "minAssociationScore": args.min_association_score,
        "max2dDistancePx": args.max_2d_distance_px,
        "minIou": args.min_iou,
        "max3dDistanceM": args.max_3d_distance_m,
        "maxRpyDeltaDeg": args.max_rpy_delta_deg,
        "splitCandidateScore": args.split_candidate_score,
        "debugCandidateLimit": args.debug_candidate_limit,
        "poseSource": args.pose_source,
    }
    result = build_instance_tracking(args)
    print(
        json.dumps(
            {
                "ok": True,
                "manifest": repo_url(instance_mapping_run_root(result["runKey"], args.output_root) / "instance_mapping.json"),
                "settings": result.get("settings"),
                "summary": result.get("summary"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
