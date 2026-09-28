"""Global gate mapping and smoothing from tracked inner-void pose samples."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover
    from .config import ProjectionConfig
    from .schema import FrameMeta, VisionGateResult
    from .solve_pnp import SOURCE_WEIGHTS
    from .tracking import clamp01, clean_float, distance3, tuple3
except ImportError:  # pragma: no cover
    from config import ProjectionConfig
    from schema import FrameMeta, VisionGateResult
    from solve_pnp import SOURCE_WEIGHTS
    from tracking import clamp01, clean_float, distance3, tuple3


@dataclass
class GateTrack:
    gate_id: str
    last_frame_ordinal: int
    position: list[float]
    raw_position: list[float]
    orientation: list[float]
    position_confidence: float
    orientation_confidence: float
    contributors: set[str] = field(default_factory=set)
    observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PoseSample:
    sample_id: str
    frame_ordinal: int
    frame_id: str
    profile_id: str
    profile_label: str
    local_instance_id: str
    observation_id: str
    void_id: str
    source: str
    xyz: list[float]
    rpy: list[float]
    depth: float
    reprojection_error_px: float
    instance_score: float
    source_quality: float
    sample_score: float
    position_confidence: float
    orientation_confidence: float
    source_candidates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def observation_key(self) -> tuple[int, str, str, str]:
        return (self.frame_ordinal, self.profile_id, self.local_instance_id, self.observation_id)

    @property
    def local_key(self) -> tuple[str, str]:
        return (self.profile_id, self.local_instance_id)

    def to_review_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "frame_ordinal": self.frame_ordinal,
            "frame_id": self.frame_id,
            "profile_id": self.profile_id,
            "profile_label": self.profile_label,
            "local_instance_id": self.local_instance_id,
            "observation_id": self.observation_id,
            "void_id": self.void_id,
            "source": self.source,
            "xyzCameraM": [clean_float(item, 6) for item in self.xyz],
            "rpyCameraDeg": [clean_float(item, 6) for item in self.rpy],
            "depthM": clean_float(self.depth, 6),
            "reprojectionErrorPx": clean_float(self.reprojection_error_px, 6),
            "instanceScore": clean_float(self.instance_score, 6),
            "sourceQuality": clean_float(self.source_quality, 6),
            "sampleScore": clean_float(self.sample_score, 6),
            "positionConfidence": clean_float(self.position_confidence, 6),
            "orientationConfidence": clean_float(self.orientation_confidence, 6),
            "sourceCandidates": list(self.source_candidates),
        }


@dataclass
class FrameAggregate:
    frame_ordinal: int
    frame_id: str
    samples: list[PoseSample] = field(default_factory=list)

    @property
    def xyz(self) -> list[float]:
        return weighted_xyz(self.samples)

    @property
    def rpy(self) -> list[float]:
        return best_orientation_sample(self.samples).rpy

    @property
    def depth(self) -> float:
        return weighted_value([sample.depth for sample in self.samples], [sample.sample_score for sample in self.samples])

    @property
    def position_confidence(self) -> float:
        return weighted_value([sample.position_confidence for sample in self.samples], [sample.sample_score for sample in self.samples])

    @property
    def orientation_confidence(self) -> float:
        return best_orientation_sample(self.samples).orientation_confidence


@dataclass
class GlobalTrack:
    internal_id: str
    frames: dict[int, FrameAggregate] = field(default_factory=dict)
    contributors: set[tuple[str, str]] = field(default_factory=set)
    decisions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return min(self.frames) if self.frames else 0

    @property
    def last_frame(self) -> int:
        return max(self.frames) if self.frames else 0

    @property
    def first_aggregate(self) -> FrameAggregate | None:
        return self.frames[self.first_frame] if self.frames else None

    @property
    def last_aggregate(self) -> FrameAggregate | None:
        return self.frames[self.last_frame] if self.frames else None

    def add_sample(self, sample: PoseSample, decision: dict[str, Any]) -> None:
        aggregate = self.frames.setdefault(sample.frame_ordinal, FrameAggregate(sample.frame_ordinal, sample.frame_id))
        aggregate.samples.append(sample)
        self.contributors.add(sample.local_key)
        self.decisions.append(decision)


def _float(settings: dict[str, Any], key: str, fallback: float) -> float:
    try:
        value = float(settings.get(key, fallback))
    except (TypeError, ValueError):
        return float(fallback)
    return value if math.isfinite(value) else float(fallback)


def _int(settings: dict[str, Any], key: str, fallback: int) -> int:
    try:
        return int(float(settings.get(key, fallback)))
    except (TypeError, ValueError):
        return int(fallback)


def plane_normal_from_rpy(rpy_deg: list[float]) -> list[float]:
    roll = math.radians(float(rpy_deg[0]))
    pitch = math.radians(float(rpy_deg[1]))
    yaw = math.radians(float(rpy_deg[2]))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    normal = [cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr]
    length = math.sqrt(sum(item * item for item in normal))
    return [0.0, 0.0, 1.0] if length <= 1e-9 else [item / length for item in normal]


def plane_angle_delta_deg(left_rpy: list[float], right_rpy: list[float]) -> float:
    left = plane_normal_from_rpy(left_rpy)
    right = plane_normal_from_rpy(right_rpy)
    dot = abs(sum(left[index] * right[index] for index in range(3)))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def opencv_camera_to_result_camera(value: list[float]) -> tuple[float, float, float]:
    return (clean_float(float(value[0]), 6), clean_float(-float(value[1]), 6), clean_float(float(value[2]), 6))


def fit_quality_score(value: Any) -> float:
    if isinstance(value, dict):
        return clamp01(value.get("overall", value.get("score", 0.0)))
    return clamp01(value)


def weighted_value(values: list[float], weights: list[float]) -> float:
    total_weight = sum(max(0.0, float(weight)) for weight in weights)
    if total_weight <= 1e-9:
        return clean_float(sum(values) / max(1, len(values)), 6)
    total = sum(float(value) * max(0.0, float(weight)) for value, weight in zip(values, weights))
    return clean_float(total / total_weight, 6)


def weighted_xyz(samples: list[PoseSample]) -> list[float]:
    weights = [sample.sample_score for sample in samples]
    return [weighted_value([sample.xyz[index] for sample in samples], weights) for index in range(3)]


def best_orientation_sample(samples: list[PoseSample]) -> PoseSample:
    return max(samples, key=lambda sample: (sample.orientation_confidence, sample.sample_score))


def batch_sample_score(source: str, observation: dict[str, Any], solve: dict[str, Any], settings: dict[str, Any]) -> tuple[float, dict[str, float]]:
    instance_score = clamp01(
        observation.get(
            "finalInstanceScore",
            observation.get("total_confidence", observation.get("instance_confidence", observation.get("association_score", 0.0))),
        )
    )
    source_quality = max(
        clamp01(solve.get("source_score")),
        clamp01(solve.get("fit_score", solve.get("fitScore"))),
        fit_quality_score(solve.get("fitQuality")),
    )
    try:
        reprojection_error = max(0.0, float(solve.get("reprojectionErrorPx") or 0.0))
    except (TypeError, ValueError):
        reprojection_error = _float(settings, "maxReprojectionErrorPx", 20.0)
    reprojection_quality = clamp01(1.0 - reprojection_error / max(1e-6, _float(settings, "maxReprojectionErrorPx", 20.0)))
    source_weight_quality = clamp01(SOURCE_WEIGHTS.get(source, 0.0) / max(SOURCE_WEIGHTS.values()))
    score = clamp01(0.45 * instance_score + 0.25 * source_quality + 0.20 * reprojection_quality + 0.10 * source_weight_quality)
    return clean_float(score, 6), {
        "instanceScore": clean_float(instance_score, 6),
        "sourceQuality": clean_float(source_quality, 6),
        "reprojectionQuality": clean_float(reprojection_quality, 6),
        "sourceWeightQuality": clean_float(source_weight_quality, 6),
    }


def _reject_sample(
    rejected: list[dict[str, Any]],
    *,
    reason: str,
    profile_id: str,
    instance_id: str,
    observation: dict[str, Any],
    solve: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "reason": reason,
        "profile_id": profile_id,
        "local_instance_id": instance_id,
        "frame_ordinal": int(observation.get("frame_ordinal") or 0),
        "observation_id": observation.get("observation_id"),
        "void_id": observation.get("void_id") or ((observation.get("void") or {}).get("void_id") if isinstance(observation.get("void"), dict) else None),
        "source": solve.get("source") if isinstance(solve, dict) else None,
    }
    payload.update(extra or {})
    rejected.append(payload)


def choose_best_samples(samples: list[PoseSample]) -> list[PoseSample]:
    grouped: dict[tuple[int, str, str, str], list[PoseSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.observation_key, []).append(sample)
    chosen: list[PoseSample] = []
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item.sample_score, SOURCE_WEIGHTS.get(item.source, 0.0)), reverse=True)
        best = ordered[0]
        best.source_candidates = [
            {
                "source": item.source,
                "sampleScore": clean_float(item.sample_score, 6),
                "reprojectionErrorPx": clean_float(item.reprojection_error_px, 6),
            }
            for item in ordered
        ]
        chosen.append(best)
    return sorted(chosen, key=lambda item: (item.frame_ordinal, -item.sample_score, item.profile_id, item.local_instance_id))


def extract_pose_samples(tracks_payload: dict[str, Any], settings: dict[str, Any]) -> tuple[list[PoseSample], list[dict[str, Any]]]:
    profile_id = str(tracks_payload.get("profile_id") or "")
    profile_label = str(tracks_payload.get("profile_label") or profile_id)
    min_score = _float(settings, "minSampleScore", 0.45)
    samples: list[PoseSample] = []
    rejected: list[dict[str, Any]] = []
    for instance in tracks_payload.get("instances") if isinstance(tracks_payload.get("instances"), list) else []:
        if not isinstance(instance, dict):
            continue
        instance_id = str(instance.get("instance_id") or "")
        observations = instance.get("observations") if isinstance(instance.get("observations"), list) else []
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            objects = observation.get("objects") if isinstance(observation.get("objects"), list) else []
            if not objects:
                _reject_sample(rejected, reason="no-solvepnp-objects", profile_id=profile_id, instance_id=instance_id, observation=observation)
                continue
            for solve in objects:
                if not isinstance(solve, dict):
                    continue
                source = str(solve.get("source") or "")
                if source not in SOURCE_WEIGHTS:
                    _reject_sample(rejected, reason="unsupported-source", profile_id=profile_id, instance_id=instance_id, observation=observation, solve=solve)
                    continue
                if not solve.get("available"):
                    _reject_sample(rejected, reason="unavailable-pose", profile_id=profile_id, instance_id=instance_id, observation=observation, solve=solve)
                    continue
                xyz = tuple3(solve.get("xyzCameraM"))
                rpy = tuple3(solve.get("rpyCameraDeg"))
                if xyz is None or rpy is None:
                    _reject_sample(rejected, reason="missing-pose", profile_id=profile_id, instance_id=instance_id, observation=observation, solve=solve)
                    continue
                try:
                    depth = float(solve.get("depthM", xyz[2]))
                    reprojection_error = float(solve.get("reprojectionErrorPx") or 0.0)
                except (TypeError, ValueError):
                    _reject_sample(rejected, reason="invalid-pose-quality", profile_id=profile_id, instance_id=instance_id, observation=observation, solve=solve)
                    continue
                if not math.isfinite(depth) or depth <= 0.0:
                    _reject_sample(rejected, reason="invalid-depth", profile_id=profile_id, instance_id=instance_id, observation=observation, solve=solve)
                    continue
                if not math.isfinite(reprojection_error) or reprojection_error > _float(settings, "maxReprojectionErrorPx", 20.0):
                    _reject_sample(rejected, reason="reprojection-too-large", profile_id=profile_id, instance_id=instance_id, observation=observation, solve=solve)
                    continue
                score, signals = batch_sample_score(source, observation, solve, settings)
                if score < min_score:
                    _reject_sample(
                        rejected,
                        reason="sample-score-too-low",
                        profile_id=profile_id,
                        instance_id=instance_id,
                        observation=observation,
                        solve=solve,
                        extra={"sampleScore": score, "minSampleScore": clean_float(min_score, 6), "signals": signals},
                    )
                    continue
                frame_ordinal = int(observation.get("frame_ordinal") or 0)
                observation_id = str(observation.get("observation_id") or "")
                void_id = str(observation.get("void_id") or ((observation.get("void") or {}).get("void_id") if isinstance(observation.get("void"), dict) else ""))
                sample_id = ":".join([profile_id, instance_id, str(frame_ordinal), observation_id, source])
                samples.append(
                    PoseSample(
                        sample_id=sample_id,
                        frame_ordinal=frame_ordinal,
                        frame_id=str(observation.get("frame_id") or f"frame_{frame_ordinal:06d}"),
                        profile_id=profile_id,
                        profile_label=profile_label,
                        local_instance_id=instance_id,
                        observation_id=observation_id,
                        void_id=void_id,
                        source=source,
                        xyz=xyz,
                        rpy=rpy,
                        depth=depth,
                        reprojection_error_px=max(0.0, reprojection_error),
                        instance_score=signals["instanceScore"],
                        source_quality=signals["sourceQuality"],
                        sample_score=score,
                        position_confidence=clamp01(0.65 * score + 0.35 * signals["instanceScore"]),
                        orientation_confidence=clamp01(0.55 * score + 0.45 * signals["reprojectionQuality"]),
                    )
                )
    return choose_best_samples(samples), rejected


def batch_association_score(sample: PoseSample, aggregate: FrameAggregate, track: GlobalTrack, settings: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    max_gap = max(0, _int(settings, "maxFrameGap", 8))
    gap = max(0, int(sample.frame_ordinal) - int(aggregate.frame_ordinal))
    if gap > max_gap:
        return 0.0, {"reason": "frame-gap-too-large", "gap": gap, "maxFrameGap": max_gap}
    max_distance = max(1e-6, _float(settings, "maxDistanceM", 2.0))
    max_depth = max(1e-6, _float(settings, "maxDepthDeltaM", 4.0))
    max_angle = max(1e-6, _float(settings, "maxPlaneAngleDeg", 45.0))
    distance = distance3(sample.xyz, aggregate.xyz)
    depth_delta = abs(float(sample.depth) - float(aggregate.depth))
    angle_delta = plane_angle_delta_deg(sample.rpy, aggregate.rpy)
    if distance is None or distance > max_distance:
        return 0.0, {"reason": "distance-too-large", "distanceM": None if distance is None else clean_float(distance, 6), "maxDistanceM": max_distance}
    if depth_delta > max_depth:
        return 0.0, {"reason": "depth-delta-too-large", "depthDeltaM": clean_float(depth_delta, 6), "maxDepthDeltaM": max_depth}
    if angle_delta > max_angle:
        return 0.0, {"reason": "plane-angle-too-large", "planeAngleDeg": clean_float(angle_delta, 6), "maxPlaneAngleDeg": max_angle}
    distance_score = clamp01(1.0 - distance / max_distance)
    depth_score = clamp01(1.0 - depth_delta / max_depth)
    angle_score = clamp01(1.0 - angle_delta / max_angle)
    gap_score = 1.0 if max_gap <= 0 else clamp01(1.0 - gap / (max_gap + 1.0))
    continuity_bonus = 0.08 if sample.local_key in track.contributors else 0.0
    score = clamp01(
        0.35 * distance_score
        + 0.20 * depth_score
        + 0.25 * angle_score
        + 0.10 * gap_score
        + 0.10 * sample.sample_score
        + continuity_bonus
    )
    return clean_float(score, 6), {
        "reason": "accepted",
        "distanceM": clean_float(distance, 6),
        "depthDeltaM": clean_float(depth_delta, 6),
        "planeAngleDeg": clean_float(angle_delta, 6),
        "gap": gap,
        "distanceScore": clean_float(distance_score, 6),
        "depthScore": clean_float(depth_score, 6),
        "angleScore": clean_float(angle_score, 6),
        "gapScore": clean_float(gap_score, 6),
        "sampleScore": clean_float(sample.sample_score, 6),
        "continuityBonus": clean_float(continuity_bonus, 6),
        "associationScore": clean_float(score, 6),
    }


def build_global_tracks(samples: list[PoseSample], settings: dict[str, Any]) -> list[GlobalTrack]:
    min_association = _float(settings, "minAssociationScore", 0.55)
    tracks: list[GlobalTrack] = []
    next_id = 1
    for sample in samples:
        best_track: GlobalTrack | None = None
        best_score = 0.0
        best_breakdown: dict[str, Any] = {}
        for track in tracks:
            aggregate = track.frames.get(sample.frame_ordinal) or track.last_aggregate
            if aggregate is None:
                continue
            score, breakdown = batch_association_score(sample, aggregate, track, settings)
            if score > best_score:
                best_track = track
                best_score = score
                best_breakdown = breakdown
        if best_track is None or best_score < min_association:
            track = GlobalTrack(internal_id=f"gate-{next_id:03d}")
            next_id += 1
            track.add_sample(
                sample,
                {
                    "type": "new-global-gate",
                    "sample_id": sample.sample_id,
                    "frame_ordinal": sample.frame_ordinal,
                    "bestAssociationScore": clean_float(best_score, 6),
                    "minAssociationScore": clean_float(min_association, 6),
                    "bestRejectedBreakdown": best_breakdown,
                },
            )
            tracks.append(track)
            continue
        best_track.add_sample(
            sample,
            {
                "type": "associated",
                "sample_id": sample.sample_id,
                "frame_ordinal": sample.frame_ordinal,
                "associationScore": clean_float(best_score, 6),
                "breakdown": best_breakdown,
            },
        )
    return merge_handoffs(tracks, settings)


def merge_handoffs(tracks: list[GlobalTrack], settings: dict[str, Any]) -> list[GlobalTrack]:
    min_association = _float(settings, "minAssociationScore", 0.55)
    changed = True
    while changed:
        changed = False
        ordered = sorted(tracks, key=lambda track: (track.first_frame, track.internal_id))
        for left in ordered:
            if changed:
                break
            left_last = left.last_aggregate
            if left_last is None:
                continue
            for right in ordered:
                if left is right or right not in tracks:
                    continue
                right_first = right.first_aggregate
                if right_first is None:
                    continue
                gap = right.first_frame - left.last_frame
                if gap <= 0 or gap > _int(settings, "maxFrameGap", 8):
                    continue
                probe = best_orientation_sample(right_first.samples)
                score, breakdown = batch_association_score(probe, left_last, left, settings)
                if score < min_association:
                    continue
                for frame_ordinal, aggregate in right.frames.items():
                    target = left.frames.setdefault(frame_ordinal, FrameAggregate(frame_ordinal, aggregate.frame_id))
                    target.samples.extend(aggregate.samples)
                left.contributors.update(right.contributors)
                left.decisions.append(
                    {
                        "type": "handoff-merge",
                        "mergedTrackId": right.internal_id,
                        "associationScore": clean_float(score, 6),
                        "breakdown": breakdown,
                    }
                )
                tracks.remove(right)
                changed = True
                break
    return sorted(tracks, key=lambda track: (track.first_frame, track.internal_id))


def smoothed_track_outputs(track: GlobalTrack, frame_ordinals: list[int], settings: dict[str, Any]) -> dict[int, dict[str, Any]]:
    alpha = clamp01(_float(settings, "positionSmoothingAlpha", 0.65))
    hold_frames = _int(settings, "holdFrames", 3)
    decay = clamp01(_float(settings, "confidenceDecay", 0.65))
    min_output = _float(settings, "minOutputConfidence", 0.35)
    outputs: dict[int, dict[str, Any]] = {}
    previous_position: list[float] | None = None
    previous_orientation: list[float] | None = None
    previous_position_confidence = 0.0
    previous_orientation_confidence = 0.0
    previous_observed_frame: int | None = None
    for frame_ordinal in frame_ordinals:
        aggregate = track.frames.get(frame_ordinal)
        if aggregate is not None and aggregate.samples:
            raw_position = aggregate.xyz
            smoothed_position = raw_position if previous_position is None else [
                clean_float(alpha * raw_position[index] + (1.0 - alpha) * previous_position[index], 6)
                for index in range(3)
            ]
            orientation = aggregate.rpy
            output = {
                "status": "observed",
                "raw_position_camera_m": [clean_float(item, 6) for item in raw_position],
                "position_camera_m": [clean_float(item, 6) for item in smoothed_position],
                "orientation_camera": [clean_float(item, 6) for item in orientation],
                "position_confidence": clean_float(aggregate.position_confidence, 6),
                "orientation_confidence": clean_float(aggregate.orientation_confidence, 6),
                "contributors": [sample.to_review_dict() for sample in aggregate.samples],
            }
            previous_position = list(smoothed_position)
            previous_orientation = list(orientation)
            previous_position_confidence = float(aggregate.position_confidence)
            previous_orientation_confidence = float(aggregate.orientation_confidence)
            previous_observed_frame = frame_ordinal
        elif (
            previous_position is not None
            and previous_orientation is not None
            and previous_observed_frame is not None
            and frame_ordinal - previous_observed_frame <= hold_frames
        ):
            gap = frame_ordinal - previous_observed_frame
            output = {
                "status": "held",
                "position_camera_m": [clean_float(item, 6) for item in previous_position],
                "orientation_camera": [clean_float(item, 6) for item in previous_orientation],
                "position_confidence": clean_float(previous_position_confidence * (decay ** gap), 6),
                "orientation_confidence": clean_float(previous_orientation_confidence * (decay ** gap), 6),
                "held_from_frame_ordinal": previous_observed_frame,
                "held_gap": gap,
                "contributors": [],
            }
        else:
            continue
        if float(output["position_confidence"]) >= min_output:
            outputs[frame_ordinal] = output
    return outputs


def solve_sample_score(instance: dict[str, Any], solve: dict[str, Any], settings: dict[str, Any]) -> tuple[float, dict[str, float]]:
    instance_score = clamp01(instance.get("finalInstanceScore"))
    fit_quality = solve.get("fitQuality") if isinstance(solve.get("fitQuality"), dict) else {}
    source_quality = max(clamp01(solve.get("source_score")), clamp01(solve.get("fit_score")), clamp01(fit_quality.get("overall")))
    reprojection_error = max(0.0, float(solve.get("reprojectionErrorPx") or 0.0))
    reprojection_quality = clamp01(1.0 - reprojection_error / max(1e-6, _float(settings, "maxReprojectionErrorPx", 20.0)))
    source_weight_quality = clamp01(SOURCE_WEIGHTS.get(str(solve.get("source")), 0.0) / max(SOURCE_WEIGHTS.values()))
    score = clamp01(0.45 * instance_score + 0.25 * source_quality + 0.20 * reprojection_quality + 0.10 * source_weight_quality)
    return clean_float(score, 6), {
        "instanceScore": clean_float(instance_score, 6),
        "sourceQuality": clean_float(source_quality, 6),
        "reprojectionQuality": clean_float(reprojection_quality, 6),
        "sourceWeightQuality": clean_float(source_weight_quality, 6),
    }


class GlobalGateMapper:
    def __init__(self, config: ProjectionConfig):
        self.settings = config.section("globalGateMapping")
        self.tracks: list[GateTrack] = []
        self.next_index = 1

    def process_run(
        self,
        profile_tracks: list[dict[str, Any]],
        frame_ordinals: list[int],
        frame_ids: dict[int, str],
    ) -> tuple[dict[int, list[VisionGateResult]], dict[str, Any]]:
        accepted_samples: list[PoseSample] = []
        rejected_samples: list[dict[str, Any]] = []
        for tracks_payload in profile_tracks:
            samples, rejected = extract_pose_samples(tracks_payload, self.settings)
            accepted_samples.extend(samples)
            rejected_samples.extend(rejected)

        ordered_frame_ordinals = sorted(set(int(item) for item in frame_ordinals))
        if not ordered_frame_ordinals:
            ordered_frame_ordinals = sorted({sample.frame_ordinal for sample in accepted_samples})
        resolved_frame_ids = {ordinal: frame_ids.get(ordinal, f"frame_{ordinal:06d}") for ordinal in ordered_frame_ordinals}

        tracks = build_global_tracks(accepted_samples, self.settings)
        gates_by_frame: dict[int, list[VisionGateResult]] = {ordinal: [] for ordinal in ordered_frame_ordinals}
        review_tracks: list[dict[str, Any]] = []
        for index, track in enumerate(tracks, start=1):
            gate_id = f"gate-{index:03d}"
            outputs = smoothed_track_outputs(track, ordered_frame_ordinals, self.settings)
            review_tracks.append(
                {
                    "gate_id": gate_id,
                    "internal_track_id": track.internal_id,
                    "first_frame_ordinal": track.first_frame,
                    "last_frame_ordinal": track.last_frame,
                    "contributor_count": len(track.contributors),
                    "contributors": [
                        {"profile_id": profile_id, "local_instance_id": local_id}
                        for profile_id, local_id in sorted(track.contributors)
                    ],
                    "decisions": list(track.decisions),
                    "observations": [
                        {
                            "frame_ordinal": frame_ordinal,
                            "frame_id": resolved_frame_ids.get(frame_ordinal, f"frame_{frame_ordinal:06d}"),
                            **output,
                        }
                        for frame_ordinal, output in sorted(outputs.items())
                    ],
                }
            )
            for frame_ordinal, output in outputs.items():
                position = tuple3(output.get("position_camera_m")) or [0.0, 0.0, 0.0]
                orientation = tuple3(output.get("orientation_camera"))
                gates_by_frame.setdefault(frame_ordinal, []).append(
                    VisionGateResult(
                        gate_id=gate_id,
                        position_camera_m=opencv_camera_to_result_camera(position),
                        position_confidence=clean_float(float(output.get("position_confidence") or 0.0), 6),
                        orientation_camera=None
                        if orientation is None
                        else (clean_float(orientation[0], 6), clean_float(orientation[1], 6), clean_float(orientation[2], 6)),
                        orientation_confidence=clean_float(float(output.get("orientation_confidence") or 0.0), 6),
                        trace={
                            "source_stage": "projection.global_mapping",
                            "status": output.get("status"),
                            "opencv_position_camera_m": [clean_float(item, 6) for item in position],
                            "coordinate_transform": "opencv-camera-to-result-camera-flip-y",
                            "contributor_count": len(output.get("contributors") or []),
                            "contributors": output.get("contributors") or [],
                        },
                    )
                )

        frame_index = [
            {
                "frame_ordinal": ordinal,
                "frame_id": resolved_frame_ids.get(ordinal, f"frame_{ordinal:06d}"),
                "gate_count": len(gates_by_frame.get(ordinal, [])),
            }
            for ordinal in ordered_frame_ordinals
        ]
        payload = {
            "schema": "projection-global-gate-mapping-run.v1",
            "frame_count": len(ordered_frame_ordinals),
            "profile_count": len(profile_tracks),
            "sample_count": len(accepted_samples),
            "rejected_sample_count": len(rejected_samples),
            "gate_count": len(review_tracks),
            "settings": dict(self.settings),
            "frame_index": frame_index,
            "tracks": review_tracks,
            "rejected_samples": rejected_samples,
        }
        return gates_by_frame, payload

    def _new_gate_id(self) -> str:
        gate_id = f"gate-{self.next_index:03d}"
        self.next_index += 1
        return gate_id

    def _sample_from_instance(self, instance: dict[str, Any]) -> dict[str, Any] | None:
        objects = [item for item in instance.get("objects") or [] if isinstance(item, dict)]
        if not objects:
            return None
        scored: list[tuple[float, dict[str, float], dict[str, Any]]] = []
        for solve in objects:
            xyz = tuple3(solve.get("xyzCameraM"))
            rpy = tuple3(solve.get("rpyCameraDeg"))
            if xyz is None or rpy is None or not solve.get("available"):
                continue
            score, signals = solve_sample_score(instance, solve, self.settings)
            if score < _float(self.settings, "minSampleScore", 0.45):
                continue
            scored.append((score, signals, solve))
        if not scored:
            return None
        score, signals, solve = max(scored, key=lambda item: (item[0], SOURCE_WEIGHTS.get(str(item[2].get("source")), 0.0)))
        xyz = tuple3(solve.get("xyzCameraM")) or [0.0, 0.0, 0.0]
        rpy = tuple3(solve.get("rpyCameraDeg")) or [0.0, 0.0, 0.0]
        return {
            "local_instance_id": instance.get("instance_id"),
            "void_id": (instance.get("void") or {}).get("void_id") if isinstance(instance.get("void"), dict) else None,
            "source": solve.get("source"),
            "xyz": xyz,
            "rpy": rpy,
            "depth": float(solve.get("depthM", xyz[2])),
            "sample_score": score,
            "position_confidence": clamp01(0.65 * score + 0.35 * signals["instanceScore"]),
            "orientation_confidence": clamp01(0.55 * score + 0.45 * signals["reprojectionQuality"]),
            "signals": signals,
            "solve": solve,
        }

    def _association(self, sample: dict[str, Any], track: GateTrack, frame_ordinal: int) -> tuple[float, dict[str, Any]]:
        gap = max(0, int(frame_ordinal) - int(track.last_frame_ordinal))
        if gap > _int(self.settings, "maxFrameGap", 8):
            return 0.0, {"reason": "frame-gap-too-large", "gap": gap}
        max_distance = max(1e-6, _float(self.settings, "maxDistanceM", 2.0))
        max_depth = max(1e-6, _float(self.settings, "maxDepthDeltaM", 4.0))
        max_angle = max(1e-6, _float(self.settings, "maxPlaneAngleDeg", 45.0))
        distance = distance3(sample["xyz"], track.raw_position)
        depth_delta = abs(float(sample["depth"]) - float(track.raw_position[2]))
        angle_delta = plane_angle_delta_deg(sample["rpy"], track.orientation)
        if distance is None or distance > max_distance:
            return 0.0, {"reason": "distance-too-large", "distanceM": distance}
        if depth_delta > max_depth:
            return 0.0, {"reason": "depth-delta-too-large", "depthDeltaM": depth_delta}
        if angle_delta > max_angle:
            return 0.0, {"reason": "plane-angle-too-large", "planeAngleDeg": angle_delta}
        score = clamp01(
            0.35 * (1.0 - distance / max_distance)
            + 0.20 * (1.0 - depth_delta / max_depth)
            + 0.25 * (1.0 - angle_delta / max_angle)
            + 0.10 * sample["sample_score"]
            + (0.10 if str(sample["local_instance_id"]) in track.contributors else 0.0)
        )
        return clean_float(score, 6), {
            "reason": "accepted",
            "distanceM": clean_float(distance, 6),
            "depthDeltaM": clean_float(depth_delta, 6),
            "planeAngleDeg": clean_float(angle_delta, 6),
            "associationScore": clean_float(score, 6),
        }

    def _update_track(self, track: GateTrack, sample: dict[str, Any], meta: FrameMeta, association: dict[str, Any]) -> None:
        alpha = clamp01(_float(self.settings, "positionSmoothingAlpha", 0.65))
        raw = [clean_float(item, 6) for item in sample["xyz"]]
        track.position = [clean_float(alpha * raw[index] + (1.0 - alpha) * track.position[index], 6) for index in range(3)]
        track.raw_position = raw
        track.orientation = [clean_float(item, 6) for item in sample["rpy"]]
        track.position_confidence = float(sample["position_confidence"])
        track.orientation_confidence = float(sample["orientation_confidence"])
        track.last_frame_ordinal = int(meta.frame_ordinal)
        track.contributors.add(str(sample["local_instance_id"]))
        track.observations.append(
            {
                "frame_ordinal": meta.frame_ordinal,
                "frame_id": meta.frame_id,
                "sample": sample,
                "association": association,
            }
        )

    def process(self, meta: FrameMeta, tracking_payload: dict[str, Any]) -> tuple[list[VisionGateResult], dict[str, Any]]:
        samples = [self._sample_from_instance(instance) for instance in tracking_payload.get("instances", [])]
        samples = [sample for sample in samples if sample is not None]
        min_association = _float(self.settings, "minAssociationScore", 0.55)
        for sample in samples:
            best_track: GateTrack | None = None
            best_score = 0.0
            best_breakdown: dict[str, Any] = {}
            for track in self.tracks:
                score, breakdown = self._association(sample, track, meta.frame_ordinal)
                if score > best_score:
                    best_track = track
                    best_score = score
                    best_breakdown = breakdown
            if best_track is None or best_score < min_association:
                best_track = GateTrack(
                    gate_id=self._new_gate_id(),
                    last_frame_ordinal=meta.frame_ordinal,
                    position=[clean_float(item, 6) for item in sample["xyz"]],
                    raw_position=[clean_float(item, 6) for item in sample["xyz"]],
                    orientation=[clean_float(item, 6) for item in sample["rpy"]],
                    position_confidence=float(sample["position_confidence"]),
                    orientation_confidence=float(sample["orientation_confidence"]),
                    contributors={str(sample["local_instance_id"])},
                )
                self.tracks.append(best_track)
                best_breakdown = {"reason": "new-global-gate"}
            else:
                self._update_track(best_track, sample, meta, best_breakdown)

        gates: list[VisionGateResult] = []
        held_count = 0
        for track in sorted(self.tracks, key=lambda item: item.gate_id):
            gap = int(meta.frame_ordinal) - int(track.last_frame_ordinal)
            if gap == 0:
                position_confidence = track.position_confidence
                orientation_confidence = track.orientation_confidence
                status = "observed"
            elif gap <= _int(self.settings, "holdFrames", 3):
                decay = clamp01(_float(self.settings, "confidenceDecay", 0.65))
                position_confidence = track.position_confidence * (decay ** gap)
                orientation_confidence = track.orientation_confidence * (decay ** gap)
                status = "held"
                held_count += 1
            else:
                continue
            if position_confidence < _float(self.settings, "minOutputConfidence", 0.35):
                continue
            gates.append(
                VisionGateResult(
                    gate_id=track.gate_id,
                    position_camera_m=opencv_camera_to_result_camera(track.position),
                    position_confidence=clean_float(position_confidence, 6),
                    orientation_camera=(clean_float(track.orientation[0], 6), clean_float(track.orientation[1], 6), clean_float(track.orientation[2], 6)),
                    orientation_confidence=clean_float(orientation_confidence, 6),
                    trace={
                        "source_stage": "projection.global_mapping",
                        "status": status,
                        "opencv_position_camera_m": [clean_float(item, 6) for item in track.position],
                        "coordinate_transform": "opencv-camera-to-result-camera-flip-y",
                        "contributor_count": len(track.contributors),
                        "last_frame_ordinal": track.last_frame_ordinal,
                    },
                )
            )
        return gates, {
            "schema": "projection-global-mapping.v1",
            "frame_ordinal": meta.frame_ordinal,
            "sample_count": len(samples),
            "gate_count": len(gates),
            "track_count": len(self.tracks),
            "held_gate_count": held_count,
            "settings": dict(self.settings),
        }
