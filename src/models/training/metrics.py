"""Decoding and metrics for multi-gate detection, keypoints, and solved pose."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from src.models.geometry import solve_gate_pose
from src.models.training.targets import (
    CameraCalibration,
    GateFrame,
    OUTER_KEYPOINT_COUNT,
)


def decode_detections(
    outputs: Sequence[dict[str, torch.Tensor]],
    samples: Sequence[GateFrame],
    calibration: CameraCalibration,
    score_threshold: float = 0.7,
    duplicate_iou_threshold: float = 0.25,
    suppress_contained_duplicates: bool = True,
    containment_area_ratio: float = 0.85,
) -> list[list[dict[str, object]]]:
    decoded: list[list[dict[str, object]]] = []
    for output, sample in zip(outputs, samples):
        frame_calibration = calibration.scaled_to(sample.width, sample.height)
        frame_detections: list[dict[str, object]] = []
        scores = output["scores"].detach().cpu().numpy()
        boxes = output["boxes"].detach().cpu().numpy()
        keypoints = output["keypoints"].detach().cpu().numpy()[..., :2]
        keypoint_scores = output.get("keypoints_scores")
        keypoint_scores_array = (
            keypoint_scores.detach().cpu().numpy()
            if keypoint_scores is not None
            else np.ones(keypoints.shape[:2], dtype=np.float32)
        )
        for index in np.flatnonzero(scores >= score_threshold):
            all_corners = keypoints[index].astype(np.float32)
            outer_corners = all_corners[:OUTER_KEYPOINT_COUNT]
            inner_corners = (
                all_corners[OUTER_KEYPOINT_COUNT:OUTER_KEYPOINT_COUNT * 2]
                if all_corners.shape[0] >= OUTER_KEYPOINT_COUNT * 2
                else None
            )
            pose = solve_gate_pose(outer_corners, frame_calibration)
            frame_detections.append(
                {
                    "score": float(scores[index]),
                    "bbox_xyxy": boxes[index].astype(np.float32),
                    "outer_corners": outer_corners.astype(np.float32),
                    "inner_corners": (
                        inner_corners.astype(np.float32)
                        if inner_corners is not None
                        else None
                    ),
                    "keypoints_2d": all_corners.astype(np.float32),
                    "keypoint_scores": keypoint_scores_array[index].astype(np.float32),
                    "pose": pose,
                }
            )
        decoded.append(
            suppress_duplicate_detections(
                frame_detections,
                duplicate_iou_threshold=duplicate_iou_threshold,
                suppress_contained_duplicates=suppress_contained_duplicates,
                containment_area_ratio=containment_area_ratio,
            )
        )
    return decoded


def suppress_duplicate_detections(
    detections: Sequence[dict[str, object]],
    duplicate_iou_threshold: float = 0.25,
    suppress_contained_duplicates: bool = True,
    containment_area_ratio: float = 0.85,
) -> list[dict[str, object]]:
    """Remove lower-score detections that likely describe an already-kept gate."""

    if not detections:
        return []
    kept: list[dict[str, object]] = []
    for detection in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
        box = np.asarray(detection["bbox_xyxy"])
        if any(
            _is_duplicate_gate(
                box,
                np.asarray(existing["bbox_xyxy"]),
                duplicate_iou_threshold,
                suppress_contained_duplicates,
                containment_area_ratio,
            )
            for existing in kept
        ):
            continue
        kept.append(dict(detection))
    return kept


def _is_duplicate_gate(
    candidate: np.ndarray,
    kept: np.ndarray,
    duplicate_iou_threshold: float,
    suppress_contained_duplicates: bool,
    containment_area_ratio: float,
) -> bool:
    if single_bbox_iou(candidate, kept) >= duplicate_iou_threshold:
        return True
    if not suppress_contained_duplicates:
        return False

    candidate_center_x = float((candidate[0] + candidate[2]) * 0.5)
    candidate_center_y = float((candidate[1] + candidate[3]) * 0.5)
    center_inside_kept = (
        float(kept[0]) <= candidate_center_x <= float(kept[2])
        and float(kept[1]) <= candidate_center_y <= float(kept[3])
    )
    if not center_inside_kept:
        return False

    candidate_area = bbox_area(candidate)
    kept_area = bbox_area(kept)
    if kept_area <= 0.0:
        return False
    return candidate_area <= kept_area * containment_area_ratio


def calculate_metrics(
    decoded: Sequence[Sequence[dict[str, object]]],
    samples: Sequence[GateFrame],
    match_iou: float = 0.5,
) -> dict[str, object]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    box_ious: list[float] = []
    corner_distances: list[np.ndarray] = []
    visible_corner_distances: list[np.ndarray] = []
    all_corner_distances: list[np.ndarray] = []
    visible_all_corner_distances: list[np.ndarray] = []
    position_errors: list[np.ndarray] = []
    orientation_errors: list[np.ndarray] = []
    visible_outer_corner_count = 0
    total_outer_corner_count = 0
    visible_label_corner_count = 0
    total_label_corner_count = 0

    for detections, sample in zip(decoded, samples):
        for target in sample.gates:
            total_outer_corner_count += len(target.keypoint_visibility)
            visible_outer_corner_count += int((target.keypoint_visibility > 0.0).sum())
            if target.all_keypoint_visibility is not None:
                total_label_corner_count += len(target.all_keypoint_visibility)
                visible_label_corner_count += int(
                    (target.all_keypoint_visibility > 0.0).sum()
                )
        matches, unmatched_predictions, unmatched_targets = match_detections(
            detections,
            sample,
            match_iou,
        )
        true_positive += len(matches)
        false_positive += len(unmatched_predictions)
        false_negative += len(unmatched_targets)
        for prediction_index, target_index, iou in matches:
            prediction = detections[prediction_index]
            target = sample.gates[target_index]
            box_ious.append(iou)
            distances = np.linalg.norm(
                np.asarray(prediction["outer_corners"]) - target.outer_corners,
                axis=1,
            )
            corner_distances.append(distances)
            visible_mask = target.keypoint_visibility > 0.0
            if visible_mask.any():
                visible_corner_distances.append(distances[visible_mask])
            prediction_keypoints = prediction.get("keypoints_2d")
            if (
                prediction_keypoints is not None
                and target.all_corners is not None
                and target.all_keypoint_visibility is not None
            ):
                predicted_all = np.asarray(prediction_keypoints)
                target_all = np.asarray(target.all_corners)
                count = min(len(predicted_all), len(target_all))
                if count:
                    all_distances = np.linalg.norm(
                        predicted_all[:count] - target_all[:count],
                        axis=1,
                    )
                    all_corner_distances.append(all_distances)
                    all_visible_mask = target.all_keypoint_visibility[:count] > 0.0
                    if all_visible_mask.any():
                        visible_all_corner_distances.append(
                            all_distances[all_visible_mask]
                        )
            pose = prediction.get("pose")
            if pose is not None:
                position_errors.append(
                    np.abs(np.asarray(pose["position_m"]) - target.position)
                )
                predicted_orientation = np.asarray(pose["orientation_deg"])
                orientation_errors.append(
                    np.asarray(
                        [
                            yaw_180_absolute_error(
                                predicted_orientation[0],
                                target.orientation_deg[0],
                            ),
                            circular_absolute_error(
                                predicted_orientation[1],
                                target.orientation_deg[1],
                            ),
                            circular_absolute_error(
                                predicted_orientation[2],
                                target.orientation_deg[2],
                            ),
                        ]
                    )
                )

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    corner_array = (
        np.stack(corner_distances)
        if corner_distances
        else np.empty((0, 4), dtype=np.float32)
    )
    visible_corner_array = (
        np.concatenate(visible_corner_distances)
        if visible_corner_distances
        else np.empty((0,), dtype=np.float32)
    )
    all_corner_array = (
        np.concatenate(all_corner_distances)
        if all_corner_distances
        else np.empty((0,), dtype=np.float32)
    )
    visible_all_corner_array = (
        np.concatenate(visible_all_corner_distances)
        if visible_all_corner_distances
        else np.empty((0,), dtype=np.float32)
    )
    position_array = (
        np.stack(position_errors)
        if position_errors
        else np.empty((0, 3), dtype=np.float32)
    )
    orientation_array = (
        np.stack(orientation_errors)
        if orientation_errors
        else np.empty((0, 3), dtype=np.float32)
    )
    return {
        "frame_count": len(samples),
        "target_gate_count": sum(len(sample.gates) for sample in samples),
        "predicted_gate_count": sum(len(frame) for frame in decoded),
        "detection_iou_threshold": match_iou,
        "detection_true_positive": true_positive,
        "detection_false_positive": false_positive,
        "detection_false_negative": false_negative,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": 2.0 * precision * recall / max(1e-12, precision + recall),
        "matched_bbox_iou_mean": _mean_or_none(np.asarray(box_ious)),
        "matched_corner_mean_distance_px": _mean_or_none(corner_array),
        "matched_visible_corner_mean_distance_px": _mean_or_none(visible_corner_array),
        "matched_all_corner_mean_distance_px": _mean_or_none(all_corner_array),
        "matched_visible_all_corner_mean_distance_px": _mean_or_none(
            visible_all_corner_array
        ),
        "matched_corner_distance_px": {
            name: _mean_or_none(corner_array[:, index])
            for index, name in enumerate(("TL", "TR", "BL", "BR"))
        },
        "target_visible_outer_corner_count": visible_outer_corner_count,
        "target_outer_corner_count": total_outer_corner_count,
        "target_visible_label_corner_count": visible_label_corner_count,
        "target_label_corner_count": total_label_corner_count,
        "pose_solution_count": len(position_errors),
        "position_mae_m": {
            name: _mean_or_none(position_array[:, index])
            for index, name in enumerate(("right", "up", "forward"))
        },
        "position_l2_mean_m": _mean_or_none(
            np.linalg.norm(position_array, axis=1)
            if len(position_array)
            else np.asarray([])
        ),
        "orientation_mae_deg": {
            name: _mean_or_none(orientation_array[:, index])
            for index, name in enumerate(("yaw_mod_180", "pitch", "roll"))
        },
        "orientation_mean_mae_deg": _mean_or_none(orientation_array),
    }


def match_detections(
    detections: Sequence[dict[str, object]],
    sample: GateFrame,
    minimum_iou: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(detections):
        for target_index, target in enumerate(sample.gates):
            iou = single_bbox_iou(
                np.asarray(prediction["bbox_xyxy"]),
                target.bbox_xyxy,
            )
            if iou >= minimum_iou:
                candidates.append((iou, prediction_index, target_index))
    matches: list[tuple[int, int, float]] = []
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    for iou, prediction_index, target_index in sorted(candidates, reverse=True):
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        matches.append((prediction_index, target_index, iou))
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
    return (
        matches,
        set(range(len(detections))) - used_predictions,
        set(range(len(sample.gates))) - used_targets,
    )


def circular_absolute_error(
    predicted_deg: float | np.ndarray,
    target_deg: float | np.ndarray,
) -> float | np.ndarray:
    return np.abs(
        (np.asarray(predicted_deg) - np.asarray(target_deg) + 180.0) % 360.0
        - 180.0
    )


def yaw_180_absolute_error(
    predicted_deg: float | np.ndarray,
    target_deg: float | np.ndarray,
) -> float | np.ndarray:
    return np.abs(
        (np.asarray(predicted_deg) - np.asarray(target_deg) + 90.0) % 180.0
        - 90.0
    )


def single_bbox_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    intersection_width = max(
        0.0,
        min(float(predicted[2]), float(target[2]))
        - max(float(predicted[0]), float(target[0])),
    )
    intersection_height = max(
        0.0,
        min(float(predicted[3]), float(target[3]))
        - max(float(predicted[1]), float(target[1])),
    )
    intersection = intersection_width * intersection_height
    predicted_area = max(0.0, float(predicted[2] - predicted[0])) * max(
        0.0, float(predicted[3] - predicted[1])
    )
    target_area = max(0.0, float(target[2] - target[0])) * max(
        0.0, float(target[3] - target[1])
    )
    union = predicted_area + target_area - intersection
    return intersection / union if union > 0.0 else 0.0


def bbox_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None
