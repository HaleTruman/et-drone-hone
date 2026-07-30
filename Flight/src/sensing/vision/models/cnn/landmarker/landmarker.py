"""Deterministic camera-local landmark fusion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


LANDMARK_STATE_SCHEMA_VERSION = "extraction_landmarker_state_v1"
DEFAULT_LANDMARK_CONFIG: dict[str, float] = {
    "association_distance_m": 6.0,
    "association_score_m": 7.0,
    "orientation_score_weight_m": 1.0,
    "min_confidence": 0.05,
    "confidence_growth_observations": 5.0,
}


def _clamp_confidence(value: Any, minimum: float) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    return max(float(minimum), min(1.0, raw))


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def _add(left: list[float], right: list[float]) -> list[float]:
    return [left[0] + right[0], left[1] + right[1], left[2] + right[2]]


def _sub(left: list[float], right: list[float]) -> list[float]:
    return [left[0] - right[0], left[1] - right[1], left[2] - right[2]]


def _scale(value: list[float], scalar: float) -> list[float]:
    return [value[0] * scalar, value[1] * scalar, value[2] * scalar]


def _dot(left: list[float], right: list[float]) -> float:
    return (left[0] * right[0]) + (left[1] * right[1]) + (left[2] * right[2])


def _norm(value: list[float]) -> float:
    return math.sqrt(max(0.0, _dot(value, value)))


def _normalize(value: list[float] | None) -> list[float] | None:
    if value is None:
        return None
    norm = _norm(value)
    if norm <= 1.0e-6 or not math.isfinite(norm):
        return None
    return _scale(value, 1.0 / norm)


def _distance(left: list[float], right: list[float]) -> float:
    return _norm(_sub(left, right))


def _orientation_angle_deg(left: list[float] | None, right: list[float] | None) -> float | None:
    left_norm = _normalize(left)
    right_norm = _normalize(right)
    if left_norm is None or right_norm is None:
        return None
    dot_abs = abs(_dot(left_norm, right_norm))
    return math.degrees(math.acos(max(0.0, min(1.0, dot_abs))))


def _official_gate_id(index: int) -> str:
    return f"gate-{int(index):03d}w"


def _landmark_confidence(weight_sum: float, scale: float) -> float:
    return 1.0 - math.exp(-max(0.0, float(weight_sum)) / max(1.0e-6, float(scale)))


def new_landmarker_state() -> dict[str, Any]:
    return {
        "schema_version": LANDMARK_STATE_SCHEMA_VERSION,
        "coordinate_frame": "camera_local_v1",
        "next_gate_index": 1,
        "landmarks": {"gates": []},
    }


def load_landmarker_state(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return new_landmarker_state()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Landmarker state is not an object: {resolved}")
    if payload.get("schema_version") != LANDMARK_STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported landmarker state schema {payload.get('schema_version')}; "
            f"expected {LANDMARK_STATE_SCHEMA_VERSION}."
        )
    if not isinstance(payload.get("landmarks"), dict) or not isinstance(payload["landmarks"].get("gates"), list):
        raise ValueError(f"Landmarker state is missing landmarks.gates: {resolved}")
    return payload


def save_landmarker_state(path: Path, state: dict[str, Any]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(state, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return resolved


def _observation_from_gate(frame_payload: dict[str, Any], gate: dict[str, Any], config: dict[str, float]) -> dict[str, Any] | None:
    position = _vector(gate.get("position_xyz"))
    if position is None:
        return None
    orientation = _normalize(_vector(gate.get("orientation_xyz")))
    run = frame_payload.get("run") if isinstance(frame_payload.get("run"), dict) else {}
    cycle = int(run.get("cycle", 0))
    frame_id = str(run.get("frame_id") or f"frame_{cycle:06d}")
    position_confidence = _clamp_confidence(gate.get("position_confidence"), float(config["min_confidence"]))
    orientation_confidence = _clamp_confidence(gate.get("orientation_confidence"), float(config["min_confidence"]))
    return {
        "source_id": str(gate.get("id", "")),
        "cycle": cycle,
        "frame_id": frame_id,
        "position": position,
        "position_confidence": position_confidence,
        "orientation": orientation,
        "orientation_confidence": orientation_confidence if orientation is not None else 0.0,
    }


def _new_landmark(state: dict[str, Any]) -> dict[str, Any]:
    gate_index = int(state.get("next_gate_index", 1))
    state["next_gate_index"] = gate_index + 1
    return {
        "id": _official_gate_id(gate_index),
        "position_xyz": [0.0, 0.0, 0.0],
        "position_weight_sum": 0.0,
        "position_confidence": 0.0,
        "orientation_xyz": None,
        "orientation_sum_xyz": [0.0, 0.0, 0.0],
        "orientation_weight_sum": 0.0,
        "orientation_confidence": 0.0,
        "observation_count": 0,
        "first_seen_cycle": None,
        "last_seen_cycle": None,
        "source_ids": [],
    }


def _association_score(landmark: dict[str, Any], observation: dict[str, Any], config: dict[str, float]) -> tuple[float, float, float]:
    position_distance = _distance(landmark["position_xyz"], observation["position"])
    orientation_angle = _orientation_angle_deg(landmark.get("orientation_xyz"), observation.get("orientation"))
    orientation_score = 0.0
    if orientation_angle is not None:
        orientation_score = (orientation_angle / 45.0) * float(config["orientation_score_weight_m"])
    return position_distance + orientation_score, position_distance, -1.0 if orientation_angle is None else orientation_angle


def _apply_observation(landmark: dict[str, Any], observation: dict[str, Any], config: dict[str, float]) -> None:
    position_weight = float(observation["position_confidence"])
    old_weight = float(landmark.get("position_weight_sum", 0.0))
    total_weight = old_weight + position_weight
    if total_weight > 1.0e-6:
        old_position = landmark["position_xyz"]
        new_position = observation["position"]
        landmark["position_xyz"] = [
            ((old_position[index] * old_weight) + (new_position[index] * position_weight)) / total_weight
            for index in range(3)
        ]
        landmark["position_weight_sum"] = total_weight
        landmark["position_confidence"] = _landmark_confidence(total_weight, float(config["confidence_growth_observations"]))

    orientation = observation.get("orientation")
    orientation_weight = float(observation.get("orientation_confidence", 0.0))
    if orientation is not None and orientation_weight > 0.0:
        current = _normalize(landmark.get("orientation_xyz"))
        if current is not None and _dot(current, orientation) < 0.0:
            orientation = _scale(orientation, -1.0)
        landmark["orientation_sum_xyz"] = _add(landmark["orientation_sum_xyz"], _scale(orientation, orientation_weight))
        landmark["orientation_weight_sum"] = float(landmark.get("orientation_weight_sum", 0.0)) + orientation_weight
        landmark["orientation_xyz"] = _normalize(landmark["orientation_sum_xyz"])
        landmark["orientation_confidence"] = _landmark_confidence(
            float(landmark["orientation_weight_sum"]),
            float(config["confidence_growth_observations"]),
        )

    landmark["observation_count"] = int(landmark.get("observation_count", 0)) + 1
    if landmark.get("first_seen_cycle") is None:
        landmark["first_seen_cycle"] = int(observation["cycle"])
    landmark["last_seen_cycle"] = int(observation["cycle"])
    if observation["source_id"]:
        landmark.setdefault("source_ids", []).append(str(observation["source_id"]))


class Landmarker:
    def __init__(self, state: dict[str, Any] | None = None, config: dict[str, float] | None = None) -> None:
        self.state = state if state is not None else new_landmarker_state()
        self.config = {**DEFAULT_LANDMARK_CONFIG, **(config or {})}

    def update_frame(self, frame_payload: dict[str, Any]) -> dict[str, Any]:
        gates = frame_payload.get("gates", [])
        if not isinstance(gates, list):
            raise ValueError("Regressor frame payload has non-list gates.")
        observations = [
            observation
            for gate in gates
            if isinstance(gate, dict) and (observation := _observation_from_gate(frame_payload, gate, self.config)) is not None
        ]
        observations.sort(key=lambda item: float(item["position_confidence"]), reverse=True)
        updated_ids: set[str] = set()
        current_gates: list[dict[str, Any]] = []
        landmarks = self.state["landmarks"]["gates"]

        for observation in observations:
            best: tuple[float, dict[str, Any]] | None = None
            for landmark in landmarks:
                if str(landmark["id"]) in updated_ids:
                    continue
                score, position_distance, _orientation_angle = _association_score(landmark, observation, self.config)
                if position_distance > float(self.config["association_distance_m"]):
                    continue
                if score <= float(self.config["association_score_m"]) and (best is None or score < best[0]):
                    best = (score, landmark)
            if best is None:
                landmark = _new_landmark(self.state)
                landmarks.append(landmark)
            else:
                landmark = best[1]
            _apply_observation(landmark, observation, self.config)
            official_id = str(landmark["id"])
            updated_ids.add(official_id)
            current_gates.append(
                {
                    "id": official_id,
                    "position_xyz": [float(value) for value in observation["position"]],
                    "position_confidence": float(observation["position_confidence"]),
                    "orientation_xyz": None
                    if observation.get("orientation") is None
                    else [float(value) for value in observation["orientation"]],
                    "orientation_confidence": float(observation.get("orientation_confidence", 0.0)),
                }
            )

        return {
            "cycle": int((frame_payload.get("run") or {}).get("cycle", 0)),
            "frame_id": str((frame_payload.get("run") or {}).get("frame_id", "")),
            "updated_landmark_ids": sorted(updated_ids),
            "current_gates": current_gates,
            "state": self.state,
        }


__all__ = [
    "DEFAULT_LANDMARK_CONFIG",
    "LANDMARK_STATE_SCHEMA_VERSION",
    "Landmarker",
    "load_landmarker_state",
    "new_landmarker_state",
    "save_landmarker_state",
]
