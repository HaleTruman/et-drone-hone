"""Compare Flight deterministic vision backends over a recorded flight run.

This module is intentionally housed in LegacyVision.  It imports the live
Flight backends for offline evaluation, but does not modify Flight code or use
LegacyVision algorithms in the live runtime.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLIGHT_SRC = PROJECT_ROOT / "Flight" / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(FLIGHT_SRC) not in sys.path:
    sys.path.insert(0, str(FLIGHT_SRC))

from core.schema import VehicleState, VisionFrame, VisionObservation
from sensing.vision.models.deterministic_v3 import DeterministicVision as DeterministicVisionV3
from sensing.vision.models.deterministic_v3_2 import DeterministicVision as DeterministicVisionV3_2


DEFAULT_FLIGHT_RUNS_ROOT = PROJECT_ROOT / "Viewer" / "logs" / "flight" / "runs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Viewer" / "logs" / "evaluation" / "runs"
DEFAULT_OVERLAY_LAYERS = (
    "composite",
    "contours",
    "ellipses",
    "outer_corners",
    "inner_corners",
    "parent_obtuse_corners",
)
BACKEND_FACTORIES = {
    "deterministic_v3": DeterministicVisionV3,
    "deterministic_v3_2": DeterministicVisionV3_2,
}
BACKEND_MODULES = {
    "deterministic_v3": "sensing.vision.models.deterministic_v3.src.void_detection",
    "deterministic_v3_2": "sensing.vision.models.deterministic_v3_2.src.void_detection",
}


class EvaluationError(RuntimeError):
    """Raised for invalid evaluation inputs."""


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(payload), separators=(",", ":")) + "\n")


def _read_json_if_present(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _vec3(payload: Any, field_name: str) -> tuple[float, float, float]:
    if not isinstance(payload, (list, tuple)) or len(payload) != 3:
        raise EvaluationError(f"vehicle_state.{field_name} must contain three values")
    return tuple(float(value) for value in payload)


def _quat(payload: Any, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(payload, (list, tuple)) or len(payload) != 4:
        raise EvaluationError(f"vehicle_state.{field_name} must contain four values")
    return tuple(float(value) for value in payload)


def _vehicle_state_from_payload(payload: dict[str, Any]) -> VehicleState:
    return VehicleState(
        sim_time_ns=int(payload["sim_time_ns"]),
        position_local_ned_m=_vec3(payload["position_local_ned_m"], "position_local_ned_m"),
        velocity_local_ned_mps=_vec3(payload["velocity_local_ned_mps"], "velocity_local_ned_mps"),
        attitude_quaternion=_quat(payload["attitude_quaternion"], "attitude_quaternion"),
        body_rates_frd_rps=_vec3(payload["body_rates_frd_rps"], "body_rates_frd_rps"),
        acceleration_local_ned_mps2=_vec3(
            payload["acceleration_local_ned_mps2"], "acceleration_local_ned_mps2"
        ),
        elapsed_time_ns=None
        if payload.get("elapsed_time_ns") is None
        else int(payload["elapsed_time_ns"]),
    )


def _load_vehicle_states_by_frame_identity(run_dir: Path) -> dict[tuple[int, int], VehicleState]:
    """Join frame manifest identity to telemetry via logged inner cycle."""
    vision_path = run_dir / "lists" / "vision_frames.jsonl"
    telemetry_path = run_dir / "lists" / "telemetry.jsonl"
    frames_path = run_dir / "frames.jsonl"
    if not vision_path.is_file() or not telemetry_path.is_file() or not frames_path.is_file():
        return {}

    telemetry_by_cycle: dict[int, VehicleState] = {}
    with telemetry_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cycle = int(record["inner_cycle"])
                state = _vehicle_state_from_payload(record["telemetry"]["vehicle_state"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationError(
                    f"invalid telemetry record at {telemetry_path}:{line_number}"
                ) from exc
            if cycle in telemetry_by_cycle:
                raise EvaluationError(f"duplicate telemetry inner_cycle in {telemetry_path}: {cycle}")
            telemetry_by_cycle[cycle] = state

    cycles_by_frame: dict[tuple[int, int], set[int]] = {}
    with vision_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)["frame"]
                frame_id = int(frame["frame_id"])
                sim_time_ns = int(frame["sim_time_ns"])
                cycle = int(frame["inner_cycle"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationError(
                    f"invalid vision-frame record at {vision_path}:{line_number}"
                ) from exc
            cycles_by_frame.setdefault((frame_id, sim_time_ns), set()).add(cycle)

    states: dict[tuple[int, int], VehicleState] = {}
    with frames_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
                identity = (int(frame["frame_id"]), int(frame["sim_time_ns"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationError(
                    f"invalid frame manifest record at {frames_path}:{line_number}"
                ) from exc
            cycles = cycles_by_frame.get(identity, set())
            if len(cycles) == 1:
                state = telemetry_by_cycle.get(next(iter(cycles)))
                if state is not None:
                    states[identity] = state
    return states


def _iter_frame_records(run_dir: Path) -> Iterable[dict[str, Any]]:
    manifest_path = run_dir / "frames.jsonl"
    if not manifest_path.is_file():
        raise EvaluationError(f"flight run has no frame manifest: {manifest_path}")
    previous_time = -1
    seen: set[tuple[int, int]] = set()
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frame_id = int(record["frame_id"])
                sim_time_ns = int(record["sim_time_ns"])
                jpeg_size = int(record["jpeg_size"])
                relative_path = Path(str(record["path"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationError(
                    f"invalid frame manifest record at {manifest_path}:{line_number}"
                ) from exc
            identity = (frame_id, sim_time_ns)
            if identity in seen:
                raise EvaluationError(f"duplicate frame identity in manifest: {identity}")
            if sim_time_ns < previous_time:
                raise EvaluationError("frame manifest timing is not monotonic")
            if relative_path.is_absolute() or not relative_path.parts:
                raise EvaluationError(f"frame path must be relative: {relative_path}")
            source_path = (run_dir / relative_path).resolve()
            if not source_path.is_relative_to(run_dir):
                raise EvaluationError(f"frame path escapes run: {relative_path}")
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            seen.add(identity)
            previous_time = sim_time_ns
            yield {
                "frame_id": frame_id,
                "sim_time_ns": sim_time_ns,
                "elapsed_time_ns": record.get("elapsed_time_ns"),
                "jpeg_size": jpeg_size,
                "relative_path": relative_path.as_posix(),
                "source_path": source_path,
            }


def _observation_payload(observation: VisionObservation) -> dict[str, Any]:
    payload = asdict(observation)
    payload["controller_payload"] = observation.to_controller_payload()
    return payload


def _error_payload(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _run_backend(
    backend: Any,
    frame_record: dict[str, Any],
    *,
    vehicle_state: VehicleState | None,
) -> dict[str, Any]:
    jpeg_bytes = frame_record["source_path"].read_bytes()
    if len(jpeg_bytes) != frame_record["jpeg_size"]:
        raise EvaluationError(
            f"JPEG size changed for {frame_record['relative_path']}: "
            f"{len(jpeg_bytes)} != {frame_record['jpeg_size']}"
        )
    frame = VisionFrame(
        frame_id=frame_record["frame_id"],
        sim_time_ns=frame_record["sim_time_ns"],
        jpeg_bytes=jpeg_bytes,
        saved_path=frame_record["relative_path"],
        elapsed_time_ns=frame_record["elapsed_time_ns"],
    )
    started = time.perf_counter_ns()
    try:
        observation = backend.process_frame(frame, vehicle_state)
    except Exception as exc:  # noqa: BLE001 - evaluation must continue per frame.
        return {
            "status": "error",
            "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
            "error": _error_payload(exc),
        }
    return {
        "status": "ok",
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "observation": _observation_payload(observation),
    }


def _overlay_filename(record: dict[str, Any]) -> str:
    return f"frame-{int(record['frame_id']):08d}-{int(record['sim_time_ns'])}.png"


def _render_backend_overlays(
    backend_name: str,
    renderer: Any,
    lut: Any,
    frame_record: dict[str, Any],
    output_dir: Path,
    overlay_layers: tuple[str, ...],
    frame_index: int,
    frame_count: int,
) -> dict[str, Any]:
    import cv2

    image = cv2.imread(str(frame_record["source_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "status": "error",
            "error": {"type": "ValueError", "message": f"unable to read {frame_record['source_path']}"},
        }
    context = {
        "id_start": 0,
        "run_id": backend_name,
        "frame_id": Path(frame_record["relative_path"]).stem,
        "frame_index": frame_index,
        "frame_count": frame_count,
    }
    started = time.perf_counter_ns()
    try:
        rendered_layers, _, _ = renderer.render_review_layers(image, context, lut)
    except Exception as exc:  # noqa: BLE001 - evaluation should keep comparing frames.
        return {
            "status": "error",
            "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
            "error": _error_payload(exc),
        }

    selected_layers = tuple(rendered_layers) if overlay_layers == ("all",) else overlay_layers
    written = []
    for layer_id in selected_layers:
        layer = rendered_layers.get(layer_id)
        if layer is None:
            continue
        path = output_dir / "overlays" / backend_name / layer_id / _overlay_filename(frame_record)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), layer):
            return {
                "status": "error",
                "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
                "error": {"type": "ValueError", "message": f"unable to write {path}"},
            }
        written.append(layer_id)
    return {
        "status": "ok",
        "elapsed_ms": (time.perf_counter_ns() - started) / 1_000_000.0,
        "layers": written,
    }


def _load_overlay_renderers() -> dict[str, dict[str, Any]]:
    renderers = {}
    for backend_name, module_name in BACKEND_MODULES.items():
        module = importlib.import_module(module_name)
        renderers[backend_name] = {
            "module": module,
            "lut": module.load_lut(module.LUT_PATH),
        }
    return renderers


def _gate_position(gate: dict[str, Any]) -> tuple[float, float, float] | None:
    value = gate.get("position_local_ned")
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(float(item) for item in value)
    return None


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def _compare_observations(left_result: dict[str, Any], right_result: dict[str, Any]) -> dict[str, Any]:
    if left_result["status"] != "ok" or right_result["status"] != "ok":
        return {
            "same_status": left_result["status"] == right_result["status"],
            "left_status": left_result["status"],
            "right_status": right_result["status"],
            "gate_count_delta": None,
            "max_position_delta_m": None,
            "changed": True,
        }

    left = left_result["observation"]
    right = right_result["observation"]
    left_gates = list(left.get("gates", ()))
    right_gates = list(right.get("gates", ()))
    gate_pairs = []
    for index, (left_gate, right_gate) in enumerate(zip(left_gates, right_gates)):
        left_position = _gate_position(left_gate)
        right_position = _gate_position(right_gate)
        position_delta = (
            None if left_position is None or right_position is None
            else _distance(left_position, right_position)
        )
        gate_pairs.append({
            "index": index,
            "left_gate_id": left_gate.get("gate_id"),
            "right_gate_id": right_gate.get("gate_id"),
            "same_gate_id": left_gate.get("gate_id") == right_gate.get("gate_id"),
            "position_delta_m": position_delta,
            "position_confidence_delta": (
                None
                if left_gate.get("position_confidence") is None
                or right_gate.get("position_confidence") is None
                else float(right_gate["position_confidence"]) - float(left_gate["position_confidence"])
            ),
        })
    position_deltas = [
        item["position_delta_m"]
        for item in gate_pairs
        if item["position_delta_m"] is not None
    ]
    gate_count_delta = len(right_gates) - len(left_gates)
    max_position_delta = max(position_deltas, default=None)
    changed = (
        left.get("frame_id") != right.get("frame_id")
        or left.get("sim_time_ns") != right.get("sim_time_ns")
        or left.get("source") != right.get("source")
        or gate_count_delta != 0
        or any(not item["same_gate_id"] for item in gate_pairs)
        or any((item["position_delta_m"] or 0.0) > 1e-9 for item in gate_pairs)
    )
    return {
        "same_status": True,
        "frame_identity_equal": (
            left.get("frame_id") == right.get("frame_id")
            and left.get("sim_time_ns") == right.get("sim_time_ns")
        ),
        "source_equal": left.get("source") == right.get("source"),
        "left_gate_count": len(left_gates),
        "right_gate_count": len(right_gates),
        "gate_count_delta": gate_count_delta,
        "max_position_delta_m": max_position_delta,
        "gate_pairs_by_index": gate_pairs,
        "changed": changed,
    }


def _resolve_run_dir(run: str | Path) -> Path:
    value = Path(run)
    candidates = [value]
    if not value.is_absolute():
        candidates.append(DEFAULT_FLIGHT_RUNS_ROOT / value)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise EvaluationError(f"flight run not found: {run}")


def compare_run(
    run: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    limit: int | None = None,
    start: int = 0,
    render_overlays: bool = False,
    overlay_layers: tuple[str, ...] = DEFAULT_OVERLAY_LAYERS,
    verbose: bool = True,
) -> Path:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if start < 0:
        raise ValueError("start must be zero or greater")

    def status(message: str) -> None:
        if verbose:
            print(f"[compare] {message}", flush=True)

    run_dir = _resolve_run_dir(run)
    run_id = run_dir.name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_root) / f"evaluation-{timestamp}-{run_id}"
    frames_output = output_dir / "frames.jsonl"
    observations_dir = output_dir / "observations"

    status(f"loading run: {run_dir}")
    records = list(_iter_frame_records(run_dir))
    selected = records[start:start + limit if limit is not None else None]
    states = _load_vehicle_states_by_frame_identity(run_dir)
    status(f"loaded {len(records)} frames; processing {len(selected)}")
    status(f"matched vehicle states for {len(states)} frame identities")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "metadata.json", {
        "evaluation_format_version": 1,
        "created_utc": timestamp,
        "source_run": {
            "run_id": run_id,
            "path": str(run_dir),
            "metadata": _read_json_if_present(run_dir / "metadata.json"),
            "summary": _read_json_if_present(run_dir / "summary.json"),
            "status": _read_json_if_present(run_dir / "status.json"),
        },
        "selection": {
            "total_frame_count": len(records),
            "start": start,
            "limit": limit,
            "selected_frame_count": len(selected),
        },
        "backends": list(BACKEND_FACTORIES),
        "overlays": {
            "enabled": render_overlays,
            "layers": list(overlay_layers),
            "root": "overlays" if render_overlays else None,
        },
        "output_files": {
            "frames": "frames.jsonl",
            "summary": "summary.json",
            "observations": {
                name: f"observations/{name}.jsonl"
                for name in BACKEND_FACTORIES
            },
        },
    })

    status("initializing Flight backends")
    backends = {
        name: factory()
        for name, factory in BACKEND_FACTORIES.items()
    }
    overlay_renderers = _load_overlay_renderers() if render_overlays else {}

    summary: dict[str, Any] = {
        "run_id": run_id,
        "processed_frame_count": 0,
        "frames_with_vehicle_state": 0,
        "frames_changed": 0,
        "frames_with_errors": 0,
        "left_backend": "deterministic_v3",
        "right_backend": "deterministic_v3_2",
        "backend_totals": {
            name: {
                "ok": 0,
                "error": 0,
                "total_gates": 0,
                "elapsed_ms_total": 0.0,
                "elapsed_ms_max": 0.0,
            }
            for name in BACKEND_FACTORIES
        },
        "gate_count_delta_histogram": {},
        "max_position_delta_m": 0.0,
        "overlay_totals": {
            name: {"ok": 0, "error": 0, "elapsed_ms_total": 0.0}
            for name in BACKEND_FACTORIES
        } if render_overlays else None,
    }

    for index, record in enumerate(selected, 1):
        identity = (record["frame_id"], record["sim_time_ns"])
        vehicle_state = states.get(identity)
        if vehicle_state is not None:
            summary["frames_with_vehicle_state"] += 1
        if verbose and (index == 1 or index == len(selected) or index % 25 == 0):
            status(
                f"frame {index}/{len(selected)} "
                f"frame_id={record['frame_id']} sim_time_ns={record['sim_time_ns']}"
            )

        backend_results = {
            name: _run_backend(backend, record, vehicle_state=vehicle_state)
            for name, backend in backends.items()
        }
        overlay_results = {}
        if render_overlays:
            for name, renderer in overlay_renderers.items():
                result = _render_backend_overlays(
                    name,
                    renderer["module"],
                    renderer["lut"],
                    record,
                    output_dir,
                    overlay_layers,
                    start + index - 1,
                    len(records),
                )
                overlay_results[name] = result
                totals = summary["overlay_totals"][name]
                totals[result["status"]] += 1
                totals["elapsed_ms_total"] += float(result.get("elapsed_ms", 0.0))
        for name, result in backend_results.items():
            totals = summary["backend_totals"][name]
            totals[result["status"]] += 1
            totals["elapsed_ms_total"] += float(result["elapsed_ms"])
            totals["elapsed_ms_max"] = max(
                float(totals["elapsed_ms_max"]), float(result["elapsed_ms"])
            )
            if result["status"] == "ok":
                totals["total_gates"] += len(result["observation"].get("gates", ()))
            _append_jsonl(observations_dir / f"{name}.jsonl", {
                "source": {
                    "run_id": run_id,
                    "frame_id": record["frame_id"],
                    "sim_time_ns": record["sim_time_ns"],
                    "relative_path": record["relative_path"],
                    "vehicle_state_available": vehicle_state is not None,
                },
                "backend": name,
                **result,
            })

        comparison = _compare_observations(
            backend_results["deterministic_v3"],
            backend_results["deterministic_v3_2"],
        )
        if comparison["changed"]:
            summary["frames_changed"] += 1
        if any(result["status"] != "ok" for result in backend_results.values()):
            summary["frames_with_errors"] += 1
        gate_delta = comparison.get("gate_count_delta")
        if gate_delta is not None:
            key = str(gate_delta)
            summary["gate_count_delta_histogram"][key] = (
                summary["gate_count_delta_histogram"].get(key, 0) + 1
            )
        if comparison.get("max_position_delta_m") is not None:
            summary["max_position_delta_m"] = max(
                float(summary["max_position_delta_m"]),
                float(comparison["max_position_delta_m"]),
            )
        summary["processed_frame_count"] += 1

        _append_jsonl(frames_output, {
            "source": {
                "run_id": run_id,
                "frame_index": start + index - 1,
                "frame_id": record["frame_id"],
                "sim_time_ns": record["sim_time_ns"],
                "elapsed_time_ns": record["elapsed_time_ns"],
                "jpeg_size": record["jpeg_size"],
                "relative_path": record["relative_path"],
                "vehicle_state_available": vehicle_state is not None,
                "vehicle_state": vehicle_state,
            },
            "backends": backend_results,
            "overlays": overlay_results,
            "comparison": comparison,
        })

    for totals in summary["backend_totals"].values():
        processed = max(1, summary["processed_frame_count"])
        totals["elapsed_ms_mean"] = totals["elapsed_ms_total"] / processed
    if render_overlays:
        for totals in summary["overlay_totals"].values():
            processed = max(1, summary["processed_frame_count"])
            totals["elapsed_ms_mean"] = totals["elapsed_ms_total"] / processed
    _write_json(output_dir / "summary.json", summary)
    status(f"wrote evaluation: {output_dir}")
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a Viewer/logs/flight run through Flight deterministic_v3 and "
            "deterministic_v3_2, then write comparison logs."
        )
    )
    parser.add_argument(
        "run",
        help=(
            "Run directory or run id under Viewer/logs/flight/runs, for example "
            "run-20260802T054152Z."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--render-overlays",
        action="store_true",
        help="Render backend review overlay PNGs for the evaluation UI.",
    )
    parser.add_argument(
        "--overlay-layer",
        action="append",
        dest="overlay_layers",
        help=(
            "Overlay layer to render; repeat for multiple layers. "
            "Use 'all' for every render_review_layers output. Defaults to "
            "the core visual overlay layers when --render-overlays is set."
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir = compare_run(
        args.run,
        args.output_root,
        start=args.start,
        limit=args.limit,
        render_overlays=args.render_overlays,
        overlay_layers=tuple(args.overlay_layers) if args.overlay_layers else DEFAULT_OVERLAY_LAYERS,
        verbose=not args.quiet,
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
