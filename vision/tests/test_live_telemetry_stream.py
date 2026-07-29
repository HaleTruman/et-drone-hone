from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mask_review_pipeline" / "src"
sys.path.insert(0, str(SRC))

from camera_odometry import MAX_TELEMETRY_GAP_S
from deterministic_vision import DeterministicVision
from schema import VehicleState, VisionFrame

RUN_DIR = ROOT / "runs" / "run-20260724T165437Z"
FRAME_SLICE = 30  # enough real frames to earn VOID_EARN_STREAK reliability and see real gates


def _load_vehicle_states(telemetry_path: Path) -> list[VehicleState]:
    data = json.loads(telemetry_path.read_text())
    states = []
    for sample in data["samples"]:
        vs = sample["telemetry"]["vehicle_state"]
        # Real-data finding: telemetry.json's own vehicle_state.sim_time_ns is on a different,
        # internal clock than VisionFrame.sim_time_ns/frames.jsonl (which is real wall-clock
        # epoch ns). wall_time_utc *is* on that same epoch, so it's what a live VehicleState's
        # sim_time_ns is expected to actually carry -- not the recorded file's internal field.
        epoch_ns = int(datetime.fromisoformat(sample["wall_time_utc"]).timestamp() * 1e9)
        states.append(VehicleState(
            sim_time_ns=epoch_ns,
            position_local_ned_m=tuple(vs["position_local_ned_m"]),
            velocity_local_ned_mps=tuple(vs["velocity_local_ned_mps"]),
            attitude_quaternion=tuple(vs["attitude_quaternion"]),
            body_rates_frd_rps=tuple(vs["body_rates_frd_rps"]),
            acceleration_local_ned_mps2=tuple(vs["acceleration_local_ned_mps2"]),
        ))
    return states


def _nearest_vehicle_state(states: list[VehicleState], sim_time_ns: int) -> VehicleState | None:
    best = min(states, key=lambda s: abs(s.sim_time_ns - sim_time_ns))
    return best if abs(best.sim_time_ns - sim_time_ns) <= MAX_TELEMETRY_GAP_S * 1e9 else None


def test_live_stream_with_real_telemetry_produces_real_gates() -> None:
    states = _load_vehicle_states(RUN_DIR / "telemetry.json")
    lo = min(s.sim_time_ns for s in states)
    hi = max(s.sim_time_ns for s in states)

    frame_records = [json.loads(line) for line in (RUN_DIR / "frames.jsonl").read_text().splitlines()]
    covered = [r for r in frame_records if lo <= r["sim_time_ns"] <= hi]
    covered.sort(key=lambda r: r["sim_time_ns"])
    assert len(covered) >= FRAME_SLICE, "expected enough telemetry-covered frames for this test"

    vision = DeterministicVision()
    saw_real_gates = False
    for record in covered[:FRAME_SLICE]:
        vehicle_state = _nearest_vehicle_state(states, record["sim_time_ns"])
        frame = VisionFrame(
            frame_id=record["frame_id"],
            sim_time_ns=record["sim_time_ns"],
            jpeg_bytes=(RUN_DIR / record["path"]).read_bytes(),
        )
        observation = vision.process_frame(frame, vehicle_state)
        if observation.gates:
            saw_real_gates = True
            for gate in observation.gates:
                assert len(gate.position_camera_m) == 3
                assert all(v == v and abs(v) != float("inf") for v in gate.position_camera_m)  # finite

    assert saw_real_gates, "expected at least one real gate once real vehicle_state was supplied"


def test_missing_or_stale_vehicle_state_degrades_gracefully() -> None:
    frame_records = [json.loads(line) for line in (RUN_DIR / "frames.jsonl").read_text().splitlines()]
    record = frame_records[0]
    frame = VisionFrame(
        frame_id=record["frame_id"],
        sim_time_ns=record["sim_time_ns"],
        jpeg_bytes=(RUN_DIR / record["path"]).read_bytes(),
    )

    vision = DeterministicVision()
    observation = vision.process_frame(frame, None)
    assert observation.gates == []

    stale_state = VehicleState(
        sim_time_ns=record["sim_time_ns"] - int((MAX_TELEMETRY_GAP_S + 1.0) * 1e9),
        position_local_ned_m=(0.0, 0.0, 0.0),
        velocity_local_ned_mps=(0.0, 0.0, 0.0),
        attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
        body_rates_frd_rps=(0.0, 0.0, 0.0),
        acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
    )
    observation = vision.process_frame(frame, stale_state)
    assert observation.gates == []


if __name__ == "__main__":
    test_live_stream_with_real_telemetry_produces_real_gates()
    test_missing_or_stale_vehicle_state_degrades_gracefully()
    print("test_live_telemetry_stream.py passed")
