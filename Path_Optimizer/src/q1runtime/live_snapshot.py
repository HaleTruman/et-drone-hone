from __future__ import annotations

import argparse
import json
import math
import select
import socket
import struct
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from sensing.vision.io.udp_protocol import VISION_HEADER_SIZE


@dataclass(frozen=True)
class LiveSnapshotConfig:
    """Receive-only telemetry and vision snapshot configuration."""

    telemetry_host: str = "0.0.0.0"
    telemetry_port: int = 14550
    vision_host: str = "0.0.0.0"
    vision_port: int = 5600
    seconds: float = 3.0
    timeline_step_s: float = 0.5
    include_vision: bool = True
    save_path: Path | None = None


def capture_snapshot(config: LiveSnapshotConfig) -> dict[str, Any]:
    started_wall = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    sockets: list[tuple[str, int, socket.socket]] = []
    bind_errors: dict[str, str] = {}
    mav_counts: Counter[str] = Counter()
    packet_counts: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    telemetry_samples: list[dict[str, Any]] = []
    attitude_samples: list[dict[str, Any]] = []
    actuator_samples: list[dict[str, Any]] = []
    heartbeat_samples: list[dict[str, Any]] = []
    imu_samples: list[dict[str, Any]] = []
    vision_frames: dict[int, dict[str, Any]] = {}
    vision_invalid = 0

    telemetry_socket = _bind_udp(config.telemetry_host, config.telemetry_port, bind_errors, "telemetry")
    if telemetry_socket is not None:
        sockets.append(("telemetry", config.telemetry_port, telemetry_socket))
    if config.include_vision:
        vision_socket = _bind_udp(config.vision_host, config.vision_port, bind_errors, "vision")
        if vision_socket is not None:
            sockets.append(("vision", config.vision_port, vision_socket))

    parser = _mavlink_parser() if telemetry_socket is not None else None
    try:
        deadline = started_monotonic + max(0.0, float(config.seconds))
        while time.monotonic() < deadline and sockets:
            readable, _, _ = select.select([item[2] for item in sockets], [], [], 0.05)
            for sock in readable:
                stream_name, _, _ = next(item for item in sockets if item[2] is sock)
                data, addr = sock.recvfrom(65536)
                packet_counts[stream_name] += 1
                sources[stream_name][f"{addr[0]}:{addr[1]}"] += 1
                t_plus = time.monotonic() - started_monotonic
                if stream_name == "telemetry" and parser is not None:
                    _parse_mavlink_packet(
                        parser,
                        data,
                        t_plus_s=t_plus,
                        counts=mav_counts,
                        telemetry_samples=telemetry_samples,
                        attitude_samples=attitude_samples,
                        actuator_samples=actuator_samples,
                        heartbeat_samples=heartbeat_samples,
                        imu_samples=imu_samples,
                    )
                elif stream_name == "vision":
                    if not _parse_vision_packet(data, t_plus_s=t_plus, frames=vision_frames):
                        vision_invalid += 1
    finally:
        for _, _, sock in sockets:
            sock.close()

    ended_wall = datetime.now().astimezone()
    summary = summarize_snapshot(
        started_wall=started_wall,
        ended_wall=ended_wall,
        duration_s=time.monotonic() - started_monotonic,
        config=config,
        bind_errors=bind_errors,
        packet_counts=packet_counts,
        sources=sources,
        mav_counts=mav_counts,
        telemetry_samples=telemetry_samples,
        attitude_samples=attitude_samples,
        actuator_samples=actuator_samples,
        heartbeat_samples=heartbeat_samples,
        imu_samples=imu_samples,
        vision_frames=vision_frames,
        vision_invalid=vision_invalid,
    )
    if config.save_path is not None:
        config.save_path.parent.mkdir(parents=True, exist_ok=True)
        config.save_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def summarize_snapshot(
    *,
    started_wall: datetime,
    ended_wall: datetime,
    duration_s: float,
    config: LiveSnapshotConfig,
    bind_errors: dict[str, str],
    packet_counts: Counter[str],
    sources: dict[str, Counter[str]],
    mav_counts: Counter[str],
    telemetry_samples: list[dict[str, Any]],
    attitude_samples: list[dict[str, Any]],
    actuator_samples: list[dict[str, Any]],
    heartbeat_samples: list[dict[str, Any]],
    imu_samples: list[dict[str, Any]],
    vision_frames: dict[int, dict[str, Any]],
    vision_invalid: int,
) -> dict[str, Any]:
    motion = analyze_motion(telemetry_samples, attitude_samples, actuator_samples, imu_samples)
    return {
        "snapshot_started_wall": started_wall.isoformat(timespec="milliseconds"),
        "snapshot_ended_wall": ended_wall.isoformat(timespec="milliseconds"),
        "duration_s": round(float(duration_s), 3),
        "relative_time_origin": "snapshot_started_wall",
        "observer": "q1runtime.live_snapshot",
        "receive_only": True,
        "ports": {
            "telemetry": {"host": config.telemetry_host, "port": config.telemetry_port},
            "vision": {"host": config.vision_host, "port": config.vision_port, "included": config.include_vision},
        },
        "bind_errors": bind_errors,
        "packet_counts": dict(packet_counts),
        "sources": {name: dict(counter) for name, counter in sources.items()},
        "mavlink_message_counts": dict(mav_counts),
        "telemetry_sample_count": len(telemetry_samples),
        "motion": motion,
        "timeline": build_timeline(
            telemetry_samples,
            attitude_samples,
            actuator_samples,
            step_s=config.timeline_step_s,
            duration_s=duration_s,
        ),
        "vision": summarize_vision(vision_frames, invalid_count=vision_invalid),
        "latest_heartbeat": heartbeat_samples[-1] if heartbeat_samples else None,
    }


def analyze_motion(
    telemetry_samples: list[dict[str, Any]],
    attitude_samples: list[dict[str, Any]],
    actuator_samples: list[dict[str, Any]],
    imu_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if not telemetry_samples:
        return {
            "state": "no_telemetry",
            "description": "No LOCAL_POSITION_NED samples were received, so movement cannot be interpreted.",
        }

    first = telemetry_samples[0]
    last = telemetry_samples[-1]
    displacement = [last["position_local_ned_m"][index] - first["position_local_ned_m"][index] for index in range(3)]
    horizontal = math.hypot(displacement[0], displacement[1])
    net = math.sqrt(sum(value * value for value in displacement))
    speeds = [_norm(sample["velocity_local_ned_mps"]) for sample in telemetry_samples]
    z_delta = displacement[2]
    z_direction = "descending in NED (+z)" if z_delta > 0.05 else "climbing in NED (-z)" if z_delta < -0.05 else "holding z"
    avg_speed = sum(speeds) / len(speeds)
    max_speed = max(speeds)
    attitude_deg = [_attitude_deg(sample) for sample in attitude_samples]
    max_abs_roll = max((abs(sample["roll_deg"]) for sample in attitude_deg), default=0.0)
    max_abs_pitch = max((abs(sample["pitch_deg"]) for sample in attitude_deg), default=0.0)

    if net < 0.25 and max_speed < 0.5:
        state = "stationary_or_hover_like"
        description = "Drone is nearly stationary over this window."
    elif max_speed >= 8.0 or abs(z_delta) >= 5.0 or horizontal >= 5.0:
        state = "high_energy_motion"
        description = (
            f"Drone is moving fast: {net:.1f} m net, {horizontal:.1f} m horizontal, "
            f"{z_delta:.1f} m local-NED z over the sample; {z_direction}."
        )
    else:
        state = "controlled_or_moderate_motion"
        description = (
            f"Drone is moving moderately: {net:.1f} m net, {horizontal:.1f} m horizontal, "
            f"{z_delta:.1f} m local-NED z; {z_direction}."
        )

    if max_abs_roll >= 30.0 or max_abs_pitch >= 30.0:
        description += f" Attitude excursions are large: roll up to {max_abs_roll:.1f} deg, pitch up to {max_abs_pitch:.1f} deg."

    return {
        "state": state,
        "description": description,
        "start_position_local_ned_m": _round_list(first["position_local_ned_m"], 3),
        "end_position_local_ned_m": _round_list(last["position_local_ned_m"], 3),
        "displacement_local_ned_m": _round_list(displacement, 3),
        "horizontal_displacement_m": round(horizontal, 3),
        "net_displacement_m": round(net, 3),
        "average_speed_mps": round(avg_speed, 3),
        "speed_minmax_mps": [round(min(speeds), 3), round(max_speed, 3)],
        "z_direction": z_direction,
        "max_abs_roll_deg": round(max_abs_roll, 3),
        "max_abs_pitch_deg": round(max_abs_pitch, 3),
        "actuator_first4_minmax": _actuator_minmax(actuator_samples),
        "latest_accel_mps2": _round_list(imu_samples[-1]["acceleration_local_ned_mps2"], 3) if imu_samples else None,
    }


def build_timeline(
    telemetry_samples: list[dict[str, Any]],
    attitude_samples: list[dict[str, Any]],
    actuator_samples: list[dict[str, Any]],
    *,
    step_s: float,
    duration_s: float,
) -> list[dict[str, Any]]:
    if not telemetry_samples:
        return []
    first = telemetry_samples[0]
    step = max(0.1, float(step_s))
    targets: list[float] = []
    current = 0.0
    while current < duration_s + 1e-9:
        targets.append(current)
        current += step
    if not targets or targets[-1] < duration_s:
        targets.append(duration_s)

    rows = []
    for target_t in targets:
        telemetry = _latest_at(telemetry_samples, max(target_t, first["t_plus_s"]))
        if telemetry is None:
            continue
        attitude = _latest_at(attitude_samples, telemetry["t_plus_s"])
        actuator = _latest_at(actuator_samples, telemetry["t_plus_s"])
        displacement = [
            telemetry["position_local_ned_m"][index] - first["position_local_ned_m"][index] for index in range(3)
        ]
        row: dict[str, Any] = {
            "t_plus_s": round(telemetry["t_plus_s"], 3),
            "position_local_ned_m": _round_list(telemetry["position_local_ned_m"], 3),
            "displacement_from_start_m": _round_list(displacement, 3),
            "velocity_local_ned_mps": _round_list(telemetry["velocity_local_ned_mps"], 3),
            "speed_mps": round(_norm(telemetry["velocity_local_ned_mps"]), 3),
        }
        if attitude is not None:
            row["attitude_deg"] = _attitude_deg(attitude)
        if actuator is not None:
            row["actuator_first4"] = _round_list(actuator["actuator_first4"], 3)
        rows.append(row)
    return rows


def summarize_vision(vision_frames: dict[int, dict[str, Any]], *, invalid_count: int) -> dict[str, Any]:
    complete = 0
    packet_total = 0
    chunk_total = 0
    sample = []
    for frame_id in sorted(vision_frames):
        frame = vision_frames[frame_id]
        unique_chunks = len(frame["chunks"])
        total_chunks = int(frame["total_chunks"])
        if unique_chunks == total_chunks:
            complete += 1
        packet_total += int(frame["packet_count"])
        chunk_total += unique_chunks
        if len(sample) < 5:
            sample.append(
                {
                    "frame_id": frame_id,
                    "t_plus_first_seen_s": round(float(frame["first_seen_t_plus_s"]), 3),
                    "total_chunks": total_chunks,
                    "unique_chunks": unique_chunks,
                    "packet_count": int(frame["packet_count"]),
                    "jpeg_size": int(frame["jpeg_size"]),
                    "sim_time_ns": int(frame["sim_time_ns"]),
                }
            )
    return {
        "unique_frames": len(vision_frames),
        "complete_frames": complete,
        "invalid_packets": invalid_count,
        "packet_count": packet_total,
        "unique_chunk_count": chunk_total,
        "duplicate_packet_estimate": max(0, packet_total - chunk_total),
        "sample_frames": sample,
    }


def format_snapshot_text(summary: dict[str, Any]) -> str:
    motion = summary["motion"]
    lines = [
        f"Snapshot: {summary['snapshot_started_wall']} to {summary['snapshot_ended_wall']}",
        f"Window: {summary['duration_s']}s, relative origin = snapshot start",
        f"Receive-only: {summary['receive_only']}",
        "",
        f"Behavior: {motion['state']}",
        motion["description"],
        "",
        "Streams:",
        f"  packets={summary['packet_counts']}",
        f"  sources={summary['sources']}",
        f"  mavlink={summary['mavlink_message_counts']}",
        f"  vision={summary['vision']}",
    ]
    if summary["bind_errors"]:
        lines.extend(["", f"Bind errors: {summary['bind_errors']}"])
    lines.extend(["", "Timeline:"])
    for row in summary["timeline"]:
        attitude = row.get("attitude_deg")
        attitude_text = (
            f" roll={attitude['roll_deg']:+.1f} pitch={attitude['pitch_deg']:+.1f} yaw={attitude['yaw_deg']:+.1f}"
            if attitude
            else ""
        )
        lines.append(
            "  "
            f"t+{row['t_plus_s']:.3f}s "
            f"pos={row['position_local_ned_m']} "
            f"disp={row['displacement_from_start_m']} "
            f"vel={row['velocity_local_ned_mps']} "
            f"speed={row['speed_mps']:.3f}m/s"
            f"{attitude_text}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive-only live simulator telemetry/vision snapshot.")
    parser.add_argument("--seconds", type=float, default=LiveSnapshotConfig.seconds)
    parser.add_argument("--timeline-step", type=float, default=LiveSnapshotConfig.timeline_step_s)
    parser.add_argument("--telemetry-host", default=LiveSnapshotConfig.telemetry_host)
    parser.add_argument("--telemetry-port", type=int, default=LiveSnapshotConfig.telemetry_port)
    parser.add_argument("--vision-host", default=LiveSnapshotConfig.vision_host)
    parser.add_argument("--vision-port", type=int, default=LiveSnapshotConfig.vision_port)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = capture_snapshot(
        LiveSnapshotConfig(
            telemetry_host=args.telemetry_host,
            telemetry_port=args.telemetry_port,
            vision_host=args.vision_host,
            vision_port=args.vision_port,
            seconds=args.seconds,
            timeline_step_s=args.timeline_step,
            include_vision=not args.no_vision,
            save_path=args.save,
        )
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(format_snapshot_text(summary))


def _bind_udp(host: str, port: int, bind_errors: dict[str, str], name: str) -> socket.socket | None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        sock.bind((host, int(port)))
        sock.setblocking(False)
        return sock
    except OSError as exc:
        bind_errors[name] = str(exc)
        return None


def _mavlink_parser() -> Any:
    from pymavlink.dialects.v20 import common as mavlink2

    return mavlink2.MAVLink(None)


def _parse_mavlink_packet(
    parser: Any,
    data: bytes,
    *,
    t_plus_s: float,
    counts: Counter[str],
    telemetry_samples: list[dict[str, Any]],
    attitude_samples: list[dict[str, Any]],
    actuator_samples: list[dict[str, Any]],
    heartbeat_samples: list[dict[str, Any]],
    imu_samples: list[dict[str, Any]],
) -> None:
    for byte in data:
        try:
            message = parser.parse_char(bytes([byte]))
        except Exception:
            continue
        if message is None:
            continue
        message_type = message.get_type()
        counts[message_type] += 1
        if message_type == "LOCAL_POSITION_NED":
            telemetry_samples.append(
                {
                    "t_plus_s": float(t_plus_s),
                    "position_local_ned_m": [float(message.x), float(message.y), float(message.z)],
                    "velocity_local_ned_mps": [float(message.vx), float(message.vy), float(message.vz)],
                }
            )
        elif message_type == "ATTITUDE":
            attitude_samples.append(
                {
                    "t_plus_s": float(t_plus_s),
                    "roll_rad": float(message.roll),
                    "pitch_rad": float(message.pitch),
                    "yaw_rad": float(message.yaw),
                }
            )
        elif message_type == "ACTUATOR_OUTPUT_STATUS":
            actuator_samples.append(
                {
                    "t_plus_s": float(t_plus_s),
                    "actuator_first4": [float(value) for value in message.actuator[:4]],
                }
            )
        elif message_type == "HEARTBEAT":
            heartbeat_samples.append(
                {
                    "t_plus_s": float(t_plus_s),
                    "base_mode": int(message.base_mode),
                    "system_status": int(message.system_status),
                    "mav_type": int(message.type),
                    "autopilot": int(message.autopilot),
                }
            )
        elif message_type == "HIGHRES_IMU":
            imu_samples.append(
                {
                    "t_plus_s": float(t_plus_s),
                    "acceleration_local_ned_mps2": [float(message.xacc), float(message.yacc), float(message.zacc)],
                }
            )


def _parse_vision_packet(data: bytes, *, t_plus_s: float, frames: dict[int, dict[str, Any]]) -> bool:
    if len(data) < VISION_HEADER_SIZE:
        return False
    try:
        frame_id, chunk_id, total_chunks, jpeg_size, payload_size, sim_time_ns = struct.unpack_from("<IHHIIQ", data)
    except struct.error:
        return False
    if total_chunks <= 0 or chunk_id >= total_chunks or payload_size != len(data) - VISION_HEADER_SIZE:
        return False
    frame = frames.setdefault(
        int(frame_id),
        {
            "first_seen_t_plus_s": float(t_plus_s),
            "total_chunks": int(total_chunks),
            "jpeg_size": int(jpeg_size),
            "sim_time_ns": int(sim_time_ns),
            "chunks": set(),
            "packet_count": 0,
        },
    )
    frame["chunks"].add(int(chunk_id))
    frame["packet_count"] += 1
    return True


def _latest_at(samples: list[dict[str, Any]], t_plus_s: float) -> dict[str, Any] | None:
    latest = None
    for sample in samples:
        if sample["t_plus_s"] <= t_plus_s + 1e-9:
            latest = sample
        else:
            break
    return latest


def _attitude_deg(sample: dict[str, Any]) -> dict[str, float]:
    return {
        "roll_deg": round(float(sample["roll_rad"]) * 180.0 / math.pi, 1),
        "pitch_deg": round(float(sample["pitch_rad"]) * 180.0 / math.pi, 1),
        "yaw_deg": round(float(sample["yaw_rad"]) * 180.0 / math.pi, 1),
    }


def _actuator_minmax(samples: list[dict[str, Any]]) -> list[list[float]] | None:
    if not samples:
        return None
    rows = [sample["actuator_first4"] for sample in samples]
    return [[round(min(row[index] for row in rows), 3), round(max(row[index] for row in rows), 3)] for index in range(4)]


def _norm(values: list[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _round_list(values: list[float], digits: int) -> list[float]:
    return [round(float(value), digits) for value in values]


if __name__ == "__main__":
    main()
