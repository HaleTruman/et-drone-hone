"""Inspect live MAVLink and vision receiver traffic.

Run from the repository root with:
    python Flight/src/validation/mavlink_vision_receiver_view.py

The script listens to MAVLink and the UDP vision stream, saves MAVLink messages
as regular JSON, and prints grouped message counts plus latest parsed telemetry.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymavlink import mavutil

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sensing.telemetry import MavlinkClient  # noqa: E402
from sensing.vision import VisionStreamReceiver  # noqa: E402


DEFAULT_MAVLINK_ENDPOINT = "udpin:127.0.0.1:14550"
DEFAULT_VISION_HOST = "0.0.0.0"
DEFAULT_VISION_PORT = 5600
DEFAULT_OUTPUT_ROOT = SRC_DIR.parent / "logs" / "validation"


def main() -> int:
    args = _parse_args()
    output_dir = _resolve_output_dir(args.output_dir)
    frames_dir = output_dir / "frames"
    mavlink_json_path = output_dir / "mavlink_messages.json"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    mavlink_client = MavlinkClient(endpoint=args.mavlink_endpoint, sim_runtime=args.sim_runtime)
    vision_rx = VisionStreamReceiver(
        host=args.vision_host,
        port=args.vision_port,
        output_dir=frames_dir,
        max_buffered_frames=args.max_vision_frames,
    )

    print(f"Saving validation output: {output_dir}", flush=True)
    print(f"Saving MAVLink JSON: {mavlink_json_path}", flush=True)
    print(f"Saving vision frames: {frames_dir}", flush=True)
    print(f"Connecting MAVLink: {args.mavlink_endpoint}", flush=True)
    connection = mavutil.mavlink_connection(args.mavlink_endpoint)
    heartbeat = connection.wait_heartbeat(timeout=args.heartbeat_timeout_s)
    if heartbeat is None:
        raise TimeoutError(f"No MAVLink heartbeat received from {args.mavlink_endpoint}")

    mavlink_client._connection = connection
    mavlink_client.connected = True
    mavlink_client._on_heartbeat(heartbeat)
    print(
        "MAVLink heartbeat received "
        f"(system={connection.target_system}, component={connection.target_component})",
        flush=True,
    )

    if args.send_timesync:
        mavlink_client.start_heartbeat()

    vision_rx.start_listener()
    print(f"Vision receiver listening: {args.vision_host}:{args.vision_port}", flush=True)
    print("Press Ctrl+C to stop.\n", flush=True)

    message_counts: Counter[str] = Counter()
    latest_messages: dict[str, dict[str, Any]] = {}
    mavlink_records: list[dict[str, Any]] = []
    latest_frames: list[dict[str, Any]] = []
    started_s = time.perf_counter()
    next_summary_s = started_s

    try:
        _record_mavlink_message(
            heartbeat,
            message_counts=message_counts,
            latest_messages=latest_messages,
            mavlink_records=mavlink_records,
            started_s=started_s,
            verbose=args.verbose,
        )

        while args.duration_s is None or time.perf_counter() - started_s < args.duration_s:
            msg = connection.recv_match(blocking=False)
            if msg is not None and msg.get_type() != "BAD_DATA":
                _record_mavlink_message(
                    msg,
                    message_counts=message_counts,
                    latest_messages=latest_messages,
                    mavlink_records=mavlink_records,
                    started_s=started_s,
                    verbose=args.verbose,
                )
                mavlink_client.handle_message(msg)

            frame = vision_rx.get_next_frame()
            while frame is not None:
                record = {
                    "frame_id": frame.frame_id,
                    "sim_time_ns": frame.sim_time_ns,
                    "jpeg_size": len(frame.jpeg_bytes),
                    "saved_path": frame.saved_path,
                }
                latest_frames.append(record)
                latest_frames = latest_frames[-args.recent_frames :]
                if args.verbose:
                    print(f"[vision] {_compact_json(record)}", flush=True)
                frame = vision_rx.get_next_frame()

            now_s = time.perf_counter()
            if now_s >= next_summary_s:
                next_summary_s = now_s + args.summary_period_s
                _print_summary(
                    elapsed_s=now_s - started_s,
                    message_counts=message_counts,
                    latest_messages=latest_messages,
                    mavlink_snapshot=mavlink_client.snapshot(),
                    vision_snapshot=vision_rx.snapshot(),
                    latest_frames=latest_frames,
                    top_n=args.top_n,
                )

            time.sleep(args.poll_sleep_s)

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        _write_summary(
            summary_path,
            elapsed_s=time.perf_counter() - started_s,
            message_counts=message_counts,
            latest_messages=latest_messages,
            mavlink_snapshot=mavlink_client.snapshot(),
            vision_snapshot=vision_rx.snapshot(),
            latest_frames=latest_frames,
        )
        _write_mavlink_messages(mavlink_json_path, mavlink_records)
        vision_rx.shutdown()
        mavlink_client.shutdown()
        print(f"MAVLink messages saved: {mavlink_json_path}", flush=True)
        print(f"Summary saved: {summary_path}", flush=True)

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mavlink-endpoint", default=DEFAULT_MAVLINK_ENDPOINT)
    parser.add_argument("--sim-runtime", choices=("VQ_1", "VQ_2"), default="VQ_2")
    parser.add_argument("--vision-host", default=DEFAULT_VISION_HOST)
    parser.add_argument("--vision-port", type=int, default=DEFAULT_VISION_PORT)
    parser.add_argument("--heartbeat-timeout-s", type=float, default=10.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--summary-period-s", type=float, default=1.0)
    parser.add_argument("--poll-sleep-s", type=float, default=0.002)
    parser.add_argument("--top-n", type=int, default=16)
    parser.add_argument("--recent-frames", type=int, default=5)
    parser.add_argument("--max-vision-frames", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=None, help="Run output directory. Defaults to Flight/logs/validation/<timestamp>.")
    parser.add_argument("--send-timesync", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (DEFAULT_OUTPUT_ROOT / f"mavlink_vision-{timestamp}").resolve()


def _record_mavlink_message(
    msg: Any,
    *,
    message_counts: Counter[str],
    latest_messages: dict[str, dict[str, Any]],
    mavlink_records: list[dict[str, Any]],
    started_s: float,
    verbose: bool,
) -> None:
    msg_type = msg.get_type()
    message_counts[msg_type] += 1
    summary = _message_summary(msg)
    latest_messages[msg_type] = summary
    record = {
        "wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.perf_counter() - started_s, 6),
        "type": msg_type,
        "count_for_type": message_counts[msg_type],
        "message": summary,
    }
    mavlink_records.append(record)
    if verbose:
        print(f"[mavlink] {msg_type}: {_compact_json(summary)}", flush=True)


def _print_summary(
    *,
    elapsed_s: float,
    message_counts: Counter[str],
    latest_messages: dict[str, dict[str, Any]],
    mavlink_snapshot: dict[str, Any],
    vision_snapshot: dict[str, Any],
    latest_frames: list[dict[str, Any]],
    top_n: int,
) -> None:
    print("\n" + "=" * 96)
    print(f"Elapsed: {elapsed_s:8.2f}s")
    print("\nMAVLink message counts")
    print(f"{'type':<30} {'count':>10} latest")
    print("-" * 96)
    for msg_type, count in message_counts.most_common(top_n):
        print(f"{msg_type:<30} {count:>10} {_compact_json(latest_messages.get(msg_type, {}))}")
    if not message_counts:
        print("(no MAVLink messages yet)")

    print("\nParsed MAVLink cache")
    parsed = {
        "armed": mavlink_snapshot.get("armed"),
        "latest_heartbeat": mavlink_snapshot.get("latest_heartbeat"),
        "latest_timesync": mavlink_snapshot.get("latest_timesync"),
        "latest_imu": mavlink_snapshot.get("latest_imu"),
        "latest_actuator_output": mavlink_snapshot.get("latest_actuator_output"),
        "latest_sim_truth": mavlink_snapshot.get("latest_sim_truth"),
        "race_status": mavlink_snapshot.get("race_status"),
        "collisions": mavlink_snapshot.get("collisions"),
    }
    print(_pretty_json(parsed))

    print("\nVision receiver")
    print(_pretty_json(vision_snapshot))
    print("Recent frames")
    if latest_frames:
        for frame in latest_frames:
            print(_compact_json(frame))
    else:
        print("(no complete vision frames yet)")


def _message_summary(msg: Any) -> dict[str, Any]:
    payload = msg.to_dict()
    payload.pop("mavpackettype", None)
    return _truncate_nested(payload, max_items=12)


def _write_summary(
    path: Path,
    *,
    elapsed_s: float,
    message_counts: Counter[str],
    latest_messages: dict[str, dict[str, Any]],
    mavlink_snapshot: dict[str, Any],
    vision_snapshot: dict[str, Any],
    latest_frames: list[dict[str, Any]],
) -> None:
    payload = {
        "elapsed_s": round(elapsed_s, 6),
        "message_counts": dict(message_counts.most_common()),
        "latest_messages": latest_messages,
        "mavlink": mavlink_snapshot,
        "vision": vision_snapshot,
        "latest_frames": latest_frames,
    }
    path.write_text(_pretty_json(payload) + "\n", encoding="utf-8")


def _write_mavlink_messages(path: Path, records: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "message_count": len(records),
        "messages": records,
    }
    path.write_text(_pretty_json(payload) + "\n", encoding="utf-8")


def _truncate_nested(value: Any, *, max_items: int) -> Any:
    if is_dataclass(value):
        return _truncate_nested(asdict(value), max_items=max_items)
    if isinstance(value, dict):
        return {
            key: _truncate_nested(item, max_items=max_items)
            for key, item in list(value.items())[:max_items]
        }
    if isinstance(value, (list, tuple)):
        items = list(value)
        truncated = [_truncate_nested(item, max_items=max_items) for item in items[:max_items]]
        if len(items) > max_items:
            truncated.append(f"... {len(items) - max_items} more")
        return truncated
    if isinstance(value, float):
        return round(value, 6)
    return value


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
