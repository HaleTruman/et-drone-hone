#!/usr/bin/env python3
"""Emit a Flight-compatible controller payload from a 0721Vision frame JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = APP_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flight_bridge.vision_observation import VisionObservation  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_json", type=Path, help="0721Vision per-frame instance JSON or latest.json")
    parser.add_argument("--sim-time-ns", type=int, default=None, help="optional simulator timestamp to attach")
    parser.add_argument("--output-dir", default="memory", help="controller payload output_dir field")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    observation = VisionObservation.from_instance_frame_path(args.frame_json, sim_time_ns=args.sim_time_ns)
    print(json.dumps(observation.to_controller_payload(output_dir=args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
