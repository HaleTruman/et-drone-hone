#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DEMO_ROOT / "src"
DEFAULT_PROFILE = DEMO_ROOT / "config" / "known_good_body_rate_guidance.json"


def _load_runtime() -> tuple[type[Any], type[Any]]:
    sys.path.insert(0, str(SRC_ROOT))
    from q1runtime import Q1Runtime, Q1RuntimeConfig

    return Q1Runtime, Q1RuntimeConfig


def _resolve_path(value: Any, *, allow_none: bool = True) -> Path | None:
    if value is None and allow_none:
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = DEMO_ROOT / path
    return path


def _load_profile(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Profile must contain a config object: {path}")
    return payload


def _build_config(config_type: type[Any], profile_config: dict[str, Any], args: argparse.Namespace) -> Any:
    known_fields = {field.name for field in fields(config_type)}
    config = {key: value for key, value in profile_config.items() if key in known_fields}

    if args.run_s is not None:
        config["run_s"] = float(args.run_s)
    if args.endpoint is not None:
        config["mavlink_endpoint"] = args.endpoint
    if args.device is not None:
        config["vision_device"] = args.device
    if args.log_dir is not None:
        config["log_dir"] = str(args.log_dir)
    if args.frame_output_dir is not None:
        config["frame_output_dir"] = str(args.frame_output_dir)
        config["save_raw_frames"] = True
    if args.no_save_raw_frames:
        config["save_raw_frames"] = False
        config["frame_output_dir"] = None
    if args.no_reset_on_start:
        config["reset_on_start"] = False
    if args.dry_run:
        config["dry_run"] = True
        config["arm_on_start"] = False

    config["log_dir"] = _resolve_path(config.get("log_dir") or "logs/q1runtime", allow_none=False)
    config["command_transform_path"] = _resolve_path(config.get("command_transform_path"))
    frame_output = config.get("frame_output_dir")
    if config.get("save_raw_frames") and frame_output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        frame_output = DEMO_ROOT / "logs" / "q1runtime" / "frames" / f"known-good-{stamp}"
    config["frame_output_dir"] = _resolve_path(frame_output)

    return config_type(**config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the self-contained AIGP Q1 known-good demo profile.")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--run-s", type=float, default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--frame-output-dir", type=Path, default=None)
    parser.add_argument("--no-save-raw-frames", action="store_true")
    parser.add_argument("--no-reset-on-start", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Q1Runtime, Q1RuntimeConfig = _load_runtime()
    profile_path = args.profile.expanduser()
    if not profile_path.is_absolute():
        profile_path = DEMO_ROOT / profile_path
    profile = _load_profile(profile_path)
    config = _build_config(Q1RuntimeConfig, profile["config"], args)

    print(f"aigpq1_demo profile={profile.get('profile')} source_run={profile.get('source_run')}")
    print(f"aigpq1_demo log_dir={config.log_dir}")
    print(f"aigpq1_demo frame_output_dir={config.frame_output_dir}")
    print(f"aigpq1_demo run_s={config.run_s} control_method={config.control_method}")
    Q1Runtime(config).run()


if __name__ == "__main__":
    main()
