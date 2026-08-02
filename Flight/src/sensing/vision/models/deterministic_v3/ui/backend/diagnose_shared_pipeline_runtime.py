"""Measure the implemented shared pipeline by module on historic frames.

All timing and reporting stay outside production ``src``.  The density stage
materializes every review profile for each density-eligible component; it is
intentionally not a claim about the future one-profile live density cost.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter, perf_counter_ns

import cv2
import numpy as np

from ...src.configurations import DEFAULT_DENSITY_CONFIGURATION
from ...src.density_bank import DensityBank
from ...src.preprocessing import (
    DEFAULT_CONFIG, decode_jpeg, load_lut, preprocess_frame)
from ...src.topology import assess_frame
from .replay_historic_run import (
    DEFAULT_OUTPUT_ROOT, DEFAULT_SOURCE_RUN, HistoricFrameRecord,
    HistoricRunSource)
from .schema_json import write_json


REPORT_VERSION = "deterministic-v3.shared-runtime-diagnostics.v2"
DEFAULT_LIMITS = Path(__file__).with_name("shared_pipeline_runtime_limits.json")
PROFILE_STAGES = tuple(
    profile.profile_id for profile in DEFAULT_DENSITY_CONFIGURATION.profiles)
STAGES = (
    "jpeg_read",
    "decode_jpeg",
    "preprocessing",
    "topology",
    *(f"density_{profile}" for profile in PROFILE_STAGES),
    "density_bank_all_profiles",
    "total_shared_compute",
    "total_review_wall",
)


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def summarize_samples(samples: list[float]) -> dict:
    """Return per-frame latency distribution and reciprocal mean throughput."""
    values = np.asarray(samples, np.float64)
    if values.size == 0:
        raise ValueError("timing samples cannot be empty")
    mean_ms = float(values.mean())
    return {
        "frames": int(values.size),
        "mean_ms_per_frame": mean_ms,
        "median_ms_per_frame": float(np.median(values)),
        "p95_ms_per_frame": float(np.percentile(values, 95)),
        "max_ms_per_frame": float(values.max()),
        "effective_hz": None if mean_ms <= 0 else 1000.0 / mean_ms,
    }


def measure_frame(
    record: HistoricFrameRecord, lut: np.ndarray
) -> dict:
    """Time each current shared module once for a single recorded frame."""
    wall_started = perf_counter_ns()
    read_started = perf_counter_ns()
    pipeline_input = record.pipeline_input()
    read_ms = _elapsed_ms(read_started)

    compute_started = perf_counter_ns()
    decode_started = perf_counter_ns()
    image = decode_jpeg(pipeline_input["jpeg_bytes"])
    decode_ms = _elapsed_ms(decode_started)

    preprocessing_started = perf_counter_ns()
    frame = preprocess_frame(
        frame_id=pipeline_input["frame_id"],
        sim_time_ns=pipeline_input["sim_time_ns"],
        image=image,
        lut=lut,
    )
    preprocessing_ms = _elapsed_ms(preprocessing_started)

    bank = DensityBank(frame)
    topology_started = perf_counter_ns()
    decisions = assess_frame(frame, density_bank=bank)
    topology_ms = _elapsed_ms(topology_started)

    density_started = perf_counter_ns()
    eligible_components = tuple(
        component for component in frame.components
        if bank.includes_component(component))
    ignored_components = tuple(
        component for component in frame.components
        if not bank.includes_component(component))
    profile_ms = {profile: 0.0 for profile in PROFILE_STAGES}
    density_records = 0
    for component in eligible_components:
        for profile in PROFILE_STAGES:
            profile_started = perf_counter_ns()
            bank.get(component, profile)
            profile_ms[profile] += _elapsed_ms(profile_started)
            density_records += 1
    density_ms = _elapsed_ms(density_started)
    compute_ms = _elapsed_ms(compute_started)
    wall_ms = _elapsed_ms(wall_started)

    stages = {
        "jpeg_read": read_ms,
        "decode_jpeg": decode_ms,
        "preprocessing": preprocessing_ms,
        "topology": topology_ms,
        **{f"density_{profile}": profile_ms[profile]
           for profile in PROFILE_STAGES},
        "density_bank_all_profiles": density_ms,
        "total_shared_compute": compute_ms,
        "total_review_wall": wall_ms,
    }
    return {
        "frame_id": record.frame_id,
        "sim_time_ns": record.sim_time_ns,
        "component_count": len(frame.components),
        "topology_decision_count": len(decisions),
        "density_eligible_component_count": len(eligible_components),
        "density_ignored_component_count": len(ignored_components),
        "density_ignored_component_ids": [
            component.component_id for component in ignored_components],
        "density_record_count": density_records,
        "gated": frame.gated,
        "stages_ms": stages,
    }


def run_diagnostics(
    records: tuple[HistoricFrameRecord, ...],
    lut: np.ndarray,
    *,
    warmup_frames: int = 5,
) -> dict:
    """Warm OpenCV, measure every supplied frame, and summarize each stage."""
    if not records:
        raise ValueError("at least one frame is required")
    if warmup_frames < 0:
        raise ValueError("warmup_frames cannot be negative")
    for record in records[:warmup_frames]:
        measure_frame(record, lut)

    wall_started = perf_counter()
    frames = [measure_frame(record, lut) for record in records]
    benchmark_wall_s = perf_counter() - wall_started
    summary = {
        stage: summarize_samples([
            frame["stages_ms"][stage] for frame in frames])
        for stage in STAGES
    }
    summary["benchmark_loop"] = {
        "frames": len(frames),
        "mean_ms_per_frame": benchmark_wall_s * 1000.0 / len(frames),
        "median_ms_per_frame": None,
        "p95_ms_per_frame": None,
        "max_ms_per_frame": None,
        "effective_hz": len(frames) / benchmark_wall_s,
    }
    return {
        "report_version": REPORT_VERSION,
        "run_id": records[0].run_id,
        "scope": {
            "pipeline_frontier": "shared_density_evidence",
            "density_mode":
                "all_configured_profiles_for_density_eligible_components",
            "density_profile_count": len(PROFILE_STAGES),
            "density_profile_ids": list(PROFILE_STAGES),
            "ignore_frame_edge_clipped_for_density":
                DEFAULT_DENSITY_CONFIGURATION.ignore_frame_edge_clipped,
            "maximum_input_components":
                DEFAULT_CONFIG.maximum_input_components,
            "lut_load_included": False,
            "json_serialization_included": False,
            "warmup_frames": min(warmup_frames, len(records)),
            "measured_frames": len(records),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "opencv_threads": cv2.getNumThreads(),
            "numpy": np.__version__,
        },
        "summary": summary,
        "frames": frames,
    }


def load_limits(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        limits = json.load(stream)
    if limits.get("metric") != "p95_ms_per_frame":
        raise ValueError("runtime limits must use p95_ms_per_frame")
    return limits


def validate_limits(report: dict, limits: dict) -> dict:
    """Compare stage P95 values with explicit review-machine ceilings."""
    checks = []
    for stage, maximum in limits["maximum_ms"].items():
        measured = report["summary"][stage][limits["metric"]]
        passed = measured is not None and measured <= float(maximum)
        checks.append({
            "stage": stage,
            "metric": limits["metric"],
            "measured_ms": measured,
            "maximum_ms": float(maximum),
            "passed": passed,
        })
    return {
        "limits_version": limits["version"],
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def _print_summary(report: dict) -> None:
    print("stage                              mean ms   p95 ms   max ms        Hz")
    print("---------------------------------  -------  -------  -------  --------")
    for stage in (*STAGES, "benchmark_loop"):
        metric = report["summary"][stage]
        p95 = "-" if metric["p95_ms_per_frame"] is None \
            else f"{metric['p95_ms_per_frame']:.3f}"
        maximum = "-" if metric["max_ms_per_frame"] is None \
            else f"{metric['max_ms_per_frame']:.3f}"
        hz = "-" if metric["effective_hz"] is None \
            else f"{metric['effective_hz']:.2f}"
        print(f"{stage:33}  {metric['mean_ms_per_frame']:7.3f}  "
              f"{p95:>7}  {maximum:>7}  {hz:>8}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--limits", type=Path, default=DEFAULT_LIMITS)
    parser.add_argument("--no-limits", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    records = tuple(HistoricRunSource(args.run_dir))
    if args.limit is not None:
        records = records[:args.limit]
    report = run_diagnostics(records, load_lut(), warmup_frames=args.warmup)
    if not args.no_limits:
        report["validation"] = validate_limits(
            report, load_limits(args.limits))
    output = args.output or (
        DEFAULT_OUTPUT_ROOT / records[0].run_id / "runtime-diagnostics.json")
    write_json(output, report)
    _print_summary(report)
    if "validation" in report:
        print("runtime limits:",
              "PASS" if report["validation"]["passed"] else "FAIL")
    print(output)
    return 0 if report.get("validation", {}).get("passed", True) else 2


if __name__ == "__main__":
    sys.exit(main())
