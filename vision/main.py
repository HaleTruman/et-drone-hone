#!/usr/bin/env python3
"""0721Vision product entrypoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline import PipelineConfig, VisionPipeline


APP_DIR = Path(__file__).resolve().parent
TOOLS_DIR = APP_DIR / "tools"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["batch", "watch", "live"], default="batch")
    parser.add_argument("--source-dir", type=Path, default=None, help="server-visible JPEG directory or source manifest")
    parser.add_argument("--live-point", default=None, help="live source path; v1 supports server-visible paths")
    parser.add_argument("--output-root", type=Path, default=APP_DIR / "assets" / "pipeline_runs")
    parser.add_argument("--artifact-root", type=Path, default=APP_DIR / "assets")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-parallel", action="store_true", help="disable stateless stage overlap for debugging")
    parser.add_argument("--no-debug", action="store_true", help="publish only compact final per-frame instance JSON outputs")
    parser.add_argument("--debug-artifacts", action="store_true", help="write per-frame stage debug artifacts")
    parser.add_argument("--aggregate-debug-manifests", action="store_true", help="write legacy cumulative stage manifests for review/benchmarking")
    parser.add_argument("--serve-ui", action="store_true", help="host the UI and pipeline launcher API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        mode=args.mode,
        source_dir=args.source_dir,
        live_point=args.live_point,
        output_root=args.output_root,
        artifact_root=args.artifact_root,
        max_frames=args.max_frames,
        parallel_stateless=not args.no_parallel,
        debug_artifacts=bool(args.debug_artifacts) and not bool(args.no_debug),
        aggregate_debug_manifests=bool(args.aggregate_debug_manifests) and not bool(args.no_debug),
    )


def run_headless(args: argparse.Namespace) -> int:
    pipeline = VisionPipeline(config_from_args(args))
    try:
        status = pipeline.run()
    except KeyboardInterrupt:
        pipeline.stop()
        status = pipeline.status_snapshot()
    print(json.dumps({"ok": status.get("state") != "error", "pipeline": status}, indent=2))
    return 1 if status.get("state") == "error" else 0


def run_with_ui(args: argparse.Namespace) -> int:
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    import serve_vision_review

    if args.source_dir or args.live_point:
        start_payload = {
            "mode": args.mode,
            "sourceDir": str(args.source_dir) if args.source_dir else None,
            "livePoint": args.live_point,
            "maxFrames": args.max_frames,
            "outputRoot": str(args.output_root),
            "artifactRoot": str(args.artifact_root),
            "debugArtifacts": bool(args.debug_artifacts) and not bool(args.no_debug),
            "aggregateDebugManifests": bool(args.aggregate_debug_manifests) and not bool(args.no_debug),
        }
        result = serve_vision_review.start_pipeline_job(start_payload)
        if result.get("conflict") or not result.get("ok", True):
            print(json.dumps(result, indent=2), file=sys.stderr)
            return 1
    serve_vision_review.serve(args.host, args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.serve_ui:
        return run_with_ui(args)
    return run_headless(args)


if __name__ == "__main__":
    raise SystemExit(main())
