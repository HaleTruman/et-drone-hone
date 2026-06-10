from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FLIGHT_SRC = _PROJECT_ROOT / "Flight" / "src"
if str(_FLIGHT_SRC) not in sys.path:
    sys.path.insert(0, str(_FLIGHT_SRC))

from sensing.vision.cnn.rgb_inference import DEFAULT_CHECKPOINT
from sensing.vision.io.udp_protocol import DEFAULT_HOST, DEFAULT_PORT
from sensing.vision.landmarker.landmarker_pipeline import LandmarkerPipelineConfig
from sensing.vision.pipeline import (
    DEFAULT_OUTPUT_ROOT,
    VisionPipelineConfig,
    run_landmarker_stage,
    run_live_pipeline,
    run_regressor_stage,
)
from sensing.vision.regressor import DEFAULT_REGRESSOR_CHECKPOINT, RegressorPipelineConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision extraction production entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="Receive live UDP frames and run vision inference.")
    live.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    live.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    live.add_argument("--bind-host", default=DEFAULT_HOST)
    live.add_argument("--port", type=int, default=DEFAULT_PORT)
    live.add_argument("--timeout-seconds", type=float)
    live.add_argument("--max-frames", type=int, default=0)
    live.add_argument("--device", default="auto")
    live.add_argument("--run-regressor", action="store_true")
    live.add_argument("--run-landmarker", action="store_true")
    live.add_argument("--regressor-checkpoint", default=str(DEFAULT_REGRESSOR_CHECKPOINT))
    live.add_argument("--top-k", type=int, default=5)
    live.add_argument("--review", action="store_true")
    live.add_argument("--review-output", default="vision/output/review")

    regress = subparsers.add_parser("regress", help="Consume raw logits and write regressor JSON.")
    regress.add_argument("--input-dir", default="vision/output/lightmask_logits")
    regress.add_argument("--output-dir", default="vision/output/regressor_json")
    regress.add_argument("--checkpoint", default=str(DEFAULT_REGRESSOR_CHECKPOINT))
    regress.add_argument("--gate-threshold", type=float, default=0.50)
    regress.add_argument("--confidence-threshold", type=float, default=0.50)
    regress.add_argument("--min-component-area", type=int, default=3)
    regress.add_argument("--max-candidates", type=int, default=32)
    regress.add_argument("--max-frames", type=int, default=0)
    regress.add_argument("--device", default="auto")

    landmark = subparsers.add_parser("landmark", help="Consume regressor JSON and write controller JSON.")
    landmark.add_argument("--input-jsonl")
    landmark.add_argument("--input-dir")
    landmark.add_argument("--output-dir", default="vision/output/landmarker_controller_json")
    landmark.add_argument("--state-path", default="vision/output/landmarker_state.json")
    landmark.add_argument("--top-k", type=int, default=5)
    landmark.add_argument("--max-frames", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "live":
        review_sink = None
        if args.review:
            from vision.tools.review import ReviewConfig, ReviewSink

            review_sink = ReviewSink(ReviewConfig(output_root=Path(args.review_output), run_mode="live", has_truth=False))

        stats = run_live_pipeline(
            VisionPipelineConfig(
                checkpoint=Path(args.checkpoint),
                output_root=Path(args.output_root),
                bind_host=str(args.bind_host),
                port=int(args.port),
                timeout_s=args.timeout_seconds,
                max_frames=int(args.max_frames),
                device=str(args.device),
                run_regressor=bool(args.run_regressor),
                run_landmarker=bool(args.run_landmarker),
                regressor_checkpoint=Path(args.regressor_checkpoint),
                top_k=int(args.top_k),
                review_sink=review_sink,
            )
        )
        fps = stats.cnn.frames_processed / stats.cnn.elapsed_seconds if stats.cnn.elapsed_seconds > 0.0 else 0.0
        print(
            f"live frames={stats.cnn.frames_processed} seconds={stats.cnn.elapsed_seconds:.3f} "
            f"hz={fps:.2f} output_root={stats.output_root}"
        )
        return

    if args.command == "regress":
        stats = run_regressor_stage(
            RegressorPipelineConfig(
                input_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir),
                checkpoint=Path(args.checkpoint),
                gate_threshold=float(args.gate_threshold),
                confidence_threshold=float(args.confidence_threshold),
                min_component_area=int(args.min_component_area),
                max_candidates=int(args.max_candidates),
                max_frames=int(args.max_frames),
                device=str(args.device),
            )
        )
        fps = stats.frames_processed / stats.elapsed_seconds if stats.elapsed_seconds > 0.0 else 0.0
        print(
            f"regressed frames={stats.frames_processed} gates={stats.gates_emitted} "
            f"seconds={stats.elapsed_seconds:.3f} hz={fps:.2f} "
            f"output_dir={stats.output_dir} jsonl={stats.jsonl_path}"
        )
        return

    stats = run_landmarker_stage(
        LandmarkerPipelineConfig(
            input_jsonl=None if args.input_jsonl is None else Path(args.input_jsonl),
            input_dir=None if args.input_dir is None else Path(args.input_dir),
            output_dir=Path(args.output_dir),
            state_path=Path(args.state_path),
            top_k=int(args.top_k),
            max_frames=int(args.max_frames),
        )
    )
    fps = stats.frames_processed / stats.elapsed_seconds if stats.elapsed_seconds > 0.0 else 0.0
    print(
        f"landmarked frames={stats.frames_processed} landmarks={stats.final_landmark_count} "
        f"seconds={stats.elapsed_seconds:.3f} hz={fps:.2f} "
        f"output_dir={stats.output_dir} state={stats.state_path} jsonl={stats.jsonl_path}"
    )


if __name__ == "__main__":
    main()
