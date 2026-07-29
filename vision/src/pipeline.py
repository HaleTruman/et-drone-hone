from __future__ import annotations
import argparse
import json
from pathlib import Path
from final_2d_void_estimate import FORECAST_HORIZON_FRAMES, FinalVoidEstimator
from final_3d_pose_estimate import Final3DPoseEstimator
from instance_tracking import TRAIL_WINDOW, InstanceTracker
from mask import frame_to_mask
from mask_bridge import bridge_outer_hull
from mask_clipping import annotate_mask_clipping
from mask_fill import fill_close_pixels
from schema import (DEFAULT_BRIDGE_MAX_PX, DEFAULT_FILL_RADIUS_PX, DEFAULT_LUT_PATH,
                    DEFAULT_MIN_MASK_REGION_PX, DEFAULT_MIN_ORIGINAL_COVERAGE_DEG,
                    DEFAULT_MIN_VOID_DISTANCE_PX, DEFAULT_MIN_VOID_PX,
                    MaskFrame, VoidAnalysis, save_review_artifacts)
from void_center import find_void_centers

def frame_paths(run_dir: Path) -> list[Path]:
    root = run_dir / "vision_frames"
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(path for path in root.iterdir() if path.suffix.lower() in suffixes)

def analysis_from_mask_frame(mask_frame: MaskFrame) -> VoidAnalysis:
    filled = fill_close_pixels(mask_frame, DEFAULT_FILL_RADIUS_PX)
    bridged = bridge_outer_hull(filled, DEFAULT_BRIDGE_MAX_PX, DEFAULT_MIN_MASK_REGION_PX)
    return annotate_mask_clipping(find_void_centers(
        bridged, DEFAULT_MIN_VOID_PX, DEFAULT_MIN_MASK_REGION_PX,
        DEFAULT_MIN_VOID_DISTANCE_PX, DEFAULT_MIN_ORIGINAL_COVERAGE_DEG,
    ))

def build_analysis(frame_path: Path, lut_path: str) -> VoidAnalysis:
    return analysis_from_mask_frame(frame_to_mask(frame_path, lut_path))

def advance_analysis(analysis: VoidAnalysis, frame_path, tracker: InstanceTracker | None = None,
                      upcoming_frame_paths=(), void_estimator: FinalVoidEstimator | None = None,
                      pose_estimator: Final3DPoseEstimator | None = None) -> VoidAnalysis:
    if tracker is not None:
        tracker.update(analysis, frame_path=str(frame_path),
                       upcoming_frame_paths=[str(p) for p in upcoming_frame_paths])
    if void_estimator is not None:
        void_estimator.update(analysis, frame_path=str(frame_path),
                               upcoming_frame_paths=[str(p) for p in upcoming_frame_paths])
    if pose_estimator is not None:
        pose_estimator.update(analysis, frame_path=str(frame_path))
    return analysis

def process_frame(frame_path: Path, out_dir: Path, lut_path: str, tracker: InstanceTracker | None = None,
                   upcoming_frame_paths=(), void_estimator: FinalVoidEstimator | None = None,
                   pose_estimator: Final3DPoseEstimator | None = None) -> dict:
    analysis = build_analysis(frame_path, lut_path)
    advance_analysis(analysis, frame_path, tracker, upcoming_frame_paths, void_estimator, pose_estimator)
    return save_review_artifacts(analysis, out_dir)

def run_pipeline(run_dir: str | Path, lut_path: str = DEFAULT_LUT_PATH,
                 review_root: str | Path = "review", limit: int | None = None) -> dict:
    run_dir = Path(run_dir)
    out_root = Path(review_root) / run_dir.name
    frames_dir = out_root / "frames"
    frames = frame_paths(run_dir)
    if limit is not None:
        frames = frames[:max(0, int(limit))]
    items = []
    tracker = InstanceTracker()
    void_estimator = FinalVoidEstimator()
    pose_estimator = Final3DPoseEstimator()
    lookahead = max(TRAIL_WINDOW, FORECAST_HORIZON_FRAMES)
    for index, frame_path in enumerate(frames):
        saved = process_frame(frame_path, frames_dir, lut_path, tracker, frames[index + 1: index + 1 + lookahead],
                               void_estimator, pose_estimator)
        items.append({
            "frame_index": index,
            "frame_path": str(frame_path),
            "json_path": saved["json"],
            "void_count": sum(len(region["voids"]) for region in saved["metadata"]["analysis"]["regions"]),
        })
    out_root.mkdir(parents=True, exist_ok=True)
    index_payload = {
        "version": "mask-review-index.v1",
        "run_id": run_dir.name,
        "frame_count": len(items),
        "settings": {
            "lut_path": lut_path,
            "fill_radius_px": DEFAULT_FILL_RADIUS_PX,
            "bridge_max_px": DEFAULT_BRIDGE_MAX_PX,
            "min_mask_region_px": DEFAULT_MIN_MASK_REGION_PX,
            "min_void_px": DEFAULT_MIN_VOID_PX,
            "min_void_distance_px": DEFAULT_MIN_VOID_DISTANCE_PX,
            "min_coverage_deg": DEFAULT_MIN_ORIGINAL_COVERAGE_DEG,
        },
        "frames": items,
    }
    (out_root / "review-index.json").write_text(json.dumps(index_payload, indent=2) + "\n", encoding="utf-8")
    return index_payload

def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute mask review artifacts for one run.")
    parser.add_argument("run_dir")
    parser.add_argument("--lut", default=DEFAULT_LUT_PATH)
    parser.add_argument("--review-root", default="review")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    result = run_pipeline(args.run_dir, args.lut, args.review_root, args.limit)
    print(json.dumps({"run_id": result["run_id"], "frame_count": result["frame_count"]}, indent=2))

if __name__ == "__main__":
    main()
