"""The live entrypoint from io_specification.md: one VisionFrame in, one VisionObservation out.
Reuses pipeline.py's analysis_from_mask_frame/advance_analysis rather than reimplementing the
stage sequence -- the file-based batch path (pipeline.py) and this live path share one real
implementation. See mask_review_pipeline/live_frame_ingestion_plan.md for the design; telemetry
ingestion is explicitly out of scope here (see that doc's last section)."""

from __future__ import annotations

from final_2d_void_estimate import FinalVoidEstimator
from final_3d_pose_estimate import Final3DPoseEstimator
from instance_tracking import InstanceTracker
from mask import image_to_mask, jpeg_bytes_to_mask
from pipeline import advance_analysis, analysis_from_mask_frame
from schema import DEFAULT_LUT_PATH, VehicleState, VisionFrame, VisionObservation


def _frame_label(frame: VisionFrame) -> str:
    return f"frame-{frame.frame_id:08d}-{frame.sim_time_ns}"


class DeterministicVision:
    def __init__(self, lut_path: str = DEFAULT_LUT_PATH) -> None:
        self._lut_path = lut_path
        self._tracker = InstanceTracker()
        self._void_estimator = FinalVoidEstimator()
        self._pose_estimator = Final3DPoseEstimator()

    def process_frame(self, frame: VisionFrame, vehicle_state: VehicleState | None = None) -> VisionObservation:
        frame_label = _frame_label(frame)
        if frame.image is not None:
            mask_frame = image_to_mask(frame.image, frame_label, self._lut_path)
        else:
            mask_frame = jpeg_bytes_to_mask(frame.jpeg_bytes, frame_label, self._lut_path)

        analysis = analysis_from_mask_frame(mask_frame)
        # a live caller can't peek at frames that haven't arrived yet -- upcoming_frame_paths=()
        # means the mc-trail/forecast lookahead in instance_tracking.py/final_2d_void_estimate.py
        # is naturally unavailable here, unlike batch mode replaying an already-recorded run.
        # pose_estimator=None here: 3D pose is handled separately below, from vehicle_state
        # directly, not from a file-based telemetry lookup.
        advance_analysis(analysis, frame_label, self._tracker, (), self._void_estimator, pose_estimator=None)
        self._pose_estimator.update_live(analysis, frame_label, vehicle_state)
        return analysis.vision_observation
