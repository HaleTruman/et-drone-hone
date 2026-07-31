"""Publish gate positions in run-local NED from per-frame void detections.

Sits between the frame-local geometry and the flight service, and owns every
concern that spans frames so `void_position.py` can stay a pure per-frame
function: gate identity, published confidence, and any later consolidation of
fragmented tracks into stable gates.

Identity currently comes straight from the instance tracker's `track_id`, which
is frame-to-frame only. It cannot bridge a detection dropout, so one physical
gate becomes several ids over a run -- measured at roughly 25x more track ids
than gates on the recorded runs. Consolidating them is the extension this module
exists to hold; `GatePublisher._gate_id` is the seam, and `distinct_gate_ids` in
the observation trace makes the fragmentation visible meanwhile.

Orientation is deliberately not published. The perspective signal is buried
under corner-localisation noise beyond roughly 11 m, so a confident orientation
at range would be invented rather than measured.
"""
from core.schema import VisionGateObservation, VisionObservation
from .void_ned import to_ned
from .void_position import estimate_position

SOURCE = "detection_v2"
MAX_SCALE_RESIDUAL = 0.20   # p90 of the outer/inner channel disagreement
EARN_FRAMES = 5             # frames before a track publishes at full confidence
MIN_CONSECUTIVE_FRAMES = 2  # a track must outlive its first frame to publish

def confidence(estimate, track):
    """Scale-channel agreement, scaled by how long the track has persisted."""
    quality = max(0.0, 1.0 - estimate.scale_residual / MAX_SCALE_RESIDUAL)
    persistence = min(1.0, track.consecutive_frame_count / EARN_FRAMES)
    return round(quality * persistence, 6)

def gate_trace(ned, track):
    """The ray and its anisotropy; `position_local_ned` alone discards both."""
    return {"camera_position_ned_m": ned.camera_position_ned_m,
            "direction_ned": ned.direction_ned,
            "range_m": round(ned.range_m, 4),
            "sigma_bearing_m": round(ned.sigma_bearing_m, 4),
            "sigma_range_m": round(ned.sigma_range_m, 4),
            "scale_residual": round(ned.scale_residual, 4),
            "pose_gap_ns": ned.pose_gap_ns,
            "track_id": track.track_id,
            "consecutive_frame_count": track.consecutive_frame_count}

class GatePublisher:
    """Turns detections into published gates, carrying identity across frames."""

    def __init__(self, source=SOURCE):
        self.source = str(source)
        self._gate_ids = {}

    def _gate_id(self, track):
        """Stable id for a track. Where fragmented tracks would be merged."""
        return self._gate_ids.setdefault(track.track_id, track.track_id)

    def observe(self, detection_frame, vehicle_state=None):
        """One VoidDetectionFrame plus a vehicle state to a VisionObservation."""
        gates = []
        if vehicle_state is not None:
            for detection in detection_frame.detections:
                if detection.track.consecutive_frame_count < MIN_CONSECUTIVE_FRAMES:
                    continue
                estimate = estimate_position(detection.geometry)
                if estimate is None:
                    continue
                ned = to_ned(estimate, vehicle_state)
                gates.append(VisionGateObservation(
                    gate_id=self._gate_id(detection.track),
                    position_local_ned=ned.position_ned_m,
                    position_confidence=confidence(estimate, detection.track),
                    trace=gate_trace(ned, detection.track)))
        return VisionObservation(
            frame_id=detection_frame.frame_id,
            sim_time_ns=detection_frame.sim_time_ns,
            gates=gates, source=self.source,
            trace={"coordinate_frame": "local_ned",
                   "position_semantics": "absolute_landmark",
                   "detections": len(detection_frame.detections),
                   "published": len(gates),
                   "distinct_gate_ids": len(self._gate_ids),
                   "pose_available": vehicle_state is not None})
