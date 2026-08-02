import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sensing.vision.models.deterministic_v3.src.void_ned import (
    camera_position_ned,
    rotation_world_from_camera,
)
from sensing.vision.models.deterministic_v3.ui.backend.pnp_world_replay import (
    build_pnp_world_replay,
)
from sensing.vision.models.deterministic_v3.ui.backend.pnp_world_replay_configuration import (
    PnpWorldReplayConfiguration,
)


RUN_ID = "run-test"
FRAME_ID = 42
FRAME_TIME_NS = 123_000


def _candidate(rank, *, position=(0.0, 0.0, 10.0)):
    return {
        "candidate_rank": rank,
        "rotation_vector_model_to_camera": [0.0, 0.0, 0.0],
        "position_camera_m": list(position),
        "reprojection_rmse_px": 0.2 + rank,
    }


def _pnp_estimate(component_id=1, *, accepted=True):
    return {
        "frame_id": FRAME_ID,
        "sim_time_ns": FRAME_TIME_NS,
        "component_id": component_id,
        "gate_index": 0,
        "route": "standard",
        "solver": "test",
        "camera_calibration_id": "camera-v1",
        "gate_model_id": "gate-v1",
        "candidates": [_candidate(0), _candidate(1, position=(0.0, 0.0, 10.5))],
        "selected_candidate_rank": 0,
        "candidate_count": 2,
        "ambiguity_gap_px": 1.0,
        "position_confidence": 0.9,
        "orientation_confidence": 0.8,
        "accepted": accepted,
        "rejection_reason": None if accepted else "test_rejection",
    }


def _final_pose(component_id=1, *, accepted=True, position=(0.0, 0.0, 11.0)):
    return {
        "frame_id": FRAME_ID,
        "sim_time_ns": FRAME_TIME_NS,
        "component_id": component_id,
        "gate_index": 0,
        "route": "standard",
        "solver": "test-regression",
        "camera_calibration_id": "camera-v1",
        "gate_model_id": "gate-v1",
        "rotation_vector_model_to_camera": [0.0, 0.0, 0.0],
        "position_camera_m": list(position),
        "reprojection_rmse_px": 0.3,
        "position_confidence": 0.9,
        "orientation_confidence": 0.8,
        "accepted": accepted,
        "rejection_reason": None if accepted else "test_rejection",
        "regression_evidence": None,
    }


def _geometry_item():
    return {
        "run_id": RUN_ID,
        "runtime_result": {
            "frame_id": FRAME_ID,
            "sim_time_ns": FRAME_TIME_NS,
            "camera_calibration": {
                "calibration_id": "camera-v1",
                "image_shape": [360, 640],
                "camera_matrix": [
                    [320.0, 0.0, 320.0],
                    [0.0, 320.0, 180.0],
                    [0.0, 0.0, 1.0],
                ],
                "distortion_coefficients": [0.0] * 5,
            },
            "gate_model": {
                "model_id": "gate-v1",
                "side_length_m": 2.0,
                "corner_order": [
                    "upper_left", "upper_right", "lower_right", "lower_left"],
                "object_points_m": [
                    [-1.0, 1.0, 0.0], [1.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
                ],
            },
            "pnp_relative_pose_estimates": [
                _pnp_estimate(), _pnp_estimate(2, accepted=False)],
            "camera_pose_estimates": [_final_pose()],
        },
    }


def _write_run(root: Path, *, include_mount=True, alignment_ms=20.0):
    (root / "lists").mkdir(parents=True)
    metadata = {}
    if include_mount:
        metadata["initialization_constants"] = {
            "VIO_CAMERA_TILT_DEG": 20.0,
            "VIO_BODY_TO_CAMERA_TRANSLATION_BODY_FRD_M": [0.0, 0.0, 0.0],
        }
    (root / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8")
    telemetry = {
        "inner_cycle": 7,
        "monotonic_s": 2.0,
        "telemetry": {"vehicle_state": {
            "sim_time_ns": 7_000,
            "position_local_ned_m": [1.0, 2.0, 3.0],
            "attitude_quaternion": [1.0, 0.0, 0.0, 0.0],
        }},
    }
    vision = {
        "monotonic_s": 2.0 + alignment_ms / 1000,
        "frame": {
            "frame_id": FRAME_ID,
            "sim_time_ns": FRAME_TIME_NS,
            "inner_cycle": 7,
        },
    }
    (root / "lists" / "telemetry.jsonl").write_text(
        json.dumps(telemetry) + "\n", encoding="utf-8")
    (root / "lists" / "vision_frames.jsonl").write_text(
        json.dumps(vision) + "\n", encoding="utf-8")


class PnpWorldReplayTests(unittest.TestCase):
    def test_exact_cycle_projects_raw_candidates_and_final_pose_into_local_ned(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / RUN_ID
            _write_run(run)
            payload = build_pnp_world_replay(
                run_id=RUN_ID, run_dir=run,
                geometry_items=[_geometry_item()])

        frame = payload["frames"][0]
        projection = frame["ui_projection"]
        self.assertEqual(payload["coordinate_system"]["world"], "local_ned")
        self.assertEqual(frame["sync"]["match_source"], "frame_cycle")
        self.assertAlmostEqual(frame["sync"]["alignment_error_ms"], 20.0)
        self.assertTrue(frame["sync"]["within_tolerance"])
        self.assertTrue(projection["available"])
        self.assertEqual(len(projection["pnp_relative_pose_estimates"]), 1)
        self.assertEqual(len(projection["camera_pose_estimates"]), 1)

        vehicle = type("Vehicle", (), {
            "position_local_ned_m": (1.0, 2.0, 3.0),
            "attitude_quaternion": (1.0, 0.0, 0.0, 0.0),
        })()
        origin = camera_position_ned(vehicle)
        trajectory_projection = payload["trajectory"][0]["ui_projection"]
        self.assertTrue(trajectory_projection["available"])
        np.testing.assert_allclose(
            trajectory_projection["camera_position_local_ned_m"], origin)
        rotation = rotation_world_from_camera(vehicle.attitude_quaternion)
        expected = origin + rotation @ np.array([0.0, 0.0, 10.0])
        raw = projection[
            "pnp_relative_pose_estimates"][0]["selected_candidate"]
        np.testing.assert_allclose(raw["position_local_ned_m"], expected)
        np.testing.assert_allclose(
            np.mean(raw["corners_local_ned_m"], axis=0), expected)
        self.assertEqual(raw["orientation_status"], "provisional")
        self.assertIsNotNone(projection[
            "pnp_relative_pose_estimates"][0]["secondary_candidate"])
        matrix = np.asarray(
            projection["rotation_local_ned_from_camera_cv"])
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0)

    def test_alignment_gate_suppresses_projection_but_keeps_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / RUN_ID
            _write_run(run, alignment_ms=20.0)
            payload = build_pnp_world_replay(
                run_id=RUN_ID, run_dir=run,
                geometry_items=[_geometry_item()],
                configuration=PnpWorldReplayConfiguration(
                    maximum_alignment_error_ms=10.0),
            )

        self.assertEqual(len(payload["trajectory"]), 1)
        frame = payload["frames"][0]
        self.assertFalse(frame["sync"]["within_tolerance"])
        self.assertFalse(frame["ui_projection"]["available"])
        self.assertEqual(
            frame["ui_projection"]["unavailable_reason"],
            "alignment_tolerance_exceeded")

    def test_missing_logged_mount_suppresses_world_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / RUN_ID
            _write_run(run, include_mount=False)
            payload = build_pnp_world_replay(
                run_id=RUN_ID, run_dir=run,
                geometry_items=[_geometry_item()])

        self.assertFalse(payload["camera_mount"]["usable"])
        self.assertFalse(payload["trajectory"][0]["ui_projection"]["available"])
        self.assertEqual(
            payload["frames"][0]["ui_projection"]["unavailable_reason"],
            "logged_camera_mount_unavailable")


if __name__ == "__main__":
    unittest.main()
