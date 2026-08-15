import math
import unittest

import numpy as np

from Vision.review_pipeline.evaluate_reference_dataset import (
    _distribution,
    _new_backend_accumulator,
    _score_backend_frame,
    _truth_camera_cv,
    _truth_gate_ned,
    mask_iou_for_gate,
    match_detections,
    normalize_camera,
)
from sensing.vision.models.deterministic_v3_2.src.void_detection import warm_gate_mask


def _gate(*, label="gate", center=(100.0, 100.0), size=40.0, stencil=16):
    half = size / 2.0
    return {
        "label": label,
        "custom_depth_stencil_value": stencil,
        "bbox_2d_px": {
            "x_min_px": center[0] - half,
            "y_min_px": center[1] - half,
            "x_max_px": center[0] + half,
            "y_max_px": center[1] + half,
        },
        "corners": {
            "inner": {
                "tl": {"pixel_px": {"x": center[0] - 5, "y": center[1] - 5}},
                "tr": {"pixel_px": {"x": center[0] + 5, "y": center[1] - 5}},
                "bl": {"pixel_px": {"x": center[0] - 5, "y": center[1] + 5}},
                "br": {"pixel_px": {"x": center[0] + 5, "y": center[1] + 5}},
            },
        },
    }


class CoordinateNormalizationTests(unittest.TestCase):
    def test_identity_unreal_camera_maps_truth_to_ned_and_cv(self):
        frame = {
            "camera": {
                "world_location_m": {"x": 0.0, "y": 0.0, "z": 0.0},
                "world_rotation_deg": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            },
        }
        gate = {
            "relative_position_camera_frame_m": {
                "right": 2.0,
                "up": 3.0,
                "forward": 4.0,
            },
            "world_location_m": {"x": 4.0, "y": 2.0, "z": 3.0},
        }
        camera = normalize_camera(frame)

        self.assertTrue(np.allclose(_truth_camera_cv(gate), (2.0, -3.0, 4.0)))
        self.assertTrue(np.allclose(_truth_gate_ned(gate), (4.0, 2.0, -3.0)))
        reconstructed = (
            np.asarray(camera.position_local_ned_m)
            + camera.rotation_ned_from_camera_cv @ _truth_camera_cv(gate)
        )
        self.assertTrue(np.allclose(reconstructed, _truth_gate_ned(gate)))
        self.assertAlmostEqual(
            sum(value * value for value in camera.attitude_body_to_ned_wxyz),
            1.0,
        )


class MatchingTests(unittest.TestCase):
    def test_global_assignment_matches_nearest_candidates(self):
        gates = [
            _gate(label="left", center=(50.0, 100.0)),
            _gate(label="right", center=(150.0, 100.0)),
        ]
        detections = [
            {"center_px": [148.0, 102.0]},
            {"center_px": [52.0, 99.0]},
            {"center_px": [300.0, 300.0]},
        ]

        matches, missed, false_positives = match_detections(gates, detections)

        self.assertEqual(matches, {0: 1, 1: 0})
        self.assertEqual(missed, [])
        self.assertEqual(false_positives, [2])

    def test_far_candidate_is_not_forced_to_match(self):
        matches, missed, false_positives = match_detections(
            [_gate(center=(20.0, 20.0), size=10.0)],
            [{"center_px": [200.0, 200.0]}],
        )

        self.assertEqual(matches, {})
        self.assertEqual(missed, [0])
        self.assertEqual(false_positives, [0])


class MetricTests(unittest.TestCase):
    def test_backend_error_counts_truth_as_missed_and_retains_latency(self):
        gate = _gate()
        accumulator = _new_backend_accumulator()

        score = _score_backend_frame(
            run_id="run_001",
            frame_index=0,
            visible_gates=[gate],
            result={"status": "error", "elapsed_ms": 7.5},
            prediction_mask=None,
            truth_mask=np.zeros((200, 200), dtype=np.uint8),
            accumulator=accumulator,
        )

        self.assertEqual(score["status"], "error")
        self.assertEqual(accumulator["truth_visible"], 1)
        self.assertEqual(accumulator["detected"], 0)
        self.assertEqual(accumulator["elapsed_ms"], [7.5])
        self.assertEqual(accumulator["by_gate"]["gate"]["truth"], 1)

    def test_warm_gate_mask_accepts_orange_and_rejects_grey_and_blue(self):
        image = np.asarray([[
            [20, 80, 230],
            [120, 120, 120],
            [230, 80, 20],
        ]], dtype=np.uint8)

        mask = warm_gate_mask(image)

        self.assertEqual(mask.tolist(), [[255, 0, 0]])

    def test_mask_iou_uses_stencil_identity_inside_gate_box(self):
        truth = np.zeros((20, 20), dtype=np.uint8)
        prediction = np.zeros_like(truth)
        truth[5:10, 5:10] = 16
        truth[5:10, 12:17] = 32
        prediction[5:10, 5:10] = 255
        gate = _gate(center=(7.0, 7.0), size=8.0, stencil=16)

        self.assertAlmostEqual(mask_iou_for_gate(prediction, truth, gate), 1.0)

    def test_distribution_reports_required_latency_percentiles(self):
        stats = _distribution([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["median"], 2.5)
        self.assertTrue(math.isclose(stats["p90"], 3.7))
        self.assertTrue(math.isclose(stats["p95"], 3.85))
        self.assertEqual(stats["max"], 4.0)


if __name__ == "__main__":
    unittest.main()
