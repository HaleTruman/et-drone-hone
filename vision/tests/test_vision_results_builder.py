from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.src.schema import InstanceFrame, PipelinePreset, utc_now
from vision.src.vision_results import VisionResultsBuilder


VISION_ROOT = Path(__file__).resolve().parents[1]


class VisionResultsBuilderTests(unittest.TestCase):
    def test_builder_converts_available_instances_and_flips_y(self) -> None:
        preset = PipelinePreset.from_path(VISION_ROOT / "assets" / "pipeline_presets.json")
        instance_frame = InstanceFrame(
            run_id="run-test",
            frame_ordinal=7,
            frame_id="frame_000007",
            source_path="/tmp/frame.jpg",
            image_width=640,
            image_height=360,
            created_at=utc_now(),
            instances=[
                {
                    "instance_id": "gate-0001",
                    "observation_id": "frame_000007-obs-001",
                    "bbox_id": "frame_000007-bbox-001",
                    "tracking_status": "new",
                    "association_score": 0.6,
                    "observationQuality": 0.8,
                    "pose": {
                        "available": True,
                        "bbox_id": "frame_000007-bbox-001",
                        "xyzCameraM": [1.0, 2.5, 9.0],
                        "rpyCameraDeg": [3.0, 4.0, 5.0],
                        "depthM": 9.0,
                        "fitQuality": {"overall": 0.7},
                    },
                },
                {
                    "instance_id": "gate-0002",
                    "observationQuality": 1.0,
                    "pose": {"available": False, "reason": "missing-quad-points"},
                },
                {
                    "instance_id": "gate-0003",
                    "observationQuality": 1.0,
                    "pose": {"available": True, "xyzCameraM": [0.0, 1.0, -1.0], "depthM": -1.0},
                },
            ],
        )
        results = VisionResultsBuilder(preset).process(instance_frame)
        self.assertEqual(results.schema, "vision-results.v1")
        self.assertEqual(results.coordinate_frame, "flight-camera-local")
        self.assertEqual(len(results.gates), 1)
        gate = results.gates[0]
        self.assertEqual(gate.gate_id, "gate-0001")
        self.assertEqual(gate.position_camera_m, (1.0, -2.5, 9.0))
        self.assertEqual(gate.orientation_camera, (3.0, 4.0, 5.0))
        self.assertEqual(gate.position_confidence, 0.8)
        self.assertEqual(gate.orientation_confidence, 0.7)
        self.assertEqual(gate.trace["source_instance_id"], "gate-0001")
        self.assertEqual(gate.trace["source_observation_id"], "frame_000007-obs-001")
        self.assertEqual(gate.trace["source_bbox_id"], "frame_000007-bbox-001")
        self.assertEqual(gate.trace["coordinate_transform"], "opencv-camera-to-result-camera-flip-y")
        self.assertEqual(results.camera_position_camera_m, (0.0, 0.0, 0.0))
        self.assertEqual(results.camera_orientation_camera, (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
