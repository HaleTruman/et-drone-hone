from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.src.schema import (
    BBoxFrame,
    BBoxObservation,
    ColorMaskFrame,
    PipelinePreset,
    SourceFrame,
    VisionGateResult,
    VisionResults,
    utc_now,
)


VISION_ROOT = Path(__file__).resolve().parents[1]


class SchemaTests(unittest.TestCase):
    def test_preset_defaults_and_camera_contract(self) -> None:
        preset = PipelinePreset.from_path(VISION_ROOT / "assets" / "pipeline_presets.json")
        self.assertEqual(preset.schema, "vision-pipeline-presets.v1")
        self.assertEqual(preset.image, {"width": 640, "height": 360})
        self.assertEqual(preset.camera["fx"], 320.0)
        self.assertEqual(preset.camera["fy"], 320.0)
        self.assertEqual(preset.camera["cx"], 320.0)
        self.assertEqual(preset.camera["cy"], 180.0)
        self.assertEqual(preset.enabled_bits(), [0, 1, 2, 3])

    def test_frame_dataclasses_round_trip_json_safe_dicts(self) -> None:
        created_at = utc_now()
        source = SourceFrame(
            run_id="run-test",
            frame_ordinal=0,
            frame_id="frame_000000",
            source_path="/tmp/frame.jpg",
            image_width=640,
            image_height=360,
            created_at=created_at,
            timing_ms={"source": 1.25},
            source_sha256="abc",
        )
        self.assertEqual(SourceFrame.from_dict(source.to_dict()).to_dict(), source.to_dict())
        self.assertEqual(source.to_dict()["frame_id"], "frame_000000")
        self.assertEqual(source.to_dict()["timing_ms"]["source"], 1.25)

        mask = ColorMaskFrame(
            run_id="run-test",
            frame_ordinal=0,
            frame_id="frame_000000",
            source_path="/tmp/frame.jpg",
            image_width=2,
            image_height=2,
            created_at=created_at,
            timing_ms={"color_masking": 0.5},
            layer_pixel_counts={"001": 1},
            mask_bits=np.array([[1, 0], [0, 0]], dtype=np.uint8),
        )
        payload = mask.to_dict()
        self.assertNotIn("mask_bits", payload)
        json.dumps(payload)
        self.assertEqual(ColorMaskFrame.from_dict(payload).to_dict(), payload)

        bbox = BBoxFrame(
            run_id="run-test",
            frame_ordinal=0,
            frame_id="frame_000000",
            source_path="/tmp/frame.jpg",
            image_width=2,
            image_height=2,
            created_at=created_at,
            timing_ms={"bboxing": 0.75},
            observations=[
                BBoxObservation(
                    bbox_id="frame_000000-bbox-001",
                    component_label=1,
                    bbox_px=[0, 0, 1, 1],
                    bbox_uv=[0.0, 0.0, 1.0, 1.0],
                    center_px=[0.5, 0.5],
                    center_uv=[0.25, 0.25],
                    width_px=2,
                    height_px=2,
                    area_px=4,
                    pixel_count=1,
                    layer_counts={"001": 1},
                    dominant_prefix="001",
                    quality={"fillRatio": 0.25},
                )
            ],
            bbox_count=1,
            selected_pixel_count=1,
        )
        round_trip = BBoxFrame.from_dict(bbox.to_dict())
        self.assertEqual(round_trip.to_dict(), bbox.to_dict())
        json.dumps(round_trip.to_dict())

    def test_vision_results_are_clean_import_contract(self) -> None:
        result = VisionResults(
            run_id="run-test",
            frame_ordinal=3,
            frame_id="frame_000003",
            source_path="/tmp/frame.jpg",
            image_width=640,
            image_height=360,
            created_at=utc_now(),
            gates=[
                VisionGateResult(
                    gate_id="gate-0001",
                    position_camera_m=[1, -2, 9],
                    position_confidence=0.8,
                    orientation_camera=[3, 4, 5],
                    orientation_confidence=0.7,
                    trace={"source_bbox_id": "frame_000003-bbox-001"},
                )
            ],
        )
        self.assertEqual(result.gates[0].position_camera_m, (1.0, -2.0, 9.0))
        self.assertEqual(result.gates[0].orientation_camera, (3.0, 4.0, 5.0))
        payload = result.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["gates"][0]["position_camera_m"], [1.0, -2.0, 9.0])
        round_trip = VisionResults.from_dict(payload)
        self.assertEqual(round_trip.gates[0].position_camera_m, (1.0, -2.0, 9.0))


if __name__ == "__main__":
    unittest.main()
