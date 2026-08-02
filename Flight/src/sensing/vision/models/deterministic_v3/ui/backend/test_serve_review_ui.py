import json
import base64
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from .serve_review_ui import (
    C_SHAPE_MASK_LAYER_ID,
    TOUCHES_FRAME_LAYER_ID,
    _confined,
    _is_supported_layer_id,
    discover_geometry_review_records,
    frame_catalog,
    geometry_frame_result,
    render_review_layer,
    run_catalog,
    topology_component_catalog,
)


def encoded_array(raw, dtype, shape):
    return {"__ndarray__": {
        "encoding": "numpy-contiguous-zlib-base64-v2",
        "dtype": dtype,
        "shape": list(shape),
        "order": "C",
        "writeable": False,
        "data": base64.b64encode(zlib.compress(raw)).decode("ascii"),
    }}


def runtime_record(name, fields):
    return {"__dataclass__": {
        "module": "sensing.vision.models.deterministic_v3.src.schema",
        "name": name,
        "fields": fields,
    }}


class ReviewUiCatalogTests(unittest.TestCase):
    def test_layer_id_validation_accepts_schema_profiles_without_a_fixed_list(self):
        self.assertTrue(_is_supported_layer_id(
            "DensityEvidence[future_profile].final_field"))
        self.assertTrue(_is_supported_layer_id(TOUCHES_FRAME_LAYER_ID))
        self.assertTrue(_is_supported_layer_id(C_SHAPE_MASK_LAYER_ID))
        self.assertFalse(_is_supported_layer_id(
            "DensityEvidence[future_profile].unknown_field"))

    def test_current_review_shape_associates_with_logged_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "Flight" / "logs"
            reviews = root / "ui" / "review_runs"
            run_id = "run-20260731T093159Z"
            stem = "frame-00259602-1785490321007611000"
            frames = logs / run_id / "vision_frames"
            frames.mkdir(parents=True)
            (frames / f"{stem}.jpg").write_bytes(b"jpeg")
            output = reviews / run_id / "frames"
            output.mkdir(parents=True)
            (reviews / run_id / "manifest.json").write_text(
                json.dumps({"run_id": run_id}), encoding="utf-8")
            (reviews / run_id / "runtime-diagnostics.json").write_text(
                json.dumps({"run_id": run_id}), encoding="utf-8")
            (output / f"{stem}.json").write_text(json.dumps({
                "review_format_version": 1,
                "source": {
                    "run_id": run_id,
                    "frame_id": 259602,
                    "sim_time_ns": 1785490321007611000,
                },
                "runtime_record_encoding": "python-schema-runtime-record-v2",
                "schema_records": [
                    runtime_record("FrameObservation", {
                        "image_shape": [2, 2],
                        "base_mask": encoded_array(b"\x00\xff\xff\x00", "|u1", (2, 2)),
                        "size_filtered_mask": encoded_array(
                            b"\x00\xff\xff\x00", "|u1", (2, 2)),
                        "closed_mask": encoded_array(
                            b"\x00\xff\xff\x00", "|u1", (2, 2)),
                        "component_labels": encoded_array(
                            struct.pack("<iiii", 0, 1, 1, 0), "<i4", (2, 2)),
                        "components": {"__tuple__": [runtime_record(
                            "ComponentObservation", {
                            "component_id": 1,
                            "bbox_xywh": {"__tuple__": [0, 0, 2, 2]},
                            "analysis_shape": [2, 2],
                            "image_origin_uv": {"__tuple__": [0, 0]},
                            "touches_frame": True,
                        })]},
                    }),
                    runtime_record("TopologyDecision", {
                        "component_id": 1,
                        "topology_label": "clipped",
                    }),
                    runtime_record("DensityEvidence", {
                        "component_id": 1,
                        "profile": runtime_record("DensityProfile", {
                            "profile_id": "scale_01",
                            "maximum_component_area_px": 591,
                        }),
                        "final_field": encoded_array(
                            struct.pack("<dddd", 0.0, 0.25, 0.5, 1.0),
                            "<f8", (2, 2)),
                        "p90_mask": encoded_array(
                            b"\x00\x00\x00\x01", "|u1", (2, 2)),
                    }),
                    runtime_record("CShapeResult", {
                        "component_id": 1,
                        "refined_mask_origin_uv": {"__tuple__": [0, 0]},
                        "refined_mask": encoded_array(
                            b"\x00\x01\x01\x00", "|u1", (2, 2)),
                    }),
                ],
            }), encoding="utf-8")

            catalog = run_catalog(logs, reviews)
            frame_result = frame_catalog(run_id, logs, reviews)
            topology_result = topology_component_catalog(run_id, logs, reviews)

            self.assertEqual(catalog["runs"][0]["review_json_count"], 1)
            self.assertEqual(
                catalog["runs"][0]["initial_review_url"],
                f"/reviews/{run_id}/frames/{stem}.json")
            self.assertEqual(frame_result["unmatched_review_json"], 0)
            self.assertEqual(frame_result["frames"][0]["review_json_count"], 1)
            self.assertEqual(
                frame_result["frames"][0]["review_urls"],
                [f"/reviews/{run_id}/frames/{stem}.json"])
            self.assertEqual(
                frame_result["frames"][0]["layers"]
                ["DensityEvidence[scale_01].final_field"],
                f"/schema-layers/{run_id}/{stem}/"
                "DensityEvidence%5Bscale_01%5D.final_field.png")
            self.assertEqual(frame_result["density_profile_ids"], ("scale_01",))
            self.assertNotIn(
                "DensityEvidence[scale_10].final_field",
                frame_result["frames"][0]["layers"])
            self.assertTrue(render_review_layer(
                output / f"{stem}.json",
                "DensityEvidence[scale_01].final_field").startswith(
                    b"\x89PNG\r\n\x1a\n"))
            self.assertTrue(render_review_layer(
                output / f"{stem}.json", TOUCHES_FRAME_LAYER_ID).startswith(
                    b"\x89PNG\r\n\x1a\n"))
            self.assertIn(
                C_SHAPE_MASK_LAYER_ID,
                frame_result["frames"][0]["layers"])
            self.assertTrue(render_review_layer(
                output / f"{stem}.json", C_SHAPE_MASK_LAYER_ID).startswith(
                    b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(
                topology_result["available_topology_labels"], ["clipped"])
            self.assertEqual(
                topology_result["missing_topology_label_count"], 0)
            self.assertEqual(
                topology_result["components"][0]["bbox_xywh"], [0, 0, 2, 2])

    def test_catalog_supports_nested_logs_runs_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            frames = logs / "runs" / "run-example" / "vision_frames"
            frames.mkdir(parents=True)
            (frames / "frame-00000001-1.png").write_bytes(b"png")

            catalog = run_catalog(logs, root / "review_runs")

            self.assertEqual(catalog["runs"][0]["frame_count"], 1)

    def test_run_level_geometry_results_match_exact_logged_frame_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "Flight" / "logs"
            reviews = root / "ui" / "review_runs"
            run_id = "run-20260801T031401Z"
            filename = "frame-00106570-1785554042291850700.jpg"
            frames = logs / run_id / "vision_frames"
            frames.mkdir(parents=True)
            (frames / filename).write_bytes(b"jpeg")
            output = reviews / run_id
            output.mkdir(parents=True)
            item = {
                "run_id": run_id,
                "runtime_result": {
                    "frame_id": 106570,
                    "sim_time_ns": 1785554042291850700,
                    "camera_calibration": {
                        "calibration_id": "sim-camera-640x360-pinhole-v1",
                        "image_shape": [360, 640],
                        "camera_matrix": [
                            [320.0, 0.0, 320.0],
                            [0.0, 320.0, 180.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "distortion_coefficients": [0.0] * 5,
                    },
                    "gate_model": {
                        "model_id": "gate-face-centerline-square-2.10m-v1",
                        "side_length_m": 2.1,
                        "object_points_m": [[-1.05, 1.05, 0.0]] * 4,
                    },
                    "camera_pose_estimates": [{
                        "component_id": 4,
                        "accepted": True,
                    }],
                },
            }
            document = {"summary": {"frames": 1}, "frames": [item]}
            (output / "standard-gate-pnp-runtime.json").write_text(
                json.dumps(document), encoding="utf-8")

            records = discover_geometry_review_records(reviews)
            catalog = run_catalog(logs, reviews)
            frame_result = frame_catalog(run_id, logs, reviews)

            self.assertEqual(len(records), 1)
            self.assertEqual(catalog["runs"][0]["geometry_frame_result_count"], 1)
            self.assertEqual(catalog["runs"][0]["camera_pose_estimate_count"], 1)
            self.assertEqual(
                catalog["runs"][0]["accepted_camera_pose_estimate_count"], 1)
            self.assertEqual(
                frame_result["frames"][0]["geometry_result_url"],
                f"/api/runs/{run_id}/geometry-frames/{filename}")
            self.assertEqual(
                geometry_frame_result(run_id, filename, logs, reviews), item)
            self.assertEqual(
                frame_result["unmatched_geometry_frame_result_count"], 0)

    def test_confined_paths_reject_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                _confined(root, "../outside.json")


if __name__ == "__main__":
    unittest.main()
