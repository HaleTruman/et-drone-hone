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
    discover_evaluation_runs,
    evaluation_frame_catalog,
    frame_catalog,
    geometry_frame_history,
    geometry_frame_result,
    pnp_world_replay,
    render_review_layer,
    render_standard_gate_component_layer,
    run_catalog,
    standard_gate_result_catalog,
    standard_gate_review_layer,
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


def grayscale_png_pixels(encoded):
    width, height = struct.unpack(">II", encoded[16:24])
    offset = 8
    compressed = bytearray()
    while offset < len(encoded):
        length = struct.unpack(">I", encoded[offset:offset + 4])[0]
        kind = encoded[offset + 4:offset + 8]
        payload = encoded[offset + 8:offset + 8 + length]
        if kind == b"IDAT":
            compressed.extend(payload)
        offset += 12 + length
    rows = zlib.decompress(compressed)
    self_filtered = []
    for row in range(height):
        start = row * (width + 1)
        if rows[start] != 0:
            raise AssertionError("fixture PNG used a non-zero row filter")
        self_filtered.extend(rows[start + 1:start + width + 1])
    return width, height, bytes(self_filtered)


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
            reviews = root / "Viewer" / "logs" / "review" / "runs"
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

            catalog = run_catalog(logs, root / "Viewer" / "logs" / "review" / "runs")

            self.assertEqual(catalog["runs"][0]["frame_count"], 1)

    def test_evaluation_catalog_exposes_overlay_urls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluations = root / "Viewer" / "logs" / "evaluation" / "runs"
            evaluation_id = "evaluation-20260802T173817Z-run-example"
            output = evaluations / evaluation_id
            overlay = (
                output / "overlays" / "deterministic_v3" / "composite" /
                "frame-00000017-123456789.png")
            overlay.parent.mkdir(parents=True)
            overlay.write_bytes(b"png")
            (output / "metadata.json").write_text(json.dumps({
                "created_utc": "20260802T173817Z",
                "source_run": {
                    "run_id": "run-example",
                    "metadata": {"large": "omitted from API"},
                },
                "selection": {"selected_frame_count": 1},
                "overlays": {
                    "enabled": True,
                    "layers": ["composite"],
                },
            }), encoding="utf-8")
            (output / "summary.json").write_text(json.dumps({
                "processed_frame_count": 1,
                "frames_changed": 1,
                "frames_with_errors": 0,
                "frames_with_vehicle_state": 1,
                "max_position_delta_m": 0.25,
            }), encoding="utf-8")
            (output / "frames.jsonl").write_text(json.dumps({
                "source": {
                    "run_id": "run-example",
                    "frame_id": 17,
                    "sim_time_ns": 123456789,
                    "relative_path": "vision_frames/frame-00000017-123456789.jpg",
                },
                "comparison": {
                    "changed": True,
                    "gate_count_delta": 1,
                },
            }) + "\n", encoding="utf-8")

            catalog = discover_evaluation_runs(evaluations)
            frames = evaluation_frame_catalog(evaluation_id, evaluations)

            self.assertEqual(catalog["evaluations"][0]["id"], evaluation_id)
            self.assertTrue(catalog["evaluations"][0]["overlays_enabled"])
            self.assertEqual(frames["frames"][0]["source_image_url"],
                             "/frames/run-example/frame-00000017-123456789.jpg")
            self.assertEqual(
                frames["frames"][0]["overlays"]["deterministic_v3"]["composite"],
                "/evaluation-overlays/evaluation-20260802T173817Z-run-example/"
                "deterministic_v3/composite/frame-00000017-123456789.png")
            self.assertNotIn("metadata", frames["metadata"]["source_run"])

    def test_run_level_geometry_results_match_exact_logged_frame_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "Flight" / "logs"
            reviews = root / "Viewer" / "logs" / "review" / "runs"
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
                    "pnp_relative_pose_estimates": [{
                        "component_id": 4,
                        "accepted": True,
                        "candidates": [
                            {"candidate_rank": 0}, {"candidate_rank": 1}],
                    }],
                    "camera_pose_estimates": [{
                        "component_id": 4,
                        "accepted": True,
                    }],
                },
            }
            document = {
                "summary": {"format_version": 2, "frames": 1},
                "frames": [item],
            }
            (output / "standard-gate-pnp-runtime.json").write_text(
                json.dumps(document), encoding="utf-8")

            records = discover_geometry_review_records(reviews)
            catalog = run_catalog(logs, reviews)
            frame_result = frame_catalog(run_id, logs, reviews)

            self.assertEqual(len(records), 1)
            self.assertEqual(catalog["runs"][0]["geometry_frame_result_count"], 1)
            self.assertEqual(
                catalog["runs"][0]["pnp_relative_pose_estimate_count"], 1)
            self.assertEqual(
                catalog["runs"][0][
                    "accepted_pnp_relative_pose_estimate_count"], 1)
            self.assertEqual(
                catalog["runs"][0]["secondary_pnp_candidate_count"], 1)
            self.assertEqual(catalog["runs"][0]["camera_pose_estimate_count"], 1)
            self.assertEqual(
                catalog["runs"][0]["accepted_camera_pose_estimate_count"], 1)
            self.assertEqual(
                frame_result["frames"][0]["geometry_result_url"],
                f"/api/runs/{run_id}/geometry-frames/{filename}")
            self.assertEqual(
                geometry_frame_result(run_id, filename, logs, reviews), item)
            self.assertEqual(
                geometry_frame_history(run_id, logs, reviews)["frames"],
                [item])
            world = pnp_world_replay(run_id, logs, reviews)
            self.assertEqual(
                world["version"], "deterministic-v3.pnp-world-replay.v2")
            self.assertEqual(world["frames"][0]["runtime_result"],
                             item["runtime_result"])
            self.assertEqual(
                frame_result["unmatched_geometry_frame_result_count"], 0)

    def test_run_level_geometry_discovery_fails_closed_before_format_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            reviews = Path(temporary) / "run-example"
            reviews.mkdir(parents=True)
            (reviews / "geometry.json").write_text(json.dumps({
                "summary": {"format_version": 1},
                "frames": [{
                    "run_id": "run-example",
                    "runtime_result": {
                        "frame_id": 1,
                        "sim_time_ns": 2,
                        "camera_calibration": {},
                        "gate_model": {},
                        "pnp_relative_pose_estimates": [],
                        "camera_pose_estimates": [],
                    },
                }],
            }), encoding="utf-8")

            self.assertEqual(
                discover_geometry_review_records(reviews.parent), [])

    def test_standard_gate_catalog_joins_exact_component_and_selected_density(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "Flight" / "logs"
            reviews = root / "Viewer" / "logs" / "review" / "runs"
            run_id = "run-20260801T031401Z"
            frame_id = 106570
            sim_time_ns = 1785554042291850700
            stem = f"frame-{frame_id:08d}-{sim_time_ns}"
            frame_dir = logs / run_id / "vision_frames"
            frame_dir.mkdir(parents=True)
            (frame_dir / f"{stem}.jpg").write_bytes(b"jpeg")
            review_frames = reviews / run_id / "frames"
            review_frames.mkdir(parents=True)

            profile = {
                "profile_id": "scale_03",
                "calibration_version": "test",
                "maximum_component_area_px": 1000,
                "density_radius_px": 3,
                "ridge_radius_px": 2,
                "relative_cap": 1.0,
                "inverse_gamma": 2.0,
                "ridge_gamma": 3.0,
            }
            components = [
                runtime_record("ComponentObservation", {
                    "component_id": 4,
                    "bbox_xywh": {"__tuple__": [1, 0, 2, 2]},
                    "image_origin_uv": {"__tuple__": [1, 0]},
                    "analysis_shape": {"__tuple__": [2, 2]},
                    "touches_frame": False,
                }),
                runtime_record("ComponentObservation", {
                    "component_id": 5,
                    "bbox_xywh": {"__tuple__": [0, 2, 1, 1]},
                    "image_origin_uv": {"__tuple__": [0, 2]},
                    "analysis_shape": {"__tuple__": [1, 1]},
                    "touches_frame": True,
                }),
            ]
            frame_payload = {
                "review_format_version": 6,
                "source": {
                    "run_id": run_id,
                    "frame_id": frame_id,
                    "sim_time_ns": sim_time_ns,
                    "relative_path": f"vision_frames/{stem}.jpg",
                },
                "schema_records": [
                    runtime_record("FrameObservation", {
                        "frame_id": frame_id,
                        "sim_time_ns": sim_time_ns,
                        "image_shape": {"__tuple__": [3, 4]},
                        "closed_mask": encoded_array(
                            b"\x00\x01\x01\x00"
                            b"\x00\x01\x01\x00"
                            b"\x01\x00\x00\x00", "|u1", (3, 4)),
                        "component_labels": encoded_array(struct.pack(
                            "<12i",
                            0, 4, 4, 0,
                            0, 4, 5, 0,
                            5, 0, 0, 0,
                        ), "<i4", (3, 4)),
                        "components": {"__tuple__": components},
                    }),
                    runtime_record("DensityEvidence", {
                        "component_id": 4,
                        "profile": runtime_record("DensityProfile", profile),
                        "final_field": encoded_array(
                            struct.pack("<dddd", 0.0, 0.25, 0.5, 1.0),
                            "<f8", (2, 2)),
                        "p70_threshold": 0.2,
                        "p70_mask": encoded_array(
                            b"\x00\x01\x01\x01", "|u1", (2, 2)),
                        "p80_threshold": 0.4,
                        "p80_mask": encoded_array(
                            b"\x00\x00\x01\x01", "|u1", (2, 2)),
                        "p90_threshold": 0.7,
                        "p90_mask": encoded_array(
                            b"\x00\x00\x00\x01", "|u1", (2, 2)),
                    }),
                ],
            }
            review_path = review_frames / f"{stem}.json"
            review_path.write_text(json.dumps(frame_payload), encoding="utf-8")

            accepted = {
                "frame_id": frame_id,
                "sim_time_ns": sim_time_ns,
                "component_id": 4,
                "route": "standard_gate",
                "fitter": "standard_component_confident_quad_v1",
                "selected_density_profile": profile,
                "fitted_corners_uv": [[1.0, 0.0], [2.0, 0.0],
                                        [2.0, 1.0], [1.0, 1.0]],
                "fit_confidence": 0.91,
                "high_confidence_threshold": 0.8,
                "high_confidence": True,
                "accepted": True,
                "rejection_reason": None,
            }
            clipped = {
                **accepted,
                "component_id": 5,
                "selected_density_profile": {**profile,
                                             "profile_id": "scale_01"},
                "fitted_corners_uv": None,
                "fit_confidence": 0.0,
                "high_confidence": False,
                "accepted": False,
                "rejection_reason": "standard_frame_edge_clipped",
            }
            runtime_result = {
                "frame_id": frame_id,
                "sim_time_ns": sim_time_ns,
                "camera_calibration": {},
                "gate_model": {},
                "pnp_relative_pose_estimates": [],
                "camera_pose_estimates": [],
                "standard_gate_results": [accepted, clipped],
            }
            (reviews / run_id / "gate-geometry-pnp-runtime.json").write_text(
                json.dumps({
                    "summary": {"format_version": 2},
                    "frames": [{
                        "run_id": run_id,
                        "runtime_result": runtime_result,
                    }],
                }), encoding="utf-8")

            catalog = standard_gate_result_catalog(run_id, logs, reviews)

            self.assertEqual(catalog["standard_gate_result_count"], 2)
            self.assertEqual(
                catalog["accepted_standard_gate_result_count"], 1)
            self.assertEqual(
                catalog["high_confidence_standard_gate_result_count"], 1)
            self.assertEqual(catalog["missing_density_evidence_count"], 1)
            first, second = catalog["results"]
            self.assertEqual(
                first["ComponentObservation"]["bbox_xywh"], [1, 0, 2, 2])
            self.assertEqual(
                first["StandardGateResult"]["fit_confidence"], 0.91)
            self.assertEqual(
                first["DensityEvidence"]["p70_threshold"], 0.2)
            self.assertIn(
                "DensityEvidence[scale_03].p80_mask", first["layers"])
            self.assertEqual(second["DensityEvidence"], None)
            self.assertEqual(
                tuple(second["layers"]), ("FrameObservation.closed_mask",))

            closed_png = render_standard_gate_component_layer(
                review_path, 4, "scale_03", "FrameObservation.closed_mask")
            p90_png = standard_gate_review_layer(
                run_id, frame_id, sim_time_ns, 4,
                "DensityEvidence[scale_03].p90_mask", logs, reviews)
            self.assertEqual(struct.unpack(">II", closed_png[16:24]), (2, 2))
            self.assertEqual(struct.unpack(">II", p90_png[16:24]), (2, 2))
            self.assertEqual(
                grayscale_png_pixels(closed_png),
                (2, 2, b"\xff\xff\xff\x00"),
            )

    def test_standard_gate_catalog_rejects_mismatched_result_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            reviews = root / "reviews"
            run_id = "run-example"
            frame_dir = logs / run_id / "vision_frames"
            frame_dir.mkdir(parents=True)
            (frame_dir / "frame-00000001-2.jpg").write_bytes(b"jpeg")
            output = reviews / run_id
            output.mkdir(parents=True)
            (output / "geometry.json").write_text(json.dumps({"frames": [{
                "run_id": run_id,
                "runtime_result": {
                    "frame_id": 1,
                    "sim_time_ns": 2,
                    "camera_calibration": {},
                    "gate_model": {},
                    "pnp_relative_pose_estimates": [],
                    "camera_pose_estimates": [],
                    "standard_gate_results": [{
                        "frame_id": 9,
                        "sim_time_ns": 2,
                        "component_id": 1,
                        "selected_density_profile": {"profile_id": "scale_01"},
                    }],
                },
            }], "summary": {"format_version": 2}}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "identity"):
                standard_gate_result_catalog(run_id, logs, reviews)

    def test_confined_paths_reject_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                _confined(root, "../outside.json")


if __name__ == "__main__":
    unittest.main()
