import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UiArchitectureTests(unittest.TestCase):
    def test_offline_viewer_assets_live_in_frontend(self):
        frontend = ROOT / "ui" / "frontend"
        for relative in (
                "index.html", "app.js", "source.html", "source_app.js",
                "topology.html", "topology_app.js",
                "standard_gate.html", "standard_gate_app.js",
                "standard_gate.css",
                "pnp_scene.html", "pnp_scene_app.js", "pnp_scene_adapter.js",
                "pnp_world_adapter.js", "pnp_world_config.js",
                "pnp_world_scene.js",
                "pnp_scene.css",
                "styles.css", "magnifier.js", "schema_json_cache.js",
                "schema_json_inspector.js", "schema_provenance.js",
                "frame_navigation_state.js",
                "data/.gitignore"):
            self.assertTrue((frontend / relative).is_file(), relative)
        self.assertFalse(any(
            "legacy" in path.name.lower() for path in frontend.iterdir()))

    def test_primary_and_source_entry_points_are_distinct(self):
        frontend = ROOT / "ui" / "frontend"
        primary = (frontend / "index.html").read_text(encoding="utf-8")
        source = (frontend / "source.html").read_text(encoding="utf-8")
        topology = (frontend / "topology.html").read_text(encoding="utf-8")
        standard_gate = (frontend / "standard_gate.html").read_text(
            encoding="utf-8")
        pnp_scene = (frontend / "pnp_scene.html").read_text(encoding="utf-8")
        self.assertIn('src="app.js?', primary)
        self.assertIn('href="source.html"', primary)
        self.assertIn('src="source_app.js?', source)
        self.assertIn('href="index.html"', source)
        self.assertIn('src="topology_app.js?', topology)
        self.assertIn('href="topology.html"', primary)
        self.assertIn('href="topology.html"', source)
        self.assertIn('src="standard_gate_app.js?', standard_gate)
        for page in (primary, source, topology, standard_gate, pnp_scene):
            self.assertIn('href="standard_gate.html"', page)
        self.assertIn('href="source.html"', standard_gate)
        self.assertIn('href="index.html"', standard_gate)
        self.assertIn('href="topology.html"', standard_gate)
        self.assertIn('href="pnp_scene.html"', standard_gate)
        self.assertIn('src="pnp_scene_app.js?', pnp_scene)
        self.assertIn('three@0.165.0/build/three.module.js', pnp_scene)
        self.assertIn('href="source.html"', pnp_scene)
        self.assertIn('href="index.html"', pnp_scene)
        self.assertIn('href="topology.html"', pnp_scene)
        self.assertIn('href="pnp_scene.html"', primary)
        self.assertIn('href="pnp_scene.html"', source)
        self.assertIn('href="pnp_scene.html"', topology)
        for page in (primary, source, topology, standard_gate, pnp_scene):
            self.assertIn(
                'styles.css?v=standard-gate-review-v1', page)
        self.assertIn(
            'standard_gate.css?v=standard-gate-review-v3', standard_gate)
        self.assertIn(
            'standard_gate_app.js?v=standard-gate-review-v3', standard_gate)
        self.assertIn('magnifier.js?v=canvas-v1', standard_gate)

    def test_authoritative_review_processes_live_in_backend(self):
        backend = ROOT / "ui" / "backend"
        for relative in (
                "serve_review_ui.py",
                "replay_historic_run.py",
                "pnp_world_replay.py",
                "pnp_world_replay_configuration.py",
                "schema_json.py",
                "diagnose_shared_pipeline_runtime.py",
                "validate_schema_review_dump.py"):
            self.assertTrue((backend / relative).is_file(), relative)

    def test_topology_crops_preserve_bbox_aspect_ratio(self):
        frontend = ROOT / "ui" / "frontend"
        script = (frontend / "topology_app.js").read_text(encoding="utf-8")
        styles = (frontend / "styles.css").read_text(encoding="utf-8")
        self.assertIn("Math.min(250 / width, 250 / height)", script)
        self.assertIn("targetX, targetY, targetWidth, targetHeight", script)
        self.assertIn("card.append(cropFrame, label)", script)
        self.assertIn("outline: 4px solid var(--topology-label-color)", styles)

    def test_standard_gate_tab_renders_only_serialized_component_evidence(self):
        frontend = ROOT / "ui" / "frontend"
        script = (frontend / "standard_gate_app.js").read_text(
            encoding="utf-8")
        page = (frontend / "standard_gate.html").read_text(encoding="utf-8")
        self.assertIn("/standard-gate-results", script)
        self.assertIn("RESULT_BATCH_SIZE = 200", script)
        self.assertIn('id="previous-batch"', page)
        self.assertIn('id="next-batch"', page)
        self.assertNotIn("IntersectionObserver", script)
        self.assertNotIn("image.loading = 'lazy'", script)
        for field in (
                "ComponentObservation.bbox_xywh",
                "FrameObservation.closed_mask",
                "fitted_corners_uv", "fit_confidence",
                "high_confidence_threshold", "accepted",
                "rejection_reason"):
            self.assertIn(field, script + page)
        self.assertNotIn("DensityEvidence.", script + page)
        self.assertNotIn("canvas: 'source'", script)
        self.assertIn(
            "if (!fitCanvas || !validCorners(corners))", script)
        self.assertNotIn("cv2", script)
        self.assertNotIn("fitLine", script)
        for owner in (
                "replay_historic_run.py::HistoricRunSource",
                "preprocessing.py::preprocess_frame",
                "quadrilateral_fitter.py::fit_standard_quadrilateral"):
            self.assertIn(owner, script)
        styles = (frontend / "standard_gate.css").read_text(encoding="utf-8")
        self.assertIn(
            "grid-template-columns: repeat(2, minmax(150px, 300px))",
            styles)

    def test_pnp_projection_consumes_exact_geometry_runtime_fields(self):
        frontend = ROOT / "ui" / "frontend"
        adapter = (frontend / "pnp_scene_adapter.js").read_text(encoding="utf-8")
        app = (frontend / "pnp_scene_app.js").read_text(encoding="utf-8")
        world_adapter = (frontend / "pnp_world_adapter.js").read_text(
            encoding="utf-8")
        world_config = (frontend / "pnp_world_config.js").read_text(
            encoding="utf-8")
        world_scene = (frontend / "pnp_world_scene.js").read_text(
            encoding="utf-8")
        page = (frontend / "pnp_scene.html").read_text(encoding="utf-8")
        self.assertIn("PNP_SCENE_SCHEMA_READY = true", adapter)
        self.assertNotIn("PNP_SCENE_CALIBRATION", adapter)
        for field in (
                "camera_calibration", "gate_model",
                "pnp_relative_pose_estimates", "candidates",
                "selected_candidate_rank", "candidate_rank",
                "camera_pose_estimates",
                "rotation_vector_model_to_camera", "position_camera_m",
                "reprojection_rmse_px", "position_confidence",
                "orientation_confidence", "accepted", "rejection_reason"):
            self.assertIn(field, adapter + app)
        self.assertIn("PnP evidence and final pose replay", page)
        self.assertIn("geometry_result_url", app)
        self.assertIn('id="raw-pnp-toggle"', page)
        self.assertIn('id="secondary-pnp-toggle"', page)
        self.assertIn('id="final-pose-toggle"', page)
        self.assertIn('id="historic-raw-pose-toggle"', page)
        self.assertIn('id="historic-final-pose-toggle"', page)
        self.assertNotIn('id="historic-pose-toggle"', page)
        self.assertIn('id="camera-projection-mode"', page)
        self.assertIn('id="open-3d-mode"', page)
        self.assertIn('id="fit-3d-evidence"', page)
        self.assertIn('id="world-camera-path-toggle"', page)
        self.assertIn('id="world-full-path-toggle"', page)
        self.assertIn('id="pnp-world-readout"', page)
        self.assertIn("RAW_PNP_COLOR = 0x52e8ff", app)
        self.assertIn("SECONDARY_PNP_COLOR = 0xffb347", app)
        self.assertIn("FINAL_POSE_COLOR = 0xff4fd8", app)
        self.assertIn("HISTORIC_RAW_PNP_COLOR = 0xffd84a", app)
        self.assertIn("HISTORIC_FINAL_POSE_COLOR = 0x55f29a", app)
        self.assertIn("/geometry-history", app)
        self.assertIn("/pnp-world-replay", app)
        self.assertIn("borderWidthPx: 8", app)
        self.assertIn("borderWidthPx: 4", app)
        self.assertIn("new Line2(geometry, material)", app)
        self.assertIn("new PnpWorldScene", app)
        self.assertIn("state.viewMode === WORLD_3D_MODE", app)
        self.assertIn("new OrbitControls(this.camera, canvas)", world_scene)
        self.assertIn("fitSelectedEvidence", world_scene)
        self.assertIn("historyRaw", world_scene)
        self.assertIn("historyFinal", world_scene)
        self.assertIn("position_local_ned_m", world_adapter + world_scene)
        self.assertIn("ui_projection", world_adapter + world_scene + page)
        self.assertIn("return new THREE.Vector3(Number(north), -Number(down)",
                      world_scene)
        self.assertIn("currentRawBorderPx: 8", world_config)
        self.assertIn("currentSecondaryBorderPx: 3", world_config)
        self.assertIn("currentFinalBorderPx: 4", world_config)
        self.assertIn("gridSizeM: 120", world_config)
        self.assertIn("gridDivisions: 60", world_config)
        self.assertNotIn("pathGrid", world_config + world_scene)
        self.assertNotIn("trajectoryDirection", world_scene)
        self.assertNotIn("rotation_world_from_camera", app + world_scene)
        self.assertNotIn("regressed_camera_pose_estimates", adapter + app +
                         world_adapter + world_scene + page)
        self.assertIn('"three/addons/"', page)

    def test_tabs_share_exact_frame_navigation_state(self):
        frontend = ROOT / "ui" / "frontend"
        navigation = (frontend / "frame_navigation_state.js").read_text(
            encoding="utf-8")
        self.assertIn("sessionStorage", navigation)
        self.assertIn("frame.filename === remembered.filename", navigation)
        for relative in (
                "app.js", "source_app.js", "topology_app.js",
                "standard_gate_app.js", "pnp_scene_app.js"):
            script = (frontend / relative).read_text(encoding="utf-8")
            self.assertIn("frame_navigation_state.js?v=1", script, relative)

    def test_production_src_does_not_import_ui(self):
        prohibited = (
            "deterministic_v3.ui", "from ..ui", "from .ui", "import ui")
        violations = []
        for path in (ROOT / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(value in text for value in prohibited):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual(violations, [])

    def test_primary_review_renders_exact_c_shape_schema_mask(self):
        frontend = ROOT / "ui" / "frontend"
        app = (frontend / "app.js").read_text(encoding="utf-8")
        inspector = (frontend / "schema_json_inspector.js").read_text(
            encoding="utf-8")
        self.assertIn("CShapeResult.refined_mask", app)
        self.assertIn("schema_records[CShapeResult]", app)
        self.assertIn("recordStarts(lines, recordName)", inspector)


if __name__ == "__main__":
    unittest.main()
