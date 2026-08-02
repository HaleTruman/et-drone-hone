import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UiArchitectureTests(unittest.TestCase):
    def test_offline_viewer_assets_live_in_frontend(self):
        frontend = ROOT / "ui" / "frontend"
        for relative in (
                "index.html", "app.js", "source.html", "source_app.js",
                "topology.html", "topology_app.js",
                "pnp_scene.html", "pnp_scene_app.js", "pnp_scene_adapter.js",
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
        pnp_scene = (frontend / "pnp_scene.html").read_text(encoding="utf-8")
        self.assertIn('src="app.js?', primary)
        self.assertIn('href="source.html"', primary)
        self.assertIn('src="source_app.js?', source)
        self.assertIn('href="index.html"', source)
        self.assertIn('src="topology_app.js?', topology)
        self.assertIn('href="topology.html"', primary)
        self.assertIn('href="topology.html"', source)
        self.assertIn('src="pnp_scene_app.js?', pnp_scene)
        self.assertIn('three@0.165.0/build/three.module.js', pnp_scene)
        self.assertIn('href="source.html"', pnp_scene)
        self.assertIn('href="index.html"', pnp_scene)
        self.assertIn('href="topology.html"', pnp_scene)
        self.assertIn('href="pnp_scene.html"', primary)
        self.assertIn('href="pnp_scene.html"', source)
        self.assertIn('href="pnp_scene.html"', topology)

    def test_authoritative_review_processes_live_in_backend(self):
        backend = ROOT / "ui" / "backend"
        for relative in (
                "serve_review_ui.py",
                "replay_historic_run.py",
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

    def test_pnp_projection_consumes_exact_geometry_runtime_fields(self):
        frontend = ROOT / "ui" / "frontend"
        adapter = (frontend / "pnp_scene_adapter.js").read_text(encoding="utf-8")
        app = (frontend / "pnp_scene_app.js").read_text(encoding="utf-8")
        page = (frontend / "pnp_scene.html").read_text(encoding="utf-8")
        self.assertIn("PNP_SCENE_SCHEMA_READY = true", adapter)
        self.assertNotIn("PNP_SCENE_CALIBRATION", adapter)
        for field in (
                "camera_calibration", "gate_model", "camera_pose_estimates",
                "rotation_vector_model_to_camera", "position_camera_m",
                "reprojection_rmse_px", "position_confidence",
                "orientation_confidence", "accepted", "rejection_reason"):
            self.assertIn(field, adapter + app)
        self.assertIn("CameraPoseEstimate projection", page)
        self.assertIn("geometry_result_url", app)

    def test_tabs_share_exact_frame_navigation_state(self):
        frontend = ROOT / "ui" / "frontend"
        navigation = (frontend / "frame_navigation_state.js").read_text(
            encoding="utf-8")
        self.assertIn("sessionStorage", navigation)
        self.assertIn("frame.filename === remembered.filename", navigation)
        for relative in (
                "app.js", "source_app.js", "topology_app.js",
                "pnp_scene_app.js"):
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
