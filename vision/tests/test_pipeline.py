import json
import sys
import tempfile
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from pipeline import PipelineConfig, VisionPipeline  # noqa: E402


class PipelineSmokeTests(unittest.TestCase):
    def test_batch_one_frame_defaults_to_compact_instance_frame_output(self):
        source_dir = APP_ROOT / "assets" / "frame_runs" / "run-20260720T023225Z" / "vision_frames"
        if not source_dir.exists():
            self.skipTest("fixture frame directory is not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pipeline = VisionPipeline(
                PipelineConfig(
                    mode="batch",
                    source_dir=source_dir,
                    output_root=temp_root / "runs",
                    artifact_root=temp_root / "assets",
                    max_frames=1,
                )
            )
            status = pipeline.run()
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["frameCountAccepted"], 1)
            self.assertEqual(status["frameCountCompleted"], 1)
            self.assertEqual(status["outputMode"], "production")

            run_root = temp_root / "runs" / status["runId"]
            frame_path = run_root / "frames" / "frame_000000.json"
            latest_path = run_root / "latest.json"
            self.assertTrue(frame_path.exists())
            self.assertTrue(latest_path.exists())
            payload = json.loads(frame_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "instance-frame-mapping-v1")
            self.assertEqual(payload["frameOrdinal"], 0)
            self.assertIn("instances", payload)
            self.assertIsNone(payload["sourceFrame"]["workingPath"])
            self.assertIn("startupToFirstOutputMs", status["timingMs"])
            self.assertFalse(list((run_root / "color_masks" / "masks").glob("*.bin")))
            self.assertFalse((run_root / "color_masks" / "mask_manifest.json").exists())
            self.assertFalse(list((run_root / "source_frames").glob("*.jpg")))
            instance_path = temp_root / "assets" / "instance_tracking" / status["runId"] / "instance_mapping.json"
            self.assertFalse(instance_path.exists())

    def test_batch_one_frame_can_write_legacy_aggregate_debug_manifest(self):
        source_dir = APP_ROOT / "assets" / "frame_runs" / "run-20260720T023225Z" / "vision_frames"
        if not source_dir.exists():
            self.skipTest("fixture frame directory is not available")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            pipeline = VisionPipeline(
                PipelineConfig(
                    mode="batch",
                    source_dir=source_dir,
                    output_root=temp_root / "runs",
                    artifact_root=temp_root / "assets",
                    max_frames=1,
                    aggregate_debug_manifests=True,
                )
            )
            status = pipeline.run()
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["frameCountCompleted"], 1)
            self.assertEqual(status["outputMode"], "aggregate-debug")

            instance_path = temp_root / "assets" / "instance_tracking" / status["runId"] / "instance_mapping.json"
            self.assertTrue(instance_path.exists())
            payload = json.loads(instance_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["kind"], "instance-mapping-pose-estimation-v1")
            self.assertEqual(payload["frameCount"], 1)
            self.assertEqual(len(payload["frames"]), 1)
            self.assertIn("instances", payload)


if __name__ == "__main__":
    unittest.main()
