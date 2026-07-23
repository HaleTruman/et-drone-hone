from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.pipeline import PipelineOptions, run_pipeline


def _make_jpeg(directory: Path, name: str = "frame_000000.jpg") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image_path = directory / name
    Image.new("RGB", (64, 36), (4, 5, 6)).save(image_path, quality=95)
    return image_path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    return payload


class PipelineSmokeTests(unittest.TestCase):
    def test_pipeline_production_writes_only_final_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            source_dir = tmp_path / "frames"
            _make_jpeg(source_dir)
            output_root = tmp_path / "runs"
            manifest = run_pipeline(
                PipelineOptions(
                    mode="batch",
                    source_dir=source_dir,
                    output_root=output_root,
                    run_id="run-prod",
                    debug=False,
                    max_frames=1,
                )
            )
            run_root = Path(manifest.run_root)
            self.assertTrue((run_root / "run_manifest.json").exists())
            self.assertTrue((run_root / "status.json").exists())
            self.assertTrue((run_root / "latest.json").exists())
            frame_files = sorted((run_root / "frames").glob("frame_*.json"))
            self.assertEqual(len(frame_files), 1)
            self.assertFalse((run_root / "debug").exists())
            self.assertEqual(_read_json(run_root / "run_manifest.json")["frame_count"], 1)
            self.assertEqual(_read_json(run_root / "status.json")["status"], "complete")

    def test_pipeline_debug_writes_one_json_per_stage_and_no_stage_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            source_dir = tmp_path / "frames"
            _make_jpeg(source_dir)
            output_root = tmp_path / "runs"
            manifest = run_pipeline(
                PipelineOptions(
                    mode="batch",
                    source_dir=source_dir,
                    output_root=output_root,
                    run_id="run-debug",
                    debug=True,
                    max_frames=1,
                )
            )
            run_root = Path(manifest.run_root)
            stages = [
                "color_masking",
                "bboxing",
                "clipping",
                "contouring",
                "pose_estimation",
                "instance_tracking",
            ]
            for stage in stages:
                frame_json = sorted((run_root / "debug" / stage / "frames").glob("frame_*.json"))
                self.assertEqual(len(frame_json), 1, stage)
                aggregate_json = list((run_root / "debug" / stage).glob("*.json"))
                self.assertEqual(aggregate_json, [])
            self.assertFalse((run_root / "debug" / "flight_bridge").exists())
            mask_files = list((run_root / "debug" / "color_masking" / "masks").glob("frame_*.bin"))
            self.assertEqual(len(mask_files), 1)
            manifest_payload = _read_json(run_root / "run_manifest.json")
            self.assertEqual(manifest_payload["frame_count"], 1)
            self.assertIn("vision_results", manifest_payload["stages"])
            self.assertNotIn("flight_bridge", manifest_payload["stages"])
            self.assertTrue(
                manifest_payload["frame_index"][0]["debug_artifacts"]["bboxing"].endswith(
                    "debug/bboxing/frames/frame_000000.json"
                )
            )
            frame_files = sorted((run_root / "frames").glob("frame_*.json"))
            self.assertEqual(len(frame_files), 1)
            frame_payload = _read_json(frame_files[0])
            self.assertIn("vision_results", frame_payload)
            self.assertNotIn("flight_bridge", frame_payload)


if __name__ == "__main__":
    unittest.main()
