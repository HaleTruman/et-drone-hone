from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.main import ReviewState


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


def _wait_for_job(state: ReviewState) -> dict:
    deadline = time.time() + 20.0
    while time.time() < deadline:
        status = state.debug_status()
        if status["status"] != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("debug launcher did not finish")


class ReviewLauncherTests(unittest.TestCase):
    def test_debug_launcher_writes_inside_selected_run_folder_and_loads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_folder = Path(temp_dir) / "run-example"
            source_dir = run_folder / "vision_frames"
            _make_jpeg(source_dir)

            state = ReviewState()
            start_status = state.start_debug_run(str(run_folder))
            self.assertEqual(start_status["status"], "running")
            self.assertEqual(start_status["sourceDir"], str(source_dir.resolve()))
            self.assertEqual(start_status["outputRoot"], str((run_folder / "vision_run").resolve()))

            final_status = _wait_for_job(state)
            self.assertEqual(final_status["status"], "complete", final_status)
            run_root = Path(final_status["runRoot"])
            self.assertEqual(run_root.parent, (run_folder / "vision_run").resolve())
            self.assertTrue((run_root / "run_manifest.json").exists())
            self.assertTrue((run_root / "frames" / "frame_000000.json").exists())
            self.assertTrue((run_root / "debug" / "bboxing" / "frames" / "frame_000000.json").exists())

            manifest = state.manifest_snapshot()
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertEqual(manifest["frame_count"], 1)
            frame_entry = state.frame_entry("0")
            self.assertIsNotNone(frame_entry)
            assert frame_entry is not None
            self.assertTrue(state.is_allowed_source_path(Path(frame_entry["source_path"])))
            self.assertTrue(state.is_allowed_run_path(Path(frame_entry["final_json"])))
            self.assertEqual(_read_json(run_root / "status.json")["status"], "complete")

    def test_debug_launcher_rejects_run_folder_without_vision_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_folder = Path(temp_dir) / "run-example"
            run_folder.mkdir()

            state = ReviewState()
            with self.assertRaises(FileNotFoundError):
                state.start_debug_run(str(run_folder))


if __name__ == "__main__":
    unittest.main()
