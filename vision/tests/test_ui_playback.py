import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools import serve_vision_review as server  # noqa: E402


class UiPlaybackTests(unittest.TestCase):
    def write_frame(self, folder: Path, name: str = "frame-000001.jpg", size=(640, 360)) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        Image.new("RGB", size, (12, 34, 56)).save(path, format="JPEG")
        return path

    def test_playback_readiness_accepts_640x360_jpeg_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "frames"
            self.write_frame(source)
            payload = server.playback_readiness_payload({"sourceDir": str(source), "targetHz": 30})
            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["frameCount"], 1)
            self.assertEqual(payload["periodMs"], 33.333)

    def test_playback_readiness_rejects_wrong_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "frames"
            self.write_frame(source, size=(320, 180))
            payload = server.playback_readiness_payload({"sourceDir": str(source), "targetHz": 30})
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["badFrames"])
            self.assertIn("frame dimensions", [check["name"] for check in payload["checks"]])

    def test_deadline_miss_is_recorded_at_next_tick(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "frames"
            frame = self.write_frame(source)
            feeder = server.PlaybackFeeder(source, [frame], 30)

            with mock.patch.object(
                feeder,
                "_pipeline_status",
                return_value={"state": "running", "frameCountCompleted": 0, "latestCompletedFrame": None, "runId": "run-test"},
            ):
                feeder._set_status(fedCount=1)
                feeder._update_pipeline_progress()
                self.assertFalse(feeder.snapshot()["deadlineMissed"])

                feeder._record_deadline_if_needed(expected_completed_count=1, missed_frame=0)
                status = feeder.snapshot()
                self.assertTrue(status["deadlineMissed"])
                self.assertEqual(status["missedDeadlineCount"], 1)
                self.assertEqual(status["lastMissedFrame"], 0)


if __name__ == "__main__":
    unittest.main()
