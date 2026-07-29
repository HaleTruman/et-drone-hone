from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mask_review_pipeline" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deterministic_vision import DeterministicVision
from pipeline import run_pipeline
from schema import VisionFrame
from test_pipeline import ring_with_gap, write_frame, write_lut


def test_live_call_matches_batch_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lut_path = root / "lut.npz"
        run_dir = root / "runs" / "run-test"
        frames_dir = run_dir / "vision_frames"
        review_dir = root / "review"
        frames_dir.mkdir(parents=True)
        write_lut(lut_path)

        mask = ring_with_gap()
        local_files = []
        for n in range(1, 8):  # 5 to earn VOID_EARN_STREAK reliability, a couple more after
            fp = frames_dir / f"frame-{n:06d}.png"
            write_frame(fp, mask)
            local_files.append(fp)

        # Batch path: existing file-based pipeline, unchanged.
        index = run_pipeline(run_dir, str(lut_path), review_dir)
        batch_track_ids, batch_reliable_at = [], None
        for i, item in enumerate(index["frames"]):
            void = json.loads(Path(item["json_path"]).read_text())["analysis"]["regions"][0]["voids"][0]
            batch_track_ids.append(void["track_id"])
            if void["reliable"] and batch_reliable_at is None:
                batch_reliable_at = i
        assert len(set(batch_track_ids)) == 1, "batch path should track one continuous void"
        assert batch_reliable_at is not None, "batch path never earned reliability"

        # Live path: read the same local files' bytes back and feed them through the live
        # entrypoint one at a time, exactly as a real caller would per io_specification.md.
        vision = DeterministicVision(lut_path=str(lut_path))
        live_reliable_at = None
        for i, fp in enumerate(local_files):
            sim_time_ns = 1_700_000_000_000_000_000 + i * (10**9 // 30)  # realistic 30Hz frame spacing
            frame = VisionFrame(frame_id=i, sim_time_ns=sim_time_ns, jpeg_bytes=fp.read_bytes())
            observation = vision.process_frame(frame)
            assert observation.frame_id == i
            assert observation.gates == [], "no telemetry.json reachable from a live frame_label -- expected"

            void_tracks = vision._tracker._void_tracks
            assert len(void_tracks) == 1, f"expected exactly one live void track, got {len(void_tracks)}"
            if void_tracks[0]["reliable"] and live_reliable_at is None:
                live_reliable_at = i

        assert live_reliable_at is not None, "live path never earned reliability"
        assert live_reliable_at == batch_reliable_at, (
            f"live path earned reliability at frame {live_reliable_at}, batch at {batch_reliable_at}"
        )
        assert vision._tracker._void_tracks[0]["id"] == batch_track_ids[0], (
            "live and batch paths minted different track_ids for the same frame sequence"
        )


if __name__ == "__main__":
    test_live_call_matches_batch_pipeline()
    print("test_deterministic_vision.py passed")
