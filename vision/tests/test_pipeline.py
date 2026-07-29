from __future__ import annotations

import json
import base64
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mask_review_pipeline" / "src"
sys.path.insert(0, str(SRC))

from final_2d_void_estimate import MAX_FINAL_VOID_SLOTS, FinalVoidEstimator
from mask import frame_to_mask
from mask_bridge import bridge_outer_hull
from mask_fill import fill_close_pixels
from pipeline import run_pipeline
from schema import FinalVoidEstimate, MaskFrame
from void_center import find_void_centers


MASK_RGB = (10, 20, 30)


def write_lut(path: Path) -> None:
    lut = np.zeros(1 << 24, dtype=np.uint8)
    lut[(MASK_RGB[0] << 16) | (MASK_RGB[1] << 8) | MASK_RGB[2]] = 1
    np.savez(path, lut=lut)


def write_frame(path: Path, mask: np.ndarray) -> None:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = (MASK_RGB[2], MASK_RGB[1], MASK_RGB[0])
    cv2.imwrite(str(path), image)


def ring_with_gap(size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[8:88, 8:88] = True
    mask[32:64, 32:64] = False
    mask[44:52, 8:33] = False
    return mask


def ring_no_gap(size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[8:88, 8:88] = True
    return mask


def ring_with_gap_at_edge(size: int = 96) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[8:88, 0:80] = True
    mask[32:64, 24:56] = False
    mask[44:52, 0:25] = False
    return mask


def two_rings(width: int = 400, height: int = 110, offset: int = 200, bridged: bool = False) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for x0 in (0, offset):
        mask[8:88, x0 + 8:x0 + 88] = True
        mask[32:64, x0 + 32:x0 + 64] = False
        mask[44:52, x0 + 8:x0 + 33] = False
    if bridged:
        mask[44:52, 88:offset + 8] = True  # directly connects the two rings into one component
    return mask


def test_mask_fill_bridge_and_void_center_stages() -> None:
    close_gap = np.zeros((16, 16), dtype=bool)
    close_gap[4:12, 3:7] = True
    close_gap[4:12, 10:14] = True
    close_frame = MaskFrame("close-frame.png", close_gap.shape[1], close_gap.shape[0], close_gap)
    assert fill_close_pixels(close_frame, 3).yellow_mask.any()

    base = ring_with_gap()
    frame = MaskFrame("frame.png", base.shape[1], base.shape[0], base)
    filled = fill_close_pixels(frame, 3)
    bridged = bridge_outer_hull(filled, bridge_max_px=25, min_mask_region_px=200)
    assert bridged.green_mask.any()
    analysis = find_void_centers(bridged, min_void_px=75, min_mask_region_px=200)
    assert analysis.regions
    assert analysis.regions[0].voids
    void = analysis.regions[0].voids[0]
    assert void.pixel_count >= 75
    assert 0 <= void.center[0] < frame.width
    assert 0 <= void.center[1] < frame.height


def test_lut_conversion_and_pipeline_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lut_path = root / "lut.npz"
        run_dir = root / "runs" / "run-test"
        frames_dir = run_dir / "vision_frames"
        review_dir = root / "review"
        frames_dir.mkdir(parents=True)
        write_lut(lut_path)
        frame_path = frames_dir / "frame-000001.png"
        source_mask = ring_with_gap()
        write_frame(frame_path, source_mask)
        for n in range(2, 6):
            write_frame(frames_dir / f"frame-00000{n}.png", source_mask)

        converted = frame_to_mask(frame_path, lut_path)
        assert np.array_equal(converted.base_mask, source_mask)

        index = run_pipeline(run_dir, str(lut_path), review_dir)
        frame_item = index["frames"][0]
        json_path = Path(frame_item["json_path"])
        assert json_path.exists()
        assert "npz_path" not in frame_item
        assert not list(review_dir.rglob("*.npz"))
        assert not list(review_dir.rglob("*.png"))

        metadata = json.loads(json_path.read_text())
        assert metadata["version"] == "mask-review.v2"
        analysis = metadata["analysis"]
        base_plane = analysis["bridge"]["fill"]["frame"]["base_mask"]
        base_bytes = base64.b64decode(base_plane["data"])
        packed_bits = np.unpackbits(np.frombuffer(base_bytes, dtype=np.uint8), bitorder="little")
        assert int(packed_bits[: source_mask.size].sum()) == int(source_mask.sum())
        assert len(base_bytes) == base_plane["byte_length"]
        assert base_plane["encoding"] == "base64-packed-lsb0"
        assert analysis["regions"][0]["track_id"].startswith("instance_track_")
        assert analysis["regions"][0]["voids"][0]["track_id"].startswith("void_track_")
        assert analysis["regions"][0]["voids"][0]["pixel_count"] >= 75
        assert analysis["regions"][0]["track_match"] is None  # first sighting: nothing to match against yet
        assert analysis["regions"][0]["voids"][0]["forecast"] is None  # only 1 history point so far, below MIN_HISTORY

        second_metadata = json.loads(Path(index["frames"][1]["json_path"]).read_text())
        second_region = second_metadata["analysis"]["regions"][0]
        assert second_region["track_match"] is not None
        assert second_region["track_match"]["area_agreement"] == 1.0
        assert second_region["track_match"]["iou"] == 1.0
        second_void = second_region["voids"][0]
        assert second_void["track_match"] is not None
        assert second_void["track_match"]["iou"] is None
        assert second_void["track_match"]["area_agreement"] == 1.0

        # streak=2 here: the void hasn't earned VOID_EARN_STREAK(5) yet, so its "cv" estimate
        # is gated off by reliability — only the (ungated) region estimate should publish.
        final_estimates = second_metadata["analysis"]["final_estimates"]
        assert final_estimates
        assert all(fe["source"] == "cv" for fe in final_estimates)
        assert all(0.0 <= fe["confidence"] <= 1.0 for fe in final_estimates)
        assert all(fe["missing"] == 0 for fe in final_estimates)
        assert {fe["track_id"] for fe in final_estimates} == {second_region["track_id"]}
        assert not second_void["reliable"]

        # frame index 3 (streak=4): still not reliable, void estimate still gated off.
        fourth_metadata = json.loads(Path(index["frames"][3]["json_path"]).read_text())
        fourth_void = fourth_metadata["analysis"]["regions"][0]["voids"][0]
        assert not fourth_void["reliable"]
        assert fourth_void["track_id"] not in {fe["track_id"] for fe in fourth_metadata["analysis"]["final_estimates"]}

        # frame index 4 (streak=5): the void just earned reliability — its "cv" estimate
        # should now appear alongside the region's.
        fifth_metadata = json.loads(Path(index["frames"][4]["json_path"]).read_text())
        fifth_region = fifth_metadata["analysis"]["regions"][0]
        fifth_void = fifth_region["voids"][0]
        assert fifth_void["reliable"]
        fifth_estimates = fifth_metadata["analysis"]["final_estimates"]
        assert {fe["track_id"] for fe in fifth_estimates} == {fifth_region["track_id"], fifth_void["track_id"]}


def test_void_ghost_pool_reconnection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lut_path = root / "lut.npz"
        run_dir = root / "runs" / "run-test"
        frames_dir = run_dir / "vision_frames"
        review_dir = root / "review"
        frames_dir.mkdir(parents=True)
        write_lut(lut_path)

        with_gap = ring_with_gap()
        no_gap = ring_no_gap()
        n = 1
        for _ in range(5):  # earn VOID_EARN_STREAK(5) reliability
            write_frame(frames_dir / f"frame-{n:06d}.png", with_gap)
            n += 1
        for _ in range(12):  # exceed VOID_MAX_MISSING(10): void gets pruned into the ghost pool
            write_frame(frames_dir / f"frame-{n:06d}.png", no_gap)
            n += 1
        for _ in range(3):  # void reappears at the identical position
            write_frame(frames_dir / f"frame-{n:06d}.png", with_gap)
            n += 1

        index = run_pipeline(run_dir, str(lut_path), review_dir)
        reconnections = []
        for item in index["frames"]:
            voids = json.loads(Path(item["json_path"]).read_text())["analysis"]["regions"][0]["voids"]
            reconnections.extend(v for v in voids if v["possible_reconnection"] is not None)

        assert reconnections, "expected the reappearing void to flag a possible_reconnection"
        assert reconnections[0]["reconnection_error_px"] < 5.0  # same position — should be a near-exact match


def test_void_survives_instance_merge() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lut_path = root / "lut.npz"
        run_dir = root / "runs" / "run-test"
        frames_dir = run_dir / "vision_frames"
        review_dir = root / "review"
        frames_dir.mkdir(parents=True)
        write_lut(lut_path)

        separate = two_rings(bridged=False)
        merged = two_rings(bridged=True)
        n = 1
        for _ in range(6):  # earn VOID_EARN_STREAK(5) reliability on both voids independently
            write_frame(frames_dir / f"frame-{n:06d}.png", separate)
            n += 1
        for _ in range(3):  # the two instance regions merge into one -- voids must survive it
            write_frame(frames_dir / f"frame-{n:06d}.png", merged)
            n += 1

        index = run_pipeline(run_dir, str(lut_path), review_dir)

        def voids_of(frame_item):
            regions = json.loads(Path(frame_item["json_path"]).read_text())["analysis"]["regions"]
            return [v for r in regions for v in r["voids"]]

        voids_before = voids_of(index["frames"][5])  # last separate frame
        assert len(voids_before) == 2, f"expected 2 distinct voids before the merge, got {voids_before}"
        assert all(v["reliable"] for v in voids_before)
        before_ids = {v["track_id"] for v in voids_before}

        for frame_item in index["frames"][6:]:  # every merged frame
            voids_after = voids_of(frame_item)
            after_ids = {v["track_id"] for v in voids_after}
            assert after_ids == before_ids, (
                f"void track_ids changed across the instance merge: {before_ids} -> {after_ids} "
                "(the merge must not orphan/re-mint void identity)"
            )
            # both voids now sit inside the one merged region
            assert len({v["region_track_id"] for v in voids_after}) == 1


def test_final_void_estimate_slot_assignment() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lut_path = root / "lut.npz"
        run_dir = root / "runs" / "run-test"
        frames_dir = run_dir / "vision_frames"
        review_dir = root / "review"
        frames_dir.mkdir(parents=True)
        write_lut(lut_path)

        mask = ring_with_gap()
        for n in range(1, 7):  # earn VOID_EARN_STREAK(5) reliability
            write_frame(frames_dir / f"frame-{n:06d}.png", mask)

        index = run_pipeline(run_dir, str(lut_path), review_dir)
        last = json.loads(Path(index["frames"][-1]["json_path"]).read_text())
        void = last["analysis"]["regions"][0]["voids"][0]
        assert void["reliable"]

        finals = last["analysis"]["final_void_estimates"]
        assert len(finals) == 1
        final = finals[0]
        assert final["final_void_id"] == 1
        assert final["source"] == "cv"
        assert final["clipped"] is False
        assert final["track_id"] == void["track_id"]
        assert 0.0 <= final["confidence"] <= 1.0


def test_final_void_estimate_clip_releases_slot() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lut_path = root / "lut.npz"
        run_dir = root / "runs" / "run-test"
        frames_dir = run_dir / "vision_frames"
        review_dir = root / "review"
        frames_dir.mkdir(parents=True)
        write_lut(lut_path)

        clean = ring_with_gap()
        edge = ring_with_gap_at_edge()
        n = 1
        for _ in range(6):  # earn reliability, hold slot 1
            write_frame(frames_dir / f"frame-{n:06d}.png", clean)
            n += 1
        for _ in range(3):  # drift to the frame edge -- clips
            write_frame(frames_dir / f"frame-{n:06d}.png", edge)
            n += 1

        index = run_pipeline(run_dir, str(lut_path), review_dir)

        before = json.loads(Path(index["frames"][5]["json_path"]).read_text())
        assert len(before["analysis"]["final_void_estimates"]) == 1

        after = json.loads(Path(index["frames"][-1]["json_path"]).read_text())
        void = after["analysis"]["regions"][0]["voids"][0]
        assert void["mask_clipping"]
        assert after["analysis"]["final_void_estimates"] == []


def test_final_void_estimate_survives_gap_then_evicts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lut_path = root / "lut.npz"
        run_dir = root / "runs" / "run-test"
        frames_dir = run_dir / "vision_frames"
        review_dir = root / "review"
        frames_dir.mkdir(parents=True)
        write_lut(lut_path)

        with_gap = ring_with_gap()
        no_gap = ring_no_gap()
        n = 1
        for _ in range(6):  # earn reliability, hold slot 1
            write_frame(frames_dir / f"frame-{n:06d}.png", with_gap)
            n += 1
        for _ in range(10):  # undetected but never clipped -- within the 15-frame horizon
            write_frame(frames_dir / f"frame-{n:06d}.png", no_gap)
            n += 1
        for _ in range(6):  # push past the 15-frame horizon -- must evict
            write_frame(frames_dir / f"frame-{n:06d}.png", no_gap)
            n += 1

        index = run_pipeline(run_dir, str(lut_path), review_dir)

        mid_gap = json.loads(Path(index["frames"][15]["json_path"]).read_text())  # missing=10
        finals = mid_gap["analysis"]["final_void_estimates"]
        assert len(finals) == 1
        assert finals[0]["final_void_id"] == 1
        assert finals[0]["source"] in ("mc", "forecast")
        assert finals[0]["clipped"] is False

        past_horizon = json.loads(Path(index["frames"][-1]["json_path"]).read_text())  # missing=16
        assert past_horizon["analysis"]["final_void_estimates"] == []


def test_final_void_estimate_slot_cap() -> None:
    estimator = FinalVoidEstimator()
    candidates = {
        f"void_track_{i:03d}": FinalVoidEstimate(0, (float(i), float(i)), "cv", 0.9, False, f"void_track_{i:03d}")
        for i in range(1, MAX_FINAL_VOID_SLOTS + 3)  # more candidates than slots
    }
    estimator._assign_slots(candidates)
    assert len(estimator._slots) == MAX_FINAL_VOID_SLOTS
    assert set(estimator._slots.values()) == set(range(1, MAX_FINAL_VOID_SLOTS + 1))


if __name__ == "__main__":
    test_mask_fill_bridge_and_void_center_stages()
    test_lut_conversion_and_pipeline_outputs()
    test_void_ghost_pool_reconnection()
    test_void_survives_instance_merge()
    test_final_void_estimate_slot_assignment()
    test_final_void_estimate_clip_releases_slot()
    test_final_void_estimate_survives_gap_then_evicts()
    test_final_void_estimate_slot_cap()
    print("test_pipeline.py passed")
