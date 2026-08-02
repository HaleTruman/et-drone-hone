from __future__ import annotations

import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "mask_review_pipeline" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_3d_pose_estimate import CX, CY, FX, FY, Final3DPoseEstimator, _new_entry
from pipeline import run_pipeline
from test_pipeline import ring_with_gap, write_frame, write_lut

BASE_EPOCH = 1700000000.0
FRAME_DT = 1.0 / 30.0


def _pixel_for(point: np.ndarray, r_wc: np.ndarray, c_t: np.ndarray) -> tuple[float, float]:
    d_cam = r_wc.T @ (point - c_t)
    return CX + FX * d_cam[0] / d_cam[2], CY + FY * d_cam[1] / d_cam[2]


def test_two_ray_recovery() -> None:
    target = np.array([0.0, 0.0, 10.0])
    r_wc = np.eye(3)
    c_a, c_b = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    estimator = Final3DPoseEstimator()
    entry = _new_entry()
    entry["last_pixel_count"] = 4000.0  # expected_distance == 10, matching the target's real depth

    estimator._fold_ray(entry, _pixel_for(target, r_wc, c_a), 1.0, r_wc, c_a)
    estimator._fold_ray(entry, _pixel_for(target, r_wc, c_b), 1.0, r_wc, c_b)
    position = estimator._solve(entry, c_b)

    assert np.linalg.norm(position - target) < 0.5, f"recovered {position}, expected near {target}"


def test_single_ray_with_prior_is_not_singular() -> None:
    target = np.array([0.0, 0.0, 10.0])
    r_wc = np.eye(3)
    c_t = np.array([-1.0, 0.0, 0.0])

    estimator = Final3DPoseEstimator()
    entry = _new_entry()
    entry["last_pixel_count"] = 4000.0
    estimator._fold_ray(entry, _pixel_for(target, r_wc, c_t), 1.0, r_wc, c_t)
    position = estimator._solve(entry, c_t)

    assert np.all(np.isfinite(position)), f"single-ray solve produced a non-finite result: {position}"
    distance = np.linalg.norm(position - c_t)
    assert 5.0 < distance < 20.0, f"expected roughly the prior's 10m depth, got distance={distance}"


def test_weak_parallax_lowers_confidence() -> None:
    target = np.array([0.0, 0.0, 10.0])
    r_wc = np.eye(3)
    estimator = Final3DPoseEstimator()

    strong = _new_entry()
    strong["last_pixel_count"] = 4000.0
    estimator._fold_ray(strong, _pixel_for(target, r_wc, np.array([-1.0, 0.0, 0.0])), 1.0, r_wc, np.array([-1.0, 0.0, 0.0]))
    estimator._fold_ray(strong, _pixel_for(target, r_wc, np.array([1.0, 0.0, 0.0])), 1.0, r_wc, np.array([1.0, 0.0, 0.0]))

    weak = _new_entry()
    weak["last_pixel_count"] = 4000.0
    estimator._fold_ray(weak, _pixel_for(target, r_wc, np.array([0.0, 0.0, 0.0])), 1.0, r_wc, np.array([0.0, 0.0, 0.0]))
    estimator._fold_ray(weak, _pixel_for(target, r_wc, np.array([0.001, 0.0, 0.0])), 1.0, r_wc, np.array([0.001, 0.0, 0.0]))

    assert estimator._confidence(weak) < estimator._confidence(strong)


def test_passed_is_sticky() -> None:
    estimator = Final3DPoseEstimator()
    entry = _new_entry()
    entry["position"] = np.array([0.0, 0.0, 10.0])

    estimator._check_passed(entry, np.eye(3), np.array([0.0, 0.0, 15.0]))  # camera now beyond the point
    assert entry["passed"]

    estimator._check_passed(entry, np.eye(3), np.array([0.0, 0.0, 0.0]))  # would otherwise say "in front"
    assert entry["passed"], "passed must be sticky -- a later frame can't un-pass a landmark"


def _write_telemetry(path: Path, poses: list[tuple[float, list[float], list[float]]]) -> None:
    samples = [
        {
            "wall_time_utc": dt.datetime.fromtimestamp(epoch, tz=dt.timezone.utc).isoformat(),
            "telemetry": {"vehicle_state": {"attitude_quaternion": quat, "position_local_ned_m": position}},
        }
        for epoch, quat, position in poses
    ]
    path.write_text(json.dumps({"samples": samples}))


def test_end_to_end_vision_observation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        lut_path = root / "lut.npz"
        run_dir = root / "runs" / "run-test"
        frames_dir = run_dir / "vision_frames"
        review_dir = root / "review"
        frames_dir.mkdir(parents=True)
        write_lut(lut_path)

        mask = ring_with_gap()
        poses = []
        last_frame_id = last_sim_time_ns = None
        for n in range(8):  # 5 to earn reliability (VOID_EARN_STREAK), a few more to accumulate rays
            epoch = BASE_EPOCH + n * FRAME_DT
            ns = int(round(epoch * 1e9))
            write_frame(frames_dir / f"frame-{n:06d}-{ns}.png", mask)
            last_frame_id, last_sim_time_ns = n, ns
            # Constant attitude, matching the frozen synthetic mask (no implied rotation) so the
            # 2D tracker's own rotation-compensated matching isn't fighting fabricated telemetry --
            # only position varies, which is all this wiring test needs.
            quat = [1.0, 0.0, 0.0, 0.0]
            position = [float(n) * 2.0, 0.0, 0.0]
            poses.append((epoch, quat, position))
        _write_telemetry(run_dir / "telemetry.json", poses)

        index = run_pipeline(run_dir, str(lut_path), review_dir)
        last = json.loads(Path(index["frames"][-1]["json_path"]).read_text())
        observation = last["analysis"]["vision_observation"]

        assert observation is not None
        assert observation["frame_id"] == last_frame_id
        assert observation["sim_time_ns"] == last_sim_time_ns
        assert observation["source"] == "vision"

        void_track_id = last["analysis"]["regions"][0]["voids"][0]["track_id"]
        gates = observation["gates"]
        assert gates, "expected at least one published gate by the last frame"
        gate = next(g for g in gates if g["gate_id"] == void_track_id)
        assert len(gate["position_local_ned"]) == 3
        assert 0.0 <= gate["position_confidence"] <= 1.0
        assert gate["orientation_local_ned_quat"] is None
        assert gate["orientation_confidence"] is None
        assert gate["trace"]["ray_count"] > 1


if __name__ == "__main__":
    test_two_ray_recovery()
    test_single_ray_with_prior_is_not_singular()
    test_weak_parallax_lowers_confidence()
    test_passed_is_sticky()
    test_end_to_end_vision_observation()
    print("test_final_3d_pose_estimate.py passed")
