"""Simple state estimator setup with sample IMU readings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.schemas import MavlinkHighresImu
from sensing.odometry.state import IDENTITY_QUATERNION, VehicleStateEstimator


SAMPLE_PATH = ROOT / "examples" / "imu_data" / "imu_samples.json"
SAMPLE_TIME_BOOT_US = 1_000_000


def load_imu_sample(name: str) -> MavlinkHighresImu:
    samples = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))["imu_samples"]
    sample = samples[name]
    return MavlinkHighresImu(
        time_boot_us=SAMPLE_TIME_BOOT_US,
        acceleration_body_frd_mps2=tuple(sample["acceleration_body_frd_mps2"]),
        gyro_body_frd_rps=tuple(sample["gyro_body_frd_rps"]),
    )


pitch_up_sample = load_imu_sample("pitch_up")
roll_right_sample = load_imu_sample("roll_right")
yaw_right_sample = load_imu_sample("yaw_right")
stationary_sample = load_imu_sample("stationary")
falling_sample = load_imu_sample("falling")
climbing_sample = load_imu_sample("climbing")
translate_right_sample = load_imu_sample("translate_right")
translate_forward_sample = load_imu_sample("translate_forward")

state_estimator = VehicleStateEstimator()
state_estimator.last_imu_time_boot_us = 0

selected_sample = pitch_up_sample

print("SAMPLE:", selected_sample)
print("GYRO BIAS - ", state_estimator.gyro_bias_body_frd_rps)

print(
    f"---PRE UPDATE-- ",
    f"Attitude quat - {state_estimator.attitude_quaternion}  |  ",
    f"Attitude euler (deg) - {state_estimator.attitude_euler_frd_deg} |  ",
    f"Angular rates - {state_estimator.angular_velocity_body_frd_rps}  |  ",
    f"Angular accel - {state_estimator.angular_acceleration_body_frd_rps2}  |  ",
    f"Position - {state_estimator.position_local_ned_m}  |  ",
    f"--PRE UPDATE---\n",
    flush=True,
)

state_estimator.update(imu_data_t=selected_sample)

print(
    f"---POST UPDATE-- ",
    f"Attitude quat - {state_estimator.attitude_quaternion}  |  ",
    f"Attitude euler (deg) - {state_estimator.attitude_euler_frd_deg} |  ",
    f"Angular rates - {state_estimator.angular_velocity_body_frd_rps}  |  ",
    f"Angular accel - {state_estimator.angular_acceleration_body_frd_rps2}  |  ",
    f"Position - {state_estimator.position_local_ned_m}  |  ",
    f"--POST UPDATE---\n",
    flush=True,
)
