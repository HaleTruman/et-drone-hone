from typing import Any

import numpy as np


class GatePoseEstimator:
    def __init__(
        self,
        fx: float = 320.0,
        fy: float = 320.0,
        cx: float = 320.0,
        cy: float = 180.0,
        camera_tilt_deg: float = 20.0,
    ):
        self.intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.camera_tilt_deg = camera_tilt_deg

    def estimate_gate_pose(self, body_frame: Any) -> dict[str, Any]:
        raise NotImplementedError("Connect CNN keypoints or relative-pose output to PnP.")

    def camera_to_body_transform(self) -> np.ndarray:
        tilt = np.deg2rad(self.camera_tilt_deg)
        c, s = np.cos(tilt), np.sin(tilt)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def body_to_local_ned(self, vector_body: np.ndarray, rotation_body_to_ned: np.ndarray) -> np.ndarray:
        return np.asarray(rotation_body_to_ned, dtype=float) @ np.asarray(vector_body, dtype=float)
