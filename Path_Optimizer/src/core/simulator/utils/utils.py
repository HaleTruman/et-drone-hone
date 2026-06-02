from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def src_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def initial_state(config: dict[str, Any]) -> np.ndarray:
    state = np.zeros(13, dtype=float)
    x0 = config["X_0"]
    state[0:3] = np.asarray(x0["p_0"], dtype=float)
    state[3:6] = np.asarray(x0["v_0"], dtype=float)
    state[6:10] = normalize_quaternion(np.asarray(x0["q_0"], dtype=float))
    state[10:13] = np.asarray(x0["omega_0"], dtype=float)
    return state


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Quaternion norm cannot be zero.")
    return q / norm


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = normalize_quaternion(q)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.sign(sinp) * np.pi / 2.0 if abs(sinp) >= 1.0 else np.arcsin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=float)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(euler, dtype=float)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return normalize_quaternion(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )
    )


def rk4_step(derivative, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    k1 = derivative(state, control)
    k2 = derivative(state + 0.5 * dt * k1, control)
    k3 = derivative(state + 0.5 * dt * k2, control)
    k4 = derivative(state + dt * k3, control)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    next_state[6:10] = normalize_quaternion(next_state[6:10])
    return next_state


def body_axes_from_quaternion(model, q: np.ndarray) -> np.ndarray:
    return model._rotation_matrix(normalize_quaternion(q))
