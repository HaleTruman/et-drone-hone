"""Void position estimates lifted from the camera frame into run-local NED.

Completes the transform a rotation-only predictor leaves out: the camera
translation. A world ray takes its origin from the vehicle position plus the
mounting offset, and its direction from the attitude composed with the camera
extrinsic. Range along that ray comes from the gate's known metric size.

The extrinsic is a fixed upward tilt about the body-right axis composed with the
CV-to-FRD axis permutation. Its sign was fixed empirically rather than assumed:
rays cast from a static gate intersect at 0.25 m median residual at +20 degrees
and 3.74 m at -20, and sweeping the angle puts the best fit at 19-20 degrees.

Any object carrying `position_local_ned_m`, `attitude_quaternion` and
`sim_time_ns` serves as the pose, so a live VehicleState passes straight through
with no adapter. The vision service re-stamps that state to the frame's own time
before a backend sees it, so no clock reconciliation happens here.
"""
import math
import numpy as np
from .schema import VoidNedEstimate

CAMERA_TILT_DEG = 20.0
CAMERA_OFFSET_BODY_M = (0.0, 0.0, 0.0)  # optical centre from the IMU origin; UNMEASURED
_CV_TO_FRD = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

def _pitch(degrees):
    cos, sin = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])

BODY_FROM_CAMERA = _pitch(CAMERA_TILT_DEG) @ _CV_TO_FRD

def rotation_world_from_body(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def rotation_world_from_camera(quaternion):
    return rotation_world_from_body(quaternion) @ BODY_FROM_CAMERA

def camera_position_ned(pose):
    """Optical centre in NED, including the body-frame mounting offset."""
    offset = rotation_world_from_body(pose.attitude_quaternion) @ np.array(CAMERA_OFFSET_BODY_M)
    return np.array(pose.position_local_ned_m, dtype=float) + offset

def _floats(vector):
    """Plain floats, so published estimates stay JSON-serialisable."""
    return tuple(float(value) for value in vector)

def to_ned(estimate, pose, gap_ns=0):
    """One camera-frame VoidPositionEstimate plus a vehicle pose, in NED."""
    direction = rotation_world_from_camera(pose.attitude_quaternion) @ np.array(estimate.bearing_unit)
    direction = direction / np.linalg.norm(direction)
    origin = camera_position_ned(pose)
    return VoidNedEstimate(
        estimate.source, int(pose.sim_time_ns), _floats(origin), _floats(direction),
        estimate.range_m, _floats(origin + estimate.range_m * direction),
        estimate.sigma_bearing_m, estimate.sigma_range_m,
        estimate.scale_residual, gap_ns)

def to_ned_all(estimates, pose, gap_ns=0):
    """Every estimate from one frame; they all share that frame's pose."""
    return [to_ned(estimate, pose, gap_ns) for estimate in estimates]

def covariance_ned(estimate):
    """Position covariance: soft along the ray, tight across it."""
    direction = np.array(estimate.direction_ned)
    along = np.outer(direction, direction)
    return estimate.sigma_range_m ** 2 * along + \
        estimate.sigma_bearing_m ** 2 * (np.eye(3) - along)
