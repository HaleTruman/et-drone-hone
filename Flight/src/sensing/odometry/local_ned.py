"""Local-NED odometry from MAVLink HIGHRES_IMU velocity output."""

from core.coordinates import quat_wxyz, vec3
from core.schemas import AttitudeSample, ImuSample, OdometryState, TelemetrySample


ZERO_VEC3 = (0.0, 0.0, 0.0)
IDENTITY_QUATERNION = (1.0, 0.0, 0.0, 0.0)


def initial_odometry_state(
    *,
    sim_time_ns: int = 0,
    position_local_ned_m: tuple[float, float, float] = ZERO_VEC3,
    velocity_local_ned_mps: tuple[float, float, float] = ZERO_VEC3,
    attitude_quaternion: tuple[float, float, float, float] = IDENTITY_QUATERNION,
    body_rates_frd_rps: tuple[float, float, float] = ZERO_VEC3,
    acceleration_local_ned_mps2: tuple[float, float, float] = ZERO_VEC3,
) -> OdometryState:
    return OdometryState(
        sim_time_ns=int(sim_time_ns),
        position_local_ned_m=vec3(position_local_ned_m),
        velocity_local_ned_mps=vec3(velocity_local_ned_mps),
        attitude_quaternion=quat_wxyz(attitude_quaternion),
        body_rates_frd_rps=vec3(body_rates_frd_rps),
        acceleration_local_ned_mps2=vec3(acceleration_local_ned_mps2),
    )


def reset_odometry(
    *,
    sim_time_ns: int = 0,
    position_local_ned_m: tuple[float, float, float] = ZERO_VEC3,
) -> OdometryState:
    return initial_odometry_state(sim_time_ns=sim_time_ns, position_local_ned_m=position_local_ned_m)


def integrate_highres_imu(
    state: OdometryState,
    imu: ImuSample,
    attitude: AttitudeSample | None = None,
) -> OdometryState:
    dt_s = max(0.0, (int(imu.sim_time_ns) - int(state.sim_time_ns)) / 1_000_000_000)
    previous_velocity = state.velocity_local_ned_mps
    current_velocity = imu.velocity_local_ned_mps
    integration_velocity = tuple(
        0.5 * (float(previous_velocity[index]) + float(current_velocity[index])) for index in range(3)
    )
    position = tuple(
        float(state.position_local_ned_m[index]) + integration_velocity[index] * dt_s for index in range(3)
    )
    return OdometryState(
        sim_time_ns=int(imu.sim_time_ns),
        position_local_ned_m=vec3(position),
        velocity_local_ned_mps=vec3(current_velocity),
        attitude_quaternion=attitude.attitude_quaternion if attitude else state.attitude_quaternion,
        body_rates_frd_rps=attitude.body_rates_frd_rps if attitude else imu.gyro_frd_rps,
        acceleration_local_ned_mps2=imu.acceleration_local_ned_mps2,
    )


def telemetry_from_odometry(
    odometry: OdometryState,
    *,
    imu: ImuSample | None = None,
    attitude: AttitudeSample | None = None,
    system_status: str | None = None,
    reset_count: int | None = None,
    diagnostic_odometry: dict | None = None,
) -> TelemetrySample:
    return TelemetrySample(
        sim_time_ns=odometry.sim_time_ns,
        odometry=odometry,
        imu=imu,
        attitude_sample=attitude,
        system_status=system_status,
        reset_count=reset_count,
        diagnostic_odometry=diagnostic_odometry,
        raw={"source": "local_ned_odometry"},
    )
