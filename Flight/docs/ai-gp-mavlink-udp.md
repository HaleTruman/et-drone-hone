# AI-GP Simulator MAVLink UDP Capture

Capture date: 2026-07-03

Endpoint tested: `udpin:127.0.0.1:14550`

## Current Runtime Contract

The runtime now uses only the MAVLink messages observed on the simulator UDP stream.

`src/sensing/telemetry/mavlink_client.py` caches raw simulator telemetry from:

- `HEARTBEAT`
- `TIMESYNC` replies
- `HIGHRES_IMU`
- `ACTUATOR_OUTPUT_STATUS`
- race-status `ENCAPSULATED_DATA`
- track-gate `DATA_TRANSMISSION_HANDSHAKE`/`ENCAPSULATED_DATA` if the simulator emits them again

The simulator did not emit MAVLink `ODOMETRY`, `LOCAL_POSITION_NED`, or `ATTITUDE` in this capture. Position, velocity, attitude, body rates, and local acceleration therefore come from `sensing/odometry/VehicleState`. The runtime flow is `telemetry = client.get_telemetry()`, then `telemetry = vehicle_state.update(telemetry)`.

The first IMU sample initializes attitude by rotating the measured accelerometer direction onto local NED up, using gravity as the only absolute reference. Yaw is unobservable from accelerometer alone, so it starts at zero and then propagates from gyro integration.

## Messages Observed

A 20 second passive capture on `127.0.0.1:14550` decoded these MAVLink message types:

| Message | Count | Approx rate | Runtime support | Notes |
| --- | ---: | ---: | --- | --- |
| `HIGHRES_IMU` | 2333 | 116.65 Hz | Cached | Raw accel/gyro is available. Pressure, mag, and temperature fields are present but `NaN`. |
| `ACTUATOR_OUTPUT_STATUS` | 1903 | 95.15 Hz | Cached | Actuator array is present; observed values were all zero while unarmed/idle. |
| `HEARTBEAT` | 199 | 9.95 Hz | Cached | Simulator reports MAV type `2`, autopilot `0`, base mode `65`, system status `3`. |
| `ENCAPSULATED_DATA` | 79 | 3.95 Hz | Partially decoded | Observed payloads were race-status packets, not track-gate chunks. |

When the client actively sent `TIMESYNC`, the simulator replied with `TIMESYNC`; this was not present in the passive-only capture.

## Messages Not Observed

These messages were not seen in the 20 second capture:

- `ODOMETRY`
- `LOCAL_POSITION_NED`
- `ATTITUDE`
- `DATA_TRANSMISSION_HANDSHAKE`
- track-gate `ENCAPSULATED_DATA` packets with payload type `2`
- `COLLISION`

The missing track-gate handshake/data means the authoritative gate map was not arriving over this UDP stream during the capture.

## Sample Payloads

`HEARTBEAT`:

```json
{
  "type": 2,
  "autopilot": 0,
  "base_mode": 65,
  "custom_mode": 0,
  "system_status": 3,
  "mavlink_version": 3
}
```

`HIGHRES_IMU`:

```json
{
  "time_usec": 299354518,
  "xacc": -2.999192237854004,
  "yacc": -0.00232541561126709,
  "zacc": -9.340286254882812,
  "xgyro": 0.0,
  "ygyro": 0.0,
  "zgyro": -0.0,
  "fields_updated": 63,
  "id": 0,
  "xmag": NaN,
  "ymag": NaN,
  "zmag": NaN,
  "abs_pressure": NaN,
  "diff_pressure": NaN,
  "pressure_alt": NaN,
  "temperature": NaN
}
```

`ACTUATOR_OUTPUT_STATUS`:

```json
{
  "time_usec": 299354518,
  "active": 15,
  "actuator": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

Race-status `ENCAPSULATED_DATA` payloads start with byte `1`, which matches the client constant `ENCAPSULATED_RACE_STATUS_MSG_ID`. Decoded through the current client, the latest observed race status was:

```json
{
  "sim_boot_time_ms": 299185,
  "race_start_boot_time_ms": -1,
  "race_finish_time_ns": -1,
  "active_gate_index": 0,
  "last_gate_race_time": -1
}
```

`TIMESYNC` reply observed when the client sent timesync requests:

```json
{
  "ts1": 1783093835109953300,
  "tc1": 1783093835111475300
}
```

## Code Impact

The current startup path is:

1. `main.py` connects to `MAVLINK_ENDPOINT`.
2. `MavlinkClient.connect()` succeeds because `HEARTBEAT` is present.
3. `MavlinkClient.subscribe_telemetry()` starts receiving messages.
4. `MavlinkClient.get_telemetry()` returns raw telemetry with `imu` populated and `odometry` unset.
5. `main.py` feeds `telemetry.imu` into its `VehicleState` instance.
6. `main.py` builds the flight-facing telemetry bundle whose `odometry` field is `vehicle_state.state`.
7. Reset readiness and guidance use that local estimate instead of a simulator pose packet.
