# Telemetry Simulator

Deterministic offline telemetry for the racing stack. It uses the existing
`quadrotor.model.Quadrotor` dynamics and `racing_stack.sim_harness` RK4 wrapper.
The drone starts powered off and can receive stabilized quaternion/thrust
targets shaped like MAVLink `SET_ATTITUDE_TARGET`. It emits only the MAVLink-side
data currently expected by `racing_stack.mavlink_bridge`:

- `HEARTBEAT`
- `TIMESYNC`
- `ATTITUDE`
- `HIGHRES_IMU`

The model position is intentionally not emitted. Position remains a
telemetry-integration estimate until vision integration is added.

Small seeded Gaussian noise is applied at the telemetry boundary to velocity,
acceleration, body rates, and attitude. Physics remains deterministic.

`HIGHRES_IMU.fields.velocity_local_ned_mps` is the expected simulator API
extension described by the local stack brief. It feeds the existing
`TelemetrySample.velocity_local_ned_mps` contract.

From `Path_Optimizer`:

```powershell
$env:PYTHONPATH = "src"
python -m telemetry_simulator --duration-s 1
```
