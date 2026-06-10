# Flight-Only Simulator

The simulator reuses `core.quadrotor.model.Quadrotor` and exposes the same
MAVLink-shaped flight surfaces used by the live stack. It does not create or
process vision frames.

Global runtime settings live in `src/simulator/config/settings.yaml`. That file
selects the default scenario, MAVLink endpoint, pacing mode, and simulator
rates. Scenario JSON files contain initial state, environment overrides, mode
transitions, references, and disturbances.

For a UDP MAVLink endpoint, omit `--transport inprocess`. Simulator runs are
saved below `logs/sim/`. New real or qualifier captures are saved below
`logs/runs/`; the Dash explorer still reads older `data/live_runs/` captures.
