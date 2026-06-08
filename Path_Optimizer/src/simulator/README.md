# Flight-Only Simulator

The simulator reuses `core.quadrotor.model.Quadrotor` and exposes the same
MAVLink-shaped flight surfaces used by the live stack. It does not create or
process vision frames.

Run an accelerated in-process scenario from `Path_Optimizer`:

```powershell
$env:PYTHONPATH = "src"
python src/sim.py --scenario scenarios/armed_hover.json --transport inprocess --accelerated
```

Global runtime settings live in `src/simulator/config/settings.yaml`. That file
selects the default scenario, MAVLink endpoint, pacing mode, and simulator
rates. Scenario JSON files contain initial state, environment overrides, mode
transitions, references, and disturbances. Command-line flags override the
global settings for one run.

For a UDP MAVLink endpoint, omit `--transport inprocess`. Simulator runs are
saved below `logs/sim/`. New real or qualifier captures are saved below
`logs/runs/`; the Dash explorer still reads older `data/live_runs/` captures.
