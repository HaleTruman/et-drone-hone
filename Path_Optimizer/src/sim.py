"""Run a flight-only drone simulator scenario."""

import argparse
from pathlib import Path

from simulator.config import DEFAULT_CONFIG, load_config
from simulator.runtime import SimulatorRuntime
from simulator.scenario import load_scenario
from simulator.mavlink_server import MavlinkServer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--scenario")
    parser.add_argument("--telemetry-hz", type=float)
    parser.add_argument("--physics-hz", type=float)
    parser.add_argument("--heartbeat-hz", type=float)
    parser.add_argument("--accelerated", action="store_true", help="Run without wall-clock pacing.")
    parser.add_argument("--transport", choices=("udp", "inprocess"))
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    scenario = load_scenario(Path(args.scenario) if args.scenario else config.scenario)
    runtime = SimulatorRuntime(
        scenario,
        telemetry_hz=args.telemetry_hz or config.telemetry_hz,
        physics_hz=args.physics_hz or config.physics_hz,
        heartbeat_hz=args.heartbeat_hz or config.heartbeat_hz,
        realtime=False if args.accelerated else config.realtime,
    )
    transport = args.transport or config.transport
    if transport == "udp":
        runtime.mavlink_server = MavlinkServer(runtime.simulator, args.endpoint or config.endpoint)
    path = runtime.run()
    print(f"Simulation log saved to {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
