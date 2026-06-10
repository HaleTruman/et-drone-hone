import json
from pathlib import Path

import numpy as np

from core.modes import ModeState, RaceMode, SystemMode
from simulator.runtime import SimulatorRuntime
from simulator.scenario import load_scenario


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def test_runtime_modes_are_system_plus_race_intent() -> None:
    modes = ModeState(SystemMode.ARMED, RaceMode.HOLD)

    assert modes.system == SystemMode.ARMED
    assert modes.race == RaceMode.HOLD


def test_scenario_normalizes_initial_quaternion() -> None:
    scenario = load_scenario(SCENARIOS / "armed_hover.json")

    np.testing.assert_allclose(np.linalg.norm(scenario.initial_state[6:10]), 1.0)
    assert scenario.initial_state[2] == -1.0


def test_invalid_system_mode_event_faults_and_is_logged(tmp_path) -> None:
    scenario_path = tmp_path / "invalid_system.json"
    scenario_path.write_text(
        json.dumps(
            {
                "name": "invalid_system",
                "duration_s": 1.0,
                "initial_modes": {"system": "IDLE"},
                "events": [{"at_s": 0.1, "type": "set_modes", "system": "NOT_A_MODE"}],
            }
        ),
        encoding="utf-8",
    )
    runtime = SimulatorRuntime(load_scenario(scenario_path), telemetry_hz=10, realtime=False)
    path = runtime.run(tmp_path / "run.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert any(event["event"] == "fault" for event in payload["events"])
    assert payload["cycles"][-1]["modes"]["system"] == "FAULT"


def test_turbulence_run_is_deterministic(tmp_path) -> None:
    scenario = load_scenario(SCENARIOS / "turbulence_recovery.json")
    first = SimulatorRuntime(scenario, telemetry_hz=10, realtime=False).run(tmp_path / "first.json")
    second = SimulatorRuntime(scenario, telemetry_hz=10, realtime=False).run(tmp_path / "second.json")

    a = json.loads(first.read_text(encoding="utf-8"))["cycles"]
    b = json.loads(second.read_text(encoding="utf-8"))["cycles"]
    assert [cycle["simulator_truth"] for cycle in a] == [cycle["simulator_truth"] for cycle in b]
    assert a[-1]["disturbance"] == b[-1]["disturbance"]


def test_reset_increments_telemetry_reset_counter(tmp_path) -> None:
    runtime = SimulatorRuntime(load_scenario(SCENARIOS / "reset.json"), telemetry_hz=10, realtime=False)
    path = runtime.run(tmp_path / "run.json")
    cycles = json.loads(path.read_text(encoding="utf-8"))["cycles"]

    assert cycles[-1]["telemetry"]["reset_count"] == 1


def test_simulator_run_writes_telemetry_sidecar(tmp_path) -> None:
    runtime = SimulatorRuntime(load_scenario(SCENARIOS / "idle_telemetry.json"), telemetry_hz=10, realtime=False)
    path = runtime.run(tmp_path / "run-20260607T120000Z" / "run.json")

    payload = json.loads((path.parent / "telemetry.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["scenario"] == "idle_telemetry"
    assert payload["samples"][0]["cycle"] == 0
    assert payload["samples"][0]["telemetry"]["sim_time_ns"] == 0
