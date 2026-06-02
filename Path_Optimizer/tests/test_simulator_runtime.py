import json
from pathlib import Path

import numpy as np
import pytest

from core.modes import ControlMode, FlightMode, ModeSelection, SystemMode, validate_modes
from simulator.runtime import SimulatorRuntime
from simulator.scenario import load_scenario


SCENARIOS = Path(__file__).resolve().parents[1] / "scenarios"


def test_mode_compatibility_and_flight_only_vision_rejection() -> None:
    validate_modes(ModeSelection(SystemMode.ARMED, FlightMode.HOVER, ControlMode.POSITION_HOLD))
    validate_modes(ModeSelection(SystemMode.ARMED, FlightMode.LANDING, ControlMode.POSITION_HOLD))

    with pytest.raises(ValueError, match="flight-only"):
        validate_modes(
            ModeSelection(SystemMode.RACING, FlightMode.GATE_TRACKING, ControlMode.VISION_GATE),
            flight_only=True,
        )


def test_scenario_normalizes_initial_quaternion() -> None:
    scenario = load_scenario(SCENARIOS / "armed_hover.json")

    np.testing.assert_allclose(np.linalg.norm(scenario.initial_state[6:10]), 1.0)
    assert scenario.initial_state[2] == -1.0


def test_invalid_mode_event_faults_and_is_logged(tmp_path) -> None:
    runtime = SimulatorRuntime(load_scenario(SCENARIOS / "invalid_mode_fault.json"), telemetry_hz=10, realtime=False)
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
