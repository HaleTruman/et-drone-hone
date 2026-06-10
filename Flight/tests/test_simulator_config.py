from simulator.config import PROJECT_ROOT, load_config


def test_default_simulator_config_selects_project_scenario() -> None:
    config = load_config()

    assert config.scenario == PROJECT_ROOT / "scenarios" / "idle_telemetry.json"
    assert config.transport == "udp"
    assert config.telemetry_hz == 100.0
    assert config.physics_hz == 100.0
    assert config.heartbeat_hz == 2.0
