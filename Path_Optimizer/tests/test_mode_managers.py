from core.modes import ControlMode, ControlModeManager, FlightMode, FlightModeManager


def test_flight_mode_manager_defaults_to_hover_and_selects_path_following() -> None:
    manager = FlightModeManager()

    assert manager.flight_mode == FlightMode.HOVER
    assert manager.set_mode(FlightMode.PATH_FOLLOWING) == FlightMode.PATH_FOLLOWING
    assert manager.flight_mode == FlightMode.PATH_FOLLOWING


def test_control_mode_manager_defaults_to_attitude_and_selects_rate() -> None:
    manager = ControlModeManager()

    assert manager.control_mode == ControlMode.ATTITUDE
    assert manager.set_mode(ControlMode.RATE) == ControlMode.RATE
    assert manager.control_mode == ControlMode.RATE
