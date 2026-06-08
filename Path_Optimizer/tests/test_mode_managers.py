from core.modes import ControlMode, ControlModeManager, FlightMode, FlightModeManager


def test_flight_mode_manager_defaults_to_hover_and_selects_waypoint() -> None:
    manager = FlightModeManager()

    assert manager.flight_mode == FlightMode.HOVER
    assert manager.set_mode(FlightMode.WAYPOINT) == FlightMode.WAYPOINT
    assert manager.flight_mode == FlightMode.WAYPOINT


def test_control_mode_manager_defaults_to_attitude_and_selects_rate() -> None:
    manager = ControlModeManager()

    assert manager.control_mode == ControlMode.ATTITUDE
    assert manager.set_mode(ControlMode.RATE) == ControlMode.RATE
    assert manager.control_mode == ControlMode.RATE
