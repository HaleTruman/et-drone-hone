"""Valid system, flight, and control mode combinations."""

from dataclasses import dataclass

from .control_mode import ControlMode
from .flight_mode import FlightMode
from .system_mode import SystemMode


@dataclass(frozen=True)
class ModeSelection:
    system: SystemMode
    flight: FlightMode | None = None
    control: ControlMode | None = None


def validate_modes(selection: ModeSelection, *, flight_only: bool = False) -> None:
    system, flight, control = selection.system, selection.flight, selection.control
    if system in (SystemMode.IDLE, SystemMode.FINISHED, SystemMode.FAULT):
        if flight is None and control is None:
            return
        raise ValueError(f"{system.value} does not allow an active flight or control mode")

    allowed = {
        (SystemMode.ARMED, FlightMode.HOVER, ControlMode.POSITION_HOLD),
        (SystemMode.ARMED, FlightMode.HOVER, ControlMode.ATTITUDE),
        (SystemMode.ARMED, FlightMode.WAYPOINT, ControlMode.WAYPOINT_FOLLOW),
        (SystemMode.RACING, FlightMode.WAYPOINT, ControlMode.WAYPOINT_FOLLOW),
        (SystemMode.RACING, FlightMode.TRAJECTORY, ControlMode.TRAJECTORY_TRACK),
        (SystemMode.RACING, FlightMode.TRAJECTORY, ControlMode.MPCC_TRACKER),
    }
    if (system, flight, control) in allowed:
        return
    if system == SystemMode.ARMED and flight is None and control in (
        ControlMode.RATE_DIRECT,
        ControlMode.ATTITUDE,
        ControlMode.VELOCITY,
    ):
        return
    if system == SystemMode.RACING and flight is None and control == ControlMode.VELOCITY:
        return
    raise ValueError(
        f"unsupported mode combination: system={system.value}, "
        f"flight={flight.value if flight else '-'}, control={control.value if control else '-'}"
    )
