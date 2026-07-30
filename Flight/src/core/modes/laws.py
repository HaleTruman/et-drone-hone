"""Flight control law definitions.

Control laws describe how controller outputs are interpreted before they are
sent to the vehicle, including which command path is used and which limits are
enforced. The command conversion itself is intentionally not implemented yet.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .system_mode import SystemMode


class ControlLaw(str, Enum):
    DIRECT = "direct"
    DIRECT_ACRO = "direct_acro"
    RATE = "rate"
    RATE_ACRO = "rate_acro"
    ATTITUDE = "attitude"


@dataclass(frozen=True)
class ControlLawLimits:
    max_roll_deg: float | None = None
    max_pitch_deg: float | None = None
    max_yaw_rate_rps: float | None = None
    max_body_rate_rps: float | None = None
    min_thrust: float | None = None
    max_thrust: float | None = None


@dataclass(frozen=True)
class ControlLawDefinition:
    law: ControlLaw
    limits: ControlLawLimits = field(default_factory=ControlLawLimits)
    description: str = ""

    def validate_command(self, command: dict[str, Any]) -> None:
        raise NotImplementedError(f"{self.law.value} command validation is not implemented.")

    def to_mavlink_target(self, command: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError(f"{self.law.value} MAVLink target conversion is not implemented.")


DEFAULT_CONTROL_LAW_DEFINITIONS: dict[ControlLaw, ControlLawDefinition] = {
    ControlLaw.DIRECT: ControlLawDefinition(
        law=ControlLaw.DIRECT,
        description="Pass through low-level command payloads with minimal interpretation.",
    ),
    ControlLaw.DIRECT_ACRO: ControlLawDefinition(
        law=ControlLaw.DIRECT_ACRO,
        description="Pass through acro-style low-level commands.",
    ),
    ControlLaw.RATE: ControlLawDefinition(
        law=ControlLaw.RATE,
        description="Interpret commands as body-rate targets with configured rate and thrust limits.",
    ),
    ControlLaw.RATE_ACRO: ControlLawDefinition(
        law=ControlLaw.RATE_ACRO,
        description="Interpret commands as acro body-rate targets with configured rate and thrust limits.",
    ),
    ControlLaw.ATTITUDE: ControlLawDefinition(
        law=ControlLaw.ATTITUDE,
        description="Interpret commands as attitude targets with configured attitude, rate, and thrust limits.",
    ),
}


class ControlLawManager:
    def __init__(
        self,
        initial_law: ControlLaw = ControlLaw.ATTITUDE,
        definitions: dict[ControlLaw, ControlLawDefinition] | None = None,
    ):
        self.definitions = definitions if definitions is not None else DEFAULT_CONTROL_LAW_DEFINITIONS
        self.control_law = initial_law
        self._require_definition(initial_law)

    @property
    def definition(self) -> ControlLawDefinition:
        return self._require_definition(self.control_law)

    def set_law(self, law: ControlLaw) -> ControlLaw:
        self._require_definition(law)
        self.control_law = law
        return self.control_law

    def validate_command(self, command: dict[str, Any]) -> None:
        self.definition.validate_command(command)

    def to_mavlink_target(self, command: dict[str, Any]) -> dict[str, Any]:
        return self.definition.to_mavlink_target(command)

    def _require_definition(self, law: ControlLaw) -> ControlLawDefinition:
        try:
            return self.definitions[law]
        except KeyError as error:
            raise ValueError(f"No control law definition registered for {law.value}.") from error


@dataclass(frozen=True)
class ModeState:
    system: SystemMode
    control_law: ControlLaw = ControlLaw.ATTITUDE
