"""Load the small global simulator configuration."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "settings.yaml"


@dataclass(frozen=True)
class SimulatorConfig:
    scenario: Path
    transport: str
    endpoint: str
    realtime: bool
    telemetry_hz: float
    physics_hz: float
    heartbeat_hz: float


def load_config(path: str | Path = DEFAULT_CONFIG) -> SimulatorConfig:
    config_path = Path(path)
    data = _read_settings(config_path)
    scenario = Path(data.get("scenario", "scenarios/idle_telemetry.json"))
    if not scenario.is_absolute():
        scenario = PROJECT_ROOT / scenario
    transport = str(data.get("transport", "udp"))
    if transport not in ("udp", "inprocess"):
        raise ValueError("transport must be udp or inprocess")
    telemetry_hz = float(data.get("telemetry_hz", 100.0))
    physics_hz = float(data.get("physics_hz", telemetry_hz))
    heartbeat_hz = float(data.get("heartbeat_hz", 2.0))
    if min(telemetry_hz, physics_hz, heartbeat_hz) <= 0.0:
        raise ValueError("simulator rates must be positive")
    return SimulatorConfig(
        scenario=scenario,
        transport=transport,
        endpoint=str(data.get("endpoint", "udpout:127.0.0.1:14550")),
        realtime=bool(data.get("realtime", True)),
        telemetry_hz=telemetry_hz,
        physics_hz=physics_hz,
        heartbeat_hz=heartbeat_hz,
    )


def _read_settings(path: Path) -> dict[str, str | bool]:
    settings: dict[str, str | bool] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"invalid simulator setting: {line}")
        value = value.strip()
        settings[key.strip()] = value.lower() == "true" if value.lower() in ("true", "false") else value
    return settings
