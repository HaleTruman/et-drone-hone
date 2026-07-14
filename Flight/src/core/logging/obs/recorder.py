from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OBSRecordingError(RuntimeError):
    """Raised when OBS recording cannot be controlled."""


@dataclass(frozen=True)
class OBSConfig:
    host: str = "192.168.1.11"
    port: int = 4455
    password: str = ""

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "OBSConfig":
        values = _read_dotenv(env_path or _find_dotenv())
        host = os.environ.get("OBS_WEBSOCKET_HOST") or values.get("OBS_WEBSOCKET_HOST") or cls.host
        port_value = os.environ.get("OBS_WEBSOCKET_PORT") or values.get("OBS_WEBSOCKET_PORT") or str(cls.port)
        password = os.environ.get("OBS_WEBSOCKET_PASSWORD") or values.get("OBS_WEBSOCKET_PASSWORD") or ""

        try:
            port = int(port_value)
        except ValueError as exc:
            raise OBSRecordingError(f"OBS_WEBSOCKET_PORT must be an integer, got {port_value!r}") from exc

        return cls(host=host, port=port, password=password)


class OBSRecorder:
    def __init__(self, config: OBSConfig | None = None) -> None:
        self.config = config or OBSConfig.from_env()
        self._client: Any | None = None
        self._target_path: Path | None = None

    def start_recording(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        client = self._connect()
        self._target_path = output_path if output_path.suffix else None
        record_dir = output_path.parent if output_path.suffix else output_path
        record_dir.mkdir(parents=True, exist_ok=True)

        try:
            client.set_record_directory(str(record_dir))
            client.start_record()
        except Exception as exc:  # noqa: BLE001 - OBS client raises transport/request-specific exceptions.
            raise OBSRecordingError(f"Failed to start OBS recording: {exc}") from exc

    def stop_recording(self) -> Path | None:
        client = self._connect()

        try:
            response = client.stop_record()
        except Exception as exc:  # noqa: BLE001
            raise OBSRecordingError(f"Failed to stop OBS recording: {exc}") from exc

        output_path = _response_output_path(response)
        target_path = self._target_path
        self._target_path = None

        if output_path is None or target_path is None:
            return output_path

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.resolve() == target_path.resolve():
            return target_path

        shutil.move(str(output_path), str(target_path))
        return target_path

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            import obsws_python as obs
        except ImportError as exc:
            raise OBSRecordingError(
                "Missing OBS WebSocket client dependency. Install requirements.txt or `pip install obsws-python`."
            ) from exc

        try:
            self._client = obs.ReqClient(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
            )
        except Exception as exc:  # noqa: BLE001
            raise OBSRecordingError(
                f"Failed to connect to OBS WebSocket at {self.config.host}:{self.config.port}: {exc}"
            ) from exc

        return self._client


_DEFAULT_RECORDER: OBSRecorder | None = None


def start_recording(path: str | Path) -> None:
    _get_default_recorder().start_recording(path)


def stop_recording() -> Path | None:
    return _get_default_recorder().stop_recording()


def _get_default_recorder() -> OBSRecorder:
    global _DEFAULT_RECORDER
    if _DEFAULT_RECORDER is None:
        _DEFAULT_RECORDER = OBSRecorder()
    return _DEFAULT_RECORDER


def _find_dotenv() -> Path | None:
    for parent in (Path.cwd(), *Path.cwd().parents):
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
    return None


def _read_dotenv(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}

    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _response_output_path(response: Any) -> Path | None:
    output_path = getattr(response, "output_path", None)
    if output_path is None:
        output_path = getattr(response, "outputPath", None)
    if output_path is None and isinstance(response, dict):
        output_path = response.get("outputPath") or response.get("output_path")
    return Path(output_path) if output_path else None
