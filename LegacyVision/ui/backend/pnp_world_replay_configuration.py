"""Code-owned settings for read-only PnP world replay."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PnpWorldReplayConfiguration:
    """Review-only gates; none of these values affect production inference."""

    maximum_alignment_error_ms: float = 100.0
    require_exact_frame_cycle: bool = True
    require_logged_camera_mount: bool = True
    render_provisional_orientation: bool = True

    def json_value(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_PNP_WORLD_REPLAY_CONFIGURATION = PnpWorldReplayConfiguration()
