from app.callbacks import _ned_point_to_plot, _playback_frame_indices, _playback_interval_ms
from app.data import RunLog


def test_playback_interval_uses_recorded_simulation_step() -> None:
    run = RunLog(
        path="run.json",
        name="run",
        timestamp=None,
        schema_version=1,
        metadata={},
        events=[],
        cycles=[
            {"sim_time_ns": 0},
            {"sim_time_ns": 20_000_000},
            {"sim_time_ns": 40_000_000},
        ],
        raw={},
    )

    assert _playback_interval_ms(run) == 33
    assert _playback_frame_indices(run) == [0, 2]


def test_playback_trajectory_converts_ned_down_to_screen_up() -> None:
    assert _ned_point_to_plot([1.0, 2.0, 3.0]) == [1.0, -2.0, -3.0]
