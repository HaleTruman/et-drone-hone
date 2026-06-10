from core.modes import RaceMode, RaceModeManager


def test_race_mode_manager_defaults_to_hold_and_selects_race() -> None:
    manager = RaceModeManager()

    assert manager.race_mode == RaceMode.HOLD
    assert manager.set_mode(RaceMode.RACE) == RaceMode.RACE
    assert manager.race_mode == RaceMode.RACE
