from pathlib import Path

from opt_engine.scenario_io import discover_scenario_files


def test_discover_scenario_files_prefers_course_model(tmp_path: Path) -> None:
    course_dir = tmp_path / "course_model"
    legacy_dir = tmp_path / "data" / "scenarios"
    course_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)

    (legacy_dir / "legacy.json").write_text("{}", encoding="utf-8")
    (course_dir / "course.json").write_text("{}", encoding="utf-8")

    discovered = discover_scenario_files(tmp_path)
    assert [p.name for p in discovered] == ["course.json"]


def test_discover_scenario_files_falls_back_to_legacy(tmp_path: Path) -> None:
    (tmp_path / "course_model").mkdir(parents=True)
    legacy_dir = tmp_path / "data" / "scenarios"
    legacy_dir.mkdir(parents=True)

    (legacy_dir / "legacy.json").write_text("{}", encoding="utf-8")

    discovered = discover_scenario_files(tmp_path)
    assert [p.name for p in discovered] == ["legacy.json"]


def test_discover_scenario_files_empty_when_none_present(tmp_path: Path) -> None:
    (tmp_path / "course_model").mkdir(parents=True)
    (tmp_path / "data" / "scenarios").mkdir(parents=True)

    discovered = discover_scenario_files(tmp_path)
    assert discovered == []

