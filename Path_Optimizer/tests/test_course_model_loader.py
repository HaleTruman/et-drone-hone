from pathlib import Path

import numpy as np

from opt_engine.coords_unreal import scenario_waypoints_to_internal_m
from opt_engine.scenario_io import load_scenario


def test_course_model_loader_parses_targets_and_excludes_origin():
    course_path = Path(__file__).resolve().parent.parent / "course_model" / "targets-SimBlank-20260216_194648.json"
    scenario = load_scenario(course_path)

    assert scenario.targets is not None
    assert scenario.origin_target is not None
    assert "origin" in scenario.origin_target.actor_label.lower()

    # The sample file contains Course_01..Course_16 plus a single Origin marker.
    assert len(scenario.targets) == 16
    assert len(scenario.waypoints) == 16
    assert all("_origin" not in (t.actor_label.lower()) for t in scenario.targets)


def test_course_model_preserves_listed_order_minus_origin():
    course_path = Path(__file__).resolve().parent.parent / "course_model" / "targets-SimBlank-20260216_194648.json"
    scenario = load_scenario(course_path)
    assert scenario.targets is not None

    labels = [t.actor_label for t in scenario.targets]
    # The file order begins with Course_13, Course_14, Course_12, Course_11, Origin, Course_15...
    assert labels[0].endswith("Course_13")
    assert labels[1].endswith("Course_14")
    assert labels[2].endswith("Course_12")
    assert labels[3].endswith("Course_11")
    assert labels[4].endswith("Course_15")


def test_course_model_cm_to_m_conversion_matches_expected():
    course_path = Path(__file__).resolve().parent.parent / "course_model" / "targets-SimBlank-20260216_194648.json"
    scenario = load_scenario(course_path)

    pts_m = scenario_waypoints_to_internal_m(scenario)
    assert pts_m.shape[1] == 3

    # First course target is Course_13 at x=-5921.5065 cm => -59.215065 m.
    np.testing.assert_allclose(pts_m[0, 0], -59.21506500274598, rtol=0, atol=1e-6)


def test_course_model_axes_are_near_orthonormal():
    course_path = Path(__file__).resolve().parent.parent / "course_model" / "targets-SimBlank-20260216_194648.json"
    scenario = load_scenario(course_path)
    assert scenario.targets is not None

    t0 = scenario.targets[0]
    x = t0.axis_x.as_np()
    y = t0.axis_y.as_np()
    z = t0.axis_z.as_np()

    np.testing.assert_allclose(np.linalg.norm(x), 1.0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(y), 1.0, rtol=0, atol=1e-3)
    np.testing.assert_allclose(np.linalg.norm(z), 1.0, rtol=0, atol=1e-3)

    assert abs(float(np.dot(x, y))) < 1e-3
    assert abs(float(np.dot(x, z))) < 1e-3
    assert abs(float(np.dot(y, z))) < 1e-3


def test_waypoints_format_still_loads():
    scenario_path = Path(__file__).resolve().parent.parent / "data" / "scenarios" / "simple_demo.json"
    scenario = load_scenario(scenario_path)
    assert scenario.targets is None
    assert scenario.origin_target is None
