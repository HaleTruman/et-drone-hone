from vision.src.config import DEFAULT_PRESET_PATH, ProjectionConfig
from vision.src.solve_pnp import solve_object_pose, square_points


def test_solve_pnp_accepts_synthetic_square_projection():
    config = ProjectionConfig.from_path(DEFAULT_PRESET_PATH)
    object_points = square_points(1.5)
    # A fronto-parallel 1.5m square centered 10m forward projects to 48px side length.
    image_points = [[296.0, 156.0], [344.0, 156.0], [344.0, 204.0], [296.0, 204.0]]
    result = solve_object_pose(
        pose_id="test-pose",
        image_points_px=image_points,
        object_points_m=object_points,
        camera=config.camera,
        max_reprojection_error_px=8.0,
        source="unit-test",
    )
    assert result["available"] is True
    assert abs(result["xyzCameraM"][2] - 10.0) < 0.05
    assert result["reprojectionErrorPx"] <= 0.01
