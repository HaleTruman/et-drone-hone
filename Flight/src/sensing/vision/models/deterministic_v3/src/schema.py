from dataclasses import dataclass


CORNER_ROLES = ("upper_left", "upper_right", "lower_left", "lower_right")


@dataclass(frozen=True, slots=True)
class EllipseEstimate:
    ellipse_id: int
    run_id: str
    frame_id: str
    frame_index: int
    frame_count: int
    frame_size_px: tuple[int, int]
    center_px: tuple[float, float]
    semi_axes_px: tuple[float, float]
    angle_degrees: float
    source_area_px: int


@dataclass(frozen=True, slots=True)
class GeometrySource:
    """Identity linking derived geometry to its detector and ellipse instance."""
    detector: str
    run_id: str
    frame_id: str
    frame_index: int
    ellipse_id: int


@dataclass(frozen=True, slots=True)
class CornerGeometry:
    """A role-bearing image-space corner selected from a named contour source."""
    role: str
    point_px: tuple[int, int]
    source_contour: str


@dataclass(frozen=True, slots=True)
class CircleGeometry:
    """A rendered circle and the role-bearing corners on its circumference."""
    role: str
    center_px: tuple[int, int]
    radius_px: int
    tangent_corner_roles: tuple[str, ...]
    tangent_points_px: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class VoidDetectionFeatures:
    """Complete geometry rendered by the production detector for one ellipse."""
    outer_corners: tuple[CornerGeometry, ...]
    inner_corners: tuple[CornerGeometry, ...]
    outer_corner_circles: tuple[CircleGeometry, ...]
    inner_corner_circles: tuple[CircleGeometry, ...]
    connection_circles: tuple[CircleGeometry, ...]


@dataclass(frozen=True, slots=True)
class VoidDetectionGeometry:
    """Published production geometry for one frame-local ellipse instance."""
    source: GeometrySource
    ellipse: EllipseEstimate
    color_bgr: tuple[int, int, int]
    outer_corners: tuple[CornerGeometry, ...]
    inner_corners: tuple[CornerGeometry, ...]
    outer_corner_circles: tuple[CircleGeometry, ...]
    inner_corner_circles: tuple[CircleGeometry, ...]
    connection_circles: tuple[CircleGeometry, ...]


@dataclass(frozen=True, slots=True)
class VoidPositionEstimate:
    """Metric camera-frame position of one void, with its anisotropic error.

    The error ellipsoid is roughly 40:1, elongated along the viewing ray:
    `sigma_bearing_m` is the tight cross-ray direction and `sigma_range_m` the
    soft along-ray one. Consumers must keep them separate rather than reducing
    the estimate to an isotropic point.
    """
    source: GeometrySource
    bearing_unit: tuple[float, float, float]
    range_m: float
    position_m: tuple[float, float, float]
    sigma_bearing_m: float
    sigma_range_m: float
    scale_residual: float
    channel_ranges_m: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VoidNedEstimate:
    """Metric void position in the run-local NED frame, published as a ray.

    The estimate is a precise direction with a soft position along it (~40:1).
    Rebuild the covariance with `void_ned.covariance_ned` rather than treating
    `position_ned_m` as an isotropic point, which discards that structure.

    The NED origin is per-run, set at vehicle initialisation, so positions are
    comparable within a run and not across runs.
    """
    source: GeometrySource
    frame_time_ns: int
    camera_position_ned_m: tuple[float, float, float]
    direction_ned: tuple[float, float, float]
    range_m: float
    position_ned_m: tuple[float, float, float]
    sigma_bearing_m: float
    sigma_range_m: float
    scale_residual: float
    pose_gap_ns: int


@dataclass(frozen=True, slots=True)
class EllipseTrackRecord:
    track_id: str
    frame_index: int
    consecutive_frame_count: int
    ellipse_id: int
    center_px: tuple[float, float]
    previous_ellipse_id: int | None
    previous_center_px: tuple[float, float] | None
    iou_score: float


@dataclass(frozen=True, slots=True)
class TrackedVoidDetection:
    """Complete frame-local geometry associated with its persistent track."""
    track: EllipseTrackRecord
    geometry: VoidDetectionGeometry


@dataclass(frozen=True, slots=True)
class VoidDetectionFrame:
    """All artifacts published for one live frame.

    `frame_count` is the number of frames accepted by this detector instance,
    including this frame. Each detection retains its stream-local frame index.
    """
    frame_id: int
    sim_time_ns: int
    frame_count: int
    detections: tuple[TrackedVoidDetection, ...]


@dataclass(frozen=True, slots=True)
class GeometryLinkEvidence:
    """Geometry continuity evidence attached to one accepted ellipse-IoU link."""
    run_id: str
    track_id: str
    previous_frame_index: int
    frame_index: int
    previous_ellipse_id: int
    ellipse_id: int
    iou_score: float
    center_shift_px: float
    normalized_center_shift: float
    radius_scale_ratio: float
    aspect_ratio_delta: float
    angle_delta_degrees: float
    outer_corner_rmse_px: float
    outer_shape_rmse_normalized: float
    inner_corner_rmse_px: float | None
    inner_shape_rmse_normalized: float | None
    circle_radius_relative_delta: float | None
    occlusion_proxy: bool
    position_evidence: str
    orientation_evidence: str


@dataclass(frozen=True, slots=True)
class GeometryEvidenceReport:
    """Final-review report containing summaries and typed per-link evidence."""
    report_type: str
    source_files: tuple[str, ...]
    methodology: dict
    global_summary: dict
    per_run: tuple[dict, ...]
    iou_buckets: dict
    occlusion_comparison: dict
    correlations: dict
    findings: tuple[str, ...]
    limitations: tuple[str, ...]
    links: tuple[GeometryLinkEvidence, ...]


def geometry_source(detector, ellipse):
    return GeometrySource(
        detector, ellipse.run_id, ellipse.frame_id, ellipse.frame_index,
        ellipse.ellipse_id)
