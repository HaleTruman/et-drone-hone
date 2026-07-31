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
