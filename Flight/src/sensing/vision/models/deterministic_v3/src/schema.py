from dataclasses import dataclass
from typing import Any


CORNER_ROLES = ("upper_left", "upper_right", "lower_left", "lower_right")

STANDARD_ROUTE = "standard_gate"
C_SHAPE_ROUTE = "c_shape"
MULTI_GATE_ROUTE = "multi_gate"

STANDARD_TOPOLOGY = "standard"
MULTI_VOID_TOPOLOGY = "multi_void"
C_SHAPE_TOPOLOGY = "c_shape"
UNKNOWN_TOPOLOGY = "unknown"
CLIPPED_TOPOLOGY = "clipped"
TOPOLOGY_LABELS = (
    STANDARD_TOPOLOGY,
    MULTI_VOID_TOPOLOGY,
    C_SHAPE_TOPOLOGY,
    UNKNOWN_TOPOLOGY,
    CLIPPED_TOPOLOGY,
)
QUADRILATERAL_CORNER_ORDER = (
    "upper_left", "upper_right", "lower_right", "lower_left")


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
class ParentContourCorner:
    point_px: tuple[int, int]
    angle_degrees: float


@dataclass(frozen=True, slots=True)
class VoidDetectionFeatures:
    """Complete geometry rendered by the production detector for one ellipse."""
    outer_corners: tuple[CornerGeometry, ...]
    inner_corners: tuple[CornerGeometry, ...]
    parent_obtuse_corners: tuple[ParentContourCorner, ...]


@dataclass(frozen=True, slots=True)
class VoidDetectionGeometry:
    """Published production geometry for one frame-local ellipse instance."""
    source: GeometrySource
    ellipse: EllipseEstimate
    color_bgr: tuple[int, int, int]
    outer_corners: tuple[CornerGeometry, ...]
    inner_corners: tuple[CornerGeometry, ...]
    parent_obtuse_corners: tuple[ParentContourCorner, ...]


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


# Geometry-pipeline contracts.  The older ellipse contracts above remain in
# place until their current consumers are migrated to the topology-first path.


@dataclass(frozen=True, slots=True)
class TopologyEvidence:
    """Raw and post-close aperture evidence for one connected parent."""

    raw_hole_contours: tuple[Any, ...]
    raw_hole_areas_px2: tuple[float, ...]
    closed_hole_contours: tuple[Any, ...]
    closed_hole_areas_px2: tuple[float, ...]

    @property
    def raw_hole_count(self) -> int:
        return len(self.raw_hole_contours)

    @property
    def closed_hole_count(self) -> int:
        return len(self.closed_hole_contours)


@dataclass(frozen=True, slots=True)
class ContourNodeEvidence:
    """One closed-mask ``RETR_TREE`` node in full-frame UV coordinates."""

    contour_id: int
    component_id: int
    parent_contour_id: int | None
    child_contour_ids: tuple[int, ...]
    depth: int
    is_hole: bool
    area_px2: float
    points_uv: Any


@dataclass(frozen=True, slots=True)
class ContourEvidence:
    """Component contours and frame-tree references.

    ``outer_raw`` and ``outer_simplified`` are component-local.  The contour
    IDs refer to the owning frame's ``closed_contour_nodes`` catalog.
    """

    raw_hierarchy: Any | None
    closed_hierarchy: Any | None
    outer_raw: Any | None
    outer_simplified: Any | None
    outer_contour_id: int | None
    parent_contour_ids: tuple[int, ...]
    parent_child_contour_ids: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ComponentObservation:
    """Shared evidence for one post-close connected component.

    Component masks are intentionally not stored here.  They are reconstructed
    from the owning ``FrameObservation`` and ``component_id`` so the full-frame
    mask planes remain the single source of truth.
    """

    component_id: int
    bbox_xywh: tuple[int, int, int, int]
    image_origin_uv: tuple[int, int]
    analysis_shape: tuple[int, int]
    touches_frame: bool
    area_px: int
    fill_ratio: float
    solidity: float
    topology: TopologyEvidence
    contours: ContourEvidence
    distance_transform: Any


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """Frame identity and authoritative shared mask planes."""

    frame_id: int
    sim_time_ns: int
    image_shape: tuple[int, int]
    preprocessing_version: str
    base_mask: Any
    size_filtered_mask: Any
    closed_mask: Any
    component_labels: Any
    closed_contour_nodes: tuple[ContourNodeEvidence, ...]
    components: tuple[ComponentObservation, ...]
    input_component_count: int
    gated: bool
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class DensityProfile:
    """One named, resolved density calibration available to every fitter."""

    profile_id: str
    calibration_version: str
    maximum_component_area_px: int | None
    density_radius_px: int
    ridge_radius_px: int
    relative_cap: float
    inverse_gamma: float
    ridge_gamma: float


@dataclass(frozen=True, slots=True)
class DensityBankConfiguration:
    """Exact density profiles and eligibility policy active for one bank."""

    calibration_version: str
    ignore_frame_edge_clipped: bool
    profiles: tuple[DensityProfile, ...]


@dataclass(frozen=True, slots=True)
class CShapeDensityProfileRule:
    """Select one named density profile by component maximum dimension."""

    maximum_component_dimension_px: int | None
    density_profile_id: str


@dataclass(frozen=True, slots=True)
class CShapeConfiguration:
    """Versioned density-profile selection for the approved C-shape fitter."""

    configuration_version: str
    density_profile_rules: tuple[CShapeDensityProfileRule, ...]


@dataclass(frozen=True, slots=True)
class DensityEvidence:
    """Cached analytical field and percentile evidence for one profile."""

    component_id: int
    profile: DensityProfile
    final_field: Any
    positive_points_xy: Any
    positive_weights: Any
    p70_threshold: float
    p70_mask: Any
    p80_threshold: float
    p80_mask: Any
    p90_threshold: float
    p90_mask: Any


@dataclass(frozen=True, slots=True)
class ApertureCenterEvidence:
    """Distance-transform center and proportionality for one child aperture."""

    parent_contour_id: int
    child_contour_id: int
    center_uv: tuple[float, float]
    distance_peak_px: float
    diameter_ratio: float
    center_offset_ratio: float
    radial_balance_ratio: float
    radial_balance_median: float
    accepted: bool
    failed_checks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TopologyDecision:
    """Explainable routing result with authoritative frame provenance."""

    frame_id: int
    sim_time_ns: int
    component_id: int
    topology_label: str
    route: str | None
    accepted: bool
    touches_frame: bool
    raw_significant_holes: int
    closed_significant_holes: int
    significant_parent_child_contour_ids: tuple[tuple[int, int], ...]
    significant_child_counts_by_parent: tuple[tuple[int, int], ...]
    aperture_center_evidence: tuple[ApertureCenterEvidence, ...]
    foreground_distance_max_px: float
    exterior_void_area_px: int
    exterior_void_area_ratio: float
    exterior_void_max_distance_px: float
    exterior_void_depth_ratio: float
    exterior_void_span_px: int
    exterior_void_span_axis: str | None
    exterior_void_segment_uv: tuple[tuple[float, float], ...] | None
    density_profile_id: str | None
    density_p90_pixel_count: int
    density_p90_component_count: int
    classification_rule: str
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class CShapeRefinedLine:
    """One of the three visible lines retained by C-shape refinement."""

    side_index: int
    support_points: int
    p90_coefficients: tuple[float, float, float]
    contour_coefficients: tuple[float, float, float]
    refined_coefficients: tuple[float, float, float]
    raw_contour_delta_degrees: float
    weighted_contour_delta_degrees: float
    applied_contour_delta_degrees: float
    correction_capped: bool
    p90_rmse_px: float


@dataclass(frozen=True, slots=True)
class CShapeResult:
    """Auditable output of the accepted three-line C-shape construction."""

    frame_id: int
    sim_time_ns: int
    component_id: int
    image_shape: tuple[int, int]
    topology_label: str
    route: str
    fitter: str
    configuration_version: str
    selected_density_profile: DensityProfile
    p90_threshold: float
    visible_lines: tuple[CShapeRefinedLine, ...]
    missing_side_index: int | None
    closed_intersections_uv: tuple[tuple[float, float], ...] | None
    contour_exits_uv: tuple[tuple[float, float], ...] | None
    extension_lengths_px: tuple[float, float] | None
    extended_endpoints_uv: tuple[tuple[float, float], ...] | None
    completed_quadrilateral_uv: tuple[tuple[float, float], ...] | None
    quadrilateral_convex: bool | None
    quadrilateral_area_px2: float | None
    refined_mask_origin_uv: tuple[int, int]
    refined_mask_line_width_px: int | None
    refined_mask: Any
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class QuadrilateralEstimate:
    """One normalized image-space quadrilateral from a routed fitter."""

    frame_id: int
    sim_time_ns: int
    component_id: int
    image_shape: tuple[int, int]
    route: str
    fitter: str
    selected_density_profile: DensityProfile
    corner_order: tuple[str, ...]
    corners_uv: tuple[tuple[float, float], ...] | None
    p90_threshold: float
    p90_evidence_points: int
    area_px2: float | None
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """Versioned pinhole parameters consumed by camera-relative PnP."""

    calibration_id: str
    image_shape: tuple[int, int]
    camera_matrix: tuple[tuple[float, float, float], ...]
    distortion_coefficients: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PlanarGateModel:
    """Metric gate-centerline model and its ordered point correspondence."""

    model_id: str
    side_length_m: float
    corner_order: tuple[str, ...]
    object_points_m: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class CameraPoseEstimate:
    """Camera-optical pose for the same frame-local component identity."""

    frame_id: int
    sim_time_ns: int
    component_id: int
    route: str
    solver: str
    camera_calibration_id: str
    gate_model_id: str
    rotation_vector_model_to_camera: tuple[float, float, float] | None
    position_camera_m: tuple[float, float, float] | None
    candidate_count: int
    reprojection_rmse_px: float | None
    secondary_reprojection_rmse_px: float | None
    ambiguity_gap_px: float | None
    position_confidence: float
    orientation_confidence: float
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True, slots=True)
class GeometryFrameResult:
    """JSON-compatible production frontier through camera-relative PnP."""

    frame_id: int
    sim_time_ns: int
    preprocessing_version: str
    density_configuration: DensityBankConfiguration
    camera_calibration: CameraCalibration
    gate_model: PlanarGateModel
    processed_routes: tuple[str, ...]
    topology_decisions: tuple[TopologyDecision, ...]
    quadrilateral_estimates: tuple[QuadrilateralEstimate, ...]
    camera_pose_estimates: tuple[CameraPoseEstimate, ...]
