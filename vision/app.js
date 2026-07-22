const API = '/api/vision-review';
const PIPELINE_API = '/api/pipeline';
const COLOR_MASK_MANIFEST_URL = '/0721Vision/src/color_masks/output/mask_manifest.json';
const SOURCE_POLL_MS = 1250;
const VIEW_GAP = 10;
const MAIN_VIEW_COUNT = 2;
const AUX_RAIL_GAP = 10;
const AUX_PANE_GAP = 8;
const AUX_MIN_WIDTH = 160;
const AUX_MAX_WIDTH = 640;
const MASK_LAYER_COLORS = {
  '001': [255, 92, 67],
  '002': [85, 247, 255],
  '003': [255, 240, 42],
  '007': [170, 126, 255]
};
const MASK_BIT_FALLBACK_COLORS = [
  [255, 92, 67],
  [85, 247, 255],
  [255, 240, 42],
  [170, 126, 255],
  [99, 255, 149],
  [255, 179, 56],
  [128, 166, 255],
  [255, 120, 186]
];
const BBOX_FLOW_DEFAULTS = {
  minPixels: 303,
  maxBboxesPerFrame: 5000,
  fitTightness: 100,
  quadFitEnabled: false,
  quadFitMode: 'free-quad',
  targetAspect: 1,
  aspectTolerance: 0.55,
  quadThicknessPx: 1,
  edgeCoverageMin: 1,
  cornerMinPixels: 13,
  voidOverlapMaxRatio: 0.75,
  voidOverlapMinPixels: 685,
  showRawBboxes: true,
  quadOverlayOpacity: 0,
  viewMode: 'baseline',
  fovClip: {
    enabled: true,
    showOverlay: true,
    marginPx: 22,
    minContactPixels: 306,
    minContactRatio: 0.055,
    requireBboxTouch: true,
    warnOnly: true
  }
};
const LAYER_GROUP_OPTIONS = [
  ['group-1', 'Group 1'],
  ['group-2', 'Group 2'],
  ['group-3', 'Group 3'],
  ['group-4', 'Group 4'],
  ['group-5', 'Group 5']
];
const CORNER_TYPE_DEFS = {
  white: { label: 'White', category: 'hull', position: 'inside', base: 'hull', legacyMin: 'minMaskInsidePoints', legacyMax: 'maxMaskInsidePoints', maxDefault: 64 },
  black: { label: 'Black', category: 'hull', position: 'outside', base: 'hull', legacyMin: 'minMaskOutsidePoints', legacyMax: 'maxMaskOutsidePoints', maxDefault: 64 },
  orange: { label: 'Orange', category: 'void', position: 'inside', base: 'void', legacyMin: 'minMaskInsidePoints', legacyMax: 'maxMaskInsidePoints', maxDefault: 24 },
  green: { label: 'Green', category: 'void', position: 'outside', base: 'void', legacyMin: 'minMaskOutsidePoints', legacyMax: 'maxMaskOutsidePoints', maxDefault: 24 }
};
const CORNER_TYPE_KEYS = Object.keys(CORNER_TYPE_DEFS);
const OPTION_HELP = {
  sourceOpacityInput: 'Controls how strongly the source frame is visible beneath review overlays. Lower values make masks, bboxes, corners, square pose, and depth previews easier to inspect.',
  squarePoseToggle: 'Shows or hides the 3D square pose viewer in the second main frame pane.',
  squarePoseViewModeSelect: 'Camera locked uses the calibrated 640 x 360 pinhole camera; orbit inspect lets you rotate around the solved square in 3D.',
  squarePoseSolutionSelect: 'Chooses which square-pose signal to emphasize. Best overall uses all four sides; side options highlight top, right, bottom, or left diagnostics.',
  squarePoseSizeInput: 'Physical size of the square plane in meters. The default is 2.7m by 2.7m.',
  squarePoseEdgeToleranceInput: 'Pixel distance used when scoring contour support along each fitted square edge.',
  squarePoseMinCoverageInput: 'Minimum fraction of each side that must be supported by contour pixels before the pose is accepted.',
  squarePoseMaxErrorInput: 'Maximum average corner reprojection error allowed before a pose is flagged as weak.',
  squarePoseClipGuardToggle: 'When enabled, square-pose measurements from fully clipped bboxes are ignored for XYZ/RPY updates while bbox and instance tracking continue.',
  squarePoseClipThresholdInput: 'FOV clip severity required before the pose measurement is ignored. The default 100% means only bboxes marked clipped by the current clipping settings are ignored.',
  squarePoseCornerFitToggle: 'Uses white hull corners and green void corners as additional candidate-fit evidence for tighter, less ambiguous square poses.',
  squarePoseMinCornerLinksInput: 'Minimum strong white-to-green corner agreement links needed for corner agreement to strongly influence candidate scoring.',
  squarePoseCompleteBonusToggle: 'Rewards candidates with complete opposite white-green agreement structures.',
  squarePoseExtraCornerLimitInput: 'Number of white or green corners allowed before the object is flagged as possible multi-gate or overlap evidence.',
  squarePoseMaxCandidatesInput: 'Maximum square-pose candidates retained and scored per bbox.',
  squarePoseTextureOpacityInput: 'Visual opacity of the textured 2.7m square in the 3D viewer.',
  squarePoseCandidateOpacityInput: 'Visual opacity for the fitted outline and side signal overlays.',
  squarePoseFrustumToggle: 'Shows or hides the calibrated camera frustum in orbit inspect mode.',
  squarePoseOutlineToggle: 'Shows or hides the projected square outline.',
  squarePoseScoresToggle: 'Shows or hides per-side pose scores in the metadata readout.',
  squarePoseLabelsToggle: 'Shows per-object prediction labels anchored to each solved square center, using OpenCV camera-space XYZ meters and XYZ-order RPY degrees.',
  bboxMinPixelsInput: 'Minimum connected mask pixels required before a region can become a bbox. Increasing it removes small specks; decreasing it allows smaller regions.',
  bboxMaxPerFrameInput: 'Maximum number of bboxes retained per frame after sorting by masked pixel count. The largest regions are kept first.',
  bboxMaxPerFrameNumber: 'Exact numeric entry for the maximum number of bboxes retained per frame after sorting by masked pixel count.',
  bboxTightnessInput: 'Controls how tightly mask regions are separated before bbox creation. Higher values preserve tighter fits; lower values can merge nearby mask pixels into larger regions.',
  bboxQuadToggle: 'Enables the quad fit review layer on top of the bbox result. It does not change the raw connected-component bbox source.',
  bboxFitModeSelect: 'Chooses the fitted review shape: axis-aligned bbox, rotated rectangle, or free quadrilateral.',
  bboxTargetAspectInput: 'Preferred width-to-height ratio used when evaluating fitted quad shapes.',
  bboxAspectToleranceInput: 'Allowed deviation from the target aspect ratio. Higher values accept more varied shapes; lower values enforce the target more strictly.',
  bboxQuadThicknessInput: 'Thickness of the edge band used when checking whether mask pixels support a fitted quad edge.',
  bboxEdgeCoverageInput: 'Minimum fraction of each fitted edge that must be supported by mask pixels.',
  bboxCornerMinPixelsInput: 'Minimum number of mask pixels required near a fitted quad corner before that corner counts as supported.',
  bboxVoidOverlapInput: 'Maximum allowed ratio of unrelated mask pixels inside the fitted void before the fit is flagged as contaminated.',
  bboxVoidMinPixelsInput: 'Minimum unrelated pixels inside a void before that void overlap is counted.',
  bboxShowRawToggle: 'Shows or hides the raw axis-aligned bbox rectangles in the overlay.',
  bboxQuadOpacityInput: 'Controls opacity for bbox and quad review overlays.',
  fovClipToggle: 'Enables diagnostic-only detection for mask instances that are likely clipped by the camera field of view. When off, bbox behavior is unchanged.',
  fovClipOverlayToggle: 'Shows or hides the FOV clipping overlay without changing bbox build artifacts.',
  fovClipMarginInput: 'Pixel distance from the frame border used as the clipping contact strip. Larger values warn earlier near the frame edge.',
  fovClipMinPixelsInput: 'Minimum component pixels inside a border strip before that side can be marked clipped.',
  fovClipMinRatioInput: 'Minimum fraction of the instance pixels inside a border strip before that side can be marked clipped.',
  fovClipBboxTouchToggle: 'Requires the observed bbox to touch or nearly touch the frame border before a side can be considered clipped.',
  fovClipWarnOnlyToggle: 'Stores clipping as measurement-quality metadata only. It does not change bbox, contour, corner, or pose scoring.',
  instanceToggle: 'Enables the instance provenance review layer. This creates and displays persistent frame-to-frame instance IDs without changing bbox or pose outputs.',
  instanceOverlayToggle: 'Shows or hides instance provenance overlays in the source frame review.',
  instanceLabelsToggle: 'Shows compact instance ID labels beside current-frame bboxes.',
  instanceLinksToggle: 'Shows accepted frame-to-frame association links from prior observation centers to current centers.',
  instanceCandidatesToggle: 'Shows rejected or alternate candidate association links for debugging. This is hidden by default to reduce clutter.',
  instanceReadoutToggle: 'Shows the current-frame instance provenance JSON readout in the right review rail.',
  instancePoseSourceSelect: 'Chooses which pose measurement source instance provenance uses for 3D association and XYZ/RPY readouts.',
  instanceMaxGapInput: 'Maximum frame gap an instance can remain active for association after its last observation.',
  instanceMinScoreInput: 'Minimum combined association score required before a current bbox can adopt an existing instance ID.',
  instanceMax2dInput: 'Maximum 2D center distance used to normalize and gate frame-to-frame association.',
  instanceMax3dInput: 'Maximum 3D XYZ distance used to normalize pose association when square pose is available.',
  instanceMaxRpyInput: 'Maximum RPY angle delta used to normalize pose orientation association when square pose is available.',
  instanceSplitScoreInput: 'Candidate score threshold used to flag possible split or merge provenance events without rewriting history.',
  instanceTrailInput: 'Number of recent frames shown in the instance trail overlay.',
  instanceCandidateLimitInput: 'Maximum candidate associations stored and shown per current-frame observation.',
  contourToggle: 'Shows or hides the contour hierarchy overlay. This is a live review toggle and does not rebuild contour data.',
  contourSourceSelect: 'Chooses which decoded mask pixels are used for contour hierarchy analysis. Enabled layers follows the current combined mask; Layer 002 only inspects the strict support layer.',
  contourMinOuterInput: 'Minimum outer contour area required before a bbox region receives an outer boundary.',
  contourMinVoidInput: 'Minimum internal void or nested contour area kept for review.',
  contourMaxVoidsInput: 'Maximum direct void boundaries kept per bbox after sorting by area. Zero hides voids from the contour manifest.',
  contourSimplifyInput: 'Contour simplification amount before paths are saved. Higher values produce cleaner, fewer-point boundaries; lower values preserve jagged pixel detail.',
  contourCloseInput: 'Morphological close radius applied to the analysis mask before contour extraction. Higher values bridge small breaks.',
  contourOpenInput: 'Morphological open radius applied before contour extraction. Higher values remove small isolated specks.',
  contourNotchInput: 'Distance threshold for flagging void boundaries that are close to the outer contour or bbox edge as notch candidates.',
  contourLineThicknessInput: 'Visual line thickness for contour hierarchy paths.',
  contourOpacityInput: 'Visual opacity for contour hierarchy paths and labels.',
  contourLabelsToggle: 'Shows labels for outer, void, nested, and notch contour paths.',
  contourSmallToggle: 'Includes below-threshold contour records in the output for debugging. This can make the manifest and overlay noisier.',
  cornerShowAmbiguousToggle: 'Shows ambiguous corner candidates that did not clearly classify as inside or outside.',
  cornerOpacityInput: 'Controls opacity for corner points and corner direction rays.',
  cornerAgreementToggle: 'Shows or hides yellow white-to-green corner agreement links. This is a live review overlay and does not rebuild corner data.',
  cornerAgreementMaxDistanceInput: 'Maximum pixel distance from a white hull-inside corner to a green void-outside corner candidate.',
  cornerAgreementWhiteToleranceInput: 'How closely the white corner direction ray must point toward the green corner. Lower is stricter.',
  cornerAgreementAngleToleranceInput: 'How close the white and green corner opening angles must be to count as compatible.',
  cornerAgreementMinScoreInput: 'Minimum combined match score after distance, white direction, and angle agreement are evaluated.',
  cornerAgreementMaxLinksInput: 'Maximum agreement links allowed per white corner and per green corner. One keeps the overlay one-to-one.',
  cornerAgreementLineThicknessInput: 'Visual thickness of the yellow agreement lines.',
  cornerAgreementOpacityInput: 'Visual opacity of the yellow agreement lines.',
  cornerAgreementStructureToggle: 'Marks complete agreement structures when four links in the same bbox and void have opposite supporting pairs.',
  cornerAgreementOppositeToleranceInput: 'Tolerance for deciding whether two agreement links are opposite enough to support a complete structure.',
  layer002QuadToggle: 'Shows or hides Layer 002 hull-sweep quads. The candidate starts around the outer mask hull, then searches inward for four straight edges with maximum Layer 002 support.',
  layer002TraceSupportSelect: 'Chooses the mask used to seed the outer hull. Any decoded mask starts from the full visible mask region; Layer 002 only starts from the strict 002 region.',
  layer002TraceForbiddenSelect: 'Chooses the hard rejection rule for edge samples. Non-002 is strict; Non-mask only allows other decoded mask layers but rejects empty space.',
  layer002QuadMaxInput: 'Maximum accepted Layer 002 hull-sweep quads shown for the current frame after scoring.',
  layer002QuadMinPixelsInput: 'Minimum Layer 002 pixels inside a bbox before hull-sweep fitting is attempted.',
  layer002QuadEdgeThicknessInput: 'Half-width of the sampled edge band used to test whether a straight candidate edge intersects Layer 002 pixels.',
  layer002QuadMinCoverageInput: 'Target fraction of edge samples that should intersect Layer 002 pixels. Lower values tolerate noisy edges; higher values penalize incomplete lines more strongly.',
  layer002QuadMaxGapInput: 'Target maximum run of edge samples without Layer 002 support. Lower values penalize unsupported stretches more strongly.',
  layer002QuadAngleSweepInput: 'Maximum inward distance searched from the outer hull toward the center.',
  layer002QuadAngleStepInput: 'Pixel increment used while sweeping inward. Smaller steps are more precise but slower.',
  layer002QuadInsetInput: 'Angular sweep around the outer hull orientation used to find the best four-edge fit.',
  layer002TraceAngleStepInput: 'Angle increment used during hull-sweep orientation search. Smaller steps are more precise but slower.',
  layer002TraceAspectToleranceInput: 'Allowed deviation from a square-like aspect ratio. Lower values enforce a squarer quad; higher values allow more rectangular fits.',
  layer002QuadLineThicknessInput: 'Visual line thickness for accepted and rejected Layer 002 hull-sweep quads.',
  layer002QuadOpacityInput: 'Visual opacity for Layer 002 hull-sweep overlays.',
  layer002QuadRejectedToggle: 'Shows rejected hull-sweep candidates as faint dashed quads for calibration.',
  maxFramesInput: 'Optional debug limit for rebuild operations. Leave this blank to process every frame in the selected run.'
};
const CORNER_FIELD_HELP = {
  show: 'Shows or hides this corner type in the overlay. This is a review visibility setting and does not rebuild the data by itself.',
  MinArea: 'Minimum internal void area required before searching for this void corner type.',
  Radius: 'Pixel radius around each candidate corner used to sample mask and void evidence.',
  MinAngle: 'Smallest accepted corner angle for this corner type.',
  MaxAngle: 'Largest accepted corner angle for this corner type.',
  Support: 'Minimum sampled pixels required to classify this corner direction as inside or outside.',
  Epsilon: 'Contour simplification amount before corner candidates are created. Higher values produce fewer, cleaner candidates; lower values preserve more detail.',
  Distance: 'Minimum spacing between kept corners of this type. Higher values remove nearby duplicates; lower values keep more neighboring candidates.',
  MinPoints: 'Minimum number of points of this type required in the local bbox or void group.',
  MaxPoints: 'Maximum number of points of this type kept after filtering and ordering.'
};
const CLASS_OPTION_HELP = {
  layerRow: 'Layer row controls: the checkbox enables or disables this layer for visualization and downstream builds. The color picker changes only the review overlay color; it does not redefine the fixed decoded color rules for the layer.',
  layerOpacityInput: 'Controls the visual opacity of this layer mask overlay. Turning the layer off effectively displays it at 0 percent opacity.',
  layerConfidenceInput: 'User-adjustable confidence for this pixel class. It defaults from the preset and is stored for confidence thresholding and future scoring logic.',
  layerGroupSelect: 'Assigns this layer to up to three review groups. A layer may belong to zero or more groups so later rules can evaluate layer combinations.'
};

const state = {
  config: null,
  rules: null,
  review: null,
  memory: null,
  precompute: null,
  dependencyStatus: null,
  colorTable: null,
  classMembership: null,
  ruleSets: new Map(),
  currentColorIds: null,
  currentImage: null,
  source: {
    mode: 'library',
    libraryRuns: [],
    libraryRunName: '',
    livePoint: '/src/color_masks/output/mask_manifest.json',
    readiness: null,
    readinessInFlight: false,
    sourceDiscovery: null,
    pipelineManifestUrl: '',
    manifest: null,
    manifestUrl: '',
    frameIndex: 0,
    followLatest: true,
    pollTimer: 0,
    pollInFlight: false,
    pollError: '',
    lastFrameCount: 0,
    imageCache: new Map(),
    imageRequestId: 0
  },
  pipeline: {
    status: null,
    playback: null,
    pollTimer: 0,
    pollInFlight: false,
    lastAppliedFrame: -1,
    lastRunId: ''
  },
  colorMasks: {
    manifest: null,
    cache: new Map(),
    requestId: 0,
    loadError: '',
    manifestUrl: ''
  },
  frameIndex: 0,
  activePrefix: '001',
  bbox: {
    discovery: null,
    manifest: null,
    renderRaf: 0
  },
  maskbitsBbox: {
    discovery: null,
    manifest: null
  },
  topBbox: {
    requestId: 0,
    renderRaf: 0,
    latency: null
  },
  bboxClipping: {
    discovery: null,
    manifest: null,
    requestId: 0,
    renderRaf: 0
  },
  bboxContours: {
    discovery: null,
    manifest: null,
    requestId: 0,
    renderRaf: 0,
    frameCache: new Map()
  },
  topPose: {
    requestId: 0,
    renderRaf: 0
  },
  topInstance: {
    requestId: 0,
    renderRaf: 0
  },
  contour: {
    discovery: null,
    manifest: null
  },
  corner: {
    discovery: null,
    manifest: null
  },
  squarePose: {
    discovery: null,
    manifest: null,
    renderRaf: 0,
    three: null,
    textureLoaded: false
  },
  instances: {
    discovery: null,
    manifest: null
  },
  instance3d: {
    renderRaf: 0,
    three: null,
    sceneObjects: [],
    lastLayoutKey: ''
  },
  viewport: {
    frameWidth: 640,
    frameHeight: 360,
    fitScale: 1,
    scale: 1,
    panX: 0,
    panY: 0,
    mainViewportWidth: 640,
    mainViewportHeight: 730,
    auxVisible: false,
    auxWidth: 0
  },
  panDrag: null,
  dirtyView: false,
  eventsInstalled: false
};

const els = {
  runLabel: document.getElementById('runLabel'),
  sourceLibraryButton: document.getElementById('sourceLibraryButton'),
  sourceLiveButton: document.getElementById('sourceLiveButton'),
  libraryRunSelect: document.getElementById('libraryRunSelect'),
  livePointInput: document.getElementById('livePointInput'),
  pipelineStartButton: document.getElementById('pipelineStartButton'),
  pipelineStopButton: document.getElementById('pipelineStopButton'),
  pipelineReadinessText: document.getElementById('pipelineReadinessText'),
  frameLabel: document.getElementById('frameLabel'),
  frameSlider: document.getElementById('frameSlider'),
  saveRulesButton: document.getElementById('saveRulesButton'),
  rebuildButton: document.getElementById('rebuildButton'),
  classList: document.getElementById('classList'),
  rulesStatus: document.getElementById('rulesStatus'),
  sourceOpacityInput: document.getElementById('sourceOpacityInput'),
  sourceOpacityValue: document.getElementById('sourceOpacityValue'),
  assetSummary: document.getElementById('assetSummary'),
  topSourceCanvas: document.getElementById('topSourceCanvas'),
  topSourceMeta: document.getElementById('topSourceMeta'),
  topSourceFile: document.getElementById('topSourceFile'),
  topSourceEmpty: document.getElementById('topSourceEmpty'),
  topMaskCanvas: document.getElementById('topMaskCanvas'),
  topMaskMeta: document.getElementById('topMaskMeta'),
  topMaskFile: document.getElementById('topMaskFile'),
  topMaskEmpty: document.getElementById('topMaskEmpty'),
  topBboxCanvas: document.getElementById('topBboxCanvas'),
  topBboxMeta: document.getElementById('topBboxMeta'),
  topBboxFile: document.getElementById('topBboxFile'),
  topBboxEmpty: document.getElementById('topBboxEmpty'),
  topClipCanvas: document.getElementById('topClipCanvas'),
  topClipMeta: document.getElementById('topClipMeta'),
  topClipFile: document.getElementById('topClipFile'),
  topClipEmpty: document.getElementById('topClipEmpty'),
  topContourCanvas: document.getElementById('topContourCanvas'),
  topContourMeta: document.getElementById('topContourMeta'),
  topContourFile: document.getElementById('topContourFile'),
  topContourEmpty: document.getElementById('topContourEmpty'),
  topPoseCanvas: document.getElementById('topPoseCanvas'),
  topPoseMeta: document.getElementById('topPoseMeta'),
  topPoseFile: document.getElementById('topPoseFile'),
  topPoseEmpty: document.getElementById('topPoseEmpty'),
  topInstanceCanvas: document.getElementById('topInstanceCanvas'),
  topInstanceMeta: document.getElementById('topInstanceMeta'),
  topInstanceFile: document.getElementById('topInstanceFile'),
  topInstanceEmpty: document.getElementById('topInstanceEmpty'),
  canvasWrap: document.getElementById('canvasWrap'),
  viewStack: document.getElementById('viewStack'),
  frameSurface: document.getElementById('frameSurface'),
  poseSurface: document.getElementById('poseSurface'),
  instance3dSurface: document.getElementById('instance3dSurface'),
  auxViewRail: document.getElementById('auxViewRail'),
  sourceCanvas: document.getElementById('sourceCanvas'),
  overlayCanvas: document.getElementById('overlayCanvas'),
  bboxOverlay: document.getElementById('bboxOverlay'),
  squarePoseCanvas: document.getElementById('squarePoseCanvas'),
  squarePoseLabelOverlay: document.getElementById('squarePoseLabelOverlay'),
  instance3dCanvas: document.getElementById('instance3dCanvas'),
  instance3dHud: document.getElementById('instance3dHud'),
  auxMaskCanvas: document.getElementById('auxMaskCanvas'),
  instanceReadout: document.getElementById('instanceReadout'),
  instanceTrackingJsonScaffold: document.getElementById('instanceTrackingJsonScaffold'),
  hoverReadout: document.getElementById('hoverReadout'),
  selectionReadout: document.getElementById('selectionReadout'),
  squarePoseStatus: document.getElementById('squarePoseStatus'),
  squarePoseToggle: document.getElementById('squarePoseToggle'),
  squarePoseViewModeSelect: document.getElementById('squarePoseViewModeSelect'),
  squarePoseSolutionSelect: document.getElementById('squarePoseSolutionSelect'),
  squarePoseSizeInput: document.getElementById('squarePoseSizeInput'),
  squarePoseSizeValue: document.getElementById('squarePoseSizeValue'),
  squarePoseEdgeToleranceInput: document.getElementById('squarePoseEdgeToleranceInput'),
  squarePoseEdgeToleranceValue: document.getElementById('squarePoseEdgeToleranceValue'),
  squarePoseMinCoverageInput: document.getElementById('squarePoseMinCoverageInput'),
  squarePoseMinCoverageValue: document.getElementById('squarePoseMinCoverageValue'),
  squarePoseMaxErrorInput: document.getElementById('squarePoseMaxErrorInput'),
  squarePoseMaxErrorValue: document.getElementById('squarePoseMaxErrorValue'),
  squarePoseClipGuardToggle: document.getElementById('squarePoseClipGuardToggle'),
  squarePoseClipThresholdInput: document.getElementById('squarePoseClipThresholdInput'),
  squarePoseClipThresholdValue: document.getElementById('squarePoseClipThresholdValue'),
  squarePoseCornerFitToggle: document.getElementById('squarePoseCornerFitToggle'),
  squarePoseMinCornerLinksInput: document.getElementById('squarePoseMinCornerLinksInput'),
  squarePoseMinCornerLinksValue: document.getElementById('squarePoseMinCornerLinksValue'),
  squarePoseCompleteBonusToggle: document.getElementById('squarePoseCompleteBonusToggle'),
  squarePoseExtraCornerLimitInput: document.getElementById('squarePoseExtraCornerLimitInput'),
  squarePoseExtraCornerLimitValue: document.getElementById('squarePoseExtraCornerLimitValue'),
  squarePoseMaxCandidatesInput: document.getElementById('squarePoseMaxCandidatesInput'),
  squarePoseMaxCandidatesValue: document.getElementById('squarePoseMaxCandidatesValue'),
  squarePoseTextureOpacityInput: document.getElementById('squarePoseTextureOpacityInput'),
  squarePoseTextureOpacityValue: document.getElementById('squarePoseTextureOpacityValue'),
  squarePoseCandidateOpacityInput: document.getElementById('squarePoseCandidateOpacityInput'),
  squarePoseCandidateOpacityValue: document.getElementById('squarePoseCandidateOpacityValue'),
  squarePoseFrustumToggle: document.getElementById('squarePoseFrustumToggle'),
  squarePoseOutlineToggle: document.getElementById('squarePoseOutlineToggle'),
  squarePoseScoresToggle: document.getElementById('squarePoseScoresToggle'),
  squarePoseLabelsToggle: document.getElementById('squarePoseLabelsToggle'),
  squarePoseBuildButton: document.getElementById('squarePoseBuildButton'),
  squarePoseStats: document.getElementById('squarePoseStats'),
  bboxStatus: document.getElementById('bboxStatus'),
  bboxMinPixelsInput: document.getElementById('bboxMinPixelsInput'),
  bboxMinPixelsValue: document.getElementById('bboxMinPixelsValue'),
  bboxMaxPerFrameInput: document.getElementById('bboxMaxPerFrameInput'),
  bboxMaxPerFrameNumber: document.getElementById('bboxMaxPerFrameNumber'),
  bboxMaxPerFrameValue: document.getElementById('bboxMaxPerFrameValue'),
  bboxTightnessInput: document.getElementById('bboxTightnessInput'),
  bboxTightnessValue: document.getElementById('bboxTightnessValue'),
  bboxQuadToggle: document.getElementById('bboxQuadToggle'),
  bboxFitModeSelect: document.getElementById('bboxFitModeSelect'),
  bboxTargetAspectInput: document.getElementById('bboxTargetAspectInput'),
  bboxTargetAspectValue: document.getElementById('bboxTargetAspectValue'),
  bboxAspectToleranceInput: document.getElementById('bboxAspectToleranceInput'),
  bboxAspectToleranceValue: document.getElementById('bboxAspectToleranceValue'),
  bboxQuadThicknessInput: document.getElementById('bboxQuadThicknessInput'),
  bboxQuadThicknessValue: document.getElementById('bboxQuadThicknessValue'),
  bboxEdgeCoverageInput: document.getElementById('bboxEdgeCoverageInput'),
  bboxEdgeCoverageValue: document.getElementById('bboxEdgeCoverageValue'),
  bboxCornerMinPixelsInput: document.getElementById('bboxCornerMinPixelsInput'),
  bboxCornerMinPixelsValue: document.getElementById('bboxCornerMinPixelsValue'),
  bboxVoidOverlapInput: document.getElementById('bboxVoidOverlapInput'),
  bboxVoidOverlapValue: document.getElementById('bboxVoidOverlapValue'),
  bboxVoidMinPixelsInput: document.getElementById('bboxVoidMinPixelsInput'),
  bboxVoidMinPixelsValue: document.getElementById('bboxVoidMinPixelsValue'),
  bboxShowRawToggle: document.getElementById('bboxShowRawToggle'),
  bboxQuadOpacityInput: document.getElementById('bboxQuadOpacityInput'),
  bboxQuadOpacityValue: document.getElementById('bboxQuadOpacityValue'),
  bboxBuildButton: document.getElementById('bboxBuildButton'),
  bboxViewModeSelect: document.getElementById('bboxViewModeSelect'),
  maskbitsBboxBuildButton: document.getElementById('maskbitsBboxBuildButton'),
  maskbitsBboxStats: document.getElementById('maskbitsBboxStats'),
  bboxStats: document.getElementById('bboxStats'),
  fovClipStatus: document.getElementById('fovClipStatus'),
  fovClipToggle: document.getElementById('fovClipToggle'),
  fovClipOverlayToggle: document.getElementById('fovClipOverlayToggle'),
  fovClipMarginInput: document.getElementById('fovClipMarginInput'),
  fovClipMarginValue: document.getElementById('fovClipMarginValue'),
  fovClipMinPixelsInput: document.getElementById('fovClipMinPixelsInput'),
  fovClipMinPixelsValue: document.getElementById('fovClipMinPixelsValue'),
  fovClipMinRatioInput: document.getElementById('fovClipMinRatioInput'),
  fovClipMinRatioValue: document.getElementById('fovClipMinRatioValue'),
  fovClipBboxTouchToggle: document.getElementById('fovClipBboxTouchToggle'),
  fovClipWarnOnlyToggle: document.getElementById('fovClipWarnOnlyToggle'),
  bboxClippingBuildButton: document.getElementById('bboxClippingBuildButton'),
  bboxClippingStats: document.getElementById('bboxClippingStats'),
  instanceStatus: document.getElementById('instanceStatus'),
  instanceToggle: document.getElementById('instanceToggle'),
  instanceOverlayToggle: document.getElementById('instanceOverlayToggle'),
  instanceLabelsToggle: document.getElementById('instanceLabelsToggle'),
  instanceLinksToggle: document.getElementById('instanceLinksToggle'),
  instanceCandidatesToggle: document.getElementById('instanceCandidatesToggle'),
  instanceReadoutToggle: document.getElementById('instanceReadoutToggle'),
  instancePoseSourceSelect: document.getElementById('instancePoseSourceSelect'),
  instanceMaxGapInput: document.getElementById('instanceMaxGapInput'),
  instanceMaxGapValue: document.getElementById('instanceMaxGapValue'),
  instanceMinScoreInput: document.getElementById('instanceMinScoreInput'),
  instanceMinScoreValue: document.getElementById('instanceMinScoreValue'),
  instanceMax2dInput: document.getElementById('instanceMax2dInput'),
  instanceMax2dValue: document.getElementById('instanceMax2dValue'),
  instanceMax3dInput: document.getElementById('instanceMax3dInput'),
  instanceMax3dValue: document.getElementById('instanceMax3dValue'),
  instanceMaxRpyInput: document.getElementById('instanceMaxRpyInput'),
  instanceMaxRpyValue: document.getElementById('instanceMaxRpyValue'),
  instanceSplitScoreInput: document.getElementById('instanceSplitScoreInput'),
  instanceSplitScoreValue: document.getElementById('instanceSplitScoreValue'),
  instanceTrailInput: document.getElementById('instanceTrailInput'),
  instanceTrailValue: document.getElementById('instanceTrailValue'),
  instanceCandidateLimitInput: document.getElementById('instanceCandidateLimitInput'),
  instanceCandidateLimitValue: document.getElementById('instanceCandidateLimitValue'),
  instanceBuildButton: document.getElementById('instanceBuildButton'),
  instanceStats: document.getElementById('instanceStats'),
  contourStatus: document.getElementById('contourStatus'),
  contourToggle: document.getElementById('contourToggle'),
  contourSourceSelect: document.getElementById('contourSourceSelect'),
  contourMinOuterInput: document.getElementById('contourMinOuterInput'),
  contourMinOuterValue: document.getElementById('contourMinOuterValue'),
  contourMinVoidInput: document.getElementById('contourMinVoidInput'),
  contourMinVoidValue: document.getElementById('contourMinVoidValue'),
  contourMaxVoidsInput: document.getElementById('contourMaxVoidsInput'),
  contourMaxVoidsValue: document.getElementById('contourMaxVoidsValue'),
  contourSimplifyInput: document.getElementById('contourSimplifyInput'),
  contourSimplifyValue: document.getElementById('contourSimplifyValue'),
  contourCloseInput: document.getElementById('contourCloseInput'),
  contourCloseValue: document.getElementById('contourCloseValue'),
  contourOpenInput: document.getElementById('contourOpenInput'),
  contourOpenValue: document.getElementById('contourOpenValue'),
  contourNotchInput: document.getElementById('contourNotchInput'),
  contourNotchValue: document.getElementById('contourNotchValue'),
  contourLineThicknessInput: document.getElementById('contourLineThicknessInput'),
  contourLineThicknessValue: document.getElementById('contourLineThicknessValue'),
  contourOpacityInput: document.getElementById('contourOpacityInput'),
  contourOpacityValue: document.getElementById('contourOpacityValue'),
  contourLabelsToggle: document.getElementById('contourLabelsToggle'),
  contourSmallToggle: document.getElementById('contourSmallToggle'),
  contourBuildButton: document.getElementById('contourBuildButton'),
  contourStats: document.getElementById('contourStats'),
  bboxContourBuildButton: document.getElementById('bboxContourBuildButton'),
  bboxContourStats: document.getElementById('bboxContourStats'),
  cornerStatus: document.getElementById('cornerStatus'),
  cornerHullRadiusInput: document.getElementById('cornerHullRadiusInput'),
  cornerHullRadiusValue: document.getElementById('cornerHullRadiusValue'),
  cornerHullMinAngleInput: document.getElementById('cornerHullMinAngleInput'),
  cornerHullMinAngleValue: document.getElementById('cornerHullMinAngleValue'),
  cornerHullMaxAngleInput: document.getElementById('cornerHullMaxAngleInput'),
  cornerHullMaxAngleValue: document.getElementById('cornerHullMaxAngleValue'),
  cornerHullMinSupportInput: document.getElementById('cornerHullMinSupportInput'),
  cornerHullMinSupportValue: document.getElementById('cornerHullMinSupportValue'),
  cornerHullEpsilonInput: document.getElementById('cornerHullEpsilonInput'),
  cornerHullEpsilonValue: document.getElementById('cornerHullEpsilonValue'),
  cornerHullDistanceInput: document.getElementById('cornerHullDistanceInput'),
  cornerHullDistanceValue: document.getElementById('cornerHullDistanceValue'),
  cornerHullMaxInput: document.getElementById('cornerHullMaxInput'),
  cornerHullMaxValue: document.getElementById('cornerHullMaxValue'),
  cornerHullWhiteMinInput: document.getElementById('cornerHullWhiteMinInput'),
  cornerHullWhiteMinNumber: document.getElementById('cornerHullWhiteMinNumber'),
  cornerHullWhiteMinValue: document.getElementById('cornerHullWhiteMinValue'),
  cornerHullWhiteMaxInput: document.getElementById('cornerHullWhiteMaxInput'),
  cornerHullWhiteMaxNumber: document.getElementById('cornerHullWhiteMaxNumber'),
  cornerHullWhiteMaxValue: document.getElementById('cornerHullWhiteMaxValue'),
  cornerHullBlackMinInput: document.getElementById('cornerHullBlackMinInput'),
  cornerHullBlackMinNumber: document.getElementById('cornerHullBlackMinNumber'),
  cornerHullBlackMinValue: document.getElementById('cornerHullBlackMinValue'),
  cornerHullBlackMaxInput: document.getElementById('cornerHullBlackMaxInput'),
  cornerHullBlackMaxNumber: document.getElementById('cornerHullBlackMaxNumber'),
  cornerHullBlackMaxValue: document.getElementById('cornerHullBlackMaxValue'),
  cornerVoidMinAreaInput: document.getElementById('cornerVoidMinAreaInput'),
  cornerVoidMinAreaValue: document.getElementById('cornerVoidMinAreaValue'),
  cornerVoidRadiusInput: document.getElementById('cornerVoidRadiusInput'),
  cornerVoidRadiusValue: document.getElementById('cornerVoidRadiusValue'),
  cornerVoidMinAngleInput: document.getElementById('cornerVoidMinAngleInput'),
  cornerVoidMinAngleValue: document.getElementById('cornerVoidMinAngleValue'),
  cornerVoidMaxAngleInput: document.getElementById('cornerVoidMaxAngleInput'),
  cornerVoidMaxAngleValue: document.getElementById('cornerVoidMaxAngleValue'),
  cornerVoidMinSupportInput: document.getElementById('cornerVoidMinSupportInput'),
  cornerVoidMinSupportValue: document.getElementById('cornerVoidMinSupportValue'),
  cornerVoidEpsilonInput: document.getElementById('cornerVoidEpsilonInput'),
  cornerVoidEpsilonValue: document.getElementById('cornerVoidEpsilonValue'),
  cornerVoidDistanceInput: document.getElementById('cornerVoidDistanceInput'),
  cornerVoidDistanceValue: document.getElementById('cornerVoidDistanceValue'),
  cornerVoidMaxInput: document.getElementById('cornerVoidMaxInput'),
  cornerVoidMaxValue: document.getElementById('cornerVoidMaxValue'),
  cornerVoidOrangeMinInput: document.getElementById('cornerVoidOrangeMinInput'),
  cornerVoidOrangeMinNumber: document.getElementById('cornerVoidOrangeMinNumber'),
  cornerVoidOrangeMinValue: document.getElementById('cornerVoidOrangeMinValue'),
  cornerVoidOrangeMaxInput: document.getElementById('cornerVoidOrangeMaxInput'),
  cornerVoidOrangeMaxNumber: document.getElementById('cornerVoidOrangeMaxNumber'),
  cornerVoidOrangeMaxValue: document.getElementById('cornerVoidOrangeMaxValue'),
  cornerVoidGreenMinInput: document.getElementById('cornerVoidGreenMinInput'),
  cornerVoidGreenMinNumber: document.getElementById('cornerVoidGreenMinNumber'),
  cornerVoidGreenMinValue: document.getElementById('cornerVoidGreenMinValue'),
  cornerVoidGreenMaxInput: document.getElementById('cornerVoidGreenMaxInput'),
  cornerVoidGreenMaxNumber: document.getElementById('cornerVoidGreenMaxNumber'),
  cornerVoidGreenMaxValue: document.getElementById('cornerVoidGreenMaxValue'),
  cornerShowHullToggle: document.getElementById('cornerShowHullToggle'),
  cornerShowVoidToggle: document.getElementById('cornerShowVoidToggle'),
  cornerShowAmbiguousToggle: document.getElementById('cornerShowAmbiguousToggle'),
  cornerOpacityInput: document.getElementById('cornerOpacityInput'),
  cornerOpacityValue: document.getElementById('cornerOpacityValue'),
  cornerAgreementToggle: document.getElementById('cornerAgreementToggle'),
  cornerAgreementMaxDistanceInput: document.getElementById('cornerAgreementMaxDistanceInput'),
  cornerAgreementMaxDistanceValue: document.getElementById('cornerAgreementMaxDistanceValue'),
  cornerAgreementWhiteToleranceInput: document.getElementById('cornerAgreementWhiteToleranceInput'),
  cornerAgreementWhiteToleranceValue: document.getElementById('cornerAgreementWhiteToleranceValue'),
  cornerAgreementAngleToleranceInput: document.getElementById('cornerAgreementAngleToleranceInput'),
  cornerAgreementAngleToleranceValue: document.getElementById('cornerAgreementAngleToleranceValue'),
  cornerAgreementMinScoreInput: document.getElementById('cornerAgreementMinScoreInput'),
  cornerAgreementMinScoreValue: document.getElementById('cornerAgreementMinScoreValue'),
  cornerAgreementMaxLinksInput: document.getElementById('cornerAgreementMaxLinksInput'),
  cornerAgreementMaxLinksValue: document.getElementById('cornerAgreementMaxLinksValue'),
  cornerAgreementLineThicknessInput: document.getElementById('cornerAgreementLineThicknessInput'),
  cornerAgreementLineThicknessValue: document.getElementById('cornerAgreementLineThicknessValue'),
  cornerAgreementOpacityInput: document.getElementById('cornerAgreementOpacityInput'),
  cornerAgreementOpacityValue: document.getElementById('cornerAgreementOpacityValue'),
  cornerAgreementStructureToggle: document.getElementById('cornerAgreementStructureToggle'),
  cornerAgreementOppositeToleranceInput: document.getElementById('cornerAgreementOppositeToleranceInput'),
  cornerAgreementOppositeToleranceValue: document.getElementById('cornerAgreementOppositeToleranceValue'),
  cornerAgreementStats: document.getElementById('cornerAgreementStats'),
  layer002QuadToggle: document.getElementById('layer002QuadToggle'),
  layer002TraceSupportSelect: document.getElementById('layer002TraceSupportSelect'),
  layer002TraceForbiddenSelect: document.getElementById('layer002TraceForbiddenSelect'),
  layer002QuadMaxInput: document.getElementById('layer002QuadMaxInput'),
  layer002QuadMaxValue: document.getElementById('layer002QuadMaxValue'),
  layer002QuadMinPixelsInput: document.getElementById('layer002QuadMinPixelsInput'),
  layer002QuadMinPixelsValue: document.getElementById('layer002QuadMinPixelsValue'),
  layer002QuadEdgeThicknessInput: document.getElementById('layer002QuadEdgeThicknessInput'),
  layer002QuadEdgeThicknessValue: document.getElementById('layer002QuadEdgeThicknessValue'),
  layer002QuadMinCoverageInput: document.getElementById('layer002QuadMinCoverageInput'),
  layer002QuadMinCoverageValue: document.getElementById('layer002QuadMinCoverageValue'),
  layer002QuadMaxGapInput: document.getElementById('layer002QuadMaxGapInput'),
  layer002QuadMaxGapValue: document.getElementById('layer002QuadMaxGapValue'),
  layer002QuadAngleSweepInput: document.getElementById('layer002QuadAngleSweepInput'),
  layer002QuadAngleSweepValue: document.getElementById('layer002QuadAngleSweepValue'),
  layer002QuadAngleStepInput: document.getElementById('layer002QuadAngleStepInput'),
  layer002QuadAngleStepValue: document.getElementById('layer002QuadAngleStepValue'),
  layer002QuadInsetInput: document.getElementById('layer002QuadInsetInput'),
  layer002QuadInsetValue: document.getElementById('layer002QuadInsetValue'),
  layer002TraceAngleStepInput: document.getElementById('layer002TraceAngleStepInput'),
  layer002TraceAngleStepValue: document.getElementById('layer002TraceAngleStepValue'),
  layer002TraceAspectToleranceInput: document.getElementById('layer002TraceAspectToleranceInput'),
  layer002TraceAspectToleranceValue: document.getElementById('layer002TraceAspectToleranceValue'),
  layer002QuadLineThicknessInput: document.getElementById('layer002QuadLineThicknessInput'),
  layer002QuadLineThicknessValue: document.getElementById('layer002QuadLineThicknessValue'),
  layer002QuadOpacityInput: document.getElementById('layer002QuadOpacityInput'),
  layer002QuadOpacityValue: document.getElementById('layer002QuadOpacityValue'),
  layer002QuadRejectedToggle: document.getElementById('layer002QuadRejectedToggle'),
  layer002QuadStats: document.getElementById('layer002QuadStats'),
  cornerBuildButton: document.getElementById('cornerBuildButton'),
  cornerStats: document.getElementById('cornerStats'),
  memoryStatus: document.getElementById('memoryStatus'),
  memorySummary: document.getElementById('memorySummary'),
  layerStatus: document.getElementById('layerStatus'),
  frameLayerList: document.getElementById('frameLayerList'),
  inspector: document.getElementById('inspector'),
  statusText: document.getElementById('statusText'),
  maxFramesInput: document.getElementById('maxFramesInput')
};

const ctx = {
  topSource: els.topSourceCanvas?.getContext('2d') || null,
  topMask: els.topMaskCanvas?.getContext('2d') || null,
  topBbox: els.topBboxCanvas?.getContext('2d') || null,
  topClip: els.topClipCanvas?.getContext('2d') || null,
  topContour: els.topContourCanvas?.getContext('2d') || null,
  topPose: els.topPoseCanvas?.getContext('2d') || null,
  topInstance: els.topInstanceCanvas?.getContext('2d') || null,
  source: els.sourceCanvas.getContext('2d'),
  overlay: els.overlayCanvas.getContext('2d'),
  auxMask: els.auxMaskCanvas.getContext('2d')
};

function setStatus(text) {
  if (els.statusText) els.statusText.textContent = text;
}

let activeOptionInfoButton = null;
let optionInfoEventsInstalled = false;

function optionLabelText(labelNode) {
  const clone = labelNode.cloneNode(true);
  clone.querySelectorAll('.optionInfoButton').forEach((button) => button.remove());
  return clone.textContent.trim().replace(/\s+/g, ' ');
}

function cornerHelpForControl(control) {
  const id = control?.id || '';
  const showMatch = id.match(/^corner(White|Black|Orange|Green)ShowToggle$/);
  if (showMatch) return CORNER_FIELD_HELP.show;
  const fieldMatch = id.match(/^corner(White|Black|Orange|Green)(MinArea|Radius|MinAngle|MaxAngle|Support|Epsilon|Distance|MinPoints|MaxPoints)(Input|Number)$/);
  if (!fieldMatch) return '';
  return CORNER_FIELD_HELP[fieldMatch[2]] || '';
}

function optionHelpForControl(control) {
  if (!control) return '';
  if (control.dataset?.optionHelpKey) return CLASS_OPTION_HELP[control.dataset.optionHelpKey] || '';
  if (control.id && OPTION_HELP[control.id]) return OPTION_HELP[control.id];
  const cornerHelp = cornerHelpForControl(control);
  if (cornerHelp) return cornerHelp;
  if (control.classList?.contains('layerOpacityInput')) return CLASS_OPTION_HELP.layerOpacityInput;
  if (control.classList?.contains('layerConfidenceInput')) return CLASS_OPTION_HELP.layerConfidenceInput;
  if (control.classList?.contains('layerGroupSelect')) return CLASS_OPTION_HELP.layerGroupSelect;
  return '';
}

function primaryOptionControl(row) {
  return row.querySelector('select, input:not([type="hidden"]), textarea');
}

function ensureOptionInfoPopover() {
  let popover = document.querySelector('.optionInfoPopover');
  if (popover) return popover;
  popover = document.createElement('div');
  popover.className = 'optionInfoPopover';
  popover.hidden = true;
  popover.innerHTML = '<strong class="optionInfoTitle"></strong><div class="optionInfoText"></div>';
  document.body.appendChild(popover);
  return popover;
}

function closeOptionInfo() {
  const popover = document.querySelector('.optionInfoPopover');
  if (popover) popover.hidden = true;
  activeOptionInfoButton?.classList.remove('active');
  activeOptionInfoButton = null;
}

function showOptionInfo(button, title, text) {
  const popover = ensureOptionInfoPopover();
  if (activeOptionInfoButton === button && !popover.hidden) {
    closeOptionInfo();
    return;
  }
  activeOptionInfoButton?.classList.remove('active');
  activeOptionInfoButton = button;
  button.classList.add('active');
  popover.querySelector('.optionInfoTitle').textContent = title || 'Option';
  popover.querySelector('.optionInfoText').textContent = text || 'No description available yet.';
  popover.hidden = false;
  popover.style.left = '12px';
  popover.style.top = '12px';
  const rect = button.getBoundingClientRect();
  const popoverRect = popover.getBoundingClientRect();
  const maxLeft = Math.max(12, window.innerWidth - popoverRect.width - 12);
  const maxTop = Math.max(12, window.innerHeight - popoverRect.height - 12);
  const left = Math.min(maxLeft, Math.max(12, rect.left));
  const top = Math.min(maxTop, Math.max(12, rect.bottom + 6));
  popover.style.left = `${Math.round(left)}px`;
  popover.style.top = `${Math.round(top)}px`;
}

function bindOptionInfoButton(button, title, text) {
  if (!button || button.dataset.infoBound === 'true') return;
  button.dataset.infoBound = 'true';
  button.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    showOptionInfo(button, title, text);
  });
}

function installOptionInfoGlobalEvents() {
  if (optionInfoEventsInstalled) return;
  optionInfoEventsInstalled = true;
  document.addEventListener('click', (event) => {
    if (event.target.closest?.('.optionInfoPopover, .optionInfoButton')) return;
    closeOptionInfo();
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') closeOptionInfo();
  });
  window.addEventListener('resize', closeOptionInfo);
}

function attachOptionInfoButtons(root = document) {
  installOptionInfoGlobalEvents();
  root.querySelectorAll('.flowControl, .flowSlider, .toggleRow, .opacityControl, .layerOpacityControl, .layerConfidenceControl, .layerGroupControl, .footerControl').forEach((row) => {
    const label = row.querySelector(':scope > span:first-child');
    const control = primaryOptionControl(row);
    const help = optionHelpForControl(control);
    if (!label || !control || !help) return;
    let button = label.querySelector('.optionInfoButton');
    if (!button) {
      button = document.createElement('button');
      button.type = 'button';
      button.className = 'optionInfoButton';
      button.textContent = 'i';
      label.appendChild(button);
    }
    const title = optionLabelText(label);
    button.setAttribute('aria-label', `About ${title}`);
    bindOptionInfoButton(button, title, help);
  });
  root.querySelectorAll('.optionInfoButton[data-option-help-key]').forEach((button) => {
    const help = CLASS_OPTION_HELP[button.dataset.optionHelpKey] || '';
    const title = button.getAttribute('aria-label')?.replace(/^About\s+/i, '') || 'Option';
    if (help) bindOptionInfoButton(button, title, help);
  });
}

function fmt(value, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : '-';
}

function pct(value) {
  return `${fmt(Number(value) * 100, 1)}%`;
}

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(1, number));
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[char]));
}

function hexToRgb(hex) {
  const clean = String(hex || '#ffffff').replace('#', '');
  if (!/^[0-9a-fA-F]{6}$/.test(clean)) return [255, 255, 255];
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16)
  ];
}

function normalizeHex(value, fallback = '#ffffff') {
  const text = String(value || '').trim();
  if (/^#[0-9a-fA-F]{6}$/.test(text)) return text.toLowerCase();
  if (/^[0-9a-fA-F]{6}$/.test(text)) return `#${text.toLowerCase()}`;
  return fallback;
}

function assetUrl(path) {
  const value = String(path || '').trim();
  if (!value) return '';
  if (/^(https?:|data:|blob:)/i.test(value)) return value;
  if (value.startsWith('/')) return value;
  return `/${value.replace(/^\.?\//, '')}`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    cache: options.method ? 'no-store' : 'no-store',
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error || `HTTP ${response.status}`);
  }
  return payload;
}

function jsonCacheKey(value) {
  return JSON.stringify(value);
}

async function fetchOptionalJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (response.status === 404) return null;
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new Error(payload?.error || `HTTP ${response.status}`);
  if (!payload || typeof payload !== 'object') throw new Error(`invalid JSON from ${url}`);
  return payload;
}

function appScopedAssetUrl(path) {
  const value = String(path || '').trim();
  if (!value) return '';
  if (/^(https?:|data:|blob:)/i.test(value)) return value;
  if (value.startsWith('/0721Vision/')) return value;
  if (value.startsWith('/src/') || value.startsWith('/assets/') || value.startsWith('/vendor/')) return `/0721Vision${value}`;
  if (value.startsWith('/')) return value;
  return `/0721Vision/src/color_masks/output/${value.replace(/^\.?\//, '')}`;
}

function cleanFileRef(path, maxLength = 116) {
  let value = String(path || '').trim();
  if (!value) return '-';
  const origin = window.location?.origin || '';
  if (origin && value.startsWith(origin)) value = value.slice(origin.length);
  value = value.replace(/^.*\/0721Vision\//, '/');
  if (value.startsWith('/0721Vision/')) value = value.slice('/0721Vision'.length) || '/';
  if (value.length <= maxLength) return value;
  const parts = value.split('/').filter(Boolean);
  if (parts.length >= 3) {
    const compact = `/${parts[0]}/.../${parts.slice(-2).join('/')}`;
    if (compact.length <= maxLength) return compact;
  }
  return `...${value.slice(Math.max(0, value.length - maxLength + 3))}`;
}

function pathMatchKey(path) {
  return cleanFileRef(path, 10000).replace(/\\/g, '/').replace(/\/+/g, '/');
}

function manifestUrlFromPoint(point) {
  let value = String(point || '').trim();
  if (!value) return COLOR_MASK_MANIFEST_URL;
  if (/^(https?:|data:|blob:)/i.test(value)) return value;
  value = value.replace(/^.*\/0721Vision\//, '/');
  if (!/\.json(?:[?#].*)?$/i.test(value)) {
    value = value.replace(/\/?$/, '/mask_manifest.json');
  }
  return appScopedAssetUrl(value);
}

function sourceFrames() {
  const frames = Array.isArray(state.source.manifest?.frames) ? state.source.manifest.frames : [];
  if (frames.length) return frames;
  if (state.source.mode !== 'library') return [];
  const run = activeLibraryRun();
  if (run && run.active === false) return [];
  return Array.isArray(state.precompute?.frames) ? state.precompute.frames : [];
}

function sourceFrameCount() {
  return sourceFrames().length;
}

function selectedSourceFrameIndex() {
  const count = sourceFrameCount();
  if (!count) return 0;
  return Math.max(0, Math.min(count - 1, Math.round(Number(state.source.frameIndex) || 0)));
}

function currentReviewFrameIndex() {
  return selectedSourceFrameIndex();
}

function sourceFrameEntry(frameIndex = selectedSourceFrameIndex()) {
  return sourceFrames()[frameIndex] || null;
}

function framePathFromEntry(entry) {
  return entry?.source || entry?.path || entry?.relativePath || entry?.filename || '';
}

function manifestHasMaskBits(manifest) {
  const frames = Array.isArray(manifest?.frames) ? manifest.frames : [];
  return frames.some((frame) => frame && typeof frame === 'object' && (frame.maskBits || String(frame.path || '').includes('.maskbits.')));
}

function activeLibraryRun() {
  const runs = Array.isArray(state.source.libraryRuns) ? state.source.libraryRuns : [];
  return runs.find((run) => run.name === state.source.libraryRunName || run.sourceDir === state.source.libraryRunName || run.id === state.source.libraryRunName)
    || runs.find((run) => run.active)
    || runs[0]
    || null;
}

function librarySourceManifestUrl() {
  const run = activeLibraryRun();
  if (run) return run.decodedManifestPath || run.precompute?.manifestPath || '';
  return state.precompute?.sourceDecodedManifest || state.precompute?.manifestPath || '';
}

function librarySourceDir() {
  const run = activeLibraryRun();
  return String(run?.sourceDir || run?.id || '').trim();
}

function pipelineIsActive(status = state.pipeline.status) {
  const pipelineState = status?.state || 'idle';
  return ['created', 'starting', 'running', 'stopping'].includes(pipelineState);
}

function playbackIsActive(playback = state.pipeline.playback) {
  const playbackState = playback?.state || 'idle';
  return ['queued', 'running', 'draining', 'stopping'].includes(playbackState);
}

function selectedSourceManifestUrl() {
  const playbackState = state.pipeline.playback?.state || 'idle';
  const playbackSelected = ['queued', 'running', 'draining', 'complete', 'stopped', 'error'].includes(playbackState);
  if (state.source.pipelineManifestUrl && (state.source.mode === 'live' || playbackSelected)) return state.source.pipelineManifestUrl;
  if (state.source.mode === 'live') return state.source.pipelineManifestUrl || manifestUrlFromPoint(state.source.livePoint);
  return appScopedAssetUrl(librarySourceManifestUrl());
}

function pipelineStageInfo(status, manifestUrl, completeFrameCount = null) {
  const url = manifestUrl ? appScopedAssetUrl(manifestUrl) : '';
  const frameCount = Number(completeFrameCount ?? status?.frameCountCompleted ?? 0);
  return {
    available: Boolean(url),
    stale: false,
    staleReasons: [],
    runKey: status?.runKey || status?.runId || '',
    root: '',
    manifestUrl: url,
    expectedSourceSignature: null,
    manifest: {
      createdAt: status?.updatedAt || null,
      kind: 'pipeline-live-output',
      frameStart: 0,
      frameCount,
      sourceFrameCount: Number(status?.frameCountAccepted || frameCount || 0),
      complete: status?.state === 'complete',
      summary: {}
    },
    dependency: { ok: true }
  };
}

function playbackPayloadForSelection() {
  const sourceDir = state.source.mode === 'library' ? librarySourceDir() : String(state.source.livePoint || '').trim();
  return {
    sourceDir,
    targetHz: 30,
    followLatest: true
  };
}

function applyPlaybackStatus(playback) {
  if (!playback || typeof playback !== 'object') return false;
  const previous = state.pipeline.playback || {};
  state.pipeline.playback = playback;
  renderSourceControls();
  return previous.state !== playback.state
    || previous.fedCount !== playback.fedCount
    || previous.completedCount !== playback.completedCount
    || previous.deadlineMissed !== playback.deadlineMissed;
}

function applyPipelineStatus(status) {
  if (!status || typeof status !== 'object') return false;
  state.pipeline.status = status;
  const urls = status.manifestUrls || {};
  const active = status.state && status.state !== 'idle';
  if (!active) {
    state.source.pipelineManifestUrl = '';
    renderSourceControls();
    return false;
  }
  const runId = status.runId || status.runKey || '';
  const latestFrame = Number(status.latestCompletedFrame ?? -1);
  const changed = runId !== state.pipeline.lastRunId || latestFrame !== state.pipeline.lastAppliedFrame;
  state.pipeline.lastRunId = runId;
  state.pipeline.lastAppliedFrame = latestFrame;
  state.source.pipelineManifestUrl = appScopedAssetUrl(urls.maskManifest || urls.sourceManifest || '');
  state.maskbitsBbox.discovery = { ok: true, maskbitsBbox: pipelineStageInfo(status, urls.bboxManifest) };
  state.bboxClipping.discovery = { ok: true, bboxClipping: pipelineStageInfo(status, urls.clippingManifest) };
  state.bboxContours.discovery = { ok: true, bboxContours: pipelineStageInfo(status, urls.contourManifest) };
  state.squarePose.discovery = { ok: true, poseEstimation: pipelineStageInfo(status, urls.poseManifest) };
  state.instances.discovery = { ok: true, instanceTracking: pipelineStageInfo(status, urls.instanceManifest) };
  renderSourceControls();
  renderBboxControls();
  renderBboxClippingControls();
  renderBboxContourControls();
  renderSquarePoseControls();
  renderInstanceControls();
  return changed;
}

async function refreshPipelineSources() {
  const payload = await fetchJson(`${PIPELINE_API}/sources`);
  state.source.sourceDiscovery = payload;
  const sources = Array.isArray(payload.sources) ? payload.sources.filter((source) => source && typeof source === 'object') : [];
  if (sources.length) {
    state.source.libraryRuns = sources;
    const current = sources.find((source) => source.name === state.source.libraryRunName || source.sourceDir === state.source.libraryRunName);
    const active = current || sources.find((source) => source.active) || sources[0] || null;
    state.source.libraryRunName = active?.name || '';
  }
  renderSourceControls();
  await refreshPlaybackReadiness().catch((error) => setStatus(`readiness failed: ${error.message}`));
  return payload;
}

async function refreshPlaybackReadiness() {
  if (state.source.readinessInFlight || playbackIsActive()) return state.source.readiness;
  const payload = playbackPayloadForSelection();
  if (!payload.sourceDir) {
    state.source.readiness = { ok: false, blockers: ['select a JPEG source folder'], frameCount: 0, targetHz: 30 };
    renderSourceControls();
    return state.source.readiness;
  }
  state.source.readinessInFlight = true;
  renderSourceControls();
  try {
    const readiness = await fetchJson(`${PIPELINE_API}/readiness`, {
      method: 'POST',
      body: JSON.stringify(payload)
    });
    state.source.readiness = readiness;
    return readiness;
  } finally {
    state.source.readinessInFlight = false;
    renderSourceControls();
  }
}

async function refreshPipelineStatus() {
  if (state.pipeline.pollInFlight) return;
  state.pipeline.pollInFlight = true;
  try {
    const payload = await fetchJson(`${PIPELINE_API}/status`);
    const status = payload.pipeline || null;
    const playbackChanged = applyPlaybackStatus(payload.playback || null);
    const changed = applyPipelineStatus(status);
    if ((changed || playbackChanged) && status?.state && status.state !== 'idle') {
      const urls = status.manifestUrls || {};
      if ((urls.sourceManifest || urls.maskManifest) && (state.source.mode === 'live' || state.source.pipelineManifestUrl)) {
        await refreshSourceManifests({ preserveIndex: !state.source.followLatest }).catch((error) => setStatus(`pipeline source failed: ${error.message}`));
      }
      if (urls.bboxManifest) await loadMaskbitsBboxManifest().catch((error) => setStatus(`pipeline bbox load failed: ${error.message}`));
      if (urls.clippingManifest) await loadBboxClippingManifest().catch((error) => setStatus(`pipeline clipping load failed: ${error.message}`));
      if (urls.contourManifest) await loadBboxContoursManifest().catch((error) => setStatus(`pipeline contour load failed: ${error.message}`));
      if (urls.poseManifest) await loadSquarePoseManifest().catch((error) => setStatus(`pipeline pose load failed: ${error.message}`));
      if (urls.instanceManifest) await loadInstanceManifest().catch((error) => setStatus(`pipeline instance load failed: ${error.message}`));
      const completed = Number(status.frameCountCompleted || 0);
      const accepted = Number(status.frameCountAccepted || completed || 0);
      const playback = state.pipeline.playback || {};
      const deadline = playback.deadlineMissed ? ` | deadline misses ${Number(playback.missedDeadlineCount || 0)}` : '';
      setStatus(`pipeline ${status.state}: ${completed}/${accepted} frames${deadline}`);
    }
  } catch (error) {
    setStatus(`pipeline status failed: ${error.message}`);
  } finally {
    state.pipeline.pollInFlight = false;
  }
}

function pipelineSourceInput() {
  if (state.source.mode === 'library') {
    return librarySourceDir();
  }
  const value = String(els.livePointInput?.value || state.source.livePoint || '').trim();
  return value || 'src/color_masks/input_frames';
}

async function startPlaybackRun() {
  const sourceValue = librarySourceDir();
  if (!sourceValue) throw new Error('select a JPEG source folder');
  state.source.followLatest = true;
  state.source.pipelineManifestUrl = '';
  renderSourceControls();
  const readiness = await refreshPlaybackReadiness();
  if (!readiness?.ok) {
    const blocker = readiness?.blockers?.[0] || 'source is not ready';
    setStatus(`not ready: ${blocker}`);
    return;
  }
  const payload = await fetchJson(`${PIPELINE_API}/playback/start`, {
    method: 'POST',
    body: JSON.stringify({ ...playbackPayloadForSelection(), sourceDir: sourceValue, noDebug: true })
  });
  applyPlaybackStatus(payload.playback);
  setStatus(`playback ${payload.playback?.state || 'queued'}: ${cleanFileRef(sourceValue, 80)}`);
  await refreshPipelineStatus();
}

async function startPipelineRun() {
  if (state.source.mode === 'library') {
    await startPlaybackRun();
    return;
  }
  const sourceValue = pipelineSourceInput();
  state.source.livePoint = sourceValue;
  state.source.mode = 'live';
  state.source.followLatest = true;
  state.source.pipelineManifestUrl = '';
  renderSourceControls();
  const mode = 'watch';
  const payload = await fetchJson(`${PIPELINE_API}/start`, {
    method: 'POST',
    body: JSON.stringify({ mode, sourceDir: sourceValue, followLatest: true, noDebug: true })
  });
  applyPipelineStatus(payload.pipeline);
  setStatus(`pipeline ${payload.pipeline?.state || 'queued'}: ${cleanFileRef(sourceValue, 80)}`);
  await refreshPipelineStatus();
}

async function stopPipelineRun() {
  const endpoint = playbackIsActive() ? 'playback/stop' : 'stop';
  const payload = await fetchJson(`${PIPELINE_API}/${endpoint}`, {
    method: 'POST',
    body: JSON.stringify({})
  });
  applyPlaybackStatus(payload.playback || { state: 'idle' });
  applyPipelineStatus(payload.pipeline);
  setStatus(`pipeline ${payload.pipeline?.state || 'stopping'}`);
  await refreshPlaybackReadiness().catch((error) => setStatus(`readiness failed: ${error.message}`));
}

function startPipelinePolling() {
  if (state.pipeline.pollTimer) window.clearInterval(state.pipeline.pollTimer);
  state.pipeline.pollTimer = window.setInterval(() => {
    refreshPipelineStatus().catch((error) => setStatus(`pipeline refresh failed: ${error.message}`));
  }, SOURCE_POLL_MS);
}

function sourceManifestReference(manifest) {
  if (!manifest || typeof manifest !== 'object') return '';
  const direct = manifest.sourceDecodedManifest || manifest.sourceManifest || manifest.manifestPath || '';
  if (direct) return appScopedAssetUrl(direct);
  const layers = Array.isArray(manifest.lut?.metadata?.layers) ? manifest.lut.metadata.layers : [];
  for (const layer of layers) {
    const source = layer?.source?.sourceDecodedManifest || layer?.source?.manifestPath || '';
    if (source) return appScopedAssetUrl(source);
  }
  return '';
}

function manifestSourceValues(manifest, keys) {
  const values = new Set();
  const visit = (node) => {
    if (!node || typeof node !== 'object') return;
    for (const key of keys) {
      if (node[key] !== undefined && node[key] !== null && String(node[key]).trim()) values.add(String(node[key]));
    }
  };
  visit(manifest);
  visit(manifest?.source);
  visit(manifest?.input);
  visit(manifest?.lut?.metadata);
  const layers = Array.isArray(manifest?.lut?.metadata?.layers) ? manifest.lut.metadata.layers : [];
  for (const layer of layers) {
    visit(layer);
    visit(layer?.source);
  }
  return values;
}

function maskManifestCanIndexSource() {
  if (!state.colorMasks.manifest) return false;
  if (state.colorMasks.manifest === state.source.manifest) return true;
  const maskUrl = pathMatchKey(state.colorMasks.manifestUrl);
  const sourceUrl = pathMatchKey(state.source.manifestUrl || selectedSourceManifestUrl());
  if (maskUrl && sourceUrl && maskUrl === sourceUrl) return true;
  const maskSource = pathMatchKey(sourceManifestReference(state.colorMasks.manifest));
  if (maskSource && sourceUrl && maskSource === sourceUrl) return true;
  const sourceHash = String(state.source.manifest?.sourceHash || '').trim();
  if (sourceHash && manifestSourceValues(state.colorMasks.manifest, ['sourceHash']).has(sourceHash)) return true;
  const sourceRun = String(state.source.manifest?.runName || state.source.manifest?.run || '').trim();
  return Boolean(sourceRun && manifestSourceValues(state.colorMasks.manifest, ['runName', 'run', 'activeRunName']).has(sourceRun));
}

function hasRequestedFrameParam() {
  const params = new URLSearchParams(window.location.search || '');
  return params.has('frame') || params.has('frameOrdinal');
}

function applySourceDiscovery(readOnlyAssets) {
  const assets = readOnlyAssets || {};
  const runs = Array.isArray(assets.runs) ? assets.runs.filter((run) => run && typeof run === 'object') : [];
  state.source.libraryRuns = runs;
  const current = runs.find((run) => run.name === state.source.libraryRunName);
  const active = current || runs.find((run) => run.active) || runs[0] || null;
  state.source.libraryRunName = active?.name || '';
  renderSourceControls();
}

function renderLibraryRunOptions() {
  if (!els.libraryRunSelect) return;
  const runs = Array.isArray(state.source.libraryRuns) ? state.source.libraryRuns : [];
  if (!runs.length) {
    els.libraryRunSelect.innerHTML = '<option value="">no JPEG source folders</option>';
    els.libraryRunSelect.value = '';
    return;
  }
  els.libraryRunSelect.innerHTML = runs.map((run) => {
    const frameCount = Number(run.frameCount || run.precompute?.frameCount || 0);
    const label = run.label || run.name || run.sourceDir || 'source';
    const suffix = label.includes('frames') ? '' : `${run.active ? ' | active' : ''}${frameCount ? ` | ${frameCount} frames` : ''}`;
    const value = run.name || run.sourceDir || run.id || '';
    return `<option value="${escapeHtml(value)}">${escapeHtml(`${label}${suffix}`)}</option>`;
  }).join('');
  els.libraryRunSelect.value = state.source.libraryRunName || activeLibraryRun()?.name || '';
}

function renderPipelineReadiness() {
  const element = els.pipelineReadinessText;
  if (!element) return;
  const controls = element.closest('.pipelineControls');
  controls?.classList.remove('ready', 'warning', 'error');
  const playback = state.pipeline.playback || {};
  if (playbackIsActive(playback) || playback.state === 'complete' || playback.state === 'stopped' || playback.state === 'error') {
    const fed = Number(playback.fedCount || 0);
    const completed = Number(playback.completedCount || 0);
    const total = Number(playback.sourceFrameCount || 0);
    if (playback.deadlineMissed) {
      controls?.classList.add('warning');
      element.textContent = `deadline miss ${Number(playback.missedDeadlineCount || 0)} | ${completed}/${fed || total}`;
    } else if (playback.state === 'complete') {
      controls?.classList.add('ready');
      element.textContent = `complete | ${completed}/${total}`;
    } else if (playback.state === 'error') {
      controls?.classList.add('error');
      element.textContent = `playback error`;
    } else {
      controls?.classList.add('ready');
      element.textContent = `30Hz ${playback.state || 'running'} | ${completed}/${fed || total}`;
    }
    return;
  }
  if (state.source.readinessInFlight) {
    element.textContent = 'checking readiness';
    return;
  }
  const readiness = state.source.readiness;
  if (!readiness) {
    element.textContent = 'readiness pending';
    return;
  }
  if (readiness.ok) {
    controls?.classList.add('ready');
    element.textContent = `ready 30Hz | ${Number(readiness.frameCount || 0)} frames`;
    return;
  }
  controls?.classList.add('error');
  element.textContent = `not ready | ${readiness.blockers?.[0] || 'source'}`;
}

function renderSourceControls() {
  const isLive = state.source.mode === 'live';
  if (els.sourceLibraryButton) {
    els.sourceLibraryButton.classList.toggle('active', !isLive);
    els.sourceLibraryButton.setAttribute('aria-pressed', String(!isLive));
  }
  if (els.sourceLiveButton) {
    els.sourceLiveButton.classList.toggle('active', isLive);
    els.sourceLiveButton.setAttribute('aria-pressed', String(isLive));
  }
  renderLibraryRunOptions();
  if (els.libraryRunSelect) els.libraryRunSelect.hidden = isLive;
  if (els.livePointInput) {
    els.livePointInput.hidden = !isLive;
    if (document.activeElement !== els.livePointInput) els.livePointInput.value = state.source.livePoint || '';
  }
  updateFrameNavigator();
  if (els.runLabel) {
    const count = sourceFrameCount();
    const sourceName = isLive ? cleanFileRef(state.source.livePoint, 54) : cleanFileRef(activeLibraryRun()?.sourceDir || activeLibraryRun()?.name || 'library', 54);
    const follow = state.source.followLatest ? 'following' : 'history';
    const pipeline = state.pipeline.status || {};
    const pipelineState = pipeline.state && pipeline.state !== 'idle' ? ` | pipeline ${pipeline.state}` : '';
    els.runLabel.textContent = `${isLive ? 'live' : 'library'} | ${sourceName} | ${count} frames | ${follow}${pipelineState}`;
  }
  const running = pipelineIsActive() || playbackIsActive();
  const readinessOk = state.source.mode === 'live' || state.source.readiness?.ok === true;
  if (els.pipelineStartButton) {
    els.pipelineStartButton.disabled = running || state.source.readinessInFlight || !readinessOk;
    els.pipelineStartButton.textContent = running ? 'Running' : 'Run';
  }
  if (els.pipelineStopButton) {
    const pipelineState = state.pipeline.status?.state || 'idle';
    els.pipelineStopButton.disabled = !running || pipelineState === 'stopping';
  }
  renderPipelineReadiness();
}

function canSyncReviewFrame(frameIndex = selectedSourceFrameIndex()) {
  if (state.source.mode !== 'library') return false;
  const run = activeLibraryRun();
  if (run && run.active === false) return false;
  const frames = Array.isArray(state.precompute?.frames) ? state.precompute.frames : [];
  return Boolean(frames[Math.max(0, Math.round(Number(frameIndex) || 0))]);
}

function visualizationState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.visualization = state.review.visualization && typeof state.review.visualization === 'object'
    ? state.review.visualization
    : {};
  const view = state.review.visualization;
  view.layerOpacityByPrefix = view.layerOpacityByPrefix && typeof view.layerOpacityByPrefix === 'object' ? view.layerOpacityByPrefix : {};
  view.layerEnabledByPrefix = view.layerEnabledByPrefix && typeof view.layerEnabledByPrefix === 'object' ? view.layerEnabledByPrefix : {};
  view.layerColorByPrefix = view.layerColorByPrefix && typeof view.layerColorByPrefix === 'object' ? view.layerColorByPrefix : {};
  return view;
}

function squarePoseState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.squarePose = state.review.squarePose && typeof state.review.squarePose === 'object'
    ? state.review.squarePose
    : {};
  const pose = state.review.squarePose;
  pose.show = pose.show !== false;
  pose.viewMode = ['camera', 'orbit'].includes(pose.viewMode) ? pose.viewMode : 'camera';
  pose.solutionMode = ['best', 'top', 'right', 'bottom', 'left'].includes(pose.solutionMode) ? pose.solutionMode : 'best';
  pose.squareSizeM = Number.isFinite(Number(pose.squareSizeM)) ? Math.max(0.1, Math.min(20, Number(pose.squareSizeM))) : 2.7;
  pose.edgeTolerancePx = Number.isFinite(Number(pose.edgeTolerancePx)) ? Math.max(0.25, Math.min(64, Number(pose.edgeTolerancePx))) : 4;
  pose.minEdgeCoverage = Number.isFinite(Number(pose.minEdgeCoverage)) ? clamp01(Number(pose.minEdgeCoverage)) : 0.45;
  pose.maxReprojectionErrorPx = Number.isFinite(Number(pose.maxReprojectionErrorPx)) ? Math.max(0.1, Math.min(200, Number(pose.maxReprojectionErrorPx))) : 8;
  pose.clipPoseGuardEnabled = pose.clipPoseGuardEnabled !== false;
  pose.clipInvalidationThreshold = Number.isFinite(Number(pose.clipInvalidationThreshold)) ? clamp01(Number(pose.clipInvalidationThreshold)) : 1;
  pose.cornerAwareFitEnabled = pose.cornerAwareFitEnabled === true;
  pose.minCornerAgreementLinks = Number.isFinite(Number(pose.minCornerAgreementLinks)) ? Math.max(0, Math.min(12, Math.round(Number(pose.minCornerAgreementLinks)))) : 2;
  pose.completeCornerAgreementBonus = pose.completeCornerAgreementBonus !== false;
  pose.multiGateExtraCornerLimit = Number.isFinite(Number(pose.multiGateExtraCornerLimit)) ? Math.max(4, Math.min(24, Math.round(Number(pose.multiGateExtraCornerLimit)))) : 4;
  pose.maxPoseCandidates = Number.isFinite(Number(pose.maxPoseCandidates)) ? Math.max(1, Math.min(64, Math.round(Number(pose.maxPoseCandidates)))) : 12;
  pose.textureOpacity = Number.isFinite(Number(pose.textureOpacity)) ? clamp01(Number(pose.textureOpacity)) : 0.85;
  pose.candidateOpacity = Number.isFinite(Number(pose.candidateOpacity)) ? clamp01(Number(pose.candidateOpacity)) : 0.85;
  pose.showFrustum = pose.showFrustum !== false;
  pose.showOutline = pose.showOutline !== false;
  pose.showScores = pose.showScores !== false;
  pose.showLabels = pose.showLabels === true;
  return pose;
}

function squarePoseBuildSettings() {
  const settings = squarePoseState();
  return {
    squareSizeM: settings.squareSizeM,
    edgeTolerancePx: settings.edgeTolerancePx,
    minEdgeCoverage: settings.minEdgeCoverage,
    maxReprojectionErrorPx: settings.maxReprojectionErrorPx,
    clipPoseGuardEnabled: settings.clipPoseGuardEnabled,
    clipInvalidationThreshold: settings.clipInvalidationThreshold,
    cornerAwareFitEnabled: false,
    minCornerAgreementLinks: settings.minCornerAgreementLinks,
    completeCornerAgreementBonus: settings.completeCornerAgreementBonus,
    multiGateExtraCornerLimit: settings.multiGateExtraCornerLimit,
    maxPoseCandidates: settings.maxPoseCandidates
  };
}

function bboxFlowState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.bboxFlow = state.review.bboxFlow && typeof state.review.bboxFlow === 'object'
    ? state.review.bboxFlow
    : {};
  const bbox = state.review.bboxFlow;
  const defaults = BBOX_FLOW_DEFAULTS;
  bbox.minPixels = Number.isFinite(Number(bbox.minPixels)) ? Math.max(1, Math.min(50000, Math.round(Number(bbox.minPixels)))) : defaults.minPixels;
  bbox.maxBboxesPerFrame = Number.isFinite(Number(bbox.maxBboxesPerFrame)) ? Math.max(1, Math.min(5000, Math.round(Number(bbox.maxBboxesPerFrame)))) : defaults.maxBboxesPerFrame;
  bbox.fitTightness = Number.isFinite(Number(bbox.fitTightness)) ? Math.max(0, Math.min(100, Math.round(Number(bbox.fitTightness)))) : defaults.fitTightness;
  bbox.quadFitEnabled = typeof bbox.quadFitEnabled === 'boolean' ? bbox.quadFitEnabled : defaults.quadFitEnabled;
  bbox.quadFitMode = ['axis-bbox', 'rotated-rect', 'free-quad'].includes(bbox.quadFitMode) ? bbox.quadFitMode : defaults.quadFitMode;
  bbox.targetAspect = Number.isFinite(Number(bbox.targetAspect)) ? Math.max(0.1, Math.min(10, Number(bbox.targetAspect))) : defaults.targetAspect;
  bbox.aspectTolerance = Number.isFinite(Number(bbox.aspectTolerance)) ? Math.max(0, Math.min(2, Number(bbox.aspectTolerance))) : defaults.aspectTolerance;
  bbox.quadThicknessPx = Number.isFinite(Number(bbox.quadThicknessPx)) ? Math.max(1, Math.min(80, Math.round(Number(bbox.quadThicknessPx)))) : defaults.quadThicknessPx;
  bbox.edgeCoverageMin = Number.isFinite(Number(bbox.edgeCoverageMin)) ? clamp01(Number(bbox.edgeCoverageMin)) : defaults.edgeCoverageMin;
  bbox.cornerMinPixels = Number.isFinite(Number(bbox.cornerMinPixels)) ? Math.max(0, Math.min(10000, Math.round(Number(bbox.cornerMinPixels)))) : defaults.cornerMinPixels;
  bbox.voidOverlapMaxRatio = Number.isFinite(Number(bbox.voidOverlapMaxRatio)) ? clamp01(Number(bbox.voidOverlapMaxRatio)) : defaults.voidOverlapMaxRatio;
  bbox.voidOverlapMinPixels = Number.isFinite(Number(bbox.voidOverlapMinPixels)) ? Math.max(1, Math.min(50000, Math.round(Number(bbox.voidOverlapMinPixels)))) : defaults.voidOverlapMinPixels;
  bbox.showRawBboxes = typeof bbox.showRawBboxes === 'boolean' ? bbox.showRawBboxes : defaults.showRawBboxes;
  bbox.quadOverlayOpacity = Number.isFinite(Number(bbox.quadOverlayOpacity)) ? clamp01(Number(bbox.quadOverlayOpacity)) : defaults.quadOverlayOpacity;
  bbox.viewMode = ['baseline', 'maskbits', 'compare'].includes(bbox.viewMode) ? bbox.viewMode : defaults.viewMode;
  bbox.fovClip = bbox.fovClip && typeof bbox.fovClip === 'object' ? bbox.fovClip : {};
  bbox.fovClip.enabled = typeof bbox.fovClip.enabled === 'boolean' ? bbox.fovClip.enabled : defaults.fovClip.enabled;
  bbox.fovClip.showOverlay = typeof bbox.fovClip.showOverlay === 'boolean' ? bbox.fovClip.showOverlay : defaults.fovClip.showOverlay;
  bbox.fovClip.marginPx = Number.isFinite(Number(bbox.fovClip.marginPx)) ? Math.max(0, Math.min(64, Math.round(Number(bbox.fovClip.marginPx)))) : defaults.fovClip.marginPx;
  bbox.fovClip.minContactPixels = Number.isFinite(Number(bbox.fovClip.minContactPixels)) ? Math.max(1, Math.min(5000, Math.round(Number(bbox.fovClip.minContactPixels)))) : defaults.fovClip.minContactPixels;
  bbox.fovClip.minContactRatio = Number.isFinite(Number(bbox.fovClip.minContactRatio)) ? clamp01(Number(bbox.fovClip.minContactRatio)) : defaults.fovClip.minContactRatio;
  bbox.fovClip.requireBboxTouch = typeof bbox.fovClip.requireBboxTouch === 'boolean' ? bbox.fovClip.requireBboxTouch : defaults.fovClip.requireBboxTouch;
  bbox.fovClip.warnOnly = typeof bbox.fovClip.warnOnly === 'boolean' ? bbox.fovClip.warnOnly : defaults.fovClip.warnOnly;
  return bbox;
}

function instanceTrackingState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.instanceTracking = state.review.instanceTracking && typeof state.review.instanceTracking === 'object'
    ? state.review.instanceTracking
    : {};
  const tracking = state.review.instanceTracking;
  tracking.enabled = tracking.enabled === true;
  tracking.showOverlay = tracking.showOverlay !== false;
  tracking.showLabels = tracking.showLabels !== false;
  tracking.showLinks = tracking.showLinks !== false;
  tracking.showCandidates = tracking.showCandidates === true;
  tracking.showReadout = tracking.showReadout !== false;
  tracking.maxFrameGap = Number.isFinite(Number(tracking.maxFrameGap)) ? Math.max(1, Math.min(120, Math.round(Number(tracking.maxFrameGap)))) : 5;
  tracking.minAssociationScore = Number.isFinite(Number(tracking.minAssociationScore)) ? clamp01(Number(tracking.minAssociationScore)) : 0.45;
  tracking.max2dDistancePx = Number.isFinite(Number(tracking.max2dDistancePx)) ? Math.max(1, Math.min(1000, Number(tracking.max2dDistancePx))) : 90;
  tracking.minIou = Number.isFinite(Number(tracking.minIou)) ? clamp01(Number(tracking.minIou)) : 0.02;
  tracking.max3dDistanceM = Number.isFinite(Number(tracking.max3dDistanceM)) ? Math.max(0.01, Math.min(100, Number(tracking.max3dDistanceM))) : 3;
  tracking.maxRpyDeltaDeg = Number.isFinite(Number(tracking.maxRpyDeltaDeg)) ? Math.max(1, Math.min(180, Number(tracking.maxRpyDeltaDeg))) : 45;
  tracking.splitCandidateScore = Number.isFinite(Number(tracking.splitCandidateScore)) ? clamp01(Number(tracking.splitCandidateScore)) : 0.35;
  tracking.debugCandidateLimit = Number.isFinite(Number(tracking.debugCandidateLimit)) ? Math.max(1, Math.min(50, Math.round(Number(tracking.debugCandidateLimit)))) : 5;
  tracking.trailLengthFrames = Number.isFinite(Number(tracking.trailLengthFrames)) ? Math.max(1, Math.min(240, Math.round(Number(tracking.trailLengthFrames)))) : 12;
  if (tracking.poseSource === 'squarePose') tracking.poseSource = 'poseFit';
  tracking.poseSource = ['poseFit', 'none'].includes(tracking.poseSource) ? tracking.poseSource : 'poseFit';
  return tracking;
}

function contourHierarchyState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.contourHierarchy = state.review.contourHierarchy && typeof state.review.contourHierarchy === 'object'
    ? state.review.contourHierarchy
    : {};
  const contour = state.review.contourHierarchy;
  contour.show = contour.show !== false;
  contour.maskSource = ['enabled-layers', '002-only'].includes(contour.maskSource) ? contour.maskSource : 'enabled-layers';
  contour.minOuterAreaPx = Number.isFinite(Number(contour.minOuterAreaPx)) ? Math.max(1, Math.min(500000, Math.round(Number(contour.minOuterAreaPx)))) : 10;
  contour.minVoidAreaPx = Number.isFinite(Number(contour.minVoidAreaPx)) ? Math.max(1, Math.min(500000, Math.round(Number(contour.minVoidAreaPx)))) : 20;
  contour.maxVoidsPerBbox = Number.isFinite(Number(contour.maxVoidsPerBbox)) ? Math.max(0, Math.min(200, Math.round(Number(contour.maxVoidsPerBbox)))) : 12;
  contour.simplifyEpsilonPx = Number.isFinite(Number(contour.simplifyEpsilonPx)) ? Math.max(0, Math.min(32, Number(contour.simplifyEpsilonPx))) : 1.5;
  contour.closeRadiusPx = Number.isFinite(Number(contour.closeRadiusPx)) ? Math.max(0, Math.min(32, Math.round(Number(contour.closeRadiusPx)))) : 0;
  contour.openRadiusPx = Number.isFinite(Number(contour.openRadiusPx)) ? Math.max(0, Math.min(32, Math.round(Number(contour.openRadiusPx)))) : 0;
  contour.notchProximityPx = Number.isFinite(Number(contour.notchProximityPx)) ? Math.max(0, Math.min(128, Math.round(Number(contour.notchProximityPx)))) : 4;
  contour.includeSmallContours = contour.includeSmallContours === true;
  contour.lineThicknessPx = Number.isFinite(Number(contour.lineThicknessPx)) ? Math.max(1, Math.min(8, Math.round(Number(contour.lineThicknessPx)))) : 1;
  contour.opacity = Number.isFinite(Number(contour.opacity)) ? clamp01(Number(contour.opacity)) : 0.95;
  contour.showLabels = contour.showLabels !== false;
  return contour;
}

function contourAnalysisSettings() {
  const settings = contourHierarchyState();
  return {
    maskSource: settings.maskSource,
    minOuterAreaPx: settings.minOuterAreaPx,
    minVoidAreaPx: settings.minVoidAreaPx,
    maxVoidsPerBbox: settings.maxVoidsPerBbox,
    simplifyEpsilonPx: settings.simplifyEpsilonPx,
    closeRadiusPx: settings.closeRadiusPx,
    openRadiusPx: settings.openRadiusPx,
    notchProximityPx: settings.notchProximityPx,
    includeSmallContours: settings.includeSmallContours
  };
}

function cornerFlowState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.cornerFlow = state.review.cornerFlow && typeof state.review.cornerFlow === 'object'
    ? state.review.cornerFlow
    : {};
  const corner = state.review.cornerFlow;
  const legacy = {
    radiusPx: Number.isFinite(Number(corner.radiusPx)) ? Number(corner.radiusPx) : 8,
    minAngleDeg: Number.isFinite(Number(corner.minAngleDeg)) ? Number(corner.minAngleDeg) : 45,
    maxAngleDeg: Number.isFinite(Number(corner.maxAngleDeg)) ? Number(corner.maxAngleDeg) : 135,
    minSupportPixels: Number.isFinite(Number(corner.minSupportPixels)) ? Number(corner.minSupportPixels) : 4,
    contourEpsilonPx: Number.isFinite(Number(corner.contourEpsilonPx)) ? Number(corner.contourEpsilonPx) : 2,
    minDistancePx: Number.isFinite(Number(corner.minDistancePx)) ? Number(corner.minDistancePx) : 8,
    maxCorners: Number.isFinite(Number(corner.maxCornersPerFrame)) ? Number(corner.maxCornersPerFrame) : 64
  };
  const normalizeGroup = (group, maxKey, maxDefault) => {
    const source = group && typeof group === 'object' ? group : {};
    const normalized = source;
    normalized.radiusPx = Number.isFinite(Number(normalized.radiusPx)) ? Math.max(2, Math.min(64, Math.round(Number(normalized.radiusPx)))) : Math.max(2, Math.min(64, Math.round(legacy.radiusPx)));
    normalized.minAngleDeg = Number.isFinite(Number(normalized.minAngleDeg)) ? Math.max(1, Math.min(179, Math.round(Number(normalized.minAngleDeg)))) : Math.max(1, Math.min(179, Math.round(legacy.minAngleDeg)));
    normalized.maxAngleDeg = Number.isFinite(Number(normalized.maxAngleDeg)) ? Math.max(1, Math.min(179, Math.round(Number(normalized.maxAngleDeg)))) : Math.max(1, Math.min(179, Math.round(legacy.maxAngleDeg)));
    if (normalized.maxAngleDeg < normalized.minAngleDeg) {
      const swap = normalized.minAngleDeg;
      normalized.minAngleDeg = normalized.maxAngleDeg;
      normalized.maxAngleDeg = swap;
    }
    normalized.minSupportPixels = Number.isFinite(Number(normalized.minSupportPixels)) ? Math.max(1, Math.min(5000, Math.round(Number(normalized.minSupportPixels)))) : Math.max(1, Math.min(5000, Math.round(legacy.minSupportPixels)));
    normalized.contourEpsilonPx = Number.isFinite(Number(normalized.contourEpsilonPx)) ? Math.max(0.25, Math.min(24, Number(normalized.contourEpsilonPx))) : Math.max(0.25, Math.min(24, legacy.contourEpsilonPx));
    normalized.minDistancePx = Number.isFinite(Number(normalized.minDistancePx)) ? Math.max(1, Math.min(80, Math.round(Number(normalized.minDistancePx)))) : Math.max(1, Math.min(80, Math.round(legacy.minDistancePx)));
    normalized[maxKey] = Number.isFinite(Number(normalized[maxKey])) ? Math.max(1, Math.min(5000, Math.round(Number(normalized[maxKey])))) : Math.max(1, Math.min(5000, Math.round(Number(maxDefault))));
    normalized.minMaskInsidePoints = Number.isFinite(Number(normalized.minMaskInsidePoints)) ? Math.max(0, Math.min(5000, Math.round(Number(normalized.minMaskInsidePoints)))) : 0;
    normalized.maxMaskInsidePoints = Number.isFinite(Number(normalized.maxMaskInsidePoints)) ? Math.max(0, Math.min(5000, Math.round(Number(normalized.maxMaskInsidePoints)))) : normalized[maxKey];
    if (normalized.maxMaskInsidePoints < normalized.minMaskInsidePoints) normalized.maxMaskInsidePoints = normalized.minMaskInsidePoints;
    normalized.minMaskOutsidePoints = Number.isFinite(Number(normalized.minMaskOutsidePoints)) ? Math.max(0, Math.min(5000, Math.round(Number(normalized.minMaskOutsidePoints)))) : 0;
    normalized.maxMaskOutsidePoints = Number.isFinite(Number(normalized.maxMaskOutsidePoints)) ? Math.max(0, Math.min(5000, Math.round(Number(normalized.maxMaskOutsidePoints)))) : normalized[maxKey];
    if (normalized.maxMaskOutsidePoints < normalized.minMaskOutsidePoints) normalized.maxMaskOutsidePoints = normalized.minMaskOutsidePoints;
    return normalized;
  };
  const normalizeType = (typeKey) => {
    const def = CORNER_TYPE_DEFS[typeKey];
    const base = def.base === 'void' ? corner.void : corner.hull;
    const source = corner.types?.[typeKey] && typeof corner.types[typeKey] === 'object' ? corner.types[typeKey] : {};
    const normalized = source;
    normalized.category = def.category;
    normalized.targetPosition = def.position;
    normalized.radiusPx = Number.isFinite(Number(normalized.radiusPx)) ? Math.max(2, Math.min(64, Math.round(Number(normalized.radiusPx)))) : base.radiusPx;
    normalized.minAngleDeg = Number.isFinite(Number(normalized.minAngleDeg)) ? Math.max(1, Math.min(179, Math.round(Number(normalized.minAngleDeg)))) : base.minAngleDeg;
    normalized.maxAngleDeg = Number.isFinite(Number(normalized.maxAngleDeg)) ? Math.max(1, Math.min(179, Math.round(Number(normalized.maxAngleDeg)))) : base.maxAngleDeg;
    if (normalized.maxAngleDeg < normalized.minAngleDeg) {
      const swap = normalized.minAngleDeg;
      normalized.minAngleDeg = normalized.maxAngleDeg;
      normalized.maxAngleDeg = swap;
    }
    normalized.minSupportPixels = Number.isFinite(Number(normalized.minSupportPixels)) ? Math.max(1, Math.min(5000, Math.round(Number(normalized.minSupportPixels)))) : base.minSupportPixels;
    normalized.contourEpsilonPx = Number.isFinite(Number(normalized.contourEpsilonPx)) ? Math.max(0.25, Math.min(24, Number(normalized.contourEpsilonPx))) : base.contourEpsilonPx;
    normalized.minDistancePx = Number.isFinite(Number(normalized.minDistancePx)) ? Math.max(1, Math.min(80, Math.round(Number(normalized.minDistancePx)))) : base.minDistancePx;
    normalized.minPoints = Number.isFinite(Number(normalized.minPoints)) ? Math.max(0, Math.min(5000, Math.round(Number(normalized.minPoints)))) : Math.max(0, Math.min(5000, Math.round(Number(base[def.legacyMin] || 0))));
    normalized.maxPoints = Number.isFinite(Number(normalized.maxPoints)) ? Math.max(0, Math.min(5000, Math.round(Number(normalized.maxPoints)))) : Math.max(0, Math.min(5000, Math.round(Number(base[def.legacyMax] ?? def.maxDefault))));
    if (normalized.maxPoints < normalized.minPoints) normalized.maxPoints = normalized.minPoints;
    if (def.category === 'void') {
      normalized.minAreaPx = Number.isFinite(Number(normalized.minAreaPx)) ? Math.max(1, Math.min(100000, Math.round(Number(normalized.minAreaPx)))) : corner.void.minAreaPx;
    }
    return normalized;
  };
  corner.hull = normalizeGroup(corner.hull, 'maxCornersPerBbox', legacy.maxCorners || 64);
  corner.void = normalizeGroup(corner.void, 'maxCornersPerVoid', legacy.maxCorners || 24);
  corner.void.minAreaPx = Number.isFinite(Number(corner.void.minAreaPx)) ? Math.max(1, Math.min(100000, Math.round(Number(corner.void.minAreaPx)))) : 20;
  corner.types = corner.types && typeof corner.types === 'object' ? corner.types : {};
  for (const typeKey of CORNER_TYPE_KEYS) {
    corner.types[typeKey] = normalizeType(typeKey);
  }
  corner.showWhite = typeof corner.showWhite === 'boolean' ? corner.showWhite : corner.showHull !== false && corner.showMaskWedge !== false;
  corner.showBlack = typeof corner.showBlack === 'boolean' ? corner.showBlack : corner.showHull !== false && corner.showVoidWedge !== false;
  corner.showOrange = typeof corner.showOrange === 'boolean' ? corner.showOrange : corner.showVoid !== false && corner.showMaskWedge !== false;
  corner.showGreen = typeof corner.showGreen === 'boolean' ? corner.showGreen : corner.showVoid !== false && corner.showVoidWedge !== false;
  corner.showHull = corner.showWhite || corner.showBlack;
  corner.showVoid = corner.showOrange || corner.showGreen;
  corner.showMaskWedge = corner.showHull;
  corner.showVoidWedge = corner.showVoid;
  corner.showAmbiguous = corner.showAmbiguous === true;
  corner.overlayOpacity = Number.isFinite(Number(corner.overlayOpacity)) ? clamp01(Number(corner.overlayOpacity)) : 0.95;
  return corner;
}

function cornerAgreementState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.cornerAgreement = state.review.cornerAgreement && typeof state.review.cornerAgreement === 'object'
    ? state.review.cornerAgreement
    : {};
  const agreement = state.review.cornerAgreement;
  agreement.show = agreement.show !== false;
  agreement.maxDistancePx = Number.isFinite(Number(agreement.maxDistancePx)) ? Math.max(4, Math.min(220, Math.round(Number(agreement.maxDistancePx)))) : 96;
  agreement.whiteToleranceDeg = Number.isFinite(Number(agreement.whiteToleranceDeg)) ? Math.max(0, Math.min(120, Math.round(Number(agreement.whiteToleranceDeg)))) : 35;
  agreement.angleToleranceDeg = Number.isFinite(Number(agreement.angleToleranceDeg)) ? Math.max(0, Math.min(120, Math.round(Number(agreement.angleToleranceDeg)))) : 25;
  agreement.minScore = Number.isFinite(Number(agreement.minScore)) ? clamp01(Number(agreement.minScore)) : 0.55;
  agreement.maxLinksPerCorner = Number.isFinite(Number(agreement.maxLinksPerCorner)) ? Math.max(1, Math.min(4, Math.round(Number(agreement.maxLinksPerCorner)))) : 1;
  agreement.lineThicknessPx = Number.isFinite(Number(agreement.lineThicknessPx)) ? Math.max(1, Math.min(30, Math.round(Number(agreement.lineThicknessPx)))) : 8;
  agreement.opacity = Number.isFinite(Number(agreement.opacity)) ? clamp01(Number(agreement.opacity)) : 0.9;
  agreement.showStructures = agreement.showStructures !== false;
  agreement.oppositeToleranceDeg = Number.isFinite(Number(agreement.oppositeToleranceDeg)) ? Math.max(0, Math.min(90, Math.round(Number(agreement.oppositeToleranceDeg)))) : 25;
  return agreement;
}

function layer002QuadState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.layer002Quad = state.review.layer002Quad && typeof state.review.layer002Quad === 'object'
    ? state.review.layer002Quad
    : {};
  const quad = state.review.layer002Quad;
  quad.show = quad.show !== false;
  quad.prefix = '002';
  quad.supportMode = ['002-only', 'any-mask'].includes(quad.supportMode) ? quad.supportMode : 'any-mask';
  quad.forbiddenMode = ['non-002', 'non-mask'].includes(quad.forbiddenMode) ? quad.forbiddenMode : 'non-002';
  quad.maxQuadsPerFrame = Number.isFinite(Number(quad.maxQuadsPerFrame)) ? Math.max(1, Math.min(50, Math.round(Number(quad.maxQuadsPerFrame)))) : 5;
  quad.minPixels = Number.isFinite(Number(quad.minPixels)) ? Math.max(1, Math.min(5000, Math.round(Number(quad.minPixels)))) : 80;
  quad.edgeBandPx = Number.isFinite(Number(quad.hullEdgeBandPx))
    ? Math.max(0, Math.min(24, Math.round(Number(quad.hullEdgeBandPx))))
    : 4;
  quad.hullEdgeBandPx = quad.edgeBandPx;
  quad.minEdgeCoverage = Number.isFinite(Number(quad.hullMinEdgeCoverage))
    ? clamp01(Number(quad.hullMinEdgeCoverage))
    : 1;
  quad.hullMinEdgeCoverage = quad.minEdgeCoverage;
  quad.maxUnsupportedGapPx = Number.isFinite(Number(quad.hullMaxUnsupportedGapPx))
    ? Math.max(0, Math.min(80, Math.round(Number(quad.hullMaxUnsupportedGapPx))))
    : 0;
  quad.hullMaxUnsupportedGapPx = quad.maxUnsupportedGapPx;
  quad.inwardMaxPx = Number.isFinite(Number(quad.inwardMaxPx))
    ? Math.max(0, Math.min(220, Math.round(Number(quad.inwardMaxPx))))
    : 80;
  quad.insetStepPx = Number.isFinite(Number(quad.insetStepPx))
    ? Math.max(1, Math.min(20, Math.round(Number(quad.insetStepPx))))
    : 2;
  quad.angleSweepDeg = Number.isFinite(Number(quad.hullAngleSweepDeg))
    ? Math.max(0, Math.min(90, Math.round(Number(quad.hullAngleSweepDeg))))
    : 20;
  quad.hullAngleSweepDeg = quad.angleSweepDeg;
  quad.angleStepDeg = Number.isFinite(Number(quad.hullAngleStepDeg))
    ? Math.max(1, Math.min(15, Math.round(Number(quad.hullAngleStepDeg))))
    : 3;
  quad.hullAngleStepDeg = quad.angleStepDeg;
  quad.aspectTolerance = Number.isFinite(Number(quad.aspectTolerance))
    ? Math.max(0, Math.min(2, Number(quad.aspectTolerance)))
    : 0.8;
  quad.lineThicknessPx = Number.isFinite(Number(quad.lineThicknessPx)) ? Math.max(1, Math.min(16, Math.round(Number(quad.lineThicknessPx)))) : 2;
  quad.opacity = Number.isFinite(Number(quad.opacity)) ? clamp01(Number(quad.opacity)) : 0.9;
  quad.showRejected = quad.showRejected === true;
  return quad;
}

function layerMetadataState() {
  state.review = state.review && typeof state.review === 'object' ? state.review : {};
  state.review.layerMetadata = state.review.layerMetadata && typeof state.review.layerMetadata === 'object'
    ? state.review.layerMetadata
    : {};
  const metadata = state.review.layerMetadata;
  metadata.layerConfidenceByPrefix = metadata.layerConfidenceByPrefix && typeof metadata.layerConfidenceByPrefix === 'object'
    ? metadata.layerConfidenceByPrefix
    : {};
  metadata.layerGroupSlotsByPrefix = metadata.layerGroupSlotsByPrefix && typeof metadata.layerGroupSlotsByPrefix === 'object'
    ? metadata.layerGroupSlotsByPrefix
    : {};
  metadata.layerGroupIdsByPrefix = metadata.layerGroupIdsByPrefix && typeof metadata.layerGroupIdsByPrefix === 'object'
    ? metadata.layerGroupIdsByPrefix
    : {};
  return metadata;
}

function layerEnabled(prefix) {
  return visualizationState().layerEnabledByPrefix[String(prefix)] !== false;
}

function setLayerEnabled(prefix, enabled) {
  visualizationState().layerEnabledByPrefix[String(prefix)] = Boolean(enabled);
  state.dirtyView = true;
}

function layerOpacity(prefix) {
  const value = visualizationState().layerOpacityByPrefix[String(prefix)];
  return Number.isFinite(Number(value)) ? clamp01(value) : 0.38;
}

function setLayerOpacity(prefix, value) {
  visualizationState().layerOpacityByPrefix[String(prefix)] = clamp01(value);
  state.dirtyView = true;
}

function layerColor(item) {
  const prefix = String(item?.prefix || '');
  return normalizeHex(visualizationState().layerColorByPrefix[prefix], normalizeHex(item?.color, '#ffffff'));
}

function setLayerColor(prefix, color) {
  visualizationState().layerColorByPrefix[String(prefix)] = normalizeHex(color);
  state.dirtyView = true;
}

function sourceOpacity() {
  const value = visualizationState().sourceOpacity;
  return Number.isFinite(Number(value)) ? clamp01(value) : 1;
}

function setSourceOpacity(value) {
  visualizationState().sourceOpacity = clamp01(value);
  state.dirtyView = true;
}

function layerConfidence(item) {
  const prefix = String(item?.prefix || '');
  const value = layerMetadataState().layerConfidenceByPrefix[prefix];
  if (Number.isFinite(Number(value))) return clamp01(value);
  return clamp01(item?.confidence ?? 0);
}

function setLayerConfidence(prefix, value) {
  layerMetadataState().layerConfidenceByPrefix[String(prefix)] = clamp01(value);
  state.dirtyView = true;
}

function layerGroupSlots(prefix) {
  const value = layerMetadataState().layerGroupSlotsByPrefix[String(prefix)];
  const slots = Array.isArray(value) ? value.slice(0, 3) : [];
  while (slots.length < 3) slots.push('');
  return slots.map((item) => String(item || ''));
}

function setLayerGroupSlot(prefix, slotIndex, value) {
  const metadata = layerMetadataState();
  const slots = layerGroupSlots(prefix);
  slots[Number(slotIndex) || 0] = String(value || '');
  const seen = [];
  for (const slot of slots) {
    if (slot && !seen.includes(slot)) seen.push(slot);
  }
  metadata.layerGroupSlotsByPrefix[String(prefix)] = slots;
  metadata.layerGroupIdsByPrefix[String(prefix)] = seen;
  state.dirtyView = true;
}

function currentFrameEntry() {
  return state.precompute?.frames?.[state.frameIndex] || null;
}

function currentMemoryFrame() {
  const frames = Array.isArray(state.memory?.frames) ? state.memory.frames : [];
  return frames.find((frame) => Number(frame.frameOrdinal) === Number(state.frameIndex)) || null;
}

function requestedInitialFrameIndex() {
  const params = new URLSearchParams(window.location.search || '');
  const raw = params.get('frame') || params.get('frameOrdinal');
  const requested = Number(raw);
  const frames = Array.isArray(state.precompute?.frames) ? state.precompute.frames : [];
  if (!Number.isFinite(requested) || !frames.length) return 0;
  const direct = frames.findIndex((frame) => Number(frame.index) === requested || Number(frame.frameOrdinal) === requested);
  if (direct >= 0) return direct;
  return Math.max(0, Math.min(frames.length - 1, Math.round(requested)));
}

function stackHeight() {
  return state.viewport.frameHeight * MAIN_VIEW_COUNT + VIEW_GAP * (MAIN_VIEW_COUNT - 1);
}

function stackPaneStart(paneIndex) {
  return paneIndex * (state.viewport.frameHeight + VIEW_GAP);
}

function auxPaneCount() {
  return Math.max(1, els.auxViewRail?.querySelectorAll('.auxPane')?.length || 1);
}

function updateSurfaceSize(width = state.viewport.frameWidth, height = state.viewport.frameHeight) {
  state.viewport.frameWidth = Math.max(1, Math.round(Number(width) || 640));
  state.viewport.frameHeight = Math.max(1, Math.round(Number(height) || 360));
  for (const surface of [els.frameSurface, els.poseSurface]) {
    if (!surface) continue;
    surface.style.width = `${state.viewport.frameWidth}px`;
    surface.style.height = `${state.viewport.frameHeight}px`;
    if (surface.parentElement) {
      surface.parentElement.style.width = `${state.viewport.frameWidth}px`;
      surface.parentElement.style.height = `${state.viewport.frameHeight}px`;
    }
  }
  for (const canvas of [els.squarePoseCanvas, els.instance3dCanvas, els.auxMaskCanvas]) {
    if (!canvas) continue;
    if (canvas.width !== state.viewport.frameWidth) canvas.width = state.viewport.frameWidth;
    if (canvas.height !== state.viewport.frameHeight) canvas.height = state.viewport.frameHeight;
  }
  if (els.viewStack) {
    els.viewStack.style.width = `${state.viewport.frameWidth}px`;
    els.viewStack.style.height = `${stackHeight()}px`;
  }
  if (els.bboxOverlay) {
    els.bboxOverlay.setAttribute('viewBox', `0 0 ${state.viewport.frameWidth} ${state.viewport.frameHeight}`);
    els.bboxOverlay.setAttribute('width', String(state.viewport.frameWidth));
    els.bboxOverlay.setAttribute('height', String(state.viewport.frameHeight));
  }
  if (els.squarePoseLabelOverlay) {
    els.squarePoseLabelOverlay.setAttribute('viewBox', `0 0 ${state.viewport.frameWidth} ${state.viewport.frameHeight}`);
    els.squarePoseLabelOverlay.setAttribute('width', String(state.viewport.frameWidth));
    els.squarePoseLabelOverlay.setAttribute('height', String(state.viewport.frameHeight));
  }
  resizeSquarePoseRenderer();
  resizeInstance3dRenderer();
  layoutFrameSurface();
}

function fitSurfaceToStage() {
  const wrap = els.canvasWrap.getBoundingClientRect();
  const height = stackHeight();
  const fit = Math.min(
    wrap.width / Math.max(1, state.viewport.frameWidth),
    wrap.height / Math.max(1, height)
  );
  state.viewport.fitScale = Number.isFinite(fit) && fit > 0 ? fit : 1;
  const mainWidth = state.viewport.frameWidth * state.viewport.fitScale;
  const mainHeight = height * state.viewport.fitScale;
  const remainingWidth = wrap.width - mainWidth - AUX_RAIL_GAP;
  const auxVisible = remainingWidth >= AUX_MIN_WIDTH;
  const auxWidth = auxVisible
    ? Math.floor(Math.min(remainingWidth, AUX_MAX_WIDTH))
    : 0;
  state.viewport.mainViewportWidth = mainWidth;
  state.viewport.mainViewportHeight = mainHeight;
  state.viewport.auxVisible = auxVisible;
  state.viewport.auxWidth = auxVisible ? auxWidth : 0;
  layoutAuxRail();
}

function clampPan() {
  const scale = state.viewport.fitScale * state.viewport.scale;
  const width = state.viewport.frameWidth * scale;
  const height = stackHeight() * scale;
  const minX = Math.min(0, state.viewport.mainViewportWidth - width);
  const minY = Math.min(0, state.viewport.mainViewportHeight - height);
  state.viewport.panX = Math.max(minX, Math.min(0, state.viewport.panX));
  state.viewport.panY = Math.max(minY, Math.min(0, state.viewport.panY));
}

function layoutFrameSurface() {
  if (!els.viewStack || !els.canvasWrap) return;
  fitSurfaceToStage();
  clampPan();
  const scale = state.viewport.fitScale * state.viewport.scale;
  els.viewStack.style.transform = `translate(${state.viewport.panX}px, ${state.viewport.panY}px) scale(${scale})`;
}

function layoutAuxRail() {
  if (!els.auxViewRail) return;
  if (!state.viewport.auxVisible) {
    els.auxViewRail.style.display = 'none';
    return;
  }
  const paneHeight = Math.max(1, Math.floor(state.viewport.auxWidth * state.viewport.frameHeight / Math.max(1, state.viewport.frameWidth)));
  els.auxViewRail.style.display = 'grid';
  els.auxViewRail.style.left = `${Math.round(state.viewport.mainViewportWidth + AUX_RAIL_GAP)}px`;
  els.auxViewRail.style.width = `${state.viewport.auxWidth}px`;
  els.auxViewRail.style.height = `${Math.max(1, Math.floor(state.viewport.mainViewportHeight))}px`;
  for (const pane of els.auxViewRail.querySelectorAll('.auxPane')) {
    pane.style.width = `${state.viewport.auxWidth}px`;
    pane.style.height = `${paneHeight}px`;
  }
  const layoutKey = `${state.viewport.auxWidth}x${paneHeight}`;
  if (state.instance3d.lastLayoutKey !== layoutKey) {
    state.instance3d.lastLayoutKey = layoutKey;
    resizeInstance3dRenderer();
    scheduleInstance3dRender();
  }
}

function stagePointFromEvent(event) {
  const rect = els.canvasWrap.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

function isAuxRailEvent(event) {
  return Boolean(event.target?.closest?.('#auxViewRail'));
}

function stagePointInMainViewport(point) {
  const width = state.viewport.mainViewportWidth || state.viewport.frameWidth * state.viewport.fitScale;
  const height = state.viewport.mainViewportHeight || stackHeight() * state.viewport.fitScale;
  return point.x >= 0 && point.y >= 0 && point.x <= width && point.y <= height;
}

function stageToFramePoint(point) {
  const local = stageToStackPoint(point);
  if (!local || local.y >= state.viewport.frameHeight) return null;
  return { x: local.x, y: local.y };
}

function stageToStackPoint(point) {
  const scale = state.viewport.fitScale * state.viewport.scale;
  if (!scale) return null;
  const x = Math.floor((point.x - state.viewport.panX) / scale);
  const y = Math.floor((point.y - state.viewport.panY) / scale);
  const height = stackHeight();
  if (x < 0 || y < 0 || x >= state.viewport.frameWidth || y >= height) return null;
  return { x, y };
}

function zoomFrameAt(stagePoint, targetScale) {
  const previousScale = state.viewport.fitScale * state.viewport.scale;
  const frameX = (stagePoint.x - state.viewport.panX) / previousScale;
  const frameY = (stagePoint.y - state.viewport.panY) / previousScale;
  state.viewport.scale = Math.max(1, Math.min(40, targetScale));
  const nextScale = state.viewport.fitScale * state.viewport.scale;
  state.viewport.panX = stagePoint.x - frameX * nextScale;
  state.viewport.panY = stagePoint.y - frameY * nextScale;
  layoutFrameSurface();
}

function expandRanges(ranges) {
  const ids = new Set();
  for (const item of ranges || []) {
    if (!Array.isArray(item) || item.length !== 2) continue;
    let start = Math.max(0, Math.round(Number(item[0]) || 0));
    let end = Math.max(0, Math.round(Number(item[1]) || 0));
    if (end < start) [start, end] = [end, start];
    for (let value = start; value <= end; value++) ids.add(value);
  }
  return ids;
}

function decodedRuleRangesByPrefix() {
  const decoded = new Map();
  const summaries = state.memory?.classRules?.summaries;
  if (!Array.isArray(summaries)) return decoded;
  const activeRunKey = String(state.precompute?.runKey || '');
  for (const summary of summaries) {
    if (!summary || typeof summary !== 'object') continue;
    const prefix = String(summary.prefix || '');
    if (!prefix) continue;
    const summaryRunKey = String(summary.activePrecomputeRunKey || state.memory?.source?.runKey || '');
    if (activeRunKey && summaryRunKey && activeRunKey !== summaryRunKey) continue;
    const ranges = Array.isArray(summary.sourceColorIdRanges) ? summary.sourceColorIdRanges : summary.colorIdRanges;
    if (Array.isArray(ranges)) decoded.set(prefix, ranges);
  }
  return decoded;
}

function staticRuleRangesForActiveRun(ranges) {
  const activeRunKey = String(state.precompute?.runKey || '');
  const sourceRunKey = String(state.rules?.source?.precomputeRunKey || '');
  if (activeRunKey && sourceRunKey && activeRunKey !== sourceRunKey) return [];
  return ranges;
}

function initializeRuleSets() {
  state.ruleSets.clear();
  const decodedRules = decodedRuleRangesByPrefix();
  const classRules = state.rules?.classRules || {};
  for (const item of state.config.classes) {
    const prefix = String(item.prefix);
    const ranges = decodedRules.has(prefix)
      ? decodedRules.get(prefix)
      : staticRuleRangesForActiveRun(classRules[prefix]?.colorIdRanges || []);
    state.ruleSets.set(prefix, expandRanges(ranges));
  }
  rebuildClassMembership();
}

function rebuildClassMembership() {
  const colorCount = Number(state.precompute?.colorTable?.count || 0);
  if (!colorCount) return;
  const membership = new Uint32Array(colorCount);
  state.config.classes.forEach((item, index) => {
    if (index >= 31) return;
    const bit = 1 << index;
    const ids = state.ruleSets.get(String(item.prefix)) || new Set();
    for (const id of ids) {
      if (id >= 0 && id < colorCount) membership[id] |= bit;
    }
  });
  state.classMembership = membership;
}

async function loadColorTable() {
  const table = state.precompute?.colorTable;
  if (!table || state.colorTable) return state.colorTable;
  const response = await fetch(assetUrl(table.path), { cache: 'force-cache' });
  if (!response.ok) throw new Error(`failed to load color table ${response.status}`);
  const buffer = await response.arrayBuffer();
  state.colorTable = {
    rgb: new Uint8Array(buffer, table.arrays.rgb.offset, table.arrays.rgb.length),
    hue: new Float32Array(buffer, table.arrays.hue.offset, table.arrays.hue.length),
    saturation: new Float32Array(buffer, table.arrays.saturation.offset, table.arrays.saturation.length),
    lightness: new Float32Array(buffer, table.arrays.lightness.offset, table.arrays.lightness.length)
  };
  return state.colorTable;
}

function rgbForColorId(id) {
  if (!state.colorTable || id < 0) return null;
  const offset = id * 3;
  if (offset + 2 >= state.colorTable.rgb.length) return null;
  return [state.colorTable.rgb[offset], state.colorTable.rgb[offset + 1], state.colorTable.rgb[offset + 2]];
}

function hnlForColorId(id) {
  if (!state.colorTable || id < 0 || id >= state.colorTable.hue.length) return null;
  return {
    hue: state.colorTable.hue[id],
    saturation: state.colorTable.saturation[id],
    lightness: state.colorTable.lightness[id]
  };
}

function classLabelsForMask(mask) {
  if (!mask) return 'none';
  const labels = [];
  state.config.classes.forEach((item, index) => {
    if (mask & (1 << index)) labels.push(String(item.prefix));
  });
  return labels.length ? labels.join(',') : 'none';
}

function colorIdAt(point) {
  if (!point || !state.currentColorIds) return -1;
  return state.currentColorIds[point.y * els.overlayCanvas.width + point.x] ?? -1;
}

async function loadFrameColorIds(frame) {
  const info = frame.arrays.colorIds;
  const response = await fetch(assetUrl(frame.cachePath), { cache: 'force-cache' });
  if (!response.ok) throw new Error(`failed to load frame cache ${response.status}`);
  const buffer = await response.arrayBuffer();
  state.currentColorIds = new Uint32Array(buffer, info.offset || 0, info.length);
}

async function loadFrame(index) {
  const frameCount = state.precompute.frames.length;
  state.frameIndex = Math.max(0, Math.min(frameCount - 1, Math.round(Number(index) || 0)));
  const frame = currentFrameEntry();
  if (!frame) return;
  setStatus(`loading frame ${state.frameIndex + 1}`);

  const image = new Image();
  image.decoding = 'async';
  image.src = assetUrl(frame.path);
  if (!image.src) throw new Error('selected frame has no source image path');
  await image.decode();
  state.currentImage = image;

  const width = Number(state.precompute.width || 640);
  const height = Number(state.precompute.height || 360);
  for (const canvas of [els.sourceCanvas, els.overlayCanvas, els.squarePoseCanvas, els.instance3dCanvas, els.auxMaskCanvas]) {
    if (!canvas) continue;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
  }
  updateSurfaceSize(width, height);
  ctx.source.clearRect(0, 0, width, height);
  ctx.source.drawImage(image, 0, 0, width, height);
  await loadFrameColorIds(frame);
  renderOverlay();
  scheduleBboxRender();
  scheduleSquarePoseRender();
  scheduleInstance3dRender();
  renderFrameLayerList();
  setStatus(`frame ${state.frameIndex + 1} ready`);
}

function renderOverlay() {
  const width = els.overlayCanvas.width;
  const height = els.overlayCanvas.height;
  ctx.overlay.clearRect(0, 0, width, height);
  if (!state.currentColorIds || !state.classMembership) {
    renderAuxMaskPreview();
    return;
  }

  const imageData = ctx.overlay.createImageData(width, height);
  const data = imageData.data;
  const classRgbs = state.config.classes.map((item) => hexToRgb(layerColor(item)));
  const classOpacities = state.config.classes.map((item) => layerEnabled(item.prefix) ? layerOpacity(item.prefix) : 0);
  for (let i = 0; i < state.currentColorIds.length; i++) {
    const mask = state.classMembership[state.currentColorIds[i]];
    if (!mask) continue;
    let outR = 0;
    let outG = 0;
    let outB = 0;
    let outA = 0;
    for (let classIndex = 0; classIndex < state.config.classes.length; classIndex++) {
      if (!(mask & (1 << classIndex))) continue;
      const srcA = classOpacities[classIndex] || 0;
      if (srcA <= 0) continue;
      const rgb = classRgbs[classIndex] || [255, 255, 255];
      const nextA = srcA + outA * (1 - srcA);
      if (nextA <= 0) continue;
      outR = (rgb[0] * srcA + outR * outA * (1 - srcA)) / nextA;
      outG = (rgb[1] * srcA + outG * outA * (1 - srcA)) / nextA;
      outB = (rgb[2] * srcA + outB * outA * (1 - srcA)) / nextA;
      outA = nextA;
    }
    if (outA <= 0) continue;
    const offset = i * 4;
    data[offset] = Math.round(outR);
    data[offset + 1] = Math.round(outG);
    data[offset + 2] = Math.round(outB);
    data[offset + 3] = Math.round(outA * 255);
  }
  ctx.overlay.putImageData(imageData, 0, 0);
  renderAuxMaskPreview();
}

function renderAuxMaskPreview() {
  if (!ctx.auxMask || !els.auxMaskCanvas) return;
  const width = els.auxMaskCanvas.width;
  const height = els.auxMaskCanvas.height;
  ctx.auxMask.clearRect(0, 0, width, height);
  ctx.auxMask.drawImage(els.overlayCanvas, 0, 0, width, height);
}

function setFrameModuleEmpty(element, text, visible) {
  if (!element) return;
  if (text) element.textContent = text;
  element.hidden = !visible;
}

function updateTopCanvasSize(canvas, width, height) {
  if (!canvas) return;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
}

function sourceFrameUrl(entry) {
  return appScopedAssetUrl(framePathFromEntry(entry));
}

function frameVersionToken(entry) {
  return String(entry?.sha1 || entry?.sourceSha1 || entry?.mtimeNs || entry?.jpegSize || entry?.updatedAt || '');
}

function versionedResourceUrl(url, version) {
  if (!url || !version) return url;
  const separator = url.includes('?') ? '&' : '?';
  return `${url}${separator}v=${encodeURIComponent(version)}`;
}

async function loadSourceFrameImage(entry) {
  const url = sourceFrameUrl(entry);
  if (!url) throw new Error('source frame has no file reference');
  const version = frameVersionToken(entry);
  const cacheKey = `${url}|${version}`;
  if (state.source.imageCache.has(cacheKey)) return state.source.imageCache.get(cacheKey);
  const promise = new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`failed to load source frame ${url}`));
    image.src = versionedResourceUrl(url, version);
  }).catch((error) => {
    state.source.imageCache.delete(cacheKey);
    throw error;
  });
  state.source.imageCache.set(cacheKey, promise);
  while (state.source.imageCache.size > 32) state.source.imageCache.delete(state.source.imageCache.keys().next().value);
  return promise;
}

function sourceFrameDimensions(entry) {
  const width = Number(entry?.width || state.source.manifest?.width || state.colorMasks.manifest?.width || state.precompute?.width || state.viewport.frameWidth || 640);
  const height = Number(entry?.height || state.source.manifest?.height || state.colorMasks.manifest?.height || state.precompute?.height || state.viewport.frameHeight || 360);
  return { width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(height)) };
}

function updateFrameNavigator() {
  const count = sourceFrameCount();
  const index = selectedSourceFrameIndex();
  state.source.frameIndex = index;
  if (els.frameSlider) {
    els.frameSlider.max = String(Math.max(0, count - 1));
    els.frameSlider.value = String(index);
  }
  if (els.frameLabel) {
    const mode = state.source.mode === 'live' ? 'Live' : 'Frame';
    els.frameLabel.textContent = count ? `${mode} ${index + 1} / ${count}` : `${mode} 0 / 0`;
  }
}

async function renderTopSourceFrame(frame = sourceFrameEntry()) {
  if (!ctx.topSource || !els.topSourceCanvas) return;
  const { width, height } = sourceFrameDimensions(frame);
  const requestId = ++state.source.imageRequestId;
  updateTopCanvasSize(els.topSourceCanvas, width, height);
  ctx.topSource.clearRect(0, 0, width, height);
  if (!frame) {
    setFrameModuleEmpty(els.topSourceEmpty, 'source manifest has no frames', true);
    if (els.topSourceMeta) els.topSourceMeta.textContent = 'waiting for source';
    if (els.topSourceFile) els.topSourceFile.textContent = '-';
    return;
  }
  setFrameModuleEmpty(els.topSourceEmpty, 'loading source', true);
  if (els.topSourceMeta) els.topSourceMeta.textContent = `${width} x ${height} | ${state.source.mode}`;
  if (els.topSourceFile) els.topSourceFile.textContent = cleanFileRef(framePathFromEntry(frame));
  try {
    const image = await loadSourceFrameImage(frame);
    if (requestId !== state.source.imageRequestId) return;
    ctx.topSource.clearRect(0, 0, width, height);
    ctx.topSource.drawImage(image, 0, 0, width, height);
    setFrameModuleEmpty(els.topSourceEmpty, '', false);
  } catch (error) {
    if (requestId !== state.source.imageRequestId) return;
    console.error(error);
    setFrameModuleEmpty(els.topSourceEmpty, error.message, true);
  }
}

function colorMaskLayers() {
  const layers = Array.isArray(state.colorMasks.manifest?.layers) ? state.colorMasks.manifest.layers : [];
  return layers
    .map((layer, index) => ({
      prefix: String(layer?.prefix || `bit-${layer?.bit ?? index}`),
      name: String(layer?.name || layer?.prefix || `bit ${layer?.bit ?? index}`),
      bit: Math.max(0, Math.min(7, Math.round(Number(layer?.bit ?? index) || 0)))
    }))
    .sort((a, b) => a.bit - b.bit);
}

function colorForMaskLayer(layer) {
  const prefix = String(layer?.prefix || '');
  return MASK_LAYER_COLORS[prefix] || MASK_BIT_FALLBACK_COLORS[Math.max(0, Math.min(7, Number(layer?.bit) || 0))] || [255, 255, 255];
}

function buildMaskPalette(layers, options = {}) {
  const emptyAlpha = Number.isFinite(Number(options.emptyAlpha)) ? Math.max(0, Math.min(255, Math.round(Number(options.emptyAlpha)))) : 255;
  const selectedAlpha = Number.isFinite(Number(options.selectedAlpha)) ? Math.max(0, Math.min(255, Math.round(Number(options.selectedAlpha)))) : 255;
  const byBit = new Map();
  for (const layer of layers) byBit.set(Number(layer.bit), colorForMaskLayer(layer));
  const palette = new Uint8ClampedArray(256 * 4);
  for (let value = 0; value < 256; value++) {
    const offset = value * 4;
    if (!value) {
      palette[offset] = 3;
      palette[offset + 1] = 5;
      palette[offset + 2] = 7;
      palette[offset + 3] = emptyAlpha;
      continue;
    }
    let red = 0;
    let green = 0;
    let blue = 0;
    let count = 0;
    for (let bit = 0; bit < 8; bit++) {
      if (!(value & (1 << bit))) continue;
      const color = byBit.get(bit) || MASK_BIT_FALLBACK_COLORS[bit] || [255, 255, 255];
      red += color[0];
      green += color[1];
      blue += color[2];
      count += 1;
    }
    palette[offset] = Math.round(red / Math.max(1, count));
    palette[offset + 1] = Math.round(green / Math.max(1, count));
    palette[offset + 2] = Math.round(blue / Math.max(1, count));
    palette[offset + 3] = selectedAlpha;
  }
  return palette;
}

function maskBitsImageData(targetCtx, maskBits, width, height, layers, options = {}) {
  const imageData = targetCtx.createImageData(width, height);
  const data = imageData.data;
  const palette = buildMaskPalette(layers, options);
  for (let index = 0; index < maskBits.length; index++) {
    const source = maskBits[index] * 4;
    const target = index * 4;
    data[target] = palette[source];
    data[target + 1] = palette[source + 1];
    data[target + 2] = palette[source + 2];
    data[target + 3] = palette[source + 3];
  }
  return imageData;
}

function drawMaskBits(maskBits, width, height, layers) {
  if (!ctx.topMask || !els.topMaskCanvas) return;
  updateTopCanvasSize(els.topMaskCanvas, width, height);
  const imageData = maskBitsImageData(ctx.topMask, maskBits, width, height, layers);
  ctx.topMask.putImageData(imageData, 0, 0);
}

function maskFrameEntry(frameIndex = selectedSourceFrameIndex()) {
  const manifest = state.colorMasks.manifest;
  const frames = Array.isArray(manifest?.frames) ? manifest.frames : [];
  if (!frames.length) return null;
  const current = sourceFrameEntry(frameIndex) || (canSyncReviewFrame(frameIndex) ? currentFrameEntry() : null);
  const sourceKey = pathMatchKey(framePathFromEntry(current));
  if (sourceKey && sourceKey !== '-') {
    const byPath = frames.find((entry) => pathMatchKey(entry?.source || entry?.sourcePath || entry?.framePath || '') === sourceKey);
    if (byPath) return byPath;
  }
  const canUseOrdinalIdentity = maskManifestCanIndexSource();
  const frameId = current?.frameId ?? current?.rawIndex ?? null;
  if (canUseOrdinalIdentity && frameId !== null && frameId !== undefined) {
    const byId = frames.find((entry) => String(entry?.frameId) === String(frameId));
    if (byId) return byId;
  }
  if (canUseOrdinalIdentity) {
    const direct = frames.find((entry) => Number(entry?.index) === Number(frameIndex) || Number(entry?.frameOrdinal) === Number(frameIndex));
    if (direct) return direct;
  }
  return null;
}

function trimMaskCache(limit = 32) {
  while (state.colorMasks.cache.size > limit) {
    state.colorMasks.cache.delete(state.colorMasks.cache.keys().next().value);
  }
}

async function loadMaskBits(entry) {
  const url = appScopedAssetUrl(entry?.maskBits || entry?.path || '');
  if (!url) throw new Error('mask frame has no file reference');
  const version = frameVersionToken(entry);
  const cacheKey = `${url}|${version}`;
  if (state.colorMasks.cache.has(cacheKey)) return state.colorMasks.cache.get(cacheKey);
  const promise = fetch(versionedResourceUrl(url, version), { cache: state.source.mode === 'live' ? 'no-store' : 'force-cache' })
    .then(async (response) => {
      if (!response.ok) throw new Error(`failed to load mask frame ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    })
    .catch((error) => {
      state.colorMasks.cache.delete(cacheKey);
      throw error;
    });
  state.colorMasks.cache.set(cacheKey, promise);
  trimMaskCache();
  return promise;
}

function maskMetaText(entry) {
  if (!entry) return 'not loaded';
  const selected = Number(entry.selectedPixels || 0).toLocaleString('en-US');
  const overlap = Number(entry.overlapPixels || 0).toLocaleString('en-US');
  const counts = entry.layerPixelCounts && typeof entry.layerPixelCounts === 'object'
    ? Object.entries(entry.layerPixelCounts).map(([prefix, count]) => `${prefix} ${Number(count || 0).toLocaleString('en-US')}`).join(' | ')
    : '';
  return counts ? `${selected} px | overlap ${overlap} | ${counts}` : `${selected} px | overlap ${overlap}`;
}

function clearTopMask(text) {
  if (ctx.topMask && els.topMaskCanvas) {
    const width = els.topMaskCanvas.width || 640;
    const height = els.topMaskCanvas.height || 360;
    ctx.topMask.clearRect(0, 0, width, height);
  }
  setFrameModuleEmpty(els.topMaskEmpty, text, true);
  if (els.topMaskMeta) els.topMaskMeta.textContent = text;
  if (els.topMaskFile) els.topMaskFile.textContent = '-';
}

async function renderTopMaskFrame() {
  if (!ctx.topMask || !els.topMaskCanvas) return;
  const requestId = ++state.colorMasks.requestId;
  if (state.colorMasks.loadError) {
    clearTopMask(state.colorMasks.loadError);
    return;
  }
  if (!state.colorMasks.manifest) {
    clearTopMask('mask manifest not found');
    return;
  }
  const entry = maskFrameEntry();
  if (!entry) {
    clearTopMask('mask not produced for this source frame');
    return;
  }
  const sourceFrame = sourceFrameEntry();
  const width = Number(entry.width || state.colorMasks.manifest.width || sourceFrame?.width || state.source.manifest?.width || state.precompute?.width || 640);
  const height = Number(entry.height || state.colorMasks.manifest.height || sourceFrame?.height || state.source.manifest?.height || state.precompute?.height || 360);
  updateTopCanvasSize(els.topMaskCanvas, width, height);
  setFrameModuleEmpty(els.topMaskEmpty, 'loading mask', true);
  if (els.topMaskMeta) els.topMaskMeta.textContent = maskMetaText(entry);
  if (els.topMaskFile) els.topMaskFile.textContent = cleanFileRef(entry.maskBits || entry.path);
  try {
    const maskBits = await loadMaskBits(entry);
    if (requestId !== state.colorMasks.requestId) return;
    if (maskBits.length !== width * height) {
      throw new Error(`mask size ${maskBits.length.toLocaleString('en-US')} does not match ${width} x ${height}`);
    }
    drawMaskBits(maskBits, width, height, colorMaskLayers());
    setFrameModuleEmpty(els.topMaskEmpty, '', false);
  } catch (error) {
    if (requestId !== state.colorMasks.requestId) return;
    console.error(error);
    clearTopMask(error.message);
    if (els.topMaskFile) els.topMaskFile.textContent = cleanFileRef(entry.maskBits || entry.path);
  }
}

function clearTopBbox(text) {
  if (ctx.topBbox && els.topBboxCanvas) {
    const width = els.topBboxCanvas.width || 640;
    const height = els.topBboxCanvas.height || 360;
    ctx.topBbox.clearRect(0, 0, width, height);
  }
  setFrameModuleEmpty(els.topBboxEmpty, text, true);
  if (els.topBboxMeta) els.topBboxMeta.textContent = text;
  if (els.topBboxFile) els.topBboxFile.textContent = '-';
}

function drawMaskBitsOverlay(targetCtx, maskBits, width, height, layers, selectedAlpha = 150) {
  const maskCanvas = document.createElement('canvas');
  maskCanvas.width = width;
  maskCanvas.height = height;
  const maskCtx = maskCanvas.getContext('2d');
  if (!maskCtx) return;
  const imageData = maskBitsImageData(maskCtx, maskBits, width, height, layers, {
    emptyAlpha: 0,
    selectedAlpha
  });
  maskCtx.putImageData(imageData, 0, 0);
  targetCtx.drawImage(maskCanvas, 0, 0, width, height);
}

function topBboxSourceList(settings, frameIndex) {
  const info = activeBboxInfo();
  const maskbitsInfo = activeMaskbitsBboxInfo();
  const baselineReady = info && info.available && !info.stale && state.bbox.manifest;
  const maskbitsReady = maskbitsInfo && maskbitsInfo.available && !maskbitsInfo.stale && state.maskbitsBbox.manifest;
  const baselineFrame = baselineReady ? bboxFrameEntry(frameIndex) : null;
  const maskbitsFrame = maskbitsReady ? maskbitsBboxFrameEntry(frameIndex) : null;
  const baselineBboxes = Array.isArray(baselineFrame?.bboxes) ? baselineFrame.bboxes : [];
  const maskbitsBboxes = Array.isArray(maskbitsFrame?.bboxes) ? maskbitsFrame.bboxes : [];
  if (settings.viewMode === 'maskbits') {
    return [{ name: 'maskbits', suffix: 'm', color: '#ff5fe6', bboxes: maskbitsBboxes, ready: Boolean(maskbitsReady) }];
  }
  if (settings.viewMode === 'compare') {
    return [
      { name: 'baseline', suffix: '', color: '#55f7ff', bboxes: baselineBboxes, ready: Boolean(baselineReady) },
      { name: 'maskbits', suffix: 'm', color: '#ff5fe6', bboxes: maskbitsBboxes, ready: Boolean(maskbitsReady) }
    ];
  }
  return [{ name: 'baseline', suffix: '', color: '#55f7ff', bboxes: baselineBboxes, ready: Boolean(baselineReady) }];
}

function drawTopBboxList(targetCtx, bboxes, settings, options = {}) {
  const color = options.color || '#55f7ff';
  const suffix = options.suffix || '';
  const maxLabels = Number(options.maxLabels ?? 10);
  targetCtx.save();
  targetCtx.lineJoin = 'miter';
  targetCtx.lineCap = 'butt';
  targetCtx.strokeStyle = color;
  targetCtx.fillStyle = color;
  targetCtx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  targetCtx.textBaseline = 'alphabetic';
  targetCtx.shadowColor = 'rgba(0,0,0,.9)';
  targetCtx.shadowBlur = 2;
  for (let index = 0; index < bboxes.length; index++) {
    const bbox = bboxes[index];
    const rectValues = Array.isArray(bbox.bboxPx) ? bbox.bboxPx.map(Number) : null;
    if (settings.showRawBboxes && rectValues && rectValues.length === 4) {
      const [x0, y0, x1, y1] = rectValues;
      targetCtx.globalAlpha = 0.95;
      targetCtx.lineWidth = 1;
      targetCtx.strokeRect(x0 + 0.5, y0 + 0.5, Math.max(0, x1 - x0), Math.max(0, y1 - y0));
    }
    const quadFit = bbox.quadFit && typeof bbox.quadFit === 'object' ? bbox.quadFit : null;
    const quadPoints = validPointList(quadFit?.pointsPx) ? quadFit.pointsPx : null;
    if (quadPoints) {
      targetCtx.beginPath();
      quadPoints.forEach((point, pointIndex) => {
        const x = Number(point[0] ?? point.x);
        const y = Number(point[1] ?? point.y);
        if (pointIndex === 0) targetCtx.moveTo(x, y);
        else targetCtx.lineTo(x, y);
      });
      targetCtx.closePath();
      targetCtx.globalAlpha = Math.max(0.08, Number(settings.quadOverlayOpacity) || 0);
      targetCtx.lineWidth = Math.max(1, Number(settings.quadThicknessPx) || 1);
      targetCtx.stroke();
    }
    if (index < maxLabels && rectValues && rectValues.length === 4) {
      const [x0, y0] = rectValues;
      const marker = quadFit?.hasVoidOverlap ? ' +void' : quadFit?.accepted ? ' quad' : quadFit?.enabled ? ' weak' : '';
      targetCtx.globalAlpha = 1;
      targetCtx.fillText(`${suffix}${index + 1} ${Number(bbox.pixelCount || 0).toLocaleString('en-US')}px${marker}`, x0 + 2, Math.max(11, y0 - 3));
    }
  }
  targetCtx.restore();
}

function bboxParityText() {
  const comparison = state.maskbitsBbox.manifest?.comparison || state.maskbitsBbox.manifest?.summary?.comparison || {};
  if (!comparison.available) return '';
  if (comparison.passed) {
    return ` | parity pass ${Number(comparison.framesCompared || 0).toLocaleString('en-US')}f`;
  }
  return ` | parity diff sel ${Number(comparison.selectedPixelMismatches || 0).toLocaleString('en-US')} bbox ${Number(comparison.bboxCountMismatches || 0).toLocaleString('en-US')} geom ${Number(comparison.geometryMismatches || 0).toLocaleString('en-US')}`;
}

function bboxTimingText(timing) {
  if (!timing) return 'timing pending';
  return `total ${fmt(timing.total, 1)}ms | src ${fmt(timing.source, 1)} | mask ${fmt(timing.maskLoad + timing.maskDraw, 1)} | bbox ${fmt(timing.bboxLookup, 1)} | draw ${fmt(timing.bboxDraw, 1)}`;
}

async function renderTopBboxFrame() {
  if (!ctx.topBbox || !els.topBboxCanvas) return;
  const frameIndex = selectedSourceFrameIndex();
  const frame = sourceFrameEntry(frameIndex);
  const { width, height } = sourceFrameDimensions(frame);
  const requestId = ++state.topBbox.requestId;
  updateTopCanvasSize(els.topBboxCanvas, width, height);
  ctx.topBbox.clearRect(0, 0, width, height);
  if (!frame) {
    clearTopBbox('source manifest has no frames');
    return;
  }
  setFrameModuleEmpty(els.topBboxEmpty, 'loading bbox', true);
  const timingStart = performance.now();
  let sourceMs = 0;
  let maskLoadMs = 0;
  let maskDrawMs = 0;
  let bboxLookupMs = 0;
  let bboxDrawMs = 0;
  let maskStatus = 'no mask';
  let bboxStatus = 'no bbox';
  try {
    const sourceStart = performance.now();
    const image = await loadSourceFrameImage(frame);
    sourceMs = performance.now() - sourceStart;
    if (requestId !== state.topBbox.requestId) return;
    ctx.topBbox.clearRect(0, 0, width, height);
    ctx.topBbox.drawImage(image, 0, 0, width, height);

    const maskEntry = maskFrameEntry(frameIndex);
    if (maskEntry) {
      const maskWidth = Number(maskEntry.width || state.colorMasks.manifest?.width || width);
      const maskHeight = Number(maskEntry.height || state.colorMasks.manifest?.height || height);
      const maskStart = performance.now();
      const maskBits = await loadMaskBits(maskEntry);
      maskLoadMs = performance.now() - maskStart;
      if (requestId !== state.topBbox.requestId) return;
      if (maskBits.length === maskWidth * maskHeight && maskWidth === width && maskHeight === height) {
        const drawStart = performance.now();
        drawMaskBitsOverlay(ctx.topBbox, maskBits, width, height, colorMaskLayers());
        maskDrawMs = performance.now() - drawStart;
        maskStatus = 'mask';
      } else {
        maskStatus = 'mask size mismatch';
      }
    }

    const settings = bboxFlowState();
    const lookupStart = performance.now();
    const sources = topBboxSourceList(settings, frameIndex);
    bboxLookupMs = performance.now() - lookupStart;
    const drawStart = performance.now();
    for (const source of sources) {
      if (source.ready && source.bboxes.length) {
        drawTopBboxList(ctx.topBbox, source.bboxes, settings, {
          color: source.color,
          suffix: source.suffix,
          maxLabels: settings.viewMode === 'compare' ? 6 : 10
        });
      }
    }
    bboxDrawMs = performance.now() - drawStart;
    const readySources = sources.filter((source) => source.ready);
    const totalBoxes = readySources.reduce((sum, source) => sum + source.bboxes.length, 0);
    bboxStatus = readySources.length ? `${totalBoxes.toLocaleString('en-US')} boxes` : `${settings.viewMode} not ready`;
    state.topBbox.latency = {
      source: sourceMs,
      maskLoad: maskLoadMs,
      maskDraw: maskDrawMs,
      bboxLookup: bboxLookupMs,
      bboxDraw: bboxDrawMs,
      total: performance.now() - timingStart
    };
    if (els.topBboxMeta) {
      els.topBboxMeta.textContent = `${bboxStatus} | ${bboxTimingText(state.topBbox.latency)}${bboxParityText()}`;
    }
    if (els.topBboxFile) {
      els.topBboxFile.textContent = `${cleanFileRef(framePathFromEntry(frame), 56)} | ${maskStatus} | ${settings.viewMode}`;
    }
    setFrameModuleEmpty(els.topBboxEmpty, '', false);
  } catch (error) {
    if (requestId !== state.topBbox.requestId) return;
    console.error(error);
    clearTopBbox(error.message);
    if (els.topBboxFile) els.topBboxFile.textContent = cleanFileRef(framePathFromEntry(frame), 72);
  }
}

function scheduleTopBboxRender() {
  if (state.topBbox.renderRaf) return;
  state.topBbox.renderRaf = requestAnimationFrame(() => {
    state.topBbox.renderRaf = 0;
    renderTopBboxFrame().catch((error) => {
      console.error(error);
      clearTopBbox(error.message);
    });
  });
}

function clearTopClip(text) {
  if (ctx.topClip && els.topClipCanvas) {
    const width = els.topClipCanvas.width || 640;
    const height = els.topClipCanvas.height || 360;
    ctx.topClip.clearRect(0, 0, width, height);
  }
  setFrameModuleEmpty(els.topClipEmpty, text, true);
  if (els.topClipMeta) els.topClipMeta.textContent = text;
  if (els.topClipFile) els.topClipFile.textContent = '-';
}

function clearTopContour(text) {
  if (ctx.topContour && els.topContourCanvas) {
    const width = els.topContourCanvas.width || 640;
    const height = els.topContourCanvas.height || 360;
    ctx.topContour.clearRect(0, 0, width, height);
  }
  setFrameModuleEmpty(els.topContourEmpty, text, true);
  if (els.topContourMeta) els.topContourMeta.textContent = text;
  if (els.topContourFile) els.topContourFile.textContent = '-';
}

function clearTopPose(text) {
  if (ctx.topPose && els.topPoseCanvas) {
    const width = els.topPoseCanvas.width || 640;
    const height = els.topPoseCanvas.height || 360;
    ctx.topPose.clearRect(0, 0, width, height);
  }
  setFrameModuleEmpty(els.topPoseEmpty, text, true);
  if (els.topPoseMeta) els.topPoseMeta.textContent = text;
  if (els.topPoseFile) els.topPoseFile.textContent = '-';
}

function clearTopInstance(text) {
  if (ctx.topInstance && els.topInstanceCanvas) {
    const width = els.topInstanceCanvas.width || 640;
    const height = els.topInstanceCanvas.height || 360;
    ctx.topInstance.clearRect(0, 0, width, height);
  }
  setFrameModuleEmpty(els.topInstanceEmpty, text, true);
  if (els.topInstanceMeta) els.topInstanceMeta.textContent = text;
  if (els.topInstanceFile) els.topInstanceFile.textContent = '-';
}

async function drawTopReviewBase(targetCtx, frame, frameIndex, width, height, isCurrent, maskAlpha = 90) {
  const image = await loadSourceFrameImage(frame);
  if (!isCurrent()) return { ok: false, maskStatus: 'cancelled' };
  targetCtx.clearRect(0, 0, width, height);
  targetCtx.drawImage(image, 0, 0, width, height);
  const maskEntry = maskFrameEntry(frameIndex);
  if (!maskEntry) return { ok: true, maskStatus: 'no mask' };
  const maskWidth = Number(maskEntry.width || state.colorMasks.manifest?.width || width);
  const maskHeight = Number(maskEntry.height || state.colorMasks.manifest?.height || height);
  const maskBits = await loadMaskBits(maskEntry);
  if (!isCurrent()) return { ok: false, maskStatus: 'cancelled' };
  if (maskBits.length !== maskWidth * maskHeight || maskWidth !== width || maskHeight !== height) {
    return { ok: true, maskStatus: 'mask size mismatch' };
  }
  drawMaskBitsOverlay(targetCtx, maskBits, width, height, colorMaskLayers(), maskAlpha);
  return { ok: true, maskStatus: 'mask' };
}

function drawTopFovClipObject(targetCtx, object, settings) {
  const rectValues = Array.isArray(object?.bboxPx) ? object.bboxPx.map(Number) : null;
  const fov = object?.fovClip && typeof object.fovClip === 'object' ? object.fovClip : null;
  if (!rectValues || rectValues.length !== 4 || !fov) return null;
  const clippedSides = Array.isArray(fov.sides) ? fov.sides : [];
  const nearSides = Array.isArray(fov.nearSides) ? fov.nearSides : [];
  const sides = clippedSides.length ? clippedSides : nearSides;
  if (!sides.length) return null;
  const [x0, y0, x1, y1] = rectValues;
  targetCtx.save();
  targetCtx.lineJoin = 'miter';
  targetCtx.lineCap = 'butt';
  targetCtx.globalAlpha = 0.8;
  targetCtx.strokeStyle = clippedSides.length ? '#ff4f45' : '#ffcc33';
  targetCtx.lineWidth = 1;
  targetCtx.strokeRect(x0 + 0.5, y0 + 0.5, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
  targetCtx.lineWidth = Math.max(2, Number(settings.fovClip?.marginPx || 0) > 0 ? 3 : 2);
  const drawSide = (side, color) => {
    const line = fovSideLine(rectValues, side);
    targetCtx.strokeStyle = color;
    targetCtx.beginPath();
    targetCtx.moveTo(line[0], line[1]);
    targetCtx.lineTo(line[2], line[3]);
    targetCtx.stroke();
  };
  for (const side of clippedSides) drawSide(side, '#ff453a');
  for (const side of nearSides) {
    if (!clippedSides.includes(side)) drawSide(side, '#ffd43b');
  }
  const severity = Math.round(Number(fov.severity || 0) * 100);
  const sideText = sides.map((side) => String(side)[0]?.toUpperCase() || '').join('/');
  targetCtx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  targetCtx.textBaseline = 'top';
  targetCtx.fillStyle = clippedSides.length ? '#ff837a' : '#ffe082';
  targetCtx.shadowColor = 'rgba(0,0,0,.9)';
  targetCtx.shadowBlur = 2;
  targetCtx.fillText(`${clippedSides.length ? 'CLIP' : 'EDGE'} ${sideText} ${severity}%`, x0 + 3, Math.max(2, y0 + 3));
  targetCtx.restore();
  return clippedSides.length ? 'clipped' : 'near';
}

function drawTopContourFeature(targetCtx, feature, color, settings) {
  const points = contourDisplayPoints(feature);
  if (!validContourPointList(points)) return false;
  targetCtx.save();
  targetCtx.globalAlpha = Math.max(0.08, Number(settings.opacity) || 0.95);
  targetCtx.strokeStyle = color;
  targetCtx.lineWidth = Math.max(1, Number(settings.lineThicknessPx) || 1);
  targetCtx.lineJoin = 'round';
  targetCtx.beginPath();
  points.forEach((point, index) => {
    const x = Number(point[0]);
    const y = Number(point[1]);
    if (index === 0) targetCtx.moveTo(x + 0.5, y + 0.5);
    else targetCtx.lineTo(x + 0.5, y + 0.5);
  });
  targetCtx.closePath();
  targetCtx.stroke();
  targetCtx.restore();
  return true;
}

async function renderTopClipFrame() {
  if (!ctx.topClip || !els.topClipCanvas) return;
  const frameIndex = selectedSourceFrameIndex();
  const frame = sourceFrameEntry(frameIndex);
  const { width, height } = sourceFrameDimensions(frame);
  const requestId = ++state.bboxClipping.requestId;
  updateTopCanvasSize(els.topClipCanvas, width, height);
  ctx.topClip.clearRect(0, 0, width, height);
  if (!frame) {
    clearTopClip('source manifest has no frames');
    return;
  }
  setFrameModuleEmpty(els.topClipEmpty, 'loading clipping', true);
  try {
    const base = await drawTopReviewBase(ctx.topClip, frame, frameIndex, width, height, () => requestId === state.bboxClipping.requestId, 75);
    if (requestId !== state.bboxClipping.requestId || !base.ok) return;
    const info = activeBboxClippingInfo();
    if (!info || !info.available || info.stale || !state.bboxClipping.manifest) {
      const text = info?.stale ? `clipping stale: ${(info.staleReasons || []).join(', ') || 'rebuild needed'}` : 'bbox clipping not built';
      if (els.topClipMeta) els.topClipMeta.textContent = text;
      if (els.topClipFile) els.topClipFile.textContent = cleanFileRef(framePathFromEntry(frame), 56);
      setFrameModuleEmpty(els.topClipEmpty, text, true);
      return;
    }
    const settings = bboxFlowState();
    const clip = settings.fovClip || {};
    if (!clip.enabled || !clip.showOverlay) {
      if (els.topClipMeta) els.topClipMeta.textContent = 'clipping overlay hidden';
      if (els.topClipFile) els.topClipFile.textContent = `${cleanFileRef(framePathFromEntry(frame), 56)} | ${base.maskStatus}`;
      setFrameModuleEmpty(els.topClipEmpty, '', false);
      return;
    }
    const clipFrame = bboxClippingFrameEntry(frameIndex);
    const objects = Array.isArray(clipFrame?.objects) ? clipFrame.objects : [];
    let clipped = 0;
    let near = 0;
    for (const object of objects) {
      const status = drawTopFovClipObject(ctx.topClip, object, settings);
      if (status === 'clipped') clipped++;
      if (status === 'near') near++;
    }
    if (els.topClipMeta) els.topClipMeta.textContent = `${objects.length} objects | ${clipped} clipped | ${near} near`;
    if (els.topClipFile) els.topClipFile.textContent = `${cleanFileRef(framePathFromEntry(frame), 56)} | ${base.maskStatus}`;
    setFrameModuleEmpty(els.topClipEmpty, '', false);
  } catch (error) {
    if (requestId !== state.bboxClipping.requestId) return;
    console.error(error);
    clearTopClip(error.message);
  }
}

async function renderTopContourFrame() {
  if (!ctx.topContour || !els.topContourCanvas) return;
  const frameIndex = selectedSourceFrameIndex();
  const frame = sourceFrameEntry(frameIndex);
  const { width, height } = sourceFrameDimensions(frame);
  const requestId = ++state.bboxContours.requestId;
  updateTopCanvasSize(els.topContourCanvas, width, height);
  ctx.topContour.clearRect(0, 0, width, height);
  if (!frame) {
    clearTopContour('source manifest has no frames');
    return;
  }
  setFrameModuleEmpty(els.topContourEmpty, 'loading contours', true);
  try {
    const base = await drawTopReviewBase(ctx.topContour, frame, frameIndex, width, height, () => requestId === state.bboxContours.requestId, 55);
    if (requestId !== state.bboxContours.requestId || !base.ok) return;
    const settings = contourHierarchyState();
    const payload = await liveBboxContourFrame(frameIndex, contourAnalysisSettings());
    if (requestId !== state.bboxContours.requestId) return;
    const contourFrame = payload?.frame || {};
    const objects = Array.isArray(contourFrame?.objects) ? contourFrame.objects : [];
    const topSettings = { ...settings, opacity: 0.8 };
    let drawn = 0;
    let voids = 0;
    for (const object of objects) {
      if (drawTopContourFeature(ctx.topContour, object.outer, '#55f7ff', topSettings)) drawn++;
      for (const outer of Array.isArray(object.additionalOuters) ? object.additionalOuters : []) {
        if (drawTopContourFeature(ctx.topContour, outer, '#4aa3ff', topSettings)) drawn++;
      }
      for (const item of Array.isArray(object.voids) ? object.voids : []) {
        if (drawTopContourFeature(ctx.topContour, item, '#ff7ad9', topSettings)) drawn++;
        voids++;
      }
      for (const item of Array.isArray(object.nestedIslands) ? object.nestedIslands : []) {
        if (drawTopContourFeature(ctx.topContour, item, '#63ff95', topSettings)) drawn++;
      }
    }
    const timing = Number(payload?.timingMs);
    const timingText = Number.isFinite(timing) ? ` | ${fmt(timing, 1)}ms live` : ' | live';
    if (els.topContourMeta) els.topContourMeta.textContent = `${drawn} paths | ${objects.length} objects | ${voids} voids${timingText}`;
    if (els.topContourFile) els.topContourFile.textContent = `${cleanFileRef(framePathFromEntry(frame), 56)} | ${base.maskStatus} | inference`;
    setFrameModuleEmpty(els.topContourEmpty, '', false);
  } catch (error) {
    if (requestId !== state.bboxContours.requestId) return;
    console.error(error);
    clearTopContour(error.message);
    if (els.topContourFile) els.topContourFile.textContent = cleanFileRef(framePathFromEntry(frame), 72);
  }
}

function drawTopPoseObject(targetCtx, object) {
  const points = validPointList(object?.refinedCornersPx)
    ? object.refinedCornersPx
    : validPointList(object?.bestPose?.reprojectedCornersPx)
      ? object.bestPose.reprojectedCornersPx
      : validPointList(object?.bestPose?.cornersPx)
        ? object.bestPose.cornersPx
        : null;
  if (!points) return false;
  const valid = poseMeasurementValid(object);
  targetCtx.save();
  targetCtx.globalAlpha = valid ? 0.95 : 0.45;
  targetCtx.lineWidth = valid ? 2 : 1.5;
  targetCtx.lineJoin = 'miter';
  targetCtx.strokeStyle = valid ? '#ffdc4a' : '#ff6b5f';
  targetCtx.fillStyle = valid ? 'rgba(255, 220, 74, 0.10)' : 'rgba(255, 107, 95, 0.08)';
  targetCtx.beginPath();
  points.forEach((point, index) => {
    const x = Number(point[0]) + 0.5;
    const y = Number(point[1]) + 0.5;
    if (index === 0) targetCtx.moveTo(x, y);
    else targetCtx.lineTo(x, y);
  });
  targetCtx.closePath();
  targetCtx.fill();
  targetCtx.stroke();
  const center = points.reduce((acc, point) => {
    acc.x += Number(point[0]);
    acc.y += Number(point[1]);
    return acc;
  }, { x: 0, y: 0 });
  center.x /= points.length;
  center.y /= points.length;
  targetCtx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  targetCtx.textBaseline = 'middle';
  targetCtx.fillStyle = valid ? '#fff2a8' : '#ffb2aa';
  targetCtx.shadowColor = 'rgba(0,0,0,.9)';
  targetCtx.shadowBlur = 2;
  const pose = object?.bestPose || {};
  targetCtx.fillText(`${shortPoseId(object, 0)} ${pose.depthM ? `${fmt(pose.depthM, 1)}m` : 'no pose'}`, center.x + 4, center.y);
  targetCtx.restore();
  return true;
}

async function renderTopPoseFrame() {
  if (!ctx.topPose || !els.topPoseCanvas) return;
  const frameIndex = selectedSourceFrameIndex();
  const frame = sourceFrameEntry(frameIndex);
  const { width, height } = sourceFrameDimensions(frame);
  const requestId = ++state.topPose.requestId;
  updateTopCanvasSize(els.topPoseCanvas, width, height);
  ctx.topPose.clearRect(0, 0, width, height);
  if (!frame) {
    clearTopPose('source manifest has no frames');
    return;
  }
  setFrameModuleEmpty(els.topPoseEmpty, 'loading pose', true);
  try {
    const base = await drawTopReviewBase(ctx.topPose, frame, frameIndex, width, height, () => requestId === state.topPose.requestId, 45);
    if (requestId !== state.topPose.requestId || !base.ok) return;
    const info = activeSquarePoseInfo();
    if (!info || !info.available || info.stale || !state.squarePose.manifest) {
      const text = info?.stale ? `pose stale: ${(info.staleReasons || []).join(', ') || 'rebuild needed'}` : 'pose not built';
      if (els.topPoseMeta) els.topPoseMeta.textContent = text;
      if (els.topPoseFile) els.topPoseFile.textContent = cleanFileRef(framePathFromEntry(frame), 56);
      setFrameModuleEmpty(els.topPoseEmpty, text, true);
      return;
    }
    const poseFrame = viewerPoseFrameEntry(frameIndex);
    const objects = Array.isArray(poseFrame?.objects) ? poseFrame.objects : [];
    let drawn = 0;
    let valid = 0;
    for (const object of objects) {
      if (drawTopPoseObject(ctx.topPose, object)) drawn++;
      if (object?.bestPose && poseMeasurementValid(object)) valid++;
    }
    if (els.topPoseMeta) els.topPoseMeta.textContent = `${valid}/${objects.length} valid poses | ${drawn} quads`;
    if (els.topPoseFile) els.topPoseFile.textContent = `${cleanFileRef(framePathFromEntry(frame), 56)} | ${base.maskStatus}`;
    setFrameModuleEmpty(els.topPoseEmpty, '', false);
  } catch (error) {
    if (requestId !== state.topPose.requestId) return;
    console.error(error);
    clearTopPose(error.message);
  }
}

function drawTopInstanceObservation(targetCtx, observation) {
  const center = instanceObservationCenter(observation);
  if (!center) return false;
  const color = observation.instanceColor || '#55f7ff';
  targetCtx.save();
  targetCtx.globalAlpha = observation?.pose?.available ? 0.95 : 0.65;
  targetCtx.fillStyle = color;
  targetCtx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
  targetCtx.lineWidth = 2;
  targetCtx.beginPath();
  targetCtx.arc(center.x + 0.5, center.y + 0.5, observation.status === 'new' ? 4 : 3.25, 0, Math.PI * 2);
  targetCtx.fill();
  targetCtx.stroke();
  const bbox = Array.isArray(observation.bboxPx) ? observation.bboxPx.map(Number) : null;
  if (bbox && bbox.length === 4) {
    targetCtx.globalAlpha = 0.85;
    targetCtx.strokeStyle = color;
    targetCtx.lineWidth = 1;
    targetCtx.strokeRect(bbox[0] + 0.5, bbox[1] + 0.5, Math.max(1, bbox[2] - bbox[0]), Math.max(1, bbox[3] - bbox[1]));
  }
  targetCtx.font = 'bold 11px ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace';
  targetCtx.textBaseline = 'middle';
  targetCtx.shadowColor = 'rgba(0,0,0,.9)';
  targetCtx.shadowBlur = 2;
  targetCtx.fillStyle = color;
  targetCtx.fillText(instanceDisplayId(observation), center.x + 6, center.y - 6);
  targetCtx.restore();
  return true;
}

async function renderTopInstanceFrame() {
  if (!ctx.topInstance || !els.topInstanceCanvas) return;
  const frameIndex = selectedSourceFrameIndex();
  const frame = sourceFrameEntry(frameIndex);
  const { width, height } = sourceFrameDimensions(frame);
  const requestId = ++state.topInstance.requestId;
  updateTopCanvasSize(els.topInstanceCanvas, width, height);
  ctx.topInstance.clearRect(0, 0, width, height);
  if (!frame) {
    clearTopInstance('source manifest has no frames');
    return;
  }
  setFrameModuleEmpty(els.topInstanceEmpty, 'loading instances', true);
  try {
    const base = await drawTopReviewBase(ctx.topInstance, frame, frameIndex, width, height, () => requestId === state.topInstance.requestId, 40);
    if (requestId !== state.topInstance.requestId || !base.ok) return;
    const info = activeInstanceInfo();
    if (!info || !info.available || info.stale || !state.instances.manifest) {
      const text = info?.stale ? `instances stale: ${(info.staleReasons || []).join(', ') || 'rebuild needed'}` : 'instances not built';
      if (els.topInstanceMeta) els.topInstanceMeta.textContent = text;
      if (els.topInstanceFile) els.topInstanceFile.textContent = cleanFileRef(framePathFromEntry(frame), 56);
      setFrameModuleEmpty(els.topInstanceEmpty, text, true);
      return;
    }
    const instanceFrame = instanceFrameEntry(frameIndex);
    const observations = Array.isArray(instanceFrame?.observations) ? instanceFrame.observations : [];
    let drawn = 0;
    let poseCount = 0;
    for (const observation of observations) {
      if (drawTopInstanceObservation(ctx.topInstance, observation)) drawn++;
      if (observation?.pose?.available) poseCount++;
    }
    if (els.topInstanceMeta) els.topInstanceMeta.textContent = `${drawn} instances | ${poseCount} pose updates`;
    if (els.topInstanceFile) els.topInstanceFile.textContent = `${cleanFileRef(framePathFromEntry(frame), 56)} | ${base.maskStatus}`;
    setFrameModuleEmpty(els.topInstanceEmpty, '', false);
  } catch (error) {
    if (requestId !== state.topInstance.requestId) return;
    console.error(error);
    clearTopInstance(error.message);
  }
}

function scheduleTopClipRender() {
  if (state.bboxClipping.renderRaf) return;
  state.bboxClipping.renderRaf = requestAnimationFrame(() => {
    state.bboxClipping.renderRaf = 0;
    renderTopClipFrame().catch((error) => {
      console.error(error);
      clearTopClip(error.message);
    });
  });
}

function scheduleTopContourRender() {
  if (state.bboxContours.renderRaf) return;
  state.bboxContours.renderRaf = requestAnimationFrame(() => {
    state.bboxContours.renderRaf = 0;
    renderTopContourFrame().catch((error) => {
      console.error(error);
      clearTopContour(error.message);
    });
  });
}

function scheduleTopPoseRender() {
  if (state.topPose.renderRaf) return;
  state.topPose.renderRaf = requestAnimationFrame(() => {
    state.topPose.renderRaf = 0;
    renderTopPoseFrame().catch((error) => {
      console.error(error);
      clearTopPose(error.message);
    });
  });
}

function scheduleTopInstanceRender() {
  if (state.topInstance.renderRaf) return;
  state.topInstance.renderRaf = requestAnimationFrame(() => {
    state.topInstance.renderRaf = 0;
    renderTopInstanceFrame().catch((error) => {
      console.error(error);
      clearTopInstance(error.message);
    });
  });
}

async function loadColorMaskManifest(url = COLOR_MASK_MANIFEST_URL) {
  const manifestUrl = appScopedAssetUrl(url || COLOR_MASK_MANIFEST_URL);
  state.colorMasks.loadError = '';
  const previousUrl = state.colorMasks.manifestUrl;
  const manifest = await fetchOptionalJson(manifestUrl);
  if (!manifest) {
    if (previousUrl !== manifestUrl) {
      state.colorMasks.manifest = null;
      state.colorMasks.cache.clear();
      clearBboxContourFrameCache();
    }
    state.colorMasks.manifestUrl = manifestUrl;
    state.colorMasks.loadError = 'mask manifest not found';
    clearTopMask(state.colorMasks.loadError);
    return null;
  }
  if (previousUrl !== manifestUrl) {
    state.colorMasks.cache.clear();
    clearBboxContourFrameCache();
  }
  state.colorMasks.manifest = manifest;
  state.colorMasks.manifestUrl = manifestUrl;
  if (els.topMaskMeta) {
    const frameCount = Number(manifest.frameCount || manifest.frames?.length || 0).toLocaleString('en-US');
    els.topMaskMeta.textContent = `${frameCount} mask frames`;
  }
  return manifest;
}

function assignColorMaskManifest(manifest, url) {
  const manifestUrl = appScopedAssetUrl(url || '');
  if (state.colorMasks.manifestUrl !== manifestUrl) {
    state.colorMasks.cache.clear();
    clearBboxContourFrameCache();
  }
  state.colorMasks.loadError = '';
  state.colorMasks.manifest = manifest || null;
  state.colorMasks.manifestUrl = manifestUrl;
}

async function renderSourceMonitor() {
  renderSourceControls();
  await renderTopSourceFrame().catch((error) => {
    console.error(error);
    setFrameModuleEmpty(els.topSourceEmpty, error.message, true);
  });
  await renderTopMaskFrame().catch((error) => {
    console.error(error);
    clearTopMask(error.message);
  });
  await renderTopBboxFrame().catch((error) => {
    console.error(error);
    clearTopBbox(error.message);
  });
  await renderTopClipFrame().catch((error) => {
    console.error(error);
    clearTopClip(error.message);
  });
  await renderTopContourFrame().catch((error) => {
    console.error(error);
    clearTopContour(error.message);
  });
  await renderTopPoseFrame().catch((error) => {
    console.error(error);
    clearTopPose(error.message);
  });
  await renderTopInstanceFrame().catch((error) => {
    console.error(error);
    clearTopInstance(error.message);
  });
  scheduleInstance3dRender();
  renderInstanceReadout();
}

function normalizeSourceFrameIndex(index) {
  const count = sourceFrameCount();
  if (!count) return 0;
  return Math.max(0, Math.min(count - 1, Math.round(Number(index) || 0)));
}

async function loadSelectedSourceFrame(index, fromUser = true) {
  state.source.frameIndex = normalizeSourceFrameIndex(index);
  const count = sourceFrameCount();
  if (fromUser) state.source.followLatest = count ? state.source.frameIndex >= count - 1 : true;
  renderSourceControls();
  await renderSourceMonitor();
  if (count) {
    setStatus(`${state.source.mode} source frame ${state.source.frameIndex + 1}/${count} ready`);
  } else {
    setStatus(`${state.source.mode} source waiting for frames`);
  }
}

async function refreshSourceManifests(options = {}) {
  if (state.source.pollInFlight) return;
  state.source.pollInFlight = true;
  const preserveIndex = Boolean(options.preserveIndex);
  const previousIndex = selectedSourceFrameIndex();
  try {
    const sourceUrl = selectedSourceManifestUrl();
    const previousSourceUrl = state.source.manifestUrl;
    let sourceManifest = null;
    let sourceError = '';
    if (sourceUrl) {
      try {
        sourceManifest = await fetchOptionalJson(sourceUrl);
      } catch (error) {
        sourceError = error.message || String(error);
      }
    } else {
      sourceError = 'no source manifest selected';
    }
    if (previousSourceUrl !== sourceUrl) state.source.imageCache.clear();
    if (!sourceError) {
      state.source.manifest = sourceManifest || null;
      state.source.manifestUrl = sourceUrl;
      state.source.pollError = '';
    } else {
      if (previousSourceUrl !== sourceUrl || state.source.mode === 'live') state.source.manifest = null;
      state.source.manifestUrl = sourceUrl;
      state.source.pollError = `source manifest load failed: ${sourceError}`;
    }

    if (sourceManifest && manifestHasMaskBits(sourceManifest)) {
      assignColorMaskManifest(sourceManifest, sourceUrl);
    } else {
      await loadColorMaskManifest(COLOR_MASK_MANIFEST_URL).catch((error) => {
        state.colorMasks.loadError = `mask manifest load failed: ${error.message}`;
        clearTopMask(state.colorMasks.loadError);
      });
    }

    const count = sourceFrameCount();
    if (!count) {
      state.source.frameIndex = 0;
      state.source.followLatest = true;
    } else if (!preserveIndex && state.source.followLatest) {
      state.source.frameIndex = count - 1;
    } else {
      state.source.frameIndex = normalizeSourceFrameIndex(state.source.frameIndex);
      if (state.source.frameIndex >= count - 1) state.source.followLatest = true;
    }
    state.source.lastFrameCount = count;
    renderSourceControls();
    await renderSourceMonitor();

    const maskCount = Number(state.colorMasks.manifest?.frameCount || state.colorMasks.manifest?.frames?.length || 0);
    if (state.source.pollError) {
      setStatus(state.source.pollError);
    } else if (count) {
      const mode = state.source.mode === 'live' ? 'live' : 'library';
      setStatus(`${mode} source ${state.source.frameIndex + 1}/${count} | masks ${maskCount}`);
    }
  } finally {
    state.source.pollInFlight = false;
    if (previousIndex !== selectedSourceFrameIndex()) renderSourceControls();
  }
}

function startSourcePolling() {
  if (state.source.pollTimer) window.clearInterval(state.source.pollTimer);
  state.source.pollTimer = window.setInterval(() => {
    refreshSourceManifests().catch((error) => setStatus(`source refresh failed: ${error.message}`));
  }, SOURCE_POLL_MS);
}

function setSourceMode(mode) {
  const nextMode = mode === 'live' ? 'live' : 'library';
  if (state.source.mode === nextMode) return;
  state.source.mode = nextMode;
  state.source.followLatest = true;
  state.source.frameIndex = Math.max(0, sourceFrameCount() - 1);
  state.source.readiness = null;
  clearBboxContourFrameCache();
  renderSourceControls();
  refreshSourceManifests().catch((error) => setStatus(`source mode failed: ${error.message}`));
  refreshPlaybackReadiness().catch((error) => setStatus(`readiness failed: ${error.message}`));
}

function activeBboxInfo() {
  return state.bbox.discovery?.bbox || null;
}

function activeMaskbitsBboxInfo() {
  return state.maskbitsBbox.discovery?.maskbitsBbox || null;
}

function activeBboxClippingInfo() {
  return state.bboxClipping.discovery?.bboxClipping || null;
}

function activeBboxContoursInfo() {
  return state.bboxContours.discovery?.bboxContours || null;
}

function clearBboxContourFrameCache() {
  state.bboxContours.frameCache?.clear?.();
}

async function liveBboxContourFrame(frameIndex, settings) {
  const cache = state.bboxContours.frameCache;
  const key = jsonCacheKey({
    source: state.source.manifestUrl || selectedSourceManifestUrl(),
    frameIndex,
    settings
  });
  if (cache.has(key)) return cache.get(key);
  const payload = await fetchJson(`${API}/bbox-contours/frame`, {
    method: 'POST',
    body: JSON.stringify({ frameOrdinal: frameIndex, settings })
  });
  cache.set(key, payload);
  if (cache.size > 120) cache.delete(cache.keys().next().value);
  return payload;
}

function activeContourInfo() {
  return state.contour.discovery?.contours || null;
}

function activeCornerInfo() {
  return state.corner.discovery?.corners || null;
}

function activeSquarePoseInfo() {
  return state.squarePose.discovery?.poseEstimation || state.squarePose.discovery?.squarePose || null;
}

function activeInstanceInfo() {
  return state.instances.discovery?.instanceTracking || state.instances.discovery?.instances || null;
}

function bboxFrameEntry(frameIndex = state.frameIndex) {
  const manifest = state.bbox.manifest;
  if (!manifest || !Array.isArray(manifest.frames)) return null;
  const start = Number(manifest.frameStart || 0);
  const offset = frameIndex - start;
  const entry = manifest.frames[offset];
  if (entry && Number(entry.frameOrdinal) === Number(frameIndex)) return entry;
  return manifest.frames.find((candidate) => Number(candidate.frameOrdinal) === Number(frameIndex)) || null;
}

function maskbitsBboxFrameEntry(frameIndex = state.frameIndex) {
  const manifest = state.maskbitsBbox.manifest;
  if (!manifest || !Array.isArray(manifest.frames)) return null;
  const start = Number(manifest.frameStart || 0);
  const offset = frameIndex - start;
  const entry = manifest.frames[offset];
  if (entry && Number(entry.frameOrdinal) === Number(frameIndex)) return entry;
  return manifest.frames.find((candidate) => Number(candidate.frameOrdinal) === Number(frameIndex)) || null;
}

function bboxClippingFrameEntry(frameIndex = state.frameIndex) {
  const manifest = state.bboxClipping.manifest;
  if (!manifest || !Array.isArray(manifest.frames)) return null;
  const start = Number(manifest.frameStart || 0);
  const offset = frameIndex - start;
  const entry = manifest.frames[offset];
  if (entry && Number(entry.frameOrdinal) === Number(frameIndex)) return entry;
  return manifest.frames.find((candidate) => Number(candidate.frameOrdinal) === Number(frameIndex)) || null;
}

function bboxContoursFrameEntry(frameIndex = state.frameIndex) {
  const manifest = state.bboxContours.manifest;
  if (!manifest || !Array.isArray(manifest.frames)) return null;
  const start = Number(manifest.frameStart || 0);
  const offset = frameIndex - start;
  const entry = manifest.frames[offset];
  if (entry && Number(entry.frameOrdinal) === Number(frameIndex)) return entry;
  return manifest.frames.find((candidate) => Number(candidate.frameOrdinal) === Number(frameIndex)) || null;
}

function contourFrameEntry(frameIndex = state.frameIndex) {
  const manifest = state.contour.manifest;
  if (!manifest || !Array.isArray(manifest.frames)) return null;
  const start = Number(manifest.frameStart || 0);
  const offset = frameIndex - start;
  const entry = manifest.frames[offset];
  if (entry && Number(entry.frameOrdinal) === Number(frameIndex)) return entry;
  return manifest.frames.find((candidate) => Number(candidate.frameOrdinal) === Number(frameIndex)) || null;
}

function cornerFrameEntry(frameIndex = state.frameIndex) {
  const manifest = state.corner.manifest;
  if (!manifest || !Array.isArray(manifest.frames)) return null;
  const start = Number(manifest.frameStart || 0);
  const offset = frameIndex - start;
  const entry = manifest.frames[offset];
  if (entry && Number(entry.frameOrdinal) === Number(frameIndex)) return entry;
  return manifest.frames.find((candidate) => Number(candidate.frameOrdinal) === Number(frameIndex)) || null;
}

function squarePoseFrameEntry(frameIndex = state.frameIndex) {
  const manifest = state.squarePose.manifest;
  if (!manifest || !Array.isArray(manifest.frames)) return null;
  const start = Number(manifest.frameStart || 0);
  const offset = frameIndex - start;
  const entry = manifest.frames[offset];
  if (entry && Number(entry.frameOrdinal) === Number(frameIndex)) return entry;
  return manifest.frames.find((candidate) => Number(candidate.frameOrdinal) === Number(frameIndex)) || null;
}

function activeViewerPoseInfo() {
  return activeSquarePoseInfo();
}

function viewerPoseManifest() {
  return state.squarePose.manifest;
}

function viewerPoseFrameEntry(frameIndex = state.frameIndex) {
  return squarePoseFrameEntry(frameIndex);
}

function instanceFrameEntry(frameIndex = state.frameIndex) {
  const manifest = state.instances.manifest;
  if (!manifest || !Array.isArray(manifest.frames)) return null;
  const start = Number(manifest.frameStart || 0);
  const offset = frameIndex - start;
  const entry = manifest.frames[offset];
  if (entry && Number(entry.frameOrdinal) === Number(frameIndex)) return entry;
  return manifest.frames.find((candidate) => Number(candidate.frameOrdinal) === Number(frameIndex)) || null;
}

function scheduleBboxRender() {
  if (state.bbox.renderRaf) return;
  state.bbox.renderRaf = requestAnimationFrame(() => {
    state.bbox.renderRaf = 0;
    renderBboxOverlay();
    scheduleTopBboxRender();
    scheduleTopClipRender();
    scheduleTopContourRender();
    scheduleTopPoseRender();
    scheduleTopInstanceRender();
  });
}

function scheduleSquarePoseRender() {
  if (state.squarePose.renderRaf) return;
  state.squarePose.renderRaf = requestAnimationFrame(() => {
    state.squarePose.renderRaf = 0;
    renderSquarePose();
  });
}

function resizeSquarePoseRenderer() {
  const three = state.squarePose.three;
  if (!three || !els.squarePoseCanvas) return;
  const width = Math.max(1, state.viewport.frameWidth || els.squarePoseCanvas.width || 640);
  const height = Math.max(1, state.viewport.frameHeight || els.squarePoseCanvas.height || 360);
  three.renderer.setSize(width, height, false);
  const aspect = width / Math.max(1, height);
  three.cameraLocked.aspect = aspect;
  three.cameraLocked.updateProjectionMatrix();
  three.cameraOrbit.aspect = aspect;
  three.cameraOrbit.updateProjectionMatrix();
}

function initSquarePoseViewer() {
  if (state.squarePose.three || !els.squarePoseCanvas || !window.THREE) return state.squarePose.three;
  const THREE = window.THREE;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x020406);
  const renderer = new THREE.WebGLRenderer({ canvas: els.squarePoseCanvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  if ('outputColorSpace' in renderer && THREE.SRGBColorSpace) {
    renderer.outputColorSpace = THREE.SRGBColorSpace;
  } else {
    renderer.outputEncoding = THREE.sRGBEncoding;
  }

  const camera = { widthPx: 640, heightPx: 360, fx: 320, fy: 320, cx: 320, cy: 180 };
  const verticalFov = 2 * Math.atan(camera.heightPx / (2 * camera.fy)) * 180 / Math.PI;
  const cameraLocked = new THREE.PerspectiveCamera(verticalFov, camera.widthPx / camera.heightPx, 0.01, 200);
  cameraLocked.position.set(0, 0, 0);
  cameraLocked.lookAt(0, 0, -1);
  const cameraOrbit = new THREE.PerspectiveCamera(45, camera.widthPx / camera.heightPx, 0.01, 200);
  cameraOrbit.position.set(3.6, 2.6, 5.2);
  cameraOrbit.lookAt(0, 0, -2.5);

  const controls = THREE.OrbitControls ? new THREE.OrbitControls(cameraOrbit, renderer.domElement) : null;
  if (controls) {
    controls.target.set(0, 0, -2.5);
    controls.enableDamping = false;
    controls.minDistance = 0.5;
    controls.maxDistance = 80;
    controls.update();
  }

  const textureLoader = new THREE.TextureLoader();
  const material = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.85,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: false,
    toneMapped: false
  });
  textureLoader.load('assets/textures/gate_face_2p7m.png', (texture) => {
    if ('colorSpace' in texture && THREE.SRGBColorSpace) texture.colorSpace = THREE.SRGBColorSpace;
    else texture.encoding = THREE.sRGBEncoding;
    material.map = texture;
    material.needsUpdate = true;
    state.squarePose.textureLoaded = true;
    scheduleSquarePoseRender();
  });

  const plane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), material);
  plane.matrixAutoUpdate = false;
  scene.add(plane);

  const outline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, 0.5, 0.002),
      new THREE.Vector3(0.5, 0.5, 0.002),
      new THREE.Vector3(0.5, -0.5, 0.002),
      new THREE.Vector3(-0.5, -0.5, 0.002)
    ]),
    new THREE.LineBasicMaterial({ color: 0x55f7ff, transparent: true, opacity: 0.85, depthTest: false })
  );
  outline.matrixAutoUpdate = false;
  scene.add(outline);

  const sideLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.5, 0.5, 0.004), new THREE.Vector3(0.5, 0.5, 0.004)]),
    new THREE.LineBasicMaterial({ color: 0xfff02a, transparent: true, opacity: 0.95, depthTest: false })
  );
  sideLine.matrixAutoUpdate = false;
  scene.add(sideLine);

  const frustum = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x7aa2ff, transparent: true, opacity: 0.45, depthTest: false }));
  scene.add(frustum);

  const frameTexture = new THREE.CanvasTexture(els.sourceCanvas);
  if ('colorSpace' in frameTexture && THREE.SRGBColorSpace) frameTexture.colorSpace = THREE.SRGBColorSpace;
  else frameTexture.encoding = THREE.sRGBEncoding;

  state.squarePose.three = {
    THREE,
    scene,
    renderer,
    cameraLocked,
    cameraOrbit,
    controls,
    camera,
    frameTexture,
    plane,
    material,
    outline,
    sideLine,
    frustum,
    lastObject: null
  };
  updateSquarePoseFrustum();
  resizeSquarePoseRenderer();
  return state.squarePose.three;
}

function updateSquarePoseFrustum() {
  const three = state.squarePose.three;
  if (!three) return;
  const { THREE, camera, frustum } = three;
  const depth = 1.35;
  const pixels = [[0, 0], [camera.widthPx, 0], [camera.widthPx, camera.heightPx], [0, camera.heightPx]];
  const corners = pixels.map(([u, v]) => {
    const x = ((u - camera.cx) / camera.fx) * depth;
    const y = ((v - camera.cy) / camera.fy) * depth;
    return new THREE.Vector3(x, -y, -depth);
  });
  const origin = new THREE.Vector3(0, 0, 0);
  const points = [];
  for (const corner of corners) points.push(origin, corner);
  for (let index = 0; index < 4; index++) points.push(corners[index], corners[(index + 1) % 4]);
  frustum.geometry.dispose();
  frustum.geometry = new THREE.BufferGeometry().setFromPoints(points);
}

function rodriguesToMatrix(rvec) {
  const [rx, ry, rz] = (Array.isArray(rvec) ? rvec : [0, 0, 0]).map(Number);
  const theta = Math.hypot(rx, ry, rz);
  if (!Number.isFinite(theta) || theta <= 1e-12) {
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  }
  const x = rx / theta;
  const y = ry / theta;
  const z = rz / theta;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  const t = 1 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c]
  ];
}

function matrixFromCvPose(THREE, pose) {
  const r = rodriguesToMatrix(pose?.rvec || [0, 0, 0]);
  const t = (pose?.tvec || [0, 0, 1]).map(Number);
  const signs = [1, -1, -1];
  const m = new THREE.Matrix4();
  m.set(
    signs[0] * signs[0] * r[0][0], signs[0] * signs[1] * r[0][1], signs[0] * signs[2] * r[0][2], signs[0] * t[0],
    signs[1] * signs[0] * r[1][0], signs[1] * signs[1] * r[1][1], signs[1] * signs[2] * r[1][2], signs[1] * t[1],
    signs[2] * signs[0] * r[2][0], signs[2] * signs[1] * r[2][1], signs[2] * signs[2] * r[2][2], signs[2] * t[2],
    0, 0, 0, 1
  );
  return m;
}

function eulerXyzDegFromMatrix(r) {
  const clampUnit = (value) => Math.max(-1, Math.min(1, Number(value) || 0));
  const pitch = Math.asin(clampUnit(r[0][2]));
  let roll;
  let yaw;
  if (Math.abs(r[0][2]) < 0.9999999) {
    roll = Math.atan2(-r[1][2], r[2][2]);
    yaw = Math.atan2(-r[0][1], r[0][0]);
  } else {
    roll = Math.atan2(r[2][1], r[1][1]);
    yaw = 0;
  }
  const toDeg = 180 / Math.PI;
  return {
    roll: roll * toDeg,
    pitch: pitch * toDeg,
    yaw: yaw * toDeg,
    order: 'XYZ'
  };
}

function poseRpyDeg(pose) {
  return eulerXyzDegFromMatrix(rodriguesToMatrix(pose?.rvec || [0, 0, 0]));
}

function applySquarePoseMatrix(object, matrix, squareSize) {
  const scale = new object.matrix.constructor().makeScale(squareSize, squareSize, 1);
  object.matrix.copy(matrix).multiply(scale);
  object.matrixWorldNeedsUpdate = true;
}

function setSquareSideLine(three, matrix, squareSize, side) {
  const { THREE, sideLine } = three;
  const sideIndex = { top: 0, right: 1, bottom: 2, left: 3 }[side] ?? -1;
  if (sideIndex < 0) {
    sideLine.visible = false;
    return;
  }
  const corners = [
    new THREE.Vector3(-0.5, 0.5, 0.006),
    new THREE.Vector3(0.5, 0.5, 0.006),
    new THREE.Vector3(0.5, -0.5, 0.006),
    new THREE.Vector3(-0.5, -0.5, 0.006)
  ];
  const scale = new THREE.Matrix4().makeScale(squareSize, squareSize, 1);
  const world = matrix.clone().multiply(scale);
  const points = [corners[sideIndex].clone().applyMatrix4(world), corners[(sideIndex + 1) % 4].clone().applyMatrix4(world)];
  sideLine.geometry.dispose();
  sideLine.geometry = new THREE.BufferGeometry().setFromPoints(points);
  sideLine.visible = true;
}

function selectSquarePoseObject(frame) {
  const objects = Array.isArray(frame?.objects) ? frame.objects : [];
  return objects.find((item) => item.accepted && item.bestPose && poseMeasurementValid(item))
    || objects.find((item) => item.bestPose && poseMeasurementValid(item))
    || objects.find((item) => item.bestPose)
    || objects[0]
    || null;
}

function poseMeasurementValid(object) {
  const validity = object?.measurementValidity;
  return !(validity && validity.valid === false);
}

function squarePoseLabelAnchor(object, settings, three) {
  const pose = object?.bestPose;
  if (!pose) return null;
  const width = state.viewport.frameWidth || 640;
  const height = state.viewport.frameHeight || 360;
  if (settings.viewMode === 'camera') {
    const points = validPointList(pose.reprojectedCornersPx) ? pose.reprojectedCornersPx : pose.cornersPx;
    if (!validPointList(points)) return null;
    const total = points.reduce((acc, point) => {
      acc.x += Number(point[0]);
      acc.y += Number(point[1]);
      return acc;
    }, { x: 0, y: 0 });
    return { x: total.x / points.length, y: total.y / points.length, visible: true };
  }
  if (!three?.THREE) return null;
  const { THREE, cameraOrbit, cameraLocked } = three;
  const camera = settings.viewMode === 'orbit' ? cameraOrbit : cameraLocked;
  const matrix = matrixFromCvPose(THREE, pose);
  const center = new THREE.Vector3(0, 0, 0).applyMatrix4(matrix).project(camera);
  if (![center.x, center.y, center.z].every(Number.isFinite)) return null;
  return {
    x: (center.x * 0.5 + 0.5) * width,
    y: (-center.y * 0.5 + 0.5) * height,
    visible: center.z >= -1 && center.z <= 1
  };
}

function shortPoseId(object, index) {
  const raw = String(object?.bboxId || `pose-${index + 1}`);
  const match = raw.match(/(\d+)$/);
  return match ? `bbox-${match[1]}` : raw;
}

function clearSquarePoseLabels() {
  if (els.squarePoseLabelOverlay) els.squarePoseLabelOverlay.replaceChildren();
}

function renderSquarePoseLabels(frame, settings, info, three) {
  const svg = els.squarePoseLabelOverlay;
  if (!svg) return;
  svg.replaceChildren();
  if (!settings.show || !settings.showLabels || !info?.available || info.stale) return;
  const objects = (Array.isArray(frame?.objects) ? frame.objects : []).filter((object) => object?.bestPose);
  if (!objects.length) return;
  const namespace = 'http://www.w3.org/2000/svg';
  const width = state.viewport.frameWidth || 640;
  const height = state.viewport.frameHeight || 360;
  const labelMargin = 4;
  objects.forEach((object, index) => {
    const pose = object.bestPose;
    const anchor = squarePoseLabelAnchor(object, settings, three);
    if (!anchor || !anchor.visible) return;
    const valid = poseMeasurementValid(object);
    const validity = object.measurementValidity || {};
    const ambiguity = object.ambiguityFlags || {};
    const t = (pose.tvec || [0, 0, 0]).map(Number);
    const rpy = poseRpyDeg(pose);
    const stateText = !valid
      ? `ignored ${validity.clipped ? 'clipped' : 'invalid'}`
      : ambiguity.multiGateEvidence
        ? 'multi-gate'
        : ambiguity.ambiguousOverlap
          ? 'ambiguous'
          : object.accepted ? 'ok' : 'weak';
    const lines = valid
      ? [
          `${shortPoseId(object, index)} ${stateText}`,
          `xyz ${fmt(t[0], 2)} ${fmt(t[1], 2)} ${fmt(t[2], 2)}m`,
          `rpy ${fmt(rpy.roll, 1)} ${fmt(rpy.pitch, 1)} ${fmt(rpy.yaw, 1)}`
        ]
      : [
          `${shortPoseId(object, index)} ${stateText}`,
          `pose not updated`,
          `${(validity.reasonCodes || object.reasonCodes || []).join(', ') || 'measurement invalid'}`
        ];
    const maxChars = Math.max(...lines.map((line) => line.length));
    const boxWidth = Math.min(178, Math.max(92, maxChars * 4.8 + 8));
    const boxHeight = 34;
    let boxX = anchor.x + 7;
    let boxY = anchor.y - boxHeight - 7;
    if (boxX + boxWidth > width - labelMargin) boxX = anchor.x - boxWidth - 7;
    if (boxY < labelMargin) boxY = anchor.y + 7;
    boxX = Math.max(labelMargin, Math.min(width - boxWidth - labelMargin, boxX));
    boxY = Math.max(labelMargin, Math.min(height - boxHeight - labelMargin, boxY));

    const group = document.createElementNS(namespace, 'g');
    group.setAttribute('class', 'poseLabel');

    const leader = document.createElementNS(namespace, 'path');
    leader.setAttribute('class', 'poseLabelLeader');
    leader.setAttribute('d', `M ${anchor.x.toFixed(2)} ${anchor.y.toFixed(2)} L ${boxX.toFixed(2)} ${(boxY + boxHeight * 0.5).toFixed(2)}`);
    group.appendChild(leader);

    const dot = document.createElementNS(namespace, 'circle');
    dot.setAttribute('class', 'poseLabelAnchor');
    dot.setAttribute('cx', anchor.x.toFixed(2));
    dot.setAttribute('cy', anchor.y.toFixed(2));
    dot.setAttribute('r', valid && object.accepted ? '2.5' : '2');
    group.appendChild(dot);

    const rect = document.createElementNS(namespace, 'rect');
    rect.setAttribute('class', 'poseLabelBox');
    rect.setAttribute('x', boxX.toFixed(2));
    rect.setAttribute('y', boxY.toFixed(2));
    rect.setAttribute('width', boxWidth.toFixed(2));
    rect.setAttribute('height', boxHeight.toFixed(2));
    rect.setAttribute('rx', '2');
    group.appendChild(rect);

    const text = document.createElementNS(namespace, 'text');
    text.setAttribute('class', 'poseLabelText');
    text.setAttribute('x', (boxX + 4).toFixed(2));
    text.setAttribute('y', (boxY + 10).toFixed(2));
    lines.forEach((line, lineIndex) => {
      const span = document.createElementNS(namespace, 'tspan');
      span.setAttribute('x', (boxX + 4).toFixed(2));
      if (lineIndex > 0) span.setAttribute('dy', '10');
      span.textContent = line;
      text.appendChild(span);
    });
    group.appendChild(text);
    svg.appendChild(group);
  });
}

function renderSquarePoseStats(object, settings, info) {
  if (!els.squarePoseStats) return;
  const label = 'Square pose';
  const job = state.squarePose.discovery?.job || null;
  if (job && ['queued', 'running'].includes(job.status)) {
    const progress = job.progress || {};
    els.squarePoseStats.textContent = `${label} ${job.status}: ${progress.phase || 'working'} ${progress.index || 0}/${progress.total || '?'}`;
    return;
  }
  if (!info) {
    els.squarePoseStats.textContent = `${label} status has not loaded.`;
    return;
  }
  if (!info.available) {
    els.squarePoseStats.textContent = info.contours?.available ? 'No square pose manifest has been built.' : 'Build contours before square pose.';
    return;
  }
  if (info.stale) {
    els.squarePoseStats.textContent = `${label} needs rebuild: ${(info.staleReasons || []).join(', ') || 'stale'}.`;
    return;
  }
  if (!object) {
    els.squarePoseStats.textContent = 'No square pose object for this frame.';
    return;
  }
  const pose = object.bestPose;
  const valid = poseMeasurementValid(object);
  const validity = object.measurementValidity || {};
  const ambiguity = object.ambiguityFlags || {};
  const edge = object.edgeSignals || {};
  const edgeText = settings.showScores
    ? ` | T ${Math.round((edge.top?.coverage || 0) * 100)} R ${Math.round((edge.right?.coverage || 0) * 100)} B ${Math.round((edge.bottom?.coverage || 0) * 100)} L ${Math.round((edge.left?.coverage || 0) * 100)}`
    : '';
  const fit = object.fitQuality?.overall != null ? ` | fit ${fmt(object.fitQuality.overall, 2)}` : '';
  if (pose && !valid) {
    els.squarePoseStats.textContent = `${object.bboxId || 'bbox'} pose ignored: ${(validity.reasonCodes || []).join(', ') || 'invalid'} | raw depth ${fmt(pose.depthM, 3)}m | raw reproj ${fmt(pose.reprojectionErrorPx, 2)}px${edgeText}`;
    return;
  }
  const ambiguityText = ambiguity.multiGateEvidence ? ' | multi-gate' : ambiguity.ambiguousOverlap ? ' | ambiguous' : '';
  els.squarePoseStats.textContent = pose
    ? `${object.bboxId || 'bbox'} ${object.accepted ? 'accepted' : 'weak'}${ambiguityText} | depth ${fmt(pose.depthM, 3)}m | reproj ${fmt(pose.reprojectionErrorPx, 2)}px${fit}${edgeText}`
    : `${object.bboxId || 'bbox'} rejected: ${(object.reasonCodes || []).join(', ') || 'no pose'}`;
}

function renderSquarePose() {
  const settings = squarePoseState();
  const info = activeViewerPoseInfo();
  const three = initSquarePoseViewer();
  if (!three) {
    if (els.squarePoseStats) els.squarePoseStats.textContent = 'Three.js is unavailable for square pose.';
    return;
  }
  const { THREE, scene, renderer, cameraLocked, cameraOrbit, controls, frameTexture, plane, material, outline, sideLine, frustum } = three;
  frameTexture.needsUpdate = true;
  material.opacity = settings.show ? settings.textureOpacity : 0;
  outline.material.opacity = settings.candidateOpacity;
  sideLine.material.opacity = settings.candidateOpacity;
  frustum.visible = settings.showFrustum && settings.viewMode === 'orbit';
  outline.visible = Boolean(settings.showOutline && settings.show);
  plane.visible = Boolean(settings.show);
  scene.background = settings.viewMode === 'camera' ? frameTexture : new THREE.Color(0x020406);

  const frame = viewerPoseFrameEntry();
  const object = selectSquarePoseObject(frame);
  renderSquarePoseStats(object, settings, info);
  if (!settings.show || !object?.bestPose || !info?.available || info.stale) {
    plane.visible = false;
    outline.visible = false;
    sideLine.visible = false;
    clearSquarePoseLabels();
    renderer.render(scene, settings.viewMode === 'orbit' ? cameraOrbit : cameraLocked);
    return;
  }
  if (!poseMeasurementValid(object)) {
    plane.visible = false;
    outline.visible = false;
    sideLine.visible = false;
    renderSquarePoseLabels(frame, settings, info, three);
    renderer.render(scene, settings.viewMode === 'orbit' ? cameraOrbit : cameraLocked);
    return;
  }

  const squareSize = Number(viewerPoseManifest()?.square?.widthM || settings.squareSizeM || 2.7);
  const matrix = matrixFromCvPose(THREE, object.bestPose);
  applySquarePoseMatrix(plane, matrix, squareSize);
  applySquarePoseMatrix(outline, matrix, squareSize);
  setSquareSideLine(three, matrix, squareSize, settings.solutionMode);
  if (settings.solutionMode === 'best') sideLine.visible = false;
  if (controls) {
    controls.enabled = settings.viewMode === 'orbit';
    if (controls.enabled) controls.update();
  }
  renderSquarePoseLabels(frame, settings, info, three);
  renderer.render(scene, settings.viewMode === 'orbit' ? cameraOrbit : cameraLocked);
}

function scheduleInstance3dRender() {
  if (state.instance3d.renderRaf) return;
  state.instance3d.renderRaf = requestAnimationFrame(() => {
    state.instance3d.renderRaf = 0;
    renderInstance3d();
  });
}

function resizeInstance3dRenderer() {
  const three = state.instance3d.three;
  if (!three || !els.instance3dCanvas) return;
  const rect = els.instance3dSurface?.getBoundingClientRect?.();
  const width = Math.max(1, Math.round(rect?.width || state.viewport.frameWidth || els.instance3dCanvas.width || 640));
  const height = Math.max(1, Math.round(rect?.height || state.viewport.frameHeight || els.instance3dCanvas.height || 360));
  three.renderer.setSize(width, height, false);
  three.camera.aspect = width / Math.max(1, height);
  three.camera.updateProjectionMatrix();
}

function instance3dFrustumGeometry(THREE, depth = 8) {
  const camera = { widthPx: 640, heightPx: 360, fx: 320, fy: 320, cx: 320, cy: 180 };
  const pixels = [[0, 0], [camera.widthPx, 0], [camera.widthPx, camera.heightPx], [0, camera.heightPx]];
  const corners = pixels.map(([u, v]) => {
    const x = ((u - camera.cx) / camera.fx) * depth;
    const y = ((v - camera.cy) / camera.fy) * depth;
    return new THREE.Vector3(x, -y, -depth);
  });
  const origin = new THREE.Vector3(0, 0, 0);
  const points = [];
  for (const corner of corners) points.push(origin, corner);
  for (let index = 0; index < 4; index++) points.push(corners[index], corners[(index + 1) % 4]);
  return new THREE.BufferGeometry().setFromPoints(points);
}

function initInstance3dViewer() {
  if (state.instance3d.three || !els.instance3dCanvas || !window.THREE) return state.instance3d.three;
  const THREE = window.THREE;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x030507);
  const renderer = new THREE.WebGLRenderer({ canvas: els.instance3dCanvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  if ('outputColorSpace' in renderer && THREE.SRGBColorSpace) renderer.outputColorSpace = THREE.SRGBColorSpace;
  else renderer.outputEncoding = THREE.sRGBEncoding;

  const camera = new THREE.PerspectiveCamera(45, 640 / 360, 0.01, 240);
  camera.position.set(4, 2.2, 3.2);
  camera.lookAt(0, -0.1, -8);

  const controls = THREE.OrbitControls ? new THREE.OrbitControls(camera, renderer.domElement) : null;
  if (controls) {
    controls.target.set(0, -0.1, -8);
    controls.enableDamping = false;
    controls.minDistance = 0.4;
    controls.maxDistance = 160;
    controls.addEventListener('change', scheduleInstance3dRender);
    controls.update();
  }

  const grid = new THREE.GridHelper(24, 24, 0x294654, 0x172733);
  grid.position.z = -8;
  grid.material.transparent = true;
  grid.material.opacity = 0.36;
  scene.add(grid);

  const axes = new THREE.AxesHelper(1.4);
  scene.add(axes);

  const origin = new THREE.Mesh(
    new THREE.SphereGeometry(0.045, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, depthTest: false })
  );
  scene.add(origin);

  const frustum = new THREE.LineSegments(
    instance3dFrustumGeometry(THREE, 8),
    new THREE.LineBasicMaterial({ color: 0x55f7ff, transparent: true, opacity: 0.36, depthTest: false })
  );
  scene.add(frustum);

  state.instance3d.three = { THREE, scene, renderer, camera, controls, sceneObjects: [] };
  resizeInstance3dRenderer();
  return state.instance3d.three;
}

function disposeThreeMaterial(material) {
  if (!material) return;
  const materials = Array.isArray(material) ? material : [material];
  for (const item of materials) {
    if (item.map) item.map.dispose?.();
    item.dispose?.();
  }
}

function disposeInstance3dObject(object) {
  object.traverse?.((child) => {
    child.geometry?.dispose?.();
    disposeThreeMaterial(child.material);
  });
}

function clearInstance3dSceneObjects(three) {
  const objects = state.instance3d.sceneObjects || [];
  for (const object of objects) {
    three.scene.remove(object);
    disposeInstance3dObject(object);
  }
  state.instance3d.sceneObjects = [];
}

function parseCssHslColor(THREE, text) {
  const match = String(text || '').trim().match(/^hsla?\(\s*([-\d.]+)(?:deg)?(?:\s+|,\s*)([-\d.]+)%(?:\s+|,\s*)([-\d.]+)%(?:\s*\/\s*[-\d.]+%?)?\s*\)$/i);
  if (!match) return null;
  const color = new THREE.Color();
  color.setHSL((((Number(match[1]) || 0) % 360) + 360) % 360 / 360, clamp01(Number(match[2]) / 100), clamp01(Number(match[3]) / 100));
  return color;
}

function instance3dColor(THREE, value, fallback = 0x55f7ff) {
  const text = String(value || '').trim();
  const color = parseCssHslColor(THREE, text) || new THREE.Color(fallback);
  if (!text) return color;
  try {
    color.setStyle(text);
    return color;
  } catch (error) {
    if (parseCssHslColor(THREE, text)) return color;
  }
  try {
    color.set(text);
  } catch (error) {
    color.set(fallback);
  }
  return color;
}

function instance3dPosePoint(THREE, pose) {
  const raw = Array.isArray(pose?.xyzCameraM) ? pose.xyzCameraM : Array.isArray(pose?.tvec) ? pose.tvec : null;
  if (!raw || raw.length < 3) return null;
  const [x, y, z] = raw.map(Number);
  if (![x, y, z].every(Number.isFinite)) return null;
  return new THREE.Vector3(x, -y, -z);
}

function instance3dPoseRenderable(pose) {
  return Boolean(pose?.available && (Array.isArray(pose.xyzCameraM) || Array.isArray(pose.tvec)));
}

function instance3dPoseValid(observation) {
  const pose = observation?.pose || {};
  if (!pose.available) return false;
  const validity = pose.measurementValidity || {};
  return validity.valid !== false && validity.clipped !== true && observation?.fovClip?.status !== 'clipped';
}

function createInstance3dLabel(THREE, text, color) {
  const canvas = document.createElement('canvas');
  canvas.width = 160;
  canvas.height = 40;
  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.font = '22px ui-monospace, Menlo, monospace';
  context.textBaseline = 'middle';
  context.fillStyle = 'rgba(0, 0, 0, 0.68)';
  context.fillRect(0, 2, canvas.width, 30);
  context.fillStyle = `#${color.getHexString()}`;
  context.fillRect(0, 2, 5, 30);
  context.strokeStyle = 'rgba(255, 255, 255, 0.22)';
  context.strokeRect(0.5, 2.5, canvas.width - 1, 29);
  context.fillStyle = `#${color.getHexString()}`;
  context.fillText(text, 12, 18);
  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.62, 0.16, 1);
  return sprite;
}

function buildInstance3dMarker(observation) {
  const three = state.instance3d.three;
  if (!three) return null;
  const { THREE } = three;
  const pose = observation?.pose || {};
  if (!instance3dPoseRenderable(pose)) return null;
  const position = instance3dPosePoint(THREE, pose);
  if (!position) return null;
  const valid = instance3dPoseValid(observation);
  const color = instance3dColor(THREE, observation?.instanceColor, 0x55f7ff);
  const group = new THREE.Group();

  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(valid ? 0.095 : 0.07, 20, 14),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: valid ? 0.96 : 0.34,
      depthTest: true,
      depthWrite: false
    })
  );
  marker.position.copy(position);
  group.add(marker);

  const label = createInstance3dLabel(THREE, instanceDisplayId(observation), color);
  label.position.copy(position).add(new THREE.Vector3(0.08, 0.16, 0));
  group.add(label);

  if (Array.isArray(pose.rvec)) {
    const poseForMatrix = {
      rvec: pose.rvec,
      tvec: Array.isArray(pose.tvec) ? pose.tvec : pose.xyzCameraM
    };
    const matrix = matrixFromCvPose(THREE, poseForMatrix);
    const squareSize = 2.7;
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: valid ? 0.08 : 0.025,
        side: THREE.DoubleSide,
        depthTest: true,
        depthWrite: false
      })
    );
    plane.matrixAutoUpdate = false;
    applySquarePoseMatrix(plane, matrix, squareSize);
    group.add(plane);

    const outline = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-0.5, 0.5, 0.006),
        new THREE.Vector3(0.5, 0.5, 0.006),
        new THREE.Vector3(0.5, -0.5, 0.006),
        new THREE.Vector3(-0.5, -0.5, 0.006)
      ]),
      new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: valid ? 0.88 : 0.24,
        depthTest: false
      })
    );
    outline.matrixAutoUpdate = false;
    applySquarePoseMatrix(outline, matrix, squareSize);
    group.add(outline);
  }
  return { group, valid };
}

function buildInstance3dTrail(observation, settings) {
  const three = state.instance3d.three;
  if (!three || !observation?.instanceId) return null;
  const { THREE } = three;
  const frameOrdinal = Number(observation.frameOrdinal);
  if (!Number.isFinite(frameOrdinal)) return null;
  const frames = Array.isArray(state.instances.manifest?.frames) ? state.instances.manifest.frames : [];
  const low = frameOrdinal - Math.max(1, Number(settings.trailLengthFrames) || 1);
  const points = [];
  for (const frame of frames) {
    const current = Number(frame.frameOrdinal);
    if (!Number.isFinite(current) || current < low || current > frameOrdinal) continue;
    const match = (frame.observations || []).find((item) => item.instanceId === observation.instanceId);
    const pose = match?.pose || {};
    if (!instance3dPoseRenderable(pose)) continue;
    const point = instance3dPosePoint(THREE, pose);
    if (point) points.push(point);
  }
  if (points.length < 2) return null;
  const color = instance3dColor(THREE, observation.instanceColor, 0x55f7ff);
  return new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.5,
      depthTest: false,
      depthWrite: false
    })
  );
}

function renderInstance3d() {
  const three = initInstance3dViewer();
  if (!three) {
    if (els.instance3dHud) els.instance3dHud.textContent = 'Three.js unavailable';
    return;
  }
  const { scene, renderer, camera, controls } = three;
  clearInstance3dSceneObjects(three);
  const info = activeInstanceInfo();
  let hudText = 'No instance provenance';
  if (!info || !info.available || !state.instances.manifest) {
    hudText = info?.stale ? 'Instance provenance stale' : 'No instance provenance';
  } else {
    const settings = instanceTrackingState();
    const frameIndex = currentReviewFrameIndex();
    const frame = instanceFrameEntry(frameIndex);
    const observations = Array.isArray(frame?.observations) ? frame.observations : [];
    let poseCount = 0;
    let invalidCount = 0;
    let trailCount = 0;
    for (const observation of observations) {
      if (settings.showLinks) {
        const trail = buildInstance3dTrail(observation, settings);
        if (trail) {
          scene.add(trail);
          state.instance3d.sceneObjects.push(trail);
          trailCount += 1;
        }
      }
      const marker = buildInstance3dMarker(observation);
      if (!marker) continue;
      poseCount += 1;
      if (!marker.valid) invalidCount += 1;
      scene.add(marker.group);
      state.instance3d.sceneObjects.push(marker.group);
    }
    hudText = `${info.stale ? 'stale | ' : ''}frame ${frameIndex + 1} | ${poseCount}/${observations.length} poses`;
    if (invalidCount) hudText += ` | ${invalidCount} dim`;
    if (trailCount) hudText += ` | ${trailCount} tracks`;
  }
  if (els.instance3dHud) els.instance3dHud.textContent = hudText;
  controls?.update?.();
  renderer.render(scene, camera);
}

function validPointList(points) {
  return Array.isArray(points)
    && points.length >= 4
    && points.every((point) => Array.isArray(point) && point.length >= 2 && Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1])));
}

function svgPointList(points) {
  return points.map((point) => `${Number(point[0])},${Number(point[1])}`).join(' ');
}

function appendSvgRect(svg, namespace, rectValues, className) {
  if (!Array.isArray(rectValues) || rectValues.length !== 4) return null;
  const [x0, y0, x1, y1] = rectValues.map(Number);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
  const rect = document.createElementNS(namespace, 'rect');
  rect.setAttribute('class', className);
  rect.setAttribute('x', String(x0 + 0.5));
  rect.setAttribute('y', String(y0 + 0.5));
  rect.setAttribute('width', String(Math.max(1, x1 - x0 + 1)));
  rect.setAttribute('height', String(Math.max(1, y1 - y0 + 1)));
  svg.appendChild(rect);
  return rect;
}

function appendSvgLine(svg, namespace, x0, y0, x1, y1, className) {
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
  const line = document.createElementNS(namespace, 'line');
  line.setAttribute('class', className);
  line.setAttribute('x1', String(x0));
  line.setAttribute('y1', String(y0));
  line.setAttribute('x2', String(x1));
  line.setAttribute('y2', String(y1));
  svg.appendChild(line);
  return line;
}

function appendSvgPolygon(svg, namespace, points, className) {
  if (!validPointList(points)) return null;
  const polygon = document.createElementNS(namespace, 'polygon');
  polygon.setAttribute('class', className);
  polygon.setAttribute('points', svgPointList(points));
  svg.appendChild(polygon);
  return polygon;
}

function validContourPointList(points) {
  return Array.isArray(points)
    && points.length >= 2
    && points.every((point) => Array.isArray(point) && point.length >= 2 && Number.isFinite(Number(point[0])) && Number.isFinite(Number(point[1])));
}

function contourDisplayPoints(feature) {
  if (validContourPointList(feature?.rawPointsPx)) return feature.rawPointsPx;
  if (validContourPointList(feature?.pointsPx)) return feature.pointsPx;
  return null;
}

function svgPathFromPoints(points, closed = true) {
  if (!validContourPointList(points)) return '';
  const [first, ...rest] = points;
  const commands = [`M ${Number(first[0])} ${Number(first[1])}`];
  for (const point of rest) {
    commands.push(`L ${Number(point[0])} ${Number(point[1])}`);
  }
  if (closed) commands.push('Z');
  return commands.join(' ');
}

function appendSvgPath(svg, namespace, points, className, closed = true) {
  const d = svgPathFromPoints(points, closed);
  if (!d) return null;
  const path = document.createElementNS(namespace, 'path');
  path.setAttribute('class', className);
  path.setAttribute('d', d);
  svg.appendChild(path);
  return path;
}

function cornerMaskPosition(corner) {
  const position = String(corner.maskAnglePosition || '');
  if (['inside', 'outside', 'ambiguous'].includes(position)) return position;
  const type = String(corner.type || 'ambiguous');
  if (type === 'mask-wedge') return 'inside';
  if (type === 'void-wedge') return 'outside';
  return 'ambiguous';
}

function cornerTypeKey(corner) {
  const explicit = String(corner.cornerType || '');
  if (CORNER_TYPE_DEFS[explicit]) return explicit;
  const category = String(corner.category || 'hull');
  const position = cornerMaskPosition(corner);
  if (category === 'hull' && position === 'inside') return 'white';
  if (category === 'hull' && position === 'outside') return 'black';
  if (category === 'void' && position === 'inside') return 'orange';
  if (category === 'void' && position === 'outside') return 'green';
  return 'ambiguous';
}

function cornerPointStyle(corner) {
  const category = String(corner.category || 'hull');
  const position = cornerMaskPosition(corner);
  if (category === 'hull' && position === 'inside') {
    return { fill: '#ffffff', stroke: '#101820', className: 'maskInside' };
  }
  if (category === 'hull' && position === 'outside') {
    return { fill: '#050505', stroke: '#f4f7fb', className: 'maskOutside' };
  }
  if (category === 'void' && position === 'inside') {
    return { fill: '#ff9f1c', stroke: '#5a2d00', className: 'maskInside' };
  }
  if (category === 'void' && position === 'outside') {
    return { fill: '#20c65a', stroke: '#073a18', className: 'maskOutside' };
  }
  return { fill: '#ffb338', stroke: '#3a2600', className: 'ambiguous' };
}

function appendCornerMark(svg, namespace, corner, settings) {
  const point = Array.isArray(corner.pointPx) ? corner.pointPx.map(Number) : null;
  if (!point || point.length < 2 || !point.every(Number.isFinite)) return;
  const position = cornerMaskPosition(corner);
  const category = String(corner.category || (String(corner.type || '') === 'void-wedge' ? 'void' : 'hull'));
  const typeKey = cornerTypeKey({ ...corner, category });
  if (category === 'hull' && !settings.showHull) return;
  if (category === 'void' && !settings.showVoid) return;
  if (typeKey !== 'ambiguous' && settings[`show${typeKey[0].toUpperCase()}${typeKey.slice(1)}`] === false) return;
  if (position === 'ambiguous' && !settings.showAmbiguous) return;
  const style = cornerPointStyle({ ...corner, category });
  const className = style.className;
  const [x, y] = point;
  const direction = Number(corner.directionDeg || 0) * Math.PI / 180;
  const rayLength = Math.max(5, Math.min(18, Number(corner.radiusPx || settings.radiusPx || 8)));
  const ray = document.createElementNS(namespace, 'line');
  ray.setAttribute('class', `cornerRay ${className} ${category}`);
  ray.setAttribute('stroke', style.fill);
  ray.setAttribute('x1', String(x));
  ray.setAttribute('y1', String(y));
  ray.setAttribute('x2', String(x + Math.cos(direction) * rayLength));
  ray.setAttribute('y2', String(y + Math.sin(direction) * rayLength));
  ray.setAttribute('opacity', String(settings.overlayOpacity));
  svg.appendChild(ray);
  const dot = document.createElementNS(namespace, 'circle');
  dot.setAttribute('class', `cornerDot ${className} ${category}`);
  dot.setAttribute('cx', String(x));
  dot.setAttribute('cy', String(y));
  dot.setAttribute('r', '3.5');
  dot.setAttribute('stroke', style.stroke);
  dot.setAttribute('fill', style.fill);
  dot.setAttribute('opacity', String(settings.overlayOpacity));
  svg.appendChild(dot);
}

function flattenedCornerFrameMarks(frame) {
  if (Array.isArray(frame?.corners)) return frame.corners;
  const marks = [];
  const groups = Array.isArray(frame?.bboxCorners) ? frame.bboxCorners : [];
  for (const group of groups) {
    const bboxId = group?.bboxId;
    const hullCorners = Array.isArray(group?.hullCorners) ? group.hullCorners : [];
    for (const corner of hullCorners) {
      marks.push({ ...corner, bboxId: corner.bboxId || bboxId, category: corner.category || 'hull' });
    }
    const voids = Array.isArray(group?.voids) ? group.voids : [];
    for (const voidEntry of voids) {
      const voidId = voidEntry?.voidId;
      const voidCorners = Array.isArray(voidEntry?.voidCorners) ? voidEntry.voidCorners : [];
      for (const corner of voidCorners) {
        marks.push({ ...corner, bboxId: corner.bboxId || bboxId, voidId: corner.voidId || voidId, category: corner.category || 'void' });
      }
    }
  }
  return marks;
}

function normalizedAngleDeg(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return ((number % 360) + 360) % 360;
}

function angleDeltaDeg(a, b) {
  const diff = Math.abs(normalizedAngleDeg(a) - normalizedAngleDeg(b)) % 360;
  return diff > 180 ? 360 - diff : diff;
}

function pointFromCorner(corner) {
  const point = Array.isArray(corner?.pointPx) ? corner.pointPx.map(Number) : null;
  if (!point || point.length < 2 || !point.every(Number.isFinite)) return null;
  return { x: point[0], y: point[1] };
}

function angleBetweenPointsDeg(from, to) {
  return normalizedAngleDeg(Math.atan2(to.y - from.y, to.x - from.x) * 180 / Math.PI);
}

function cornerAgreementGroupKey(link) {
  return `${link.bboxId || 'bbox'}|${link.voidId || 'void'}`;
}

function completeAgreementKeys(links, settings) {
  if (!settings.showStructures) return new Set();
  const groups = new Map();
  for (const link of links) {
    const key = cornerAgreementGroupKey(link);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(link);
  }
  const complete = new Set();
  for (const [key, groupLinks] of groups.entries()) {
    if (groupLinks.length < 4) continue;
    let oppositePairs = 0;
    for (let i = 0; i < groupLinks.length; i++) {
      for (let j = i + 1; j < groupLinks.length; j++) {
        const delta = angleDeltaDeg(groupLinks[i].vectorAngleDeg, groupLinks[j].vectorAngleDeg);
        if (Math.abs(180 - delta) <= settings.oppositeToleranceDeg) oppositePairs++;
      }
    }
    if (oppositePairs >= 2) complete.add(key);
  }
  return complete;
}

function cornerAgreementResults(settings = cornerAgreementState()) {
  const frame = cornerFrameEntry();
  const corners = flattenedCornerFrameMarks(frame);
  const whites = [];
  const greens = [];
  for (const corner of corners) {
    const typeKey = cornerTypeKey(corner);
    if (typeKey === 'white') whites.push(corner);
    if (typeKey === 'green') greens.push(corner);
  }

  const candidates = [];
  for (const white of whites) {
    const whitePoint = pointFromCorner(white);
    const whiteBboxId = String(white.bboxId || '');
    if (!whitePoint || !whiteBboxId) continue;
    for (const green of greens) {
      const greenPoint = pointFromCorner(green);
      if (!greenPoint || String(green.bboxId || '') !== whiteBboxId) continue;
      const dx = greenPoint.x - whitePoint.x;
      const dy = greenPoint.y - whitePoint.y;
      const distance = Math.hypot(dx, dy);
      if (!Number.isFinite(distance) || distance <= 0.5 || distance > settings.maxDistancePx) continue;
      const vectorAngle = angleBetweenPointsDeg(whitePoint, greenPoint);
      const whiteDelta = angleDeltaDeg(white.directionDeg, vectorAngle);
      if (whiteDelta > settings.whiteToleranceDeg) continue;
      const angleDelta = angleDeltaDeg(white.angleDeg, green.angleDeg);
      if (angleDelta > settings.angleToleranceDeg) continue;
      const directionScore = 1 - (whiteDelta / Math.max(1, settings.whiteToleranceDeg));
      const cornerAngleScore = 1 - (angleDelta / Math.max(1, settings.angleToleranceDeg));
      const distanceScore = 1 - (distance / Math.max(1, settings.maxDistancePx));
      const score = (directionScore * 0.45) + (cornerAngleScore * 0.35) + (distanceScore * 0.2);
      if (score < settings.minScore) continue;
      candidates.push({
        white,
        green,
        whitePoint,
        greenPoint,
        bboxId: whiteBboxId,
        voidId: String(green.voidId || ''),
        distance,
        vectorAngleDeg: vectorAngle,
        whiteDirectionDeltaDeg: whiteDelta,
        cornerAngleDeltaDeg: angleDelta,
        score
      });
    }
  }

  candidates.sort((a, b) => (
    b.score - a.score
    || a.distance - b.distance
    || String(a.white.cornerId || '').localeCompare(String(b.white.cornerId || ''))
    || String(a.green.cornerId || '').localeCompare(String(b.green.cornerId || ''))
  ));
  const maxLinks = Math.max(1, Math.round(settings.maxLinksPerCorner));
  const whiteCounts = new Map();
  const greenCounts = new Map();
  const links = [];
  for (const candidate of candidates) {
    const whiteId = String(candidate.white.cornerId || `${candidate.whitePoint.x},${candidate.whitePoint.y}`);
    const greenId = String(candidate.green.cornerId || `${candidate.greenPoint.x},${candidate.greenPoint.y}`);
    const whiteCount = whiteCounts.get(whiteId) || 0;
    const greenCount = greenCounts.get(greenId) || 0;
    if (whiteCount >= maxLinks || greenCount >= maxLinks) continue;
    whiteCounts.set(whiteId, whiteCount + 1);
    greenCounts.set(greenId, greenCount + 1);
    links.push(candidate);
  }

  const completeKeys = completeAgreementKeys(links, settings);
  for (const link of links) {
    link.complete = completeKeys.has(cornerAgreementGroupKey(link));
  }
  return { links, completeGroupCount: completeKeys.size };
}

function axisAngleDeg(value) {
  let angle = normalizedAngleDeg(value);
  if (angle >= 180) angle -= 180;
  return angle;
}

function bboxPixelBounds(bbox, width, height) {
  const values = Array.isArray(bbox?.bboxPx) ? bbox.bboxPx.map(Number) : null;
  if (!values || values.length !== 4 || !values.every(Number.isFinite)) return null;
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(values[0])));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(values[1])));
  const x1 = Math.max(0, Math.min(width - 1, Math.ceil(values[2])));
  const y1 = Math.max(0, Math.min(height - 1, Math.ceil(values[3])));
  if (x1 < x0 || y1 < y0) return null;
  return { x0, y0, x1, y1 };
}

function classBitForPrefix(prefix) {
  const index = state.config?.classes?.findIndex((item) => String(item.prefix) === String(prefix)) ?? -1;
  if (index < 0 || index >= 31) return 0;
  return 1 << index;
}

function buildMaskIntegral(mask, width, height) {
  const stride = width + 1;
  const integral = new Uint32Array((height + 1) * stride);
  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    const sourceRow = y * width;
    const targetRow = (y + 1) * stride;
    const previousRow = y * stride;
    for (let x = 0; x < width; x++) {
      rowSum += mask[sourceRow + x] ? 1 : 0;
      integral[targetRow + x + 1] = integral[previousRow + x + 1] + rowSum;
    }
  }
  return { integral, stride };
}

function layerMaskForPrefix(prefix) {
  if (!state.currentColorIds || !state.classMembership || !state.config?.classes) return null;
  const width = els.overlayCanvas.width;
  const height = els.overlayCanvas.height;
  const bit = classBitForPrefix(prefix);
  if (!bit || width <= 0 || height <= 0) return null;
  const total = Math.min(width * height, state.currentColorIds.length);
  const mask = new Uint8Array(width * height);
  for (let index = 0; index < total; index++) {
    const colorId = state.currentColorIds[index];
    if (colorId >= 0 && colorId < state.classMembership.length && (state.classMembership[colorId] & bit)) {
      mask[index] = 1;
    }
  }
  return { mask, width, height, ...buildMaskIntegral(mask, width, height) };
}

function integralRectSum(maskInfo, x0, y0, x1, y1) {
  const left = Math.max(0, Math.min(maskInfo.width - 1, Math.floor(x0)));
  const top = Math.max(0, Math.min(maskInfo.height - 1, Math.floor(y0)));
  const right = Math.max(0, Math.min(maskInfo.width - 1, Math.ceil(x1)));
  const bottom = Math.max(0, Math.min(maskInfo.height - 1, Math.ceil(y1)));
  if (right < left || bottom < top) return 0;
  const stride = maskInfo.stride;
  const integral = maskInfo.integral;
  const ax = left;
  const ay = top;
  const bx = right + 1;
  const by = bottom + 1;
  return integral[by * stride + bx] - integral[ay * stride + bx] - integral[by * stride + ax] + integral[ay * stride + ax];
}

function decodedMaskInfo() {
  if (!state.currentColorIds || !state.classMembership) return null;
  const width = els.overlayCanvas.width;
  const height = els.overlayCanvas.height;
  if (width <= 0 || height <= 0) return null;
  const total = Math.min(width * height, state.currentColorIds.length);
  const mask = new Uint8Array(width * height);
  for (let index = 0; index < total; index++) {
    const colorId = state.currentColorIds[index];
    if (colorId >= 0 && colorId < state.classMembership.length && state.classMembership[colorId]) {
      mask[index] = 1;
    }
  }
  return { mask, width, height, ...buildMaskIntegral(mask, width, height) };
}

function maskNeighborOffsets(connectivity = 8) {
  const fourWay = [[0, -1], [1, 0], [0, 1], [-1, 0]];
  if (Number(connectivity) === 4) return fourWay;
  return [...fourWay, [1, -1], [1, 1], [-1, 1], [-1, -1]];
}

function topLeftLocalIndex(indices, width) {
  let best = -1;
  let bestScore = Infinity;
  for (const index of indices) {
    const x = index % width;
    const y = Math.floor(index / width);
    const score = (x + y) * 10000 + y * 100 + x;
    if (score < bestScore) {
      bestScore = score;
      best = index;
    }
  }
  return best;
}

function cropHullMasks(hullInfo, layer002Info, bounds) {
  const width = bounds.x1 - bounds.x0 + 1;
  const height = bounds.y1 - bounds.y0 + 1;
  const hullMask = new Uint8Array(width * height);
  const layer002Mask = new Uint8Array(width * height);
  let hullCount = 0;
  let layer002Count = 0;
  for (let y = 0; y < height; y++) {
    const sourceRow = (bounds.y0 + y) * hullInfo.width;
    const targetRow = y * width;
    for (let x = 0; x < width; x++) {
      const sourceIndex = sourceRow + bounds.x0 + x;
      const targetIndex = targetRow + x;
      if (hullInfo.mask[sourceIndex]) {
        hullMask[targetIndex] = 1;
        hullCount++;
      }
      if (layer002Info.mask[sourceIndex]) {
        layer002Mask[targetIndex] = 1;
        layer002Count++;
      }
    }
  }
  return { hullMask, layer002Mask, width, height, hullCount, layer002Count, offsetX: bounds.x0, offsetY: bounds.y0 };
}

function connectedMaskComponents(mask, width, height, layer002Mask, connectivity = 8) {
  const visited = new Uint8Array(mask.length);
  const components = [];
  const offsets = maskNeighborOffsets(connectivity);
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || visited[start]) continue;
    const queue = [start];
    const pixels = [];
    let head = 0;
    let layer002Count = 0;
    visited[start] = 1;
    while (head < queue.length) {
      const index = queue[head++];
      pixels.push(index);
      if (layer002Mask?.[index]) layer002Count++;
      const x = index % width;
      const y = Math.floor(index / width);
      for (const [dx, dy] of offsets) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const next = ny * width + nx;
        if (!mask[next] || visited[next]) continue;
        visited[next] = 1;
        queue.push(next);
      }
    }
    components.push({ pixels, pixelCount: pixels.length, layer002Count });
  }
  components.sort((a, b) => b.layer002Count - a.layer002Count || b.pixelCount - a.pixelCount || topLeftLocalIndex(a.pixels, width) - topLeftLocalIndex(b.pixels, width));
  return components;
}

function boundaryPointsForComponent(component, width, height, offsetX, offsetY) {
  const set = new Set(component.pixels);
  const points = [];
  const offsets = maskNeighborOffsets(4);
  for (const index of component.pixels) {
    const x = index % width;
    const y = Math.floor(index / width);
    let boundary = x === 0 || y === 0 || x === width - 1 || y === height - 1;
    if (!boundary) {
      for (const [dx, dy] of offsets) {
        if (!set.has((y + dy) * width + x + dx)) {
          boundary = true;
          break;
        }
      }
    }
    if (boundary) points.push({ x: offsetX + x + 0.5, y: offsetY + y + 0.5 });
  }
  return points;
}

function convexHull(points) {
  if (points.length <= 3) return points.slice();
  const sorted = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (origin, a, b) => (a.x - origin.x) * (b.y - origin.y) - (a.y - origin.y) * (b.x - origin.x);
  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper = [];
  for (let index = sorted.length - 1; index >= 0; index--) {
    const point = sorted[index];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function uvForPoint(point, cosTheta, sinTheta) {
  return {
    u: point.x * cosTheta + point.y * sinTheta,
    v: -point.x * sinTheta + point.y * cosTheta
  };
}

function pointForUv(u, v, cosTheta, sinTheta) {
  return {
    x: u * cosTheta - v * sinTheta,
    y: u * sinTheta + v * cosTheta
  };
}

function hullExtentsForAngle(points, angleDeg) {
  const theta = angleDeg * Math.PI / 180;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);
  let uMin = Infinity;
  let uMax = -Infinity;
  let vMin = Infinity;
  let vMax = -Infinity;
  for (const point of points) {
    const projected = uvForPoint(point, cosTheta, sinTheta);
    uMin = Math.min(uMin, projected.u);
    uMax = Math.max(uMax, projected.u);
    vMin = Math.min(vMin, projected.v);
    vMax = Math.max(vMax, projected.v);
  }
  return { angleDeg, cosTheta, sinTheta, uMin, uMax, vMin, vMax, width: uMax - uMin, height: vMax - vMin };
}

function baseHullAngle(points) {
  let bestAngle = 0;
  let bestArea = Infinity;
  for (let angle = 0; angle < 180; angle += 5) {
    const extents = hullExtentsForAngle(points, angle);
    const area = Math.max(1, extents.width) * Math.max(1, extents.height);
    if (area < bestArea) {
      bestArea = area;
      bestAngle = angle;
    }
  }
  return axisAngleDeg(bestAngle);
}

function hullSweepAngles(baseAngle, sweepDeg, stepDeg) {
  const step = Math.max(1, Math.round(Number(stepDeg) || 1));
  const sweep = Math.max(0, Math.round(Number(sweepDeg) || 0));
  const values = [];
  const seen = new Set();
  for (let offset = -sweep; offset <= sweep + 1e-6; offset += step) {
    const angle = axisAngleDeg(baseAngle + offset);
    const key = String(Math.round(angle * 1000));
    if (!seen.has(key)) {
      seen.add(key);
      values.push(angle);
    }
  }
  values.sort((a, b) => angleDeltaDeg(a, baseAngle) - angleDeltaDeg(b, baseAngle));
  return values;
}

function sampleEdgeBand(layer002Info, validationInfo, start, end, settings) {
  const length = Math.hypot(end.x - start.x, end.y - start.y);
  const steps = Math.max(1, Math.ceil(length));
  const radius = Math.max(0, Math.round(Number(settings.edgeBandPx) || 0));
  let supported = 0;
  let allowed = 0;
  let supportPixelHits = 0;
  let maxGap = 0;
  let currentGap = 0;
  let maxForbiddenGap = 0;
  let currentForbiddenGap = 0;
  for (let step = 0; step <= steps; step++) {
    const t = step / steps;
    const x = start.x + (end.x - start.x) * t;
    const y = start.y + (end.y - start.y) * t;
    const cx = Math.round(x);
    const cy = Math.round(y);
    const supportCount = integralRectSum(layer002Info, cx - radius, cy - radius, cx + radius, cy + radius);
    const allowedCount = integralRectSum(validationInfo, cx - radius, cy - radius, cx + radius, cy + radius);
    if (supportCount > 0) {
      supported++;
      supportPixelHits += supportCount;
      currentGap = 0;
    } else {
      currentGap++;
      maxGap = Math.max(maxGap, currentGap);
    }
    if (allowedCount > 0) {
      allowed++;
      currentForbiddenGap = 0;
    } else {
      currentForbiddenGap++;
      maxForbiddenGap = Math.max(maxForbiddenGap, currentForbiddenGap);
    }
  }
  const stations = steps + 1;
  return {
    length,
    stations,
    supported,
    allowed,
    supportPixelHits,
    coverage: stations ? supported / stations : 0,
    allowedCoverage: stations ? allowed / stations : 0,
    maxGap,
    maxForbiddenGap
  };
}

function lineForSide(extents, side, offset) {
  const uLeft = extents.uMin;
  const uRight = extents.uMax;
  const vTop = extents.vMin;
  const vBottom = extents.vMax;
  if (side === 'top') return [
    pointForUv(uLeft, vTop + offset, extents.cosTheta, extents.sinTheta),
    pointForUv(uRight, vTop + offset, extents.cosTheta, extents.sinTheta)
  ];
  if (side === 'right') return [
    pointForUv(uRight - offset, vTop, extents.cosTheta, extents.sinTheta),
    pointForUv(uRight - offset, vBottom, extents.cosTheta, extents.sinTheta)
  ];
  if (side === 'bottom') return [
    pointForUv(uRight, vBottom - offset, extents.cosTheta, extents.sinTheta),
    pointForUv(uLeft, vBottom - offset, extents.cosTheta, extents.sinTheta)
  ];
  return [
    pointForUv(uLeft + offset, vBottom, extents.cosTheta, extents.sinTheta),
    pointForUv(uLeft + offset, vTop, extents.cosTheta, extents.sinTheta)
  ];
}

function sideOffsetCandidates(extents, side, layer002Info, validationInfo, settings, targetUv) {
  const span = side === 'top' || side === 'bottom' ? extents.height : extents.width;
  const maxOffset = Math.max(0, Math.min(settings.inwardMaxPx, Math.floor(span / 2) - 1));
  const step = Math.max(1, settings.insetStepPx);
  const candidates = [];
  for (let offset = 0; offset <= maxOffset + 1e-6; offset += step) {
    const [start, end] = lineForSide(extents, side, offset);
    const sample = sampleEdgeBand(layer002Info, validationInfo, start, end, settings);
    const targetCoverage = sideTargetCoverage(extents, side, offset, targetUv, settings);
    const targetBonus = targetCoverage.covered * 80 + targetCoverage.coverageRatio * 4000;
    const sampleBonus = sample.supported * 2 + sample.coverage * 300;
    const gapPenalty = sample.maxGap * 8 + sample.maxForbiddenGap * 12;
    candidates.push({ side, offset, sample, targetCoverage, score: targetBonus + sampleBonus + sample.supportPixelHits * 0.01 - gapPenalty });
  }
  candidates.sort((a, b) => b.score - a.score || a.offset - b.offset);
  return candidates.slice(0, 5);
}

function layer002TargetsForComponent(component, crop) {
  const targets = [];
  for (const index of component.pixels) {
    if (!crop.layer002Mask[index]) continue;
    targets.push({
      x: crop.offsetX + (index % crop.width) + 0.5,
      y: crop.offsetY + Math.floor(index / crop.width) + 0.5
    });
  }
  return targets;
}

function projectTargetsForExtents(targets, extents) {
  return targets.map((point) => uvForPoint(point, extents.cosTheta, extents.sinTheta));
}

function hullEdgeBandRadius(settings) {
  return Math.max(0.5, (Number(settings.edgeBandPx) || 0) + 0.5);
}

function sideTargetCoverage(extents, side, offset, targetUv, settings) {
  const radius = hullEdgeBandRadius(settings);
  const uMin = extents.uMin - radius;
  const uMax = extents.uMax + radius;
  const vMin = extents.vMin - radius;
  const vMax = extents.vMax + radius;
  const u = side === 'left' ? extents.uMin + offset : extents.uMax - offset;
  const v = side === 'top' ? extents.vMin + offset : extents.vMax - offset;
  let covered = 0;
  for (const target of targetUv) {
    if ((side === 'top' || side === 'bottom') && target.u >= uMin && target.u <= uMax && Math.abs(target.v - v) <= radius) {
      covered++;
    } else if ((side === 'left' || side === 'right') && target.v >= vMin && target.v <= vMax && Math.abs(target.u - u) <= radius) {
      covered++;
    }
  }
  return {
    covered,
    coverageRatio: targetUv.length ? covered / targetUv.length : 0
  };
}

function targetCoverageForQuad(extents, offsets, targetUv, settings) {
  const radius = hullEdgeBandRadius(settings);
  const uLeft = extents.uMin + offsets.left;
  const uRight = extents.uMax - offsets.right;
  const vTop = extents.vMin + offsets.top;
  const vBottom = extents.vMax - offsets.bottom;
  const edgeCounts = [0, 0, 0, 0];
  let covered = 0;
  for (const target of targetUv) {
    const inU = target.u >= uLeft - radius && target.u <= uRight + radius;
    const inV = target.v >= vTop - radius && target.v <= vBottom + radius;
    const top = inU && Math.abs(target.v - vTop) <= radius;
    const right = inV && Math.abs(target.u - uRight) <= radius;
    const bottom = inU && Math.abs(target.v - vBottom) <= radius;
    const left = inV && Math.abs(target.u - uLeft) <= radius;
    if (top) edgeCounts[0]++;
    if (right) edgeCounts[1]++;
    if (bottom) edgeCounts[2]++;
    if (left) edgeCounts[3]++;
    if (top || right || bottom || left) covered++;
  }
  const total = targetUv.length;
  const maxEdge = Math.max(1, ...edgeCounts);
  const minEdge = Math.min(...edgeCounts);
  return {
    covered,
    uncovered: Math.max(0, total - covered),
    total,
    coverageRatio: total ? covered / total : 0,
    uncoveredRatio: total ? Math.max(0, total - covered) / total : 1,
    edgeCounts,
    edgeBalance: minEdge / maxEdge
  };
}

function pointsForQuadOffsets(extents, offsets) {
  const uLeft = extents.uMin + offsets.left;
  const uRight = extents.uMax - offsets.right;
  const vTop = extents.vMin + offsets.top;
  const vBottom = extents.vMax - offsets.bottom;
  if (uRight <= uLeft + 1 || vBottom <= vTop + 1) return null;
  return [
    pointForUv(uLeft, vTop, extents.cosTheta, extents.sinTheta),
    pointForUv(uRight, vTop, extents.cosTheta, extents.sinTheta),
    pointForUv(uRight, vBottom, extents.cosTheta, extents.sinTheta),
    pointForUv(uLeft, vBottom, extents.cosTheta, extents.sinTheta)
  ];
}

function evaluateHullSweepQuad(extents, offsets, layer002Info, validationInfo, settings, meta, targetUv) {
  const points = pointsForQuadOffsets(extents, offsets);
  if (!points) return null;
  const edges = [
    sampleEdgeBand(layer002Info, validationInfo, points[0], points[1], settings),
    sampleEdgeBand(layer002Info, validationInfo, points[1], points[2], settings),
    sampleEdgeBand(layer002Info, validationInfo, points[2], points[3], settings),
    sampleEdgeBand(layer002Info, validationInfo, points[3], points[0], settings)
  ];
  const widths = [edges[0].length, edges[2].length];
  const heights = [edges[1].length, edges[3].length];
  const width = (widths[0] + widths[1]) / 2;
  const height = (heights[0] + heights[1]) / 2;
  const aspect = height > 0 ? width / height : Infinity;
  const aspectDistance = Number.isFinite(aspect) && aspect > 0 ? Math.max(aspect, 1 / aspect) - 1 : Infinity;
  const minCoverage = Math.min(...edges.map((edge) => edge.coverage));
  const maxGap = Math.max(...edges.map((edge) => edge.maxGap));
  const maxForbiddenGap = Math.max(...edges.map((edge) => edge.maxForbiddenGap));
  const totalSupported = edges.reduce((sum, edge) => sum + edge.supported, 0);
  const totalHits = edges.reduce((sum, edge) => sum + edge.supportPixelHits, 0);
  const targetCoverage = targetCoverageForQuad(extents, offsets, targetUv, settings);
  const supportValues = edges.map((edge) => edge.supported);
  const supportMax = Math.max(1, ...supportValues);
  const supportMin = Math.min(...supportValues);
  const supportBalance = supportMin / supportMax;
  const reasonCodes = [];
  if (minCoverage + 1e-9 < settings.minEdgeCoverage) reasonCodes.push('edge-coverage-low');
  if (maxGap > settings.maxUnsupportedGapPx) reasonCodes.push('unsupported-gap');
  if (maxForbiddenGap > settings.maxUnsupportedGapPx) reasonCodes.push('forbidden-gap');
  if (aspectDistance > settings.aspectTolerance) reasonCodes.push('aspect-mismatch');
  if (targetCoverage.covered <= 0) reasonCodes.push('no-002-edge-coverage');
  const lineGapPenalty = Math.max(0, maxGap - settings.maxUnsupportedGapPx) * 45
    + Math.max(0, maxForbiddenGap - settings.maxUnsupportedGapPx) * 65;
  const lineCoveragePenalty = Math.max(0, settings.minEdgeCoverage - minCoverage) * 2000;
  const score = targetCoverage.covered * 16
    + targetCoverage.coverageRatio * 30000
    + targetCoverage.edgeBalance * 1000
    + totalSupported * 2
    + totalHits * 0.015
    + minCoverage * 400
    + supportBalance * 250
    - targetCoverage.uncovered * 7
    - targetCoverage.uncoveredRatio * 7000
    - lineCoveragePenalty
    - lineGapPenalty
    - Math.max(0, aspectDistance - settings.aspectTolerance) * 600;
  const accepted = targetCoverage.covered > 0 && aspectDistance <= settings.aspectTolerance;
  return {
    ...meta,
    pointsPx: points.map((point) => [point.x, point.y]),
    edges,
    offsets,
    angleDeg: extents.angleDeg,
    pixelCount: meta.pixelCount,
    layer002Count: meta.layer002Count,
    minCoverage,
    maxGap,
    maxForbiddenGap,
    aspect,
    aspectDistance,
    supportBalance,
    totalSupported,
    targetCovered: targetCoverage.covered,
    targetTotal: targetCoverage.total,
    targetCoverageRatio: targetCoverage.coverageRatio,
    targetUncovered: targetCoverage.uncovered,
    targetUncoveredRatio: targetCoverage.uncoveredRatio,
    targetEdgeCounts: targetCoverage.edgeCounts,
    targetEdgeBalance: targetCoverage.edgeBalance,
    score,
    accepted,
    quadClosed: true,
    reasonCodes
  };
}

function bestHullSweepForComponent(component, crop, layer002Info, validationInfo, settings, bbox, componentIndex) {
  const targetPoints = layer002TargetsForComponent(component, crop);
  if (targetPoints.length < settings.minPixels) return { skipped: true, reason: 'too-few-layer-002-targets' };
  const boundary = boundaryPointsForComponent(component, crop.width, crop.height, crop.offsetX, crop.offsetY);
  const hull = convexHull(boundary.length >= 4 ? boundary : component.pixels.map((index) => ({
    x: crop.offsetX + (index % crop.width) + 0.5,
    y: crop.offsetY + Math.floor(index / crop.width) + 0.5
  })));
  if (hull.length < 4) return { skipped: true, reason: 'too-few-hull-points' };
  const baseAngle = baseHullAngle(hull);
  const angles = hullSweepAngles(baseAngle, settings.angleSweepDeg, settings.angleStepDeg);
  const meta = {
    bboxId: String(bbox.bboxId || ''),
    componentIndex,
    pixelCount: component.pixelCount,
    layer002Count: component.layer002Count,
    targetPixelCount: targetPoints.length,
    baseAngleDeg: baseAngle,
    hullPointCount: hull.length
  };
  let bestAccepted = null;
  let bestRejected = null;
  for (const angle of angles) {
    const extents = hullExtentsForAngle(hull, angle);
    if (extents.width < 3 || extents.height < 3) continue;
    const targetUv = projectTargetsForExtents(targetPoints, extents);
    const top = sideOffsetCandidates(extents, 'top', layer002Info, validationInfo, settings, targetUv);
    const right = sideOffsetCandidates(extents, 'right', layer002Info, validationInfo, settings, targetUv);
    const bottom = sideOffsetCandidates(extents, 'bottom', layer002Info, validationInfo, settings, targetUv);
    const left = sideOffsetCandidates(extents, 'left', layer002Info, validationInfo, settings, targetUv);
    for (const topCandidate of top) {
      for (const rightCandidate of right) {
        for (const bottomCandidate of bottom) {
          for (const leftCandidate of left) {
            const candidate = evaluateHullSweepQuad(extents, {
              top: topCandidate.offset,
              right: rightCandidate.offset,
              bottom: bottomCandidate.offset,
              left: leftCandidate.offset
            }, layer002Info, validationInfo, settings, meta, targetUv);
            if (!candidate) continue;
            if (candidate.accepted) {
              if (!bestAccepted || candidate.score > bestAccepted.score) bestAccepted = candidate;
            } else if (!bestRejected || candidate.score > bestRejected.score) {
              bestRejected = candidate;
            }
          }
        }
      }
    }
  }
  return { accepted: bestAccepted, rejected: bestAccepted ? null : bestRejected, skipped: false };
}

function sweepLayer002Bbox(layer002Info, hullInfo, validationInfo, bbox, settings) {
  const bounds = bboxPixelBounds(bbox, layer002Info.width, layer002Info.height);
  if (!bounds) return { accepted: [], rejected: [], skipped: 1, components: 0, angles: 0, quadCount: 0 };
  const crop = cropHullMasks(hullInfo, layer002Info, bounds);
  if (crop.layer002Count < settings.minPixels) {
    return { accepted: [], rejected: [], skipped: 1, components: 0, angles: 0, quadCount: 0 };
  }
  const components = connectedMaskComponents(crop.hullMask, crop.width, crop.height, crop.layer002Mask, 8)
    .filter((component) => component.layer002Count >= settings.minPixels);
  const accepted = [];
  const rejected = [];
  let skipped = 0;
  let angleCount = 0;
  for (let componentIndex = 0; componentIndex < components.length; componentIndex++) {
    const fit = bestHullSweepForComponent(components[componentIndex], crop, layer002Info, validationInfo, settings, bbox, componentIndex);
    if (fit.skipped) {
      skipped++;
      continue;
    }
    if (fit.accepted) accepted.push(fit.accepted);
    else if (fit.rejected) rejected.push(fit.rejected);
    angleCount += hullSweepAngles(fit.accepted?.baseAngleDeg ?? fit.rejected?.baseAngleDeg ?? 0, settings.angleSweepDeg, settings.angleStepDeg).length;
  }
  if (!accepted.length && !rejected.length) {
    rejected.push({
      bboxId: String(bbox.bboxId || ''),
      pixelCount: crop.hullCount,
      layer002Count: crop.layer002Count,
      pointsPx: [],
      edges: [],
      minCoverage: 0,
      maxGap: 0,
      maxForbiddenGap: 0,
      score: 0,
      accepted: false,
      quadClosed: false,
      reasonCodes: ['no-hull-sweep-candidate']
    });
  }
  return { accepted, rejected, skipped, components: components.length, angles: angleCount, quadCount: accepted.length };
}

function boundsForPointList(points) {
  if (!validPointList(points)) return null;
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const point of points) {
    const x = Number(point[0]);
    const y = Number(point[1]);
    x0 = Math.min(x0, x);
    y0 = Math.min(y0, y);
    x1 = Math.max(x1, x);
    y1 = Math.max(y1, y);
  }
  return { x0, y0, x1, y1, area: Math.max(0, x1 - x0) * Math.max(0, y1 - y0) };
}

function overlapRatioForBounds(a, b) {
  if (!a || !b || a.area <= 0 || b.area <= 0) return 0;
  const width = Math.max(0, Math.min(a.x1, b.x1) - Math.max(a.x0, b.x0));
  const height = Math.max(0, Math.min(a.y1, b.y1) - Math.max(a.y0, b.y0));
  const intersection = width * height;
  return intersection / Math.max(1, Math.min(a.area, b.area));
}

function suppressOverlappingQuads(candidates, limit) {
  const kept = [];
  const keptBounds = [];
  for (const candidate of candidates) {
    const bounds = boundsForPointList(candidate.pointsPx);
    if (!bounds) continue;
    const duplicate = keptBounds.some((existing) => overlapRatioForBounds(bounds, existing) > 0.75);
    if (duplicate) continue;
    kept.push(candidate);
    keptBounds.push(bounds);
    if (kept.length >= limit) break;
  }
  return kept;
}

function layer002QuadResults(settings = layer002QuadState()) {
  const info = activeBboxInfo();
  const frame = bboxFrameEntry();
  if (!settings.show || !info || !info.available || info.stale || !state.bbox.manifest || !frame) {
    return { accepted: [], rejected: [], skipped: 0, components: 0, angles: 0, quadCount: 0 };
  }
  const layer002Info = layerMaskForPrefix(settings.prefix || '002');
  if (!layer002Info) return { accepted: [], rejected: [], skipped: 0, components: 0, angles: 0, quadCount: 0 };
  const anyMaskInfo = decodedMaskInfo() || layer002Info;
  const hullInfo = settings.supportMode === '002-only' ? layer002Info : anyMaskInfo;
  const validationInfo = settings.forbiddenMode === 'non-mask' ? anyMaskInfo : layer002Info;
  const accepted = [];
  const rejected = [];
  let skipped = 0;
  let components = 0;
  let angles = 0;
  const bboxes = Array.isArray(frame.bboxes) ? frame.bboxes : [];
  for (const bbox of bboxes) {
    const result = sweepLayer002Bbox(layer002Info, hullInfo, validationInfo, bbox, settings);
    accepted.push(...result.accepted);
    rejected.push(...result.rejected);
    skipped += result.skipped;
    components += result.components;
    angles += result.angles;
  }
  accepted.sort((a, b) => (b.targetCoverageRatio || 0) - (a.targetCoverageRatio || 0) || b.score - a.score || b.layer002Count - a.layer002Count || String(a.bboxId).localeCompare(String(b.bboxId)));
  rejected.sort((a, b) => (b.targetCoverageRatio || 0) - (a.targetCoverageRatio || 0) || b.score - a.score || b.layer002Count - a.layer002Count || String(a.bboxId).localeCompare(String(b.bboxId)));
  const acceptedLimited = suppressOverlappingQuads(accepted, settings.maxQuadsPerFrame);
  return {
    accepted: acceptedLimited,
    rejected: rejected.slice(0, settings.maxQuadsPerFrame),
    skipped,
    components,
    angles,
    quadCount: acceptedLimited.length
  };
}

function renderFovClipFrameMargin(svg, namespace, settings) {
  const clip = settings.fovClip || {};
  if (!clip.enabled || !clip.showOverlay) return;
  const margin = Math.max(0, Number(clip.marginPx) || 0);
  if (margin <= 0) return;
  const width = state.viewport.frameWidth || 640;
  const height = state.viewport.frameHeight || 360;
  const rects = [
    [0, 0, margin, height],
    [Math.max(0, width - margin), 0, width, height],
    [0, 0, width, margin],
    [0, Math.max(0, height - margin), width, height]
  ];
  for (const [x, y, w, h] of rects) {
    if (w <= 0 || h <= 0) continue;
    const rect = document.createElementNS(namespace, 'rect');
    rect.setAttribute('class', 'fovClipMargin');
    rect.setAttribute('x', String(x));
    rect.setAttribute('y', String(y));
    rect.setAttribute('width', String(w));
    rect.setAttribute('height', String(h));
    svg.appendChild(rect);
  }
}

function fovSideLine(rectValues, side) {
  const [x0, y0, x1, y1] = rectValues;
  if (side === 'left') return [x0 + 0.5, y0 + 0.5, x0 + 0.5, y1 + 0.5];
  if (side === 'right') return [x1 + 0.5, y0 + 0.5, x1 + 0.5, y1 + 0.5];
  if (side === 'top') return [x0 + 0.5, y0 + 0.5, x1 + 0.5, y0 + 0.5];
  return [x0 + 0.5, y1 + 0.5, x1 + 0.5, y1 + 0.5];
}

function renderFovClipMark(svg, namespace, bbox, rectValues, settings) {
  const clipSettings = settings.fovClip || {};
  const fov = bbox?.fovClip && typeof bbox.fovClip === 'object' ? bbox.fovClip : null;
  if (!clipSettings.enabled || !clipSettings.showOverlay || !fov || !rectValues) return;
  const clippedSides = Array.isArray(fov.sides) ? fov.sides : [];
  const nearSides = Array.isArray(fov.nearSides) ? fov.nearSides : [];
  const sides = clippedSides.length ? clippedSides : nearSides;
  if (!sides.length) return;
  for (const side of clippedSides) {
    appendSvgLine(svg, namespace, ...fovSideLine(rectValues, side), 'fovClipSide clipped');
  }
  for (const side of nearSides) {
    if (clippedSides.includes(side)) continue;
    appendSvgLine(svg, namespace, ...fovSideLine(rectValues, side), 'fovClipSide near');
  }
  const [x0, y0, x1] = rectValues;
  const text = document.createElementNS(namespace, 'text');
  text.setAttribute('class', `fovClipLabel ${clippedSides.length ? 'clipped' : 'near'}`);
  text.setAttribute('x', String(Math.min(Math.max(2, x0 + 2), Math.max(2, (state.viewport.frameWidth || 640) - 82))));
  text.setAttribute('y', String(Math.max(10, y0 + 10)));
  const sideText = sides.map((side) => side[0].toUpperCase()).join('/');
  const severity = Math.round(Number(fov.severity || 0) * 100);
  text.textContent = `${clippedSides.length ? 'CLIP' : 'EDGE'} ${sideText} ${severity}%`;
  if (x1 - x0 > 10) svg.appendChild(text);
}

function instanceObservationCenter(observation) {
  const point = Array.isArray(observation?.centroidPx) ? observation.centroidPx.map(Number) : null;
  if (point && point.length >= 2 && point.every(Number.isFinite)) return { x: point[0], y: point[1] };
  const bbox = Array.isArray(observation?.bboxPx) ? observation.bboxPx.map(Number) : null;
  if (bbox && bbox.length === 4 && bbox.every(Number.isFinite)) return { x: (bbox[0] + bbox[2]) / 2, y: (bbox[1] + bbox[3]) / 2 };
  return null;
}

function previousInstanceObservation(instanceId, frameOrdinal = state.frameIndex) {
  const frames = Array.isArray(state.instances.manifest?.frames) ? state.instances.manifest.frames : [];
  for (let index = frames.length - 1; index >= 0; index--) {
    const frame = frames[index];
    if (Number(frame.frameOrdinal) >= Number(frameOrdinal)) continue;
    const match = (frame.observations || []).find((observation) => observation.instanceId === instanceId);
    if (match) return match;
  }
  return null;
}

function recentInstanceObservations(instanceId, frameOrdinal, trailLength) {
  const frames = Array.isArray(state.instances.manifest?.frames) ? state.instances.manifest.frames : [];
  const low = Number(frameOrdinal) - Math.max(1, Number(trailLength) || 1);
  const points = [];
  for (const frame of frames) {
    const current = Number(frame.frameOrdinal);
    if (current < low || current > Number(frameOrdinal)) continue;
    const match = (frame.observations || []).find((observation) => observation.instanceId === instanceId);
    const center = instanceObservationCenter(match);
    if (center) points.push({ ...center, frameOrdinal: current });
  }
  return points;
}

function instanceDisplayId(observation) {
  const ordinal = Number(observation?.instanceOrdinal);
  if (Number.isFinite(ordinal) && ordinal > 0) return `I${String(Math.round(ordinal)).padStart(3, '0')}`;
  const id = String(observation?.instanceId || 'instance');
  const match = id.match(/(\d+)$/);
  return match ? `I${match[1].slice(-3)}` : id;
}

function appendInstancePolyline(svg, namespace, points, className, color) {
  if (!Array.isArray(points) || points.length < 2) return null;
  const polyline = document.createElementNS(namespace, 'polyline');
  polyline.setAttribute('class', className);
  polyline.setAttribute('points', points.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' '));
  polyline.setAttribute('stroke', color);
  svg.appendChild(polyline);
  return polyline;
}

function renderInstanceMarks(svg, namespace) {
  const settings = instanceTrackingState();
  const info = activeInstanceInfo();
  if (!settings.enabled || !settings.showOverlay || !info || !info.available || info.stale || !state.instances.manifest) {
    renderInstanceReadout();
    return;
  }
  const frame = instanceFrameEntry();
  const observations = Array.isArray(frame?.observations) ? frame.observations : [];
  for (const observation of observations) {
    const center = instanceObservationCenter(observation);
    if (!center) continue;
    const color = observation.instanceColor || '#55f7ff';
    const trail = recentInstanceObservations(observation.instanceId, observation.frameOrdinal, settings.trailLengthFrames);
    if (trail.length > 1) appendInstancePolyline(svg, namespace, trail, 'instanceTrail', color);
    const previous = previousInstanceObservation(observation.instanceId, observation.frameOrdinal);
    const previousCenter = instanceObservationCenter(previous);
    if (settings.showLinks && previousCenter) {
      appendSvgLine(svg, namespace, previousCenter.x, previousCenter.y, center.x, center.y, 'instanceLink')?.setAttribute('stroke', color);
    }
    if (settings.showCandidates) {
      for (const candidate of observation.candidates || []) {
        if (candidate.instanceId === observation.instanceId) continue;
        const candidatePrevious = previousInstanceObservation(candidate.instanceId, observation.frameOrdinal);
        const candidateCenter = instanceObservationCenter(candidatePrevious);
        if (!candidateCenter) continue;
        appendSvgLine(svg, namespace, candidateCenter.x, candidateCenter.y, center.x, center.y, 'instanceCandidateLink')?.setAttribute('stroke', color);
      }
    }
    const dot = document.createElementNS(namespace, 'circle');
    dot.setAttribute('class', 'instancePoint');
    dot.setAttribute('cx', center.x.toFixed(2));
    dot.setAttribute('cy', center.y.toFixed(2));
    dot.setAttribute('r', observation.status === 'new' ? '2.5' : '2');
    dot.setAttribute('fill', color);
    svg.appendChild(dot);
    if (settings.showLabels) {
      const text = document.createElementNS(namespace, 'text');
      const statusClass = observation.status === 'split-candidate' ? 'split' : observation.mergeCandidateFrom ? 'merge' : observation.status === 'new' ? 'new' : 'matched';
      text.setAttribute('class', `instanceLabel ${statusClass}`);
      text.setAttribute('x', String(center.x + 4));
      text.setAttribute('y', String(center.y - 4));
      text.setAttribute('fill', color);
      const score = Number(observation.associationScore || 0);
      text.textContent = `${instanceDisplayId(observation)} ${observation.status === 'new' ? 'new' : fmt(score, 2)}`;
      svg.appendChild(text);
    }
  }
  renderInstanceReadout();
}

function compactObservationForReadout(observation) {
  const pose = observation.pose || {};
  const clip = observation.fovClip || {};
  return {
    id: observation.instanceId,
    bbox: observation.bboxId,
    status: observation.status,
    score: Number(observation.associationScore || 0),
    xyz: pose.available ? pose.xyzCameraM : null,
    rpy: pose.available ? pose.rpyCameraDeg : null,
    clip: clip.status && clip.status !== 'clear' ? { status: clip.status, sides: clip.sides || clip.nearSides || [], severity: clip.severity } : null,
    candidates: (observation.candidates || []).slice(0, 3).map((candidate) => ({
      id: candidate.instanceId,
      score: candidate.score,
      signals: candidate.signals
    }))
  };
}

function renderInstanceReadout() {
  const settings = instanceTrackingState();
  const setReadoutText = (text, options = {}) => {
    if (els.instanceTrackingJsonScaffold) els.instanceTrackingJsonScaffold.textContent = text;
    if (els.instanceReadout) {
      els.instanceReadout.textContent = options.forcePanel || settings.showReadout ? text : 'Instance tracking hidden.';
    }
  };
  const info = activeInstanceInfo();
  if (!info || !info.available || !state.instances.manifest) {
    setReadoutText(info?.stale ? `Instance tracking stale:\n${(info.staleReasons || []).join('\n')}` : 'No instance tracking manifest.', { forcePanel: true });
    return;
  }
  const frameIndex = currentReviewFrameIndex();
  const frame = instanceFrameEntry(frameIndex);
  const observations = Array.isArray(frame?.observations) ? frame.observations : [];
  const payload = {
    frame: frameIndex,
    frameDisplay: frameIndex + 1,
    frameOrdinal: frame?.frameOrdinal ?? frameIndex,
    stale: Boolean(info.stale),
    staleReasons: info.stale ? info.staleReasons || [] : undefined,
    count: observations.length,
    instances: observations.map(compactObservationForReadout)
  };
  setReadoutText(JSON.stringify(payload, null, 2));
}

function drawBboxList(svg, namespace, bboxes, settings, options = {}) {
  if (!bboxes.length) return;
  const maxLabels = Number(options.maxLabels ?? 20);
  const suffix = options.labelSuffix || '';
  const classSuffix = options.classSuffix || '';
  for (let index = 0; index < bboxes.length; index++) {
    const bbox = bboxes[index];
    const rectValues = Array.isArray(bbox.bboxPx) ? bbox.bboxPx.map(Number) : null;
    if (settings.showRawBboxes) appendSvgRect(svg, namespace, rectValues, `bboxRect${classSuffix}`);
    if (options.showFov !== false) renderFovClipMark(svg, namespace, bbox, rectValues, settings);
    const quadFit = bbox.quadFit && typeof bbox.quadFit === 'object' ? bbox.quadFit : null;
    const quadPoints = validPointList(quadFit?.pointsPx) ? quadFit.pointsPx : null;
    if (quadPoints) {
      const statusClass = quadFit.accepted ? 'accepted' : 'rejected';
      const band = appendSvgPolygon(svg, namespace, quadPoints, `quadBand ${statusClass}${classSuffix}`);
      if (band) band.setAttribute('stroke-width', String(settings.quadThicknessPx));
      const line = appendSvgPolygon(svg, namespace, quadPoints, `quadLine ${statusClass}${classSuffix}`);
      if (line) line.setAttribute('opacity', String(settings.quadOverlayOpacity));
    }
    const children = Array.isArray(bbox.overlapChildren) ? bbox.overlapChildren : [];
    for (const child of children) {
      if (validPointList(child.quadPointsPx)) {
        appendSvgPolygon(svg, namespace, child.quadPointsPx, `overlapQuad${classSuffix}`);
      } else {
        appendSvgRect(svg, namespace, child.bboxPx, `overlapRect${classSuffix}`);
      }
    }
    if (index < maxLabels) {
      if (!rectValues || rectValues.length !== 4) continue;
      const [x0, y0] = rectValues;
      const text = document.createElementNS(namespace, 'text');
      text.setAttribute('class', `bboxLabel${classSuffix}`);
      text.setAttribute('x', String(x0 + 2));
      text.setAttribute('y', String(Math.max(10, y0 - 3)));
      const marker = quadFit?.hasVoidOverlap ? ' +void' : quadFit?.accepted ? ' quad' : quadFit?.enabled ? ' weak' : '';
      text.textContent = `${suffix}${index + 1} ${Number(bbox.pixelCount || 0).toLocaleString('en-US')}px${marker}`;
      svg.appendChild(text);
    }
  }
}

function renderBboxMarks(svg, namespace) {
  const settings = bboxFlowState();
  const info = activeBboxInfo();
  const maskbitsInfo = activeMaskbitsBboxInfo();
  const baselineReady = info && info.available && !info.stale && state.bbox.manifest;
  const maskbitsReady = maskbitsInfo && maskbitsInfo.available && !maskbitsInfo.stale && state.maskbitsBbox.manifest;
  const baselineFrame = baselineReady ? bboxFrameEntry() : null;
  const maskbitsFrame = maskbitsReady ? maskbitsBboxFrameEntry() : null;
  const baselineBboxes = Array.isArray(baselineFrame?.bboxes) ? baselineFrame.bboxes : [];
  const maskbitsBboxes = Array.isArray(maskbitsFrame?.bboxes) ? maskbitsFrame.bboxes : [];
  if (settings.viewMode === 'maskbits') {
    drawBboxList(svg, namespace, maskbitsBboxes, settings, { labelSuffix: 'm', classSuffix: ' maskbits' });
    return;
  }
  if (settings.viewMode === 'compare') {
    drawBboxList(svg, namespace, baselineBboxes, settings, { maxLabels: 12 });
    drawBboxList(svg, namespace, maskbitsBboxes, settings, { maxLabels: 12, labelSuffix: 'm', classSuffix: ' maskbits', showFov: false });
    return;
  }
  drawBboxList(svg, namespace, baselineBboxes, settings);
}

function renderBboxClippingMarks(svg, namespace) {
  const info = activeBboxClippingInfo();
  if (!info || !info.available || info.stale || !state.bboxClipping.manifest) return;
  const settings = bboxFlowState();
  const clip = settings.fovClip || {};
  if (!clip.enabled || !clip.showOverlay) return;
  const frame = bboxClippingFrameEntry();
  const objects = Array.isArray(frame?.objects) ? frame.objects : [];
  for (const object of objects) {
    const rectValues = Array.isArray(object.bboxPx) ? object.bboxPx.map(Number) : null;
    renderFovClipMark(svg, namespace, { fovClip: object.fovClip }, rectValues, settings);
  }
}

function contourLabelText(feature, fallback) {
  const role = String(feature?.role || fallback || 'contour');
  if (role === 'outer') return 'outer';
  if (role === 'void') return 'void';
  if (role === 'nested-island') return 'nested';
  if (role === 'notch-candidate') return 'notch';
  return role;
}

function appendContourLabel(svg, namespace, feature, text, settings) {
  if (!settings.showLabels) return;
  const center = Array.isArray(feature?.centerPx) ? feature.centerPx.map(Number) : null;
  if (!center || center.length < 2 || !center.every(Number.isFinite)) return;
  const label = document.createElementNS(namespace, 'text');
  label.setAttribute('class', 'contourLabel');
  label.setAttribute('x', String(center[0] + 2));
  label.setAttribute('y', String(center[1] - 2));
  label.setAttribute('opacity', String(settings.opacity));
  label.textContent = text;
  svg.appendChild(label);
}

function appendContourFeature(svg, namespace, feature, className, settings, labelText) {
  const points = contourDisplayPoints(feature);
  if (!feature || !points) return;
  const path = appendSvgPath(svg, namespace, points, className, true);
  if (!path) return;
  path.setAttribute('stroke-width', String(settings.lineThicknessPx));
  path.setAttribute('opacity', String(settings.opacity));
  path.setAttribute('data-contour-id', String(feature.contourId || ''));
  path.setAttribute('data-bbox-id', String(feature.bboxId || ''));
  path.setAttribute('data-role', String(feature.role || ''));
  appendContourLabel(svg, namespace, feature, labelText || contourLabelText(feature), settings);
}

function renderContourMarks(svg, namespace) {
  const settings = contourHierarchyState();
  if (!settings.show) {
    renderContourControls({ objectCount: 0, contourCount: 0, voidCount: 0, nestedIslandCount: 0, notchCandidateCount: 0, overlapCandidateCount: 0 });
    return;
  }
  const info = activeContourInfo();
  if (!info || !info.available || info.stale || !state.contour.manifest) {
    renderContourControls();
    return;
  }
  const frame = contourFrameEntry();
  const objects = Array.isArray(frame?.objects) ? frame.objects : [];
  const summary = { objectCount: objects.length, contourCount: 0, voidCount: 0, nestedIslandCount: 0, notchCandidateCount: 0, overlapCandidateCount: 0 };
  let labels = 0;
  const drawFeature = (feature, className, label) => {
    appendContourFeature(svg, namespace, feature, className, { ...settings, showLabels: settings.showLabels && labels < 80 }, label);
    labels += 1;
  };
  for (const object of objects) {
    if (object.outer) {
      drawFeature(object.outer, 'contourPath outer', 'outer');
    }
    for (const outer of Array.isArray(object.additionalOuters) ? object.additionalOuters : []) {
      drawFeature(outer, 'contourPath outer secondary', 'outer');
    }
    const voids = Array.isArray(object.voids) ? object.voids : [];
    for (let index = 0; index < voids.length; index++) {
      drawFeature(voids[index], 'contourPath void', `void ${index + 1}`);
    }
    const nested = Array.isArray(object.nestedIslands) ? object.nestedIslands : [];
    for (let index = 0; index < nested.length; index++) {
      drawFeature(nested[index], 'contourPath nested', `nested ${index + 1}`);
    }
    const notches = Array.isArray(object.notchCandidates) ? object.notchCandidates : [];
    summary.contourCount += Number(object.contourCount || 0);
    summary.voidCount += voids.length;
    summary.nestedIslandCount += nested.length;
    summary.notchCandidateCount += notches.length;
    summary.overlapCandidateCount += Array.isArray(object.overlapCandidates) ? object.overlapCandidates.length : 0;
  }
  renderContourControls(summary);
}

function renderBboxContourMarks(svg, namespace) {
  const settings = contourHierarchyState();
  if (!settings.show) return;
  const info = activeBboxContoursInfo();
  if (!info || !info.available || info.stale || !state.bboxContours.manifest) return;
  const frame = bboxContoursFrameEntry();
  const objects = Array.isArray(frame?.objects) ? frame.objects : [];
  let labels = 0;
  const drawFeature = (feature, className, label) => {
    appendContourFeature(svg, namespace, feature, className, { ...settings, showLabels: settings.showLabels && labels < 80 }, label);
    labels += 1;
  };
  for (const object of objects) {
    if (object.outer) drawFeature(object.outer, 'contourPath outer', 'bbox outer');
    for (const outer of Array.isArray(object.additionalOuters) ? object.additionalOuters : []) {
      drawFeature(outer, 'contourPath outer secondary', 'bbox outer');
    }
    const voids = Array.isArray(object.voids) ? object.voids : [];
    for (let index = 0; index < voids.length; index++) {
      drawFeature(voids[index], 'contourPath void', `bbox void ${index + 1}`);
    }
    const nested = Array.isArray(object.nestedIslands) ? object.nestedIslands : [];
    for (let index = 0; index < nested.length; index++) {
      drawFeature(nested[index], 'contourPath nested', `bbox nested ${index + 1}`);
    }
  }
}

function renderCornerMarks(svg, namespace) {
  const settings = cornerFlowState();
  const info = activeCornerInfo();
  if (!info || !info.available || info.stale || !state.corner.manifest) return;
  const frame = cornerFrameEntry();
  const corners = flattenedCornerFrameMarks(frame);
  for (const corner of corners) {
    appendCornerMark(svg, namespace, corner, settings);
  }
}

function renderCornerAgreementMarks(svg, namespace) {
  const settings = cornerAgreementState();
  if (!settings.show) {
    renderCornerAgreementControls({ links: [], completeGroupCount: 0 });
    return;
  }
  const info = activeCornerInfo();
  if (!info || !info.available || info.stale || !state.corner.manifest) {
    renderCornerAgreementControls({ links: [], completeGroupCount: 0 });
    return;
  }
  const result = cornerAgreementResults(settings);
  for (const link of result.links) {
    const line = document.createElementNS(namespace, 'line');
    line.setAttribute('class', `cornerAgreementLine${link.complete ? ' complete' : ''}`);
    line.setAttribute('x1', String(link.whitePoint.x));
    line.setAttribute('y1', String(link.whitePoint.y));
    line.setAttribute('x2', String(link.greenPoint.x));
    line.setAttribute('y2', String(link.greenPoint.y));
    line.setAttribute('stroke-width', String(settings.lineThicknessPx));
    line.setAttribute('opacity', String(settings.opacity));
    line.setAttribute('data-score', fmt(link.score, 3));
    svg.appendChild(line);
  }
  renderCornerAgreementControls(result);
}

function renderLayer002QuadMarks(svg, namespace) {
  const settings = layer002QuadState();
  if (!settings.show) {
    renderLayer002QuadControls({ accepted: [], rejected: [], skipped: 0, components: 0, angles: 0, quadCount: 0 });
    return;
  }
  const result = layer002QuadResults(settings);
  const drawCandidate = (candidate, accepted) => {
    if (!validPointList(candidate.pointsPx)) return;
    const polygon = appendSvgPolygon(svg, namespace, candidate.pointsPx, `layer002Quad ${accepted ? 'accepted' : 'rejected'} closed`);
    if (!polygon) return;
    polygon.setAttribute('stroke-width', String(settings.lineThicknessPx));
    polygon.setAttribute('opacity', String(settings.opacity));
    polygon.setAttribute('data-bbox-id', String(candidate.bboxId || ''));
    polygon.setAttribute('data-score', fmt(candidate.score, 3));
    polygon.setAttribute('data-min-coverage', fmt(candidate.minCoverage, 3));
    polygon.setAttribute('data-target-coverage', fmt(candidate.targetCoverageRatio || 0, 3));
    polygon.setAttribute('data-target-covered', String(candidate.targetCovered || 0));
    polygon.setAttribute('data-target-total', String(candidate.targetTotal || 0));
    polygon.setAttribute('data-max-gap', String(candidate.maxGap || 0));
    polygon.setAttribute('data-angle', fmt(candidate.angleDeg, 2));
    for (const point of candidate.pointsPx) {
      const dot = document.createElementNS(namespace, 'circle');
      dot.setAttribute('class', `layer002QuadCorner ${accepted ? 'accepted' : 'rejected'}`);
      dot.setAttribute('cx', String(point[0] ?? point.x));
      dot.setAttribute('cy', String(point[1] ?? point.y));
      dot.setAttribute('r', String(Math.max(2, settings.lineThicknessPx * 1.35)));
      dot.setAttribute('opacity', String(settings.opacity));
      svg.appendChild(dot);
    }
  };
  for (const candidate of result.accepted) drawCandidate(candidate, true);
  if (settings.showRejected) {
    for (const candidate of result.rejected) drawCandidate(candidate, false);
  }
  renderLayer002QuadControls(result);
}

function renderBboxOverlay() {
  const svg = els.bboxOverlay;
  if (!svg) return;
  svg.replaceChildren();
  svg.setAttribute('viewBox', `0 0 ${state.viewport.frameWidth} ${state.viewport.frameHeight}`);
  const namespace = 'http://www.w3.org/2000/svg';
  renderFovClipFrameMargin(svg, namespace, bboxFlowState());
  renderBboxMarks(svg, namespace);
  renderBboxClippingMarks(svg, namespace);
  renderContourMarks(svg, namespace);
  renderBboxContourMarks(svg, namespace);
  renderCornerMarks(svg, namespace);
  renderCornerAgreementMarks(svg, namespace);
  renderLayer002QuadMarks(svg, namespace);
  renderInstanceMarks(svg, namespace);
}

function renderBboxControls() {
  const settings = bboxFlowState();
  const info = activeBboxInfo();
  const maskbitsInfo = activeMaskbitsBboxInfo();
  const dependency = info?.dependency || state.dependencyStatus?.maskBboxes || {};
  const job = state.bbox.discovery?.job || null;
  const maskbitsJob = state.maskbitsBbox.discovery?.job || null;
  const progress = job?.progress || {};
  const running = job && ['queued', 'running'].includes(job.status);
  const maskbitsProgress = maskbitsJob?.progress || {};
  const maskbitsRunning = maskbitsJob && ['queued', 'running'].includes(maskbitsJob.status);
  if (els.bboxMinPixelsInput) els.bboxMinPixelsInput.value = String(settings.minPixels);
  if (els.bboxMinPixelsValue) els.bboxMinPixelsValue.textContent = String(settings.minPixels);
  if (els.bboxMaxPerFrameInput) els.bboxMaxPerFrameInput.value = String(settings.maxBboxesPerFrame);
  if (els.bboxMaxPerFrameNumber) els.bboxMaxPerFrameNumber.value = String(settings.maxBboxesPerFrame);
  if (els.bboxMaxPerFrameValue) els.bboxMaxPerFrameValue.textContent = String(settings.maxBboxesPerFrame);
  if (els.bboxTightnessInput) els.bboxTightnessInput.value = String(settings.fitTightness);
  if (els.bboxTightnessValue) els.bboxTightnessValue.textContent = String(settings.fitTightness);
  if (els.bboxQuadToggle) els.bboxQuadToggle.checked = Boolean(settings.quadFitEnabled);
  if (els.bboxFitModeSelect) els.bboxFitModeSelect.value = settings.quadFitMode;
  if (els.bboxTargetAspectInput) els.bboxTargetAspectInput.value = String(settings.targetAspect);
  if (els.bboxTargetAspectValue) els.bboxTargetAspectValue.textContent = fmt(settings.targetAspect, 2);
  if (els.bboxAspectToleranceInput) els.bboxAspectToleranceInput.value = String(settings.aspectTolerance);
  if (els.bboxAspectToleranceValue) els.bboxAspectToleranceValue.textContent = fmt(settings.aspectTolerance, 2);
  if (els.bboxQuadThicknessInput) els.bboxQuadThicknessInput.value = String(settings.quadThicknessPx);
  if (els.bboxQuadThicknessValue) els.bboxQuadThicknessValue.textContent = `${settings.quadThicknessPx}px`;
  if (els.bboxEdgeCoverageInput) els.bboxEdgeCoverageInput.value = String(settings.edgeCoverageMin);
  if (els.bboxEdgeCoverageValue) els.bboxEdgeCoverageValue.textContent = `${fmt(settings.edgeCoverageMin * 100, 0)}%`;
  if (els.bboxCornerMinPixelsInput) els.bboxCornerMinPixelsInput.value = String(settings.cornerMinPixels);
  if (els.bboxCornerMinPixelsValue) els.bboxCornerMinPixelsValue.textContent = `${settings.cornerMinPixels}px`;
  if (els.bboxVoidOverlapInput) els.bboxVoidOverlapInput.value = String(settings.voidOverlapMaxRatio);
  if (els.bboxVoidOverlapValue) els.bboxVoidOverlapValue.textContent = `${fmt(settings.voidOverlapMaxRatio * 100, 1)}%`;
  if (els.bboxVoidMinPixelsInput) els.bboxVoidMinPixelsInput.value = String(settings.voidOverlapMinPixels);
  if (els.bboxVoidMinPixelsValue) els.bboxVoidMinPixelsValue.textContent = `${settings.voidOverlapMinPixels}px`;
  if (els.bboxShowRawToggle) els.bboxShowRawToggle.checked = Boolean(settings.showRawBboxes);
  if (els.bboxQuadOpacityInput) els.bboxQuadOpacityInput.value = String(Math.round(settings.quadOverlayOpacity * 100));
  if (els.bboxQuadOpacityValue) els.bboxQuadOpacityValue.textContent = `${Math.round(settings.quadOverlayOpacity * 100)}%`;
  if (els.bboxViewModeSelect) els.bboxViewModeSelect.value = settings.viewMode;
  const clip = settings.fovClip || {};
  if (els.fovClipToggle) els.fovClipToggle.checked = Boolean(clip.enabled);
  if (els.fovClipOverlayToggle) els.fovClipOverlayToggle.checked = Boolean(clip.showOverlay);
  if (els.fovClipMarginInput) els.fovClipMarginInput.value = String(clip.marginPx);
  if (els.fovClipMarginValue) els.fovClipMarginValue.textContent = `${clip.marginPx}px`;
  if (els.fovClipMinPixelsInput) els.fovClipMinPixelsInput.value = String(clip.minContactPixels);
  if (els.fovClipMinPixelsValue) els.fovClipMinPixelsValue.textContent = `${clip.minContactPixels}px`;
  if (els.fovClipMinRatioInput) els.fovClipMinRatioInput.value = String(clip.minContactRatio);
  if (els.fovClipMinRatioValue) els.fovClipMinRatioValue.textContent = `${fmt(clip.minContactRatio * 100, 1)}%`;
  if (els.fovClipBboxTouchToggle) els.fovClipBboxTouchToggle.checked = Boolean(clip.requireBboxTouch);
  if (els.fovClipWarnOnlyToggle) els.fovClipWarnOnlyToggle.checked = Boolean(clip.warnOnly);
  const fovSummary = info?.manifest?.summary?.fovClip || null;
  if (els.fovClipStatus) {
    if (!clip.enabled) els.fovClipStatus.textContent = 'off';
    else if (running) els.fovClipStatus.textContent = 'building';
    else if (!info?.available) els.fovClipStatus.textContent = 'needs bbox';
    else if (info.stale) els.fovClipStatus.textContent = 'stale';
    else if (!fovSummary?.enabled) els.fovClipStatus.textContent = 'needs rebuild';
    else els.fovClipStatus.textContent = 'ready';
  }

  if (!info) {
    if (els.bboxStatus) els.bboxStatus.textContent = 'not checked';
    if (els.bboxStats) els.bboxStats.textContent = 'BBox status has not loaded.';
  } else if (running) {
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    if (els.bboxStatus) els.bboxStatus.textContent = job.status;
    if (els.bboxStats) els.bboxStats.textContent = `BBox ${job.status}: ${progress.phase || 'working'} ${index}/${total}`;
  } else if (!dependency.ok) {
    if (els.bboxStatus) els.bboxStatus.textContent = 'dependency';
    if (els.bboxStats) els.bboxStats.textContent = `OpenCV unavailable for bbox build: ${dependency.error || 'missing dependency'}`;
  } else if (!info.available) {
    if (els.bboxStatus) els.bboxStatus.textContent = 'missing';
    if (els.bboxStats) els.bboxStats.textContent = 'No color-mask bbox run has been built for this calibration.';
  } else if (info.stale) {
    if (els.bboxStatus) els.bboxStatus.textContent = 'stale';
    if (els.bboxStats) els.bboxStats.textContent = `BBoxes need rebuild: ${(info.staleReasons || []).join(', ') || 'stale'}.`;
  } else {
    const summary = info.manifest?.summary || {};
    const frameCount = Number(info.manifest?.frameCount || 0).toLocaleString('en-US');
    const sourceCount = Number(info.manifest?.sourceFrameCount || 0).toLocaleString('en-US');
    const bboxCount = Number(summary.bboxCount || 0).toLocaleString('en-US');
    const quadAccepted = Number(summary.quadAcceptedCount || 0).toLocaleString('en-US');
    const voidOverlaps = Number(summary.voidOverlapCount || 0).toLocaleString('en-US');
    const mean = Number(summary.meanBboxesPerFrame || 0);
    const frameCap = Number(summary.bboxFrameCap || info.manifest?.settings?.maxBboxesPerFrame || settings.maxBboxesPerFrame || 0).toLocaleString('en-US');
    const observedMax = Number(summary.observedMaxBboxesPerFrame ?? summary.maxBboxesPerFrame ?? 0).toLocaleString('en-US');
    const clipText = clip.enabled && summary.fovClip?.enabled
      ? ` | clip ${Number(summary.fovClip.clippedBboxCount || 0).toLocaleString('en-US')} | near ${Number(summary.fovClip.nearEdgeBboxCount || 0).toLocaleString('en-US')} | max ${fmt(summary.fovClip.maxSeverity || 0, 2)}`
      : clip.enabled ? ' | clip pending' : '';
    if (els.bboxStatus) els.bboxStatus.textContent = 'ready';
    if (els.bboxStats) els.bboxStats.textContent = `${bboxCount} boxes | observed max ${observedMax}/frame | cap ${frameCap}/frame | ${quadAccepted} quads | ${voidOverlaps} void overlaps | ${frameCount}/${sourceCount} frames | avg ${fmt(mean, 2)}/frame${clipText}`;
  }
  if (els.bboxBuildButton) {
    els.bboxBuildButton.disabled = running || dependency.ok === false;
    els.bboxBuildButton.textContent = running ? 'Building BBoxes' : 'Build BBoxes';
  }
  if (els.maskbitsBboxStats) {
    if (maskbitsRunning) {
      const total = maskbitsProgress.total ?? '?';
      const index = maskbitsProgress.index ?? 0;
      els.maskbitsBboxStats.textContent = `Maskbits bbox ${maskbitsJob.status}: ${maskbitsProgress.phase || 'working'} ${index}/${total}`;
    } else if (!dependency.ok) {
      els.maskbitsBboxStats.textContent = `OpenCV unavailable for maskbits bbox build: ${dependency.error || 'missing dependency'}`;
    } else if (!maskbitsInfo) {
      els.maskbitsBboxStats.textContent = 'Maskbits bbox status has not loaded.';
    } else if (!maskbitsInfo.available) {
      els.maskbitsBboxStats.textContent = `No maskbits bbox run built. Source: ${cleanFileRef(maskbitsInfo.input?.maskManifest || '/src/color_masks/output/mask_manifest.json', 72)}`;
    } else if (maskbitsInfo.stale) {
      els.maskbitsBboxStats.textContent = `Maskbits bboxes need rebuild: ${(maskbitsInfo.staleReasons || []).join(', ') || 'stale'}.`;
    } else {
      const summary = maskbitsInfo.manifest?.summary || {};
      const comparison = maskbitsInfo.manifest?.comparison || summary.comparison || {};
      const bboxCount = Number(summary.bboxCount || 0).toLocaleString('en-US');
      const selected = Number(summary.selectedPixelCount || 0).toLocaleString('en-US');
      const compareText = comparison.available
        ? comparison.passed
          ? ' | parity pass'
          : ` | parity diff sel ${Number(comparison.selectedPixelMismatches || 0).toLocaleString('en-US')} bbox ${Number(comparison.bboxCountMismatches || 0).toLocaleString('en-US')} geom ${Number(comparison.geometryMismatches || 0).toLocaleString('en-US')} max ${Number(comparison.maxCoordinateDeltaPx || 0).toLocaleString('en-US')}px`
        : ' | no baseline compare';
      els.maskbitsBboxStats.textContent = `${bboxCount} boxes | ${selected} selected px | ${Number(summary.frameCount || maskbitsInfo.manifest?.frameCount || 0).toLocaleString('en-US')} frames${compareText}`;
    }
  }
  if (els.maskbitsBboxBuildButton) {
    els.maskbitsBboxBuildButton.disabled = maskbitsRunning || dependency.ok === false;
    els.maskbitsBboxBuildButton.textContent = maskbitsRunning ? 'Building Maskbits BBoxes' : 'Build Maskbits BBoxes';
  }
  renderBboxClippingControls();
}

function renderBboxClippingControls() {
  const info = activeBboxClippingInfo();
  const job = state.bboxClipping.discovery?.job || null;
  const progress = job?.progress || {};
  const running = job && ['queued', 'running'].includes(job.status);
  const dependency = info?.dependency || state.dependencyStatus?.bboxClipping || {};
  if (els.bboxClippingStats) {
    if (running) {
      const total = progress.total ?? '?';
      const index = progress.index ?? 0;
      els.bboxClippingStats.textContent = `BBox clipping ${job.status}: ${progress.phase || 'working'} ${index}/${total}`;
    } else if (!dependency.ok) {
      els.bboxClippingStats.textContent = `OpenCV unavailable for bbox clipping: ${dependency.error || 'missing dependency'}`;
    } else if (!info) {
      els.bboxClippingStats.textContent = 'BBox clipping status has not loaded.';
    } else if (!info.bbox?.available || info.bbox?.stale) {
      els.bboxClippingStats.textContent = `Build fresh maskbits bboxes first: ${(info.bbox?.staleReasons || []).join(', ') || 'maskbits bbox missing'}.`;
    } else if (!info.available) {
      els.bboxClippingStats.textContent = 'No bbox clipping manifest has been built.';
    } else if (info.stale) {
      els.bboxClippingStats.textContent = `BBox clipping needs rebuild: ${(info.staleReasons || []).join(', ') || 'stale'}.`;
    } else {
      const summary = info.manifest?.summary || {};
      const clipped = Number(summary.clippedBboxCount || 0).toLocaleString('en-US');
      const near = Number(summary.nearEdgeBboxCount || 0).toLocaleString('en-US');
      const objects = Number(summary.objectCount || 0).toLocaleString('en-US');
      const max = fmt(summary.maxSeverity || 0, 2);
      const frameCount = Number(info.manifest?.frameCount || 0).toLocaleString('en-US');
      els.bboxClippingStats.textContent = `${objects} objects | clipped ${clipped} | near ${near} | max severity ${max} | ${frameCount} frames`;
    }
  }
  if (els.bboxClippingBuildButton) {
    els.bboxClippingBuildButton.disabled = running || dependency.ok === false || !info?.bbox?.available || Boolean(info?.bbox?.stale);
    els.bboxClippingBuildButton.textContent = running ? 'Building BBox Clipping' : 'Build BBox Clipping';
  }
}

function renderInstanceControls() {
  const settings = instanceTrackingState();
  const info = activeInstanceInfo();
  const job = state.instances.discovery?.job || null;
  const progress = job?.progress || {};
  const running = job && ['queued', 'running'].includes(job.status);
  if (els.instanceToggle) els.instanceToggle.checked = Boolean(settings.enabled);
  if (els.instanceOverlayToggle) els.instanceOverlayToggle.checked = Boolean(settings.showOverlay);
  if (els.instanceLabelsToggle) els.instanceLabelsToggle.checked = Boolean(settings.showLabels);
  if (els.instanceLinksToggle) els.instanceLinksToggle.checked = Boolean(settings.showLinks);
  if (els.instanceCandidatesToggle) els.instanceCandidatesToggle.checked = Boolean(settings.showCandidates);
  if (els.instanceReadoutToggle) els.instanceReadoutToggle.checked = Boolean(settings.showReadout);
  if (els.instancePoseSourceSelect) els.instancePoseSourceSelect.value = settings.poseSource;
  if (els.instanceMaxGapInput) els.instanceMaxGapInput.value = String(settings.maxFrameGap);
  if (els.instanceMaxGapValue) els.instanceMaxGapValue.textContent = String(settings.maxFrameGap);
  if (els.instanceMinScoreInput) els.instanceMinScoreInput.value = String(settings.minAssociationScore);
  if (els.instanceMinScoreValue) els.instanceMinScoreValue.textContent = fmt(settings.minAssociationScore, 2);
  if (els.instanceMax2dInput) els.instanceMax2dInput.value = String(settings.max2dDistancePx);
  if (els.instanceMax2dValue) els.instanceMax2dValue.textContent = `${fmt(settings.max2dDistancePx, 0)}px`;
  if (els.instanceMax3dInput) els.instanceMax3dInput.value = String(settings.max3dDistanceM);
  if (els.instanceMax3dValue) els.instanceMax3dValue.textContent = `${fmt(settings.max3dDistanceM, 1)}m`;
  if (els.instanceMaxRpyInput) els.instanceMaxRpyInput.value = String(settings.maxRpyDeltaDeg);
  if (els.instanceMaxRpyValue) els.instanceMaxRpyValue.textContent = `${fmt(settings.maxRpyDeltaDeg, 0)}deg`;
  if (els.instanceSplitScoreInput) els.instanceSplitScoreInput.value = String(settings.splitCandidateScore);
  if (els.instanceSplitScoreValue) els.instanceSplitScoreValue.textContent = fmt(settings.splitCandidateScore, 2);
  if (els.instanceTrailInput) els.instanceTrailInput.value = String(settings.trailLengthFrames);
  if (els.instanceTrailValue) els.instanceTrailValue.textContent = String(settings.trailLengthFrames);
  if (els.instanceCandidateLimitInput) els.instanceCandidateLimitInput.value = String(settings.debugCandidateLimit);
  if (els.instanceCandidateLimitValue) els.instanceCandidateLimitValue.textContent = String(settings.debugCandidateLimit);

  if (!info) {
    if (els.instanceStatus) els.instanceStatus.textContent = 'not checked';
    if (els.instanceStats) els.instanceStats.textContent = 'Instance tracking status has not loaded.';
  } else if (running) {
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    if (els.instanceStatus) els.instanceStatus.textContent = job.status;
    if (els.instanceStats) els.instanceStats.textContent = `Instances ${job.status}: ${progress.phase || 'working'} ${index}/${total}`;
  } else if (!info.bboxes?.available || info.bboxes?.stale) {
    if (els.instanceStatus) els.instanceStatus.textContent = 'needs bbox';
    if (els.instanceStats) els.instanceStats.textContent = `Build fresh bboxes first: ${(info.bboxes?.staleReasons || []).join(', ') || 'bboxes missing'}.`;
  } else if (!info.available) {
    if (els.instanceStatus) els.instanceStatus.textContent = 'missing';
    if (els.instanceStats) els.instanceStats.textContent = 'No instance provenance run has been built.';
  } else if (info.stale) {
    if (els.instanceStatus) els.instanceStatus.textContent = 'stale';
    if (els.instanceStats) els.instanceStats.textContent = `Instances need rebuild: ${(info.staleReasons || []).join(', ') || 'stale'}.`;
  } else {
    const summary = info.manifest?.summary || {};
    if (els.instanceStatus) els.instanceStatus.textContent = settings.enabled ? 'ready' : 'ready overlay off';
    if (els.instanceStats) {
      els.instanceStats.textContent = `${Number(summary.instanceCount || 0).toLocaleString('en-US')} instances | ${Number(summary.observationCount || 0).toLocaleString('en-US')} obs | matched ${Number(summary.matchedCount || 0).toLocaleString('en-US')} | split ${Number(summary.splitCandidateCount || 0).toLocaleString('en-US')} | mean ${fmt(summary.meanAssociationScore || 0, 2)} | pose ${summary.poseSignalsAvailable ? 'on' : 'off'}`;
    }
  }
  if (els.instanceBuildButton) {
    const bboxesReady = info?.bboxes ? info.bboxes.available && !info.bboxes.stale : false;
    els.instanceBuildButton.disabled = running || !bboxesReady;
    els.instanceBuildButton.textContent = running ? 'Building Instances' : 'Build Instances';
  }
  renderInstanceReadout();
}

function renderContourControls(frameSummary = null) {
  const settings = contourHierarchyState();
  const info = activeContourInfo();
  const dependency = info?.dependency || state.dependencyStatus?.maskContours || {};
  const job = state.contour.discovery?.job || null;
  const progress = job?.progress || {};
  const running = job && ['queued', 'running'].includes(job.status);
  if (els.contourToggle) els.contourToggle.checked = Boolean(settings.show);
  if (els.contourSourceSelect) els.contourSourceSelect.value = settings.maskSource;
  if (els.contourMinOuterInput) els.contourMinOuterInput.value = String(settings.minOuterAreaPx);
  if (els.contourMinOuterValue) els.contourMinOuterValue.textContent = `${settings.minOuterAreaPx}px`;
  if (els.contourMinVoidInput) els.contourMinVoidInput.value = String(settings.minVoidAreaPx);
  if (els.contourMinVoidValue) els.contourMinVoidValue.textContent = `${settings.minVoidAreaPx}px`;
  if (els.contourMaxVoidsInput) els.contourMaxVoidsInput.value = String(settings.maxVoidsPerBbox);
  if (els.contourMaxVoidsValue) els.contourMaxVoidsValue.textContent = String(settings.maxVoidsPerBbox);
  if (els.contourSimplifyInput) els.contourSimplifyInput.value = String(settings.simplifyEpsilonPx);
  if (els.contourSimplifyValue) els.contourSimplifyValue.textContent = `${fmt(settings.simplifyEpsilonPx, 2)}px`;
  if (els.contourCloseInput) els.contourCloseInput.value = String(settings.closeRadiusPx);
  if (els.contourCloseValue) els.contourCloseValue.textContent = `${settings.closeRadiusPx}px`;
  if (els.contourOpenInput) els.contourOpenInput.value = String(settings.openRadiusPx);
  if (els.contourOpenValue) els.contourOpenValue.textContent = `${settings.openRadiusPx}px`;
  if (els.contourNotchInput) els.contourNotchInput.value = String(settings.notchProximityPx);
  if (els.contourNotchValue) els.contourNotchValue.textContent = `${settings.notchProximityPx}px`;
  if (els.contourLineThicknessInput) els.contourLineThicknessInput.value = String(settings.lineThicknessPx);
  if (els.contourLineThicknessValue) els.contourLineThicknessValue.textContent = `${settings.lineThicknessPx}px`;
  if (els.contourOpacityInput) els.contourOpacityInput.value = String(Math.round(settings.opacity * 100));
  if (els.contourOpacityValue) els.contourOpacityValue.textContent = `${Math.round(settings.opacity * 100)}%`;
  if (els.contourLabelsToggle) els.contourLabelsToggle.checked = Boolean(settings.showLabels);
  if (els.contourSmallToggle) els.contourSmallToggle.checked = Boolean(settings.includeSmallContours);

  if (!info) {
    if (els.contourStatus) els.contourStatus.textContent = 'not checked';
    if (els.contourStats) els.contourStats.textContent = 'Contour status has not loaded.';
  } else if (running) {
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    if (els.contourStatus) els.contourStatus.textContent = job.status;
    if (els.contourStats) els.contourStats.textContent = `Contours ${job.status}: ${progress.phase || 'working'} ${index}/${total}`;
  } else if (!settings.show) {
    if (els.contourStatus) els.contourStatus.textContent = 'hidden';
    if (els.contourStats) els.contourStats.textContent = 'Contour overlay hidden.';
  } else if (!dependency.ok) {
    if (els.contourStatus) els.contourStatus.textContent = 'dependency';
    if (els.contourStats) els.contourStats.textContent = `OpenCV unavailable for contour build: ${dependency.error || 'missing dependency'}`;
  } else if (!info.bbox?.available || info.bbox?.stale) {
    if (els.contourStatus) els.contourStatus.textContent = 'needs bbox';
    if (els.contourStats) els.contourStats.textContent = `Build fresh bboxes first: ${(info.bbox?.staleReasons || []).join(', ') || 'bbox missing'}.`;
  } else if (!info.available) {
    if (els.contourStatus) els.contourStatus.textContent = 'missing';
    if (els.contourStats) els.contourStats.textContent = 'No contour hierarchy has been built for this calibration.';
  } else if (info.stale) {
    if (els.contourStatus) els.contourStatus.textContent = 'stale';
    if (els.contourStats) els.contourStats.textContent = `Contours need rebuild: ${(info.staleReasons || []).join(', ') || 'stale'}.`;
  } else {
    const summary = info.manifest?.summary || {};
    const frameCount = Number(info.manifest?.frameCount || 0).toLocaleString('en-US');
    const sourceCount = Number(info.manifest?.sourceFrameCount || 0).toLocaleString('en-US');
    const contourCount = Number(summary.contourCount || 0).toLocaleString('en-US');
    const objectCount = Number(summary.objectCount || 0).toLocaleString('en-US');
    const voidCount = Number(summary.voidCount || 0).toLocaleString('en-US');
    const nestedCount = Number(summary.nestedIslandCount || 0).toLocaleString('en-US');
    const notchCount = Number(summary.notchCandidateCount || 0).toLocaleString('en-US');
    const frameText = frameSummary
      ? ` | frame ${Number(frameSummary.contourCount || 0).toLocaleString('en-US')} contours, ${Number(frameSummary.voidCount || 0).toLocaleString('en-US')} voids`
      : '';
    if (els.contourStatus) els.contourStatus.textContent = 'ready';
    if (els.contourStats) els.contourStats.textContent = `${contourCount} contours | ${objectCount} objects | ${voidCount} voids | ${nestedCount} nested | ${notchCount} notches | ${frameCount}/${sourceCount} frames${frameText}`;
  }
  if (els.contourBuildButton) {
    const bboxReady = info?.bbox ? info.bbox.available && !info.bbox.stale : true;
    els.contourBuildButton.disabled = running || dependency.ok === false || bboxReady === false;
    els.contourBuildButton.textContent = running ? 'Building Contours' : 'Build Contours';
  }
  renderBboxContourControls();
}

function renderBboxContourControls() {
  const info = activeBboxContoursInfo();
  const job = state.bboxContours.discovery?.job || null;
  const progress = job?.progress || {};
  const running = job && ['queued', 'running'].includes(job.status);
  const dependency = info?.dependency || state.dependencyStatus?.bboxContours || {};
  if (els.bboxContourStats) {
    if (running) {
      const total = progress.total ?? '?';
      const index = progress.index ?? 0;
      els.bboxContourStats.textContent = `BBox contours ${job.status}: ${progress.phase || 'working'} ${index}/${total}`;
    } else if (!dependency.ok) {
      els.bboxContourStats.textContent = `OpenCV unavailable for bbox contours: ${dependency.error || 'missing dependency'}`;
    } else if (!info) {
      els.bboxContourStats.textContent = 'BBox contour status has not loaded.';
    } else if (!info.bbox?.available || info.bbox?.stale) {
      els.bboxContourStats.textContent = `Build fresh maskbits bboxes first: ${(info.bbox?.staleReasons || []).join(', ') || 'maskbits bbox missing'}.`;
    } else if (!info.available) {
      els.bboxContourStats.textContent = 'No bbox-local contour manifest has been built.';
    } else if (info.stale) {
      els.bboxContourStats.textContent = `BBox contours need rebuild: ${(info.staleReasons || []).join(', ') || 'stale'}.`;
    } else {
      const summary = info.manifest?.summary || {};
      const contours = Number(summary.contourCount || 0).toLocaleString('en-US');
      const objects = Number(summary.objectCount || 0).toLocaleString('en-US');
      const voids = Number(summary.voidCount || 0).toLocaleString('en-US');
      const nested = Number(summary.nestedIslandCount || 0).toLocaleString('en-US');
      const frameCount = Number(info.manifest?.frameCount || 0).toLocaleString('en-US');
      els.bboxContourStats.textContent = `${contours} contours | ${objects} objects | ${voids} voids | ${nested} nested | ${frameCount} frames`;
    }
  }
  if (els.bboxContourBuildButton) {
    els.bboxContourBuildButton.disabled = running || dependency.ok === false || !info?.bbox?.available || Boolean(info?.bbox?.stale);
    els.bboxContourBuildButton.textContent = running ? 'Building BBox Contours' : 'Build BBox Contours';
  }
}

async function loadBboxManifest() {
  const info = activeBboxInfo();
  if (!info || !info.available || info.stale || !info.manifestUrl) {
    state.bbox.manifest = null;
    scheduleBboxRender();
    return;
  }
  state.bbox.manifest = await fetchJson(info.manifestUrl);
  scheduleBboxRender();
}

async function loadMaskbitsBboxManifest() {
  const info = activeMaskbitsBboxInfo();
  if (!info || !info.available || info.stale || !info.manifestUrl) {
    state.maskbitsBbox.manifest = null;
    scheduleBboxRender();
    return;
  }
  state.maskbitsBbox.manifest = await fetchJson(info.manifestUrl);
  scheduleBboxRender();
}

async function loadContourManifest() {
  const info = activeContourInfo();
  if (!info || !info.available || info.stale || !info.manifestUrl) {
    state.contour.manifest = null;
    scheduleBboxRender();
    return;
  }
  state.contour.manifest = await fetchJson(info.manifestUrl);
  scheduleBboxRender();
}

async function loadContourDiscovery() {
  state.contour.discovery = await fetchJson(`${API}/contours`);
  await loadContourManifest().catch((error) => {
    state.contour.manifest = null;
    if (els.contourStats) els.contourStats.textContent = `Contour manifest load failed: ${error.message}`;
  });
  renderContourControls();
}

async function buildContours() {
  await saveReview();
  const settings = contourAnalysisSettings();
  const response = await fetchJson(`${API}/contours/build`, {
    method: 'POST',
    body: JSON.stringify(settings)
  });
  if (!state.contour.discovery) state.contour.discovery = {};
  state.contour.discovery.job = response;
  renderContourControls();
  setStatus(`Contour build queued ${response.jobId}`);
  await pollContourBuild();
}

async function pollContourBuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const status = await fetchJson(`${API}/contours/build/status`);
    state.contour.discovery = status.contours || state.contour.discovery;
    if (state.contour.discovery) state.contour.discovery.job = status.job;
    renderContourControls();
    const job = status.job;
    if (!job) {
      setStatus('no contour job');
      return;
    }
    const progress = job.progress || {};
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    setStatus(`contours ${job.status}: ${progress.phase || 'working'} ${index}/${total}`);
    if (job.status === 'complete') {
      setStatus('Contours ready');
      await loadContourDiscovery();
      await loadSquarePoseDiscovery().catch((error) => setStatus(`square pose status failed: ${error.message}`));
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'Contour build failed');
    }
  }
}

function renderSquarePoseControls() {
  const settings = squarePoseState();
  const info = activeSquarePoseInfo();
  const job = state.squarePose.discovery?.job || null;
  const running = job && ['queued', 'running'].includes(job.status);
  const dependency = info?.dependency || state.dependencyStatus?.squarePose || {};
  if (els.squarePoseToggle) els.squarePoseToggle.checked = Boolean(settings.show);
  if (els.squarePoseViewModeSelect) els.squarePoseViewModeSelect.value = settings.viewMode;
  if (els.squarePoseSolutionSelect) els.squarePoseSolutionSelect.value = settings.solutionMode;
  if (els.squarePoseSizeInput) els.squarePoseSizeInput.value = String(settings.squareSizeM);
  if (els.squarePoseSizeValue) els.squarePoseSizeValue.textContent = `${fmt(settings.squareSizeM, 2)}m`;
  if (els.squarePoseEdgeToleranceInput) els.squarePoseEdgeToleranceInput.value = String(settings.edgeTolerancePx);
  if (els.squarePoseEdgeToleranceValue) els.squarePoseEdgeToleranceValue.textContent = `${fmt(settings.edgeTolerancePx, 2)}px`;
  if (els.squarePoseMinCoverageInput) els.squarePoseMinCoverageInput.value = String(settings.minEdgeCoverage);
  if (els.squarePoseMinCoverageValue) els.squarePoseMinCoverageValue.textContent = `${Math.round(settings.minEdgeCoverage * 100)}%`;
  if (els.squarePoseMaxErrorInput) els.squarePoseMaxErrorInput.value = String(settings.maxReprojectionErrorPx);
  if (els.squarePoseMaxErrorValue) els.squarePoseMaxErrorValue.textContent = `${fmt(settings.maxReprojectionErrorPx, 1)}px`;
  if (els.squarePoseClipGuardToggle) els.squarePoseClipGuardToggle.checked = Boolean(settings.clipPoseGuardEnabled);
  if (els.squarePoseClipThresholdInput) els.squarePoseClipThresholdInput.value = String(settings.clipInvalidationThreshold);
  if (els.squarePoseClipThresholdValue) els.squarePoseClipThresholdValue.textContent = `${Math.round(settings.clipInvalidationThreshold * 100)}%`;
  if (els.squarePoseCornerFitToggle) els.squarePoseCornerFitToggle.checked = Boolean(settings.cornerAwareFitEnabled);
  if (els.squarePoseMinCornerLinksInput) els.squarePoseMinCornerLinksInput.value = String(settings.minCornerAgreementLinks);
  if (els.squarePoseMinCornerLinksValue) els.squarePoseMinCornerLinksValue.textContent = String(settings.minCornerAgreementLinks);
  if (els.squarePoseCompleteBonusToggle) els.squarePoseCompleteBonusToggle.checked = Boolean(settings.completeCornerAgreementBonus);
  if (els.squarePoseExtraCornerLimitInput) els.squarePoseExtraCornerLimitInput.value = String(settings.multiGateExtraCornerLimit);
  if (els.squarePoseExtraCornerLimitValue) els.squarePoseExtraCornerLimitValue.textContent = String(settings.multiGateExtraCornerLimit);
  if (els.squarePoseMaxCandidatesInput) els.squarePoseMaxCandidatesInput.value = String(settings.maxPoseCandidates);
  if (els.squarePoseMaxCandidatesValue) els.squarePoseMaxCandidatesValue.textContent = String(settings.maxPoseCandidates);
  if (els.squarePoseTextureOpacityInput) els.squarePoseTextureOpacityInput.value = String(Math.round(settings.textureOpacity * 100));
  if (els.squarePoseTextureOpacityValue) els.squarePoseTextureOpacityValue.textContent = `${Math.round(settings.textureOpacity * 100)}%`;
  if (els.squarePoseCandidateOpacityInput) els.squarePoseCandidateOpacityInput.value = String(Math.round(settings.candidateOpacity * 100));
  if (els.squarePoseCandidateOpacityValue) els.squarePoseCandidateOpacityValue.textContent = `${Math.round(settings.candidateOpacity * 100)}%`;
  if (els.squarePoseFrustumToggle) els.squarePoseFrustumToggle.checked = Boolean(settings.showFrustum);
  if (els.squarePoseOutlineToggle) els.squarePoseOutlineToggle.checked = Boolean(settings.showOutline);
  if (els.squarePoseScoresToggle) els.squarePoseScoresToggle.checked = Boolean(settings.showScores);
  if (els.squarePoseLabelsToggle) els.squarePoseLabelsToggle.checked = Boolean(settings.showLabels);

  if (!info) {
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'not checked';
    if (els.squarePoseStats) els.squarePoseStats.textContent = 'Square pose status has not loaded.';
  } else if (running) {
    const progress = job.progress || {};
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = job.status;
    if (els.squarePoseStats) els.squarePoseStats.textContent = `Square pose ${job.status}: ${progress.phase || 'working'} ${progress.index || 0}/${progress.total || '?'}`;
  } else if (!settings.show) {
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'hidden';
    if (els.squarePoseStats) els.squarePoseStats.textContent = 'Square pose viewer hidden.';
  } else if (!dependency.ok) {
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'dependency';
    if (els.squarePoseStats) els.squarePoseStats.textContent = `OpenCV unavailable for square pose: ${dependency.error || 'missing dependency'}`;
  } else if (settings.clipPoseGuardEnabled && (!info.bboxes?.available || info.bboxes?.stale)) {
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'needs bboxes';
    if (els.squarePoseStats) els.squarePoseStats.textContent = `Build fresh bboxes first: ${(info.bboxes?.staleReasons || []).join(', ') || 'bboxes missing'}.`;
  } else if (!info.contours?.available || info.contours?.stale) {
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'needs contours';
    if (els.squarePoseStats) els.squarePoseStats.textContent = `Build fresh contours first: ${(info.contours?.staleReasons || []).join(', ') || 'contours missing'}.`;
  } else if (settings.cornerAwareFitEnabled && (!info.corners?.available || info.corners?.stale)) {
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'needs corners';
    if (els.squarePoseStats) els.squarePoseStats.textContent = `Build fresh corners first: ${(info.corners?.staleReasons || []).join(', ') || 'corners missing'}.`;
  } else if (!info.available) {
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'missing';
    if (els.squarePoseStats) els.squarePoseStats.textContent = 'No square pose manifest has been built.';
  } else if (info.stale) {
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'stale';
    if (els.squarePoseStats) els.squarePoseStats.textContent = `Square pose needs rebuild: ${(info.staleReasons || []).join(', ') || 'stale'}.`;
  } else {
    const summary = info.manifest?.summary || {};
    if (els.squarePoseStatus) els.squarePoseStatus.textContent = 'ready';
    if (els.squarePoseStats) {
      const usable = summary.usablePoseCount ?? summary.acceptedCount ?? 0;
      const clipped = summary.clippedIgnoredCount ?? 0;
      const ambiguous = summary.ambiguousOverlapCount ?? 0;
      const multi = summary.multiGateEvidenceCount ?? 0;
      els.squarePoseStats.textContent = `${Number(summary.poseCount || 0).toLocaleString('en-US')} poses | usable ${Number(usable).toLocaleString('en-US')} | clipped ${Number(clipped).toLocaleString('en-US')} | ambiguous ${Number(ambiguous).toLocaleString('en-US')} | multi ${Number(multi).toLocaleString('en-US')} | mean error ${fmt(summary.meanReprojectionErrorPx, 2)}px`;
    }
  }
  if (els.squarePoseBuildButton) {
    const contoursReady = info?.contours ? info.contours.available && !info.contours.stale : false;
    const bboxesReady = !settings.clipPoseGuardEnabled || (info?.bboxes ? info.bboxes.available && !info.bboxes.stale : false);
    const cornersReady = !settings.cornerAwareFitEnabled || (info?.corners ? info.corners.available && !info.corners.stale : false);
    els.squarePoseBuildButton.disabled = running || dependency.ok === false || !contoursReady || !bboxesReady || !cornersReady;
    els.squarePoseBuildButton.textContent = running ? 'Building Square Pose' : 'Build Square Pose';
  }
}

async function loadSquarePoseManifest() {
  const info = activeSquarePoseInfo();
  if (!info || !info.available || info.stale || !info.manifestUrl) {
    state.squarePose.manifest = null;
    scheduleSquarePoseRender();
    scheduleTopPoseRender();
    return;
  }
  state.squarePose.manifest = await fetchJson(info.manifestUrl);
  scheduleSquarePoseRender();
  scheduleTopPoseRender();
}

async function loadSquarePoseDiscovery() {
  state.squarePose.discovery = await fetchJson(`${API}/pose-estimation`);
  await loadSquarePoseManifest().catch((error) => {
    state.squarePose.manifest = null;
    if (els.squarePoseStats) els.squarePoseStats.textContent = `Square pose manifest load failed: ${error.message}`;
  });
  renderSquarePoseControls();
}

async function buildSquarePose() {
  await saveReview();
  const settings = squarePoseBuildSettings();
  const maxFrames = Number(els.maxFramesInput.value);
  if (Number.isFinite(maxFrames) && maxFrames > 0) settings.maxFrames = Math.round(maxFrames);
  const response = await fetchJson(`${API}/pose-estimation/build`, {
    method: 'POST',
    body: JSON.stringify(settings)
  });
  if (!state.squarePose.discovery) state.squarePose.discovery = {};
  state.squarePose.discovery.job = response;
  renderSquarePoseControls();
  setStatus(`Pose estimation build queued ${response.jobId}`);
  await pollSquarePoseBuild();
}

async function pollSquarePoseBuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const status = await fetchJson(`${API}/pose-estimation/build/status`);
    state.squarePose.discovery = status.poseEstimation || status.squarePose || state.squarePose.discovery;
    if (state.squarePose.discovery) state.squarePose.discovery.job = status.job;
    renderSquarePoseControls();
    const job = status.job;
    if (!job) {
      setStatus('no square pose job');
      return;
    }
    const progress = job.progress || {};
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    setStatus(`pose estimation ${job.status}: ${progress.phase || 'working'} ${index}/${total}`);
    if (job.status === 'complete') {
      setStatus('Pose estimation ready');
      await loadSquarePoseDiscovery();
      await loadInstanceDiscovery().catch((error) => setStatus(`instance status failed: ${error.message}`));
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'Square pose build failed');
    }
  }
}

async function loadBboxDiscovery() {
  state.bbox.discovery = await fetchJson(`${API}/bboxes`);
  await loadBboxManifest().catch((error) => {
    state.bbox.manifest = null;
    if (els.bboxStats) els.bboxStats.textContent = `BBox manifest load failed: ${error.message}`;
  });
  renderBboxControls();
}

async function loadMaskbitsBboxDiscovery() {
  state.maskbitsBbox.discovery = await fetchJson(`${API}/maskbits-bboxes`);
  await loadMaskbitsBboxManifest().catch((error) => {
    state.maskbitsBbox.manifest = null;
    if (els.maskbitsBboxStats) els.maskbitsBboxStats.textContent = `Maskbits bbox manifest load failed: ${error.message}`;
  });
  clearBboxContourFrameCache();
  renderBboxControls();
}

async function loadBboxClippingManifest() {
  const info = activeBboxClippingInfo();
  if (!info || !info.available || info.stale || !info.manifestUrl) {
    state.bboxClipping.manifest = null;
    scheduleBboxRender();
    return;
  }
  state.bboxClipping.manifest = await fetchJson(info.manifestUrl);
  scheduleBboxRender();
}

async function loadBboxClippingDiscovery() {
  state.bboxClipping.discovery = await fetchJson(`${API}/bbox-clipping`);
  await loadBboxClippingManifest().catch((error) => {
    state.bboxClipping.manifest = null;
    if (els.bboxClippingStats) els.bboxClippingStats.textContent = `BBox clipping manifest load failed: ${error.message}`;
  });
  renderBboxClippingControls();
}

async function loadBboxContoursManifest() {
  const info = activeBboxContoursInfo();
  if (!info || !info.available || info.stale || !info.manifestUrl) {
    state.bboxContours.manifest = null;
    scheduleBboxRender();
    return;
  }
  state.bboxContours.manifest = await fetchJson(info.manifestUrl);
  scheduleBboxRender();
}

async function loadBboxContoursDiscovery() {
  state.bboxContours.discovery = await fetchJson(`${API}/bbox-contours`);
  await loadBboxContoursManifest().catch((error) => {
    state.bboxContours.manifest = null;
    if (els.bboxContourStats) els.bboxContourStats.textContent = `BBox contour manifest load failed: ${error.message}`;
  });
  renderBboxContourControls();
}

async function loadInstanceManifest() {
  const info = activeInstanceInfo();
  if (!info || !info.available || !info.manifestUrl) {
    state.instances.manifest = null;
    renderInstanceControls();
    scheduleBboxRender();
    scheduleInstance3dRender();
    scheduleTopInstanceRender();
    return;
  }
  state.instances.manifest = await fetchJson(info.manifestUrl);
  renderInstanceControls();
  scheduleBboxRender();
  scheduleInstance3dRender();
  scheduleTopInstanceRender();
}

async function loadInstanceDiscovery() {
  state.instances.discovery = await fetchJson(`${API}/instance-tracking`);
  await loadInstanceManifest().catch((error) => {
    state.instances.manifest = null;
    if (els.instanceStats) els.instanceStats.textContent = `Instance manifest load failed: ${error.message}`;
  });
  renderInstanceControls();
}

async function buildBboxes() {
  await saveReview();
  const settings = bboxFlowState();
  const response = await fetchJson(`${API}/bboxes/build`, {
    method: 'POST',
    body: JSON.stringify(settings)
  });
  if (!state.bbox.discovery) state.bbox.discovery = {};
  state.bbox.discovery.job = response;
  renderBboxControls();
  setStatus(`BBox build queued ${response.jobId}`);
  await pollBboxBuild();
}

async function buildMaskbitsBboxes() {
  await saveReview();
  const settings = bboxFlowState();
  const response = await fetchJson(`${API}/maskbits-bboxes/build`, {
    method: 'POST',
    body: JSON.stringify(settings)
  });
  if (!state.maskbitsBbox.discovery) state.maskbitsBbox.discovery = {};
  state.maskbitsBbox.discovery.job = response;
  renderBboxControls();
  setStatus(`Maskbits bbox build queued ${response.jobId}`);
  await pollMaskbitsBboxBuild();
}

async function buildBboxClipping() {
  await saveReview();
  const settings = bboxFlowState().fovClip || {};
  const response = await fetchJson(`${API}/bbox-clipping/build`, {
    method: 'POST',
    body: JSON.stringify(settings)
  });
  if (!state.bboxClipping.discovery) state.bboxClipping.discovery = {};
  state.bboxClipping.discovery.job = response;
  renderBboxClippingControls();
  setStatus(`BBox clipping build queued ${response.jobId}`);
  await pollBboxClippingBuild();
}

async function buildBboxContours() {
  await saveReview();
  const settings = contourBuildSettings();
  const response = await fetchJson(`${API}/bbox-contours/build`, {
    method: 'POST',
    body: JSON.stringify(settings)
  });
  if (!state.bboxContours.discovery) state.bboxContours.discovery = {};
  state.bboxContours.discovery.job = response;
  renderBboxContourControls();
  setStatus(`BBox contour build queued ${response.jobId}`);
  await pollBboxContourBuild();
}

async function buildInstances() {
  await saveReview();
  const settings = instanceTrackingState();
  const response = await fetchJson(`${API}/instance-tracking/build`, {
    method: 'POST',
    body: JSON.stringify(settings)
  });
  if (!state.instances.discovery) state.instances.discovery = {};
  state.instances.discovery.job = response;
  renderInstanceControls();
  setStatus(`Instance tracking build queued ${response.jobId}`);
  await pollInstanceBuild();
}

async function pollBboxBuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const status = await fetchJson(`${API}/bboxes/build/status`);
    state.bbox.discovery = status.bboxes || state.bbox.discovery;
    if (state.bbox.discovery) state.bbox.discovery.job = status.job;
    renderBboxControls();
    const job = status.job;
    if (!job) {
      setStatus('no bbox job');
      return;
    }
    const progress = job.progress || {};
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    setStatus(`bbox ${job.status}: ${progress.phase || 'working'} ${index}/${total}`);
    if (job.status === 'complete') {
      setStatus('BBoxes ready');
      await loadBboxDiscovery();
      await loadMaskbitsBboxDiscovery().catch((error) => setStatus(`maskbits bbox status failed: ${error.message}`));
      await loadBboxClippingDiscovery().catch((error) => setStatus(`bbox clipping status failed: ${error.message}`));
      await loadBboxContoursDiscovery().catch((error) => setStatus(`bbox contour status failed: ${error.message}`));
      await loadContourDiscovery().catch((error) => setStatus(`contour status failed: ${error.message}`));
      await loadSquarePoseDiscovery().catch((error) => setStatus(`square pose status failed: ${error.message}`));
      await loadInstanceDiscovery().catch((error) => setStatus(`instance status failed: ${error.message}`));
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'BBox build failed');
    }
  }
}

async function pollMaskbitsBboxBuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const status = await fetchJson(`${API}/maskbits-bboxes/build/status`);
    state.maskbitsBbox.discovery = status.maskbitsBboxes || state.maskbitsBbox.discovery;
    if (state.maskbitsBbox.discovery) state.maskbitsBbox.discovery.job = status.job;
    renderBboxControls();
    const job = status.job;
    if (!job) {
      setStatus('no maskbits bbox job');
      return;
    }
    const progress = job.progress || {};
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    setStatus(`maskbits bbox ${job.status}: ${progress.phase || 'working'} ${index}/${total}`);
    if (job.status === 'complete') {
      setStatus('Maskbits bboxes ready');
      await loadMaskbitsBboxDiscovery();
      await loadBboxClippingDiscovery().catch((error) => setStatus(`bbox clipping status failed: ${error.message}`));
      await loadBboxContoursDiscovery().catch((error) => setStatus(`bbox contour status failed: ${error.message}`));
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'Maskbits bbox build failed');
    }
  }
}

async function pollBboxClippingBuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const status = await fetchJson(`${API}/bbox-clipping/build/status`);
    state.bboxClipping.discovery = status.bboxClipping || state.bboxClipping.discovery;
    if (state.bboxClipping.discovery) state.bboxClipping.discovery.job = status.job;
    renderBboxClippingControls();
    const job = status.job;
    if (!job) {
      setStatus('no bbox clipping job');
      return;
    }
    const progress = job.progress || {};
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    setStatus(`bbox clipping ${job.status}: ${progress.phase || 'working'} ${index}/${total}`);
    if (job.status === 'complete') {
      setStatus('BBox clipping ready');
      await loadBboxClippingDiscovery();
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'BBox clipping build failed');
    }
  }
}

async function pollBboxContourBuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const status = await fetchJson(`${API}/bbox-contours/build/status`);
    state.bboxContours.discovery = status.bboxContours || state.bboxContours.discovery;
    if (state.bboxContours.discovery) state.bboxContours.discovery.job = status.job;
    renderBboxContourControls();
    const job = status.job;
    if (!job) {
      setStatus('no bbox contour job');
      return;
    }
    const progress = job.progress || {};
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    setStatus(`bbox contours ${job.status}: ${progress.phase || 'working'} ${index}/${total}`);
    if (job.status === 'complete') {
      setStatus('BBox contours ready');
      await loadBboxContoursDiscovery();
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'BBox contour build failed');
    }
  }
}

async function pollInstanceBuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const status = await fetchJson(`${API}/instance-tracking/build/status`);
    state.instances.discovery = status.instanceTracking || status.instances || state.instances.discovery;
    if (state.instances.discovery) state.instances.discovery.job = status.job;
    renderInstanceControls();
    const job = status.job;
    if (!job) {
      setStatus('no instance tracking job');
      return;
    }
    const progress = job.progress || {};
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    setStatus(`instance tracking ${job.status}: ${progress.phase || 'working'} ${index}/${total}`);
    if (job.status === 'complete') {
      setStatus('Instance tracking ready');
      await loadInstanceDiscovery();
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'Instance provenance build failed');
    }
  }
}

function renderCornerControls() {
  const settings = cornerFlowState();
  const hull = settings.hull;
  const voids = settings.void;
  const info = activeCornerInfo();
  const dependency = info?.dependency || state.dependencyStatus?.maskCorners || {};
  const job = state.corner.discovery?.job || null;
  const progress = job?.progress || {};
  const running = job && ['queued', 'running'].includes(job.status);
  const syncTypeControls = (typeKey) => {
    const def = CORNER_TYPE_DEFS[typeKey];
    const prefix = `corner${def.label}`;
    const values = settings.types?.[typeKey] || {};
    const setInput = (suffix, value) => {
      const input = document.getElementById(`${prefix}${suffix}Input`);
      if (input) input.value = String(value);
    };
    const setNumber = (suffix, value) => {
      const input = document.getElementById(`${prefix}${suffix}Number`);
      if (input) input.value = String(value);
    };
    const setOutput = (suffix, text) => {
      const output = document.getElementById(`${prefix}${suffix}Value`);
      if (output) output.textContent = text;
    };
    const showToggle = document.getElementById(`${prefix}ShowToggle`);
    if (showToggle) showToggle.checked = settings[`show${def.label}`] !== false;
    if (def.category === 'void') {
      setInput('MinArea', values.minAreaPx);
      setOutput('MinArea', `${values.minAreaPx}px`);
    }
    setInput('Radius', values.radiusPx);
    setOutput('Radius', `${values.radiusPx}px`);
    setInput('MinAngle', values.minAngleDeg);
    setOutput('MinAngle', `${values.minAngleDeg}deg`);
    setInput('MaxAngle', values.maxAngleDeg);
    setOutput('MaxAngle', `${values.maxAngleDeg}deg`);
    setInput('Support', values.minSupportPixels);
    setOutput('Support', `${values.minSupportPixels}px`);
    setInput('Epsilon', values.contourEpsilonPx);
    setOutput('Epsilon', fmt(values.contourEpsilonPx, 2));
    setInput('Distance', values.minDistancePx);
    setOutput('Distance', `${values.minDistancePx}px`);
    setInput('MinPoints', values.minPoints);
    setNumber('MinPoints', values.minPoints);
    setOutput('MinPoints', String(values.minPoints));
    setInput('MaxPoints', values.maxPoints);
    setNumber('MaxPoints', values.maxPoints);
    setOutput('MaxPoints', String(values.maxPoints));
  };
  CORNER_TYPE_KEYS.forEach(syncTypeControls);
  if (els.cornerHullRadiusInput) els.cornerHullRadiusInput.value = String(hull.radiusPx);
  if (els.cornerHullRadiusValue) els.cornerHullRadiusValue.textContent = `${hull.radiusPx}px`;
  if (els.cornerHullMinAngleInput) els.cornerHullMinAngleInput.value = String(hull.minAngleDeg);
  if (els.cornerHullMinAngleValue) els.cornerHullMinAngleValue.textContent = `${hull.minAngleDeg}deg`;
  if (els.cornerHullMaxAngleInput) els.cornerHullMaxAngleInput.value = String(hull.maxAngleDeg);
  if (els.cornerHullMaxAngleValue) els.cornerHullMaxAngleValue.textContent = `${hull.maxAngleDeg}deg`;
  if (els.cornerHullMinSupportInput) els.cornerHullMinSupportInput.value = String(hull.minSupportPixels);
  if (els.cornerHullMinSupportValue) els.cornerHullMinSupportValue.textContent = `${hull.minSupportPixels}px`;
  if (els.cornerHullEpsilonInput) els.cornerHullEpsilonInput.value = String(hull.contourEpsilonPx);
  if (els.cornerHullEpsilonValue) els.cornerHullEpsilonValue.textContent = fmt(hull.contourEpsilonPx, 2);
  if (els.cornerHullDistanceInput) els.cornerHullDistanceInput.value = String(hull.minDistancePx);
  if (els.cornerHullDistanceValue) els.cornerHullDistanceValue.textContent = `${hull.minDistancePx}px`;
  if (els.cornerHullMaxInput) els.cornerHullMaxInput.value = String(hull.maxCornersPerBbox);
  if (els.cornerHullMaxValue) els.cornerHullMaxValue.textContent = String(hull.maxCornersPerBbox);
  if (els.cornerHullWhiteMinInput) els.cornerHullWhiteMinInput.value = String(hull.minMaskInsidePoints);
  if (els.cornerHullWhiteMinNumber) els.cornerHullWhiteMinNumber.value = String(hull.minMaskInsidePoints);
  if (els.cornerHullWhiteMinValue) els.cornerHullWhiteMinValue.textContent = String(hull.minMaskInsidePoints);
  if (els.cornerHullWhiteMaxInput) els.cornerHullWhiteMaxInput.value = String(hull.maxMaskInsidePoints);
  if (els.cornerHullWhiteMaxNumber) els.cornerHullWhiteMaxNumber.value = String(hull.maxMaskInsidePoints);
  if (els.cornerHullWhiteMaxValue) els.cornerHullWhiteMaxValue.textContent = String(hull.maxMaskInsidePoints);
  if (els.cornerHullBlackMinInput) els.cornerHullBlackMinInput.value = String(hull.minMaskOutsidePoints);
  if (els.cornerHullBlackMinNumber) els.cornerHullBlackMinNumber.value = String(hull.minMaskOutsidePoints);
  if (els.cornerHullBlackMinValue) els.cornerHullBlackMinValue.textContent = String(hull.minMaskOutsidePoints);
  if (els.cornerHullBlackMaxInput) els.cornerHullBlackMaxInput.value = String(hull.maxMaskOutsidePoints);
  if (els.cornerHullBlackMaxNumber) els.cornerHullBlackMaxNumber.value = String(hull.maxMaskOutsidePoints);
  if (els.cornerHullBlackMaxValue) els.cornerHullBlackMaxValue.textContent = String(hull.maxMaskOutsidePoints);
  if (els.cornerVoidMinAreaInput) els.cornerVoidMinAreaInput.value = String(voids.minAreaPx);
  if (els.cornerVoidMinAreaValue) els.cornerVoidMinAreaValue.textContent = `${voids.minAreaPx}px`;
  if (els.cornerVoidRadiusInput) els.cornerVoidRadiusInput.value = String(voids.radiusPx);
  if (els.cornerVoidRadiusValue) els.cornerVoidRadiusValue.textContent = `${voids.radiusPx}px`;
  if (els.cornerVoidMinAngleInput) els.cornerVoidMinAngleInput.value = String(voids.minAngleDeg);
  if (els.cornerVoidMinAngleValue) els.cornerVoidMinAngleValue.textContent = `${voids.minAngleDeg}deg`;
  if (els.cornerVoidMaxAngleInput) els.cornerVoidMaxAngleInput.value = String(voids.maxAngleDeg);
  if (els.cornerVoidMaxAngleValue) els.cornerVoidMaxAngleValue.textContent = `${voids.maxAngleDeg}deg`;
  if (els.cornerVoidMinSupportInput) els.cornerVoidMinSupportInput.value = String(voids.minSupportPixels);
  if (els.cornerVoidMinSupportValue) els.cornerVoidMinSupportValue.textContent = `${voids.minSupportPixels}px`;
  if (els.cornerVoidEpsilonInput) els.cornerVoidEpsilonInput.value = String(voids.contourEpsilonPx);
  if (els.cornerVoidEpsilonValue) els.cornerVoidEpsilonValue.textContent = fmt(voids.contourEpsilonPx, 2);
  if (els.cornerVoidDistanceInput) els.cornerVoidDistanceInput.value = String(voids.minDistancePx);
  if (els.cornerVoidDistanceValue) els.cornerVoidDistanceValue.textContent = `${voids.minDistancePx}px`;
  if (els.cornerVoidMaxInput) els.cornerVoidMaxInput.value = String(voids.maxCornersPerVoid);
  if (els.cornerVoidMaxValue) els.cornerVoidMaxValue.textContent = String(voids.maxCornersPerVoid);
  if (els.cornerVoidOrangeMinInput) els.cornerVoidOrangeMinInput.value = String(voids.minMaskInsidePoints);
  if (els.cornerVoidOrangeMinNumber) els.cornerVoidOrangeMinNumber.value = String(voids.minMaskInsidePoints);
  if (els.cornerVoidOrangeMinValue) els.cornerVoidOrangeMinValue.textContent = String(voids.minMaskInsidePoints);
  if (els.cornerVoidOrangeMaxInput) els.cornerVoidOrangeMaxInput.value = String(voids.maxMaskInsidePoints);
  if (els.cornerVoidOrangeMaxNumber) els.cornerVoidOrangeMaxNumber.value = String(voids.maxMaskInsidePoints);
  if (els.cornerVoidOrangeMaxValue) els.cornerVoidOrangeMaxValue.textContent = String(voids.maxMaskInsidePoints);
  if (els.cornerVoidGreenMinInput) els.cornerVoidGreenMinInput.value = String(voids.minMaskOutsidePoints);
  if (els.cornerVoidGreenMinNumber) els.cornerVoidGreenMinNumber.value = String(voids.minMaskOutsidePoints);
  if (els.cornerVoidGreenMinValue) els.cornerVoidGreenMinValue.textContent = String(voids.minMaskOutsidePoints);
  if (els.cornerVoidGreenMaxInput) els.cornerVoidGreenMaxInput.value = String(voids.maxMaskOutsidePoints);
  if (els.cornerVoidGreenMaxNumber) els.cornerVoidGreenMaxNumber.value = String(voids.maxMaskOutsidePoints);
  if (els.cornerVoidGreenMaxValue) els.cornerVoidGreenMaxValue.textContent = String(voids.maxMaskOutsidePoints);
  if (els.cornerShowHullToggle) els.cornerShowHullToggle.checked = Boolean(settings.showHull);
  if (els.cornerShowVoidToggle) els.cornerShowVoidToggle.checked = Boolean(settings.showVoid);
  if (els.cornerShowAmbiguousToggle) els.cornerShowAmbiguousToggle.checked = Boolean(settings.showAmbiguous);
  if (els.cornerOpacityInput) els.cornerOpacityInput.value = String(Math.round(settings.overlayOpacity * 100));
  if (els.cornerOpacityValue) els.cornerOpacityValue.textContent = `${Math.round(settings.overlayOpacity * 100)}%`;

  if (!info) {
    if (els.cornerStatus) els.cornerStatus.textContent = 'not checked';
    if (els.cornerStats) els.cornerStats.textContent = 'Corner status has not loaded.';
  } else if (running) {
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    if (els.cornerStatus) els.cornerStatus.textContent = job.status;
    if (els.cornerStats) els.cornerStats.textContent = `Corner ${job.status}: ${progress.phase || 'working'} ${index}/${total}`;
  } else if (!dependency.ok) {
    if (els.cornerStatus) els.cornerStatus.textContent = 'dependency';
    if (els.cornerStats) els.cornerStats.textContent = `OpenCV unavailable for corner build: ${dependency.error || 'missing dependency'}`;
  } else if (info.bbox && !info.bbox.available) {
    if (els.cornerStatus) els.cornerStatus.textContent = 'needs bbox';
    if (els.cornerStats) els.cornerStats.textContent = 'Build BBoxes before building bbox-local corners.';
  } else if (info.bbox?.stale) {
    if (els.cornerStatus) els.cornerStatus.textContent = 'bbox stale';
    if (els.cornerStats) els.cornerStats.textContent = `BBoxes need rebuild first: ${(info.bbox.staleReasons || []).join(', ') || 'stale'}.`;
  } else if (!info.available) {
    if (els.cornerStatus) els.cornerStatus.textContent = 'missing';
    if (els.cornerStats) els.cornerStats.textContent = 'No bbox-local corner run has been built for this calibration.';
  } else if (info.stale) {
    if (els.cornerStatus) els.cornerStatus.textContent = 'stale';
    if (els.cornerStats) els.cornerStats.textContent = `Corners need rebuild: ${(info.staleReasons || []).join(', ') || 'stale'}.`;
  } else {
    const summary = info.manifest?.summary || {};
    const total = Number(summary.cornerCount || 0).toLocaleString('en-US');
    const hullCount = Number(summary.hullCornerCount ?? summary.maskWedgeCount ?? 0).toLocaleString('en-US');
    const voidCount = Number(summary.voidCornerCount ?? summary.voidWedgeCount ?? 0).toLocaleString('en-US');
    const internalVoids = Number(summary.voidCount || 0).toLocaleString('en-US');
    const ambiguous = Number(summary.ambiguousCount || 0).toLocaleString('en-US');
    const whiteHull = Number(summary.hullMaskInsideCount ?? summary.hullMaskWedgeCount ?? 0).toLocaleString('en-US');
    const blackHull = Number(summary.hullMaskOutsideCount ?? summary.hullVoidWedgeCount ?? 0).toLocaleString('en-US');
    const orangeVoid = Number(summary.voidMaskInsideCount ?? summary.voidMaskWedgeCount ?? 0).toLocaleString('en-US');
    const greenVoid = Number(summary.voidMaskOutsideCount ?? summary.voidVoidWedgeCount ?? 0).toLocaleString('en-US');
    const frameCount = Number(info.manifest?.frameCount || 0).toLocaleString('en-US');
    const sourceCount = Number(info.manifest?.sourceFrameCount || 0).toLocaleString('en-US');
    if (els.cornerStatus) els.cornerStatus.textContent = 'ready';
    if (els.cornerStats) els.cornerStats.textContent = `${total} corners | hull ${hullCount} | void ${voidCount} | white hull ${whiteHull} | black hull ${blackHull} | orange void ${orangeVoid} | green void ${greenVoid} | ${internalVoids} voids | amber ${ambiguous} | ${frameCount}/${sourceCount} frames`;
  }
  if (els.cornerBuildButton) {
    const bboxReady = info?.bbox ? info.bbox.available && !info.bbox.stale : true;
    els.cornerBuildButton.disabled = running || dependency.ok === false || bboxReady === false;
    els.cornerBuildButton.textContent = running ? 'Building Corners' : 'Build Corners';
  }
  renderCornerAgreementControls();
  renderLayer002QuadControls();
}

function renderCornerAgreementControls(summary = null) {
  const settings = cornerAgreementState();
  if (els.cornerAgreementToggle) els.cornerAgreementToggle.checked = Boolean(settings.show);
  if (els.cornerAgreementMaxDistanceInput) els.cornerAgreementMaxDistanceInput.value = String(settings.maxDistancePx);
  if (els.cornerAgreementMaxDistanceValue) els.cornerAgreementMaxDistanceValue.textContent = `${settings.maxDistancePx}px`;
  if (els.cornerAgreementWhiteToleranceInput) els.cornerAgreementWhiteToleranceInput.value = String(settings.whiteToleranceDeg);
  if (els.cornerAgreementWhiteToleranceValue) els.cornerAgreementWhiteToleranceValue.textContent = `${settings.whiteToleranceDeg}deg`;
  if (els.cornerAgreementAngleToleranceInput) els.cornerAgreementAngleToleranceInput.value = String(settings.angleToleranceDeg);
  if (els.cornerAgreementAngleToleranceValue) els.cornerAgreementAngleToleranceValue.textContent = `${settings.angleToleranceDeg}deg`;
  if (els.cornerAgreementMinScoreInput) els.cornerAgreementMinScoreInput.value = String(settings.minScore);
  if (els.cornerAgreementMinScoreValue) els.cornerAgreementMinScoreValue.textContent = fmt(settings.minScore, 2);
  if (els.cornerAgreementMaxLinksInput) els.cornerAgreementMaxLinksInput.value = String(settings.maxLinksPerCorner);
  if (els.cornerAgreementMaxLinksValue) els.cornerAgreementMaxLinksValue.textContent = String(settings.maxLinksPerCorner);
  if (els.cornerAgreementLineThicknessInput) els.cornerAgreementLineThicknessInput.value = String(settings.lineThicknessPx);
  if (els.cornerAgreementLineThicknessValue) els.cornerAgreementLineThicknessValue.textContent = `${settings.lineThicknessPx}px`;
  if (els.cornerAgreementOpacityInput) els.cornerAgreementOpacityInput.value = String(Math.round(settings.opacity * 100));
  if (els.cornerAgreementOpacityValue) els.cornerAgreementOpacityValue.textContent = `${Math.round(settings.opacity * 100)}%`;
  if (els.cornerAgreementStructureToggle) els.cornerAgreementStructureToggle.checked = Boolean(settings.showStructures);
  if (els.cornerAgreementOppositeToleranceInput) els.cornerAgreementOppositeToleranceInput.value = String(settings.oppositeToleranceDeg);
  if (els.cornerAgreementOppositeToleranceValue) els.cornerAgreementOppositeToleranceValue.textContent = `${settings.oppositeToleranceDeg}deg`;

  if (!els.cornerAgreementStats) return;
  if (!settings.show) {
    els.cornerAgreementStats.textContent = 'Agreement links hidden.';
    return;
  }
  const info = activeCornerInfo();
  if (!info || !info.available || info.stale || !state.corner.manifest) {
    els.cornerAgreementStats.textContent = 'No corner manifest loaded.';
    return;
  }
  const result = summary || cornerAgreementResults(settings);
  els.cornerAgreementStats.textContent = `${Number(result.links.length || 0).toLocaleString('en-US')} links | ${Number(result.completeGroupCount || 0).toLocaleString('en-US')} complete`;
}

function renderLayer002QuadControls(summary = null) {
  const settings = layer002QuadState();
  if (els.layer002QuadToggle) els.layer002QuadToggle.checked = Boolean(settings.show);
  if (els.layer002TraceSupportSelect) els.layer002TraceSupportSelect.value = settings.supportMode;
  if (els.layer002TraceForbiddenSelect) els.layer002TraceForbiddenSelect.value = settings.forbiddenMode;
  if (els.layer002QuadMaxInput) els.layer002QuadMaxInput.value = String(settings.maxQuadsPerFrame);
  if (els.layer002QuadMaxValue) els.layer002QuadMaxValue.textContent = String(settings.maxQuadsPerFrame);
  if (els.layer002QuadMinPixelsInput) els.layer002QuadMinPixelsInput.value = String(settings.minPixels);
  if (els.layer002QuadMinPixelsValue) els.layer002QuadMinPixelsValue.textContent = String(settings.minPixels);
  if (els.layer002QuadEdgeThicknessInput) els.layer002QuadEdgeThicknessInput.value = String(settings.edgeBandPx);
  if (els.layer002QuadEdgeThicknessValue) els.layer002QuadEdgeThicknessValue.textContent = `${settings.edgeBandPx}px`;
  if (els.layer002QuadMinCoverageInput) els.layer002QuadMinCoverageInput.value = String(settings.minEdgeCoverage);
  if (els.layer002QuadMinCoverageValue) els.layer002QuadMinCoverageValue.textContent = `${fmt(settings.minEdgeCoverage * 100, 0)}%`;
  if (els.layer002QuadMaxGapInput) els.layer002QuadMaxGapInput.value = String(settings.maxUnsupportedGapPx);
  if (els.layer002QuadMaxGapValue) els.layer002QuadMaxGapValue.textContent = `${settings.maxUnsupportedGapPx}px`;
  if (els.layer002QuadAngleSweepInput) els.layer002QuadAngleSweepInput.value = String(settings.inwardMaxPx);
  if (els.layer002QuadAngleSweepValue) els.layer002QuadAngleSweepValue.textContent = `${settings.inwardMaxPx}px`;
  if (els.layer002QuadAngleStepInput) els.layer002QuadAngleStepInput.value = String(settings.insetStepPx);
  if (els.layer002QuadAngleStepValue) els.layer002QuadAngleStepValue.textContent = `${settings.insetStepPx}px`;
  if (els.layer002QuadInsetInput) els.layer002QuadInsetInput.value = String(settings.angleSweepDeg);
  if (els.layer002QuadInsetValue) els.layer002QuadInsetValue.textContent = `${settings.angleSweepDeg}deg`;
  if (els.layer002TraceAngleStepInput) els.layer002TraceAngleStepInput.value = String(settings.angleStepDeg);
  if (els.layer002TraceAngleStepValue) els.layer002TraceAngleStepValue.textContent = `${settings.angleStepDeg}deg`;
  if (els.layer002TraceAspectToleranceInput) els.layer002TraceAspectToleranceInput.value = String(settings.aspectTolerance);
  if (els.layer002TraceAspectToleranceValue) els.layer002TraceAspectToleranceValue.textContent = fmt(settings.aspectTolerance, 2);
  if (els.layer002QuadLineThicknessInput) els.layer002QuadLineThicknessInput.value = String(settings.lineThicknessPx);
  if (els.layer002QuadLineThicknessValue) els.layer002QuadLineThicknessValue.textContent = `${settings.lineThicknessPx}px`;
  if (els.layer002QuadOpacityInput) els.layer002QuadOpacityInput.value = String(Math.round(settings.opacity * 100));
  if (els.layer002QuadOpacityValue) els.layer002QuadOpacityValue.textContent = `${Math.round(settings.opacity * 100)}%`;
  if (els.layer002QuadRejectedToggle) els.layer002QuadRejectedToggle.checked = Boolean(settings.showRejected);

  if (!els.layer002QuadStats) return;
  if (!settings.show) {
    els.layer002QuadStats.textContent = 'Layer 002 hull sweep hidden.';
    return;
  }
  const info = activeBboxInfo();
  if (!info || !info.available || info.stale || !state.bbox.manifest) {
    els.layer002QuadStats.textContent = 'No bbox manifest loaded.';
    return;
  }
  if (!state.currentColorIds || !state.classMembership) {
    els.layer002QuadStats.textContent = 'No layer-002 frame mask loaded.';
    return;
  }
  const result = summary || layer002QuadResults(settings);
  const bestCoverage = Math.max(0, ...[...(result.accepted || []), ...(result.rejected || [])].map((candidate) => Number(candidate.targetCoverageRatio) || 0));
  els.layer002QuadStats.textContent = `${Number(result.accepted.length || 0).toLocaleString('en-US')} quads | cover ${fmt(bestCoverage * 100, 0)}% | ${Number(result.components || 0).toLocaleString('en-US')} comps | ${Number(result.angles || 0).toLocaleString('en-US')} angles | ${Number(result.rejected.length || 0).toLocaleString('en-US')} rejected | ${Number(result.skipped || 0).toLocaleString('en-US')} skipped`;
}

async function loadCornerManifest() {
  const info = activeCornerInfo();
  if (!info || !info.available || info.stale || !info.manifestUrl) {
    state.corner.manifest = null;
    scheduleBboxRender();
    return;
  }
  state.corner.manifest = await fetchJson(info.manifestUrl);
  scheduleBboxRender();
}

async function loadCornerDiscovery() {
  state.corner.discovery = await fetchJson(`${API}/corners`);
  await loadCornerManifest().catch((error) => {
    state.corner.manifest = null;
    if (els.cornerStats) els.cornerStats.textContent = `Corner manifest load failed: ${error.message}`;
  });
  renderCornerControls();
}

async function buildCorners() {
  await saveReview();
  const settings = cornerFlowState();
  const response = await fetchJson(`${API}/corners/build`, {
    method: 'POST',
    body: JSON.stringify(settings)
  });
  if (!state.corner.discovery) state.corner.discovery = {};
  state.corner.discovery.job = response;
  renderCornerControls();
  setStatus(`Corner build queued ${response.jobId}`);
  await pollCornerBuild();
}

async function pollCornerBuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 700));
    const status = await fetchJson(`${API}/corners/build/status`);
    state.corner.discovery = status.corners || state.corner.discovery;
    if (state.corner.discovery) state.corner.discovery.job = status.job;
    renderCornerControls();
    const job = status.job;
    if (!job) {
      setStatus('no corner job');
      return;
    }
    const progress = job.progress || {};
    const total = progress.total ?? '?';
    const index = progress.index ?? 0;
    setStatus(`corner ${job.status}: ${progress.phase || 'working'} ${index}/${total}`);
    if (job.status === 'complete') {
      setStatus('Corners ready');
      await loadCornerDiscovery();
      await loadSquarePoseDiscovery().catch((error) => setStatus(`square pose status failed: ${error.message}`));
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'Corner build failed');
    }
  }
}

function applySourceOpacityControl() {
  const opacity = sourceOpacity();
  els.sourceCanvas.style.opacity = String(opacity);
  if (els.sourceOpacityInput) els.sourceOpacityInput.value = String(Math.round(opacity * 100));
  if (els.sourceOpacityValue) els.sourceOpacityValue.textContent = `${Math.round(opacity * 100)}%`;
}

function renderClassList() {
  els.classList.innerHTML = state.config.classes.map((item) => {
    const prefix = String(item.prefix);
    const active = prefix === state.activePrefix ? ' active' : '';
    const enabled = layerEnabled(prefix);
    const disabled = enabled ? '' : ' disabled';
    const count = state.ruleSets.get(prefix)?.size || 0;
    const opacity = Math.round(layerOpacity(prefix) * 100);
    const color = layerColor(item);
    const confidence = layerConfidence(item);
    const groupSlots = layerGroupSlots(prefix);
    const groupSelects = groupSlots.map((slotValue, slotIndex) => `
      <select class="layerGroupSelect" data-prefix="${escapeHtml(prefix)}" data-slot="${slotIndex}" aria-label="Group ${slotIndex + 1} for ${escapeHtml(item.displayName)}">
        <option value="">None</option>
        ${LAYER_GROUP_OPTIONS.map(([value, label]) => `
          <option value="${escapeHtml(value)}" ${slotValue === value ? 'selected' : ''}>${escapeHtml(label)}</option>
        `).join('')}
      </select>
    `).join('');
    return `
      <div class="classRow${active}${disabled}" data-prefix="${escapeHtml(prefix)}">
        <div class="classTop">
          <input class="layerEnabledInput" data-prefix="${escapeHtml(prefix)}" type="checkbox" aria-label="Enable ${escapeHtml(item.displayName)}" ${enabled ? 'checked' : ''}>
          <input class="layerColorInput" data-prefix="${escapeHtml(prefix)}" type="color" aria-label="Overlay color ${escapeHtml(item.displayName)}" title="Overlay color only" value="${escapeHtml(color)}">
          <button class="optionInfoButton layerTopInfoButton" type="button" data-option-help-key="layerRow" aria-label="About layer row controls">i</button>
          <button class="classPick" type="button" data-prefix="${escapeHtml(prefix)}">
            <span class="className" title="${escapeHtml(item.displayName)}">${escapeHtml(item.displayName)}</span>
            <span class="classMeta">${fmt(confidence, 2)} | ${count.toLocaleString('en-US')} colors</span>
          </button>
        </div>
        <label class="layerOpacityControl">
          <span>Opacity</span>
          <input class="layerOpacityInput" data-prefix="${escapeHtml(prefix)}" type="range" min="0" max="100" step="1" value="${opacity}">
          <output>${enabled ? opacity : 0}%</output>
        </label>
        <label class="layerConfidenceControl">
          <span>Confidence</span>
          <input class="layerConfidenceInput" data-prefix="${escapeHtml(prefix)}" type="range" min="0" max="100" step="1" value="${Math.round(confidence * 100)}">
          <output>${fmt(confidence, 2)}</output>
        </label>
        <div class="layerGroupControl">
          <span>Groups</span>
          <div class="layerGroupSelects">${groupSelects}</div>
        </div>
      </div>
    `;
  }).join('');
  attachOptionInfoButtons(els.classList);

  els.classList.querySelectorAll('.classPick').forEach((button) => {
    button.addEventListener('click', () => {
      state.activePrefix = button.dataset.prefix;
      updateViewStatus();
      renderClassList();
      renderInspector();
    });
  });
  els.classList.querySelectorAll('.layerEnabledInput').forEach((input) => {
    input.addEventListener('change', () => {
      setLayerEnabled(input.dataset.prefix, input.checked);
      clearBboxContourFrameCache();
      renderClassList();
      renderOverlay();
      renderFrameLayerList();
      updateViewStatus();
      saveReview()
        .then(() => Promise.all([loadBboxDiscovery(), loadMaskbitsBboxDiscovery(), loadContourDiscovery(), loadCornerDiscovery(), loadSquarePoseDiscovery()]))
        .catch((error) => setStatus(`view save failed: ${error.message}`));
    });
  });
  els.classList.querySelectorAll('.layerColorInput').forEach((input) => {
    input.addEventListener('input', () => {
      setLayerColor(input.dataset.prefix, input.value);
      renderOverlay();
    });
    input.addEventListener('change', () => saveReview().catch((error) => setStatus(`view save failed: ${error.message}`)));
  });
  els.classList.querySelectorAll('.layerOpacityInput').forEach((input) => {
    input.addEventListener('input', () => {
      const value = clamp01(Number(input.value) / 100);
      setLayerOpacity(input.dataset.prefix, value);
      input.nextElementSibling.textContent = `${layerEnabled(input.dataset.prefix) ? Math.round(value * 100) : 0}%`;
      renderOverlay();
    });
    input.addEventListener('change', () => saveReview().catch((error) => setStatus(`view save failed: ${error.message}`)));
  });
  els.classList.querySelectorAll('.layerConfidenceInput').forEach((input) => {
    input.addEventListener('input', () => {
      const value = clamp01(Number(input.value) / 100);
      setLayerConfidence(input.dataset.prefix, value);
      input.nextElementSibling.textContent = fmt(value, 2);
    });
    input.addEventListener('change', () => {
      renderClassList();
      saveReview().catch((error) => setStatus(`view save failed: ${error.message}`));
    });
  });
  els.classList.querySelectorAll('.layerGroupSelect').forEach((select) => {
    select.addEventListener('change', () => {
      setLayerGroupSlot(select.dataset.prefix, select.dataset.slot, select.value);
      renderClassList();
      saveReview().catch((error) => setStatus(`view save failed: ${error.message}`));
    });
  });
}

function renderMemorySummary() {
  const summary = state.memory?.summary || {};
  const source = state.memory?.source || {};
  const rows = [
    ['Mode', state.memory?.analysisMode || 'fresh-layer-decode'],
    ['Processed', summary.processedFrameCount ?? source.processedFrameCount ?? 0],
    ['Layers', `${summary.enabledLayerCount ?? 0} / ${summary.decodedLayerCount ?? 0}`],
    ['Classified px', Number(summary.classifiedPixelCount || 0).toLocaleString('en-US')],
    ['Active colors', Number(summary.enabledClassColorIdCount || 0).toLocaleString('en-US')],
    ['Source colors', Number(summary.sourceClassColorIdCount || 0).toLocaleString('en-US')],
    ['Hash', state.memory?.classRules?.hash || '-']
  ];
  els.memoryStatus.textContent = state.memory?.status || '-';
  els.memorySummary.innerHTML = rows.map(([label, value]) => `
    <div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>
  `).join('');
}

function renderAssetSummary(readOnlyAssets) {
  const assets = readOnlyAssets || {};
  const active = (assets.runs || []).find((run) => run.active) || {};
  const rows = [
    ['Frame runs', assets.frameRunsRoot || '-'],
    ['Color layers', assets.layerSetsPath || '-'],
    ['Active run', assets.activeRunName || '-'],
    ['Frames', active.frameCount || '-'],
    ['Precompute', active.precompute?.ready ? active.precompute.runKey : 'missing']
  ];
  els.assetSummary.innerHTML = rows.map(([label, value]) => `
    <div class="kv"><span>${escapeHtml(label)}</span><span title="${escapeHtml(value)}">${escapeHtml(value)}</span></div>
  `).join('');
}

function renderFrameLayerList() {
  const frame = currentMemoryFrame();
  const counts = Array.isArray(frame?.layerCounts) ? frame.layerCounts : [];
  if (els.layerStatus) els.layerStatus.textContent = frame ? `${Number(frame.classifiedPixelCount || 0).toLocaleString('en-US')} px` : 'not decoded';
  if (!counts.length) {
    els.frameLayerList.innerHTML = '<div class="inspectorEmpty">No decoded layer counts for this frame.</div>';
    return;
  }
  els.frameLayerList.innerHTML = counts.map((item) => {
    const classItem = state.config.classes.find((candidate) => String(candidate.prefix) === String(item.prefix)) || {};
    const color = layerColor(classItem);
    const enabled = layerEnabled(item.prefix);
    return `
      <button class="frameLayerRow${String(item.prefix) === state.activePrefix ? ' active' : ''}${enabled ? '' : ' disabled'}" type="button" data-prefix="${escapeHtml(item.prefix)}">
        <span class="swatch" style="background:${escapeHtml(color)}"></span>
        <span>${escapeHtml(item.prefix)}</span>
        <strong>${Number(item.pixelCount || 0).toLocaleString('en-US')}</strong>
      </button>
    `;
  }).join('');
  els.frameLayerList.querySelectorAll('.frameLayerRow').forEach((button) => {
    button.addEventListener('click', () => {
      state.activePrefix = button.dataset.prefix;
      renderClassList();
      renderFrameLayerList();
      renderInspector();
      updateViewStatus();
    });
  });
}

function renderInspector() {
  const item = state.config.classes.find((candidate) => String(candidate.prefix) === String(state.activePrefix));
  if (!item) {
    els.inspector.innerHTML = 'No layer selected.';
    return;
  }
  const summary = (state.memory?.classRules?.summaries || []).find((candidate) => String(candidate.prefix) === String(state.activePrefix)) || {};
  const total = (state.memory?.layerTotals || []).find((candidate) => String(candidate.prefix) === String(state.activePrefix)) || {};
  const rows = [
    ['Layer', item.displayName || item.label || item.prefix],
    ['Prefix', item.prefix],
    ['Visible', layerEnabled(item.prefix) ? 'yes' : 'no'],
    ['Confidence', fmt(layerConfidence(item), 2)],
    ['Color IDs', Number(summary.sourceColorIdCount || 0).toLocaleString('en-US')],
    ['Active run IDs', Number(summary.colorIdCount || 0).toLocaleString('en-US')],
    ['Rule source', summary.ruleSource || '-'],
    ['Total frame pixels', Number(total.pixelCount || 0).toLocaleString('en-US')],
    ['Groups', (layerGroupSlots(item.prefix).filter(Boolean).join(', ') || '-')]
  ];
  els.inspector.innerHTML = rows.map(([label, value]) => `
    <div class="kv"><span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span></div>
  `).join('');
}

function updateViewStatus() {
  const activeCount = state.ruleSets.get(state.activePrefix)?.size || 0;
  els.rulesStatus.textContent = state.dirtyView ? 'view unsaved' : 'view saved';
  els.selectionReadout.textContent = `${state.activePrefix}: ${activeCount.toLocaleString('en-US')} decoded color IDs`;
}

async function saveReview() {
  await fetchJson(`${API}/review`, {
    method: 'PUT',
    body: JSON.stringify(state.review || {})
  });
  state.dirtyView = false;
  updateViewStatus();
}

async function rebuildMemory() {
  await saveReview();
  const maxFrames = Number(els.maxFramesInput.value);
  const payload = {};
  if (Number.isFinite(maxFrames) && maxFrames > 0) payload.maxFrames = Math.round(maxFrames);
  const response = await fetchJson(`${API}/rebuild`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
  setStatus(`decode queued ${response.jobId}`);
  await pollRebuild();
}

async function pollRebuild() {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 900));
    const status = await fetchJson(`${API}/rebuild/status`);
    const job = status.job;
    if (!job) {
      setStatus('no decode job');
      return;
    }
    const progress = job.progress || {};
    setStatus(`decode ${job.status}: ${progress.phase || 'working'} ${progress.index || 0}/${progress.total || '?'}`);
    if (job.status === 'complete') {
      await loadMemory();
      await loadBboxDiscovery().catch((error) => setStatus(`bbox status failed: ${error.message}`));
      await loadMaskbitsBboxDiscovery().catch((error) => setStatus(`maskbits bbox status failed: ${error.message}`));
      await loadBboxClippingDiscovery().catch((error) => setStatus(`bbox clipping status failed: ${error.message}`));
      await loadBboxContoursDiscovery().catch((error) => setStatus(`bbox contour status failed: ${error.message}`));
      await loadContourDiscovery().catch((error) => setStatus(`contour status failed: ${error.message}`));
      await loadCornerDiscovery().catch((error) => setStatus(`corner status failed: ${error.message}`));
      await loadSquarePoseDiscovery().catch((error) => setStatus(`square pose status failed: ${error.message}`));
      await loadFrame(state.frameIndex);
      return;
    }
    if (job.status === 'error') {
      throw new Error(job.error || 'decode failed');
    }
  }
}

async function loadMemory() {
  state.memory = await fetchJson(`${API}/memory`);
  initializeRuleSets();
  renderClassList();
  renderMemorySummary();
  renderFrameLayerList();
  renderInspector();
  updateViewStatus();
  renderOverlay();
}

async function loadReview() {
  state.review = await fetchJson(`${API}/review`);
}

function onCanvasWheel(event) {
  if (!event.shiftKey) return;
  if (isAuxRailEvent(event)) return;
  event.preventDefault();
  const stagePoint = stagePointFromEvent(event);
  if (!stagePointInMainViewport(stagePoint) || !stageToStackPoint(stagePoint)) return;
  const mainDelta = Math.abs(event.deltaY) >= Math.abs(event.deltaX) ? event.deltaY : event.deltaX;
  if (!mainDelta) return;
  const units = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? 16 : event.deltaMode === WheelEvent.DOM_DELTA_PAGE ? 400 : 1;
  const multiplier = Math.exp(-(mainDelta * units) * 0.0015);
  zoomFrameAt(stagePoint, state.viewport.scale * multiplier);
  setStatus(`zoom ${Math.round(state.viewport.scale * 100)}%`);
}

function onPointerDown(event) {
  if (!event.shiftKey) return;
  if (isAuxRailEvent(event)) return;
  event.preventDefault();
  const stagePoint = stagePointFromEvent(event);
  if (!stagePointInMainViewport(stagePoint) || !stageToStackPoint(stagePoint)) return;
  state.panDrag = {
    start: stagePoint,
    panX: state.viewport.panX,
    panY: state.viewport.panY,
    captureTarget: event.currentTarget
  };
  els.canvasWrap.classList.add('panning');
  event.currentTarget?.setPointerCapture?.(event.pointerId);
}

function onPointerMove(event) {
  if (state.panDrag) {
    const stagePoint = stagePointFromEvent(event);
    state.viewport.panX = state.panDrag.panX + stagePoint.x - state.panDrag.start.x;
    state.viewport.panY = state.panDrag.panY + stagePoint.y - state.panDrag.start.y;
    layoutFrameSurface();
    return;
  }
  const stagePoint = stagePointFromEvent(event);
  if (!stagePointInMainViewport(stagePoint)) return;
  const local = stageToStackPoint(stagePoint);
  const point = stageToFramePoint(stagePoint);
  if (!point) return;
  const id = colorIdAt(point);
  const rgb = rgbForColorId(id);
  const hnl = hnlForColorId(id);
  const classLabel = classLabelsForMask(state.classMembership?.[id] || 0);
  els.hoverReadout.textContent = rgb
    ? `x ${point.x} y ${point.y} | id ${id} | rgb ${rgb.join(',')} | h ${fmt(hnl?.hue, 3)} l ${fmt(hnl?.lightness, 3)} | layers ${classLabel}`
    : `x ${point.x} y ${point.y} | id ${id} | layers ${classLabel}`;
}

function onPointerUp(event) {
  if (!state.panDrag) return;
  const captureTarget = state.panDrag.captureTarget;
  state.panDrag = null;
  els.canvasWrap.classList.remove('panning');
  captureTarget?.releasePointerCapture?.(event.pointerId);
}

function installEvents() {
  if (state.eventsInstalled) return;
  state.eventsInstalled = true;
  if (window.ResizeObserver) {
    new ResizeObserver(() => layoutFrameSurface()).observe(els.canvasWrap);
  }
  window.addEventListener('resize', layoutFrameSurface);
  els.canvasWrap.addEventListener('wheel', onCanvasWheel, { passive: false });
  els.canvasWrap.addEventListener('pointerdown', onPointerDown);
  els.canvasWrap.addEventListener('pointermove', onPointerMove);
  els.canvasWrap.addEventListener('pointerup', onPointerUp);
  els.canvasWrap.addEventListener('pointercancel', onPointerUp);
  if (els.sourceLibraryButton) els.sourceLibraryButton.addEventListener('click', () => setSourceMode('library'));
  if (els.sourceLiveButton) els.sourceLiveButton.addEventListener('click', () => setSourceMode('live'));
  if (els.libraryRunSelect) {
    els.libraryRunSelect.addEventListener('change', () => {
      state.source.libraryRunName = els.libraryRunSelect.value || '';
      state.source.followLatest = true;
      state.source.frameIndex = 0;
      state.source.readiness = null;
      clearBboxContourFrameCache();
      refreshSourceManifests().catch((error) => setStatus(`library source failed: ${error.message}`));
      refreshPlaybackReadiness().catch((error) => setStatus(`readiness failed: ${error.message}`));
    });
  }
  if (els.livePointInput) {
    const commitLivePoint = () => {
      state.source.livePoint = els.livePointInput.value.trim() || '/src/color_masks/output/mask_manifest.json';
      state.source.followLatest = true;
      state.source.frameIndex = 0;
      state.source.readiness = null;
      clearBboxContourFrameCache();
      refreshSourceManifests().catch((error) => setStatus(`live source failed: ${error.message}`));
      refreshPlaybackReadiness().catch((error) => setStatus(`readiness failed: ${error.message}`));
    };
    els.livePointInput.addEventListener('change', commitLivePoint);
    els.livePointInput.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter') return;
      event.preventDefault();
      commitLivePoint();
    });
  }
  if (els.pipelineStartButton) {
    els.pipelineStartButton.addEventListener('click', () => {
      startPipelineRun().catch((error) => setStatus(`pipeline start failed: ${error.message}`));
    });
  }
  if (els.pipelineStopButton) {
    els.pipelineStopButton.addEventListener('click', () => {
      stopPipelineRun().catch((error) => setStatus(`pipeline stop failed: ${error.message}`));
    });
  }
  els.frameSlider.addEventListener('input', () => loadSelectedSourceFrame(Number(els.frameSlider.value)).catch((error) => setStatus(error.message)));
  els.saveRulesButton.addEventListener('click', () => saveReview().catch((error) => setStatus(`save failed: ${error.message}`)));
  els.rebuildButton.addEventListener('click', () => rebuildMemory().catch((error) => setStatus(`decode failed: ${error.message}`)));
  if (els.sourceOpacityInput) {
    els.sourceOpacityInput.addEventListener('input', () => {
      setSourceOpacity(Number(els.sourceOpacityInput.value) / 100);
      applySourceOpacityControl();
    });
    els.sourceOpacityInput.addEventListener('change', () => saveReview().catch((error) => setStatus(`view save failed: ${error.message}`)));
  }
  const saveSquarePoseViewSetting = () => {
    saveReview()
      .then(() => {
        renderSquarePoseControls();
        scheduleSquarePoseRender();
      })
      .catch((error) => setStatus(`square pose view save failed: ${error.message}`));
  };
  const saveSquarePoseBuildSetting = () => {
    saveReview()
      .then(loadSquarePoseDiscovery)
      .then(loadInstanceDiscovery)
      .catch((error) => setStatus(`square pose setting save failed: ${error.message}`));
  };
  if (els.squarePoseToggle) {
    els.squarePoseToggle.addEventListener('change', () => {
      squarePoseState().show = Boolean(els.squarePoseToggle.checked);
      state.dirtyView = true;
      saveSquarePoseViewSetting();
    });
  }
  if (els.squarePoseViewModeSelect) {
    els.squarePoseViewModeSelect.addEventListener('change', () => {
      squarePoseState().viewMode = els.squarePoseViewModeSelect.value === 'orbit' ? 'orbit' : 'camera';
      state.dirtyView = true;
      saveSquarePoseViewSetting();
    });
  }
  if (els.squarePoseSolutionSelect) {
    els.squarePoseSolutionSelect.addEventListener('change', () => {
      squarePoseState().solutionMode = ['best', 'top', 'right', 'bottom', 'left'].includes(els.squarePoseSolutionSelect.value)
        ? els.squarePoseSolutionSelect.value
        : 'best';
      state.dirtyView = true;
      saveSquarePoseViewSetting();
    });
  }
  const bindSquarePoseSlider = (input, assign, save) => {
    if (!input) return;
    input.addEventListener('input', () => {
      assign(Number(input.value));
      state.dirtyView = true;
      renderSquarePoseControls();
      scheduleSquarePoseRender();
    });
    input.addEventListener('change', save);
  };
  bindSquarePoseSlider(els.squarePoseSizeInput, (value) => {
    squarePoseState().squareSizeM = Math.max(0.1, Math.min(20, value || 2.7));
  }, saveSquarePoseBuildSetting);
  bindSquarePoseSlider(els.squarePoseEdgeToleranceInput, (value) => {
    squarePoseState().edgeTolerancePx = Math.max(0.25, Math.min(64, value || 4));
  }, saveSquarePoseBuildSetting);
  bindSquarePoseSlider(els.squarePoseMinCoverageInput, (value) => {
    squarePoseState().minEdgeCoverage = clamp01(value || 0);
  }, saveSquarePoseBuildSetting);
  bindSquarePoseSlider(els.squarePoseMaxErrorInput, (value) => {
    squarePoseState().maxReprojectionErrorPx = Math.max(0.1, Math.min(200, value || 8));
  }, saveSquarePoseBuildSetting);
  if (els.squarePoseClipGuardToggle) {
    els.squarePoseClipGuardToggle.addEventListener('change', () => {
      squarePoseState().clipPoseGuardEnabled = Boolean(els.squarePoseClipGuardToggle.checked);
      state.dirtyView = true;
      saveSquarePoseBuildSetting();
    });
  }
  bindSquarePoseSlider(els.squarePoseClipThresholdInput, (value) => {
    squarePoseState().clipInvalidationThreshold = clamp01(value || 0);
  }, saveSquarePoseBuildSetting);
  if (els.squarePoseCornerFitToggle) {
    els.squarePoseCornerFitToggle.addEventListener('change', () => {
      squarePoseState().cornerAwareFitEnabled = Boolean(els.squarePoseCornerFitToggle.checked);
      state.dirtyView = true;
      saveSquarePoseBuildSetting();
    });
  }
  bindSquarePoseSlider(els.squarePoseMinCornerLinksInput, (value) => {
    squarePoseState().minCornerAgreementLinks = Math.max(0, Math.min(12, Math.round(Number(value) || 0)));
  }, saveSquarePoseBuildSetting);
  if (els.squarePoseCompleteBonusToggle) {
    els.squarePoseCompleteBonusToggle.addEventListener('change', () => {
      squarePoseState().completeCornerAgreementBonus = Boolean(els.squarePoseCompleteBonusToggle.checked);
      state.dirtyView = true;
      saveSquarePoseBuildSetting();
    });
  }
  bindSquarePoseSlider(els.squarePoseExtraCornerLimitInput, (value) => {
    squarePoseState().multiGateExtraCornerLimit = Math.max(4, Math.min(24, Math.round(Number(value) || 4)));
  }, saveSquarePoseBuildSetting);
  bindSquarePoseSlider(els.squarePoseMaxCandidatesInput, (value) => {
    squarePoseState().maxPoseCandidates = Math.max(1, Math.min(64, Math.round(Number(value) || 1)));
  }, saveSquarePoseBuildSetting);
  bindSquarePoseSlider(els.squarePoseTextureOpacityInput, (value) => {
    squarePoseState().textureOpacity = clamp01((value || 0) / 100);
  }, saveSquarePoseViewSetting);
  bindSquarePoseSlider(els.squarePoseCandidateOpacityInput, (value) => {
    squarePoseState().candidateOpacity = clamp01((value || 0) / 100);
  }, saveSquarePoseViewSetting);
  if (els.squarePoseFrustumToggle) {
    els.squarePoseFrustumToggle.addEventListener('change', () => {
      squarePoseState().showFrustum = Boolean(els.squarePoseFrustumToggle.checked);
      state.dirtyView = true;
      saveSquarePoseViewSetting();
    });
  }
  if (els.squarePoseOutlineToggle) {
    els.squarePoseOutlineToggle.addEventListener('change', () => {
      squarePoseState().showOutline = Boolean(els.squarePoseOutlineToggle.checked);
      state.dirtyView = true;
      saveSquarePoseViewSetting();
    });
  }
  if (els.squarePoseScoresToggle) {
    els.squarePoseScoresToggle.addEventListener('change', () => {
      squarePoseState().showScores = Boolean(els.squarePoseScoresToggle.checked);
      state.dirtyView = true;
      saveSquarePoseViewSetting();
    });
  }
  if (els.squarePoseLabelsToggle) {
    els.squarePoseLabelsToggle.addEventListener('change', () => {
      squarePoseState().showLabels = Boolean(els.squarePoseLabelsToggle.checked);
      state.dirtyView = true;
      saveSquarePoseViewSetting();
    });
  }
  if (els.squarePoseBuildButton) {
    els.squarePoseBuildButton.addEventListener('click', () => buildSquarePose().catch((error) => {
      els.squarePoseStatus.textContent = 'error';
      els.squarePoseStats.textContent = error.message;
      setStatus(`Square pose failed: ${error.message}`);
    }));
  }
  const saveBboxAnalysisSetting = () => {
    saveReview()
      .then(loadBboxDiscovery)
      .then(loadMaskbitsBboxDiscovery)
      .then(loadBboxClippingDiscovery)
      .then(loadBboxContoursDiscovery)
      .then(loadContourDiscovery)
      .then(loadSquarePoseDiscovery)
      .then(loadInstanceDiscovery)
      .catch((error) => setStatus(`bbox setting save failed: ${error.message}`));
  };
  const saveBboxVisualSetting = () => {
    saveReview()
      .then(() => scheduleBboxRender())
      .then(loadBboxClippingDiscovery)
      .catch((error) => setStatus(`bbox view save failed: ${error.message}`));
  };
  const bindBboxSlider = (input, setter, saveHandler = saveBboxAnalysisSetting) => {
    if (!input) return;
    input.addEventListener('input', () => {
      setter(input.value);
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
    });
    input.addEventListener('change', saveHandler);
  };
  bindBboxSlider(els.bboxMinPixelsInput, (value) => {
    bboxFlowState().minPixels = Math.max(1, Math.min(50000, Math.round(Number(value) || 1)));
  });
  const bindBboxNumberSlider = (slider, numberInput, setter, saveHandler = saveBboxAnalysisSetting) => {
    const applyValue = (value) => {
      setter(value);
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
    };
    if (slider) {
      slider.addEventListener('input', () => applyValue(slider.value));
      slider.addEventListener('change', saveHandler);
    }
    if (numberInput) {
      numberInput.addEventListener('input', () => applyValue(numberInput.value));
      numberInput.addEventListener('change', saveHandler);
    }
  };
  bindBboxNumberSlider(els.bboxMaxPerFrameInput, els.bboxMaxPerFrameNumber, (value) => {
    bboxFlowState().maxBboxesPerFrame = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
  });
  bindBboxSlider(els.bboxTightnessInput, (value) => {
    bboxFlowState().fitTightness = Math.max(0, Math.min(100, Math.round(Number(value) || 0)));
  });
  bindBboxSlider(els.bboxTargetAspectInput, (value) => {
    bboxFlowState().targetAspect = Math.max(0.1, Math.min(10, Number(value) || 1));
  });
  bindBboxSlider(els.bboxAspectToleranceInput, (value) => {
    bboxFlowState().aspectTolerance = Math.max(0, Math.min(2, Number(value) || 0));
  });
  bindBboxSlider(els.bboxQuadThicknessInput, (value) => {
    bboxFlowState().quadThicknessPx = Math.max(1, Math.min(80, Math.round(Number(value) || 1)));
  });
  bindBboxSlider(els.bboxEdgeCoverageInput, (value) => {
    bboxFlowState().edgeCoverageMin = clamp01(Number(value) || 0);
  });
  bindBboxSlider(els.bboxCornerMinPixelsInput, (value) => {
    bboxFlowState().cornerMinPixels = Math.max(0, Math.min(10000, Math.round(Number(value) || 0)));
  });
  bindBboxSlider(els.bboxVoidOverlapInput, (value) => {
    bboxFlowState().voidOverlapMaxRatio = clamp01(Number(value) || 0);
  });
  bindBboxSlider(els.bboxVoidMinPixelsInput, (value) => {
    bboxFlowState().voidOverlapMinPixels = Math.max(1, Math.min(50000, Math.round(Number(value) || 1)));
  });
  bindBboxSlider(els.bboxQuadOpacityInput, (value) => {
    bboxFlowState().quadOverlayOpacity = clamp01((Number(value) || 0) / 100);
  }, saveBboxVisualSetting);
  bindBboxSlider(els.fovClipMarginInput, (value) => {
    bboxFlowState().fovClip.marginPx = Math.max(0, Math.min(64, Math.round(Number(value) || 0)));
  });
  bindBboxSlider(els.fovClipMinPixelsInput, (value) => {
    bboxFlowState().fovClip.minContactPixels = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
  });
  bindBboxSlider(els.fovClipMinRatioInput, (value) => {
    bboxFlowState().fovClip.minContactRatio = clamp01(Number(value) || 0);
  });
  if (els.bboxQuadToggle) {
    els.bboxQuadToggle.addEventListener('change', () => {
      bboxFlowState().quadFitEnabled = Boolean(els.bboxQuadToggle.checked);
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
      saveBboxAnalysisSetting();
    });
  }
  if (els.bboxFitModeSelect) {
    els.bboxFitModeSelect.addEventListener('change', () => {
      bboxFlowState().quadFitMode = els.bboxFitModeSelect.value;
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
      saveBboxAnalysisSetting();
    });
  }
  if (els.bboxShowRawToggle) {
    els.bboxShowRawToggle.addEventListener('change', () => {
      bboxFlowState().showRawBboxes = Boolean(els.bboxShowRawToggle.checked);
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
      saveBboxVisualSetting();
    });
  }
  if (els.bboxViewModeSelect) {
    els.bboxViewModeSelect.addEventListener('change', () => {
      bboxFlowState().viewMode = els.bboxViewModeSelect.value;
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
      saveBboxVisualSetting();
    });
  }
  if (els.fovClipToggle) {
    els.fovClipToggle.addEventListener('change', () => {
      bboxFlowState().fovClip.enabled = Boolean(els.fovClipToggle.checked);
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
      saveBboxAnalysisSetting();
    });
  }
  if (els.fovClipOverlayToggle) {
    els.fovClipOverlayToggle.addEventListener('change', () => {
      bboxFlowState().fovClip.showOverlay = Boolean(els.fovClipOverlayToggle.checked);
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
      saveBboxVisualSetting();
    });
  }
  if (els.fovClipBboxTouchToggle) {
    els.fovClipBboxTouchToggle.addEventListener('change', () => {
      bboxFlowState().fovClip.requireBboxTouch = Boolean(els.fovClipBboxTouchToggle.checked);
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
      saveBboxAnalysisSetting();
    });
  }
  if (els.fovClipWarnOnlyToggle) {
    els.fovClipWarnOnlyToggle.addEventListener('change', () => {
      bboxFlowState().fovClip.warnOnly = Boolean(els.fovClipWarnOnlyToggle.checked);
      state.dirtyView = true;
      renderBboxControls();
      scheduleBboxRender();
      saveBboxAnalysisSetting();
    });
  }
  els.bboxBuildButton.addEventListener('click', () => buildBboxes().catch((error) => {
    els.bboxStatus.textContent = 'error';
    els.bboxStats.textContent = error.message;
    setStatus(`BBox build failed: ${error.message}`);
  }));
  if (els.maskbitsBboxBuildButton) {
    els.maskbitsBboxBuildButton.addEventListener('click', () => buildMaskbitsBboxes().catch((error) => {
      if (els.maskbitsBboxStats) els.maskbitsBboxStats.textContent = error.message;
      setStatus(`Maskbits bbox build failed: ${error.message}`);
    }));
  }
  if (els.bboxClippingBuildButton) {
    els.bboxClippingBuildButton.addEventListener('click', () => buildBboxClipping().catch((error) => {
      if (els.bboxClippingStats) els.bboxClippingStats.textContent = error.message;
      setStatus(`BBox clipping build failed: ${error.message}`);
    }));
  }
  if (els.bboxContourBuildButton) {
    els.bboxContourBuildButton.addEventListener('click', () => buildBboxContours().catch((error) => {
      if (els.bboxContourStats) els.bboxContourStats.textContent = error.message;
      setStatus(`BBox contour build failed: ${error.message}`);
    }));
  }
  const saveInstanceAnalysisSetting = () => {
    saveReview()
      .then(loadInstanceDiscovery)
      .catch((error) => setStatus(`instance setting save failed: ${error.message}`));
  };
  const saveInstanceVisualSetting = () => {
    saveReview()
      .then(() => {
        renderInstanceControls();
        scheduleBboxRender();
      })
      .catch((error) => setStatus(`instance view save failed: ${error.message}`));
  };
  const bindInstanceSlider = (input, setter, saveHandler = saveInstanceAnalysisSetting) => {
    if (!input) return;
    input.addEventListener('input', () => {
      setter(input.value);
      state.dirtyView = true;
      renderInstanceControls();
      scheduleBboxRender();
    });
    input.addEventListener('change', saveHandler);
  };
  bindInstanceSlider(els.instanceMaxGapInput, (value) => {
    instanceTrackingState().maxFrameGap = Math.max(0, Math.min(120, Math.round(Number(value) || 0)));
  });
  bindInstanceSlider(els.instanceMinScoreInput, (value) => {
    instanceTrackingState().minAssociationScore = clamp01(Number(value) || 0);
  });
  bindInstanceSlider(els.instanceMax2dInput, (value) => {
    instanceTrackingState().max2dDistancePx = Math.max(1, Math.min(640, Number(value) || 1));
  });
  bindInstanceSlider(els.instanceMax3dInput, (value) => {
    instanceTrackingState().max3dDistanceM = Math.max(0.1, Math.min(50, Number(value) || 0.1));
  });
  bindInstanceSlider(els.instanceMaxRpyInput, (value) => {
    instanceTrackingState().maxRpyDeltaDeg = Math.max(1, Math.min(180, Number(value) || 1));
  });
  bindInstanceSlider(els.instanceSplitScoreInput, (value) => {
    instanceTrackingState().splitCandidateScore = clamp01(Number(value) || 0);
  });
  bindInstanceSlider(els.instanceTrailInput, (value) => {
    instanceTrackingState().trailLengthFrames = Math.max(0, Math.min(240, Math.round(Number(value) || 0)));
  }, saveInstanceVisualSetting);
  bindInstanceSlider(els.instanceCandidateLimitInput, (value) => {
    instanceTrackingState().debugCandidateLimit = Math.max(0, Math.min(25, Math.round(Number(value) || 0)));
  });
  if (els.instanceToggle) {
    els.instanceToggle.addEventListener('change', () => {
      instanceTrackingState().enabled = Boolean(els.instanceToggle.checked);
      state.dirtyView = true;
      renderInstanceControls();
      scheduleBboxRender();
      saveInstanceVisualSetting();
    });
  }
  if (els.instanceOverlayToggle) {
    els.instanceOverlayToggle.addEventListener('change', () => {
      instanceTrackingState().showOverlay = Boolean(els.instanceOverlayToggle.checked);
      state.dirtyView = true;
      renderInstanceControls();
      scheduleBboxRender();
      saveInstanceVisualSetting();
    });
  }
  if (els.instanceLabelsToggle) {
    els.instanceLabelsToggle.addEventListener('change', () => {
      instanceTrackingState().showLabels = Boolean(els.instanceLabelsToggle.checked);
      state.dirtyView = true;
      renderInstanceControls();
      scheduleBboxRender();
      saveInstanceVisualSetting();
    });
  }
  if (els.instanceLinksToggle) {
    els.instanceLinksToggle.addEventListener('change', () => {
      instanceTrackingState().showLinks = Boolean(els.instanceLinksToggle.checked);
      state.dirtyView = true;
      renderInstanceControls();
      scheduleBboxRender();
      saveInstanceVisualSetting();
    });
  }
  if (els.instanceCandidatesToggle) {
    els.instanceCandidatesToggle.addEventListener('change', () => {
      instanceTrackingState().showCandidates = Boolean(els.instanceCandidatesToggle.checked);
      state.dirtyView = true;
      renderInstanceControls();
      scheduleBboxRender();
      saveInstanceVisualSetting();
    });
  }
  if (els.instanceReadoutToggle) {
    els.instanceReadoutToggle.addEventListener('change', () => {
      instanceTrackingState().showReadout = Boolean(els.instanceReadoutToggle.checked);
      state.dirtyView = true;
      renderInstanceControls();
      saveInstanceVisualSetting();
    });
  }
  if (els.instancePoseSourceSelect) {
    els.instancePoseSourceSelect.addEventListener('change', () => {
      const value = els.instancePoseSourceSelect.value;
      instanceTrackingState().poseSource = value === 'none' ? 'none' : 'poseFit';
      state.dirtyView = true;
      renderInstanceControls();
      scheduleSquarePoseRender();
      saveInstanceAnalysisSetting();
    });
  }
  if (els.instanceBuildButton) {
    els.instanceBuildButton.addEventListener('click', () => buildInstances().catch((error) => {
      if (els.instanceStatus) els.instanceStatus.textContent = 'error';
      if (els.instanceStats) els.instanceStats.textContent = error.message;
      setStatus(`Instance provenance failed: ${error.message}`);
    }));
  }
  const saveContourAnalysisSetting = () => {
    saveReview()
      .then(loadContourDiscovery)
      .then(loadBboxContoursDiscovery)
      .then(loadSquarePoseDiscovery)
      .catch((error) => setStatus(`contour setting save failed: ${error.message}`));
  };
  const saveContourVisualSetting = () => {
    saveReview()
      .then(() => scheduleBboxRender())
      .catch((error) => setStatus(`contour view save failed: ${error.message}`));
  };
  const bindContourSlider = (input, setter, saveHandler = saveContourAnalysisSetting, invalidateLive = true) => {
    if (!input) return;
    input.addEventListener('input', () => {
      setter(input.value);
      if (invalidateLive) clearBboxContourFrameCache();
      state.dirtyView = true;
      renderContourControls();
      scheduleBboxRender();
    });
    input.addEventListener('change', saveHandler);
  };
  if (els.contourToggle) {
    els.contourToggle.addEventListener('change', () => {
      contourHierarchyState().show = Boolean(els.contourToggle.checked);
      state.dirtyView = true;
      renderContourControls();
      scheduleBboxRender();
      saveContourVisualSetting();
    });
  }
  if (els.contourSourceSelect) {
    els.contourSourceSelect.addEventListener('change', () => {
      contourHierarchyState().maskSource = els.contourSourceSelect.value === '002-only' ? '002-only' : 'enabled-layers';
      clearBboxContourFrameCache();
      state.dirtyView = true;
      renderContourControls();
      scheduleBboxRender();
      saveContourAnalysisSetting();
    });
  }
  bindContourSlider(els.contourMinOuterInput, (value) => {
    contourHierarchyState().minOuterAreaPx = Math.max(1, Math.min(500000, Math.round(Number(value) || 1)));
  });
  bindContourSlider(els.contourMinVoidInput, (value) => {
    contourHierarchyState().minVoidAreaPx = Math.max(1, Math.min(500000, Math.round(Number(value) || 1)));
  });
  bindContourSlider(els.contourMaxVoidsInput, (value) => {
    contourHierarchyState().maxVoidsPerBbox = Math.max(0, Math.min(200, Math.round(Number(value) || 0)));
  });
  bindContourSlider(els.contourSimplifyInput, (value) => {
    contourHierarchyState().simplifyEpsilonPx = Math.max(0, Math.min(32, Number(value) || 0));
  });
  bindContourSlider(els.contourCloseInput, (value) => {
    contourHierarchyState().closeRadiusPx = Math.max(0, Math.min(32, Math.round(Number(value) || 0)));
  });
  bindContourSlider(els.contourOpenInput, (value) => {
    contourHierarchyState().openRadiusPx = Math.max(0, Math.min(32, Math.round(Number(value) || 0)));
  });
  bindContourSlider(els.contourNotchInput, (value) => {
    contourHierarchyState().notchProximityPx = Math.max(0, Math.min(128, Math.round(Number(value) || 0)));
  });
  bindContourSlider(els.contourLineThicknessInput, (value) => {
    contourHierarchyState().lineThicknessPx = Math.max(1, Math.min(8, Math.round(Number(value) || 1)));
  }, saveContourVisualSetting, false);
  bindContourSlider(els.contourOpacityInput, (value) => {
    contourHierarchyState().opacity = clamp01((Number(value) || 0) / 100);
  }, saveContourVisualSetting, false);
  if (els.contourLabelsToggle) {
    els.contourLabelsToggle.addEventListener('change', () => {
      contourHierarchyState().showLabels = Boolean(els.contourLabelsToggle.checked);
      state.dirtyView = true;
      renderContourControls();
      scheduleBboxRender();
      saveContourVisualSetting();
    });
  }
  if (els.contourSmallToggle) {
    els.contourSmallToggle.addEventListener('change', () => {
      contourHierarchyState().includeSmallContours = Boolean(els.contourSmallToggle.checked);
      clearBboxContourFrameCache();
      state.dirtyView = true;
      renderContourControls();
      scheduleBboxRender();
      saveContourAnalysisSetting();
    });
  }
  if (els.contourBuildButton) {
    els.contourBuildButton.addEventListener('click', () => buildContours().catch((error) => {
      els.contourStatus.textContent = 'error';
      els.contourStats.textContent = error.message;
      setStatus(`Contour build failed: ${error.message}`);
    }));
  }
  const saveCornerAnalysisSetting = () => {
    saveReview()
      .then(loadCornerDiscovery)
      .then(loadSquarePoseDiscovery)
      .catch((error) => setStatus(`corner setting save failed: ${error.message}`));
  };
  const saveCornerVisualSetting = () => {
    saveReview()
      .then(() => scheduleBboxRender())
      .catch((error) => setStatus(`corner view save failed: ${error.message}`));
  };
  const bindCornerSlider = (input, setter, saveHandler = saveCornerAnalysisSetting) => {
    if (!input) return;
    input.addEventListener('input', () => {
      setter(input.value);
      state.dirtyView = true;
      renderCornerControls();
      scheduleBboxRender();
    });
    input.addEventListener('change', saveHandler);
  };
  const bindCornerPointControl = (slider, numberInput, setter, saveHandler = saveCornerAnalysisSetting) => {
    const applyValue = (value) => {
      setter(value);
      state.dirtyView = true;
      renderCornerControls();
      scheduleBboxRender();
    };
    if (slider) {
      slider.addEventListener('input', () => applyValue(slider.value));
      slider.addEventListener('change', saveHandler);
    }
    if (numberInput) {
      numberInput.addEventListener('input', () => applyValue(numberInput.value));
      numberInput.addEventListener('change', saveHandler);
    }
  };
  const clampCornerAngle = (value) => Math.max(1, Math.min(179, Math.round(Number(value) || 1)));
  const clampCornerPointCount = (value) => Math.max(0, Math.min(5000, Math.round(Number(value) || 0)));
  const bindCornerTypeSlider = (typeKey, suffix, setter) => {
    const def = CORNER_TYPE_DEFS[typeKey];
    const input = document.getElementById(`corner${def.label}${suffix}Input`);
    bindCornerSlider(input, (value) => {
      setter(cornerFlowState().types[typeKey], value);
    });
  };
  const bindCornerTypePointControl = (typeKey, suffix, setter) => {
    const def = CORNER_TYPE_DEFS[typeKey];
    const slider = document.getElementById(`corner${def.label}${suffix}Input`);
    const numberInput = document.getElementById(`corner${def.label}${suffix}Number`);
    bindCornerPointControl(slider, numberInput, (value) => {
      setter(cornerFlowState().types[typeKey], value);
    });
  };
  for (const typeKey of CORNER_TYPE_KEYS) {
    const def = CORNER_TYPE_DEFS[typeKey];
    const showToggle = document.getElementById(`corner${def.label}ShowToggle`);
    if (showToggle) {
      showToggle.addEventListener('change', () => {
        cornerFlowState()[`show${def.label}`] = Boolean(showToggle.checked);
        state.dirtyView = true;
        renderCornerControls();
        scheduleBboxRender();
        saveCornerVisualSetting();
      });
    }
    if (def.category === 'void') {
      bindCornerTypeSlider(typeKey, 'MinArea', (settings, value) => {
        settings.minAreaPx = Math.max(1, Math.min(100000, Math.round(Number(value) || 1)));
      });
    }
    bindCornerTypeSlider(typeKey, 'Radius', (settings, value) => {
      settings.radiusPx = Math.max(2, Math.min(64, Math.round(Number(value) || 2)));
    });
    bindCornerTypeSlider(typeKey, 'MinAngle', (settings, value) => {
      settings.minAngleDeg = clampCornerAngle(value);
      if (settings.minAngleDeg > settings.maxAngleDeg) settings.maxAngleDeg = settings.minAngleDeg;
    });
    bindCornerTypeSlider(typeKey, 'MaxAngle', (settings, value) => {
      settings.maxAngleDeg = clampCornerAngle(value);
      if (settings.maxAngleDeg < settings.minAngleDeg) settings.minAngleDeg = settings.maxAngleDeg;
    });
    bindCornerTypeSlider(typeKey, 'Support', (settings, value) => {
      settings.minSupportPixels = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
    });
    bindCornerTypeSlider(typeKey, 'Epsilon', (settings, value) => {
      settings.contourEpsilonPx = Math.max(0.25, Math.min(24, Number(value) || 0.25));
    });
    bindCornerTypeSlider(typeKey, 'Distance', (settings, value) => {
      settings.minDistancePx = Math.max(1, Math.min(80, Math.round(Number(value) || 1)));
    });
    bindCornerTypePointControl(typeKey, 'MinPoints', (settings, value) => {
      settings.minPoints = clampCornerPointCount(value);
      if (settings.minPoints > settings.maxPoints) settings.maxPoints = settings.minPoints;
    });
    bindCornerTypePointControl(typeKey, 'MaxPoints', (settings, value) => {
      settings.maxPoints = clampCornerPointCount(value);
      if (settings.maxPoints < settings.minPoints) settings.minPoints = settings.maxPoints;
    });
  }
  bindCornerSlider(els.cornerHullRadiusInput, (value) => {
    cornerFlowState().hull.radiusPx = Math.max(2, Math.min(64, Math.round(Number(value) || 2)));
  });
  bindCornerSlider(els.cornerHullMinAngleInput, (value) => {
    const settings = cornerFlowState().hull;
    settings.minAngleDeg = clampCornerAngle(value);
    if (settings.minAngleDeg > settings.maxAngleDeg) settings.maxAngleDeg = settings.minAngleDeg;
  });
  bindCornerSlider(els.cornerHullMaxAngleInput, (value) => {
    const settings = cornerFlowState().hull;
    settings.maxAngleDeg = clampCornerAngle(value);
    if (settings.maxAngleDeg < settings.minAngleDeg) settings.minAngleDeg = settings.maxAngleDeg;
  });
  bindCornerSlider(els.cornerHullMinSupportInput, (value) => {
    cornerFlowState().hull.minSupportPixels = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
  });
  bindCornerSlider(els.cornerHullEpsilonInput, (value) => {
    cornerFlowState().hull.contourEpsilonPx = Math.max(0.25, Math.min(24, Number(value) || 0.25));
  });
  bindCornerSlider(els.cornerHullDistanceInput, (value) => {
    cornerFlowState().hull.minDistancePx = Math.max(1, Math.min(80, Math.round(Number(value) || 1)));
  });
  bindCornerSlider(els.cornerHullMaxInput, (value) => {
    cornerFlowState().hull.maxCornersPerBbox = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
  });
  bindCornerPointControl(els.cornerHullWhiteMinInput, els.cornerHullWhiteMinNumber, (value) => {
    const settings = cornerFlowState().hull;
    settings.minMaskInsidePoints = clampCornerPointCount(value);
    if (settings.minMaskInsidePoints > settings.maxMaskInsidePoints) settings.maxMaskInsidePoints = settings.minMaskInsidePoints;
  });
  bindCornerPointControl(els.cornerHullWhiteMaxInput, els.cornerHullWhiteMaxNumber, (value) => {
    const settings = cornerFlowState().hull;
    settings.maxMaskInsidePoints = clampCornerPointCount(value);
    if (settings.maxMaskInsidePoints < settings.minMaskInsidePoints) settings.minMaskInsidePoints = settings.maxMaskInsidePoints;
  });
  bindCornerPointControl(els.cornerHullBlackMinInput, els.cornerHullBlackMinNumber, (value) => {
    const settings = cornerFlowState().hull;
    settings.minMaskOutsidePoints = clampCornerPointCount(value);
    if (settings.minMaskOutsidePoints > settings.maxMaskOutsidePoints) settings.maxMaskOutsidePoints = settings.minMaskOutsidePoints;
  });
  bindCornerPointControl(els.cornerHullBlackMaxInput, els.cornerHullBlackMaxNumber, (value) => {
    const settings = cornerFlowState().hull;
    settings.maxMaskOutsidePoints = clampCornerPointCount(value);
    if (settings.maxMaskOutsidePoints < settings.minMaskOutsidePoints) settings.minMaskOutsidePoints = settings.maxMaskOutsidePoints;
  });
  bindCornerSlider(els.cornerVoidMinAreaInput, (value) => {
    cornerFlowState().void.minAreaPx = Math.max(1, Math.min(100000, Math.round(Number(value) || 1)));
  });
  bindCornerSlider(els.cornerVoidRadiusInput, (value) => {
    cornerFlowState().void.radiusPx = Math.max(2, Math.min(64, Math.round(Number(value) || 2)));
  });
  bindCornerSlider(els.cornerVoidMinAngleInput, (value) => {
    const settings = cornerFlowState().void;
    settings.minAngleDeg = clampCornerAngle(value);
    if (settings.minAngleDeg > settings.maxAngleDeg) settings.maxAngleDeg = settings.minAngleDeg;
  });
  bindCornerSlider(els.cornerVoidMaxAngleInput, (value) => {
    const settings = cornerFlowState().void;
    settings.maxAngleDeg = clampCornerAngle(value);
    if (settings.maxAngleDeg < settings.minAngleDeg) settings.minAngleDeg = settings.maxAngleDeg;
  });
  bindCornerSlider(els.cornerVoidMinSupportInput, (value) => {
    cornerFlowState().void.minSupportPixels = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
  });
  bindCornerSlider(els.cornerVoidEpsilonInput, (value) => {
    cornerFlowState().void.contourEpsilonPx = Math.max(0.25, Math.min(24, Number(value) || 0.25));
  });
  bindCornerSlider(els.cornerVoidDistanceInput, (value) => {
    cornerFlowState().void.minDistancePx = Math.max(1, Math.min(80, Math.round(Number(value) || 1)));
  });
  bindCornerSlider(els.cornerVoidMaxInput, (value) => {
    cornerFlowState().void.maxCornersPerVoid = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
  });
  bindCornerPointControl(els.cornerVoidOrangeMinInput, els.cornerVoidOrangeMinNumber, (value) => {
    const settings = cornerFlowState().void;
    settings.minMaskInsidePoints = clampCornerPointCount(value);
    if (settings.minMaskInsidePoints > settings.maxMaskInsidePoints) settings.maxMaskInsidePoints = settings.minMaskInsidePoints;
  });
  bindCornerPointControl(els.cornerVoidOrangeMaxInput, els.cornerVoidOrangeMaxNumber, (value) => {
    const settings = cornerFlowState().void;
    settings.maxMaskInsidePoints = clampCornerPointCount(value);
    if (settings.maxMaskInsidePoints < settings.minMaskInsidePoints) settings.minMaskInsidePoints = settings.maxMaskInsidePoints;
  });
  bindCornerPointControl(els.cornerVoidGreenMinInput, els.cornerVoidGreenMinNumber, (value) => {
    const settings = cornerFlowState().void;
    settings.minMaskOutsidePoints = clampCornerPointCount(value);
    if (settings.minMaskOutsidePoints > settings.maxMaskOutsidePoints) settings.maxMaskOutsidePoints = settings.minMaskOutsidePoints;
  });
  bindCornerPointControl(els.cornerVoidGreenMaxInput, els.cornerVoidGreenMaxNumber, (value) => {
    const settings = cornerFlowState().void;
    settings.maxMaskOutsidePoints = clampCornerPointCount(value);
    if (settings.maxMaskOutsidePoints < settings.minMaskOutsidePoints) settings.minMaskOutsidePoints = settings.maxMaskOutsidePoints;
  });
  bindCornerSlider(els.cornerOpacityInput, (value) => {
    cornerFlowState().overlayOpacity = clamp01((Number(value) || 0) / 100);
  }, saveCornerVisualSetting);
  if (els.cornerShowHullToggle) {
    els.cornerShowHullToggle.addEventListener('change', () => {
      cornerFlowState().showHull = Boolean(els.cornerShowHullToggle.checked);
      state.dirtyView = true;
      renderCornerControls();
      scheduleBboxRender();
      saveCornerVisualSetting();
    });
  }
  if (els.cornerShowVoidToggle) {
    els.cornerShowVoidToggle.addEventListener('change', () => {
      cornerFlowState().showVoid = Boolean(els.cornerShowVoidToggle.checked);
      state.dirtyView = true;
      renderCornerControls();
      scheduleBboxRender();
      saveCornerVisualSetting();
    });
  }
  if (els.cornerShowAmbiguousToggle) {
    els.cornerShowAmbiguousToggle.addEventListener('change', () => {
      cornerFlowState().showAmbiguous = Boolean(els.cornerShowAmbiguousToggle.checked);
      state.dirtyView = true;
      renderCornerControls();
      scheduleBboxRender();
      saveCornerVisualSetting();
    });
  }
  const saveCornerAgreementVisualSetting = () => {
    saveReview()
      .then(() => scheduleBboxRender())
      .catch((error) => setStatus(`corner agreement save failed: ${error.message}`));
  };
  const updateCornerAgreementView = () => {
    state.dirtyView = true;
    renderCornerAgreementControls();
    scheduleBboxRender();
  };
  const bindCornerAgreementSlider = (input, setter) => {
    if (!input) return;
    input.addEventListener('input', () => {
      setter(input.value);
      updateCornerAgreementView();
    });
    input.addEventListener('change', saveCornerAgreementVisualSetting);
  };
  if (els.cornerAgreementToggle) {
    els.cornerAgreementToggle.addEventListener('change', () => {
      cornerAgreementState().show = Boolean(els.cornerAgreementToggle.checked);
      updateCornerAgreementView();
      saveCornerAgreementVisualSetting();
    });
  }
  bindCornerAgreementSlider(els.cornerAgreementMaxDistanceInput, (value) => {
    cornerAgreementState().maxDistancePx = Math.max(4, Math.min(220, Math.round(Number(value) || 4)));
  });
  bindCornerAgreementSlider(els.cornerAgreementWhiteToleranceInput, (value) => {
    cornerAgreementState().whiteToleranceDeg = Math.max(0, Math.min(120, Math.round(Number(value) || 0)));
  });
  bindCornerAgreementSlider(els.cornerAgreementAngleToleranceInput, (value) => {
    cornerAgreementState().angleToleranceDeg = Math.max(0, Math.min(120, Math.round(Number(value) || 0)));
  });
  bindCornerAgreementSlider(els.cornerAgreementMinScoreInput, (value) => {
    cornerAgreementState().minScore = clamp01(Number(value) || 0);
  });
  bindCornerAgreementSlider(els.cornerAgreementMaxLinksInput, (value) => {
    cornerAgreementState().maxLinksPerCorner = Math.max(1, Math.min(4, Math.round(Number(value) || 1)));
  });
  bindCornerAgreementSlider(els.cornerAgreementLineThicknessInput, (value) => {
    cornerAgreementState().lineThicknessPx = Math.max(1, Math.min(30, Math.round(Number(value) || 1)));
  });
  bindCornerAgreementSlider(els.cornerAgreementOpacityInput, (value) => {
    cornerAgreementState().opacity = clamp01((Number(value) || 0) / 100);
  });
  if (els.cornerAgreementStructureToggle) {
    els.cornerAgreementStructureToggle.addEventListener('change', () => {
      cornerAgreementState().showStructures = Boolean(els.cornerAgreementStructureToggle.checked);
      updateCornerAgreementView();
      saveCornerAgreementVisualSetting();
    });
  }
  bindCornerAgreementSlider(els.cornerAgreementOppositeToleranceInput, (value) => {
    cornerAgreementState().oppositeToleranceDeg = Math.max(0, Math.min(90, Math.round(Number(value) || 0)));
  });
  const saveLayer002QuadVisualSetting = () => {
    saveReview()
      .then(() => scheduleBboxRender())
      .catch((error) => setStatus(`layer 002 quad save failed: ${error.message}`));
  };
  const updateLayer002QuadView = () => {
    state.dirtyView = true;
    renderLayer002QuadControls();
    scheduleBboxRender();
  };
  const bindLayer002QuadSlider = (input, setter) => {
    if (!input) return;
    input.addEventListener('input', () => {
      setter(input.value);
      updateLayer002QuadView();
    });
    input.addEventListener('change', saveLayer002QuadVisualSetting);
  };
  if (els.layer002QuadToggle) {
    els.layer002QuadToggle.addEventListener('change', () => {
      layer002QuadState().show = Boolean(els.layer002QuadToggle.checked);
      updateLayer002QuadView();
      saveLayer002QuadVisualSetting();
    });
  }
  const bindLayer002QuadSelect = (select, setter) => {
    if (!select) return;
    select.addEventListener('change', () => {
      setter(select.value);
      updateLayer002QuadView();
      saveLayer002QuadVisualSetting();
    });
  };
  bindLayer002QuadSelect(els.layer002TraceSupportSelect, (value) => {
    layer002QuadState().supportMode = value === 'any-mask' ? 'any-mask' : '002-only';
  });
  bindLayer002QuadSelect(els.layer002TraceForbiddenSelect, (value) => {
    layer002QuadState().forbiddenMode = value === 'non-mask' ? 'non-mask' : 'non-002';
  });
  bindLayer002QuadSlider(els.layer002QuadMaxInput, (value) => {
    layer002QuadState().maxQuadsPerFrame = Math.max(1, Math.min(50, Math.round(Number(value) || 1)));
  });
  bindLayer002QuadSlider(els.layer002QuadMinPixelsInput, (value) => {
    layer002QuadState().minPixels = Math.max(1, Math.min(5000, Math.round(Number(value) || 1)));
  });
  bindLayer002QuadSlider(els.layer002QuadEdgeThicknessInput, (value) => {
    const settings = layer002QuadState();
    settings.hullEdgeBandPx = Math.max(0, Math.min(24, Math.round(Number(value) || 0)));
    settings.edgeBandPx = settings.hullEdgeBandPx;
  });
  bindLayer002QuadSlider(els.layer002QuadMinCoverageInput, (value) => {
    const settings = layer002QuadState();
    settings.hullMinEdgeCoverage = clamp01(Number(value) || 0);
    settings.minEdgeCoverage = settings.hullMinEdgeCoverage;
  });
  bindLayer002QuadSlider(els.layer002QuadMaxGapInput, (value) => {
    const settings = layer002QuadState();
    settings.hullMaxUnsupportedGapPx = Math.max(0, Math.min(80, Math.round(Number(value) || 0)));
    settings.maxUnsupportedGapPx = settings.hullMaxUnsupportedGapPx;
  });
  bindLayer002QuadSlider(els.layer002QuadAngleSweepInput, (value) => {
    layer002QuadState().inwardMaxPx = Math.max(0, Math.min(220, Math.round(Number(value) || 0)));
  });
  bindLayer002QuadSlider(els.layer002QuadAngleStepInput, (value) => {
    layer002QuadState().insetStepPx = Math.max(1, Math.min(20, Math.round(Number(value) || 1)));
  });
  bindLayer002QuadSlider(els.layer002QuadInsetInput, (value) => {
    const settings = layer002QuadState();
    settings.hullAngleSweepDeg = Math.max(0, Math.min(90, Math.round(Number(value) || 0)));
    settings.angleSweepDeg = settings.hullAngleSweepDeg;
  });
  bindLayer002QuadSlider(els.layer002TraceAngleStepInput, (value) => {
    const settings = layer002QuadState();
    settings.hullAngleStepDeg = Math.max(1, Math.min(15, Math.round(Number(value) || 1)));
    settings.angleStepDeg = settings.hullAngleStepDeg;
  });
  bindLayer002QuadSlider(els.layer002TraceAspectToleranceInput, (value) => {
    layer002QuadState().aspectTolerance = Math.max(0, Math.min(2, Number(value) || 0));
  });
  bindLayer002QuadSlider(els.layer002QuadLineThicknessInput, (value) => {
    layer002QuadState().lineThicknessPx = Math.max(1, Math.min(16, Math.round(Number(value) || 1)));
  });
  bindLayer002QuadSlider(els.layer002QuadOpacityInput, (value) => {
    layer002QuadState().opacity = clamp01((Number(value) || 0) / 100);
  });
  if (els.layer002QuadRejectedToggle) {
    els.layer002QuadRejectedToggle.addEventListener('change', () => {
      layer002QuadState().showRejected = Boolean(els.layer002QuadRejectedToggle.checked);
      updateLayer002QuadView();
      saveLayer002QuadVisualSetting();
    });
  }
  els.cornerBuildButton.addEventListener('click', () => buildCorners().catch((error) => {
    els.cornerStatus.textContent = 'error';
    els.cornerStats.textContent = error.message;
    setStatus(`Corner build failed: ${error.message}`);
  }));
}

async function init() {
  setStatus('loading config');
  const payload = await fetchJson(`${API}/config`);
  state.config = payload.config;
  state.precompute = payload.precompute;
  state.dependencyStatus = payload.dependencyStatus || null;
  state.bbox.discovery = payload.bboxes || null;
  state.maskbitsBbox.discovery = payload.maskbitsBboxes || null;
  state.bboxClipping.discovery = payload.bboxClipping || null;
  state.bboxContours.discovery = payload.bboxContours || null;
  state.contour.discovery = payload.contours || null;
  state.corner.discovery = payload.corners || null;
  state.squarePose.discovery = payload.poseEstimation || payload.squarePose || null;
  state.instances.discovery = payload.instanceTracking || payload.instances || null;
  state.rules = await fetchJson(`${API}/rules`);
  await loadReview();
  applySourceOpacityControl();
  renderAssetSummary(payload.readOnlyAssets);
  applySourceDiscovery(payload.readOnlyAssets);
  await refreshPipelineSources().catch((error) => setStatus(`source discovery failed: ${error.message}`));
  renderBboxControls();
  renderBboxClippingControls();
  renderContourControls();
  renderBboxContourControls();
  renderCornerControls();
  renderSquarePoseControls();
  renderInstanceControls();
  attachOptionInfoButtons();
  const depsOk = payload.dependencyStatus?.ok === true;
  if (hasRequestedFrameParam()) {
    state.source.frameIndex = requestedInitialFrameIndex();
    state.source.followLatest = false;
  }
  renderSourceControls();
  installEvents();
  await refreshPipelineStatus().catch((error) => setStatus(`pipeline status failed: ${error.message}`));
  startPipelinePolling();
  renderClassList();
  await loadColorTable();
  await loadMemory();
  await loadBboxManifest().catch((error) => setStatus(`BBox load failed: ${error.message}`));
  await loadMaskbitsBboxManifest().catch((error) => setStatus(`Maskbits bbox load failed: ${error.message}`));
  await loadBboxClippingManifest().catch((error) => setStatus(`BBox clipping load failed: ${error.message}`));
  await loadBboxContoursManifest().catch((error) => setStatus(`BBox contour load failed: ${error.message}`));
  await loadContourManifest().catch((error) => setStatus(`Contour load failed: ${error.message}`));
  await loadCornerManifest().catch((error) => setStatus(`Corner load failed: ${error.message}`));
  await loadSquarePoseManifest().catch((error) => setStatus(`Square pose load failed: ${error.message}`));
  await loadInstanceManifest().catch((error) => setStatus(`Instance provenance load failed: ${error.message}`));
  renderBboxControls();
  renderBboxClippingControls();
  renderContourControls();
  renderBboxContourControls();
  renderCornerControls();
  renderSquarePoseControls();
  renderInstanceControls();
  await loadFrame(requestedInitialFrameIndex());
  await refreshSourceManifests({ preserveIndex: hasRequestedFrameParam() });
  startSourcePolling();
  setStatus(depsOk ? 'ready' : `dependency issue: ${payload.dependencyStatus?.error || 'missing dependency'}`);
}

init().catch((error) => {
  console.error(error);
  setStatus(`startup failed: ${error.message}`);
});
