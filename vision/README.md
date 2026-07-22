# 0721Vision

0721Vision is a deterministic frame-by-frame vision pipeline with a small UI
for launching local runs and inspecting the outputs.

The product path is:

```text
source JPEG frame
  -> deterministic RGB LUT maskbits
  -> mask-derived bboxes
  -> bbox-local clipping diagnostics
  -> bbox-local contours
  -> 3D square pose fit
  -> frame-to-frame instance mapping
  -> UI visualization and JSON review
```

The UI is not the core inference engine. It starts or stops a local pipeline run
through a small launcher API, then watches the manifests written by `pipeline.py`.

## Setup

Use Python 3.10 or newer. From this directory:

```bash
pip install -r requirements.txt
python tools/check_deps.py
```

## Entrypoint

All commands below are run from inside `vision/`.

Run a finite directory or source manifest in production mode:

```bash
python main.py --source-dir path/to/frames --mode batch --no-debug
```

Run a watched directory without dropping accepted frames:

```bash
python main.py --source-dir path/to/frames --mode watch
```

Run the UI and the pipeline launcher API:

```bash
python main.py --serve-ui
```

Start the UI with a pipeline already running:

```bash
python main.py --source-dir path/to/frames --mode watch --serve-ui
```

The UI is served by default at:

```text
http://127.0.0.1:8772/0721Vision/index.html
```

## Runtime Shape

`main.py` is the public wrapper. It parses source mode, source location, output
roots, optional frame caps, and UI hosting options.

`pipeline.py` is the ordered executor. It accepts each frame, snapshots it into
the run folder, runs every enabled stage, records latency, and atomically
publishes compact per-frame outputs plus status.

The core pipeline types are:

```text
PipelineConfig
SourceAdapter
FramePacket
VisionPipeline
```

The current v1 runtime is ordered and lossless. If it falls behind a live source,
it reports backlog instead of skipping frames. Instance tracking remains ordered
because it carries state across frames.

## Frame Contract

V1 expects source JPEG frames at 640x360.

Frames can come from:

```text
directory of .jpg/.jpeg files
source manifest JSON
watched directory receiving new .jpg/.jpeg files
```

Watched-directory files are accepted only after their size and mtime are stable.
Accepted frames are copied into the pipeline run so history remains inspectable
even if the original live source overwrites files later.

In v1, `live` mode is a watched server-visible path. HTTP live points are
reserved for a future adapter.

## Outputs

In production mode, each pipeline run writes compact per-frame instance output:

```text
assets/pipeline_runs/<run_id>/
  status.json
  run_manifest.json
  latest.json
  frames/
    frame_XXXXXX.json
```

Each frame JSON is the final product output for that frame:

```text
schema: 0721vision-instance-frame.v1
kind: instance-frame-mapping-v1
frameOrdinal
frameId
instances[]
```

Debug and compatibility outputs are opt in. Use `--debug-artifacts` for
per-frame stage debug JSON and `--aggregate-debug-manifests` for legacy
cumulative manifests in the stage roots:

```text
assets/mask_bboxes_maskbits/<run_id>/bbox_manifest.json
assets/bbox_clipping/<run_id>/clipping_manifest.json
assets/bbox_contours/<run_id>/contours_manifest.json
assets/pose_estimation/<run_id>/3d_pose_fit.json
assets/instance_tracking/<run_id>/instance_mapping.json
```

The aggregate instance mapping preserves the legacy UI review shape:

```text
instances[]
frames[].observations[]
observation.instanceId
observation.instanceColor
observation.bboxPx
observation.centroidPx
observation.pose
observation.status
observation.candidates
```

## Flight Observation Bridge

The Flight-facing adapter lives inside this vision module at:

```text
src/flight_bridge/vision_observation.py
tools/emit_flight_observation.py
```

It intentionally does not import from `Flight`. The Flight module remains the
consumer contract, while vision owns the conversion from
`0721vision-instance-frame.v1` into a controller payload with:

```text
run.cycle
run.frame_id
run.sim_time_ns
gates[].id
gates[].position_xyz
gates[].position_confidence
gates[].orientation_xyz
gates[].orientation_confidence
obstacles[]
```

The current conversion assumes the camera is the local origin for each frame:

```text
camera position = 0,0,0
camera orientation = 0,0,0
instance position = camera-local meters
```

Vision pose positions are produced in OpenCV camera coordinates
`+x right, +y down, +z forward`. The bridge flips the y axis once so the Flight
payload receives `+x right, +y up, +z forward`. Orientation is passed through as
camera-relative RPY degrees for v1 and is marked in code as a handedness item to
validate before high-authority control use.

## UI Launcher API

The local server exposes:

```text
GET  /api/pipeline/status
GET  /api/pipeline/runs
POST /api/pipeline/start
POST /api/pipeline/stop
```

`POST /api/pipeline/start` accepts:

```json
{
  "mode": "batch | watch | live",
  "sourceDir": "server-visible path",
  "livePoint": null,
  "maxFrames": null
}
```

Relative paths are resolved from the `/0721Vision` app directory. UI-style paths
such as `/src/...` and `/assets/...` are also resolved inside the app directory.

## Stage Boundaries

The `/src` modules own algorithms, not orchestration.

```text
src/color_masks       exact RGB -> maskbits
src/mask_bbox         maskbits -> bboxes
src/bbox_clipping     bboxes + maskbits -> FOV clipping diagnostics
src/bbox_contours     bboxes + maskbits -> bbox-local contour hierarchy
src/pose_estimation   contours + clipping -> 3D pose fit
src/instance_tracking bboxes + pose -> stable instance IDs
```

Batch builders remain available for review and comparison, but the production
wrapper calls frame-level stage functions through `pipeline.py`.

## Performance

The production pass is ordered and deterministic. BBox clipping and bbox
contours are independent after bbox generation, so they overlap on a small
stateless worker pool by default. Instance tracking remains ordered.

Every frame status records per-stage latency in `status.json`:

```text
stageLatencyMs
meanFrameLatencyMs
behindByFrames
latestAcceptedFrame
latestCompletedFrame
```

Use `--no-parallel` only when debugging the strictly sequential path.
