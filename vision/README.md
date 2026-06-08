# Vision
`vision/src/` owns inference
`vision/tools/` owns replay, review, and sample data

## Layout

```text
vision/
  vision_entrypoint.py          # thin CLI/composition script
  src/
    pipeline.py                 # live end-to-end orchestration
    io/                         # UDP packet protocol and ingress
    cnn/                        # RGB normalization, CNN inference, logits output
    regressor/                  # raw logits to camera-local gate JSON
    landmarker/                 # landmark fusion and controller JSON
  tools/
    udp_spoof/                  # sample/replay UDP sender
    review/                     # optional capture and overlay tooling
    sample_runs/                # replay inputs and truth metadata
  tests/
```

## Data Flow

Live UDP input is configured in vision_entrypoint.py then enters through `vision.src.io`, then every stage writes to configured `--output-root`.

```text
external UDP sender
  -> vision.src.io UDP ingress
  -> vision.src.cnn
       output_root/lightmask_logits/frame_000001.bin
  -> optional vision.src.regressor
       output_root/regressor_json/frame_000001.json
       output_root/regressor_json/regressor_frames.jsonl
  -> optional vision.src.landmarker
       output_root/landmarker_controller_json/frame_000001_controller.json
       output_root/landmarker_controller_json/controller_frames.jsonl
       output_root/landmarker_state.json
  -> optional vision.tools.review
       review_output/review_manifest.json
       review_output/frames/frame_000001/rgb.jpg
       review_output/frames/frame_000001/landmarker_overlay.jpg
```

`vision_entrypoint.py` only parses CLI options, builds the config, attaches review if requested, and prints a summary. `vision/src/pipeline.py` orchestrates the execution.

## CNN Model Source

The production checkpoint and architecture are both kept with the runtime package:

```text
vision/src/cnn/cnn_last.pt
vision/src/cnn/lightmask_model.py
```

`lightmask_model.py` is the standalone inference copy of the stable lightmask architecture. It contains only the runtime contract:

```text
schema_version: gate_non_gate_lightmask_v1
input image:    640x360 RGB
output stride:  4
mask logits:    2 x 90 x 160
depth logits:   1 x 90 x 160
backbone:       mobilenet_v3_small
```

`vision/src/cnn/rgb_inference.py` imports the local runtime model directly:

```python
from vision.src.cnn.lightmask_model import MASK_CHANNELS, SCHEMA_VERSION
from vision.src.cnn.lightmask_model import build_model, load_compatible_state_dict
```

The runtime no longer imports `Convolutional_Neural_Network/training_pipeline` to construct the CNN. Training code may still use its own research/training modules, but production inference depends on the local `vision/src/cnn` model contract.

## UDP Input Contract

The live receiver binds a UDP socket with:

```bash
python -m vision.vision_entrypoint live \
  --bind-host 127.0.0.1 \
  --port 5600 \
  --output-root vision/output \
  --device auto
```

Use `--bind-host 0.0.0.0` when the sender is on another machine or interface and the host firewall allows it. Defaults are:

```text
bind host: 127.0.0.1
port:      5600
```

Each UDP datagram uses the `4.6 Vision Stream` JPEG chunk format:

```text
header: little-endian struct "<IHHIIQ" (24 bytes)

uint32 frame_id
uint16 chunk_id          # zero-based
uint16 total_chunks
uint32 jpeg_size         # full frame byte size
uint32 payload_size      # bytes in this datagram payload
uint64 sim_time_ns
bytes  payload           # JPEG chunk bytes
```

The receiver reassembles all chunks with the same `frame_id`, validates chunk counts and payload sizes, decodes the JPEG, and passes the frame to CNN inference. `--timeout-seconds` stops the receiver after socket inactivity. `--max-frames 0` means keep receiving until timeout or process exit.

## Live Runs

cnn + regressor + landmarker:

```bash
python -m vision.vision_entrypoint live \
  --bind-host 127.0.0.1 \
  --port 5600 \
  --output-root vision/output/live_001 \
  --run-landmarker \
  --top-k 5 \
  --device auto
```

`--run-landmarker` automatically runs the regressor first because landmarker consumes regressor JSONL.

## Landmarker Egress

In a live full-stage run, landmarker output paths are derived from `--output-root`:

`--top-k` controls how many current-frame official gates are emitted for the controller. The landmarker still persists fused official IDs across frames, but controller output uses the current frame's matched regressor `position_xyz`, sorts those camera-relative XYZ values by distance, and emits the nearest targets. `gates[0]` is the nearest current-frame regression target.

Per-frame controller JSON shape:

```json
{
  "run": {
    "output_dir": "vision/output/live_001/landmarker_controller_json",
    "cycle": 1,
    "frame_id": "frame_000001",
    "sim_time_ns": 0
  },
  "gates": [
    {
      "id": "gate-001w",
      "position_xyz": [12.33, 1.72, 15.99],
      "position_confidence": 0.18,
      "orientation_xyz": [0.28, 0.84, -0.46],
      "orientation_confidence": 0.18
    }
  ],
  "obstacles": []
}
```

Notes:

- `position_xyz` is the current frame's camera-local regressor output in meters.
- `orientation_xyz` is either a 3-value direction vector or `null` when unavailable.
- `orientation_confidence` is always present.
- `obstacles` is currently an empty list.
- `controller_frames.jsonl` contains the same per-frame payloads, one JSON object per line.
- `landmarker_state.json` persists fused official landmark IDs across frames/runs.

## Review Capture

Review is optional and observes the same production path. It does not create a separate inference flow.

```bash
python -m vision.vision_entrypoint live \
  --bind-host 127.0.0.1 \
  --port 5600 \
  --output-root vision/output/live_001 \
  --run-landmarker \
  --review \
  --review-output vision/output/reviews/live_001
```

## Replay And Spoofing

Replay is a review tool. 

Start the live receiver first:
```bash
python -m vision.vision_entrypoint live \
  --bind-host 127.0.0.1 \
  --port 5600 \
  --max-frames 20 \
  --timeout-seconds 10 \
  --output-root vision/output/replay_001 \
  --run-landmarker \
  --review \
  --review-output vision/output/reviews/replay_001
```

Then send sample frames over the same UDP wire contract:
```bash
python -m vision.tools.udp_spoof.replay \
  vision/tools/sample_runs/universe_75m75g50d_nearest_front_facing_75gates_fullrun_retry720_20260605 \
  --host 127.0.0.1 \
  --port 5600 \
  --fps 0 \
  --max-frames 20
```

Replay options:
```text
--host        UDP receiver host
--port        UDP receiver port
--fps         pacing; 0 sends as fast as possible
--chunk-size  JPEG payload bytes per UDP packet, default 1200
--max-frames  0 sends all available sample frames
```

This keeps live and replay behavior comparable: both enter the system as UDP packets and both pass through `vision.src.io`.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest vision/tests -q
```

The UDP integration test opens a local UDP socket on `127.0.0.1`.
