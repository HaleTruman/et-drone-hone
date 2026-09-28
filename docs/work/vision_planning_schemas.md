# Vision And Planning Schemas

The hover/vision/MPCC test stack used typed in-process contracts, then wrote
JSON-compatible records to `logs/runs/run-*/run.json`.

## Vision Frame To Planning V1

Schema id: `vision_frame_to_planning_v1`

Producer:

- `sensing.vision.io.vision_stream.VisionStreamReceiver`
- `sensing.vision.service.VisionPerceptionService` (reexported from Vision)
- `sensing.perception.GatePoseEstimator`

Consumer:

- `autonomy.planning.mpcc.MPCCPlanner`

Frame log shape:

```json
{
  "schema_version": "vision_frame_to_planning_v1",
  "frame_id": 1,
  "sim_time_ns": 123456789,
  "saved_path": "logs/runs/run-.../frames/frame-00000001-123456789.jpg",
  "gate_count": 1,
  "observation": {
    "run": {
      "output_dir": "memory",
      "cycle": 1,
      "frame_id": "frame_000001",
      "sim_time_ns": 123456789
    },
    "gates": [
      {
        "id": "gate-current-000",
        "position_xyz": [0.0, 0.0, 10.0],
        "position_confidence": 0.8,
        "orientation_xyz": [0.0, 0.0, 1.0],
        "orientation_confidence": 0.7
      }
    ],
    "obstacles": []
  },
  "mapped_gates": [
    {
      "id": "gate-current-000",
      "pos": [9.397, 0.0, -3.42],
      "quat": [1.0, 0.0, 0.0, 0.0],
      "confidence": 0.8,
      "sequence": 0
    }
  ]
}
```

Coordinate frames:

- `observation.gates[].position_xyz` is camera optical meters from the CNN/regressor path in `[right_m, up_m, forward_m]` order.
- `observation.gates[].orientation_xyz` is a camera optical direction vector in `[right, up, forward]` order.
- Flight-test mapping bypasses the camera-local landmarker state and assigns current-frame IDs `gate-current-000`, `gate-current-001`, etc. before local-NED mapping.
- `mapped_gates[].pos` is local NED meters after applying camera optical-to-body FRD conversion, the 20 degree upward camera tilt, vehicle attitude, and vehicle position.
- `mapped_gates[].quat` is a local NED gate orientation quaternion in `[w, x, y, z]` order.

## Planned Path Local NED V1

Schema id: `planned_path_local_ned_v1`

Producer:

- `autonomy.planning.mpcc.MPCCPlanner`

Consumer:

- Flight-test logs
- Dash trajectory viewer

Log shape:

```json
{
  "schema_version": "planned_path_local_ned_v1",
  "frame_id": 1,
  "sim_time_ns": 123456789,
  "elapsed_ms": 42.0,
  "status": "Solve_Succeeded",
  "points_local_ned_m": [[0.0, 0.0, 0.0], [1.0, 0.0, -0.1]],
  "reference_path_local_ned_m": [[0.0, 0.0, 0.0], [1.0, 0.0, -0.1]],
  "gates": [
    {
      "id": "gate-001w",
      "pos": [12.0, 3.0, -1.0],
      "quat": [1.0, 0.0, 0.0, 0.0],
      "confidence": 0.8,
      "sequence": 0
    }
  ],
  "gate_misses_m": [0.1]
}
```

The current MPCC formulation constrains the immediate next gate, but accepts a
multi-gate sequence for reference-path generation and planned-path logging.
