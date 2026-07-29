# Dataset Collection

Code in this folder should only be responsible for collecting training data from a generated gate layout.

This includes:

- generating camera poses
- syncing `SceneCaptureComponent2D` to the camera
- exporting render target frames
- projecting gate corners into the 640x360 frame
- calculating gate bounding boxes
- writing `metadata.json` and `metadata.jsonl`

It should not own the algorithm that lays out the race track or randomizes gate positions.
