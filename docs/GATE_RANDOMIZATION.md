# Gate Randomization

Code in this folder should only be responsible for generating the gate layout.

This includes:

- finding actors in the `Gates/` outliner folder
- generating random spline-like race tracks
- placing every gate along that track
- orienting each gate toward the next gate
- rebuilding the blue 0.75m spline track indicator on `Generated_Gate_Track_Spline`

It should not save frames, write dataset metadata, or control the camera capture loop.
