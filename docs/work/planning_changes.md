# Planning Changes

## Problem

When the drone can only see the current last perceived gate, the planner can collapse to a path from the drone's current position directly to that gate. As the drone moves, that line rotates around the gate center. At high speed this can produce a large sweeping turn around the gate instead of a stable path through the gate and onward toward the unseen next gate.

## Rolling Corridor Direction

The preferred long-term fix is a rolling path corridor:

- Keep a stable active corridor for control.
- Only project onto path samples forward of monotonic path progress.
- Keep crossed gates as geometric context for path shape, but remove them as targets.
- Splice new plans after a short committed horizon instead of replacing the immediate control path.
- Keep a full path history for logging/UI, but control against the active forward corridor.

## Terminal Tail

The planned path should not terminate at the final perceived gate center. It should pass through that gate and continue along a synthetic tail.

For the current implementation, the synthetic tail should use the tangent from the second-to-last gate to the last gate when at least two gate anchors are available. If later only the last gate remains visible/plannable, retain the last known terminal gate-to-gate tangent so the tail does not rotate with the drone's current position.

## Expected Effect

This should reduce path endpoint rotation and control-target jitter near the final perceived gate. It does not fully replace the rolling corridor design, but it addresses the most obvious failure mode where the terminal path direction depends on the vehicle's current position instead of the course geometry.
