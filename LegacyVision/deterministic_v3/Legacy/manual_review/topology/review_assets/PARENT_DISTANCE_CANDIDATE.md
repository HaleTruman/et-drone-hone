# Parent-distance standard-gate candidate

Status: isolated manual-review candidate, not production.

Timestamp: 2026-08-01 PDT.

The previous standard test considered the shape of the child aperture without
confirming where that aperture sat within its full parent contour. This allowed
a small balanced hole to label a much larger, asymmetric parent as standard.

For every exact `(parent_contour_id, child_contour_id)` pair, the candidate now:

1. Fills the complete parent contour with a one-pixel zero border.
2. Runs `cv2.distanceTransform(..., cv2.DIST_L2, 5)`.
3. Samples that field at the subpixel child-aperture center.
4. Divides the sampled depth by the parent field's maximum depth.
5. Requires a ratio of at least `0.80` for a standard classification.

The supplied frame `106569`, component `4`, measures approximately `8.625 / 12
= 0.719` with bilinear sampling. It is therefore routed to `unknown` with:

```text
classification_rule = one_child_parent_distance_below_threshold
rejection_reason = standard_parent_distance_depth_below_minimum
failed_check = parent_distance_depth_ratio_below_minimum
```

The threshold is an initial review setting. It must not be promoted into
production until the regenerated candidate sweep is visually assessed.

