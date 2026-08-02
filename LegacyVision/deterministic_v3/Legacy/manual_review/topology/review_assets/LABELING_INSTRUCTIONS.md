# Manual topology review labeling

This directory holds subjective review evidence for the isolated topology
workspace. Generated predictions are candidates, not ground truth.

## Identity and timestamps

Every annotation must retain this complete identity tuple:

```text
run_id + frame_id + sim_time_ns + component_id
```

- `sim_time_ns` is the authoritative source timestamp recorded by the run.
- `reviewed_at_utc` records when a person made the judgment. Use an ISO-8601
  UTC timestamp such as `2026-08-02T04:12:30Z`.
- Never replace `sim_time_ns` with the review timestamp.
- Record the generated review ID and candidate topology SHA-256 so a judgment
  remains tied to the exact implementation that produced it.

## Subjective judgment

Use one of these `judgment` values:

- `agree`: the predicted topology is visually supported.
- `disagree`: a different topology is visually supported.
- `uncertain`: the available pixels do not justify a reliable choice.

When `judgment` is `disagree`, record one `expected_topology_label` using the
runtime terminology: `standard`, `c_shape`, `multi_void`, `unknown`, or
`clipped`. For `uncertain`, prefer `unknown` unless a likely label is useful as
a non-binding note.

Also record:

- `reviewer`: reviewer name or stable identifier.
- `confidence`: `low`, `medium`, or `high`.
- `reason_codes`: a list of short observable reasons.
- `notes`: optional free text that distinguishes visible evidence from an
  inference about it.

Suggested reason codes include `aperture_count_wrong`, `component_merge`,
`component_split`, `distance_center_wrong`, `distance_balance_wrong`,
`frame_clipping_wrong`, `opening_wrong`, and `insufficient_pixels`.

## Distance-transform review criteria

Each component image shows the source crop, final closed component mask,
contour hierarchy, the filled parent-contour distance transform used by the
isolated candidate, and the child-aperture distance transform. The parent and
child transforms share the parent peak as their display scale.

For a plausible standard gate, review whether:

1. The indicated child contour is the visible aperture.
2. The aperture distance-transform maximum lies within that aperture.
3. The selected center reasonably represents the aperture center.
4. The aperture center lies inside the outlined 80% depth region of its exact
   filled parent contour.
5. Closing or component merging did not manufacture the apparent aperture.

An `N/A` distance panel means the current topology branch did not execute that
test. It must not be interpreted as a zero-valued result.

## Annotation record

Keep annotations append-only in a reviewer-specific JSONL file. One line should
have this shape:

```json
{
  "run_id": "run-20260801T031401Z",
  "frame_id": 106738,
  "sim_time_ns": 1785554050000000000,
  "component_id": 1,
  "review_id": "20260802T041230Z",
  "candidate_topology_sha256": "...",
  "reviewed_at_utc": "2026-08-02T04:20:00Z",
  "reviewer": "reviewer-id",
  "judgment": "disagree",
  "expected_topology_label": "c_shape",
  "confidence": "medium",
  "reason_codes": ["opening_wrong"],
  "notes": "Visible opening is not an enclosed aperture."
}
```

Do not edit an older judgment in place. Append a replacement record with a new
`reviewed_at_utc` value so changes in subjective interpretation remain visible.
