# Repository Standards

> **Working draft:** These standards are incomplete and must be reviewed, clarified, and improved as the deterministic V3 architecture matures.

## Scope

These standards apply to all work under `Flight/src/sensing/vision/models/deterministic_v3` on the `ft-vision-gate-geometry-a` branch. The production inference path, calibration tools, visual interfaces, generated assets, tests, and documentation must remain clearly separated and easy to audit.

## Production inference and `/src`

Code under `/src` is production inference code. Its first priorities are runtime speed, deterministic behavior, numerical precision, and predictable resource use.

- Keep the runtime path direct, bounded, and measurable.
- Avoid unnecessary allocations, repeated computation, redundant image conversion, and avoidable file access.
- Preserve numerical precision wherever it materially affects geometry, confidence, pose, or validation.
- Make shared computation explicit and perform it once at the narrowest correct scope.
- Keep rejection behavior deterministic and machine-readable.
- Do not add visualization, exploratory analysis, report generation, or calibration-only behavior to the production inference path.
- Production inference must not depend on UI state, review artifacts, generated images, local developer paths, or optional calibration datasets.

## UI, calibration, and review assets

The `/ui`, viewer, scripts, and other review assets must provide a visual system for inspecting and calibrating deterministic V3 behavior without changing the production runtime contract.

- Calibration tools may observe production outputs and invoke stable production APIs, but production code must not contain special cases solely to support a particular visualization or experiment.
- One-off computations, experimental comparisons, annotations, and report rendering belong in calibration or review tooling, not in `/src` runtime logic.
- Calibration reads and writes must remain outside the inference path and must use explicit inputs and output locations.
- Generated images, manifests, contact sheets, caches, and review runs must never become hidden runtime dependencies.
- UI controls must map to documented configuration or calibration inputs rather than mutating implicit global state.
- A production result must remain reproducible without launching the UI or loading review assets.

## Architecture and code organization

Optimize the system for minimal line count consistent with correctness, clarity, and testability. Fewer lines are valuable when they remove duplication and incidental complexity; compressed or obscure code is not.

- Give each module one clear responsibility.
- Prefer small, explainable functions and explicit data flow.
- Use shared helpers only when they express a stable concept used in more than one place.
- Keep imports minimal, directional, and free of circular dependencies.
- Do not use wildcard imports or import-time side effects.
- Keep production modules independent of scripts, UI code, generated assets, and test fixtures.
- Remove obsolete compatibility layers, duplicate implementations, and abandoned experimental paths once their replacement is validated.
- Use names that communicate geometry, evidence ownership, units, coordinate frames, and lifecycle.

## Documentation standards

Documentation must explain the current system rather than preserve ambiguous or obsolete branch history.

- Describe module ownership, inputs, outputs, invariants, rejection conditions, and performance-sensitive behavior.
- Keep paths relative to the repository wherever possible; avoid developer-specific absolute paths.
- Identify whether a procedure is production, calibration, experimental, or legacy.
- Update architecture documents when module boundaries or public contracts change.
- Prefer concise explanations tied to actual code and schemas over duplicated narrative.
- Record assumptions, units, coordinate frames, calibration versions, and reproducibility requirements explicitly.

## Repository cleanliness

- Keep temporary files, caches, `.DS_Store`, bytecode, generated review runs, and local manifests out of version control unless an artifact is deliberately selected as durable evidence.
- Keep durable assets organized, named, and accompanied by enough metadata to explain their origin and purpose.
- Do not commit hard-coded references to another branch or worktree as operational paths.
- Stage and commit only files owned by the current task and branch.
- Keep tests close to the behavior they validate and remove fixtures that no longer represent supported behavior.
- Leave the deterministic V3 tree understandable from its tracked files alone.

## Standard for accepting changes

A change is ready when its ownership is clear, its runtime impact is understood, production and calibration concerns remain separated, relevant behavior is tested, documentation matches the implementation, and no unnecessary files or dependencies are introduced.
