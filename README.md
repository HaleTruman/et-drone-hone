# et-drone-hone

Monorepository for autonomous drone-racing experiments. The active stack is split into small Python subprojects: `Flight` owns the live control loop, `Vision` owns perception models and gate observations, and `Viewer` owns local run review. `Legacy`, `Examples`, and `Docs` preserve earlier work, reference implementations, and design notes.

<p align="center">
  <video src="Examples/video/racing-preview.mp4" poster="Examples/video/racing-preview-poster.jpg" controls muted playsinline width="900">
    <source src="Examples/video/racing-preview.webm" type="video/webm">
    <source src="Examples/video/racing-preview.mp4" type="video/mp4">
    <a href="Examples/video/racing-preview.mp4">Watch the racing preview video</a>
  </video>
</p>

## Repository Layout

```text
et-drone-hone/
  Flight/    Live flight runtime, MAVLink bridge, control loop, state estimation,
             gate mapping, vision-frame ingestion, logging, and validation scripts.

  Vision/    Standalone perception library. Contains callable gate-detection
             services, shared vision schemas, CNN and deterministic backends,
             model assets, and parity tests against Flight integration.

  Viewer/    Local browser UI for reviewing Flight, evaluation, and review logs.
             Includes a Flask server and static HTML/CSS/JS frontend.

  Legacy/    Archived Unreal Engine tooling, old simulator projects, generated
             synthetic data pipelines, and historical experimentation code.

  Examples/  Reference snippets and prototype runtimes used while developing
             MAVLink, vision ingress, Q1 guidance, and pose-review workflows.

  Docs/      Project briefs, architecture notes, math references, protocol specs,
             planning notes, AI Grand Prix source material, and research papers.
```

## Start Here

1. For the live stack, read `Flight/README.md`.
2. For perception backends and model tests, read `Vision/README.md`.
3. For log review, run the app in `Viewer/`.
4. For historical context or old Unreal workflows, browse `Legacy/` and `Docs/work/`.

## Active Data Flow

```text
Simulator / vehicle bridge
  -> MAVLink telemetry
  -> Flight state estimation and gate-aware control
  -> MAVLink attitude-target commands

Simulator FPV stream
  -> Flight vision UDP ingestion
  -> Vision perception service
  -> Flight gate map
  -> controller target selection
```

The repository does not include the simulator runtime itself. `Flight` expects MAVLink telemetry and FPV image packets from an external simulator or vehicle bridge.
