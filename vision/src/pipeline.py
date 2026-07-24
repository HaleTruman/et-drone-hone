"""Projection production pipeline orchestrator."""

from __future__ import annotations

import argparse
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover
    from .bbox import BBoxer
    from .color_mask import ColorMasker, write_mask_binary
    from .config import DEFAULT_PRESET_PATH, ProjectionConfig
    from .geometry_fits import GeometryFitter
    from .global_mapping import GlobalGateMapper
    from .inner_voids import InnerVoidDetector
    from .schema import (
        PROJECTION_FRAME_SCHEMA,
        RUN_MANIFEST_SCHEMA,
        RUN_STATUS_SCHEMA,
        FrameMeta,
        ProjectionFrameResult,
        RunManifest,
        RunStatus,
        StageDebugEnvelope,
        VisionResults,
        atomic_write_json,
        sha256_file,
        utc_now,
    )
    from .solve_pnp import SolvePnPRunner
    from .tracking import InnerVoidTracker
except ImportError:  # pragma: no cover
    from bbox import BBoxer
    from color_mask import ColorMasker, write_mask_binary
    from config import DEFAULT_PRESET_PATH, ProjectionConfig
    from geometry_fits import GeometryFitter
    from global_mapping import GlobalGateMapper
    from inner_voids import InnerVoidDetector
    from schema import (
        PROJECTION_FRAME_SCHEMA,
        RUN_MANIFEST_SCHEMA,
        RUN_STATUS_SCHEMA,
        FrameMeta,
        ProjectionFrameResult,
        RunManifest,
        RunStatus,
        StageDebugEnvelope,
        VisionResults,
        atomic_write_json,
        sha256_file,
        utc_now,
    )
    from solve_pnp import SolvePnPRunner
    from tracking import InnerVoidTracker


PROFILE_STAGES = [
    "bbox",
    "inner_voids",
    "geometry_fits",
    "solve_pnp",
    "tracking",
]

BASE_STAGES = [
    "color_mask",
    "global_mapping",
    "vision_results",
]


def utc_run_id() -> str:
    stamp = utc_now().replace("-", "").replace(":", "").split(".")[0]
    return f"run-{stamp}Z-{uuid.uuid4().hex[:8]}"


def jpeg_paths(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"})


@dataclass
class PipelineOptions:
    mode: str
    output_root: Path
    source_dir: Path | None = None
    single_frame: Path | None = None
    run_id: str | None = None
    debug: bool = False
    max_frames: int | None = None
    preset_path: Path = DEFAULT_PRESET_PATH


@dataclass
class ProfileRuntime:
    profile_id: str
    profile_label: str
    config: ProjectionConfig
    bboxer: BBoxer
    inner_voids: InnerVoidDetector
    geometry: GeometryFitter
    solve_pnp: SolvePnPRunner
    tracker: InnerVoidTracker


@dataclass
class ProcessedFrame:
    meta: FrameMeta
    source_sha256: str
    color_payload: dict[str, Any]
    profile_payloads: dict[str, dict[str, Any]]
    stage_timings_ms: dict[str, float]
    profile_timings_ms: dict[str, dict[str, float]]
    debug_artifacts: dict[str, Any]


class ProjectionPipeline:
    def __init__(self, options: PipelineOptions):
        self.options = options
        self.config = ProjectionConfig.from_path(options.preset_path)
        self.run_id = options.run_id or utc_run_id()
        self.run_root = (options.output_root / self.run_id).resolve()
        self.frames_dir = self.run_root / "frames"
        self.debug_root = self.run_root / "debug"
        self.status_path = self.run_root / "status.json"
        self.latest_path = self.run_root / "latest.json"
        self.manifest_path = self.run_root / "run_manifest.json"
        self.started_at = utc_now()
        self.frame_index: list[dict[str, Any]] = []
        self.timing_samples: dict[str, list[float]] = {stage: [] for stage in BASE_STAGES}
        self.processed_frames: list[ProcessedFrame] = []

        self.profiles = self.config.profiles
        self.color_masker = ColorMasker(self.config)
        self.profile_runtimes: dict[str, ProfileRuntime] = {}
        for profile in self.profiles:
            profile_config = self.config.profile_config(profile)
            profile_id = str(profile["profile_id"])
            self.profile_runtimes[profile_id] = ProfileRuntime(
                profile_id=profile_id,
                profile_label=str(profile["profile_label"]),
                config=profile_config,
                bboxer=BBoxer(profile_config),
                inner_voids=InnerVoidDetector(profile_config),
                geometry=GeometryFitter(profile_config),
                solve_pnp=SolvePnPRunner(profile_config),
                tracker=InnerVoidTracker(profile_config),
            )
        self.global_mapper = GlobalGateMapper(self.config)

    def _timed(self, stage: str, func: Callable[[], Any]) -> tuple[Any, float]:
        start = time.perf_counter()
        value = func()
        elapsed = round((time.perf_counter() - start) * 1000.0, 4)
        self.timing_samples.setdefault(stage, []).append(elapsed)
        return value, elapsed

    def _stage_names(self) -> list[str]:
        stages = ["color_mask"]
        for profile in self.profiles:
            profile_id = str(profile["profile_id"])
            stages.extend(f"profiles.{profile_id}.{stage}" for stage in PROFILE_STAGES)
        stages.extend(["global_mapping", "vision_results"])
        return stages

    def _write_status(
        self,
        status: str,
        *,
        latest_frame: str | None = None,
        completed_at: str | None = None,
        error: str | None = None,
        frame_count: int | None = None,
    ) -> None:
        errors = [error] if error else []
        atomic_write_json(
            self.status_path,
            RunStatus(
                schema=RUN_STATUS_SCHEMA,
                run_id=self.run_id,
                status=status,
                run_root=str(self.run_root),
                started_at=self.started_at,
                updated_at=utc_now(),
                frame_count=len(self.frame_index) if frame_count is None else int(frame_count),
                latest_frame=latest_frame,
                completed_at=completed_at,
                errors=errors,
            ),
        )

    def _timing_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        for stage, values in self.timing_samples.items():
            if not values:
                continue
            summary[stage] = {
                "count": len(values),
                "total": round(sum(values), 4),
                "mean": round(sum(values) / len(values), 4),
                "max": round(max(values), 4),
            }
        return summary

    def _manifest(self, completed_at: str | None = None) -> RunManifest:
        return RunManifest(
            schema=RUN_MANIFEST_SCHEMA,
            run_id=self.run_id,
            mode=self.options.mode,
            run_root=str(self.run_root),
            source_dir=str(self.options.source_dir) if self.options.source_dir else None,
            single_frame=str(self.options.single_frame) if self.options.single_frame else None,
            output_root=str(self.options.output_root),
            created_at=self.started_at,
            completed_at=completed_at,
            debug=bool(self.options.debug),
            preset_path=str(self.options.preset_path.resolve()),
            preset_sha256=sha256_file(self.options.preset_path.resolve()),
            stages=self._stage_names(),
            frame_count=len(self.frame_index),
            frame_index=list(self.frame_index),
            timings_ms=self._timing_summary(),
            output_paths={
                "run_manifest": str(self.manifest_path),
                "status": str(self.status_path),
                "latest": str(self.latest_path),
                "frames": str(self.frames_dir),
                "debug": str(self.debug_root) if self.options.debug else "",
            },
        )

    def _write_manifest(self, completed_at: str | None = None) -> None:
        atomic_write_json(self.manifest_path, self._manifest(completed_at=completed_at))

    def _write_debug(self, stage: str, meta: FrameMeta, payload: dict[str, Any], timing_ms: float, artifacts: dict[str, str] | None = None) -> str:
        path = self.debug_root / stage / "frames" / f"{meta.frame_id}.json"
        atomic_write_json(
            path,
            StageDebugEnvelope(
                schema="projection-stage-debug.v1",
                run_id=self.run_id,
                frame_ordinal=meta.frame_ordinal,
                frame_id=meta.frame_id,
                stage=stage,
                created_at=utc_now(),
                timing_ms=float(timing_ms),
                payload=payload,
                artifacts=dict(artifacts or {}),
            ),
        )
        return str(path)

    def _write_profile_debug(
        self,
        profile_id: str,
        stage: str,
        meta: FrameMeta,
        payload: dict[str, Any],
        timing_ms: float,
        artifacts: dict[str, str] | None = None,
    ) -> str:
        path = self.debug_root / "profiles" / profile_id / stage / "frames" / f"{meta.frame_id}.json"
        atomic_write_json(
            path,
            StageDebugEnvelope(
                schema="projection-profile-stage-debug.v1",
                run_id=self.run_id,
                frame_ordinal=meta.frame_ordinal,
                frame_id=meta.frame_id,
                stage=f"profiles.{profile_id}.{stage}",
                created_at=utc_now(),
                timing_ms=float(timing_ms),
                payload=payload,
                artifacts=dict(artifacts or {}),
            ),
        )
        return str(path)

    def process_frame(self, source_path: Path, frame_ordinal: int) -> ProcessedFrame:
        meta = FrameMeta(
            run_id=self.run_id,
            frame_ordinal=frame_ordinal,
            frame_id=f"frame_{frame_ordinal:06d}",
            source_path=str(source_path.resolve()),
        )
        debug_artifacts: dict[str, Any] = {}
        stage_timings: dict[str, float] = {}
        profile_timings: dict[str, dict[str, float]] = {}
        profile_payloads: dict[str, dict[str, Any]] = {}

        (mask_bits, color_payload), stage_timings["color_mask"] = self._timed("color_mask", lambda: self.color_masker.process(meta))
        meta.image_width = int(color_payload["image_width"])
        meta.image_height = int(color_payload["image_height"])

        if self.options.debug:
            mask_path = self.debug_root / "color_mask" / "masks" / f"{meta.frame_id}.bin"
            write_mask_binary(mask_path, mask_bits)
            debug_artifacts["color_mask_mask"] = str(mask_path)
            debug_artifacts["color_mask"] = self._write_debug("color_mask", meta, color_payload, stage_timings["color_mask"], {"mask": str(mask_path)})

        for profile_id, runtime in self.profile_runtimes.items():
            profile_stage_timings: dict[str, float] = {}
            profile_debug: dict[str, str] = {}
            (bbox_payload, labels, selected_mask), profile_stage_timings["bbox"] = self._timed(
                f"profiles.{profile_id}.bbox",
                lambda runtime=runtime: runtime.bboxer.process(meta, mask_bits),
            )
            inner_payload, profile_stage_timings["inner_voids"] = self._timed(
                f"profiles.{profile_id}.inner_voids",
                lambda runtime=runtime, bbox_payload=bbox_payload, labels=labels, selected_mask=selected_mask: runtime.inner_voids.process(
                    meta, bbox_payload, labels, selected_mask, mask_bits
                ),
            )
            fit_payload, profile_stage_timings["geometry_fits"] = self._timed(
                f"profiles.{profile_id}.geometry_fits",
                lambda runtime=runtime, inner_payload=inner_payload: runtime.geometry.process(meta, inner_payload),
            )
            solve_payload, profile_stage_timings["solve_pnp"] = self._timed(
                f"profiles.{profile_id}.solve_pnp",
                lambda runtime=runtime, inner_payload=inner_payload, fit_payload=fit_payload: runtime.solve_pnp.process(meta, inner_payload, fit_payload),
            )
            tracking_payload, profile_stage_timings["tracking"] = self._timed(
                f"profiles.{profile_id}.tracking",
                lambda runtime=runtime, inner_payload=inner_payload, fit_payload=fit_payload, solve_payload=solve_payload: runtime.tracker.process(
                    meta, inner_payload, fit_payload, solve_payload
                ),
            )
            profile_timings[profile_id] = profile_stage_timings
            profile_payloads[profile_id] = {
                "profile_id": profile_id,
                "profile_label": runtime.profile_label,
                "bbox": bbox_payload,
                "inner_voids": inner_payload,
                "geometry_fits": fit_payload,
                "solve_pnp": solve_payload,
                "tracking": tracking_payload,
            }
            if self.options.debug:
                profile_debug["bbox"] = self._write_profile_debug(profile_id, "bbox", meta, bbox_payload, profile_stage_timings["bbox"])
                profile_debug["inner_voids"] = self._write_profile_debug(profile_id, "inner_voids", meta, inner_payload, profile_stage_timings["inner_voids"])
                profile_debug["geometry_fits"] = self._write_profile_debug(
                    profile_id, "geometry_fits", meta, fit_payload, profile_stage_timings["geometry_fits"]
                )
                profile_debug["solve_pnp"] = self._write_profile_debug(profile_id, "solve_pnp", meta, solve_payload, profile_stage_timings["solve_pnp"])
                profile_debug["tracking"] = self._write_profile_debug(profile_id, "tracking", meta, tracking_payload, profile_stage_timings["tracking"])
                debug_artifacts.setdefault("profiles", {})[profile_id] = profile_debug

        return ProcessedFrame(
            meta=meta,
            source_sha256=sha256_file(source_path),
            color_payload=color_payload,
            profile_payloads=profile_payloads,
            stage_timings_ms=stage_timings,
            profile_timings_ms=profile_timings,
            debug_artifacts=debug_artifacts,
        )

    def _profile_tracks(self, processed_frames: list[ProcessedFrame]) -> list[dict[str, Any]]:
        tracks_by_profile: list[dict[str, Any]] = []
        for profile in self.profiles:
            profile_id = str(profile["profile_id"])
            runtime = self.profile_runtimes[profile_id]
            instances: dict[str, dict[str, Any]] = {}
            frame_index: list[dict[str, Any]] = []
            for frame in processed_frames:
                payloads = frame.profile_payloads.get(profile_id, {})
                tracking_payload = payloads.get("tracking", {})
                frame_index.append(
                    {
                        "frame_ordinal": frame.meta.frame_ordinal,
                        "frame_id": frame.meta.frame_id,
                        "source_path": frame.meta.source_path,
                        "observation_count": len(tracking_payload.get("observations") or []),
                    }
                )
                for observation in tracking_payload.get("observations") or []:
                    instance_id = str(observation.get("instance_id") or "")
                    if not instance_id:
                        continue
                    instance = instances.setdefault(
                        instance_id,
                        {
                            "instance_id": instance_id,
                            "profile_id": profile_id,
                            "profile_label": runtime.profile_label,
                            "observations": [],
                        },
                    )
                    instance["observations"].append(observation)
            tracks_by_profile.append(
                {
                    "schema": "projection-inner-void-profile-tracks.v1",
                    "run_id": self.run_id,
                    "profile_id": profile_id,
                    "profile_label": runtime.profile_label,
                    "frame_count": len(processed_frames),
                    "frame_index": frame_index,
                    "settings": {
                        "bbox": runtime.config.section("bbox"),
                        "innerVoids": runtime.config.section("innerVoids"),
                        "innerVoidQuadFit": runtime.config.section("innerVoidQuadFit"),
                        "innerVoidEllipseFit": runtime.config.section("innerVoidEllipseFit"),
                        "innerVoidSolvePnP": runtime.config.section("innerVoidSolvePnP"),
                        "innerVoidInstanceTracking": runtime.config.section("innerVoidInstanceTracking"),
                    },
                    "instances": sorted(instances.values(), key=lambda item: item["instance_id"]),
                }
            )
        return tracks_by_profile

    def _profile_counts(self, frame: ProcessedFrame) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for profile_id, payloads in frame.profile_payloads.items():
            fit_payload = payloads.get("geometry_fits", {})
            counts[profile_id] = {
                "bbox": int((payloads.get("bbox", {}) or {}).get("bbox_count") or 0),
                "inner_voids": int((payloads.get("inner_voids", {}) or {}).get("accepted_count") or 0),
                "geometry_quad": int(((fit_payload.get("quad") or {}).get("available_count")) or 0),
                "geometry_ellipse": int(((fit_payload.get("ellipse") or {}).get("available_count")) or 0),
                "solve_pnp": int((payloads.get("solve_pnp", {}) or {}).get("available_count") or 0),
                "tracking_instances": len((payloads.get("tracking", {}) or {}).get("instances") or []),
            }
        return counts

    def _aggregate_counts(self, profile_counts: dict[str, dict[str, int]], gate_count: int) -> dict[str, int]:
        keys = ["bbox", "inner_voids", "geometry_quad", "geometry_ellipse", "solve_pnp", "tracking_instances"]
        counts = {key: sum(profile.get(key, 0) for profile in profile_counts.values()) for key in keys}
        counts["profiles"] = len(profile_counts)
        counts["gates"] = int(gate_count)
        return counts

    def _write_final_frames(
        self,
        processed_frames: list[ProcessedFrame],
        gates_by_frame: dict[int, list[Any]],
        global_payload: dict[str, Any],
        global_timing_ms: float,
    ) -> None:
        global_debug_path = ""
        if self.options.debug:
            global_debug_path = str(self.debug_root / "global_mapping" / "run.json")
            atomic_write_json(Path(global_debug_path), global_payload)

        for frame in processed_frames:
            gates = list(gates_by_frame.get(frame.meta.frame_ordinal, []))
            profile_counts = self._profile_counts(frame)
            stage_timings = dict(frame.stage_timings_ms)
            stage_timings["global_mapping"] = global_timing_ms
            stage_timings["vision_results"] = 0.0
            vision_start = time.perf_counter()
            vision_results = VisionResults(
                run_id=self.run_id,
                frame_ordinal=frame.meta.frame_ordinal,
                frame_id=frame.meta.frame_id,
                source_path=frame.meta.source_path,
                image_width=frame.meta.image_width,
                image_height=frame.meta.image_height,
                created_at=utc_now(),
                timing_ms=dict(stage_timings),
                gates=gates,
                obstacles=[],
                trace={
                    "source_stage": "projection.global_mapping",
                    "profile_counts": profile_counts,
                    "global_mapping": {
                        "sample_count": int(global_payload.get("sample_count") or 0),
                        "gate_count": int(global_payload.get("gate_count") or 0),
                    },
                },
            )
            stage_timings["vision_results"] = round((time.perf_counter() - vision_start) * 1000.0, 4)
            vision_results.timing_ms = dict(stage_timings)
            self.timing_samples.setdefault("vision_results", []).append(stage_timings["vision_results"])
            debug_artifacts = dict(frame.debug_artifacts)
            if global_debug_path:
                debug_artifacts["global_mapping_run"] = global_debug_path
            result = ProjectionFrameResult(
                schema=PROJECTION_FRAME_SCHEMA,
                run_id=self.run_id,
                frame_ordinal=frame.meta.frame_ordinal,
                frame_id=frame.meta.frame_id,
                source_path=frame.meta.source_path,
                image_width=frame.meta.image_width,
                image_height=frame.meta.image_height,
                created_at=utc_now(),
                vision_results=vision_results,
                stage_counts=self._aggregate_counts(profile_counts, len(gates)),
                stage_timings_ms=stage_timings,
                artifacts={"debug": debug_artifacts} if self.options.debug else {},
            )
            final_path = self.frames_dir / f"{frame.meta.frame_id}.json"
            atomic_write_json(final_path, result)
            atomic_write_json(self.latest_path, result)
            self.frame_index.append(
                {
                    "frame_ordinal": frame.meta.frame_ordinal,
                    "frame_id": frame.meta.frame_id,
                    "source_path": frame.meta.source_path,
                    "source_sha256": frame.source_sha256,
                    "final_json": str(final_path),
                    "debug_artifacts": debug_artifacts if self.options.debug else {},
                    "stage_timings_ms": stage_timings,
                    "profile_timings_ms": frame.profile_timings_ms,
                }
            )

    def sources(self) -> list[Path]:
        if self.options.single_frame is not None:
            return [self.options.single_frame]
        if self.options.source_dir is None:
            raise ValueError("source_dir is required for batch mode")
        return jpeg_paths(self.options.source_dir)

    def run(self) -> RunManifest:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._write_status("running")
        try:
            for ordinal, source_path in enumerate(self.sources()):
                if self.options.max_frames is not None and ordinal >= self.options.max_frames:
                    break
                self.processed_frames.append(self.process_frame(source_path, ordinal))
                self._write_status("running", latest_frame=f"frame_{ordinal:06d}", frame_count=len(self.processed_frames))
            profile_tracks = self._profile_tracks(self.processed_frames)
            frame_ordinals = [frame.meta.frame_ordinal for frame in self.processed_frames]
            frame_ids = {frame.meta.frame_ordinal: frame.meta.frame_id for frame in self.processed_frames}
            (gates_by_frame, global_payload), global_timing_ms = self._timed(
                "global_mapping",
                lambda: self.global_mapper.process_run(profile_tracks, frame_ordinals, frame_ids),
            )
            global_payload["profile_tracks"] = profile_tracks if self.options.debug else [
                {
                    "profile_id": payload.get("profile_id"),
                    "profile_label": payload.get("profile_label"),
                    "frame_count": payload.get("frame_count"),
                    "instance_count": len(payload.get("instances") or []),
                }
                for payload in profile_tracks
            ]
            self._write_final_frames(self.processed_frames, gates_by_frame, global_payload, global_timing_ms)
            completed_at = utc_now()
            latest_frame = str(self.frames_dir / f"{self.processed_frames[-1].meta.frame_id}.json") if self.processed_frames else None
            self._write_status("complete", latest_frame=latest_frame, completed_at=completed_at)
            self._write_manifest(completed_at=completed_at)
            return self._manifest(completed_at=completed_at)
        except Exception as exc:
            self._write_status("failed", error=str(exc))
            self._write_manifest()
            raise


def run_pipeline(options: PipelineOptions) -> RunManifest:
    return ProjectionPipeline(options).run()


def options_from_args(args: argparse.Namespace) -> PipelineOptions:
    if args.single_frame and args.source_dir:
        raise ValueError("choose either --source-dir or --single-frame, not both")
    if not args.single_frame and not args.source_dir:
        raise ValueError("one of --source-dir or --single-frame is required")
    return PipelineOptions(
        mode="single" if args.single_frame else "batch",
        source_dir=args.source_dir,
        single_frame=args.single_frame,
        output_root=args.output_root,
        run_id=args.run_id,
        debug=bool(args.debug),
        max_frames=args.max_frames,
        preset_path=args.preset_path or DEFAULT_PRESET_PATH,
    )
