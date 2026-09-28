from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from schema import VisionGateObservation, VisionObservation
from models.deterministic.config import BACKEND_DIR, DeterministicVisionConfig
from schema import VisionFrame


TOOLS_DIR = BACKEND_DIR / "tools"
SRC_DIR = BACKEND_DIR / "src"

from models.deterministic.src.bbox_clipping.build import build_bbox_clipping_frame  # noqa: E402
from models.deterministic.src.bbox_contours.build import build_bbox_contour_frame  # noqa: E402
from models.deterministic.tools.build_mask_bboxes import (  # noqa: E402
    bbox_settings_from_review,
    clean_float,
    normalized_fov_clip_settings,
)
from models.deterministic.tools.build_mask_contours import normalized_contour_settings  # noqa: E402
from models.deterministic.src.color_masks.mask_frames import (  # noqa: E402
    DEFAULT_MANIFEST_NAME,
    FrameRef,
    build_manifest as build_mask_manifest,
    empty_summary as empty_mask_summary,
    layers_from_metadata,
    load_lut,
    merge_summary as merge_mask_summary,
    process_frame as process_mask_frame,
)
from models.deterministic.src.instance_tracking.instance_mapping import (  # noqa: E402
    InstanceTrackingRuntime,
    instance_tracking_settings_from_review,
)
from models.deterministic.src.mask_bbox.build_from_maskbits import build_maskbits_bbox_frame  # noqa: E402
from models.deterministic.src.mask_bbox.maskbits_input import build_mask_frame_lookup, build_maskbits_context  # noqa: E402
from models.deterministic.src.pose_estimation.pose_fit import build_pose_frame, pose_settings_from_review  # noqa: E402


class DeterministicVisionBackend:
    """Frame-native adapter around the deterministic 0721Vision stages."""

    def __init__(self, config: DeterministicVisionConfig | None = None) -> None:
        self.config = config or DeterministicVisionConfig()
        self.run_id = f"flight-deterministic-{int(time.time() * 1000)}"
        self.run_root = self.config.scratch_root / self.run_id
        self.source_frames_dir = self.run_root / "source_frames"
        self.mask_output_dir = self.run_root / "color_masks"
        self.mask_manifest_path = self.mask_output_dir / DEFAULT_MANIFEST_NAME
        self.frame_count = 0
        self.stage_latency_ms: dict[str, float] = {}
        self._loaded = False
        self._executor: ThreadPoolExecutor | None = None

    def process_vision_frame(self, frame: VisionFrame) -> VisionObservation:
        return self.process_frame(
            frame_id=frame.frame_id,
            sim_time_ns=frame.sim_time_ns,
            jpeg_bytes=frame.jpeg_bytes,
        )

    def process_frame(self, *, frame_id: int, sim_time_ns: int, jpeg_bytes: bytes) -> VisionObservation:
        self._prepare()
        frame_index = self.frame_count
        self.frame_count += 1
        source_path = self._write_source_frame(frame_index, frame_id, jpeg_bytes)
        frame_entry = self._frame_entry(frame_index, frame_id, sim_time_ns, source_path, jpeg_bytes)

        mask_entry, mask_stats = self._timed(
            "colorMaskbits",
            lambda: process_mask_frame(
                FrameRef(index=frame_index, frame_id=str(frame_id), path=source_path, source_record=frame_entry),
                self.lut,
                self.layers,
                self.mask_output_dir / "masks",
            ),
        )
        mask_summary = empty_mask_summary(self.layers)
        merge_mask_summary(mask_summary, mask_stats)
        mask_manifest = build_mask_manifest(self.config.lut_path, self.lut_metadata, self.layers, [mask_entry], mask_summary)
        mask_manifest.update(
            {
                "app": "flight_deterministic_vision",
                "kind": "rgb-lut-maskbits",
                "runKey": self.run_id,
                "runId": self.run_id,
                "complete": False,
                "source": "flight-vision-frame",
            }
        )
        self._atomic_write_json(self.mask_manifest_path, mask_manifest)
        self._assert_dimensions(mask_manifest)

        mask_lookup = build_mask_frame_lookup(mask_manifest)
        mask_context = build_maskbits_context(self.config_data, self.review_data, mask_manifest, self.mask_manifest_path)

        bbox_frame, _ = self._timed(
            "maskBboxes",
            lambda: build_maskbits_bbox_frame(
                frame_entry,
                frame_index,
                mask_lookup,
                mask_context,
                self.config.expected_width,
                self.config.expected_height,
                self.bbox_settings,
            ),
        )

        if self._executor is not None:
            clipping_future = self._executor.submit(
                lambda: self._timed(
                    "bboxClipping",
                    lambda: build_bbox_clipping_frame(
                        self.cv2,
                        frame_entry,
                        frame_index,
                        bbox_frame,
                        mask_lookup,
                        mask_context,
                        self.config.expected_width,
                        self.config.expected_height,
                        self.clipping_settings,
                    ),
                )
            )
            contour_future = self._executor.submit(
                lambda: self._timed(
                    "bboxContours",
                    lambda: build_bbox_contour_frame(
                        self.cv2,
                        frame_entry,
                        frame_index,
                        bbox_frame,
                        mask_lookup,
                        mask_context,
                        self.config.expected_width,
                        self.config.expected_height,
                        self.contour_settings,
                    ),
                )
            )
            clipping_frame, _ = clipping_future.result()
            contour_frame = contour_future.result()
        else:
            clipping_frame, _ = self._timed(
                "bboxClipping",
                lambda: build_bbox_clipping_frame(
                    self.cv2,
                    frame_entry,
                    frame_index,
                    bbox_frame,
                    mask_lookup,
                    mask_context,
                    self.config.expected_width,
                    self.config.expected_height,
                    self.clipping_settings,
                ),
            )
            contour_frame = self._timed(
                "bboxContours",
                lambda: build_bbox_contour_frame(
                    self.cv2,
                    frame_entry,
                    frame_index,
                    bbox_frame,
                    mask_lookup,
                    mask_context,
                    self.config.expected_width,
                    self.config.expected_height,
                    self.contour_settings,
                ),
            )

        pose_frame, _ = self._timed(
            "poseEstimation",
            lambda: build_pose_frame(self.cv2, frame_entry, frame_index, contour_frame, clipping_frame, self.pose_settings),
        )
        instance_frame = self._timed("instanceTracking", lambda: self.instance_runtime.process_frame(bbox_frame, pose_frame))
        trace = self._observation_trace(
            frame_entry=frame_entry,
            mask_entry=mask_entry,
            mask_stats=mask_stats,
            mask_manifest=mask_manifest,
            bbox_frame=bbox_frame,
            clipping_frame=clipping_frame,
            contour_frame=contour_frame,
            pose_frame=pose_frame,
            instance_frame=instance_frame,
        )
        observation = self._vision_observation_from_instance_frame(
            instance_frame,
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            trace=trace,
        )

        if not self.config.keep_scratch:
            self._cleanup_frame_work(source_path, mask_entry)
        return observation

    def shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if not self.config.keep_scratch:
            shutil.rmtree(self.run_root, ignore_errors=True)

    def snapshot(self) -> dict[str, Any]:
        return {
            "backend": "deterministic_0721",
            "run_id": self.run_id,
            "loaded": self._loaded,
            "frame_count": self.frame_count,
            "scratch_root": str(self.config.scratch_root),
            "keep_scratch": self.config.keep_scratch,
            "parallel_stateless": self.config.parallel_stateless,
            "stage_latency_ms": dict(self.stage_latency_ms),
        }

    def _prepare(self) -> None:
        if self._loaded:
            return
        self.source_frames_dir.mkdir(parents=True, exist_ok=True)
        self.mask_output_dir.mkdir(parents=True, exist_ok=True)
        self.config_data = self._read_json(self.config.config_path)
        self.review_data = self._read_json(self.config.review_path) if self.config.review_path.exists() else {"annotations": [], "decisions": []}
        self.lut, self.lut_metadata = load_lut(self.config.lut_path)
        self.layers = layers_from_metadata(self.lut_metadata)
        self.bbox_settings = bbox_settings_from_review(self.review_data)
        self.clipping_settings = normalized_fov_clip_settings(
            (self.bbox_settings.get("fovClip") or {}) if isinstance(self.bbox_settings, dict) else None
        )
        self.contour_settings = normalized_contour_settings(
            self.review_data.get("contourHierarchy") if isinstance(self.review_data.get("contourHierarchy"), dict) else None
        )
        self.pose_settings = pose_settings_from_review(self.review_data)
        self.instance_settings = instance_tracking_settings_from_review(self.review_data)
        if self.instance_settings.get("poseSource") == "squarePose":
            self.instance_settings["poseSource"] = "poseFit"
        self.instance_runtime = InstanceTrackingRuntime(self.instance_settings)
        import cv2  # type: ignore

        self.cv2 = cv2
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="deterministic-vision") if self.config.parallel_stateless else None
        self._loaded = True

    def _write_source_frame(self, frame_index: int, frame_id: int, jpeg_bytes: bytes) -> Path:
        path = self.source_frames_dir / f"frame_{frame_index:06d}_{frame_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(jpeg_bytes)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        return path

    def _frame_entry(self, frame_index: int, frame_id: int, sim_time_ns: int, source_path: Path, jpeg_bytes: bytes) -> dict:
        stat = source_path.stat()
        return {
            "frameOrdinal": int(frame_index),
            "frameId": str(frame_id),
            "index": int(frame_index),
            "path": str(source_path),
            "sourcePath": str(source_path),
            "originalPath": f"vision_frame:{int(frame_id)}",
            "sha1": hashlib.sha1(jpeg_bytes).hexdigest(),
            "mtimeNs": int(stat.st_mtime_ns),
            "jpegSize": int(stat.st_size),
            "simTimeNs": int(sim_time_ns),
        }

    def _assert_dimensions(self, mask_manifest: dict) -> None:
        width = mask_manifest.get("width")
        height = mask_manifest.get("height")
        if int(width or 0) != self.config.expected_width or int(height or 0) != self.config.expected_height:
            raise ValueError(
                f"deterministic vision expects {self.config.expected_width}x{self.config.expected_height} frames, "
                f"got {width}x{height}"
            )

    def _vision_observation_from_instance_frame(
        self,
        instance_frame: dict,
        *,
        frame_id: int,
        sim_time_ns: int,
        trace: dict[str, Any] | None = None,
    ) -> VisionObservation:
        gates = []
        for instance in instance_frame.get("observations") or []:
            gate = self._gate_from_instance(instance)
            if gate is not None:
                gates.append(gate)
        return VisionObservation(
            frame_id=int(frame_id),
            sim_time_ns=int(sim_time_ns),
            gates=gates,
            source="deterministic_0721",
            trace=trace or {},
        )

    def _observation_trace(
        self,
        *,
        frame_entry: dict,
        mask_entry: dict,
        mask_stats: dict,
        mask_manifest: dict,
        bbox_frame: dict,
        clipping_frame: dict,
        contour_frame: dict,
        pose_frame: dict,
        instance_frame: dict,
    ) -> dict[str, Any]:
        return {
            "backend": "deterministic_0721",
            "runId": self.run_id,
            "frameOrdinal": frame_entry.get("frameOrdinal"),
            "frameEntry": frame_entry,
            "settings": {
                "bbox": self.bbox_settings,
                "clipping": self.clipping_settings,
                "contours": self.contour_settings,
                "pose": self.pose_settings,
                "instanceTracking": self.instance_settings,
            },
            "stageLatencyMs": dict(self.stage_latency_ms),
            "stages": {
                "colorMaskbits": {
                    "entry": mask_entry,
                    "stats": mask_stats,
                    "manifest": mask_manifest,
                },
                "maskBboxes": bbox_frame,
                "bboxClipping": clipping_frame,
                "bboxContours": contour_frame,
                "poseEstimation": pose_frame,
                "instanceTracking": instance_frame,
            },
        }

    def _gate_from_instance(self, instance: dict) -> VisionGateObservation | None:
        pose = instance.get("pose") if isinstance(instance.get("pose"), dict) else {}
        if not pose.get("available"):
            return None
        xyz_camera_m = self._float3(pose.get("xyzCameraM"))
        if xyz_camera_m is None:
            return None
        depth_m = self._finite_float(pose.get("depthM", xyz_camera_m[2]))
        if depth_m is None or depth_m <= 0.0:
            return None
        gate_id = str(instance.get("instanceId") or instance.get("observationId") or "")
        if not gate_id:
            return None
        fit_quality = pose.get("fitQuality") if isinstance(pose.get("fitQuality"), dict) else {}
        return VisionGateObservation(
            gate_id=gate_id,
            position_local_ned=self._opencv_camera_to_flight_camera(xyz_camera_m),
            position_confidence=self._clamp01(instance.get("observationQuality")),
            orientation_local_ned_quat=None,
            orientation_confidence=self._clamp01(fit_quality.get("overall")),
            trace={
                "backend": "deterministic_0721",
                "legacy_position_camera_m": self._opencv_camera_to_flight_camera(xyz_camera_m),
                "legacy_orientation_camera": self._float3(pose.get("rpyCameraDeg")),
                "sourceInstance": instance,
                "bbox": instance.get("source", {}).get("bbox") if isinstance(instance.get("source"), dict) else None,
                "fovClip": instance.get("fovClip") if isinstance(instance.get("fovClip"), dict) else {},
                "pose": pose,
            },
        )

    @staticmethod
    def _opencv_camera_to_flight_camera(value: tuple[float, float, float]) -> tuple[float, float, float]:
        x_right, y_down, z_forward = value
        return (x_right, -y_down, z_forward)

    @staticmethod
    def _float3(value: Any) -> tuple[float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in result):
            return None
        return result  # type: ignore[return-value]

    @staticmethod
    def _finite_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _clamp01(cls, value: Any, default: float = 0.0) -> float:
        number = cls._finite_float(value)
        if number is None:
            number = float(default)
        return max(0.0, min(1.0, number))

    def _timed(self, stage: str, callback):
        start = time.perf_counter()
        result = callback()
        self.stage_latency_ms[stage] = clean_float((time.perf_counter() - start) * 1000.0, 3)
        return result

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"expected object JSON: {path}")
        return payload

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _cleanup_frame_work(self, source_path: Path, mask_entry: dict) -> None:
        paths = [
            source_path,
            self._path_from_mask_ref(mask_entry.get("maskBits")),
            self.mask_manifest_path,
        ]
        for path in paths:
            if path is None:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _path_from_mask_ref(self, value: Any) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text)
        if path.is_absolute():
            return path
        if text.startswith("/"):
            path = BACKEND_DIR / text.lstrip("/")
        else:
            path = BACKEND_DIR / text
        return path
