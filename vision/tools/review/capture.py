from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewConfig:
    output_root: Path
    run_mode: str = "live"
    has_truth: bool = False


class ReviewSink:
    def __init__(self, config: ReviewConfig) -> None:
        self.config = config
        self.output_root = Path(config.output_root).expanduser()
        self.frames_dir = self.output_root / "frames"
        self.metadata: dict[str, Any] = {}
        self.frames: list[dict[str, Any]] = []

    def start_run(self, metadata: dict[str, Any] | None = None) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})
        self._write_manifest()

    def record_cnn_frame(self, *, frame: Any, logits_path: Path) -> None:
        frame_id = int(frame.frame_id)
        frame_name = f"frame_{frame_id:06d}"
        frame_dir = self.frames_dir / frame_name
        frame_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = frame_dir / "rgb.jpg"
        rgb_path.write_bytes(frame.jpeg_bytes)
        sidecar_path = frame_dir / "ingress_review.json"
        sidecar = {
            "frame_id": frame_id,
            "sim_time_ns": int(frame.sim_time_ns),
            "rgb_path": self._relative(rgb_path),
            "logits_path": self._relative(Path(logits_path)),
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        self.frames.append(
            {
                "frame_id": frame_id,
                "frame_name": frame_name,
                "sim_time_ns": int(frame.sim_time_ns),
                "rgb_path": self._relative(rgb_path),
                "ingress_review": self._relative(sidecar_path),
                "logits_path": self._relative(Path(logits_path)),
            }
        )

    def record_landmarker_outputs(self, *, controller_jsonl: Path) -> None:
        from vision.tools.review.review_render import frame_name, load_jsonl_by_frame_id, render_gate_overlay_image

        controller_path = Path(controller_jsonl).expanduser()
        controller_frames = load_jsonl_by_frame_id(controller_path)
        for frame_record in self.frames:
            name = str(frame_record.get("frame_name") or frame_name(int(frame_record["frame_id"])))
            payload = controller_frames.get(name)
            gates = payload.get("gates", []) if isinstance(payload, dict) else []
            if not isinstance(gates, list):
                gates = []

            frame_dir = self.frames_dir / name
            rgb_path = self.output_root / str(frame_record["rgb_path"])
            overlay_path = frame_dir / "landmarker_overlay.jpg"
            review_path = frame_dir / "landmarker_review.json"
            nearest_target = gates[0] if gates and isinstance(gates[0], dict) else None

            render_gate_overlay_image(
                rgb_path=rgb_path,
                payload=payload,
                output_path=overlay_path,
                title=f"{name} | Landmarker Target Overlay",
                subtitle="controller egress; highlighted marker uses nearest current-frame XYZ",
            )
            sidecar = {
                "frame_id": int(frame_record["frame_id"]),
                "frame_name": name,
                "controller_jsonl": self._relative(controller_path),
                "gate_count": len(gates),
                "nearest_target": nearest_target,
                "overlay_path": self._relative(overlay_path),
            }
            review_path.write_text(json.dumps(sidecar, indent=2, sort_keys=False) + "\n", encoding="utf-8")
            frame_record["landmarker_review"] = self._relative(review_path)
            frame_record["landmarker_overlay"] = self._relative(overlay_path)
        self._write_manifest()

    def finish_run(self, stats: Any | None = None) -> None:
        summary = {
            "schema_version": 1,
            "run_mode": self.config.run_mode,
            "has_truth": bool(self.config.has_truth),
            "metadata": self.metadata,
            "stats": _jsonable(stats),
        }
        (self.output_root / "run_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        self._write_manifest()

    def _write_manifest(self) -> None:
        manifest = {
            "schema_version": 1,
            "run_mode": self.config.run_mode,
            "has_truth": bool(self.config.has_truth),
            "frame_count": len(self.frames),
            "metadata": self.metadata,
            "frames": self.frames,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "review_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    def _relative(self, path: Path) -> str:
        resolved = Path(path).expanduser()
        try:
            return str(resolved.resolve().relative_to(self.output_root.resolve()))
        except ValueError:
            return str(resolved)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


__all__ = ["ReviewConfig", "ReviewSink"]
