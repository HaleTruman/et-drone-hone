#!/usr/bin/env python3
"""CLI for the vision runtime."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline import DEFAULT_PRESET_PATH, PipelineOptions, run_pipeline, utc_run_id
else:  # pragma: no cover
    from .pipeline import DEFAULT_PRESET_PATH, PipelineOptions, run_pipeline, utc_run_id


VISION_ROOT = Path(__file__).resolve().parent
SOURCE_FRAMES_DIRNAME = "vision_frames"
LAUNCH_OUTPUT_DIRNAME = "vision_run"
JSON_BODY_LIMIT_BYTES = 1024 * 1024


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def build_allowed_paths(run_root: Path, manifest: dict) -> tuple[set[Path], set[Path]]:
    run_root = run_root.resolve()
    run_paths = {run_root / "run_manifest.json", run_root / "status.json", run_root / "latest.json"}
    source_paths: set[Path] = set()
    for entry in manifest.get("frame_index", []):
        if not isinstance(entry, dict):
            continue
        final_json = entry.get("final_json")
        if final_json:
            run_paths.add(Path(str(final_json)).resolve())
        source_path = entry.get("source_path")
        if source_path:
            source_paths.add(Path(str(source_path)).resolve())
        debug_artifacts = entry.get("debug_artifacts") if isinstance(entry.get("debug_artifacts"), dict) else {}
        for value in debug_artifacts.values():
            if value:
                run_paths.add(Path(str(value)).resolve())
    safe_run_paths = set()
    for path in run_paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(run_root)
        except ValueError:
            continue
        safe_run_paths.add(resolved)
    return safe_run_paths, source_paths


def has_jpeg_frames(source_dir: Path) -> bool:
    return any(path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"} for path in source_dir.iterdir())


class ReviewState:
    def __init__(self, run_root: Path | None = None):
        self.lock = threading.RLock()
        self.run_root: Path | None = None
        self.manifest: dict | None = None
        self.allowed_run_paths: set[Path] = set()
        self.allowed_source_paths: set[Path] = set()
        self.debug_thread: threading.Thread | None = None
        self.debug_job = self._idle_debug_job()
        if run_root is not None:
            self.load_run_root(run_root)

    @staticmethod
    def _idle_debug_job() -> dict:
        return {
            "status": "idle",
            "runFolder": None,
            "sourceDir": None,
            "outputRoot": None,
            "runRoot": None,
            "manifestPath": None,
            "frameCount": 0,
            "error": None,
        }

    def _load_run_root_unlocked(self, run_root: Path) -> None:
        run_root = run_root.resolve()
        manifest_path = run_root / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"run_manifest.json not found under {run_root}")
        manifest = read_json(manifest_path)
        self.run_root = run_root
        self.manifest = manifest
        self.allowed_run_paths, self.allowed_source_paths = build_allowed_paths(run_root, manifest)

    def load_run_root(self, run_root: Path) -> None:
        with self.lock:
            self._load_run_root_unlocked(run_root)

    def manifest_snapshot(self) -> dict | None:
        with self.lock:
            if self.run_root is not None and (self.run_root / "run_manifest.json").exists():
                self._load_run_root_unlocked(self.run_root)
            return dict(self.manifest) if self.manifest is not None else None

    def frame_entry(self, ordinal_text: str) -> dict | None:
        try:
            ordinal = int(ordinal_text)
        except ValueError:
            return None
        with self.lock:
            manifest = self.manifest or {}
            for entry in manifest.get("frame_index", []):
                if isinstance(entry, dict) and int(entry.get("frame_ordinal", -1)) == ordinal:
                    return dict(entry)
        return None

    def is_allowed_source_path(self, path: Path) -> bool:
        with self.lock:
            return path.resolve() in self.allowed_source_paths

    def is_allowed_run_path(self, path: Path) -> bool:
        with self.lock:
            return path.resolve() in self.allowed_run_paths

    def is_allowed_file_path(self, path: Path) -> bool:
        resolved = path.resolve()
        with self.lock:
            return resolved in self.allowed_source_paths or resolved in self.allowed_run_paths

    def start_debug_run(self, run_folder: str) -> dict:
        run_folder_path = Path(str(run_folder).strip()).expanduser()
        if not run_folder_path.is_absolute():
            raise ValueError("runFolder must be an absolute path")
        run_folder_path = run_folder_path.resolve()
        if not run_folder_path.exists() or not run_folder_path.is_dir():
            raise FileNotFoundError(f"run folder not found: {run_folder_path}")

        source_dir = run_folder_path / SOURCE_FRAMES_DIRNAME
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"{SOURCE_FRAMES_DIRNAME} folder not found: {source_dir}")
        if not has_jpeg_frames(source_dir):
            raise ValueError(f"{SOURCE_FRAMES_DIRNAME} does not contain JPEG frames: {source_dir}")

        output_root = run_folder_path / LAUNCH_OUTPUT_DIRNAME
        run_id = utc_run_id()
        run_root = output_root / run_id
        with self.lock:
            if self.debug_thread is not None and self.debug_thread.is_alive():
                raise RuntimeError("a debug run is already running")
            self.debug_job = {
                "status": "running",
                "runFolder": str(run_folder_path),
                "sourceDir": str(source_dir),
                "outputRoot": str(output_root),
                "runRoot": str(run_root),
                "manifestPath": str(run_root / "run_manifest.json"),
                "frameCount": 0,
                "error": None,
            }
            self.debug_thread = threading.Thread(
                target=self._run_debug_job,
                args=(source_dir, output_root, run_id),
                name=f"vision-debug-run-{run_id}",
                daemon=True,
            )
            self.debug_thread.start()
            return dict(self.debug_job)

    def _run_debug_job(self, source_dir: Path, output_root: Path, run_id: str) -> None:
        try:
            manifest = run_pipeline(
                PipelineOptions(
                    mode="batch",
                    source_dir=source_dir,
                    output_root=output_root,
                    run_id=run_id,
                    debug=True,
                )
            )
            with self.lock:
                self._load_run_root_unlocked(Path(manifest.run_root))
                self.debug_job.update(
                    {
                        "status": "complete",
                        "runRoot": manifest.run_root,
                        "manifestPath": str(Path(manifest.run_root) / "run_manifest.json"),
                        "frameCount": manifest.frame_count,
                        "error": None,
                    }
                )
        except Exception as exc:
            with self.lock:
                self.debug_job.update({"status": "failed", "error": str(exc)})

    def debug_status(self) -> dict:
        with self.lock:
            snapshot = dict(self.debug_job)
        run_root = snapshot.get("runRoot")
        if snapshot.get("status") == "running" and run_root:
            status_path = Path(str(run_root)) / "status.json"
            if status_path.exists():
                try:
                    status_payload = read_json(status_path)
                    snapshot["frameCount"] = int(status_payload.get("frame_count") or 0)
                    snapshot["latestFrame"] = status_payload.get("latest_frame")
                except Exception:
                    pass
        return snapshot


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, state: ReviewState, static_root: Path, **kwargs):
        self.state = state
        self.static_root = static_root.resolve()
        super().__init__(*args, directory=str(self.static_root), **kwargs)

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length <= 0:
            raise ValueError("request body is required")
        if length > JSON_BODY_LIMIT_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_file(self, path: Path, *, content_type: str | None = None) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "missing artifact", "path": str(path)}, status=404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _frame_entry(self, ordinal_text: str) -> dict | None:
        return self.state.frame_entry(ordinal_text)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/manifest":
            manifest = self.state.manifest_snapshot()
            if manifest is None:
                self._send_json({"error": "no run loaded"}, status=404)
                return
            self._send_json(manifest)
            return
        if path == "/api/run-debug/status":
            self._send_json(self.state.debug_status())
            return
        if path == "/api/file":
            query = parse_qs(parsed.query)
            raw = unquote((query.get("path") or [""])[0])
            requested = Path(raw).resolve()
            if self.state.is_allowed_file_path(requested):
                self._send_file(requested)
                return
            self._send_json({"error": "path is not listed in this run manifest", "path": raw}, status=403)
            return
        parts = [part for part in path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "api" and parts[1] == "frame":
            entry = self._frame_entry(parts[2])
            if entry is None:
                self._send_json({"error": "frame not found"}, status=404)
                return
            if len(parts) == 4 and parts[3] == "source":
                source = Path(str(entry.get("source_path", ""))).resolve()
                if self.state.is_allowed_source_path(source):
                    self._send_file(source)
                    return
                self._send_json({"error": "source path not allowed"}, status=403)
                return
            if len(parts) == 4 and parts[3] == "final":
                final_path = Path(str(entry.get("final_json", ""))).resolve()
                if self.state.is_allowed_run_path(final_path):
                    self._send_file(final_path, content_type="application/json; charset=utf-8")
                    return
                self._send_json({"error": "final path not allowed"}, status=403)
                return
            if len(parts) == 5 and parts[3] == "debug":
                stage = parts[4]
                debug_artifacts = entry.get("debug_artifacts") if isinstance(entry.get("debug_artifacts"), dict) else {}
                artifact = debug_artifacts.get(stage)
                if not artifact:
                    self._send_json({"error": "debug artifact missing", "stage": stage}, status=404)
                    return
                artifact_path = Path(str(artifact)).resolve()
                if self.state.is_allowed_run_path(artifact_path):
                    self._send_file(artifact_path, content_type="application/json; charset=utf-8")
                    return
                self._send_json({"error": "debug path not allowed"}, status=403)
                return
            if len(parts) == 4 and parts[3] == "mask":
                debug_artifacts = entry.get("debug_artifacts") if isinstance(entry.get("debug_artifacts"), dict) else {}
                artifact = debug_artifacts.get("color_masking_mask")
                if not artifact:
                    self._send_json({"error": "mask artifact missing"}, status=404)
                    return
                artifact_path = Path(str(artifact)).resolve()
                if self.state.is_allowed_run_path(artifact_path):
                    self._send_file(artifact_path, content_type="application/octet-stream")
                    return
                self._send_json({"error": "mask path not allowed"}, status=403)
                return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/run-debug":
            self._send_json({"error": "unknown endpoint"}, status=404)
            return
        try:
            payload = self._read_json_body()
            run_folder = payload.get("runFolder")
            if not isinstance(run_folder, str) or not run_folder.strip():
                raise ValueError("runFolder must be a non-empty string")
            status_payload = self.state.start_debug_run(run_folder)
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=409)
            return
        except (FileNotFoundError, ValueError) as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        self._send_json(status_payload, status=202)


def serve_review(run_root: Path | None, host: str, port: int) -> None:
    static_root = VISION_ROOT / "review_vision_ui"
    state = ReviewState(run_root.resolve() if run_root is not None else None)
    handler = partial(ReviewHandler, state=state, static_root=static_root)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Vision review UI: http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic vision runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="process a batch directory or one explicit frame")
    run.add_argument("--source-dir", type=Path)
    run.add_argument("--single-frame", type=Path)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--run-id")
    run.add_argument("--debug", action="store_true")
    run.add_argument("--max-frames", type=int)
    run.add_argument("--preset", type=Path, default=DEFAULT_PRESET_PATH)

    watch = subparsers.add_parser("watch", help="poll a directory for stable new JPEGs")
    watch.add_argument("--source-dir", type=Path, required=True)
    watch.add_argument("--output-root", type=Path, required=True)
    watch.add_argument("--run-id")
    watch.add_argument("--debug", action="store_true")
    watch.add_argument("--max-frames", type=int)
    watch.add_argument("--preset", type=Path, default=DEFAULT_PRESET_PATH)
    watch.add_argument("--poll-interval-s", type=float, default=0.1)
    watch.add_argument("--stable-frame-s", type=float, default=0.1)

    review = subparsers.add_parser("review", help="serve the review UI and optional debug launcher")
    review.add_argument("--run-root", type=Path)
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8788)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "review":
        serve_review(args.run_root, args.host, args.port)
        return 0
    if args.command == "run":
        if bool(args.source_dir) == bool(args.single_frame):
            parser.error("run requires exactly one of --source-dir or --single-frame")
        options = PipelineOptions(
            mode="single" if args.single_frame else "batch",
            source_dir=args.source_dir,
            single_frame=args.single_frame,
            output_root=args.output_root,
            run_id=args.run_id,
            debug=args.debug,
            max_frames=args.max_frames,
            preset_path=args.preset,
        )
    else:
        options = PipelineOptions(
            mode="watch",
            source_dir=args.source_dir,
            output_root=args.output_root,
            run_id=args.run_id,
            debug=args.debug,
            max_frames=args.max_frames,
            preset_path=args.preset,
            poll_interval_s=args.poll_interval_s,
            stable_frame_s=args.stable_frame_s,
        )
    manifest = run_pipeline(options)
    print(manifest.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
