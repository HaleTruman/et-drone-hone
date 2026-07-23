from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from ui.data import discover_runs, frame_image_path, frame_payload, load_run_cached, run_option, run_summary


ROOT_DIR = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(root_dir: str | Path = ROOT_DIR) -> Flask:
    root = Path(root_dir).resolve()
    app = Flask(__name__, static_folder=None)

    @app.after_request
    def prevent_viewer_cache(response):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:filename>")
    def static_asset(filename: str):
        return send_from_directory(STATIC_DIR, filename)

    @app.get("/api/runs")
    def api_runs():
        return jsonify(_json_safe({"runs": [run_option(path) for path in discover_runs(root)]}))

    @app.get("/api/run")
    def api_run():
        run = _selected_run(root)
        return jsonify(_json_safe({"summary": run_summary(run)}))

    @app.get("/api/frame")
    def api_frame():
        run = _selected_run(root)
        frame_index = int(request.args.get("frame", "0"))
        return jsonify(_json_safe(frame_payload(run, frame_index)))

    @app.get("/api/frame-image")
    def api_frame_image():
        run = _selected_run(root)
        frame_index = int(request.args.get("frame", "0"))
        if frame_index < 0 or frame_index >= len(run.frames):
            abort(404)
        image_path = frame_image_path(run, frame_index)
        if not image_path.is_file():
            abort(404)
        return send_file(image_path, mimetype="image/jpeg", conditional=True)

    return app


def run(host: str = "127.0.0.1", port: int = 8050, debug: bool = True) -> None:
    app = create_app()
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Flight log review UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()
    run(host=args.host, port=args.port, debug=not args.no_debug)
    return 0


def _selected_run(root: Path):
    requested = request.args.get("run")
    runs = discover_runs(root)
    if not runs:
        abort(404, "No run logs found.")
    if requested:
        resolved = Path(requested).resolve()
        allowed = {path.resolve() for path in runs}
        if resolved not in allowed:
            abort(404, "Unknown run.")
        return load_run_cached(str(resolved))
    return load_run_cached(str(runs[0]))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
