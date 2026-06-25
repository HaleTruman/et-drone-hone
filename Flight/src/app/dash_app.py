from pathlib import Path

from dash import Dash
from flask import abort, request, send_file

from app.layout import build_layout
from app.live_callbacks import register_live_callbacks
from app.live_data import discover_live_run_dirs, frame_image_path, live_run_option, load_live_run_cached


ROOT_DIR = str(Path(__file__).resolve().parents[2])


def create_app() -> Dash:
    live_run_options = _live_run_options()
    app = Dash(__name__, title="Live Run Explorer")
    app.layout = build_layout(live_run_options=live_run_options)
    _register_frame_route(app, root_dir=ROOT_DIR)
    register_live_callbacks(app, root_dir=ROOT_DIR)
    return app


def _live_run_options() -> list[dict[str, str]]:
    return [live_run_option(path) for path in discover_live_run_dirs(ROOT_DIR)]


def _register_frame_route(app: Dash, *, root_dir: str) -> None:
    @app.server.route("/live-frame")
    def _live_frame():
        run_path = request.args.get("run")
        index_value = request.args.get("index", "0")
        if run_path is None or not _is_discovered_run(root_dir, run_path):
            abort(404)
        try:
            run = load_live_run_cached(run_path)
            index = int(index_value)
            if index < 0 or index >= len(run.frames):
                abort(404)
            image_path = frame_image_path(run, run.frames[index])
        except (OSError, ValueError):
            abort(404)
        if not image_path.is_file():
            abort(404)
        return send_file(image_path, mimetype="image/jpeg", conditional=True)


def _is_discovered_run(root_dir: str, run_path: str) -> bool:
    requested = str(Path(run_path).resolve())
    return requested in set(discover_live_run_dirs(root_dir))


def run() -> None:
    app = create_app()
    app.run(debug=True, use_reloader=False)
