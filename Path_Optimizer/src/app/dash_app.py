from pathlib import Path

from dash import Dash

from app.layout import build_layout
from app.live_callbacks import register_live_callbacks
from app.live_data import discover_live_run_dirs, live_run_option


ROOT_DIR = str(Path(__file__).resolve().parents[2])


def create_app() -> Dash:
    live_run_options = _live_run_options()
    app = Dash(__name__, title="Live Run Explorer")
    app.layout = build_layout(live_run_options=live_run_options)
    register_live_callbacks(app, root_dir=ROOT_DIR)
    return app


def _live_run_options() -> list[dict[str, str]]:
    return [live_run_option(path) for path in discover_live_run_dirs(ROOT_DIR)]


def run() -> None:
    app = create_app()
    app.run(debug=True, use_reloader=False)
