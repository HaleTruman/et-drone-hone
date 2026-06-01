from __future__ import annotations

from pathlib import Path

from dash import Dash

from app.callbacks import register_callbacks
from app.data import discover_run_files, load_run
from app.live_callbacks import register_live_callbacks
from app.live_data import discover_live_run_dirs, load_live_run
from app.layout import build_layout


ROOT_DIR = str(Path(__file__).resolve().parents[2])


def create_app() -> Dash:
    run_options = _run_options()
    live_run_options = _live_run_options()
    app = Dash(__name__, title="Racing Stack Run Explorer")
    app.layout = build_layout(run_options=run_options, live_run_options=live_run_options)
    register_callbacks(app, root_dir=ROOT_DIR)
    register_live_callbacks(app, root_dir=ROOT_DIR)
    return app


def _run_options() -> list[dict[str, str]]:
    options = []
    for path in discover_run_files(ROOT_DIR):
        run = load_run(path)
        options.append({"label": f"{run.label}  |  {run.name}", "value": path})
    return options


def _live_run_options() -> list[dict[str, str]]:
    options = []
    for path in discover_live_run_dirs(ROOT_DIR):
        run = load_live_run(path)
        options.append({"label": f"{run.label}  |  {run.name}", "value": path})
    return options


def run() -> None:
    app = create_app()
    app.run(debug=True, use_reloader=False)
