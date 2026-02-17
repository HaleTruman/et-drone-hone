from __future__ import annotations

from pathlib import Path

from dash import Dash

from app.callbacks import register_callbacks
from app.layout import build_layout
from opt_engine.scenario_io import discover_scenario_files


def create_app() -> Dash:
    here = Path(__file__).resolve().parent
    root_dir = here.parent
    scenario_paths = discover_scenario_files(root_dir)
    scenario_options = [{"label": p.name, "value": str(p)} for p in scenario_paths]

    app = Dash(__name__)
    app.layout = build_layout(scenario_options=scenario_options)
    register_callbacks(app)
    return app


def run() -> None:
    app = create_app()
    app.run(debug=True)
