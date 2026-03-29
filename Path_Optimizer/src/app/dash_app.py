from __future__ import annotations

import os

from dash import Dash

from app.callbacks import register_callbacks
from app.data import discover_course_files
from app.layout import build_layout


def create_app() -> Dash:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    scenario_paths = discover_course_files(root_dir)
    scenario_options = [{"label": os.path.basename(path), "value": path} for path in scenario_paths]

    app = Dash(__name__)
    app.layout = build_layout(scenario_options=scenario_options)
    register_callbacks(app)
    return app


def run() -> None:
    app = create_app()
    app.run(debug=True)
