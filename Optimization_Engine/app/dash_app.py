from __future__ import annotations

from pathlib import Path

from dash import Dash

from app.callbacks import register_callbacks
from app.layout import build_layout
from opt_engine.scenario_io import list_scenarios


def create_app() -> Dash:
    here = Path(__file__).resolve().parent
    scenarios_dir = (here.parent / "data" / "scenarios").resolve()
    scenario_paths = list_scenarios(scenarios_dir)
    scenario_options = [{"label": p.stem, "value": str(p)} for p in scenario_paths]

    app = Dash(__name__)
    app.layout = build_layout(scenario_options=scenario_options)
    register_callbacks(app)
    return app


def run() -> None:
    app = create_app()
    app.run(debug=True)

