import argparse
import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_bp_courselight_main"
SCRIPT_VERSION = "1.0.0"
SCRIPT_DATE = "2026-03-09"


def _load_assets_module():
    module_path = Path(__file__).resolve().parent.parent / "assemble_course_assets.py"
    spec = importlib.util.spec_from_file_location("assemble_course_assets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--asset-dir", dest="asset_dir", default="/Game/Course_Content")
    parser.add_argument("--blueprint-name", dest="blueprint_name", default="BP_CourseLight_Main")
    return parser.parse_known_args()[0]


def generate_light_blueprint(cfg):
    assets_module = _load_assets_module()
    return assets_module.create_or_update_directional_light_blueprint(cfg)


def main() -> None:
    args = _parse_args()
    assets_module = _load_assets_module()
    assets_module.log_script_metadata()
    assets_module.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")
    cfg = assets_module.BlueprintAssetConfig(
        asset_dir=str(args.asset_dir),
        blueprint_name=str(args.blueprint_name),
    )
    assets_module.log(f"Blueprint generation: {cfg.blueprint_path}")
    _, created = generate_light_blueprint(cfg)
    assets_module.log(f"Blueprint asset: {cfg.blueprint_path} ({'created' if created else 'reused'})")
    assets_module.log("Parent class: DirectionalLight")
    assets_module.log("Success")


if __name__ == "__main__":
    main()
