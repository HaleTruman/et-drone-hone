import argparse
import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_m_coursetorus_red"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-09"

COURSE_TORUS_RED = (1.0, 0.0, 0.0, 1.0)


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
    parser.add_argument("--material-name", dest="material_name", default="M_CourseTorus_Red")
    return parser.parse_known_args()[0]


def generate_material(cfg):
    assets_module = _load_assets_module()
    assets_module.ensure_directory(cfg.asset_dir)
    return assets_module.create_or_update_color_material(
        cfg,
        assets_module.unreal.LinearColor(*COURSE_TORUS_RED),
    )


def run() -> None:
    args = _parse_args()
    assets_module = _load_assets_module()
    assets_module.log_script_metadata()
    assets_module.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")
    cfg = assets_module.TorusAssetConfig(
        asset_dir=str(args.asset_dir),
        material_name=str(args.material_name),
        mesh_name="SM_CourseTorus_1mOpening",
        inner_diameter_cm=100.0,
        tube_radius_cm=12.0,
        major_steps=48,
        minor_steps=24,
    )

    assets_module.log(f"Material-only generation: {cfg.material_path}")
    _, created = generate_material(cfg)
    assets_module.log(f"Material asset: {cfg.material_path} ({'created' if created else 'reused'})")
    assets_module.log("Success")


if __name__ == "__main__":
    run()
