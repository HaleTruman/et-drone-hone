import argparse
import importlib.util
from pathlib import Path

SCRIPT_NAME = "gen_sm_coursetorus_1mopening"
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
    parser.add_argument("--mesh-name", dest="mesh_name", default="SM_CourseTorus_1mOpening")
    parser.add_argument("--inner-diameter-cm", dest="inner_diameter_cm", type=float, default=100.0)
    parser.add_argument("--tube-radius-cm", dest="tube_radius_cm", type=float, default=12.0)
    parser.add_argument("--major-steps", dest="major_steps", type=int, default=48)
    parser.add_argument("--minor-steps", dest="minor_steps", type=int, default=24)
    return parser.parse_known_args()[0]


def generate_mesh(cfg):
    assets_module = _load_assets_module()
    assets_module.ensure_directory(cfg.asset_dir)
    return assets_module.create_or_update_torus_mesh(cfg)


def run() -> None:
    args = _parse_args()
    assets_module = _load_assets_module()
    assets_module.log_script_metadata()
    assets_module.log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")
    cfg = assets_module.TorusAssetConfig(
        asset_dir=str(args.asset_dir),
        material_name="M_CourseTorus_Red",
        mesh_name=str(args.mesh_name),
        inner_diameter_cm=float(args.inner_diameter_cm),
        tube_radius_cm=float(args.tube_radius_cm),
        major_steps=int(args.major_steps),
        minor_steps=int(args.minor_steps),
    )

    assets_module.log(f"Mesh-only generation: {cfg.mesh_path}")
    _, created = generate_mesh(cfg)
    assets_module.log(f"Torus mesh asset: {cfg.mesh_path} ({'created' if created else 'reused'})")
    assets_module.log("Success")


if __name__ == "__main__":
    run()
