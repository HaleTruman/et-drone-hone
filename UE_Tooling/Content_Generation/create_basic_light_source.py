import argparse
from dataclasses import dataclass

import unreal


@dataclass(frozen=True)
class Config:
    asset_dir: str
    blueprint_name: str
    intensity_lux: float

    @property
    def blueprint_path(self) -> str:
        return f"{self.asset_dir}/{self.blueprint_name}"


def _log(message: str) -> None:
    unreal.log(f"[CreateBasicLightSource] {message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--asset-dir", dest="asset_dir", default="/Game/Course_Content")
    parser.add_argument("--blueprint-name", dest="blueprint_name", default="BP_CourseLight_Main")
    parser.add_argument("--intensity-lux", dest="intensity_lux", type=float, default=10000.0)
    args, _ = parser.parse_known_args()

    return Config(
        asset_dir=str(args.asset_dir),
        blueprint_name=str(args.blueprint_name),
        intensity_lux=float(args.intensity_lux),
    )


def _ensure_directory(asset_dir: str) -> None:
    if unreal.EditorAssetLibrary.does_directory_exist(asset_dir):
        return
    _require(unreal.EditorAssetLibrary.make_directory(asset_dir), f"Failed to create directory: {asset_dir}")


def _create_or_load_blueprint(cfg: Config) -> tuple[object, bool]:
    if unreal.EditorAssetLibrary.does_asset_exist(cfg.blueprint_path):
        blueprint = unreal.EditorAssetLibrary.load_asset(cfg.blueprint_path)
        _require(blueprint is not None, f"Failed to load blueprint: {cfg.blueprint_path}")
        return blueprint, False

    factory = unreal.BlueprintFactory()
    factory.set_editor_property("parent_class", unreal.DirectionalLight)
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    blueprint = asset_tools.create_asset(cfg.blueprint_name, cfg.asset_dir, unreal.Blueprint, factory)
    _require(blueprint is not None, f"Failed to create blueprint: {cfg.blueprint_path}")
    return blueprint, True


def _configure_blueprint_defaults(blueprint: object, cfg: Config) -> None:
    generated_class = blueprint.generated_class
    _require(generated_class is not None, f"Blueprint has no generated class: {cfg.blueprint_path}")

    cdo = unreal.get_default_object(generated_class)
    _require(cdo is not None, f"Failed to access CDO for: {cfg.blueprint_path}")

    light_component = cdo.get_editor_property("light_component")
    _require(light_component is not None, f"DirectionalLightComponent missing on blueprint CDO: {cfg.blueprint_path}")
    light_component.set_editor_property("intensity", cfg.intensity_lux)
    light_component.set_editor_property("light_color", unreal.Color(255, 255, 255, 255))

    unreal.EditorAssetLibrary.save_asset(cfg.blueprint_path, only_if_is_dirty=False)


def main() -> None:
    cfg = _parse_args()

    _log(f"Asset dir: {cfg.asset_dir}")
    _log(f"Blueprint: {cfg.blueprint_path}")

    _ensure_directory(cfg.asset_dir)
    blueprint, created = _create_or_load_blueprint(cfg)
    _configure_blueprint_defaults(blueprint, cfg)

    _log(f"Result: {'created' if created else 'already_exists'}")
    _log(f"Blueprint asset: {cfg.blueprint_path}")
    _log(f"Parent class: {blueprint.parent_class.get_name()}")
    _log(f"Intensity: {cfg.intensity_lux}")
    _log("Success")


if __name__ == "__main__":
    main()
