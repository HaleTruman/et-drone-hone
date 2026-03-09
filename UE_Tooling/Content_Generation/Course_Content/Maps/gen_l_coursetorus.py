import argparse
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path

import unreal

SCRIPT_NAME = "gen_l_coursetorus"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-09"


@dataclass(frozen=True)
class Config:
    level_path: str
    asset_dir: str
    material_name: str
    mesh_name: str
    light_blueprint_name: str
    light_actor_label: str
    light_actor_tag: str
    light_height_cm: float
    light_pitch_deg: float
    light_yaw_deg: float
    light_roll_deg: float
    actor_label_prefix: str
    actor_tag: str
    count: int
    seed: int
    x_min_cm: float
    x_max_cm: float
    y_min_cm: float
    y_max_cm: float
    ground_z_cm: float
    ground_offset_cm: float
    z_jitter_cm: float
    inner_diameter_cm: float
    tube_radius_cm: float
    major_steps: int
    minor_steps: int
    equal_xy_variation: bool
    equal_xyz_variation: bool

    @property
    def material_path(self) -> str:
        return f"{self.asset_dir}/{self.material_name}"

    @property
    def mesh_path(self) -> str:
        return f"{self.asset_dir}/{self.mesh_name}"

    @property
    def light_blueprint_path(self) -> str:
        return f"{self.asset_dir}/{self.light_blueprint_name}"

    @property
    def level_name(self) -> str:
        return self.level_path.rsplit("/", 1)[-1]

    @property
    def level_object_path(self) -> str:
        return f"{self.level_path}.{self.level_name}"

    @property
    def level_dir(self) -> str:
        return self.level_path.rsplit("/", 1)[0]


def _log(message: str) -> None:
    unreal.log(f"[CreateRandomTorusCourse] {message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_module(module_name: str, relative_path: str):
    module_path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    _require(spec is not None and spec.loader is not None, f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--level", dest="level_path", default="/Game/Course_Content/Maps/L_CourseTorus")
    parser.add_argument("--asset-dir", dest="asset_dir", default="/Game/Course_Content")
    parser.add_argument("--material-name", dest="material_name", default="M_CourseTorus_Red")
    parser.add_argument("--mesh-name", dest="mesh_name", default="SM_CourseTorus_1mOpening")
    parser.add_argument("--light-blueprint-name", dest="light_blueprint_name", default="BP_CourseLight_Main")
    parser.add_argument("--light-actor-label", dest="light_actor_label", default="CourseLight_Main")
    parser.add_argument("--light-actor-tag", dest="light_actor_tag", default="CourseLightAuto")
    parser.add_argument("--light-height-cm", dest="light_height_cm", type=float, default=3000.0)
    parser.add_argument("--light-pitch-deg", dest="light_pitch_deg", type=float, default=-45.0)
    parser.add_argument("--light-yaw-deg", dest="light_yaw_deg", type=float, default=-30.0)
    parser.add_argument("--light-roll-deg", dest="light_roll_deg", type=float, default=0.0)
    parser.add_argument("--actor-label-prefix", dest="actor_label_prefix", default="CourseTorus_")
    parser.add_argument("--actor-tag", dest="actor_tag", default="CourseTorusAuto")
    parser.add_argument("--count", dest="count", type=int, default=64)
    parser.add_argument("--seed", dest="seed", type=int, default=1337)
    parser.add_argument("--x-min-cm", dest="x_min_cm", type=float, default=-9000.0)
    parser.add_argument("--x-max-cm", dest="x_max_cm", type=float, default=9000.0)
    parser.add_argument("--y-min-cm", dest="y_min_cm", type=float, default=-9000.0)
    parser.add_argument("--y-max-cm", dest="y_max_cm", type=float, default=9000.0)
    parser.add_argument("--ground-z-cm", dest="ground_z_cm", type=float, default=0.0)
    parser.add_argument("--ground-offset-cm", dest="ground_offset_cm", type=float, default=180.0)
    parser.add_argument("--z-jitter-cm", dest="z_jitter_cm", type=float, default=120.0)
    parser.add_argument("--inner-diameter-cm", dest="inner_diameter_cm", type=float, default=100.0)
    parser.add_argument("--tube-radius-cm", dest="tube_radius_cm", type=float, default=12.0)
    parser.add_argument("--major-steps", dest="major_steps", type=int, default=48)
    parser.add_argument("--minor-steps", dest="minor_steps", type=int, default=24)
    parser.add_argument("--allow-unequal-xy-variation", dest="allow_unequal_xy_variation", action="store_true")
    parser.add_argument("--allow-unequal-xyz-variation", dest="allow_unequal_xyz_variation", action="store_true")

    args, _ = parser.parse_known_args()

    return Config(
        level_path=str(args.level_path),
        asset_dir=str(args.asset_dir),
        material_name=str(args.material_name),
        mesh_name=str(args.mesh_name),
        light_blueprint_name=str(args.light_blueprint_name),
        light_actor_label=str(args.light_actor_label),
        light_actor_tag=str(args.light_actor_tag),
        light_height_cm=float(args.light_height_cm),
        light_pitch_deg=float(args.light_pitch_deg),
        light_yaw_deg=float(args.light_yaw_deg),
        light_roll_deg=float(args.light_roll_deg),
        actor_label_prefix=str(args.actor_label_prefix),
        actor_tag=str(args.actor_tag),
        count=max(1, int(args.count)),
        seed=int(args.seed),
        x_min_cm=float(args.x_min_cm),
        x_max_cm=float(args.x_max_cm),
        y_min_cm=float(args.y_min_cm),
        y_max_cm=float(args.y_max_cm),
        ground_z_cm=float(args.ground_z_cm),
        ground_offset_cm=float(args.ground_offset_cm),
        z_jitter_cm=max(0.0, float(args.z_jitter_cm)),
        inner_diameter_cm=float(args.inner_diameter_cm),
        tube_radius_cm=float(args.tube_radius_cm),
        major_steps=max(3, int(args.major_steps)),
        minor_steps=max(3, int(args.minor_steps)),
        equal_xy_variation=not bool(args.allow_unequal_xy_variation),
        equal_xyz_variation=not bool(args.allow_unequal_xyz_variation),
    )


def _level_exists(cfg: Config) -> bool:
    return unreal.EditorAssetLibrary.does_asset_exist(cfg.level_path) or unreal.EditorAssetLibrary.does_asset_exist(cfg.level_object_path)


def _load_or_create_level(cfg: Config) -> bool:
    if _level_exists(cfg):
        _require(unreal.EditorLevelLibrary.load_level(cfg.level_path), f"Failed to load level: {cfg.level_path}")
        return False

    if not unreal.EditorAssetLibrary.does_directory_exist(cfg.level_dir):
        _require(unreal.EditorAssetLibrary.make_directory(cfg.level_dir), f"Failed to create directory: {cfg.level_dir}")
    _require(hasattr(unreal.EditorLevelLibrary, "new_level"), "EditorLevelLibrary.new_level is not available.")
    _require(unreal.EditorLevelLibrary.new_level(cfg.level_path), f"Failed to create level: {cfg.level_path}")
    _require(unreal.EditorLevelLibrary.save_current_level(), f"Failed to save newly created level: {cfg.level_path}")
    _require(unreal.EditorLevelLibrary.load_level(cfg.level_path), f"Failed to load created level: {cfg.level_path}")
    return True


def _is_managed_actor(actor: object, cfg: Config) -> bool:
    try:
        tags = [str(tag) for tag in list(actor.tags)]
    except Exception:
        tags = []
    if cfg.actor_tag in tags:
        return True
    try:
        label = actor.get_actor_label()
    except Exception:
        return False
    return str(label).startswith(cfg.actor_label_prefix)


def _clear_existing_toruses(cfg: Config) -> int:
    removed = 0
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if not _is_managed_actor(actor, cfg):
            continue
        if unreal.EditorLevelLibrary.destroy_actor(actor):
            removed += 1
    return removed


def _axis_bounds(cfg: Config) -> tuple[float, float, float, float, float, float]:
    x_min = min(cfg.x_min_cm, cfg.x_max_cm)
    x_max = max(cfg.x_min_cm, cfg.x_max_cm)
    y_min = min(cfg.y_min_cm, cfg.y_max_cm)
    y_max = max(cfg.y_min_cm, cfg.y_max_cm)
    z_min = cfg.ground_z_cm + cfg.ground_offset_cm
    z_max = z_min + max(0.0, cfg.z_jitter_cm)

    x_span = max(0.0, x_max - x_min)
    y_span = max(0.0, y_max - y_min)
    z_span = max(0.0, z_max - z_min)

    if cfg.equal_xyz_variation:
        target_span = max(x_span, y_span, z_span)
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        x_min = x_center - (target_span / 2.0)
        x_max = x_center + (target_span / 2.0)
        y_min = y_center - (target_span / 2.0)
        y_max = y_center + (target_span / 2.0)
        z_max = z_min + target_span
    elif cfg.equal_xy_variation:
        target_span = max(x_span, y_span)
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        x_min = x_center - (target_span / 2.0)
        x_max = x_center + (target_span / 2.0)
        y_min = y_center - (target_span / 2.0)
        y_max = y_center + (target_span / 2.0)

    return x_min, x_max, y_min, y_max, z_min, z_max


def _actor_location(cfg: Config, rng: random.Random) -> unreal.Vector:
    x_min, x_max, y_min, y_max, z_min, z_max = _axis_bounds(cfg)
    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    z = rng.uniform(z_min, z_max)
    return unreal.Vector(x, y, z)


def _actor_rotation(rng: random.Random) -> unreal.Rotator:
    pitch = rng.uniform(-180.0, 180.0)
    yaw = rng.uniform(-180.0, 180.0)
    roll = rng.uniform(-180.0, 180.0)
    return unreal.Rotator(pitch, yaw, roll)


def _apply_actor_tag(actor: object, tag: str) -> None:
    tags = []
    try:
        tags = [str(item) for item in list(actor.tags)]
    except Exception:
        tags = []
    if tag in tags:
        return
    tags.append(tag)
    actor.set_editor_property("tags", tags)


def _is_managed_light_actor(actor: object, cfg: Config) -> bool:
    try:
        tags = [str(tag) for tag in list(actor.tags)]
    except Exception:
        tags = []
    if cfg.light_actor_tag in tags:
        return True
    try:
        label = str(actor.get_actor_label())
    except Exception:
        return False
    return label == cfg.light_actor_label


def _load_blueprint_class(blueprint_path: str):
    if hasattr(unreal.EditorAssetLibrary, "load_blueprint_class"):
        loaded_class = unreal.EditorAssetLibrary.load_blueprint_class(blueprint_path)
        if loaded_class is not None:
            return loaded_class

    blueprint_asset = unreal.EditorAssetLibrary.load_asset(blueprint_path)
    _require(blueprint_asset is not None, f"Failed to load blueprint asset: {blueprint_path}")
    generated_class = None
    try:
        generated_class = blueprint_asset.generated_class
    except Exception:
        generated_class = None
    if generated_class is None:
        try:
            generated_class = blueprint_asset.get_editor_property("generated_class")
        except Exception:
            generated_class = None
    if isinstance(generated_class, str):
        generated_class = unreal.load_object(None, generated_class)
    _require(generated_class is not None, f"Failed to resolve generated class: {blueprint_path}")
    return generated_class


def _ensure_course_light_actor(cfg: Config, light_blueprint_path: str) -> tuple[bool, int]:
    managed_lights = []
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if _is_managed_light_actor(actor, cfg):
            managed_lights.append(actor)

    removed = 0
    light_actor = None
    for actor in managed_lights:
        if light_actor is None:
            light_actor = actor
            continue
        if unreal.EditorLevelLibrary.destroy_actor(actor):
            removed += 1

    created = False
    if light_actor is None:
        light_class = _load_blueprint_class(light_blueprint_path)
        spawn_location = unreal.Vector(0.0, 0.0, cfg.ground_z_cm + cfg.light_height_cm)
        spawn_rotation = unreal.Rotator(cfg.light_pitch_deg, cfg.light_yaw_deg, cfg.light_roll_deg)
        light_actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            light_class,
            spawn_location,
            spawn_rotation,
            False,
        )
        _require(light_actor is not None, f"Failed to spawn light actor from blueprint: {light_blueprint_path}")
        created = True

    light_actor.set_actor_label(cfg.light_actor_label)
    _apply_actor_tag(light_actor, cfg.light_actor_tag)
    return created, removed


def _spawn_toruses(cfg: Config, static_mesh: object, material: object) -> int:
    rng = random.Random(cfg.seed)
    spawned = 0

    for index in range(1, int(cfg.count) + 1):
        location = _actor_location(cfg, rng)
        rotation = _actor_rotation(rng)
        actor = unreal.EditorLevelLibrary.spawn_actor_from_class(
            unreal.StaticMeshActor,
            location,
            rotation,
            False,
        )
        _require(actor is not None, f"Failed to spawn torus actor index={index}")

        label = f"{cfg.actor_label_prefix}{index:03d}"
        actor.set_actor_label(label)
        _apply_actor_tag(actor, cfg.actor_tag)

        smc = actor.get_component_by_class(unreal.StaticMeshComponent)
        _require(smc is not None, f"StaticMeshComponent missing on '{label}'")
        smc.set_static_mesh(static_mesh)
        smc.set_material(0, material)

        spawned += 1
    return spawned


def main() -> None:
    cfg = _parse_args()
    _log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")
    material_module = _load_module("gen_m_coursetorus_red", "Materials/gen_m_coursetorus_red.py")
    mesh_module = _load_module("gen_sm_coursetorus_1mopening", "Meshes/gen_sm_coursetorus_1mopening.py")
    light_module = _load_module("gen_bp_courselight_main", "Course_Blueprints/gen_bp_courselight_main.py")
    assets_module = _load_module("asset_assembly", "asset_assembly.py")
    assets_module.log_script_metadata()
    x_min, x_max, y_min, y_max, z_min, z_max = _axis_bounds(cfg)

    _log(f"Level: {cfg.level_path}")
    _log(f"Asset dir: {cfg.asset_dir}")
    _log(f"Seed: {cfg.seed}")
    _log(f"Spawn count: {cfg.count}")
    _log(f"Bounds XY cm (effective): x[{x_min},{x_max}] y[{y_min},{y_max}]")
    _log(f"Bounds Z cm (effective): z[{z_min},{z_max}]")
    _log(
        "Light placement: "
        f"label={cfg.light_actor_label} tag={cfg.light_actor_tag} "
        f"height_cm={cfg.light_height_cm} rot=({cfg.light_pitch_deg},{cfg.light_yaw_deg},{cfg.light_roll_deg})"
    )
    _log(f"Equal XY variation: {cfg.equal_xy_variation}")
    _log(f"Equal XYZ variation: {cfg.equal_xyz_variation}")
    _log(f"Z offset cm: ground_z={cfg.ground_z_cm} offset={cfg.ground_offset_cm} jitter={cfg.z_jitter_cm}")

    asset_cfg = assets_module.TorusAssetConfig(
        asset_dir=cfg.asset_dir,
        material_name=cfg.material_name,
        mesh_name=cfg.mesh_name,
        inner_diameter_cm=cfg.inner_diameter_cm,
        tube_radius_cm=cfg.tube_radius_cm,
        major_steps=cfg.major_steps,
        minor_steps=cfg.minor_steps,
    )

    material, material_created = material_module.generate_material(asset_cfg)
    static_mesh, mesh_created = mesh_module.generate_mesh(asset_cfg)
    light_cfg = assets_module.BlueprintAssetConfig(
        asset_dir=cfg.asset_dir,
        blueprint_name=cfg.light_blueprint_name,
    )
    _, light_created = light_module.generate_light_blueprint(light_cfg)
    level_created = _load_or_create_level(cfg)
    placed_light_created, removed_lights = _ensure_course_light_actor(cfg, light_cfg.blueprint_path)
    removed = _clear_existing_toruses(cfg)
    spawned = _spawn_toruses(cfg, static_mesh, material)

    _require(unreal.EditorLevelLibrary.save_current_level(), "Failed to save current level")

    _log(f"Material asset: {cfg.material_path} ({'created' if material_created else 'reused'})")
    _log(f"Torus mesh asset: {cfg.mesh_path} ({'created' if mesh_created else 'reused'})")
    _log(f"Light blueprint asset: {light_cfg.blueprint_path} ({'created' if light_created else 'reused'})")
    _log(f"Course light actor: {cfg.light_actor_label} ({'created' if placed_light_created else 'reused'})")
    _log(f"Managed lights removed before placement: {removed_lights}")
    _log(f"Level: {cfg.level_path} ({'created' if level_created else 'loaded'})")
    _log(f"Managed toruses removed before spawn: {removed}")
    _log(f"Spawned toruses: {spawned}")
    _log("Success")


if __name__ == "__main__":
    main()
