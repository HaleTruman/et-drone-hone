import argparse
import random
from dataclasses import dataclass

import unreal


@dataclass(frozen=True)
class Config:
    level_path: str
    asset_dir: str
    material_name: str
    mesh_name: str
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

    @property
    def material_path(self) -> str:
        return f"{self.asset_dir}/{self.material_name}"

    @property
    def mesh_path(self) -> str:
        return f"{self.asset_dir}/{self.mesh_name}"

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


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--level", dest="level_path", default="/Game/Course_Content/Maps/L_CourseTorus")
    parser.add_argument("--asset-dir", dest="asset_dir", default="/Game/Course_Content")
    parser.add_argument("--material-name", dest="material_name", default="M_CourseTorus_Red")
    parser.add_argument("--mesh-name", dest="mesh_name", default="SM_CourseTorus_1mOpening")
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

    args, _ = parser.parse_known_args()

    return Config(
        level_path=str(args.level_path),
        asset_dir=str(args.asset_dir),
        material_name=str(args.material_name),
        mesh_name=str(args.mesh_name),
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
    )


def _ensure_directory(asset_dir: str) -> None:
    if unreal.EditorAssetLibrary.does_directory_exist(asset_dir):
        return
    _require(unreal.EditorAssetLibrary.make_directory(asset_dir), f"Failed to create directory: {asset_dir}")


def _create_or_load_asset(asset_path: str, asset_name: str, asset_dir: str, asset_class, factory) -> tuple[object, bool]:
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        asset = unreal.EditorAssetLibrary.load_asset(asset_path)
        _require(asset is not None, f"Failed to load asset: {asset_path}")
        return asset, False
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset = asset_tools.create_asset(asset_name, asset_dir, asset_class, factory)
    _require(asset is not None, f"Failed to create asset: {asset_path}")
    return asset, True


def _load_asset_if_exists(asset_path: str) -> object | None:
    if not unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return None
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    _require(asset is not None, f"Failed to load asset: {asset_path}")
    return asset


def _create_or_update_red_material(cfg: Config) -> tuple[object, bool]:
    material, created = _create_or_load_asset(
        cfg.material_path,
        cfg.material_name,
        cfg.asset_dir,
        unreal.Material,
        unreal.MaterialFactoryNew(),
    )

    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
    color_expr = unreal.MaterialEditingLibrary.create_material_expression(
        material,
        unreal.MaterialExpressionConstant3Vector,
        -200,
        0,
    )
    _require(color_expr is not None, "Failed to create Constant3Vector expression")
    color_expr.constant = unreal.LinearColor(1.0, 0.0, 0.0, 1.0)
    _require(
        unreal.MaterialEditingLibrary.connect_material_property(
            color_expr,
            "",
            unreal.MaterialProperty.MP_BASE_COLOR,
        ),
        "Failed to connect Constant3Vector to BaseColor",
    )
    unreal.MaterialEditingLibrary.recompile_material(material)
    _require(
        unreal.EditorAssetLibrary.save_asset(cfg.material_path, only_if_is_dirty=False),
        f"Failed to save material: {cfg.material_path}",
    )
    return material, created


def _create_or_update_torus_mesh(cfg: Config) -> tuple[object, bool]:
    _require(cfg.inner_diameter_cm > 0.0, "--inner-diameter-cm must be > 0")
    _require(cfg.tube_radius_cm > 0.0, "--tube-radius-cm must be > 0")

    inner_radius_cm = cfg.inner_diameter_cm / 2.0
    major_radius_cm = inner_radius_cm + cfg.tube_radius_cm
    minor_radius_cm = cfg.tube_radius_cm

    dyn_mesh = unreal.new_object(unreal.DynamicMesh)
    _require(dyn_mesh is not None, "Failed to allocate DynamicMesh")

    prim_opts = unreal.GeometryScriptPrimitiveOptions()
    revolve_opts = unreal.GeometryScriptRevolveOptions()
    unreal.GeometryScript_Primitives.append_torus(
        dyn_mesh,
        prim_opts,
        unreal.Transform(),
        revolve_opts,
        float(major_radius_cm),
        float(minor_radius_cm),
        int(cfg.major_steps),
        int(cfg.minor_steps),
        unreal.GeometryScriptPrimitiveOriginMode.CENTER,
    )

    static_mesh = _load_asset_if_exists(cfg.mesh_path)
    created = static_mesh is None

    if created:
        _require(
            hasattr(unreal, "GeometryScript_NewAssetUtils"),
            "Missing GeometryScript_NewAssetUtils (GeometryScripting plugin/editor module not available).",
        )
        create_opts = unreal.GeometryScriptCreateNewStaticMeshAssetOptions()
        result = unreal.GeometryScript_NewAssetUtils.create_new_static_mesh_asset_from_mesh(
            dyn_mesh,
            cfg.mesh_path,
            create_opts,
        )
        if isinstance(result, tuple):
            static_mesh = result[0]
            outcome = next((x for x in result if isinstance(x, unreal.GeometryScriptOutcomePins)), None)
            if outcome is not None:
                _require(outcome == unreal.GeometryScriptOutcomePins.SUCCESS, f"CreateNewStaticMeshAssetFromMesh failed: {outcome}")
        else:
            static_mesh = result
        _require(static_mesh is not None, f"Failed to create static mesh: {cfg.mesh_path}")
    else:
        copy_opts = unreal.GeometryScriptCopyMeshToAssetOptions()
        write_lod = unreal.GeometryScriptMeshWriteLOD()
        result = unreal.GeometryScript_AssetUtils.copy_mesh_to_static_mesh(
            dyn_mesh,
            static_mesh,
            copy_opts,
            write_lod,
            True,
        )
        if isinstance(result, tuple):
            outcome = next((x for x in result if isinstance(x, unreal.GeometryScriptOutcomePins)), None)
            if outcome is not None:
                _require(outcome == unreal.GeometryScriptOutcomePins.SUCCESS, f"CopyMeshToStaticMesh failed: {outcome}")

    _require(
        unreal.EditorAssetLibrary.save_asset(cfg.mesh_path, only_if_is_dirty=False),
        f"Failed to save static mesh: {cfg.mesh_path}",
    )
    return static_mesh, created


def _level_exists(cfg: Config) -> bool:
    return unreal.EditorAssetLibrary.does_asset_exist(cfg.level_path) or unreal.EditorAssetLibrary.does_asset_exist(cfg.level_object_path)


def _load_or_create_level(cfg: Config) -> bool:
    if _level_exists(cfg):
        _require(unreal.EditorLevelLibrary.load_level(cfg.level_path), f"Failed to load level: {cfg.level_path}")
        return False

    _ensure_directory(cfg.level_dir)
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


def _actor_location(cfg: Config, rng: random.Random) -> unreal.Vector:
    x_min = min(cfg.x_min_cm, cfg.x_max_cm)
    x_max = max(cfg.x_min_cm, cfg.x_max_cm)
    y_min = min(cfg.y_min_cm, cfg.y_max_cm)
    y_max = max(cfg.y_min_cm, cfg.y_max_cm)
    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    z = cfg.ground_z_cm + cfg.ground_offset_cm + rng.uniform(0.0, cfg.z_jitter_cm)
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

    _log(f"Level: {cfg.level_path}")
    _log(f"Asset dir: {cfg.asset_dir}")
    _log(f"Seed: {cfg.seed}")
    _log(f"Spawn count: {cfg.count}")
    _log(f"Bounds XY cm: x[{cfg.x_min_cm},{cfg.x_max_cm}] y[{cfg.y_min_cm},{cfg.y_max_cm}]")
    _log(f"Z offset cm: ground_z={cfg.ground_z_cm} offset={cfg.ground_offset_cm} jitter={cfg.z_jitter_cm}")

    _ensure_directory(cfg.asset_dir)
    material, material_created = _create_or_update_red_material(cfg)
    static_mesh, mesh_created = _create_or_update_torus_mesh(cfg)
    level_created = _load_or_create_level(cfg)
    removed = _clear_existing_toruses(cfg)
    spawned = _spawn_toruses(cfg, static_mesh, material)

    _require(unreal.EditorLevelLibrary.save_current_level(), "Failed to save current level")

    _log(f"Material asset: {cfg.material_path} ({'created' if material_created else 'reused'})")
    _log(f"Torus mesh asset: {cfg.mesh_path} ({'created' if mesh_created else 'reused'})")
    _log(f"Level: {cfg.level_path} ({'created' if level_created else 'loaded'})")
    _log(f"Managed toruses removed before spawn: {removed}")
    _log(f"Spawned toruses: {spawned}")
    _log("Success")


if __name__ == "__main__":
    main()
