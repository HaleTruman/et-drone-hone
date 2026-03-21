"""
Status: in development.
Renamed from `asset_assembly.py` to `assemble_course_assets.py` to mirror the
Course/Drone/IO/WebSocket build-layer naming pattern.

Role:
- Execute reusable course-domain asset assembly work inside Unreal.
- Centralize create/load/save helpers for course materials, meshes, and blueprints.
- Stay focused on assembly execution, not build orchestration.
"""

import unreal

SCRIPT_NAME = "assemble_course_assets"
SCRIPT_VERSION = "1.2.0"
SCRIPT_DATE = "2026-03-15"


class TorusAssetConfig:
    def __init__(
        self,
        asset_dir: str,
        material_name: str,
        mesh_name: str,
        inner_diameter_cm: float,
        tube_radius_cm: float,
        major_steps: int,
        minor_steps: int,
    ) -> None:
        self.asset_dir = asset_dir
        self.material_name = material_name
        self.mesh_name = mesh_name
        self.inner_diameter_cm = inner_diameter_cm
        self.tube_radius_cm = tube_radius_cm
        self.major_steps = major_steps
        self.minor_steps = minor_steps

    @property
    def material_path(self) -> str:
        return f"{self.asset_dir}/{self.material_name}"

    @property
    def mesh_path(self) -> str:
        return f"{self.asset_dir}/{self.mesh_name}"


class BlueprintAssetConfig:
    def __init__(
        self,
        asset_dir: str,
        blueprint_name: str,
    ) -> None:
        self.asset_dir = asset_dir
        self.blueprint_name = blueprint_name

    @property
    def blueprint_path(self) -> str:
        return f"{self.asset_dir}/{self.blueprint_name}"


def log(message: str) -> None:
    unreal.log(f"[CourseAssetAssembly] {message}")


def log_script_metadata() -> None:
    log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def ensure_directory(asset_dir: str) -> None:
    if unreal.EditorAssetLibrary.does_directory_exist(asset_dir):
        return
    require(unreal.EditorAssetLibrary.make_directory(asset_dir), f"Failed to create directory: {asset_dir}")


def save_asset(asset_path: str) -> None:
    require(
        unreal.EditorAssetLibrary.save_asset(asset_path, only_if_is_dirty=False),
        f"Failed to save asset: {asset_path}",
    )


def create_or_load_asset(asset_path: str, asset_name: str, asset_dir: str, asset_class, factory) -> tuple[object, bool]:
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        asset = unreal.EditorAssetLibrary.load_asset(asset_path)
        require(asset is not None, f"Failed to load asset: {asset_path}")
        return asset, False
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset = asset_tools.create_asset(asset_name, asset_dir, asset_class, factory)
    require(asset is not None, f"Failed to create asset: {asset_path}")
    return asset, True


def load_asset_if_exists(asset_path: str) -> object | None:
    if not unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return None
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    require(asset is not None, f"Failed to load asset: {asset_path}")
    return asset


def create_or_update_color_material(cfg: TorusAssetConfig, color: unreal.LinearColor) -> tuple[object, bool]:
    material, created = create_or_load_asset(
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
    require(color_expr is not None, "Failed to create Constant3Vector expression")
    color_expr.constant = color
    require(
        unreal.MaterialEditingLibrary.connect_material_property(
            color_expr,
            "",
            unreal.MaterialProperty.MP_BASE_COLOR,
        ),
        "Failed to connect Constant3Vector to BaseColor",
    )
    unreal.MaterialEditingLibrary.recompile_material(material)
    save_asset(cfg.material_path)
    return material, created


def create_or_update_torus_mesh(cfg: TorusAssetConfig) -> tuple[object, bool]:
    require(cfg.inner_diameter_cm > 0.0, "--inner-diameter-cm must be > 0")
    require(cfg.tube_radius_cm > 0.0, "--tube-radius-cm must be > 0")

    inner_radius_cm = cfg.inner_diameter_cm / 2.0
    major_radius_cm = inner_radius_cm + cfg.tube_radius_cm
    minor_radius_cm = cfg.tube_radius_cm

    dyn_mesh = unreal.new_object(unreal.DynamicMesh)
    require(dyn_mesh is not None, "Failed to allocate DynamicMesh")
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

    static_mesh = load_asset_if_exists(cfg.mesh_path)
    created = static_mesh is None

    if created:
        require(
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
                require(
                    outcome == unreal.GeometryScriptOutcomePins.SUCCESS,
                    f"CreateNewStaticMeshAssetFromMesh failed: {outcome}",
                )
        else:
            static_mesh = result
        require(static_mesh is not None, f"Failed to create static mesh: {cfg.mesh_path}")
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
                require(outcome == unreal.GeometryScriptOutcomePins.SUCCESS, f"CopyMeshToStaticMesh failed: {outcome}")

    save_asset(cfg.mesh_path)
    return static_mesh, created


def create_or_load_blueprint(cfg: BlueprintAssetConfig, parent_class) -> tuple[object, bool]:
    if unreal.EditorAssetLibrary.does_asset_exist(cfg.blueprint_path):
        blueprint = unreal.EditorAssetLibrary.load_asset(cfg.blueprint_path)
        require(blueprint is not None, f"Failed to load blueprint: {cfg.blueprint_path}")
        return blueprint, False

    factory = unreal.BlueprintFactory()
    factory.set_editor_property("parent_class", parent_class)
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    blueprint = asset_tools.create_asset(cfg.blueprint_name, cfg.asset_dir, unreal.Blueprint, factory)
    require(blueprint is not None, f"Failed to create blueprint: {cfg.blueprint_path}")
    return blueprint, True


def create_or_update_directional_light_blueprint(cfg: BlueprintAssetConfig) -> tuple[object, bool]:
    ensure_directory(cfg.asset_dir)
    blueprint, created = create_or_load_blueprint(cfg, unreal.DirectionalLight)
    save_asset(cfg.blueprint_path)
    return blueprint, created
