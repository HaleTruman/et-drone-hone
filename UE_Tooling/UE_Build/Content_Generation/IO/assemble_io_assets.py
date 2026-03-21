"""
Status: in development.
Renamed from `io_asset_assembly.py` to `assemble_io_assets.py` to mirror the
Course/Drone/IO/WebSocket build-layer naming pattern.

Desired behavior:
- Provide shared, minimal helpers for generating valid IO runtime assets under `/Game/io`.
- Ensure placeholder/invalid assets are replaced deterministically during regeneration.
- Keep asset creation, compile/save, and result logging consistent across IO generation scripts.

Interfaces:
- Imported by IO generators in `IO/Data_Config`, `IO/Drone_Controller`, and `IO/Data_Interface`.
- Exposes helpers for: directory creation, asset deletion/recreation, Blueprint/Struct creation, variable setup, and save/validation.

Assumptions:
- Scripts run in Unreal Editor Python (headless compatible).
- IO assets are owned under `/Game/io/...` and can be safely regenerated in place for deterministic builds.

Success conditions:
- Generated IO `.uasset` files are valid Unreal assets (no unloadable placeholder artifacts).
- Common generation patterns remain centralized and reusable.
- Generators emit explicit `[IOContentResult]` records for audit/debug.
"""

from __future__ import annotations

import json
import os
from typing import Any

import unreal

SCRIPT_NAME = "assemble_io_assets"
SCRIPT_VERSION = "1.1.0"
SCRIPT_DATE = "2026-03-15"

IO_ROOT = "/Game/io"
DATA_CONFIG_DIR = f"{IO_ROOT}/Data_Config"
DRONE_CONTROLLER_DIR = f"{IO_ROOT}/Drone_Controller"
DATA_INTERFACE_DIR = f"{IO_ROOT}/Data_Interface"


def log(message: str) -> None:
    unreal.log(f"[IOAssetAssembly] {message}")


def log_warning(message: str) -> None:
    unreal.log_warning(f"[IOAssetAssembly] {message}")


def log_script_metadata() -> None:
    log(f"Script={SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE}")


def log_result(asset_path: str, status: str, details: dict[str, Any] | None = None) -> None:
    payload = {
        "asset_path": asset_path,
        "status": status,
        "details": details or {},
    }
    log(f"[IOContentResult] {json.dumps(payload, sort_keys=True)}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def split_asset_path(asset_path: str) -> tuple[str, str]:
    require(asset_path.startswith("/Game/"), f"Expected /Game path: {asset_path}")
    parts = asset_path.rsplit("/", 1)
    require(len(parts) == 2 and parts[1], f"Invalid asset path: {asset_path}")
    return parts[0], parts[1]


def ensure_directory(asset_dir: str) -> None:
    if unreal.EditorAssetLibrary.does_directory_exist(asset_dir):
        return
    require(unreal.EditorAssetLibrary.make_directory(asset_dir), f"Failed to create directory: {asset_dir}")


def ensure_standard_directories() -> None:
    for asset_dir in [IO_ROOT, DATA_CONFIG_DIR, DRONE_CONTROLLER_DIR, DATA_INTERFACE_DIR]:
        ensure_directory(asset_dir)


def asset_path_to_disk_path(asset_path: str) -> str:
    package_path = asset_path.split(".", 1)[0]
    relative = package_path.replace("/Game/", "", 1)
    return os.path.join(unreal.Paths.project_content_dir(), relative + ".uasset")


def delete_asset_file_if_exists(asset_path: str) -> bool:
    disk_path = asset_path_to_disk_path(asset_path)
    if not os.path.exists(disk_path):
        return False
    os.remove(disk_path)
    return True


def delete_legacy_file_if_exists(relative_content_path: str) -> bool:
    disk_path = os.path.join(unreal.Paths.project_content_dir(), relative_content_path)
    if not os.path.exists(disk_path):
        return False
    os.remove(disk_path)
    return True


def delete_asset_if_exists(asset_path: str) -> bool:
    removed = False
    try:
        if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
            removed = bool(unreal.EditorAssetLibrary.delete_asset(asset_path))
    except Exception as exc:
        log_warning(f"Editor asset delete failed for {asset_path}: {exc}")
    file_removed = delete_asset_file_if_exists(asset_path)
    return removed or file_removed


def load_asset(asset_path: str) -> Any:
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    require(asset is not None, f"Failed to load asset: {asset_path}")
    return asset


def save_asset(asset_path: str) -> None:
    require(unreal.EditorAssetLibrary.save_asset(asset_path, only_if_is_dirty=False), f"Failed to save {asset_path}")


def set_asset_metadata(asset: Any, key: str, value: str) -> None:
    unreal.EditorAssetLibrary.set_metadata_tag(asset, key, value)


def set_asset_metadata_json(asset: Any, key: str, payload: Any) -> None:
    set_asset_metadata(asset, key, json.dumps(payload, sort_keys=True))


def create_blueprint(asset_path: str, parent_class: type, regenerate: bool = True) -> Any:
    ensure_standard_directories()
    if regenerate:
        delete_asset_if_exists(asset_path)
    elif unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return load_asset(asset_path)

    created = unreal.BlueprintEditorLibrary.create_blueprint_asset_with_parent(asset_path, parent_class)
    if created is None:
        asset_dir, asset_name = split_asset_path(asset_path)
        factory = unreal.BlueprintFactory()
        factory.set_editor_property("parent_class", parent_class)
        created = unreal.AssetToolsHelpers.get_asset_tools().create_asset(asset_name, asset_dir, unreal.Blueprint, factory)
    require(created is not None, f"Failed to create blueprint: {asset_path}")
    return created


def create_user_defined_struct(asset_path: str, regenerate: bool = True) -> Any:
    ensure_standard_directories()
    if regenerate:
        delete_asset_if_exists(asset_path)
    elif unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return load_asset(asset_path)

    struct_factory_cls = getattr(unreal, "StructureFactory", None)
    struct_cls = getattr(unreal, "UserDefinedStruct", None)
    require(struct_factory_cls is not None, "StructureFactory unavailable in this Unreal build.")
    require(struct_cls is not None, "UserDefinedStruct unavailable in this Unreal build.")

    asset_dir, asset_name = split_asset_path(asset_path)
    created = unreal.AssetToolsHelpers.get_asset_tools().create_asset(asset_name, asset_dir, struct_cls, struct_factory_cls())
    require(created is not None, f"Failed to create user-defined struct: {asset_path}")
    return created


def compile_blueprint(blueprint_asset: Any) -> None:
    unreal.BlueprintEditorLibrary.compile_blueprint(blueprint_asset)
    generated = unreal.BlueprintEditorLibrary.generated_class(blueprint_asset)
    require(generated is not None, f"Blueprint failed to compile: {blueprint_asset.get_path_name()}")


def save_blueprint_asset(asset_path: str, blueprint_asset: Any) -> None:
    compile_blueprint(blueprint_asset)
    save_asset(asset_path)


def blueprint_default_object(blueprint_asset: Any) -> Any:
    generated = unreal.BlueprintEditorLibrary.generated_class(blueprint_asset)
    require(generated is not None, f"Missing generated class: {blueprint_asset.get_path_name()}")
    if hasattr(unreal, "get_default_object"):
        cdo = unreal.get_default_object(generated)
    else:
        cdo = generated.get_default_object()
    require(cdo is not None, f"Missing CDO for {blueprint_asset.get_path_name()}")
    return cdo


def set_blueprint_defaults(blueprint_asset: Any, defaults: dict[str, Any]) -> None:
    compile_blueprint(blueprint_asset)
    cdo = blueprint_default_object(blueprint_asset)
    for key, value in defaults.items():
        try:
            cdo.set_editor_property(key, value)
        except Exception as exc:
            raise RuntimeError(f"Failed to set default '{key}' on {blueprint_asset.get_path_name()}: {exc}") from exc
    compile_blueprint(blueprint_asset)


def _variable_names(blueprint_asset: Any) -> set[str]:
    names: set[str] = set()
    try:
        variables = blueprint_asset.get_editor_property("new_variables") or []
    except Exception:
        return names
    for variable_desc in variables:
        try:
            names.add(str(variable_desc.var_name))
        except Exception:
            continue
    return names


def add_member_variable_if_missing(blueprint_asset: Any, variable_name: str, pin_type: unreal.EdGraphPinType) -> None:
    if variable_name in _variable_names(blueprint_asset):
        return
    ok = unreal.BlueprintEditorLibrary.add_member_variable(blueprint_asset, variable_name, pin_type)
    require(ok, f"Failed to add variable '{variable_name}' to {blueprint_asset.get_path_name()}")


def pin_bool() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("bool")


def pin_string() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("string")


def pin_int() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("int")


def pin_real() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("real")


def pin_class_reference(object_class: type) -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_class_reference_type(object_class)


def reset_function_graphs(blueprint_asset: Any, function_names: list[str]) -> None:
    desired = list(dict.fromkeys(function_names))
    try:
        existing_graphs = list(blueprint_asset.get_editor_property("function_graphs") or [])
    except Exception:
        existing_graphs = []
    for graph in existing_graphs:
        graph_name = str(graph.get_name())
        if graph_name not in desired:
            if hasattr(unreal.BlueprintEditorLibrary, "remove_graph"):
                unreal.BlueprintEditorLibrary.remove_graph(blueprint_asset, graph)

    for function_name in desired:
        graph = unreal.BlueprintEditorLibrary.find_graph(blueprint_asset, function_name)
        if graph is None:
            unreal.BlueprintEditorLibrary.add_function_graph(blueprint_asset, function_name)

    for legacy_name in ["NewFunction_0"]:
        if legacy_name in desired:
            continue
        graph = unreal.BlueprintEditorLibrary.find_graph(blueprint_asset, legacy_name)
        if graph is not None:
            if hasattr(unreal.BlueprintEditorLibrary, "remove_function_graph"):
                unreal.BlueprintEditorLibrary.remove_function_graph(blueprint_asset, legacy_name)
            elif hasattr(unreal.BlueprintEditorLibrary, "remove_graph"):
                unreal.BlueprintEditorLibrary.remove_graph(blueprint_asset, graph)

    compile_blueprint(blueprint_asset)


def ensure_loaded_asset_type(asset_path: str, class_name_contains: str) -> Any:
    asset = load_asset(asset_path)
    class_name = asset.get_class().get_name()
    require(
        class_name_contains in class_name,
        f"Unexpected class for {asset_path}: expected contains '{class_name_contains}', got '{class_name}'",
    )
    return asset
