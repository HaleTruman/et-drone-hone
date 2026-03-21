"""
Status: in development.
Renamed from `drone_asset_assembly.py` to `assemble_drone_assets.py` to mirror
the Course/Drone/IO/WebSocket build-layer naming pattern.

Desired behavior:
- Provide one shared, minimal Unreal Python helper module for all Drone_Content generation scripts.
- Ensure every generator can deterministically rebuild canonical `/Game/Drone_Content/...` assets.
- Keep logs, metadata, validation helpers, and explicit build-time capability reporting consistent across scripts.

Interfaces:
 - Called by every `gen_*` script in `UE_Tooling/UE_Build/Content_Generation/Drone`.
- Exposes helpers for: directory/asset lifecycle, blueprint/interface/struct creation, metadata stamping,
  and validation assertions.

Assumptions:
- Script runs inside Unreal Editor Python (headless compatible) with editor scripting enabled.
- `BlueprintEditorLibrary`, `BlueprintInterfaceFactory`, and `StructureFactory` are available in UE 5.7.
- Canonical output root is stable: `/Game/Drone_Content`.

Success conditions:
- Generators can rebuild assets from scratch without manual editor actions.
- Failures are explicit (no silent pass-through).
- Shared schema and result metadata are stamped onto generated assets for traceability.
"""

from __future__ import annotations

import json
import os
from typing import Any

import unreal

SCRIPT_NAME = "assemble_drone_assets"
SCRIPT_VERSION = "1.4.0"
SCRIPT_DATE = "2026-03-15"

SCHEMA_VERSION = "1.0"

DRONE_CONTENT_ROOT = "/Game/Drone_Content"
INTERFACES_DIR = f"{DRONE_CONTENT_ROOT}/Interfaces"
DATA_DIR = f"{DRONE_CONTENT_ROOT}/Data"
MATERIALS_DIR = f"{DRONE_CONTENT_ROOT}/Materials"
MESHES_DIR = f"{DRONE_CONTENT_ROOT}/Meshes"
BLUEPRINTS_DIR = f"{DRONE_CONTENT_ROOT}/Blueprints"
STRUCTS_DIR = f"{DRONE_CONTENT_ROOT}/Structs"

def log(message: str) -> None:
    unreal.log(f"[DroneAssetAssembly] {message}")


def log_warning(message: str) -> None:
    unreal.log_warning(f"[DroneAssetAssembly] {message}")


def log_script_metadata() -> None:
    log(f"Script: {SCRIPT_NAME} version={SCRIPT_VERSION} date={SCRIPT_DATE} schema={SCHEMA_VERSION}")


def log_result(asset_path: str, status: str, details: dict[str, Any] | None = None) -> None:
    payload = {
        "asset_path": asset_path,
        "status": status,
        "details": details or {},
    }
    log(f"[DroneContentResult] {json.dumps(payload, sort_keys=True)}")


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
    for directory in [DRONE_CONTENT_ROOT, INTERFACES_DIR, DATA_DIR, MATERIALS_DIR, MESHES_DIR, BLUEPRINTS_DIR, STRUCTS_DIR]:
        ensure_directory(directory)


def delete_asset_if_exists(asset_path: str) -> bool:
    if not unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return False
    deleted = unreal.EditorAssetLibrary.delete_asset(asset_path)
    require(deleted, f"Failed to delete existing asset: {asset_path}")
    return True


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


def load_asset(asset_path: str) -> Any:
    asset = unreal.EditorAssetLibrary.load_asset(asset_path)
    require(asset is not None, f"Failed to load asset: {asset_path}")
    return asset


def save_asset(asset_path: str) -> None:
    require(
        unreal.EditorAssetLibrary.save_asset(asset_path, only_if_is_dirty=False),
        f"Failed to save asset: {asset_path}",
    )


def save_loaded_asset(asset: Any) -> None:
    package_name = str(asset.get_path_name()).split(".", 1)[0]
    save_asset(package_name)


def set_asset_metadata(asset: Any, key: str, value: str) -> None:
    unreal.EditorAssetLibrary.set_metadata_tag(asset, key, value)


def set_asset_metadata_json(asset: Any, key: str, payload: Any) -> None:
    set_asset_metadata(asset, key, json.dumps(payload, sort_keys=True))


def stamp_common_metadata(asset: Any, generated_by: str, extra: dict[str, Any] | None = None) -> None:
    set_asset_metadata(asset, "drone.schema_version", SCHEMA_VERSION)
    set_asset_metadata(asset, "drone.generated_by", generated_by)
    set_asset_metadata(asset, "drone.generated_at_note", "Generated by UE_Tooling Drone build scripts")
    if extra:
        set_asset_metadata_json(asset, "drone.extra", extra)


def create_asset_with_factory(asset_path: str, asset_class: type, factory: Any, regenerate: bool = True) -> Any:
    ensure_standard_directories()
    asset_dir, asset_name = split_asset_path(asset_path)
    ensure_directory(asset_dir)
    if regenerate:
        delete_asset_if_exists(asset_path)
        delete_asset_file_if_exists(asset_path)
    elif unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        return load_asset(asset_path)
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    created = asset_tools.create_asset(asset_name, asset_dir, asset_class, factory)
    require(created is not None, f"Failed to create asset: {asset_path}")
    return created


def create_blueprint(asset_path: str, parent_class: type, regenerate: bool = True) -> Any:
    ensure_standard_directories()
    if regenerate:
        delete_asset_if_exists(asset_path)
        delete_asset_file_if_exists(asset_path)
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


def create_blueprint_interface(asset_path: str, regenerate: bool = True) -> Any:
    factory_cls = getattr(unreal, "BlueprintInterfaceFactory", None)
    require(factory_cls is not None, "BlueprintInterfaceFactory is unavailable in this Unreal build.")
    interface_blueprint = create_asset_with_factory(asset_path, unreal.Blueprint, factory_cls(), regenerate=regenerate)
    return interface_blueprint


def create_user_defined_struct(asset_path: str, regenerate: bool = True) -> Any:
    factory_cls = getattr(unreal, "StructureFactory", None)
    struct_cls = getattr(unreal, "UserDefinedStruct", None)
    require(factory_cls is not None, "StructureFactory is unavailable in this Unreal build.")
    require(struct_cls is not None, "UserDefinedStruct is unavailable in this Unreal build.")
    struct_asset = create_asset_with_factory(asset_path, struct_cls, factory_cls(), regenerate=regenerate)
    return struct_asset


def struct_member_authoring_capability() -> dict[str, Any]:
    utils = getattr(unreal, "StructureEditorUtils", None)
    if utils is None:
        return {
            "mode": "metadata_only",
            "available": False,
            "reason": "unreal.StructureEditorUtils is not exposed to Unreal Python in this environment.",
        }

    required_names = [
        "add_variable",
        "rename_variable",
        "change_variable_type",
        "get_var_desc",
        "on_structure_changed",
    ]
    missing = [name for name in required_names if not hasattr(utils, name)]
    if missing:
        return {
            "mode": "metadata_only",
            "available": False,
            "reason": "StructureEditorUtils binding is incomplete for Python struct-member authoring.",
            "missing_python_members": missing,
        }

    return {
        "mode": "reflected_editor_utils",
        "available": True,
        "reason": "StructureEditorUtils Python binding appears available for future deep struct authoring.",
    }


def generate_struct_asset(
    *,
    asset_path: str,
    generated_by: str,
    schema: dict[str, Any],
    regenerate: bool = True,
) -> Any:
    ensure_directory(STRUCTS_DIR)
    _, struct_name = split_asset_path(asset_path)
    capability = struct_member_authoring_capability()
    struct_asset = create_user_defined_struct(asset_path, regenerate=regenerate)
    stamp_common_metadata(
        struct_asset,
        generated_by=generated_by,
        extra={
            "asset_kind": "UserDefinedStruct",
            "struct_name": struct_name,
            "schema": schema,
            "member_authoring": capability,
        },
    )
    set_asset_metadata(asset=struct_asset, key="drone.struct_member_authoring_mode", value=str(capability["mode"]))
    set_asset_metadata_json(struct_asset, "drone.struct_member_authoring", capability)
    save_asset(asset_path)
    return struct_asset


def create_material(asset_path: str, regenerate: bool = True) -> Any:
    return create_asset_with_factory(asset_path, unreal.Material, unreal.MaterialFactoryNew(), regenerate=regenerate)


def _resolve_data_asset_parent_class() -> type:
    primary_cls = getattr(unreal, "PrimaryDataAsset", None)
    if primary_cls is not None:
        return primary_cls
    data_cls = getattr(unreal, "DataAsset", None)
    if data_cls is not None:
        return data_cls
    raise RuntimeError("No DataAsset parent class found in this Unreal build.")


def _set_data_asset_factory_class(factory: Any, data_asset_class: Any) -> None:
    set_attempts = ["data_asset_class", "asset_class", "DataAssetClass"]
    for property_name in set_attempts:
        try:
            factory.set_editor_property(property_name, data_asset_class)
            return
        except Exception:
            continue
    raise RuntimeError(
        "Unable to set DataAsset class on factory. Tried properties: "
        + ", ".join(set_attempts)
    )


def create_concrete_data_asset_class(
    class_asset_path: str,
    regenerate: bool = True,
    base_class: type | None = None,
) -> tuple[Any, Any]:
    parent_class = base_class or _resolve_data_asset_parent_class()
    class_blueprint = create_blueprint(class_asset_path, parent_class, regenerate=regenerate)
    compile_blueprint(class_blueprint)
    generated_class = blueprint_generated_class(class_blueprint)
    save_blueprint_asset(class_asset_path, class_blueprint)
    return class_blueprint, generated_class


def create_data_asset(
    asset_path: str,
    class_asset_path: str,
    regenerate_asset: bool = True,
    regenerate_class: bool = True,
) -> Any:
    factory = None
    for factory_name in ["DataAssetFactory", "DataAssetFactoryNew"]:
        factory_cls = getattr(unreal, factory_name, None)
        if factory_cls is not None:
            factory = factory_cls()
            break
    require(factory is not None, "No DataAsset factory class found in this Unreal build.")

    _, data_asset_class = create_concrete_data_asset_class(
        class_asset_path=class_asset_path,
        regenerate=regenerate_class,
    )
    _set_data_asset_factory_class(factory, data_asset_class)
    return create_asset_with_factory(asset_path, data_asset_class, factory, regenerate=regenerate_asset)


def create_material_instance(asset_path: str, regenerate: bool = True) -> Any:
    factory_cls = getattr(unreal, "MaterialInstanceConstantFactoryNew", None)
    require(factory_cls is not None, "MaterialInstanceConstantFactoryNew is unavailable in this Unreal build.")
    return create_asset_with_factory(asset_path, unreal.MaterialInstanceConstant, factory_cls(), regenerate=regenerate)


def duplicate_static_mesh_asset(source_mesh_path: str, target_mesh_path: str, regenerate: bool = True) -> Any:
    ensure_directory(split_asset_path(target_mesh_path)[0])
    if regenerate:
        delete_asset_if_exists(target_mesh_path)
        delete_asset_file_if_exists(target_mesh_path)
    source_mesh = load_asset(source_mesh_path)
    target_dir, target_name = split_asset_path(target_mesh_path)
    duplicated = unreal.AssetToolsHelpers.get_asset_tools().duplicate_asset(target_name, target_dir, source_mesh)
    require(duplicated is not None, f"Failed to duplicate static mesh from {source_mesh_path} to {target_mesh_path}")
    save_asset(target_mesh_path)
    return duplicated


def set_static_mesh_material(static_mesh_asset: Any, material_asset: Any, slot: int = 0) -> None:
    require(hasattr(static_mesh_asset, "set_material"), "Static mesh does not expose set_material in this Unreal build.")
    static_mesh_asset.set_material(int(slot), material_asset)


def compile_blueprint(blueprint_asset: Any) -> None:
    unreal.BlueprintEditorLibrary.compile_blueprint(blueprint_asset)
    generated = unreal.BlueprintEditorLibrary.generated_class(blueprint_asset)
    require(generated is not None, f"Blueprint failed to compile: {blueprint_asset.get_path_name()}")


def blueprint_generated_class(blueprint_asset: Any) -> Any:
    generated = unreal.BlueprintEditorLibrary.generated_class(blueprint_asset)
    require(generated is not None, f"Blueprint has no generated class: {blueprint_asset.get_path_name()}")
    return generated


def blueprint_default_object(blueprint_asset: Any) -> Any:
    generated = blueprint_generated_class(blueprint_asset)
    if hasattr(unreal, "get_default_object"):
        cdo = unreal.get_default_object(generated)
    else:
        cdo = generated.get_default_object()
    require(cdo is not None, f"Blueprint CDO not found: {blueprint_asset.get_path_name()}")
    return cdo


def set_blueprint_defaults(blueprint_asset: Any, defaults: dict[str, Any]) -> None:
    compile_blueprint(blueprint_asset)
    cdo = blueprint_default_object(blueprint_asset)
    for key, value in defaults.items():
        try:
            cdo.set_editor_property(key, value)
        except Exception as exc:
            raise RuntimeError(f"Failed to set default {key} on {blueprint_asset.get_path_name()}: {exc}") from exc
    compile_blueprint(blueprint_asset)


def _variable_names(blueprint_asset: Any) -> set[str]:
    variable_names: set[str] = set()
    try:
        variables = blueprint_asset.get_editor_property("new_variables") or []
    except Exception:
        return variable_names
    for variable_desc in variables:
        try:
            variable_names.add(str(variable_desc.var_name))
        except Exception:
            continue
    return variable_names


def add_member_variable_if_missing(blueprint_asset: Any, variable_name: str, pin_type: unreal.EdGraphPinType) -> None:
    if variable_name in _variable_names(blueprint_asset):
        return
    ok = unreal.BlueprintEditorLibrary.add_member_variable(blueprint_asset, variable_name, pin_type)
    require(ok, f"Failed to add variable '{variable_name}' to {blueprint_asset.get_path_name()}")


def set_variable_instance_editable(blueprint_asset: Any, variable_name: str, enabled: bool) -> None:
    unreal.BlueprintEditorLibrary.set_blueprint_variable_instance_editable(blueprint_asset, variable_name, bool(enabled))


def set_variable_expose_on_spawn(blueprint_asset: Any, variable_name: str, enabled: bool) -> None:
    unreal.BlueprintEditorLibrary.set_blueprint_variable_expose_on_spawn(blueprint_asset, variable_name, bool(enabled))


def pin_real() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("real")


def pin_bool() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("bool")


def pin_string() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("string")


def pin_name() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("name")


def pin_int() -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_basic_type_by_name("int")


def pin_object_reference(object_class: type) -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_object_reference_type(object_class)


def pin_class_reference(object_class: type) -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_class_reference_type(object_class)


def pin_array(contained_pin_type: unreal.EdGraphPinType) -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_array_type(contained_pin_type)


def pin_struct(struct_asset: Any) -> unreal.EdGraphPinType:
    return unreal.BlueprintEditorLibrary.get_struct_type(struct_asset)


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


def save_blueprint_asset(asset_path: str, blueprint_asset: Any) -> None:
    compile_blueprint(blueprint_asset)
    save_asset(asset_path)


def _subobject_data_api() -> tuple[Any, Any, Any]:
    subsystem_cls = getattr(unreal, "SubobjectDataSubsystem", None)
    params_cls = getattr(unreal, "AddNewSubobjectParams", None)
    data_lib = getattr(unreal, "SubobjectDataBlueprintFunctionLibrary", None)
    require(subsystem_cls is not None, "SubobjectDataSubsystem is unavailable in this Unreal build.")
    require(params_cls is not None, "AddNewSubobjectParams is unavailable in this Unreal build.")
    require(data_lib is not None, "SubobjectDataBlueprintFunctionLibrary is unavailable in this Unreal build.")
    require(hasattr(unreal, "get_engine_subsystem"), "get_engine_subsystem is unavailable in this Unreal build.")
    subsystem = unreal.get_engine_subsystem(subsystem_cls)
    require(subsystem is not None, "Failed to acquire SubobjectDataSubsystem engine subsystem.")
    return subsystem, params_cls, data_lib


def _set_struct_property(target: Any, candidate_names: list[str], value: Any) -> str:
    errors: list[str] = []
    for name in candidate_names:
        try:
            target.set_editor_property(name, value)
            return name
        except Exception as exc:
            errors.append(f"set_editor_property:{name}:{exc}")
        try:
            setattr(target, name, value)
            return name
        except Exception as exc:
            errors.append(f"setattr:{name}:{exc}")
    raise RuntimeError(f"Failed to set any candidate property {candidate_names}: {errors}")


def gather_blueprint_subobject_handles(blueprint_asset: Any) -> list[Any]:
    subsystem, _, _ = _subobject_data_api()
    handles = subsystem.k2_gather_subobject_data_for_blueprint(blueprint_asset)
    require(handles is not None, f"Failed to gather subobject data for {blueprint_asset.get_path_name()}")
    gathered = list(handles)
    require(gathered, f"No subobject handles returned for {blueprint_asset.get_path_name()}")
    return gathered


def _subobject_data_from_handle(data_lib: Any, handle: Any) -> Any:
    raw = data_lib.get_data(handle)
    if isinstance(raw, tuple):
        require(raw, "SubobjectDataBlueprintFunctionLibrary.get_data returned an empty tuple.")
        return raw[0]
    require(raw is not None, "SubobjectDataBlueprintFunctionLibrary.get_data returned None.")
    return raw


def _associated_subobject_object(data_lib: Any, data: Any, blueprint_asset: Any) -> Any:
    associated = None
    if hasattr(data_lib, "get_object_for_blueprint"):
        try:
            associated = data_lib.get_object_for_blueprint(data, blueprint_asset)
        except Exception:
            associated = None
    if associated is None and hasattr(data_lib, "get_associated_object"):
        try:
            associated = data_lib.get_associated_object(data)
        except Exception:
            associated = None
    return associated


def subobject_handle_summary(blueprint_asset: Any, handle: Any) -> dict[str, Any]:
    _, _, data_lib = _subobject_data_api()
    data = _subobject_data_from_handle(data_lib, handle)
    summary: dict[str, Any] = {
        "variable_name": str(data_lib.get_variable_name(data)),
        "is_component": bool(data_lib.is_component(data)),
        "is_root_component": bool(data_lib.is_root_component(data)),
    }
    if hasattr(data_lib, "get_display_name"):
        try:
            summary["display_name"] = str(data_lib.get_display_name(data))
        except Exception:
            pass
    associated = _associated_subobject_object(data_lib, data, blueprint_asset)
    if associated is not None:
        summary["object_name"] = str(associated.get_name())
        summary["object_class"] = str(associated.get_class().get_name())
        summary["object_class_path"] = str(associated.get_class().get_path_name())
    return summary


def rename_blueprint_subobject_member_variable(blueprint_asset: Any, handle: Any, new_name: str) -> None:
    subsystem_cls = getattr(unreal, "SubobjectDataSubsystem", None)
    require(subsystem_cls is not None, "SubobjectDataSubsystem is unavailable in this Unreal build.")
    rename_fn = getattr(subsystem_cls, "rename_subobject_member_variable", None)
    require(rename_fn is not None, "rename_subobject_member_variable is unavailable in this Unreal build.")
    name_value = unreal.Name(str(new_name)) if hasattr(unreal, "Name") else str(new_name)
    rename_fn(blueprint_asset, handle, name_value)


def add_blueprint_hosted_component(
    blueprint_asset: Any,
    *,
    component_blueprint_path: str,
    component_name: str,
    conform_transform_to_parent: bool = True,
) -> dict[str, Any]:
    subsystem, params_cls, data_lib = _subobject_data_api()
    component_blueprint = load_asset(component_blueprint_path)
    component_class = blueprint_generated_class(component_blueprint)

    handles_before = gather_blueprint_subobject_handles(blueprint_asset)
    context_handle = handles_before[0]

    params = params_cls()
    bindings = {
        "parent_handle": _set_struct_property(params, ["parent_handle", "ParentHandle"], context_handle),
        "new_class": _set_struct_property(params, ["new_class", "NewClass"], component_class),
        "blueprint_context": _set_struct_property(params, ["blueprint_context", "BlueprintContext"], blueprint_asset),
        "skip_mark_blueprint_modified": _set_struct_property(
            params,
            ["skip_mark_blueprint_modified", "bSkipMarkBlueprintModified"],
            False,
        ),
        "conform_transform_to_parent": _set_struct_property(
            params,
            ["conform_transform_to_parent", "bConformTransformToParent"],
            bool(conform_transform_to_parent),
        ),
    }

    add_result = subsystem.add_new_subobject(params)
    if isinstance(add_result, tuple):
        new_handle = add_result[0]
        fail_reason = add_result[1] if len(add_result) > 1 else None
    else:
        new_handle = add_result
        fail_reason = None
    require(data_lib.is_handle_valid(new_handle), f"AddNewSubobject returned invalid handle for component '{component_name}'.")
    if fail_reason:
        require(str(fail_reason).strip() == "", f"AddNewSubobject returned fail reason for '{component_name}': {fail_reason}")

    rename_blueprint_subobject_member_variable(blueprint_asset, new_handle, component_name)
    compile_blueprint(blueprint_asset)
    save_blueprint_asset(str(blueprint_asset.get_path_name()).split(".", 1)[0], blueprint_asset)
    return {
        "component_name": component_name,
        "component_blueprint_path": component_blueprint_path,
        "component_generated_class": component_class.get_name(),
        "property_bindings": bindings,
    }


def hosted_blueprint_component_summaries(blueprint_asset: Any) -> list[dict[str, Any]]:
    summaries = [subobject_handle_summary(blueprint_asset, handle) for handle in gather_blueprint_subobject_handles(blueprint_asset)]
    return [summary for summary in summaries if summary.get("is_component")]


def rebuild_blueprint_hosted_components(
    blueprint_asset: Any,
    component_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    added: list[dict[str, Any]] = []
    for spec in component_specs:
        added.append(
            add_blueprint_hosted_component(
                blueprint_asset,
                component_blueprint_path=str(spec["component_blueprint_path"]),
                component_name=str(spec["component_name"]),
                conform_transform_to_parent=bool(spec.get("conform_transform_to_parent", True)),
            )
        )
    compile_blueprint(blueprint_asset)
    save_blueprint_asset(str(blueprint_asset.get_path_name()).split(".", 1)[0], blueprint_asset)
    reloaded = load_asset(str(blueprint_asset.get_path_name()).split(".", 1)[0])
    summaries = hosted_blueprint_component_summaries(reloaded)
    return summaries


def ensure_blueprint_has_hosted_components(
    blueprint_asset: Any,
    component_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = hosted_blueprint_component_summaries(blueprint_asset)
    by_variable = {summary["variable_name"]: summary for summary in summaries}
    missing_names: list[str] = []
    mismatched_classes: list[str] = []
    for spec in component_specs:
        expected_name = str(spec["component_name"])
        expected_blueprint = load_asset(str(spec["component_blueprint_path"]))
        expected_class = blueprint_generated_class(expected_blueprint).get_name()
        summary = by_variable.get(expected_name)
        if summary is None:
            missing_names.append(expected_name)
            continue
        actual_class = str(summary.get("object_class", ""))
        if actual_class != expected_class:
            mismatched_classes.append(f"{expected_name}:{actual_class}!={expected_class}")
    require(not missing_names, f"Missing hosted components on {blueprint_asset.get_path_name()}: {missing_names}")
    require(
        not mismatched_classes,
        f"Hosted component class mismatches on {blueprint_asset.get_path_name()}: {mismatched_classes}",
    )
    return summaries


def blueprint_function_names(blueprint_asset: Any) -> list[str]:
    names: list[str] = []
    try:
        graphs = list(blueprint_asset.get_editor_property("function_graphs") or [])
    except Exception:
        graphs = []
    for graph in graphs:
        names.append(str(graph.get_name()))
    return sorted(set(names))


def ensure_blueprint_has_functions(blueprint_asset: Any, expected_names: list[str]) -> None:
    missing = [name for name in expected_names if unreal.BlueprintEditorLibrary.find_graph(blueprint_asset, name) is None]
    require(not missing, f"Missing function graphs on {blueprint_asset.get_path_name()}: {missing}")


def ensure_blueprint_has_variables(blueprint_asset: Any, expected_names: list[str]) -> None:
    existing = _variable_names(blueprint_asset)
    if not existing:
        log_warning(
            f"Variable introspection unavailable for {blueprint_asset.get_path_name()}; "
            "skipping strict variable-existence check."
        )
        return
    missing = [name for name in expected_names if name not in existing]
    require(not missing, f"Missing variables on {blueprint_asset.get_path_name()}: {missing}")


def ensure_asset_exists(asset_path: str) -> None:
    require(unreal.EditorAssetLibrary.does_asset_exist(asset_path), f"Missing expected asset: {asset_path}")


def ensure_loaded_asset_type(asset_path: str, class_name: str) -> Any:
    asset = load_asset(asset_path)
    asset_class_name = asset.get_class().get_name()
    require(
        class_name in asset_class_name,
        f"Unexpected class for {asset_path}: expected contains '{class_name}', got '{asset_class_name}'",
    )
    return asset


def ensure_loaded_data_asset(asset_path: str, expected_class_asset_path: str | None = None) -> Any:
    asset = load_asset(asset_path)
    if expected_class_asset_path:
        _, expected_class_name_base = split_asset_path(expected_class_asset_path)
        expected_generated_class_name = f"{expected_class_name_base}_C"
        actual_class_name = asset.get_class().get_name()
        require(
            actual_class_name == expected_generated_class_name,
            (
                f"Unexpected data asset class for {asset_path}: "
                f"expected {expected_generated_class_name}, got {actual_class_name}"
            ),
        )
    return asset
