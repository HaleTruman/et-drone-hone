import json
import re
from pathlib import Path

import unreal


OUTPUT = Path(__file__).resolve().parent / "hangar_columns_pylons.jsonl"
TARGET_ACTOR_LABEL = "AI_Hangar_Enclosed_Interior_150x300m"
TARGET_EXTERNAL_ACTOR = "/Game/__ExternalActors__/Hangar/9/HV/QBXWTM9RBYMCDVQYYY0XAD"
NAME_RE = re.compile(r"^(Outer_Structure_Column_\S+|Station_\d+_Tall_Dark_Pylon)$")


def vec(value):
    return {"x": value.x, "y": value.y, "z": value.z}


def rot(value):
    return {"pitch": value.pitch, "yaw": value.yaw, "roll": value.roll}


def component_name(component):
    return str(component.get_name())


def actor_label(actor):
    try:
        return str(actor.get_actor_label())
    except Exception:
        return str(actor.get_name())


def bounds_sizes(component):
    local_min = local_max = None
    local_size = scaled_local_size = None
    try:
        local_min, local_max = component.get_local_bounds()
        local_size = unreal.Vector(
            local_max.x - local_min.x,
            local_max.y - local_min.y,
            local_max.z - local_min.z,
        )
        scale = component.get_component_scale()
        scaled_local_size = unreal.Vector(
            abs(local_size.x * scale.x),
            abs(local_size.y * scale.y),
            abs(local_size.z * scale.z),
        )
    except Exception:
        pass

    bounds_origin = bounds_extent = bounds_size = None
    try:
        bounds = component.bounds
        bounds_origin = bounds.origin
        bounds_extent = bounds.box_extent
        bounds_size = unreal.Vector(
            bounds_extent.x * 2.0,
            bounds_extent.y * 2.0,
            bounds_extent.z * 2.0,
        )
    except Exception:
        pass

    return local_min, local_max, local_size, scaled_local_size, bounds_origin, bounds_extent, bounds_size


def find_target_actor():
    external_asset = unreal.load_asset(TARGET_EXTERNAL_ACTOR)
    unreal.log(f"External actor asset load result: {external_asset!r}")
    if isinstance(external_asset, unreal.Actor):
        return external_asset

    unreal.EditorLevelLibrary.load_level("/Game/Hangar")
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        if actor_label(actor) == TARGET_ACTOR_LABEL or actor.get_name() == TARGET_ACTOR_LABEL:
            return actor
    return None


def main():
    target_actor = find_target_actor()
    if target_actor is None:
        raise RuntimeError(f"Could not find actor {TARGET_ACTOR_LABEL!r}")

    rows = []
    for component in target_actor.get_components_by_class(unreal.StaticMeshComponent):
        name = component_name(component)
        if not NAME_RE.match(name):
            continue

        (
            local_min,
            local_max,
            local_size,
            scaled_local_size,
            bounds_origin,
            bounds_extent,
            bounds_size,
        ) = bounds_sizes(component)

        rows.append(
            {
                "component_name": name,
                "kind": "column" if name.startswith("Outer_Structure_Column_") else "pylon",
                "actor_label": TARGET_ACTOR_LABEL,
                "world_location_cm": vec(component.get_component_location()),
                "world_rotation_deg": rot(component.get_component_rotation()),
                "world_scale": vec(component.get_component_scale()),
                "local_bounds_min_cm": vec(local_min) if local_min else None,
                "local_bounds_max_cm": vec(local_max) if local_max else None,
                "local_size_cm": vec(local_size) if local_size else None,
                "scaled_local_size_cm": vec(scaled_local_size) if scaled_local_size else None,
                "world_bounds_origin_cm": vec(bounds_origin) if bounds_origin else None,
                "world_bounds_extent_cm": vec(bounds_extent) if bounds_extent else None,
                "world_bounds_size_cm": vec(bounds_size) if bounds_size else None,
            }
        )

    rows.sort(key=lambda row: (row["kind"], row["component_name"]))
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    unreal.log(f"Wrote {len(rows)} rows to {OUTPUT}")


main()
