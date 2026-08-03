import json
import re
import struct
from pathlib import Path


ROOT = Path.cwd()
ASSET = ROOT / "Content" / "__ExternalActors__" / "Hangar" / "9" / "HV" / "QBXWTM9RBYMCDVQYYY0XAD.uasset"
OUTPUT = Path(__file__).resolve().parent / "hangar_columns_pylons.jsonl"

NAME_MAP_OFFSET = 577
SERIALIZED_COMPONENT_SIZE = 429
STATIC_MESH_COMPONENT_CLASS_INDEX = 8

PROP_RELATIVE_LOCATION = 26
PROP_RELATIVE_ROTATION = 27
PROP_RELATIVE_SCALE3D = 28


def parse_name_map(data):
    names = []
    offset = NAME_MAP_OFFSET
    while True:
        length = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        if not (0 < length < 500):
            break
        names.append(data[offset : offset + length - 1].decode("ascii", "replace"))
        offset += length + 4
    return names


def fname_display(base_name, number):
    return base_name if number == 0 else f"{base_name}_{number - 1}"


def vec_prop(data, serial_offset, prop_idx, default=None):
    pos = data.find(struct.pack("<ii", prop_idx, 0), serial_offset, serial_offset + SERIALIZED_COMPONENT_SIZE)
    if pos == -1:
        return default
    return struct.unpack_from("<ddd", data, pos + 49)


def xyz_m(values):
    return {"x": values[0] / 100.0, "y": values[1] / 100.0, "z": values[2] / 100.0}


def main():
    data = ASSET.read_bytes()
    names = parse_name_map(data)

    wanted = [
        (idx, name)
        for idx, name in enumerate(names)
        if name.startswith("Outer_Structure_Column")
        or re.match(r"^Station_\d+_Tall_Dark_Pylon$", name)
    ]

    rows = []
    seen = set()
    for idx, base_name in wanted:
        needle = struct.pack("<i", idx)
        for match in re.finditer(re.escape(needle), data):
            export_record_offset = match.start()
            try:
                fname_number = struct.unpack_from("<i", data, export_record_offset + 4)[0]
                class_index = struct.unpack_from("<i", data, export_record_offset + 8)[0]
                serial_size = struct.unpack_from("<i", data, export_record_offset + 12)[0]
                serial_offset = struct.unpack_from("<i", data, export_record_offset + 20)[0]
            except struct.error:
                continue

            if class_index != STATIC_MESH_COMPONENT_CLASS_INDEX:
                continue
            if serial_size != SERIALIZED_COMPONENT_SIZE:
                continue
            if not (90000 < serial_offset < len(data)):
                continue

            component_name = fname_display(base_name, fname_number)
            if not (
                component_name.startswith("Outer_Structure_Column")
                or re.match(r"^Station_\d+_Tall_Dark_Pylon$", component_name)
            ):
                continue

            key = (idx, fname_number, serial_offset)
            if key in seen:
                continue
            seen.add(key)

            loc = vec_prop(data, serial_offset, PROP_RELATIVE_LOCATION)
            rot = vec_prop(data, serial_offset, PROP_RELATIVE_ROTATION, (0.0, 0.0, 0.0))
            scale = vec_prop(data, serial_offset, PROP_RELATIVE_SCALE3D)
            if loc is None or scale is None:
                raise RuntimeError(f"Missing transform for {component_name} at serial offset {serial_offset}")

            size_m = tuple(abs(value) for value in scale)
            rows.append(
                {
                    "component_name": component_name,
                    "name_map_entry": base_name,
                    "fname_number": fname_number,
                    "kind": "column" if component_name.startswith("Outer_Structure_Column") else "pylon",
                    "actor_label": "AI_Hangar_Enclosed_Interior_150x300m",
                    "actor_asset": "Content/__ExternalActors__/Hangar/9/HV/QBXWTM9RBYMCDVQYYY0XAD.uasset",
                    "world_location_m": xyz_m(loc),
                    "world_rotation_deg": {"pitch": rot[0], "yaw": rot[1], "roll": rot[2]},
                    "relative_scale3d": {"x": scale[0], "y": scale[1], "z": scale[2]},
                    "size_m": {"x": size_m[0], "y": size_m[1], "z": size_m[2]},
                    "size_source": "RelativeScale3D multiplied by 1 m Unreal basic-shape mesh bounds",
                    "export_record_offset": export_record_offset,
                    "serial_offset": serial_offset,
                }
            )

    rows.sort(
        key=lambda row: (
            row["kind"],
            row["world_location_m"]["x"],
            row["world_location_m"]["y"],
            row["component_name"],
        )
    )

    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    columns = sum(1 for row in rows if row["kind"] == "column")
    pylons = sum(1 for row in rows if row["kind"] == "pylon")
    print(f"wrote {len(rows)} rows: {columns} columns, {pylons} pylons")


if __name__ == "__main__":
    main()
