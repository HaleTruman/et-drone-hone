"""Small shared helpers for Unreal automation."""

import unreal

def distance(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5

def clamp(value, bounds):
    return max(bounds[0], min(bounds[1], value))

def lerp(a, b, alpha):
    return a + (b - a) * alpha

def vector_to_dict(vector):
    return {
        "x": float(vector.x),
        "y": float(vector.y),
        "z": float(vector.z),
    }

def rotator_to_dict(rotator):
    return {
        "pitch": float(rotator.pitch),
        "yaw": float(rotator.yaw),
        "roll": float(rotator.roll),
    }

def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z

def object_path_name(obj):
    if not obj:
        return ""

    get_path_name = getattr(obj, "get_path_name", None)
    if get_path_name:
        try:
            return str(get_path_name())
        except Exception:
            pass

    return str(obj)

def actor_label(actor):
    if not actor:
        return None
    try:
        return actor.get_actor_label()
    except Exception:
        return str(actor)

def component_owner(component):
    get_owner = getattr(component, "get_owner", None)
    if get_owner:
        try:
            return get_owner()
        except Exception:
            pass
    return None

def component_name(component):
    for getter_name in ("get_name", "get_fname"):
        getter = getattr(component, getter_name, None)
        if getter:
            try:
                return str(getter())
            except Exception:
                pass
    for property_name in ("component_name", "name"):
        try:
            value = component.get_editor_property(property_name)
            if value:
                return str(value)
        except Exception:
            pass
    return ""
