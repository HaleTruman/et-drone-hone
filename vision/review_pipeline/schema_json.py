"""Lossless JSON adaptation for production dataclasses used in review only."""

from __future__ import annotations

import base64
import json
import math
import zlib
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from Vision.deterministic_v3.src_unused import schema as production_schema


RUNTIME_RECORD_ENCODING = "python-schema-runtime-record-v2"
ARRAY_ENCODING = "numpy-contiguous-zlib-base64-v2"
NUMPY_SCALAR_ENCODING = "numpy-scalar-base64-v1"


def _schema_types() -> dict[str, type]:
    return {
        name: value
        for name, value in vars(production_schema).items()
        if (isinstance(value, type) and is_dataclass(value)
            and value.__module__ == production_schema.__name__)
    }


def schema_contracts(
    module: ModuleType = production_schema,
) -> dict[str, list[str]]:
    """List every schema dataclass and its fields in declaration order."""
    return {
        name: [field.name for field in fields(value)]
        for name, value in vars(module).items()
        if (isinstance(value, type) and is_dataclass(value)
            and value.__module__ == module.__name__)
    }


def json_value(value: Any) -> Any:
    """Convert ordinary review metadata; schema values use runtime envelopes."""
    if is_dataclass(value) and not isinstance(value, type):
        return runtime_value(value)
    if isinstance(value, np.ndarray):
        return runtime_value(value)
    if isinstance(value, np.generic):
        return runtime_value(value)
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return runtime_value(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported review JSON value: {type(value).__name__}")


def runtime_value(value: Any) -> Any:
    """Encode one inference value with enough information for exact recovery."""
    if is_dataclass(value) and not isinstance(value, type):
        value_type = type(value)
        if value_type.__module__ != production_schema.__name__:
            raise TypeError(
                f"not a schema.py dataclass: {value_type.__module__}."
                f"{value_type.__name__}")
        return {
            "__dataclass__": {
                "module": value_type.__module__,
                "name": value_type.__name__,
                "fields": {
                    field.name: runtime_value(getattr(value, field.name))
                    for field in fields(value)
                },
            }
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays cannot be retained without pickle")
        if value.flags.c_contiguous:
            order = "C"
        elif value.flags.f_contiguous:
            order = "F"
        else:
            raise TypeError("non-contiguous schema arrays require an explicit copy")
        compressed = zlib.compress(value.tobytes(order=order))
        return {
            "__ndarray__": {
                "encoding": ARRAY_ENCODING,
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "strides": list(value.strides),
                "order": order,
                "writeable": bool(value.flags.writeable),
                "data": base64.b64encode(compressed).decode("ascii"),
            }
        }
    if isinstance(value, np.generic):
        scalar = np.asarray(value)
        return {
            "__numpy_scalar__": {
                "encoding": NUMPY_SCALAR_ENCODING,
                "dtype": scalar.dtype.str,
                "data": base64.b64encode(scalar.tobytes()).decode("ascii"),
            }
        }
    if isinstance(value, tuple):
        return {"__tuple__": [runtime_value(item) for item in value]}
    if isinstance(value, list):
        return {"__list__": [runtime_value(item) for item in value]}
    if isinstance(value, dict):
        return {"__dict__": [
            [runtime_value(key), runtime_value(item)]
            for key, item in value.items()
        ]}
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(
        f"unsupported inference value; evidence was not written: "
        f"{type(value).__name__}")


def ndarray_value(value: dict[str, Any]) -> np.ndarray:
    """Rebuild an ndarray produced by :func:`runtime_value`."""
    envelope = value["__ndarray__"]
    if envelope["encoding"] not in {
            ARRAY_ENCODING, "numpy-c-order-zlib-base64-v1"}:
        raise ValueError(f"unsupported array encoding: {envelope['encoding']}")
    raw = zlib.decompress(base64.b64decode(envelope["data"]))
    array = np.frombuffer(raw, dtype=np.dtype(envelope["dtype"])).copy()
    order = envelope.get("order", "C")
    array = array.reshape(tuple(envelope["shape"]), order=order)
    if "strides" in envelope and list(array.strides) != envelope["strides"]:
        raise ValueError(
            f"array strides cannot be reconstructed exactly: "
            f"{array.strides} != {tuple(envelope['strides'])}")
    array.setflags(write=bool(envelope["writeable"]))
    return array


def runtime_object(value: Any) -> Any:
    """Reconstruct the exact Python value represented by a runtime envelope."""
    if not isinstance(value, dict):
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError(f"invalid untagged runtime value: {type(value).__name__}")
    if set(value) == {"__dataclass__"}:
        envelope = value["__dataclass__"]
        if envelope["module"] != production_schema.__name__:
            raise ValueError(f"unexpected schema module: {envelope['module']}")
        try:
            contract = _schema_types()[envelope["name"]]
        except KeyError as error:
            raise ValueError(
                f"unknown schema dataclass: {envelope['name']}") from error
        names = [field.name for field in fields(contract)]
        if list(envelope["fields"]) != names:
            raise ValueError(
                f"{contract.__name__} field mismatch: "
                f"{list(envelope['fields'])} != {names}")
        return contract(**{
            name: runtime_object(envelope["fields"][name]) for name in names
        })
    if set(value) == {"__ndarray__"}:
        return ndarray_value(value)
    if set(value) == {"__numpy_scalar__"}:
        envelope = value["__numpy_scalar__"]
        if envelope["encoding"] != NUMPY_SCALAR_ENCODING:
            raise ValueError(
                f"unsupported scalar encoding: {envelope['encoding']}")
        raw = base64.b64decode(envelope["data"])
        return np.frombuffer(raw, dtype=np.dtype(envelope["dtype"]))[0]
    if set(value) == {"__tuple__"}:
        return tuple(runtime_object(item) for item in value["__tuple__"])
    if set(value) == {"__list__"}:
        return [runtime_object(item) for item in value["__list__"]]
    if set(value) == {"__dict__"}:
        return {
            runtime_object(key): runtime_object(item)
            for key, item in value["__dict__"]
        }
    if set(value) == {"__bytes__"}:
        return base64.b64decode(value["__bytes__"])
    if set(value) == {"__float__"}:
        return float(value["__float__"])
    raise ValueError(f"unknown runtime value envelope: {tuple(value)}")


def assert_runtime_equal(expected: Any, actual: Any, path: str = "record") -> None:
    """Prove type, structure, metadata, and values survived JSON round-trip."""
    if type(expected) is not type(actual):
        raise AssertionError(
            f"{path}: type changed from {type(expected)} to {type(actual)}")
    if is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            assert_runtime_equal(
                getattr(expected, field.name), getattr(actual, field.name),
                f"{path}.{field.name}")
        return
    if isinstance(expected, np.ndarray):
        attributes = ("dtype", "shape", "strides")
        for attribute in attributes:
            if getattr(expected, attribute) != getattr(actual, attribute):
                raise AssertionError(f"{path}: array {attribute} changed")
        if expected.flags.writeable != actual.flags.writeable:
            raise AssertionError(f"{path}: array writeability changed")
        order = "C" if expected.flags.c_contiguous else "F"
        if expected.tobytes(order=order) != actual.tobytes(order=order):
            raise AssertionError(f"{path}: array bytes changed")
        return
    if isinstance(expected, np.generic):
        if expected.dtype != actual.dtype or expected.tobytes() != actual.tobytes():
            raise AssertionError(f"{path}: NumPy scalar changed")
        return
    if isinstance(expected, (tuple, list)):
        if len(expected) != len(actual):
            raise AssertionError(f"{path}: sequence length changed")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_runtime_equal(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, dict):
        if tuple(expected) != tuple(actual):
            raise AssertionError(f"{path}: dictionary keys or order changed")
        for key in expected:
            assert_runtime_equal(expected[key], actual[key], f"{path}[{key!r}]")
        return
    if isinstance(expected, float) and math.isnan(expected):
        if not math.isnan(actual):
            raise AssertionError(f"{path}: NaN changed")
        return
    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def write_json(path: str | Path, value: Any) -> Path:
    """Atomically write one deterministic, compact review JSON document."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            json_value(value), stream, ensure_ascii=True, allow_nan=False,
            separators=(",", ":"),
        )
        stream.write("\n")
    temporary.replace(destination)
    return destination
