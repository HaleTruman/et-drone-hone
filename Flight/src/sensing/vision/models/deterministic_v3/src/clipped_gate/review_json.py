"""Load historic deterministic-v3 review JSON and run clipped-mask discovery."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
from typing import Any, Iterable
import zlib

import numpy as np

if __package__:
    from .density_fit import DENSITY_FIT_VERSION, fit_clipped_density
    from .process import PROCESS_VERSION, process_clipped_mask
else:  # Direct ``python review_json.py ...`` execution.
    from density_fit import DENSITY_FIT_VERSION, fit_clipped_density
    from process import PROCESS_VERSION, process_clipped_mask


OUTPUT_FORMAT = "clipped-gate-review-v1"
MAX_ARRAY_BYTES = 256 * 1024 * 1024


def _decode_array(envelope: dict[str, Any]) -> np.ndarray:
    dtype = np.dtype(envelope["dtype"])
    shape = tuple(int(value) for value in envelope["shape"])
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if expected_bytes < 0 or expected_bytes > MAX_ARRAY_BYTES:
        raise ValueError(f"serialized array exceeds {MAX_ARRAY_BYTES} bytes")
    compressed = base64.b64decode(envelope["data"], validate=True)
    raw = zlib.decompress(compressed)
    if len(raw) != expected_bytes:
        raise ValueError(
            f"serialized array byte count differs: {len(raw)} != {expected_bytes}")
    order = envelope.get("order", "C")
    if order not in {"C", "F"}:
        raise ValueError(f"unsupported ndarray order: {order}")
    array = np.frombuffer(raw, dtype=dtype).copy().reshape(shape, order=order)
    strides = envelope.get("strides")
    if strides is not None and tuple(strides) != array.strides:
        raise ValueError(
            f"serialized array strides differ: {tuple(strides)} != {array.strides}")
    array.setflags(write=bool(envelope.get("writeable", False)))
    return array


def decode_runtime_value(value: Any) -> Any:
    """Decode the review serializer without importing UI or runtime schemas."""
    if isinstance(value, list):
        return [decode_runtime_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__dataclass__" in value:
        envelope = value["__dataclass__"]
        fields = {
            key: decode_runtime_value(item)
            for key, item in envelope["fields"].items()
        }
        return {"__type__": envelope["name"], **fields}
    if "__ndarray__" in value:
        return _decode_array(value["__ndarray__"])
    if "__numpy_scalar__" in value:
        envelope = value["__numpy_scalar__"]
        raw = base64.b64decode(envelope["data"], validate=True)
        return np.frombuffer(raw, dtype=np.dtype(envelope["dtype"]))[0]
    if "__tuple__" in value:
        return tuple(decode_runtime_value(item) for item in value["__tuple__"])
    if "__list__" in value:
        return [decode_runtime_value(item) for item in value["__list__"]]
    if "__dict__" in value:
        return {
            decode_runtime_value(key): decode_runtime_value(item)
            for key, item in value["__dict__"]
        }
    if "__bytes__" in value:
        return base64.b64decode(value["__bytes__"], validate=True)
    if "__float__" in value:
        return float(value["__float__"])
    return {key: decode_runtime_value(item) for key, item in value.items()}


def _records(document: dict[str, Any], record_type: str) -> list[dict[str, Any]]:
    records = []
    for encoded in document.get("schema_records", ()):
        record = decode_runtime_value(encoded)
        if isinstance(record, dict) and record.get("__type__") == record_type:
            records.append(record)
    return records


def process_review_document(
    document: dict[str, Any], *, review_json_path: str
) -> list[dict[str, Any]]:
    """Return all trusted clipped predictions in one review-frame document."""
    frames = _records(document, "FrameObservation")
    if not frames:
        return []
    if len(frames) != 1:
        raise ValueError("review document must contain exactly one FrameObservation")
    frame = frames[0]
    density_configurations = _records(document, "DensityBankConfiguration")
    if len(density_configurations) > 1:
        raise ValueError(
            "review document must contain at most one DensityBankConfiguration"
        )
    density_configuration = (
        density_configurations[0] if density_configurations else None
    )
    density_evidence = _records(document, "DensityEvidence")
    components = {
        int(component["component_id"]): component
        for component in frame.get("components", ())
    }
    source = dict(document.get("source") or {})
    source["review_json_path"] = review_json_path
    predictions = []
    for decision in _records(document, "TopologyDecision"):
        # The historic topology label is authoritative by design.  No clipping
        # reclassification or confidence test is performed here.
        if decision.get("topology_label") != "clipped":
            continue
        component_id = int(decision["component_id"])
        try:
            component = components[component_id]
        except KeyError as error:
            raise ValueError(
                f"clipped component {component_id} missing from FrameObservation"
            ) from error
        prediction = process_clipped_mask(
            source=source,
            frame=frame,
            component=component,
            decision=decision,
        )
        prediction["density_fit"] = fit_clipped_density(
            frame=frame,
            component=component,
            density_configuration=density_configuration,
            serialized_evidence=density_evidence,
        )
        predictions.append(prediction)
    return predictions


def discover_review_json(paths: Iterable[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for source in paths:
        source = source.resolve()
        if source.is_file():
            discovered.add(source)
            continue
        if not source.is_dir():
            raise FileNotFoundError(source)
        frame_dir = source / "frames"
        search_root = frame_dir if frame_dir.is_dir() else source
        discovered.update(search_root.rglob("frame-*.json"))
    return sorted(discovered)


def build_review_output(
    paths: Iterable[Path], *, limit: int | None = None, strict: bool = False
) -> dict[str, Any]:
    files = discover_review_json(paths)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    scanned = 0
    for path in files:
        if limit is not None and len(records) >= limit:
            break
        scanned += 1
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            predictions = process_review_document(
                document, review_json_path=str(path))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            if strict:
                raise
            errors.append({"path": str(path), "error": str(error)})
            continue
        if limit is not None:
            predictions = predictions[:max(0, limit - len(records))]
        records.extend(predictions)

    records.sort(key=lambda item: (
        item["source"]["run_id"], item["frame_id"], item["component_id"]))
    return {
        "format": OUTPUT_FORMAT,
        "process_version": PROCESS_VERSION,
        "density_fit_version": DENSITY_FIT_VERSION,
        "summary": {
            "files_discovered": len(files),
            "files_scanned": scanned,
            "clipped_component_predictions": len(records),
            "errors": len(errors),
        },
        "errors": errors,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read deterministic-v3 historic review JSON and emit isolated "
            "clipped-mask density and quadrilateral review predictions."
        )
    )
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="Review frame JSON, frames directory, or review run directory.")
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Destination clipped-gate-review-v1 JSON file.")
    parser.add_argument(
        "--limit", type=int,
        help="Stop after this many clipped component predictions.")
    parser.add_argument(
        "--strict", action="store_true",
        help="Stop on the first malformed review document.")
    parser.add_argument(
        "--pretty", action="store_true",
        help="Indent output JSON for manual inspection.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    output = build_review_output(
        args.paths, limit=args.limit, strict=args.strict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            output,
            indent=2 if args.pretty else None,
            separators=None if args.pretty else (",", ":"),
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    summary = output["summary"]
    print(
        f"wrote {summary['clipped_component_predictions']} clipped predictions "
        f"from {summary['files_scanned']} files to {args.output}"
    )
    if summary["errors"]:
        print(f"retained {summary['errors']} input errors", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
