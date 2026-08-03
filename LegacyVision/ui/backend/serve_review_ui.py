"""Serve the static geometry UI with logged frames and review JSON catalogs."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from functools import lru_cache, partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import shutil
import struct
from urllib.parse import quote, unquote, urlsplit
import zlib

from .pnp_world_replay import build_pnp_world_replay


UI_ROOT = Path(__file__).resolve().parents[1]
LEGACYVISION_ROOT = UI_ROOT.parent
PROJECT_ROOT = LEGACYVISION_ROOT.parent
FRONTEND_ROOT = UI_ROOT / "frontend"
LOGS_ROOT = PROJECT_ROOT / "Logs" / "flight" / "runs"
REVIEW_ROOT = PROJECT_ROOT / "Logs" / "review" / "runs"
EVALUATION_ROOT = PROJECT_ROOT / "Logs" / "evaluation" / "runs"
IMAGE_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png"))
RUN_PATTERN = re.compile(r"^run-[A-Za-z0-9_.-]+$")
EVALUATION_PATTERN = re.compile(r"^evaluation-[A-Za-z0-9_.-]+$")
BACKEND_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
OVERLAY_LAYER_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
FRAME_NUMBER_PATTERN = re.compile(r"frame-(\d+)")
FRAME_IDENTITY_PATTERN = re.compile(r"^frame-(\d+)-(\d+)$")
FRAME_MASK_FIELDS = (
    "base_mask",
    "size_filtered_mask",
    "closed_mask",
)
TOUCHES_FRAME_LAYER_ID = "ComponentObservation.touches_frame"
C_SHAPE_MASK_LAYER_ID = "CShapeResult.refined_mask"
DENSITY_ARRAY_FIELDS = ("final_field", "p70_mask", "p80_mask", "p90_mask")
STANDARD_GATE_COMPONENT_LAYER_FIELDS = (
    "FrameObservation.closed_mask",
)


def _density_layer_id(profile_id, field_name):
    return f"DensityEvidence[{profile_id}].{field_name}"


def _is_supported_layer_id(layer_id):
    if layer_id in {
            *(f"FrameObservation.{field_name}"
              for field_name in FRAME_MASK_FIELDS),
            TOUCHES_FRAME_LAYER_ID,
            C_SHAPE_MASK_LAYER_ID}:
        return True
    match = re.fullmatch(
        r"DensityEvidence\[([^]]+)\]\.([A-Za-z0-9_]+)", layer_id)
    return match is not None and match.group(2) in DENSITY_ARRAY_FIELDS


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    path: Path
    run_id: str | None
    frame_keys: frozenset[str]


@dataclass(frozen=True, slots=True)
class GeometryReviewRecord:
    """One exact GeometryFrameResult entry in a run-level JSON document."""

    path: Path
    run_id: str
    frame_id: int
    sim_time_ns: int
    frame_index: int
    pnp_relative_pose_estimate_count: int
    accepted_pnp_relative_pose_estimate_count: int
    secondary_pnp_candidate_count: int
    camera_pose_estimate_count: int
    accepted_camera_pose_estimate_count: int


def _frame_keys(*values):
    keys = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        keys.add(text)
        keys.add(Path(text).stem)
        match = FRAME_NUMBER_PATTERN.search(text)
        if match:
            keys.add(match.group(1))
            keys.add(str(int(match.group(1))))
        if text.isdigit():
            keys.add(str(int(text)))
    return frozenset(keys)


def discover_review_records(review_root=REVIEW_ROOT):
    records = []
    if not review_root.is_dir():
        return records
    for run_path in sorted(review_root.glob("run-*")):
        if not run_path.is_dir() or not RUN_PATTERN.fullmatch(run_path.name):
            continue
        for path in sorted((run_path / "frames").glob("*.json")):
            frame_values = [path.name]
            if FRAME_NUMBER_PATTERN.search(path.name) is None:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    payload = {}
                source = (
                    payload.get("source", {}) if isinstance(payload, dict) else {})
                frame_values.extend((
                    source.get("frame_id"), source.get("sim_time_ns"),
                    payload.get("frame_id"), payload.get("sim_time_ns")))
            records.append(ReviewRecord(
                path, run_path.name, _frame_keys(path.name, *frame_values)))
    return records


@lru_cache(maxsize=8)
def _read_geometry_document(path_text, modified_ns, size):
    del modified_ns, size
    return json.loads(Path(path_text).read_text(encoding="utf-8"))


def _geometry_document(path):
    statistics = path.stat()
    return _read_geometry_document(
        str(path.resolve()), statistics.st_mtime_ns, statistics.st_size)


def discover_geometry_review_records(review_root=REVIEW_ROOT):
    """Discover exact run-level GeometryFrameResult JSON without reshaping it."""
    records = []
    if not review_root.is_dir():
        return records
    for run_path in sorted(review_root.glob("run-*")):
        if not run_path.is_dir() or not RUN_PATTERN.fullmatch(run_path.name):
            continue
        for path in sorted(run_path.glob("*.json")):
            try:
                payload = _geometry_document(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            frames = payload.get("frames") if isinstance(payload, dict) else None
            summary = payload.get("summary") if isinstance(payload, dict) else None
            if (not isinstance(summary, dict) or
                    summary.get("format_version") != 2 or
                    not isinstance(frames, list)):
                continue
            for index, item in enumerate(frames):
                runtime_result = (
                    item.get("runtime_result") if isinstance(item, dict) else None)
                if (item.get("run_id") != run_path.name or
                        not isinstance(runtime_result, dict) or
                        not all(field in runtime_result for field in (
                            "frame_id", "sim_time_ns", "camera_calibration",
                            "gate_model", "pnp_relative_pose_estimates",
                            "camera_pose_estimates")) or
                        not isinstance(
                            runtime_result["pnp_relative_pose_estimates"], list) or
                        not isinstance(
                            runtime_result["camera_pose_estimates"], list)):
                    continue
                try:
                    frame_id = int(runtime_result["frame_id"])
                    sim_time_ns = int(runtime_result["sim_time_ns"])
                except (TypeError, ValueError):
                    continue
                pnp_estimates = runtime_result["pnp_relative_pose_estimates"]
                final_poses = runtime_result["camera_pose_estimates"]
                records.append(GeometryReviewRecord(
                    path=path,
                    run_id=run_path.name,
                    frame_id=frame_id,
                    sim_time_ns=sim_time_ns,
                    frame_index=index,
                    pnp_relative_pose_estimate_count=len(pnp_estimates),
                    accepted_pnp_relative_pose_estimate_count=sum(
                        pose.get("accepted") is True
                        for pose in pnp_estimates if isinstance(pose, dict)),
                    secondary_pnp_candidate_count=sum(
                        max(0, len(pose.get("candidates", ())) - 1)
                        for pose in pnp_estimates if isinstance(pose, dict) and
                        isinstance(pose.get("candidates"), list)),
                    camera_pose_estimate_count=len(final_poses),
                    accepted_camera_pose_estimate_count=sum(
                        pose.get("accepted") is True
                        for pose in final_poses if isinstance(pose, dict)),
                ))
    return records


def _geometry_matches_frame(frame_path, records):
    identity = FRAME_IDENTITY_PATTERN.fullmatch(frame_path.stem)
    if identity is None:
        return []
    frame_id, sim_time_ns = map(int, identity.groups())
    return [record for record in records
            if record.frame_id == frame_id and
            record.sim_time_ns == sim_time_ns]


def _geometry_review_item(record):
    payload = _geometry_document(record.path)
    return payload["frames"][record.frame_index]


def discover_logged_runs(logs_root=LOGS_ROOT):
    roots = [logs_root]
    nested = logs_root / "runs"
    if nested.is_dir():
        roots.insert(0, nested)
    runs = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("run-*")):
            if path.is_dir() and (path / "vision_frames").is_dir():
                runs[path.name] = path
    return runs


def _logged_frames(run_path):
    return sorted(
        path for path in (run_path / "vision_frames").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _density_bank_configuration(review_path):
    if review_path is None:
        return {"profiles": []}
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    records = payload.get("schema_records", ())
    configuration = next((
        record.get("__dataclass__", {}).get("fields", {})
        for record in records
        if record.get("__dataclass__", {}).get("name") ==
        "DensityBankConfiguration"
    ), None)
    if configuration is not None:
        profile_values = configuration.get("profiles", {})
        profile_values = profile_values.get("__tuple__", profile_values)
    else:
        profile_values = [
            record.get("__dataclass__", {}).get("fields", {}).get("profile", {})
            for record in records
            if record.get("__dataclass__", {}).get("name") == "DensityEvidence"
        ]

    profiles = []
    seen = set()
    for value in profile_values:
        fields = value.get("__dataclass__", {}).get("fields", value)
        profile_id = fields.get("profile_id")
        if not profile_id or profile_id in seen:
            continue
        seen.add(profile_id)
        profiles.append({
            key: field_value for key, field_value in fields.items()
            if field_value is None or isinstance(
                field_value, (str, int, float, bool))
        })

    result = {"profiles": profiles}
    if configuration is not None:
        result.update({
            key: value for key, value in configuration.items()
            if key != "profiles" and (
                value is None or isinstance(value, (str, int, float, bool)))
        })
    return result


def _review_layer_urls(run_id, review_path, profile_ids=()):
    if review_path is None:
        return {}
    base = "/schema-layers/{}/{}".format(
        quote(run_id), quote(review_path.stem))
    return {
        layer_id: f"{base}/{quote(layer_id)}.png"
        for layer_id in (
            *(f"FrameObservation.{field_name}"
              for field_name in FRAME_MASK_FIELDS),
            TOUCHES_FRAME_LAYER_ID,
            C_SHAPE_MASK_LAYER_ID,
            *(_density_layer_id(profile_id, field_name)
              for profile_id in profile_ids
              for field_name in DENSITY_ARRAY_FIELDS),
        )
    }


def _review_url(review_path, review_root):
    if review_path is None:
        return None
    return "/reviews/" + quote(
        review_path.relative_to(review_root).as_posix(), safe="/")


def _geometry_review_url(run_id, frame_path, matches):
    if len(matches) != 1:
        return None
    return "/api/runs/{}/geometry-frames/{}".format(
        quote(run_id), quote(frame_path.name))


def run_catalog(logs_root=LOGS_ROOT, review_root=REVIEW_ROOT):
    logged = discover_logged_runs(logs_root)
    reviews = discover_review_records(review_root)
    geometry_reviews = discover_geometry_review_records(review_root)
    reviews_by_run = {}
    for record in reviews:
        reviews_by_run.setdefault(record.run_id, []).append(record)
    geometry_by_run = {}
    for record in geometry_reviews:
        geometry_by_run.setdefault(record.run_id, []).append(record)
    result = []
    for run_id in sorted(set(logged) | {
            value for value in reviews_by_run if value is not None} |
            set(geometry_by_run)):
        run_path = logged.get(run_id)
        frame_count = len(_logged_frames(run_path)) if run_path else 0
        run_reviews = reviews_by_run.get(run_id, ())
        run_geometry = geometry_by_run.get(run_id, ())
        result.append({
            "id": run_id,
            "name": run_id,
            "frame_count": frame_count,
            "review_json_count": len(run_reviews),
            "geometry_review_json_count": len({
                record.path for record in run_geometry}),
            "geometry_frame_result_count": len(run_geometry),
            "pnp_relative_pose_estimate_count": sum(
                record.pnp_relative_pose_estimate_count
                for record in run_geometry),
            "accepted_pnp_relative_pose_estimate_count": sum(
                record.accepted_pnp_relative_pose_estimate_count
                for record in run_geometry),
            "secondary_pnp_candidate_count": sum(
                record.secondary_pnp_candidate_count
                for record in run_geometry),
            "camera_pose_estimate_count": sum(
                record.camera_pose_estimate_count
                for record in run_geometry),
            "accepted_camera_pose_estimate_count": sum(
                record.accepted_camera_pose_estimate_count
                for record in run_geometry),
            "has_logged_frames": run_path is not None,
            "initial_review_url": _review_url(
                run_reviews[0].path if run_reviews else None, review_root),
        })
    return {
        "version": "deterministic-v3.review-catalog.v2",
        "runs": result,
        "unassociated_review_json": len(reviews_by_run.get(None, ())),
    }


def frame_catalog(run_id, logs_root=LOGS_ROOT, review_root=REVIEW_ROOT):
    run_path = discover_logged_runs(logs_root).get(run_id)
    if run_path is None:
        raise FileNotFoundError(run_id)
    reviews = [record for record in discover_review_records(review_root)
               if record.run_id == run_id]
    geometry_reviews = [
        record for record in discover_geometry_review_records(review_root)
        if record.run_id == run_id]
    density_configuration = _density_bank_configuration(
        reviews[0].path if reviews else None)
    density_profile_ids = tuple(
        profile["profile_id"]
        for profile in density_configuration["profiles"])
    frames = []
    matched_reviews = set()
    matched_geometry = set()
    for path in _logged_frames(run_path):
        keys = _frame_keys(path.name)
        matches = [record for record in reviews if keys & record.frame_keys]
        matched_reviews.update(record.path for record in matches)
        geometry_matches = _geometry_matches_frame(path, geometry_reviews)
        matched_geometry.update(geometry_matches)
        image_url = f"/frames/{quote(run_id)}/{quote(path.name)}"
        frames.append({
            "id": path.stem,
            "filename": path.name,
            "image_url": image_url,
            "review_json_count": len(matches),
            "review_urls": [
                _review_url(record.path, review_root)
                for record in matches
            ],
            "geometry_frame_result_count": len(geometry_matches),
            "geometry_result_url": _geometry_review_url(
                run_id, path, geometry_matches),
            "layers": {
                "source.relative_path": image_url,
                **_review_layer_urls(
                    run_id, matches[0].path if matches else None,
                    density_profile_ids),
            },
        })
    return {
        "version": "deterministic-v3.frame-catalog.v2",
        "id": run_id,
        "frames": frames,
        "review_json_count": len(reviews),
        "geometry_frame_result_count": len(geometry_reviews),
        "density_profile_ids": density_profile_ids,
        "review_contract": {
            "DensityBankConfiguration": density_configuration,
        },
        "unmatched_review_json": len(
            [record for record in reviews if record.path not in matched_reviews]),
        "unmatched_geometry_frame_result_count": len(
            [record for record in geometry_reviews
             if record not in matched_geometry]),
    }


def geometry_frame_result(
    run_id, filename, logs_root=LOGS_ROOT, review_root=REVIEW_ROOT
):
    """Return the exact run-level JSON item matching one logged source frame."""
    run_path = discover_logged_runs(logs_root).get(run_id)
    if run_path is None or Path(filename).name != filename:
        raise FileNotFoundError(filename)
    frame_path = _confined(run_path / "vision_frames", filename)
    if not frame_path.is_file():
        raise FileNotFoundError(filename)
    records = [record for record in discover_geometry_review_records(review_root)
               if record.run_id == run_id]
    matches = _geometry_matches_frame(frame_path, records)
    if not matches:
        raise FileNotFoundError(filename)
    if len(matches) != 1:
        raise ValueError("source frame has multiple GeometryFrameResult records")
    return _geometry_review_item(matches[0])


def geometry_frame_history(
    run_id, logs_root=LOGS_ROOT, review_root=REVIEW_ROOT
):
    """Group exact prior-frame GeometryFrameResult JSON for UI history."""
    if run_id not in discover_logged_runs(logs_root):
        raise FileNotFoundError(run_id)
    records = [
        record for record in discover_geometry_review_records(review_root)
        if record.run_id == run_id
    ]
    identities = {}
    for record in records:
        identity = (record.frame_id, record.sim_time_ns)
        if identity in identities:
            raise ValueError("duplicate GeometryFrameResult frame identity")
        identities[identity] = record
    ordered = sorted(
        identities.values(), key=lambda record: (
            record.sim_time_ns, record.frame_id))
    return {
        "version": "deterministic-v3.geometry-frame-history.v2",
        "id": run_id,
        "frames": [_geometry_review_item(record) for record in ordered],
    }


def pnp_world_replay(
    run_id, logs_root=LOGS_ROOT, review_root=REVIEW_ROOT
):
    """Join exact run-level PnP JSON to logged vehicle states in LOCAL_NED."""
    run_path = discover_logged_runs(logs_root).get(run_id)
    if run_path is None:
        raise FileNotFoundError(run_id)
    records = [
        record for record in discover_geometry_review_records(review_root)
        if record.run_id == run_id
    ]
    if not records:
        raise FileNotFoundError(f"GeometryFrameResult unavailable: {run_id}")
    return build_pnp_world_replay(
        run_id=run_id,
        run_dir=run_path,
        geometry_items=[_geometry_review_item(record) for record in records],
    )


def topology_component_catalog(
    run_id, logs_root=LOGS_ROOT, review_root=REVIEW_ROOT
):
    """Return JSON-authored topology labels and source-space component boxes."""
    run_path = discover_logged_runs(logs_root).get(run_id)
    if run_path is None:
        raise FileNotFoundError(run_id)
    reviews = [record for record in discover_review_records(review_root)
               if record.run_id == run_id]
    entries = []
    decision_count = 0
    missing_label_count = 0
    missing_bbox_count = 0
    matched_reviews = set()
    for frame_path in _logged_frames(run_path):
        keys = _frame_keys(frame_path.name)
        matches = [record for record in reviews if keys & record.frame_keys]
        if not matches:
            continue
        review = matches[0]
        matched_reviews.add(review.path)
        payload = json.loads(review.path.read_text(encoding="utf-8"))
        records = payload.get("schema_records", ())
        frame = next((
            record.get("__dataclass__", {}).get("fields", {})
            for record in records
            if record.get("__dataclass__", {}).get("name") ==
            "FrameObservation"
        ), None)
        if frame is None:
            continue
        components = {
            fields.get("component_id"): fields
            for fields in (
                _dataclass_fields(value, "ComponentObservation")
                if "__dataclass__" in value else value
                for value in _tuple_items(frame.get("components", ())))
        }
        source = payload.get("source", {})
        for record in records:
            descriptor = record.get("__dataclass__", {})
            if descriptor.get("name") != "TopologyDecision":
                continue
            decision_count += 1
            fields = descriptor.get("fields", {})
            if "topology_label" not in fields:
                missing_label_count += 1
                continue
            component_id = fields.get("component_id")
            component = components.get(component_id)
            bbox = _tuple_items(
                component.get("bbox_xywh", ())) if component else ()
            if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or
                    float(bbox[2]) <= 0 or float(bbox[3]) <= 0):
                missing_bbox_count += 1
                continue
            entries.append({
                "topology_label": fields["topology_label"],
                "component_id": component_id,
                "bbox_xywh": list(bbox),
                "source": {
                    "frame_id": source.get("frame_id"),
                    "sim_time_ns": source.get("sim_time_ns"),
                    "relative_path": source.get("relative_path"),
                },
                "frame": {
                    "filename": frame_path.name,
                    "image_url":
                        f"/frames/{quote(run_id)}/{quote(frame_path.name)}",
                },
            })
    return {
        "version": "deterministic-v3.topology-component-catalog.v1",
        "id": run_id,
        "decision_count": decision_count,
        "labeled_decision_count": decision_count - missing_label_count,
        "missing_topology_label_count": missing_label_count,
        "missing_bbox_xywh_count": missing_bbox_count,
        "available_topology_labels": sorted({
            entry["topology_label"] for entry in entries
        }),
        "components": entries,
        "unmatched_review_json": len(
            [record for record in reviews if record.path not in matched_reviews]),
    }


def _standard_gate_frame_evidence(payload):
    """Extract exact shared records needed to review standard-gate results."""
    frame, density_records, _ = _schema_evidence(payload)
    return frame, density_records


def _review_evidence_by_identity(records, expected_run_id=None):
    evidence = {}
    for record in records:
        payload = json.loads(record.path.read_text(encoding="utf-8"))
        source = payload.get("source", {})
        try:
            source_run_id = str(source["run_id"])
            identity = (int(source["frame_id"]), int(source["sim_time_ns"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("review JSON is missing exact frame identity") \
                from error
        if expected_run_id is not None and source_run_id != expected_run_id:
            raise ValueError("review JSON source.run_id does not match run")
        if identity in evidence:
            raise ValueError("duplicate review JSON frame identity")
        evidence[identity] = (record, payload)
    return evidence


def _logged_frame_by_identity(run_path):
    result = {}
    for path in _logged_frames(run_path):
        match = FRAME_IDENTITY_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        identity = tuple(map(int, match.groups()))
        if identity in result:
            raise ValueError("duplicate logged source frame identity")
        result[identity] = path
    return result


def _component_fields_by_id(frame):
    components = {}
    for value in _tuple_items(frame.get("components", ())):
        fields = (_dataclass_fields(value, "ComponentObservation")
                  if "__dataclass__" in value else value)
        component_id = int(fields["component_id"])
        if component_id in components:
            raise ValueError("duplicate ComponentObservation.component_id")
        components[component_id] = fields
    return components


def _density_profile_fields(record):
    profile = record["profile"]
    return (_dataclass_fields(profile, "DensityProfile")
            if "__dataclass__" in profile else profile)


def _selected_density_record(density_records, component_id, profile_id):
    matches = [
        record for record in density_records
        if int(record["component_id"]) == component_id
        and _density_profile_fields(record).get("profile_id") == profile_id
    ]
    if len(matches) > 1:
        raise ValueError("duplicate selected DensityEvidence")
    return matches[0] if matches else None


def _density_catalog_fields(record):
    if record is None:
        return None
    return {
        "component_id": record["component_id"],
        "profile": _density_profile_fields(record),
        "p70_threshold": record["p70_threshold"],
        "p80_threshold": record["p80_threshold"],
        "p90_threshold": record["p90_threshold"],
    }


def _standard_gate_layer_url(
    run_id, frame_id, sim_time_ns, component_id, layer_id
):
    return "/standard-gate-layers/{}/{}/{}/{}/{}.png".format(
        quote(run_id), frame_id, sim_time_ns, component_id, quote(layer_id))


def standard_gate_result_catalog(
    run_id, logs_root=LOGS_ROOT, review_root=REVIEW_ROOT
):
    """Join serialized standard results to exact shared component evidence."""
    run_path = discover_logged_runs(logs_root).get(run_id)
    if run_path is None:
        raise FileNotFoundError(run_id)
    review_records = [
        record for record in discover_review_records(review_root)
        if record.run_id == run_id
    ]
    review_by_identity = _review_evidence_by_identity(
        review_records, expected_run_id=run_id)
    logged_by_identity = _logged_frame_by_identity(run_path)
    geometry_records = sorted((
        record for record in discover_geometry_review_records(review_root)
        if record.run_id == run_id
    ), key=lambda record: (record.sim_time_ns, record.frame_id))

    results = []
    missing_frame = 0
    missing_component = 0
    missing_density = 0
    for geometry_record in geometry_records:
        item = _geometry_review_item(geometry_record)
        runtime = item["runtime_result"]
        identity = (geometry_record.frame_id, geometry_record.sim_time_ns)
        standard_results = runtime.get("standard_gate_results", ())
        if not isinstance(standard_results, list):
            raise ValueError(
                "GeometryFrameResult.standard_gate_results must be a list")
        review_item = review_by_identity.get(identity)
        frame = None
        density_records = ()
        components = {}
        source = {
            "frame_id": identity[0],
            "sim_time_ns": identity[1],
            "relative_path": None,
        }
        if review_item is not None:
            _, review_payload = review_item
            frame, density_records = _standard_gate_frame_evidence(
                review_payload)
            if (int(frame["frame_id"]), int(frame["sim_time_ns"])) != identity:
                raise ValueError(
                    "FrameObservation identity does not match GeometryFrameResult")
            components = _component_fields_by_id(frame)
            review_source = review_payload.get("source", {})
            source = {
                "frame_id": review_source.get("frame_id"),
                "sim_time_ns": review_source.get("sim_time_ns"),
                "relative_path": review_source.get("relative_path"),
            }

        frame_path = logged_by_identity.get(identity)
        frame_view = {
            "filename": None if frame_path is None else frame_path.name,
            "image_url": (None if frame_path is None else
                          f"/frames/{quote(run_id)}/{quote(frame_path.name)}"),
        }
        for standard in standard_results:
            if not isinstance(standard, dict):
                raise ValueError("StandardGateResult must be an object")
            try:
                standard_identity = (
                    int(standard["frame_id"]), int(standard["sim_time_ns"]))
                component_id = int(standard["component_id"])
                selected_profile_id = str(
                    standard["selected_density_profile"]["profile_id"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid serialized StandardGateResult") \
                    from error
            if standard_identity != identity:
                raise ValueError(
                    "StandardGateResult identity does not match GeometryFrameResult")
            component = components.get(component_id)
            density = _selected_density_record(
                density_records, component_id, selected_profile_id)
            if (density is not None and
                    _density_profile_fields(density) !=
                    standard["selected_density_profile"]):
                raise ValueError(
                    "selected DensityProfile differs from StandardGateResult")
            if frame is None:
                missing_frame += 1
            elif component is None:
                missing_component += 1
            if density is None:
                missing_density += 1

            layers = {}
            if component is not None:
                layer_id = "FrameObservation.closed_mask"
                layers[layer_id] = _standard_gate_layer_url(
                    run_id, *identity, component_id, layer_id)
            if density is not None:
                for field_name in DENSITY_ARRAY_FIELDS:
                    layer_id = _density_layer_id(
                        selected_profile_id, field_name)
                    layers[layer_id] = _standard_gate_layer_url(
                        run_id, *identity, component_id, layer_id)
            component_catalog = None if component is None else {
                "component_id": component.get("component_id"),
                "bbox_xywh": list(_tuple_items(component.get(
                    "bbox_xywh", ()))),
                "image_origin_uv": list(_tuple_items(component.get(
                    "image_origin_uv", ()))),
                "analysis_shape": list(_tuple_items(component.get(
                    "analysis_shape", ()))),
                "touches_frame": component.get("touches_frame"),
            }
            results.append({
                "frame_id": identity[0],
                "sim_time_ns": identity[1],
                "component_id": component_id,
                "source": source,
                "frame": frame_view,
                "ComponentObservation": component_catalog,
                "StandardGateResult": standard,
                "DensityEvidence": _density_catalog_fields(density),
                "layers": layers,
            })

    return {
        "version": "deterministic-v3.standard-gate-result-catalog.v1",
        "id": run_id,
        "standard_gate_result_count": len(results),
        "accepted_standard_gate_result_count": sum(
            item["StandardGateResult"].get("accepted") is True
            for item in results),
        "high_confidence_standard_gate_result_count": sum(
            item["StandardGateResult"].get("high_confidence") is True
            for item in results),
        "missing_frame_observation_count": missing_frame,
        "missing_component_observation_count": missing_component,
        "missing_density_evidence_count": missing_density,
        "results": results,
    }


def _confined(root, relative):
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise FileNotFoundError(relative) from error
    return candidate


def discover_evaluation_runs(evaluation_root=EVALUATION_ROOT):
    evaluations = []
    if not evaluation_root.is_dir():
        return {
            "version": "deterministic-v3.backend-evaluation-catalog.v1",
            "evaluations": evaluations,
        }
    for path in sorted(evaluation_root.glob("evaluation-*")):
        if not path.is_dir() or not EVALUATION_PATTERN.fullmatch(path.name):
            continue
        metadata = _read_json_file(path / "metadata.json")
        summary = _read_json_file(path / "summary.json")
        if not isinstance(metadata, dict) or not isinstance(summary, dict):
            continue
        source = metadata.get("source_run", {})
        overlays = metadata.get("overlays", {})
        evaluations.append({
            "id": path.name,
            "path": str(path),
            "created_utc": metadata.get("created_utc"),
            "source_run_id": source.get("run_id") if isinstance(source, dict) else None,
            "selected_frame_count": metadata.get("selection", {}).get("selected_frame_count"),
            "processed_frame_count": summary.get("processed_frame_count"),
            "frames_changed": summary.get("frames_changed"),
            "frames_with_errors": summary.get("frames_with_errors"),
            "frames_with_vehicle_state": summary.get("frames_with_vehicle_state"),
            "max_position_delta_m": summary.get("max_position_delta_m"),
            "overlays_enabled": overlays.get("enabled") is True if isinstance(overlays, dict) else False,
            "overlay_layers": overlays.get("layers", ()) if isinstance(overlays, dict) else (),
        })
    return {
        "version": "deterministic-v3.backend-evaluation-catalog.v1",
        "evaluations": evaluations,
    }


def _read_json_file(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def _evaluation_frame_overlay_urls(evaluation_id, evaluation_dir, frame_record):
    overlays = {}
    filename = "frame-{:08d}-{}.png".format(
        int(frame_record["source"]["frame_id"]),
        int(frame_record["source"]["sim_time_ns"]),
    )
    overlays_root = evaluation_dir / "overlays"
    if not overlays_root.is_dir():
        return overlays
    for backend_dir in sorted(path for path in overlays_root.iterdir() if path.is_dir()):
        if not BACKEND_PATTERN.fullmatch(backend_dir.name):
            continue
        backend_layers = {}
        for layer_dir in sorted(path for path in backend_dir.iterdir() if path.is_dir()):
            if not OVERLAY_LAYER_PATTERN.fullmatch(layer_dir.name):
                continue
            overlay_path = layer_dir / filename
            if overlay_path.is_file():
                backend_layers[layer_dir.name] = (
                    "/evaluation-overlays/{}/{}/{}/{}".format(
                        quote(evaluation_id),
                        quote(backend_dir.name),
                        quote(layer_dir.name),
                        quote(filename),
                    )
                )
        if backend_layers:
            overlays[backend_dir.name] = backend_layers
    return overlays


def evaluation_frame_catalog(evaluation_id, evaluation_root=EVALUATION_ROOT):
    if not EVALUATION_PATTERN.fullmatch(evaluation_id):
        raise FileNotFoundError(evaluation_id)
    evaluation_dir = _confined(evaluation_root, evaluation_id)
    metadata = _read_json_file(evaluation_dir / "metadata.json")
    summary = _read_json_file(evaluation_dir / "summary.json")
    frames_path = evaluation_dir / "frames.jsonl"
    if (not evaluation_dir.is_dir() or not isinstance(metadata, dict) or
            not isinstance(summary, dict) or not frames_path.is_file()):
        raise FileNotFoundError(evaluation_id)
    frames = []
    for frame in _read_jsonl(frames_path):
        source = frame.get("source", {})
        run_id = source.get("run_id")
        relative_path = source.get("relative_path")
        filename = Path(str(relative_path)).name
        frame_view = {
            "source": source,
            "comparison": frame.get("comparison", {}),
            "source_image_url": (
                "/frames/{}/{}".format(quote(str(run_id)), quote(filename))
                if run_id and filename else None
            ),
            "overlays": _evaluation_frame_overlay_urls(
                evaluation_id, evaluation_dir, frame),
        }
        frames.append(frame_view)
    source_run = metadata.get("source_run", {})
    metadata_view = {
        key: value for key, value in metadata.items()
        if key != "source_run"
    }
    metadata_view["source_run"] = {
        key: value for key, value in source_run.items()
        if key not in {"metadata", "summary", "status"}
    } if isinstance(source_run, dict) else source_run
    return {
        "version": "deterministic-v3.backend-evaluation-frames.v1",
        "id": evaluation_id,
        "metadata": metadata_view,
        "summary": summary,
        "frames": frames,
    }


def _decode_array(value, expected_dtype):
    descriptor = value["__ndarray__"]
    if descriptor.get("encoding") not in {
        "numpy-c-order-zlib-base64-v1",
        "numpy-contiguous-zlib-base64-v2",
    }:
        raise ValueError("unsupported array encoding")
    if descriptor.get("order", "C") != "C":
        raise ValueError("only C-order arrays are supported")
    if descriptor.get("dtype") != expected_dtype:
        raise ValueError("unexpected array dtype")
    shape = tuple(int(dimension) for dimension in descriptor["shape"])
    if len(shape) != 2 or any(dimension < 1 for dimension in shape):
        raise ValueError("review layers must be non-empty 2D arrays")
    raw = zlib.decompress(base64.b64decode(descriptor["data"], validate=True))
    item_sizes = {"|u1": 1, "<i4": 4, "<f8": 8}
    item_size = item_sizes.get(expected_dtype)
    if item_size is None:
        raise ValueError("unsupported array dtype")
    if len(raw) != shape[0] * shape[1] * item_size:
        raise ValueError("array payload length does not match its shape")
    return shape, raw


def _dataclass_fields(value, expected_name):
    descriptor = value.get("__dataclass__", {})
    if descriptor.get("name") != expected_name:
        raise ValueError(f"expected runtime record {expected_name}")
    return descriptor["fields"]


def _tuple_items(value):
    return value.get("__tuple__", value) if isinstance(value, dict) else value


def _schema_evidence(payload):
    if "schema" in payload:
        schema = payload["schema"]
        return (schema["FrameObservation"], schema["DensityEvidence"],
                schema.get("CShapeResult", ()))
    frame = None
    density_records = []
    c_shape_records = []
    for record in payload.get("schema_records", ()):
        descriptor = record.get("__dataclass__", {})
        if descriptor.get("name") == "FrameObservation":
            frame = descriptor["fields"]
        elif descriptor.get("name") == "DensityEvidence":
            density_records.append(descriptor["fields"])
        elif descriptor.get("name") == "CShapeResult":
            c_shape_records.append(descriptor["fields"])
    if frame is None:
        raise ValueError("FrameObservation runtime record is missing")
    return frame, density_records, c_shape_records


def _compose_component_layer(
    frame, density_records, profile_id, field_name, dtype
):
    height, width = map(int, _tuple_items(frame["image_shape"])[:2])
    canvas = bytearray(height * width)
    components = {
        component["component_id"]: component
        for component in (
            _dataclass_fields(value, "ComponentObservation")
            if "__dataclass__" in value else value
            for value in _tuple_items(frame.get("components", ())))
    }
    for record in density_records:
        profile_value = record["profile"]
        profile = (_dataclass_fields(profile_value, "DensityProfile")
                   if "__dataclass__" in profile_value else profile_value)
        if profile["profile_id"] != profile_id:
            continue
        component = components.get(record["component_id"])
        if component is None:
            continue
        shape, raw = _decode_array(record[field_name], dtype)
        origin_x, origin_y = map(
            int, _tuple_items(component["image_origin_uv"]))
        values = raw if dtype == "|u1" else (
            value[0] for value in struct.iter_unpack("<d", raw))
        for index, value in enumerate(values):
            local_y, local_x = divmod(index, shape[1])
            x, y = origin_x + local_x, origin_y + local_y
            if 0 <= x < width and 0 <= y < height:
                intensity = (255 if value else 0) if dtype == "|u1" else round(
                    min(max(float(value), 0.0), 1.0) * 255)
                offset = y * width + x
                canvas[offset] = max(canvas[offset], intensity)
    return width, height, bytes(canvas)


def _compose_touches_frame_layer(frame):
    shape, raw = _decode_array(frame["component_labels"], "<i4")
    touching_ids = {
        component["component_id"]
        for component in (
            _dataclass_fields(value, "ComponentObservation")
            if "__dataclass__" in value else value
            for value in _tuple_items(frame.get("components", ())))
        if component.get("touches_frame") is True
    }
    pixels = bytes(
        255 if label[0] in touching_ids else 0
        for label in struct.iter_unpack("<i", raw))
    height, width = shape
    return width, height, pixels


def _compose_c_shape_layer(frame, c_shape_records):
    height, width = map(int, _tuple_items(frame["image_shape"])[:2])
    canvas = bytearray(height * width)
    for record in c_shape_records:
        shape, raw = _decode_array(record["refined_mask"], "|u1")
        origin_x, origin_y = map(
            int, _tuple_items(record["refined_mask_origin_uv"]))
        for index, value in enumerate(raw):
            if not value:
                continue
            local_y, local_x = divmod(index, shape[1])
            x, y = origin_x + local_x, origin_y + local_y
            if 0 <= x < width and 0 <= y < height:
                canvas[y * width + x] = 255
    return width, height, bytes(canvas)


def _png_chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload +
            struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))


def _grayscale_png(width, height, pixels):
    rows = b"".join(
        b"\0" + pixels[y * width:(y + 1) * width]
        for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" +
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) +
            _png_chunk(b"IDAT", zlib.compress(rows)) +
            _png_chunk(b"IEND", b""))


def render_review_layer(review_path, layer_id):
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    frame, density_records, c_shape_records = _schema_evidence(payload)
    if layer_id.startswith("FrameObservation."):
        field_name = layer_id.removeprefix("FrameObservation.")
        if field_name not in FRAME_MASK_FIELDS:
            raise KeyError(layer_id)
        shape, pixels = _decode_array(frame[field_name], "|u1")
        pixels = bytes(255 if value else 0 for value in pixels)
        height, width = shape
    elif layer_id == TOUCHES_FRAME_LAYER_ID:
        width, height, pixels = _compose_touches_frame_layer(frame)
    elif layer_id == C_SHAPE_MASK_LAYER_ID:
        width, height, pixels = _compose_c_shape_layer(
            frame, c_shape_records)
    else:
        match = re.fullmatch(
            r"DensityEvidence\[([^]]+)\]\.([A-Za-z0-9_]+)", layer_id)
        if match is None or match.group(2) not in DENSITY_ARRAY_FIELDS:
            raise KeyError(layer_id)
        profile_id, field_name = match.groups()
        available_profile_ids = {
            (_dataclass_fields(record["profile"], "DensityProfile")
             if "__dataclass__" in record["profile"] else record["profile"])
            ["profile_id"]
            for record in density_records
        }
        if profile_id not in available_profile_ids:
            raise KeyError(layer_id)
        dtype = "<f8" if field_name == "final_field" else "|u1"
        width, height, pixels = _compose_component_layer(
            frame, density_records, profile_id, field_name, dtype)
    return _grayscale_png(width, height, pixels)


def _component_bbox(component):
    bbox = tuple(map(int, _tuple_items(component.get("bbox_xywh", ()))))
    if len(bbox) != 4 or bbox[2] < 1 or bbox[3] < 1:
        raise ValueError("invalid ComponentObservation.bbox_xywh")
    return bbox


def _closed_mask_component_pixels(frame, component):
    shape, raw = _decode_array(frame["closed_mask"], "|u1")
    labels_shape, labels = _decode_array(frame["component_labels"], "<i4")
    if labels_shape != shape:
        raise ValueError(
            "FrameObservation.component_labels shape does not match closed_mask")
    height, width = shape
    x, y, crop_width, crop_height = _component_bbox(component)
    component_id = int(component["component_id"])
    if (x < 0 or y < 0 or x + crop_width > width or
            y + crop_height > height):
        raise ValueError("ComponentObservation.bbox_xywh escapes closed_mask")
    pixels = bytearray(crop_width * crop_height)
    for crop_y in range(crop_height):
        row = y + crop_y
        for crop_x in range(crop_width):
            column = x + crop_x
            index = row * width + column
            label = struct.unpack_from("<i", labels, index * 4)[0]
            if raw[index] and label == component_id:
                pixels[crop_y * crop_width + crop_x] = 255
    return crop_width, crop_height, bytes(pixels)


def _density_component_pixels(record, component, field_name):
    dtype = "<f8" if field_name == "final_field" else "|u1"
    shape, raw = _decode_array(record[field_name], dtype)
    local_height, local_width = shape
    origin_x, origin_y = map(
        int, _tuple_items(component.get("image_origin_uv", ())))
    x, y, crop_width, crop_height = _component_bbox(component)
    pixels = bytearray(crop_width * crop_height)
    for crop_y in range(crop_height):
        local_y = y + crop_y - origin_y
        if not 0 <= local_y < local_height:
            continue
        for crop_x in range(crop_width):
            local_x = x + crop_x - origin_x
            if not 0 <= local_x < local_width:
                continue
            index = local_y * local_width + local_x
            if dtype == "|u1":
                intensity = 255 if raw[index] else 0
            else:
                value = struct.unpack_from("<d", raw, index * 8)[0]
                intensity = round(min(max(float(value), 0.0), 1.0) * 255)
            pixels[crop_y * crop_width + crop_x] = intensity
    return crop_width, crop_height, bytes(pixels)


def render_standard_gate_component_layer(
    review_path, component_id, selected_profile_id, layer_id
):
    """Render one exact selected-profile layer cropped to one component."""
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    frame, density_records = _standard_gate_frame_evidence(payload)
    component = _component_fields_by_id(frame).get(int(component_id))
    if component is None:
        raise KeyError("ComponentObservation.component_id")
    if layer_id == "FrameObservation.closed_mask":
        width, height, pixels = _closed_mask_component_pixels(frame, component)
    else:
        match = re.fullmatch(
            r"DensityEvidence\[([^]]+)\]\.([A-Za-z0-9_]+)", layer_id)
        if (match is None or match.group(1) != selected_profile_id or
                match.group(2) not in DENSITY_ARRAY_FIELDS):
            raise KeyError(layer_id)
        density = _selected_density_record(
            density_records, int(component_id), selected_profile_id)
        if density is None:
            raise KeyError("DensityEvidence")
        width, height, pixels = _density_component_pixels(
            density, component, match.group(2))
    return _grayscale_png(width, height, pixels)


def standard_gate_review_layer(
    run_id, frame_id, sim_time_ns, component_id,
    layer_id, logs_root=LOGS_ROOT, review_root=REVIEW_ROOT,
):
    """Resolve a layer only for an exact serialized StandardGateResult."""
    if run_id not in discover_logged_runs(logs_root):
        raise FileNotFoundError(run_id)
    identity = (int(frame_id), int(sim_time_ns))
    geometry_matches = [
        record for record in discover_geometry_review_records(review_root)
        if record.run_id == run_id
        and (record.frame_id, record.sim_time_ns) == identity
    ]
    if not geometry_matches:
        raise FileNotFoundError(identity)
    if len(geometry_matches) != 1:
        raise ValueError("duplicate GeometryFrameResult frame identity")
    runtime = _geometry_review_item(geometry_matches[0])["runtime_result"]
    standard_matches = [
        result for result in runtime.get("standard_gate_results", ())
        if isinstance(result, dict)
        and int(result.get("component_id", -1)) == int(component_id)
    ]
    if not standard_matches:
        raise FileNotFoundError(component_id)
    if len(standard_matches) != 1:
        raise ValueError("duplicate StandardGateResult component identity")
    standard = standard_matches[0]
    if (int(standard["frame_id"]), int(standard["sim_time_ns"])) != identity:
        raise ValueError(
            "StandardGateResult identity does not match GeometryFrameResult")
    profile_id = str(standard["selected_density_profile"]["profile_id"])

    reviews = [
        record for record in discover_review_records(review_root)
        if record.run_id == run_id
    ]
    review_item = _review_evidence_by_identity(
        reviews, expected_run_id=run_id).get(identity)
    if review_item is None:
        raise FileNotFoundError(identity)
    return render_standard_gate_component_layer(
        review_item[0].path, component_id, profile_id, layer_id)


class ReviewUiHandler(SimpleHTTPRequestHandler):
    """Static frontend plus read-only catalogs for logs and review dumps."""

    server_version = "DeterministicV3Review/1"

    def __init__(self, *args, frontend_root, logs_root, review_root, **kwargs):
        self.logs_root = Path(logs_root)
        self.review_root = Path(review_root)
        self.evaluation_root = EVALUATION_ROOT
        super().__init__(*args, directory=str(frontend_root), **kwargs)

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/api/runs":
            return self._send_json(run_catalog(self.logs_root, self.review_root))
        if path == "/api/evaluations":
            return self._send_json(discover_evaluation_runs(self.evaluation_root))
        match = re.fullmatch(r"/api/evaluations/([^/]+)/frames", path)
        if match:
            try:
                payload = evaluation_frame_catalog(
                    match.group(1), self.evaluation_root)
            except FileNotFoundError:
                return self.send_error(
                    HTTPStatus.NOT_FOUND, "Unknown evaluation")
            except (TypeError, ValueError, OSError, json.JSONDecodeError):
                return self.send_error(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "Evaluation catalog is unavailable")
            return self._send_json(payload)
        match = re.fullmatch(r"/api/runs/([^/]+)/frames", path)
        if match:
            try:
                payload = frame_catalog(
                    match.group(1), self.logs_root, self.review_root)
            except FileNotFoundError:
                return self.send_error(HTTPStatus.NOT_FOUND, "Unknown run")
            return self._send_json(payload)
        match = re.fullmatch(
            r"/api/runs/([^/]+)/geometry-frames/([^/]+)", path)
        if match:
            try:
                payload = geometry_frame_result(
                    match.group(1), match.group(2), self.logs_root,
                    self.review_root)
            except FileNotFoundError:
                return self.send_error(
                    HTTPStatus.NOT_FOUND, "GeometryFrameResult unavailable")
            except (TypeError, ValueError, OSError, json.JSONDecodeError):
                return self.send_error(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "GeometryFrameResult is ambiguous or invalid")
            return self._send_json(payload)
        match = re.fullmatch(r"/api/runs/([^/]+)/geometry-history", path)
        if match:
            try:
                payload = geometry_frame_history(
                    match.group(1), self.logs_root, self.review_root)
            except FileNotFoundError:
                return self.send_error(HTTPStatus.NOT_FOUND, "Unknown run")
            except (TypeError, ValueError, OSError, json.JSONDecodeError):
                return self.send_error(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "GeometryFrameResult history is ambiguous or invalid")
            return self._send_json(payload)
        match = re.fullmatch(r"/api/runs/([^/]+)/pnp-world-replay", path)
        if match:
            try:
                payload = pnp_world_replay(
                    match.group(1), self.logs_root, self.review_root)
            except FileNotFoundError:
                return self.send_error(
                    HTTPStatus.NOT_FOUND, "PnP world replay unavailable")
            except (TypeError, ValueError, OSError, json.JSONDecodeError):
                return self.send_error(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "PnP world replay is ambiguous or invalid")
            return self._send_json(payload)
        match = re.fullmatch(r"/api/runs/([^/]+)/topology-components", path)
        if match:
            try:
                payload = topology_component_catalog(
                    match.group(1), self.logs_root, self.review_root)
            except FileNotFoundError:
                return self.send_error(HTTPStatus.NOT_FOUND, "Unknown run")
            except (TypeError, ValueError, OSError, json.JSONDecodeError):
                return self.send_error(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "TopologyDecision review is unavailable")
            return self._send_json(payload)
        match = re.fullmatch(
            r"/api/runs/([^/]+)/standard-gate-results", path)
        if match:
            try:
                payload = standard_gate_result_catalog(
                    match.group(1), self.logs_root, self.review_root)
            except FileNotFoundError:
                return self.send_error(HTTPStatus.NOT_FOUND, "Unknown run")
            except (KeyError, TypeError, ValueError, OSError,
                    json.JSONDecodeError):
                return self.send_error(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    "StandardGateResult review is unavailable")
            return self._send_json(payload)
        match = re.fullmatch(r"/frames/([^/]+)/([^/]+)", path)
        if match:
            return self._send_frame(match.group(1), match.group(2))
        match = re.fullmatch(
            r"/evaluation-overlays/([^/]+)/([^/]+)/([^/]+)/([^/]+)", path)
        if match:
            return self._send_evaluation_overlay(*match.groups())
        match = re.fullmatch(
            r"/standard-gate-layers/([^/]+)/(\d+)/(\d+)/(\d+)/"
            r"([^/]+)\.png", path)
        if match:
            return self._send_standard_gate_layer(*match.groups())
        match = re.fullmatch(r"/schema-layers/([^/]+)/([^/]+)/([^/]+)\.png", path)
        if match:
            return self._send_schema_layer(*match.groups())
        if path.startswith("/reviews/"):
            return self._send_review(path.removeprefix("/reviews/"))
        return super().do_GET()

    def _send_json(self, payload):
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_frame(self, run_id, filename):
        if not RUN_PATTERN.fullmatch(run_id) or Path(filename).name != filename:
            return self.send_error(HTTPStatus.NOT_FOUND)
        run_path = discover_logged_runs(self.logs_root).get(run_id)
        if run_path is None:
            return self.send_error(HTTPStatus.NOT_FOUND, "Unknown run")
        path = _confined(run_path / "vision_frames", filename)
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            return self.send_error(HTTPStatus.NOT_FOUND, "Unknown frame")
        return self._send_file(path, "public, max-age=60")

    def _send_evaluation_overlay(self, evaluation_id, backend, layer_id, filename):
        if (not EVALUATION_PATTERN.fullmatch(evaluation_id) or
                not BACKEND_PATTERN.fullmatch(backend) or
                not OVERLAY_LAYER_PATTERN.fullmatch(layer_id) or
                Path(filename).name != filename or
                Path(filename).suffix.lower() != ".png"):
            return self.send_error(HTTPStatus.NOT_FOUND)
        path = _confined(
            self.evaluation_root / evaluation_id / "overlays" / backend /
            layer_id,
            filename,
        )
        if not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND, "Unknown overlay")
        return self._send_file(path, "no-store", "image/png")

    def _send_review(self, relative):
        path = _confined(self.review_root, relative)
        if not path.is_file() or path.suffix.lower() != ".json":
            return self.send_error(HTTPStatus.NOT_FOUND, "Unknown review JSON")
        return self._send_file(path, "no-store", "application/json")

    def _send_schema_layer(self, run_id, frame_stem, layer_id):
        if (not RUN_PATTERN.fullmatch(run_id) or
                Path(frame_stem).name != frame_stem or
                not _is_supported_layer_id(layer_id)):
            return self.send_error(HTTPStatus.NOT_FOUND)
        path = _confined(
            self.review_root / run_id / "frames", f"{frame_stem}.json")
        if not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND, "Unknown review frame")
        try:
            encoded = render_review_layer(path, layer_id)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            return self.send_error(HTTPStatus.UNPROCESSABLE_ENTITY,
                                   "Review layer is unavailable")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_standard_gate_layer(
        self, run_id, frame_id, sim_time_ns, component_id, layer_id
    ):
        density_match = re.fullmatch(
            r"DensityEvidence\[([^]]+)\]\.([A-Za-z0-9_]+)", layer_id)
        if (not RUN_PATTERN.fullmatch(run_id) or
                (layer_id not in STANDARD_GATE_COMPONENT_LAYER_FIELDS and
                 (density_match is None or
                  density_match.group(2) not in DENSITY_ARRAY_FIELDS))):
            return self.send_error(HTTPStatus.NOT_FOUND)
        try:
            encoded = standard_gate_review_layer(
                run_id, int(frame_id), int(sim_time_ns), int(component_id),
                layer_id, self.logs_root, self.review_root)
        except FileNotFoundError:
            return self.send_error(
                HTTPStatus.NOT_FOUND, "StandardGateResult layer unavailable")
        except (KeyError, TypeError, ValueError, OSError,
                json.JSONDecodeError, struct.error):
            return self.send_error(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "StandardGateResult layer is invalid")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path, cache_control, content_type=None):
        content_type = content_type or mimetypes.guess_type(path.name)[0]
        with path.open("rb") as source:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type or "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            shutil.copyfileobj(source, self.wfile)


def build_server(host="127.0.0.1", port=8765, *, frontend_root=FRONTEND_ROOT,
                 logs_root=LOGS_ROOT):
    handler = partial(
        ReviewUiHandler, frontend_root=frontend_root, logs_root=logs_root,
        review_root=REVIEW_ROOT)
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--frontend-root", type=Path, default=FRONTEND_ROOT)
    parser.add_argument("--logs-root", type=Path, default=LOGS_ROOT)
    args = parser.parse_args(argv)
    REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    server = build_server(
        args.host, args.port, frontend_root=args.frontend_root,
        logs_root=args.logs_root)
    print(f"review UI: http://{args.host}:{server.server_port}/", flush=True)
    print(f"logs: {args.logs_root.resolve()}", flush=True)
    print(f"reviews: {REVIEW_ROOT.resolve()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
