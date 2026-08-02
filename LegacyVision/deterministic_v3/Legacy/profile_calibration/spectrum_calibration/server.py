"""Serve the isolated ten-column density-spectrum calibration laboratory."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from hashlib import sha256
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import sqlite3
from time import perf_counter
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from ...configurations import DEFAULT_DENSITY_CONFIGURATION
from ...preprocessing import DEFAULT_CONFIG
from ...schema import DensityProfile
from ..density_runtime import CandidateSettings, compute_density_evidence
from .rendering import encode_png, vertical_evidence_sheet


HERE = Path(__file__).resolve().parent
UI_ROOT = HERE / "ui"
DEFAULT_DATABASE_PATH = HERE / "assets" / "spectrum_assets.sqlite3"
DEFAULT_PORT = 8786
BIN_COUNT = 10
MAX_PAGE_SIZE = 100
COMPONENT_IMAGE = re.compile(
    r"^/api/component(?:s)?/(\d+)/candidate\.png$")
PROFILE_FIELDS = (
    "density_radius_px",
    "ridge_radius_px",
    "relative_cap",
    "inverse_gamma",
    "ridge_gamma",
    "open_kernel_radius_px",
    "close_kernel_radius_px",
)
DENSITY_PROFILE_FIELDS = PROFILE_FIELDS[:5]
INTEGER_FIELDS = frozenset((
    "density_radius_px",
    "ridge_radius_px",
    "open_kernel_radius_px",
    "close_kernel_radius_px",
))
MORPHOLOGY_FIELDS = (
    "open_kernel_radius_px", "close_kernel_radius_px")
ENDPOINT_BOUNDS = {
    "density_radius_px": {"minimum": 1, "maximum": 24, "step": 1},
    "ridge_radius_px": {"minimum": 1, "maximum": 20, "step": 1},
    "relative_cap": {"minimum": 1.0, "maximum": 5.0, "step": 0.05},
    "inverse_gamma": {"minimum": 0.01, "maximum": 100.0, "step": 0.01},
    "ridge_gamma": {"minimum": 0.01, "maximum": 100.0, "step": 0.01},
    "open_kernel_radius_px": {"minimum": 0, "maximum": 10, "step": 1},
    "close_kernel_radius_px": {"minimum": 0, "maximum": 10, "step": 1},
}
GAMMA_QUANTUM = Decimal("0.000001")


def _compact(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_compatible(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _column(columns: set[str], *choices: str) -> str:
    for choice in choices:
        if choice in columns:
            return choice
    raise ValueError(f"asset database is missing one of columns: {choices}")


def _nested_minimum_area(value):
    if isinstance(value, dict):
        if "minimum_component_area_px" in value:
            return value["minimum_component_area_px"]
        for child in value.values():
            found = _nested_minimum_area(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _nested_minimum_area(child)
            if found is not None:
                return found
    return None


@dataclass(frozen=True, slots=True)
class BinSnapshot:
    bin_index: int
    count: int
    min_area: int
    max_area: int
    median_area: float
    default_profile_id: str

    def payload(self) -> dict:
        return {
            "bin_index": self.bin_index,
            "count": self.count,
            "component_count": self.count,
            "min_area": self.min_area,
            "minimum_area_px": self.min_area,
            "max_area": self.max_area,
            "maximum_area_px": self.max_area,
            "median_area": self.median_area,
            "median_area_px": self.median_area,
            "area_label": f"{self.min_area}–{self.max_area} px",
            "default_profile_id": self.default_profile_id,
        }


@dataclass(frozen=True, slots=True)
class ComponentAsset:
    values: dict
    source_png: bytes
    closed_mask_png: bytes


@dataclass(frozen=True, slots=True)
class CandidateMorphology:
    open_kernel_radius_px: int
    close_kernel_radius_px: int

    @property
    def open_kernel_width_px(self) -> int:
        return (2 * self.open_kernel_radius_px + 1
                if self.open_kernel_radius_px else 0)

    @property
    def close_kernel_width_px(self) -> int:
        return (2 * self.close_kernel_radius_px + 1
                if self.close_kernel_radius_px else 0)


@dataclass(frozen=True, slots=True)
class MorphologyResult:
    mask: np.ndarray
    provenance: dict


@dataclass(frozen=True, slots=True)
class ResolvedCandidate:
    signature: str
    canonical_endpoints_json: str
    endpoints: dict
    resolved_bins: tuple[dict, ...]

    def _values_for_bin(self, bin_index: int) -> dict:
        try:
            values = self.resolved_bins[bin_index]
        except IndexError as error:
            raise KeyError(f"unknown spectrum bin: {bin_index}") from error
        if values["bin_index"] != bin_index:
            raise ValueError("resolved spectrum bins are not ordinal")
        return values

    def settings_for_bin(self, bin_index: int) -> CandidateSettings:
        values = self._values_for_bin(bin_index)
        return CandidateSettings(
            density_radius_px=values["density_radius_px"],
            ridge_radius_px=values["ridge_radius_px"],
            relative_cap=values["relative_cap"],
            inverse_gamma=values["inverse_gamma"],
            ridge_gamma=values["ridge_gamma"],
        )

    def morphology_for_bin(self, bin_index: int) -> CandidateMorphology:
        values = self._values_for_bin(bin_index)
        return CandidateMorphology(
            open_kernel_radius_px=values["open_kernel_radius_px"],
            close_kernel_radius_px=values["close_kernel_radius_px"],
        )

    def payload(self) -> dict:
        return {
            "candidate_signature": self.signature,
            "endpoints": self.endpoints,
            "resolved_bins": list(self.resolved_bins),
        }


@dataclass(frozen=True, slots=True)
class RenderedCandidate:
    png: bytes
    compute_ms: float
    settings: dict
    morphology: dict
    thresholds: dict
    provenance: dict
    signature: str


class AssetStore:
    """Validated, read-only access to one immutable spectrum snapshot."""

    def __init__(self, database: Path) -> None:
        self.database = database.resolve()
        if not self.database.is_file():
            raise FileNotFoundError(
                f"spectrum asset database not found: {self.database}")
        self._columns = self._schema_columns()
        self.metadata = self._metadata()
        embedded_manifest = self.metadata.get("manifest")
        if not isinstance(embedded_manifest, dict):
            raise ValueError("asset database metadata has no manifest object")
        self.manifest_snapshot = self._validated_sidecar(embedded_manifest)
        self.profiles = self._profiles()
        self.profile_map = {
            profile.profile_id: profile for profile in self.profiles}
        self.bins = self._bins()
        self.bin_map = {item.bin_index: item for item in self.bins}
        self._validate_snapshot()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database.as_posix()}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def _schema_columns(self) -> dict[str, set[str]]:
        with self._connect() as connection:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            required = {"metadata", "profiles", "bins", "components"}
            missing = required - tables
            if missing:
                raise ValueError(
                    f"asset database is missing tables: {sorted(missing)}")
            return {
                table: {row[1] for row in connection.execute(
                    f"PRAGMA table_info({table})")}
                for table in required
            }

    def _metadata(self) -> dict:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key, value_json FROM metadata").fetchall()
        try:
            return {row["key"]: json.loads(row["value_json"]) for row in rows}
        except json.JSONDecodeError as error:
            raise ValueError("asset metadata contains invalid JSON") from error

    def _validated_sidecar(self, embedded: dict) -> dict:
        path = self.database.parent / "manifest.json"
        if not path.is_file():
            raise ValueError("immutable spectrum database requires manifest.json")
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("spectrum sidecar manifest contains invalid JSON") from error
        for field in (
            "asset_schema_version", "generation_id", "source_runs",
            "preprocessing", "density_configuration", "sort", "counts", "bins"):
            if sidecar.get(field) != embedded.get(field):
                raise ValueError(
                    f"sidecar and database manifests differ at {field}")
        snapshot = sidecar.get("database_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("sidecar manifest has no immutable database snapshot")
        if (snapshot.get("filename") != self.database.name or
                int(snapshot.get("size_bytes", -1)) != self.database.stat().st_size):
            raise ValueError("database filename or size differs from immutable snapshot")
        digest = sha256()
        with self.database.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != snapshot.get("sha256"):
            raise ValueError("database SHA-256 differs from immutable snapshot")
        return sidecar

    def _profiles(self) -> tuple[DensityProfile, ...]:
        columns = self._columns["profiles"]
        ordinal = _column(columns, "profile_ordinal", "ordinal")
        required = {
            "profile_id", "calibration_version", "maximum_component_area_px",
            "density_radius_px", "ridge_radius_px", "relative_cap",
            "inverse_gamma", "ridge_gamma",
        }
        if missing := required - columns:
            raise ValueError(f"profiles table is missing columns: {sorted(missing)}")
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM profiles ORDER BY {ordinal}").fetchall()
        return tuple(DensityProfile(
            profile_id=row["profile_id"],
            calibration_version=row["calibration_version"],
            maximum_component_area_px=row["maximum_component_area_px"],
            density_radius_px=int(row["density_radius_px"]),
            ridge_radius_px=int(row["ridge_radius_px"]),
            relative_cap=float(row["relative_cap"]),
            inverse_gamma=float(row["inverse_gamma"]),
            ridge_gamma=float(row["ridge_gamma"]),
        ) for row in rows)

    def _bins(self) -> tuple[BinSnapshot, ...]:
        columns = self._columns["bins"]
        count = _column(columns, "component_count", "count")
        minimum = _column(columns, "minimum_area_px", "min_area")
        maximum = _column(columns, "maximum_area_px", "max_area")
        median = _column(columns, "median_area_px", "median_area")
        profile = _column(columns, "default_profile_id", "profile_id")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM bins ORDER BY bin_index").fetchall()
        return tuple(BinSnapshot(
            bin_index=int(row["bin_index"]),
            count=int(row[count]),
            min_area=int(row[minimum]),
            max_area=int(row[maximum]),
            median_area=float(row[median]),
            default_profile_id=str(row[profile]),
        ) for row in rows)

    def _validate_snapshot(self) -> None:
        if tuple(item.bin_index for item in self.bins) != tuple(range(BIN_COUNT)):
            raise ValueError("spectrum database must contain bins 0 through 9")
        current_profiles = {
            profile.profile_id: _json_compatible(asdict(profile))
            for profile in DEFAULT_DENSITY_CONFIGURATION.profiles}
        stored_profiles = {
            profile.profile_id: _json_compatible(asdict(profile))
            for profile in self.profiles}
        if stored_profiles != current_profiles:
            raise ValueError(
                "asset density-profile snapshot differs from current configuration")
        if not DEFAULT_DENSITY_CONFIGURATION.ignore_frame_edge_clipped:
            raise ValueError("current density configuration no longer excludes clipping")
        if _json_compatible(self.manifest_snapshot.get("density_configuration")) != \
                _json_compatible(asdict(DEFAULT_DENSITY_CONFIGURATION)):
            raise ValueError(
                "manifest density configuration differs from current configuration")
        for item in self.bins:
            if item.default_profile_id not in self.profile_map:
                raise ValueError(
                    f"bin {item.bin_index} has an unknown default profile")
        for left, right in zip(self.bins, self.bins[1:]):
            if left.min_area < right.max_area:
                raise ValueError("spectrum bins are not ordered largest to smallest")
        minimum_area = _nested_minimum_area(self.metadata)
        if minimum_area is None:
            minimum_area = _nested_minimum_area(self.manifest_snapshot)
        if int(minimum_area) != 250:
            raise ValueError("spectrum assets must use minimum component area 250")
        preprocessing = self.manifest_snapshot.get("preprocessing", {})
        config_snapshot = preprocessing.get("config", preprocessing)
        version = str(config_snapshot.get("version", ""))
        if "a250" not in version:
            raise ValueError("spectrum preprocessing version must explicitly contain a250")
        for field in (
            "maximum_input_components", "close_kernel_px", "connectivity",
            "contour_simplify_epsilon_px", "distance_mask_size"):
            if field in config_snapshot and config_snapshot[field] != getattr(
                    DEFAULT_CONFIG, field):
                raise ValueError(
                    f"spectrum preprocessing snapshot differs at {field}")
        if preprocessing.get("post_close_touches_frame_policy") != "discard":
            raise ValueError("spectrum snapshot must discard frame-edge components")
        component_columns = self._columns["components"]
        required = {
            "asset_id", "bin_index", "run_id", "frame_id", "sim_time_ns",
            "component_id", "bbox_x", "bbox_y", "bbox_width", "bbox_height",
            "image_origin_u", "image_origin_v", "analysis_height",
            "analysis_width", "area_px", "fill_ratio", "solidity",
            "touches_frame", "source_png", "closed_mask_png",
        }
        if missing := required - component_columns:
            raise ValueError(
                f"components table is missing columns: {sorted(missing)}")
        _column(component_columns, "production_profile_id", "profile_id")
        with self._connect() as connection:
            actual = dict(connection.execute(
                "SELECT bin_index, COUNT(*) FROM components GROUP BY bin_index"))
            clipped = connection.execute(
                "SELECT COUNT(*) FROM components WHERE touches_frame != 0"
            ).fetchone()[0]
        if clipped:
            raise ValueError("spectrum database contains frame-edge components")
        for item in self.bins:
            if int(actual.get(item.bin_index, 0)) != item.count:
                raise ValueError(
                    f"bin {item.bin_index} component count differs from snapshot")

    def manifest(self) -> dict:
        payload = dict(self.manifest_snapshot)
        payload.update({
            "server_contract": "spectrum-calibration.v1",
            "profiles": [asdict(profile) for profile in self.profiles],
            "bins": [item.payload() for item in self.bins],
            "endpoint_defaults": self.default_endpoints(),
            "endpoint_bounds": ENDPOINT_BOUNDS,
            "interpolation": {
                "bin_indices": list(range(BIN_COUNT)),
                "largest_bin_index": 0,
                "smallest_bin_index": 9,
                "formula": "large + (small - large) * (bin_index / 9)",
                "integer_rounding": "half_up",
                "decimal_fields": [
                    "relative_cap", "inverse_gamma", "ridge_gamma"],
                "decimal_places": 6,
            },
        })
        return payload

    def default_endpoints(self) -> dict:
        largest = self.profile_map["scale_10"]
        smallest = self.profile_map["scale_01"]
        return {
            **{f"large_{name}": getattr(largest, name)
               for name in DENSITY_PROFILE_FIELDS},
            **{f"small_{name}": getattr(smallest, name)
               for name in DENSITY_PROFILE_FIELDS},
            "large_open_kernel_radius_px": 1,
            "large_close_kernel_radius_px": 1,
            "small_open_kernel_radius_px": 0,
            "small_close_kernel_radius_px": 0,
        }

    def components(self, offset: int, limit: int) -> dict:
        columns = self._columns["components"]
        profile_column = _column(
            columns, "production_profile_id", "profile_id")
        rank_order = "rank_index" if "rank_index" in columns else (
            "area_px DESC, run_id, frame_id, sim_time_ns, component_id")
        selected = (
            "asset_id, bin_index, run_id, frame_id, sim_time_ns, "
            "source_relative_path, component_id, "
            f"{profile_column} AS production_profile_id, "
            "bbox_x, bbox_y, bbox_width, bbox_height, image_origin_u, "
            "image_origin_v, analysis_height, analysis_width, area_px, "
            "fill_ratio, solidity"
        )
        if "touches_frame" in columns:
            selected += ", touches_frame"
        result = []
        with self._connect() as connection:
            for item in self.bins:
                rows = connection.execute(
                    f"SELECT {selected} FROM components "
                    f"WHERE bin_index = ? ORDER BY {rank_order} "
                    "LIMIT ? OFFSET ?",
                    (item.bin_index, limit, offset),
                ).fetchall()
                bin_payload = item.payload()
                bin_payload.update({
                    "total": item.count,
                    "offset": offset,
                    "limit": limit,
                    "records": [dict(row) for row in rows],
                })
                result.append(bin_payload)
        return {"offset": offset, "limit": limit, "bins": result}

    @lru_cache(maxsize=128)
    def component(self, asset_id: int) -> ComponentAsset:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM components WHERE asset_id = ?", (asset_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown component asset: {asset_id}")
        values = dict(row)
        return ComponentAsset(
            values={key: value for key, value in values.items()
                    if key not in {"source_png", "closed_mask_png"}},
            source_png=bytes(values["source_png"]),
            closed_mask_png=bytes(values["closed_mask_png"]),
        )


def _decimal(value, name: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _endpoint_values(query: dict, defaults: dict) -> dict:
    resolved = {}
    for endpoint in ("large", "small"):
        for field in PROFILE_FIELDS:
            name = f"{endpoint}_{field}"
            raw = query.get(name, [defaults[name]])[-1]
            value = _decimal(raw, name)
            bounds = ENDPOINT_BOUNDS[field]
            if not Decimal(str(bounds["minimum"])) <= value <= Decimal(
                    str(bounds["maximum"])):
                raise ValueError(
                    f"{name} must be within {bounds['minimum']}..{bounds['maximum']}")
            if field in INTEGER_FIELDS:
                integral = value.to_integral_value(rounding=ROUND_HALF_UP)
                if value != integral:
                    raise ValueError(f"{name} must be an integer")
                resolved[name] = int(integral)
            else:
                resolved[name] = float(value.quantize(
                    GAMMA_QUANTUM, rounding=ROUND_HALF_UP))
    return resolved


def resolve_candidate(store: AssetStore, query: dict) -> ResolvedCandidate:
    endpoints = _endpoint_values(query, store.default_endpoints())
    resolved_bins = []
    for item in store.bins:
        t = Decimal(item.bin_index) / Decimal(BIN_COUNT - 1)
        values = {
            "bin_index": item.bin_index,
            "profile_id": item.default_profile_id,
        }
        for field in PROFILE_FIELDS:
            large = Decimal(str(endpoints[f"large_{field}"]))
            small = Decimal(str(endpoints[f"small_{field}"]))
            value = large + (small - large) * t
            values[field] = (int(value.to_integral_value(rounding=ROUND_HALF_UP))
                             if field in INTEGER_FIELDS else
                             float(value.quantize(
                                 GAMMA_QUANTUM, rounding=ROUND_HALF_UP)))
        resolved_bins.append(values)
    canonical = _compact({
        "density_calibration_version":
            DEFAULT_DENSITY_CONFIGURATION.calibration_version,
        "endpoints": endpoints,
    })
    signature = sha256(canonical.encode()).hexdigest()[:20]
    return ResolvedCandidate(
        signature, _compact(endpoints), endpoints, tuple(resolved_bins))


def _decode_png(blob: bytes, mode: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), mode)
    if image is None:
        raise ValueError("asset database contains an invalid PNG")
    return image


def _analysis_source(values: dict, source: np.ndarray) -> np.ndarray:
    shape = (int(values["analysis_height"]), int(values["analysis_width"]))
    if source.shape[:2] == shape:
        return source
    bbox_shape = (int(values["bbox_height"]), int(values["bbox_width"]))
    if source.shape[:2] != bbox_shape:
        raise ValueError("source PNG matches neither bbox nor analysis shape")
    canvas = np.zeros((*shape, 3), np.uint8)
    left = int(values["bbox_x"]) - int(values["image_origin_u"])
    top = int(values["bbox_y"]) - int(values["image_origin_v"])
    if (left < 0 or top < 0 or left + bbox_shape[1] > shape[1]
            or top + bbox_shape[0] > shape[0]):
        raise ValueError("source bbox cannot be placed in its analysis crop")
    canvas[top:top + bbox_shape[0], left:left + bbox_shape[1]] = source
    return canvas


def apply_candidate_morphology(
    persisted_closed_mask: np.ndarray,
    settings: CandidateMorphology,
) -> MorphologyResult:
    """Apply the isolated candidate open/close sequence with zero padding."""
    mask = (np.asarray(persisted_closed_mask) != 0).astype(np.uint8)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("persisted closed component mask must be non-empty and 2D")

    before_area = int(np.count_nonzero(mask))
    before_components = int(cv2.connectedComponents(
        mask, connectivity=8)[0] - 1)
    padding_px = (
        settings.open_kernel_radius_px + settings.close_kernel_radius_px)
    work = np.pad(
        mask,
        padding_px,
        mode="constant",
        constant_values=0,
    ) if padding_px else mask.copy()

    operation_specs = (
        (cv2.MORPH_OPEN, settings.open_kernel_radius_px),
        (cv2.MORPH_CLOSE, settings.close_kernel_radius_px),
    )
    for operation, radius in operation_specs:
        if radius == 0:
            continue
        width = 2 * radius + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (width, width))
        work = cv2.morphologyEx(
            work,
            operation,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if padding_px:
        height, width = mask.shape
        work = work[
            padding_px:padding_px + height,
            padding_px:padding_px + width,
        ]
    candidate_mask = (work != 0).astype(np.uint8)
    after_area = int(np.count_nonzero(candidate_mask))
    if after_area == 0:
        raise ValueError("candidate_mask_empty")
    after_components = int(cv2.connectedComponents(
        candidate_mask, connectivity=8)[0] - 1)

    provenance = {
        "source_mask": "persisted_a250_post_close_component_mask",
        "operation_order": ["MORPH_OPEN", "MORPH_CLOSE"],
        "kernel_shape": "MORPH_RECT",
        "border_type": "BORDER_CONSTANT",
        "border_value": 0,
        "padding_px": padding_px,
        "open_kernel_radius_px": settings.open_kernel_radius_px,
        "open_kernel_width_px": settings.open_kernel_width_px,
        "open_enabled": settings.open_kernel_radius_px != 0,
        "close_kernel_radius_px": settings.close_kernel_radius_px,
        "close_kernel_width_px": settings.close_kernel_width_px,
        "close_enabled": settings.close_kernel_radius_px != 0,
        "foreground_area_before_px": before_area,
        "foreground_area_after_px": after_area,
        "connected_component_count_before": before_components,
        "connected_component_count_after": after_components,
        "split_count": max(0, after_components - before_components),
    }
    return MorphologyResult(candidate_mask, provenance)


class SpectrumService:
    def __init__(self, store: AssetStore) -> None:
        self.store = store

    def resolve(self, query: dict) -> ResolvedCandidate:
        return resolve_candidate(self.store, query)

    @lru_cache(maxsize=512)
    def _render(
        self, asset_id: int, canonical_endpoints_json: str
    ) -> RenderedCandidate:
        endpoints = json.loads(canonical_endpoints_json)
        query = {key: [str(value)] for key, value in endpoints.items()}
        candidate = resolve_candidate(self.store, query)
        asset = self.store.component(asset_id)
        values = asset.values
        bin_index = int(values["bin_index"])
        bin_snapshot = self.store.bin_map.get(bin_index)
        if bin_snapshot is None:
            raise ValueError("component refers to an unknown spectrum bin")
        settings = candidate.settings_for_bin(bin_index)
        source = _decode_png(asset.source_png, cv2.IMREAD_COLOR)
        mask = (_decode_png(
            asset.closed_mask_png, cv2.IMREAD_GRAYSCALE) != 0).astype(np.uint8)
        expected_shape = (
            int(values["analysis_height"]), int(values["analysis_width"]))
        if mask.shape != expected_shape:
            raise ValueError("closed mask does not match stored analysis shape")
        if int(np.count_nonzero(mask)) != int(values["area_px"]):
            raise ValueError("closed mask area differs from component provenance")
        morphology_settings = candidate.morphology_for_bin(bin_index)
        morphology = apply_candidate_morphology(mask, morphology_settings)
        started = perf_counter()
        evidence = compute_density_evidence(
            morphology.mask,
            self.store.profiles,
            bin_snapshot.default_profile_id,
            settings,
        )
        compute_ms = (perf_counter() - started) * 1000.0
        image = vertical_evidence_sheet(
            _analysis_source(values, source), mask, morphology.mask, evidence)
        provenance = {
            "asset_id": asset_id,
            "bin_index": bin_index,
            "run_id": values["run_id"],
            "frame_id": values["frame_id"],
            "sim_time_ns": values["sim_time_ns"],
            "component_id": values["component_id"],
            "production_profile_id": values.get(
                "production_profile_id", values.get("profile_id")),
            "candidate_profile_id": bin_snapshot.default_profile_id,
        }
        return RenderedCandidate(
            png=encode_png(image),
            compute_ms=compute_ms,
            settings={**asdict(settings), **asdict(morphology_settings)},
            morphology=morphology.provenance,
            thresholds={
                "p70_threshold": evidence.p70_threshold,
                "p80_threshold": evidence.p80_threshold,
                "p90_threshold": evidence.p90_threshold,
            },
            provenance=provenance,
            signature=candidate.signature,
        )

    def render(
        self, asset_id: int, candidate: ResolvedCandidate
    ) -> RenderedCandidate:
        return self._render(asset_id, candidate.canonical_endpoints_json)


class SpectrumHandler(BaseHTTPRequestHandler):
    service: SpectrumService

    def _send(
        self,
        body: bytes,
        content_type: str,
        status=HTTPStatus.OK,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, status=HTTPStatus.OK) -> None:
        self._send(
            _compact(payload).encode(), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        request = urlsplit(self.path)
        query = parse_qs(request.query)
        try:
            if request.path == "/api/health":
                return self._json({
                    "status": "ok",
                    "server_contract": "spectrum-calibration.v1",
                    "decoded_asset_cache":
                        self.service.store.component.cache_info()._asdict(),
                    "render_cache": self.service._render.cache_info()._asdict(),
                })
            if request.path == "/api/manifest":
                return self._json(self.service.store.manifest())
            if request.path == "/api/components":
                offset = max(0, int(query.get("offset", ["0"])[-1]))
                limit = min(MAX_PAGE_SIZE, max(
                    1, int(query.get("limit", [str(MAX_PAGE_SIZE)])[-1])))
                return self._json(self.service.store.components(offset, limit))
            if request.path == "/api/resolve":
                return self._json(self.service.resolve(query).payload())
            match = COMPONENT_IMAGE.fullmatch(request.path)
            if match:
                candidate = self.service.resolve(query)
                rendered = self.service.render(int(match.group(1)), candidate)
                return self._send(rendered.png, "image/png", headers={
                    "X-Candidate-Signature": rendered.signature,
                    "X-Bin-Index": str(rendered.provenance["bin_index"]),
                    "X-Candidate-Profile-Id": str(
                        rendered.provenance["candidate_profile_id"]),
                    "X-Density-Calibration-Version":
                        DEFAULT_DENSITY_CONFIGURATION.calibration_version,
                    "X-Resolved-Settings": _compact(rendered.settings),
                    "X-Candidate-Morphology": _compact(rendered.morphology),
                    "X-Candidate-Thresholds": _compact(rendered.thresholds),
                    "X-Component-Provenance": _compact(rendered.provenance),
                    "X-Density-Compute-Ms": f"{rendered.compute_ms:.3f}",
                })
            relative = "index.html" if request.path == "/" else request.path[1:]
            target = (UI_ROOT / relative).resolve()
            if not target.is_relative_to(UI_ROOT) or not target.is_file():
                return self._json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            content_type = mimetypes.guess_type(target.name)[0] or \
                "application/octet-stream"
            return self._send(target.read_bytes(), content_type)
        except (KeyError, ValueError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as error:
            self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except Exception as error:  # expose one-off laboratory failures
            self._json(
                {"error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    SpectrumHandler.service = SpectrumService(AssetStore(args.database))
    server = ThreadingHTTPServer((args.bind, args.port), SpectrumHandler)
    print(
        f"spectrum calibration UI: http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
