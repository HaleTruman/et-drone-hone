"""Build the isolated component-spectrum calibration asset database.

The builder replays exactly three historic runs through production
``preprocess_frame`` with one explicit review-only change: the pre-close
minimum component area is 250 pixels.  Post-close components touching the
frame are discarded.  Every retained component is persisted with an aligned
analysis-square source crop and its exact analysis-square closed mask.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterator
from uuid import uuid4

import cv2
import numpy as np

from ...configurations import DEFAULT_DENSITY_CONFIGURATION
from ...preprocessing import (
    DEFAULT_CONFIG,
    PreprocessingConfig,
    component_mask,
    decode_jpeg,
    load_lut,
    preprocess_frame,
)


HERE = Path(__file__).resolve().parent
FLIGHT_ROOT = Path(__file__).resolve().parents[8]
ASSET_ROOT = HERE / "assets"
DATABASE_PATH = ASSET_ROOT / "spectrum_assets.sqlite3"
MANIFEST_PATH = ASSET_ROOT / "manifest.json"

ASSET_SCHEMA_VERSION = 1
BIN_COUNT = 10
SPECTRUM_PREPROCESSING_VERSION = (
    "deterministic-v3-mask-v3:a250:c0-disabled:k5:conn8:tree"
)
SPECTRUM_PREPROCESSING_CONFIG = replace(
    DEFAULT_CONFIG,
    version=SPECTRUM_PREPROCESSING_VERSION,
    minimum_component_area_px=250,
)

KNOWN_CORPUS_RUN_IDS = (
    "run-20260731T093616Z",
    "run-20260731T093732Z",
    "run-20260801T031401Z",
)
KNOWN_CORPUS_COMPONENT_COUNT = 2_441
SORT_CONTRACT = (
    "area_px DESC, run_id ASC, frame_id ASC, component_id ASC"
)


@dataclass(frozen=True, slots=True)
class HistoricFrameRecord:
    """One path-confined and byte-size-validated live-ingress record."""

    run_id: str
    frame_id: int
    sim_time_ns: int
    jpeg_size: int
    relative_path: str
    source_path: Path

    def jpeg_bytes(self) -> bytes:
        payload = self.source_path.read_bytes()
        if len(payload) != self.jpeg_size:
            raise ValueError(
                f"JPEG size changed for {self.relative_path}: "
                f"{len(payload)} != {self.jpeg_size}"
            )
        return payload


@dataclass(frozen=True, slots=True)
class ComponentAsset:
    """Database-ready retained component before global rank assignment."""

    run_id: str
    frame_id: int
    sim_time_ns: int
    source_relative_path: str
    component_id: int
    production_profile_id: str
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    image_origin_u: int
    image_origin_v: int
    analysis_height: int
    analysis_width: int
    area_px: int
    fill_ratio: float
    solidity: float
    source_png: bytes
    closed_mask_png: bytes

    def sort_key(self) -> tuple[int, str, int, int]:
        return (-self.area_px, self.run_id, self.frame_id, self.component_id)


@dataclass(frozen=True, slots=True)
class BinRecord:
    """One persisted, zero-based descending rank bin."""

    bin_index: int
    rank_start: int
    rank_end: int
    component_count: int
    minimum_area_px: int
    maximum_area_px: int
    median_area_px: float
    default_profile_id: str


def _iter_run(run_dir: Path) -> Iterator[HistoricFrameRecord]:
    """Yield one strictly validated historic frame manifest."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "frames.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"historic frame manifest not found: {manifest_path}")
    if not (run_dir / "vision_frames").is_dir():
        raise FileNotFoundError(
            f"historic vision frame directory not found: {run_dir}")

    seen_ids: set[int] = set()
    previous_time = -1
    yielded = 0
    with manifest_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frame_id = int(record["frame_id"])
                sim_time_ns = int(record["sim_time_ns"])
                jpeg_size = int(record["jpeg_size"])
                relative = Path(str(record["path"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid frame manifest record at line {line_number}"
                ) from error
            if frame_id in seen_ids:
                raise ValueError(f"duplicate frame_id in manifest: {frame_id}")
            if sim_time_ns < previous_time:
                raise ValueError("frame manifest timing is not monotonic")
            if jpeg_size < 1:
                raise ValueError(f"invalid JPEG size at line {line_number}")
            if (relative.is_absolute() or not relative.parts or
                    relative.parts[0] != "vision_frames"):
                raise ValueError(
                    f"frame path must be relative to vision_frames: {relative}")
            source_path = (run_dir / relative).resolve()
            if not source_path.is_relative_to(run_dir):
                raise ValueError(f"frame path escapes run: {relative}")
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if source_path.stat().st_size != jpeg_size:
                raise ValueError(
                    f"JPEG size changed for {relative}: "
                    f"{source_path.stat().st_size} != {jpeg_size}"
                )
            seen_ids.add(frame_id)
            previous_time = sim_time_ns
            yielded += 1
            yield HistoricFrameRecord(
                run_id=run_dir.name,
                frame_id=frame_id,
                sim_time_ns=sim_time_ns,
                jpeg_size=jpeg_size,
                relative_path=relative.as_posix(),
                source_path=source_path,
            )
    if yielded == 0:
        raise ValueError(f"historic frame manifest is empty: {manifest_path}")


def _validated_records(run_dir: Path) -> tuple[HistoricFrameRecord, ...]:
    return tuple(_iter_run(run_dir))


def discover_latest_runs(log_root: Path, count: int = 3) -> tuple[Path, ...]:
    """Return the newest structurally and manifest-valid run directories."""
    if count < 1:
        raise ValueError("run count must be positive")
    valid: list[Path] = []
    for candidate in sorted(log_root.resolve().glob("run-*")):
        if not candidate.is_dir():
            continue
        try:
            _validated_records(candidate)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            continue
        valid.append(candidate.resolve())
    if len(valid) < count:
        raise FileNotFoundError(
            f"expected at least {count} valid runs under {log_root}")
    return tuple(valid[-count:])


def resolve_run_dirs(
    explicit_runs: list[Path] | None,
    *,
    log_root: Path = FLIGHT_ROOT / "logs",
) -> tuple[Path, ...]:
    """Validate exactly three explicit runs or discover the latest three."""
    if explicit_runs is None:
        return discover_latest_runs(log_root, 3)
    if len(explicit_runs) != 3:
        raise ValueError("--run must be supplied exactly three times")
    resolved = tuple(path.resolve() for path in explicit_runs)
    if len(set(resolved)) != 3:
        raise ValueError("the three explicit run directories must be distinct")
    for run_dir in resolved:
        _validated_records(run_dir)
    return resolved


def _profile_for_area(area_px: int):
    for profile in DEFAULT_DENSITY_CONFIGURATION.profiles:
        if (profile.maximum_component_area_px is None or
                area_px <= profile.maximum_component_area_px):
            return profile
    raise AssertionError("density profile deck must end unbounded")


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", image, (cv2.IMWRITE_PNG_COMPRESSION, 3))
    if not ok:
        raise OSError("unable to encode calibration asset")
    return encoded.tobytes()


def _analysis_square_crop(
    image: np.ndarray,
    origin_uv: tuple[int, int],
    shape: tuple[int, int],
) -> np.ndarray:
    """Copy an analysis-aligned BGR crop, padding beyond the frame with black."""
    height, width = (int(shape[0]), int(shape[1]))
    if height < 1 or width < 1:
        raise ValueError(f"invalid analysis shape: {shape}")
    origin_u, origin_v = (int(origin_uv[0]), int(origin_uv[1]))
    output = np.zeros((height, width, 3), dtype=np.uint8)
    image_height, image_width = image.shape[:2]

    source_left = max(origin_u, 0)
    source_top = max(origin_v, 0)
    source_right = min(origin_u + width, image_width)
    source_bottom = min(origin_v + height, image_height)
    if source_left >= source_right or source_top >= source_bottom:
        return output

    destination_left = source_left - origin_u
    destination_top = source_top - origin_v
    destination_right = destination_left + source_right - source_left
    destination_bottom = destination_top + source_bottom - source_top
    output[destination_top:destination_bottom,
           destination_left:destination_right] = image[
               source_top:source_bottom, source_left:source_right]
    return output


def _collect_assets(
    run_dirs: tuple[Path, ...],
) -> tuple[list[ComponentAsset], dict[str, dict[str, int]]]:
    lut = load_lut()
    assets: list[ComponentAsset] = []
    run_counts: dict[str, dict[str, int]] = {}

    for run_dir in run_dirs:
        records = _validated_records(run_dir)
        retained_count = 0
        discarded_count = 0
        post_close_count = 0
        print(f"collecting {run_dir.name}", flush=True)
        for record in records:
            jpeg_bytes = record.jpeg_bytes()
            image = decode_jpeg(jpeg_bytes)
            frame = preprocess_frame(
                frame_id=record.frame_id,
                sim_time_ns=record.sim_time_ns,
                image=image,
                lut=lut,
                config=SPECTRUM_PREPROCESSING_CONFIG,
            )
            post_close_count += len(frame.components)
            for component in frame.components:
                if component.touches_frame:
                    discarded_count += 1
                    continue
                closed = component_mask(frame, component, stage="closed")
                source = _analysis_square_crop(
                    image,
                    component.image_origin_uv,
                    component.analysis_shape,
                )
                if source.shape[:2] != closed.shape:
                    raise AssertionError(
                        "source crop and closed mask analysis shapes differ")
                x, y, width, height = component.bbox_xywh
                profile = _profile_for_area(component.area_px)
                assets.append(ComponentAsset(
                    run_id=record.run_id,
                    frame_id=record.frame_id,
                    sim_time_ns=record.sim_time_ns,
                    source_relative_path=record.relative_path,
                    component_id=component.component_id,
                    production_profile_id=profile.profile_id,
                    bbox_x=x,
                    bbox_y=y,
                    bbox_width=width,
                    bbox_height=height,
                    image_origin_u=component.image_origin_uv[0],
                    image_origin_v=component.image_origin_uv[1],
                    analysis_height=component.analysis_shape[0],
                    analysis_width=component.analysis_shape[1],
                    area_px=component.area_px,
                    fill_ratio=component.fill_ratio,
                    solidity=component.solidity,
                    source_png=_encode_png(source),
                    closed_mask_png=_encode_png(closed * 255),
                ))
                retained_count += 1
        run_counts[run_dir.name] = {
            "frames": len(records),
            "post_close_components": post_close_count,
            "discarded_touches_frame": discarded_count,
            "retained_nonclipped": retained_count,
        }

    assets.sort(key=ComponentAsset.sort_key)
    return assets, run_counts


def _balanced_bin_sizes(total: int, count: int = BIN_COUNT) -> tuple[int, ...]:
    if total < count:
        raise ValueError(f"cannot divide {total} components into {count} bins")
    quotient, remainder = divmod(total, count)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(count)
    )


def _bin_records(
    assets: list[ComponentAsset],
) -> tuple[tuple[BinRecord, ...], tuple[int, ...]]:
    profiles = DEFAULT_DENSITY_CONFIGURATION.profiles
    if len(profiles) != BIN_COUNT:
        raise ValueError(
            f"spectrum calibration requires exactly {BIN_COUNT} profiles")
    sizes = _balanced_bin_sizes(len(assets))
    records: list[BinRecord] = []
    assignments: list[int] = [-1] * len(assets)
    start = 0
    for bin_index, size in enumerate(sizes):
        end = start + size
        part = assets[start:end]
        areas = np.asarray([asset.area_px for asset in part], np.int64)
        assignments[start:end] = [bin_index] * size
        records.append(BinRecord(
            bin_index=bin_index,
            rank_start=start,
            rank_end=end - 1,
            component_count=size,
            minimum_area_px=int(areas.min()),
            maximum_area_px=int(areas.max()),
            median_area_px=float(np.median(areas)),
            default_profile_id=profiles[-1 - bin_index].profile_id,
        ))
        start = end
    if start != len(assets) or any(index < 0 for index in assignments):
        raise AssertionError("rank-balanced assignment did not cover corpus")
    return tuple(records), tuple(assignments)


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA page_size = 4096;
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = FULL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );

        CREATE TABLE source_runs (
            run_ordinal INTEGER NOT NULL UNIQUE,
            run_id TEXT PRIMARY KEY,
            frame_count INTEGER NOT NULL CHECK(frame_count > 0)
        );

        CREATE TABLE profiles (
            profile_ordinal INTEGER NOT NULL UNIQUE,
            profile_id TEXT PRIMARY KEY,
            calibration_version TEXT NOT NULL,
            minimum_component_area_px INTEGER NOT NULL,
            maximum_component_area_px INTEGER,
            density_radius_px INTEGER NOT NULL,
            ridge_radius_px INTEGER NOT NULL,
            relative_cap REAL NOT NULL,
            inverse_gamma REAL NOT NULL,
            ridge_gamma REAL NOT NULL
        );

        CREATE TABLE bins (
            bin_index INTEGER PRIMARY KEY CHECK(bin_index BETWEEN 0 AND 9),
            rank_start INTEGER NOT NULL,
            rank_end INTEGER NOT NULL,
            component_count INTEGER NOT NULL CHECK(component_count > 0),
            minimum_area_px INTEGER NOT NULL,
            maximum_area_px INTEGER NOT NULL,
            median_area_px REAL NOT NULL,
            default_profile_id TEXT NOT NULL REFERENCES profiles(profile_id)
        );

        CREATE TABLE components (
            asset_id INTEGER PRIMARY KEY,
            rank_index INTEGER NOT NULL UNIQUE,
            bin_index INTEGER NOT NULL REFERENCES bins(bin_index),
            run_id TEXT NOT NULL REFERENCES source_runs(run_id),
            frame_id INTEGER NOT NULL,
            sim_time_ns INTEGER NOT NULL,
            source_relative_path TEXT NOT NULL,
            component_id INTEGER NOT NULL,
            production_profile_id TEXT NOT NULL REFERENCES profiles(profile_id),
            bbox_x INTEGER NOT NULL,
            bbox_y INTEGER NOT NULL,
            bbox_width INTEGER NOT NULL CHECK(bbox_width > 0),
            bbox_height INTEGER NOT NULL CHECK(bbox_height > 0),
            image_origin_u INTEGER NOT NULL,
            image_origin_v INTEGER NOT NULL,
            analysis_height INTEGER NOT NULL CHECK(analysis_height > 0),
            analysis_width INTEGER NOT NULL CHECK(analysis_width > 0),
            area_px INTEGER NOT NULL CHECK(area_px > 0),
            fill_ratio REAL NOT NULL,
            solidity REAL NOT NULL,
            touches_frame INTEGER NOT NULL CHECK(touches_frame = 0),
            source_png BLOB NOT NULL,
            closed_mask_png BLOB NOT NULL,
            UNIQUE(run_id, frame_id, component_id)
        );

        CREATE INDEX components_bin_rank
            ON components(bin_index, rank_index);
        CREATE INDEX components_production_profile
            ON components(production_profile_id, rank_index);
        CREATE INDEX components_frame
            ON components(run_id, frame_id, component_id);
    """)


def _insert_profiles(connection: sqlite3.Connection) -> None:
    minimum = 1
    for ordinal, profile in enumerate(
            DEFAULT_DENSITY_CONFIGURATION.profiles):
        connection.execute(
            """INSERT INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ordinal,
                profile.profile_id,
                profile.calibration_version,
                minimum,
                profile.maximum_component_area_px,
                profile.density_radius_px,
                profile.ridge_radius_px,
                profile.relative_cap,
                profile.inverse_gamma,
                profile.ridge_gamma,
            ),
        )
        if profile.maximum_component_area_px is not None:
            minimum = profile.maximum_component_area_px + 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_temporary(path: Path, payload: dict) -> Path:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.building")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def validate_database_snapshot(
    database_path: Path,
    manifest: dict,
) -> dict:
    """Validate hashes, relational invariants, ranking, and every PNG shape."""
    snapshot = manifest["database_snapshot"]
    if database_path.stat().st_size != int(snapshot["size_bytes"]):
        raise ValueError("database byte size does not match manifest")
    digest = _sha256(database_path)
    if digest != snapshot["sha256"]:
        raise ValueError("database SHA-256 does not match manifest")

    uri = f"file:{database_path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"SQLite integrity check failed: {integrity}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise ValueError(f"SQLite foreign-key violations: {foreign_keys}")

        expected_count = int(manifest["counts"]["retained_nonclipped"])
        actual_count = connection.execute(
            "SELECT COUNT(*) FROM components").fetchone()[0]
        if actual_count != expected_count:
            raise ValueError(
                f"component count mismatch: {actual_count} != {expected_count}")
        if connection.execute("SELECT COUNT(*) FROM bins").fetchone()[0] != 10:
            raise ValueError("database must contain ten bins")
        if connection.execute(
                "SELECT COUNT(*) FROM profiles").fetchone()[0] != 10:
            raise ValueError("database must contain ten profile snapshots")
        if connection.execute(
                "SELECT COUNT(*) FROM source_runs").fetchone()[0] != 3:
            raise ValueError("database must contain three source runs")

        ordered = connection.execute("""
            SELECT rank_index, bin_index, area_px, run_id, frame_id,
                   component_id, analysis_height, analysis_width,
                   source_png, closed_mask_png
            FROM components
            ORDER BY area_px DESC, run_id ASC, frame_id ASC, component_id ASC
        """).fetchall()
        expected_bins = {
            int(record["bin_index"]): record for record in manifest["bins"]}
        for expected_rank, row in enumerate(ordered):
            (rank_index, bin_index, _area, _run_id, _frame_id, _component_id,
             analysis_height, analysis_width, source_png,
             closed_mask_png) = row
            if rank_index != expected_rank:
                raise ValueError("persisted rank violates deterministic sort")
            bin_record = expected_bins[bin_index]
            if not (bin_record["rank_start"] <= rank_index <=
                    bin_record["rank_end"]):
                raise ValueError("component rank is outside its persisted bin")
            source = cv2.imdecode(
                np.frombuffer(source_png, np.uint8), cv2.IMREAD_COLOR)
            mask = cv2.imdecode(
                np.frombuffer(closed_mask_png, np.uint8),
                cv2.IMREAD_GRAYSCALE,
            )
            shape = (analysis_height, analysis_width)
            if source is None or source.shape[:2] != shape:
                raise ValueError("source PNG does not match analysis shape")
            if mask is None or mask.shape != shape:
                raise ValueError("closed-mask PNG does not match analysis shape")
            if not np.all((mask == 0) | (mask == 255)):
                raise ValueError("closed-mask PNG is not binary")

        for bin_index, record in expected_bins.items():
            areas = np.asarray([
                row[0] for row in connection.execute(
                    "SELECT area_px FROM components "
                    "WHERE bin_index = ? ORDER BY rank_index",
                    (bin_index,),
                )
            ], np.int64)
            measured = {
                "component_count": int(areas.size),
                "minimum_area_px": int(areas.min()),
                "maximum_area_px": int(areas.max()),
                "median_area_px": float(np.median(areas)),
            }
            for field, value in measured.items():
                if value != record[field]:
                    raise ValueError(
                        f"bin {bin_index} {field} mismatch: "
                        f"{value} != {record[field]}")
    finally:
        connection.close()
    return {
        "passed": True,
        "components": expected_count,
        "sha256": digest,
    }


def build_database(
    run_dirs: tuple[Path, ...],
    *,
    database_path: Path = DATABASE_PATH,
    manifest_path: Path = MANIFEST_PATH,
    expected_component_count: int | None = None,
) -> dict:
    """Build, fully validate, and atomically publish one asset snapshot."""
    if len(run_dirs) != 3 or len({path.resolve() for path in run_dirs}) != 3:
        raise ValueError("spectrum calibration requires three distinct runs")
    run_dirs = tuple(path.resolve() for path in run_dirs)
    run_ids = tuple(path.name for path in run_dirs)
    assets, run_counts = _collect_assets(run_dirs)
    if (expected_component_count is None and
            tuple(sorted(run_ids)) == KNOWN_CORPUS_RUN_IDS):
        expected_component_count = KNOWN_CORPUS_COMPONENT_COUNT
    if (expected_component_count is not None and
            len(assets) != expected_component_count):
        raise ValueError(
            f"retained component count {len(assets)} does not match expected "
            f"{expected_component_count}")
    bins, assignments = _bin_records(assets)

    database_path = database_path.resolve()
    manifest_path = manifest_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    generation_id = uuid4().hex
    temporary_database = database_path.with_name(
        f".{database_path.name}.{generation_id}.building")
    temporary_manifest: Path | None = None

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest_core = {
        "asset_schema_version": ASSET_SCHEMA_VERSION,
        "generation_id": generation_id,
        "generated_at_utc": generated_at,
        "purpose": "temporary_component_spectrum_calibration_only",
        "source_runs": list(run_ids),
        "preprocessing": {
            "config": asdict(SPECTRUM_PREPROCESSING_CONFIG),
            "minimum_component_area_stage": "pre_close",
            "post_close_touches_frame_policy": "discard",
        },
        "density_configuration": asdict(DEFAULT_DENSITY_CONFIGURATION),
        "sort": {
            "contract": SORT_CONTRACT,
            "rank_index_base": 0,
            "bin_index_base": 0,
            "bin_order": "largest_area_to_smallest_area",
            "binning": "balanced_contiguous_rank_partition",
            "equal_area_ties": "run_id_then_frame_id_then_component_id",
        },
        "counts": {
            "frames": sum(item["frames"] for item in run_counts.values()),
            "post_close_components": sum(
                item["post_close_components"] for item in run_counts.values()),
            "discarded_touches_frame": sum(
                item["discarded_touches_frame"] for item in run_counts.values()),
            "retained_nonclipped": len(assets),
            "by_run": run_counts,
        },
        "bins": [asdict(record) for record in bins],
    }

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary_database)
        _initialize_database(connection)
        for ordinal, run_id in enumerate(run_ids):
            connection.execute(
                "INSERT INTO source_runs VALUES (?, ?, ?)",
                (ordinal, run_id, run_counts[run_id]["frames"]),
            )
        _insert_profiles(connection)
        for record in bins:
            connection.execute(
                "INSERT INTO bins VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.bin_index,
                    record.rank_start,
                    record.rank_end,
                    record.component_count,
                    record.minimum_area_px,
                    record.maximum_area_px,
                    record.median_area_px,
                    record.default_profile_id,
                ),
            )
        for rank_index, (asset, bin_index) in enumerate(
                zip(assets, assignments, strict=True)):
            connection.execute(
                """INSERT INTO components (
                    rank_index, bin_index, run_id, frame_id, sim_time_ns,
                    source_relative_path, component_id, production_profile_id,
                    bbox_x, bbox_y, bbox_width, bbox_height, image_origin_u,
                    image_origin_v, analysis_height, analysis_width, area_px,
                    fill_ratio, solidity, touches_frame, source_png,
                    closed_mask_png
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    0, ?, ?
                )""",
                (
                    rank_index,
                    bin_index,
                    asset.run_id,
                    asset.frame_id,
                    asset.sim_time_ns,
                    asset.source_relative_path,
                    asset.component_id,
                    asset.production_profile_id,
                    asset.bbox_x,
                    asset.bbox_y,
                    asset.bbox_width,
                    asset.bbox_height,
                    asset.image_origin_u,
                    asset.image_origin_v,
                    asset.analysis_height,
                    asset.analysis_width,
                    asset.area_px,
                    asset.fill_ratio,
                    asset.solidity,
                    sqlite3.Binary(asset.source_png),
                    sqlite3.Binary(asset.closed_mask_png),
                ),
            )
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("manifest", json.dumps(manifest_core, separators=(",", ":"))),
        )
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        connection = None

        manifest = {
            **manifest_core,
            "database_snapshot": {
                "filename": database_path.name,
                "size_bytes": temporary_database.stat().st_size,
                "sha256": _sha256(temporary_database),
            },
        }
        validate_database_snapshot(temporary_database, manifest)
        temporary_manifest = _atomic_json_temporary(manifest_path, manifest)

        temporary_database.replace(database_path)
        temporary_manifest.replace(manifest_path)
        temporary_manifest = None
        return manifest
    finally:
        if connection is not None:
            connection.close()
        if temporary_database.exists():
            temporary_database.unlink()
        if temporary_manifest is not None and temporary_manifest.exists():
            temporary_manifest.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        help="explicit run directory; provide exactly three times",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=FLIGHT_ROOT / "logs",
        help="root searched when --run is omitted",
    )
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--expected-components",
        type=int,
        help="optional retained-component count assertion",
    )
    args = parser.parse_args(argv)
    if args.expected_components is not None and args.expected_components < 1:
        parser.error("--expected-components must be positive")
    try:
        run_dirs = resolve_run_dirs(args.run, log_root=args.log_root)
        manifest = build_database(
            run_dirs,
            database_path=args.database,
            manifest_path=args.manifest,
            expected_component_count=args.expected_components,
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest["counts"], indent=2))
    print(args.database.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
