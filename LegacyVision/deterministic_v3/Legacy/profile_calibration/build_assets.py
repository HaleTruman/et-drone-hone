"""Build a self-contained bounding-box asset database for profile review."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import cv2

from ..configurations import DEFAULT_DENSITY_CONFIGURATION
from ..preprocessing import (
    DEFAULT_CONFIG,
    component_mask,
    decode_jpeg,
    load_lut,
    preprocess_frame,
)
from .historic_source import HistoricRunSource


HERE = Path(__file__).resolve().parent
FLIGHT_ROOT = Path(__file__).resolve().parents[7]
ASSET_ROOT = HERE / "assets"
DATABASE_PATH = ASSET_ROOT / "profile_assets.sqlite3"
MANIFEST_PATH = ASSET_ROOT / "manifest.json"
ASSET_SCHEMA_VERSION = 1


def discover_latest_runs(log_root: Path, count: int = 3) -> tuple[Path, ...]:
    runs = sorted(
        path.resolve()
        for path in log_root.glob("run-*")
        if (path / "frames.jsonl").is_file()
        and (path / "vision_frames").is_dir()
    )
    if len(runs) < count:
        raise FileNotFoundError(
            f"expected at least {count} valid runs under {log_root}")
    return tuple(runs[-count:])


def _profile_for_area(area_px: int):
    for profile in DEFAULT_DENSITY_CONFIGURATION.profiles:
        if (profile.maximum_component_area_px is None or
                area_px <= profile.maximum_component_area_px):
            return profile
    raise AssertionError("density profile deck must end unbounded")


def _png(image) -> bytes:
    ok, encoded = cv2.imencode(
        ".png", image, (cv2.IMWRITE_PNG_COMPRESSION, 3))
    if not ok:
        raise OSError("unable to encode calibration asset")
    return encoded.tobytes()


def _initialize(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE profiles (
            ordinal INTEGER NOT NULL,
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
        CREATE TABLE components (
            asset_id INTEGER PRIMARY KEY,
            run_id TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            sim_time_ns INTEGER NOT NULL,
            source_relative_path TEXT NOT NULL,
            component_id INTEGER NOT NULL,
            profile_id TEXT NOT NULL REFERENCES profiles(profile_id),
            bbox_x INTEGER NOT NULL,
            bbox_y INTEGER NOT NULL,
            bbox_width INTEGER NOT NULL,
            bbox_height INTEGER NOT NULL,
            image_origin_u INTEGER NOT NULL,
            image_origin_v INTEGER NOT NULL,
            analysis_height INTEGER NOT NULL,
            analysis_width INTEGER NOT NULL,
            area_px INTEGER NOT NULL,
            fill_ratio REAL NOT NULL,
            solidity REAL NOT NULL,
            touches_frame INTEGER NOT NULL,
            density_eligible INTEGER NOT NULL,
            source_png BLOB NOT NULL,
            closed_mask_png BLOB NOT NULL,
            UNIQUE(run_id, frame_id, component_id)
        );
        CREATE INDEX components_profile_eligible
            ON components(profile_id, density_eligible, asset_id);
    """)


def _insert_profiles(connection: sqlite3.Connection) -> None:
    minimum = 1
    for ordinal, profile in enumerate(
            DEFAULT_DENSITY_CONFIGURATION.profiles, 1):
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


def build_database(run_dirs: tuple[Path, ...]) -> dict:
    if len(run_dirs) != 3:
        raise ValueError("profile calibration requires exactly three runs")
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = ASSET_ROOT / ".profile_assets.building.sqlite3"
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(temporary)
    _initialize(connection)
    _insert_profiles(connection)
    lut = load_lut()
    total_frames = 0
    total_components = 0
    run_counts: dict[str, dict[str, int]] = {}
    profile_counts = Counter()
    eligible_profile_counts = Counter()

    try:
        for run_dir in run_dirs:
            frames = components = eligible = 0
            print(f"building assets from {run_dir.name}", flush=True)
            for record in HistoricRunSource(run_dir):
                payload = record.pipeline_input()
                image = decode_jpeg(payload["jpeg_bytes"])
                frame = preprocess_frame(
                    frame_id=record.frame_id,
                    sim_time_ns=record.sim_time_ns,
                    image=image,
                    lut=lut,
                )
                frames += 1
                total_frames += 1
                for component in frame.components:
                    x, y, width, height = component.bbox_xywh
                    profile = _profile_for_area(component.area_px)
                    closed = component_mask(frame, component, stage="closed")
                    source_crop = image[y:y + height, x:x + width]
                    density_eligible = not (
                        DEFAULT_DENSITY_CONFIGURATION.ignore_frame_edge_clipped
                        and component.touches_frame)
                    connection.execute(
                        """INSERT INTO components (
                            run_id, frame_id, sim_time_ns, source_relative_path,
                            component_id, profile_id, bbox_x, bbox_y, bbox_width,
                            bbox_height, image_origin_u, image_origin_v,
                            analysis_height, analysis_width, area_px, fill_ratio,
                            solidity, touches_frame, density_eligible, source_png,
                            closed_mask_png
                        ) VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?
                        )""",
                        (
                            record.run_id,
                            record.frame_id,
                            record.sim_time_ns,
                            record.relative_path,
                            component.component_id,
                            profile.profile_id,
                            x, y, width, height,
                            component.image_origin_uv[0],
                            component.image_origin_uv[1],
                            component.analysis_shape[0],
                            component.analysis_shape[1],
                            component.area_px,
                            component.fill_ratio,
                            component.solidity,
                            int(component.touches_frame),
                            int(density_eligible),
                            sqlite3.Binary(_png(source_crop)),
                            sqlite3.Binary(_png(closed * 255)),
                        ),
                    )
                    components += 1
                    total_components += 1
                    profile_counts[profile.profile_id] += 1
                    if density_eligible:
                        eligible += 1
                        eligible_profile_counts[profile.profile_id] += 1
                if total_frames % 100 == 0:
                    connection.commit()
                    print(
                        f"frames={total_frames} components={total_components}",
                        flush=True,
                    )
            run_counts[run_dir.name] = {
                "frames": frames,
                "components": components,
                "density_eligible": eligible,
                "frame_edge_clipped": components - eligible,
            }

        manifest = {
            "asset_schema_version": ASSET_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "temporary_profile_calibration_only",
            "source_runs": [path.name for path in run_dirs],
            "preprocessing_version": DEFAULT_CONFIG.version,
            "density_configuration": asdict(DEFAULT_DENSITY_CONFIGURATION),
            "counts": {
                "frames": total_frames,
                "components": total_components,
                "density_eligible": sum(eligible_profile_counts.values()),
                "frame_edge_clipped": (
                    total_components - sum(eligible_profile_counts.values())),
                "by_run": run_counts,
                "by_profile": {
                    profile.profile_id: {
                        "all": profile_counts[profile.profile_id],
                        "density_eligible": eligible_profile_counts[
                            profile.profile_id],
                        "frame_edge_clipped": (
                            profile_counts[profile.profile_id] -
                            eligible_profile_counts[profile.profile_id]),
                    }
                    for profile in DEFAULT_DENSITY_CONFIGURATION.profiles
                },
            },
        }
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?)",
            ("manifest", json.dumps(manifest, separators=(",", ":"))),
        )
        connection.commit()
    finally:
        connection.close()

    temporary.replace(DATABASE_PATH)
    manifest_temporary = ASSET_ROOT / ".manifest.building.json"
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_temporary.replace(MANIFEST_PATH)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="append", type=Path,
        help="Explicit run directory; repeat exactly three times")
    args = parser.parse_args()
    run_dirs = (tuple(path.resolve() for path in args.run)
                if args.run else discover_latest_runs(FLIGHT_ROOT / "logs"))
    manifest = build_database(run_dirs)
    print(json.dumps(manifest["counts"], indent=2))
    print(DATABASE_PATH)


if __name__ == "__main__":
    main()
