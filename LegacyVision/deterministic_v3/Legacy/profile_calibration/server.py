"""Serve the isolated profile-calibration asset browser and live recompute API."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
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

from ..configurations import DEFAULT_DENSITY_CONFIGURATION
from ..preprocessing import DEFAULT_CONFIG
from ..schema import DensityProfile
from .build_assets import DATABASE_PATH
from .density_runtime import (
    CandidateSettings,
    comparison_image,
    compute_density_evidence,
)


HERE = Path(__file__).resolve().parent
UI_ROOT = HERE / "ui"
COMPONENT_IMAGE = re.compile(r"^/api/component/(\d+)/comparison\.png$")
MAX_PAGE_SIZE = 24


class AssetStore:
    def __init__(self, database: Path) -> None:
        self.database = database.resolve()
        if not self.database.is_file():
            raise FileNotFoundError(
                f"asset database not found: {self.database}; run build_assets first")
        self.profiles = self._profiles()
        self.profile_map = {
            profile.profile_id: profile for profile in self.profiles}
        self._validate_snapshot()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.database.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _profiles(self) -> tuple[DensityProfile, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM profiles ORDER BY ordinal").fetchall()
        return tuple(DensityProfile(
            profile_id=row["profile_id"],
            calibration_version=row["calibration_version"],
            maximum_component_area_px=row["maximum_component_area_px"],
            density_radius_px=row["density_radius_px"],
            ridge_radius_px=row["ridge_radius_px"],
            relative_cap=row["relative_cap"],
            inverse_gamma=row["inverse_gamma"],
            ridge_gamma=row["ridge_gamma"],
        ) for row in rows)

    def manifest(self) -> dict:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value_json FROM metadata WHERE key = 'manifest'"
            ).fetchone()
            profile_rows = connection.execute(
                "SELECT * FROM profiles ORDER BY ordinal").fetchall()
        if row is None:
            raise ValueError("asset database has no manifest")
        payload = json.loads(row["value_json"])
        payload["profiles"] = [dict(item) for item in profile_rows]
        payload["control_bounds"] = {
            "density_radius_px": {"minimum": 1, "maximum": 24, "step": 1},
            "ridge_radius_px": {"minimum": 1, "maximum": 20, "step": 1},
            "inverse_gamma": {
                "minimum": 0.01,
                "maximum": 100.0,
                "step": 0.01,
            },
            "ridge_gamma": {
                "minimum": 0.01,
                "maximum": 100.0,
                "step": 0.01,
            },
        }
        return payload

    def _validate_snapshot(self) -> None:
        """Fail visibly instead of mixing stale assets with current math."""
        payload = self.manifest()
        active_configuration = json.loads(json.dumps(
            asdict(DEFAULT_DENSITY_CONFIGURATION), separators=(",", ":")))
        if payload.get("density_configuration") != active_configuration:
            raise ValueError(
                "asset density configuration differs from active production; "
                "rebuild the isolated asset database")
        if payload.get("preprocessing_version") != DEFAULT_CONFIG.version:
            raise ValueError(
                "asset preprocessing version differs from active production; "
                "rebuild the isolated asset database")

    def components(
        self, profile_id: str, include_clipped: bool, offset: int, limit: int
    ) -> dict:
        if profile_id not in self.profile_map:
            raise KeyError(f"unknown profile: {profile_id}")
        eligibility = "" if include_clipped else "AND density_eligible = 1"
        parameters = (profile_id,)
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM components "
                f"WHERE profile_id = ? {eligibility}", parameters,
            ).fetchone()[0]
            rows = connection.execute(
                f"""SELECT
                    asset_id, run_id, frame_id, sim_time_ns,
                    source_relative_path, component_id, profile_id,
                    bbox_x, bbox_y, bbox_width, bbox_height,
                    image_origin_u, image_origin_v,
                    analysis_height, analysis_width, area_px, fill_ratio,
                    solidity, touches_frame, density_eligible
                FROM components
                WHERE profile_id = ? {eligibility}
                ORDER BY area_px, run_id, frame_id, component_id
                LIMIT ? OFFSET ?""",
                (*parameters, limit, offset),
            ).fetchall()
        return {
            "profile_id": profile_id,
            "include_frame_edge_clipped": include_clipped,
            "total": total,
            "offset": offset,
            "limit": limit,
            "records": [dict(row) for row in rows],
        }

    def component(self, asset_id: int) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM components WHERE asset_id = ?", (asset_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown asset: {asset_id}")
        return row

    @lru_cache(maxsize=24)
    def baseline_asset(self, asset_id: int):
        """Cache immutable baseline work across repeated slider requests."""
        record = self.component(asset_id)
        source = _decode_png(record["source_png"], cv2.IMREAD_COLOR)
        mask = (_decode_png(
            record["closed_mask_png"], cv2.IMREAD_GRAYSCALE) != 0
        ).astype(np.uint8)
        profile = self.profile_map[record["profile_id"]]
        baseline = (compute_density_evidence(
            mask, self.profiles, profile.profile_id)
            if record["density_eligible"] else None)
        return record, source, mask, profile, baseline


def _decode_png(blob: bytes, mode: int) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(blob, np.uint8), mode)
    if image is None:
        raise ValueError("invalid PNG in calibration asset database")
    return image


def _number(query: dict, name: str, default, convert):
    values = query.get(name)
    return default if not values else convert(values[-1])


def _candidate(query: dict, profile: DensityProfile) -> CandidateSettings:
    settings = CandidateSettings(
        density_radius_px=_number(
            query, "density_radius_px", profile.density_radius_px, int),
        ridge_radius_px=_number(
            query, "ridge_radius_px", profile.ridge_radius_px, int),
        relative_cap=profile.relative_cap,
        inverse_gamma=_number(
            query, "inverse_gamma", profile.inverse_gamma, float),
        ridge_gamma=_number(
            query, "ridge_gamma", profile.ridge_gamma, float),
    ).validate()
    bounds = {
        "density_radius_px": (1, 24),
        "ridge_radius_px": (1, 20),
        "inverse_gamma": (0.01, 100.0),
        "ridge_gamma": (0.01, 100.0),
    }
    for name, (minimum, maximum) in bounds.items():
        value = getattr(settings, name)
        if not minimum <= value <= maximum:
            raise ValueError(f"{name} must be within {minimum}..{maximum}")
    return settings


def _comparison(store: AssetStore, asset_id: int, query: dict):
    record, source, mask, profile, baseline = store.baseline_asset(asset_id)
    settings = _candidate(query, profile)
    started = perf_counter()
    if record["density_eligible"]:
        candidate = compute_density_evidence(
            mask, store.profiles, profile.profile_id, settings)
    else:
        candidate = None
    image = comparison_image(source, mask, baseline, candidate)
    compute_ms = 1000.0 * (perf_counter() - started)
    ok, encoded = cv2.imencode(
        ".png", image, (cv2.IMWRITE_PNG_COMPRESSION, 2))
    if not ok:
        raise OSError("unable to encode comparison image")
    thresholds = {} if candidate is None else {
        "p70": candidate.p70_threshold,
        "p80": candidate.p80_threshold,
        "p90": candidate.p90_threshold,
    }
    return encoded.tobytes(), compute_ms, thresholds, asdict(settings)


class CalibrationHandler(BaseHTTPRequestHandler):
    store: AssetStore

    def _send(self, body: bytes, content_type: str, status=HTTPStatus.OK,
              headers: dict[str, str] | None = None) -> None:
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
            json.dumps(payload, separators=(",", ":")).encode(),
            "application/json; charset=utf-8",
            status,
        )

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        request = urlsplit(self.path)
        query = parse_qs(request.query)
        try:
            if request.path == "/api/health":
                return self._json({"status": "ok"})
            if request.path == "/api/manifest":
                return self._json(self.store.manifest())
            if request.path == "/api/components":
                profile_id = query.get("profile_id", [""])[-1]
                include_clipped = query.get("include_clipped", ["0"])[-1] == "1"
                offset = max(0, int(query.get("offset", ["0"])[-1]))
                limit = min(
                    MAX_PAGE_SIZE,
                    max(1, int(query.get("limit", ["12"])[-1])),
                )
                return self._json(self.store.components(
                    profile_id, include_clipped, offset, limit))
            match = COMPONENT_IMAGE.fullmatch(request.path)
            if match:
                body, compute_ms, thresholds, settings = _comparison(
                    self.store, int(match.group(1)), query)
                return self._send(body, "image/png", headers={
                    "X-Profile-Compute-Ms": f"{compute_ms:.3f}",
                    "X-Candidate-Thresholds": json.dumps(
                        thresholds, separators=(",", ":")),
                    "X-Candidate-Settings": json.dumps(
                        settings, separators=(",", ":")),
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
        except Exception as error:  # keep one-off server failures visible
            self._json(
                {"error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8784)
    args = parser.parse_args()
    CalibrationHandler.store = AssetStore(args.database)
    server = ThreadingHTTPServer((args.bind, args.port), CalibrationHandler)
    print(f"profile calibration UI: http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
