"""Serve the isolated whole-frame local-thickness discovery UI."""

from __future__ import annotations

import argparse
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlsplit

import cv2

from ...configurations import DEFAULT_DENSITY_CONFIGURATION
from ...preprocessing import decode_jpeg, load_lut, preprocess_frame
from ..build_assets import discover_latest_runs
from ..historic_source import HistoricFrameRecord, HistoricRunSource
from .analysis import ASSIGNMENT_MODES, analyze_frame, render_analysis


HERE = Path(__file__).resolve().parent
UI_ROOT = HERE / "ui"
FLIGHT_ROOT = Path(__file__).resolve().parents[8]
DEFAULT_RUN_ID = "run-20260801T031401Z"
DEFAULT_FRAME_ID = 106765


class ThicknessService:
    def __init__(self, run_dirs: tuple[Path, ...]) -> None:
        self.records_by_run = {
            path.name: tuple(HistoricRunSource(path)) for path in run_dirs}
        self.record_lookup = {
            (run_id, record.frame_id): record
            for run_id, records in self.records_by_run.items()
            for record in records
        }
        self.lut = load_lut()

    def catalog(self) -> dict:
        run_ids = tuple(self.records_by_run)
        default_run = (DEFAULT_RUN_ID if DEFAULT_RUN_ID in self.records_by_run
                       else run_ids[-1])
        default_frame = DEFAULT_FRAME_ID
        if (default_run, default_frame) not in self.record_lookup:
            default_frame = self.records_by_run[default_run][0].frame_id
        return {
            "purpose": "temporary_local_thickness_discovery_only",
            "runs": {
                run_id: [{
                    "frame_id": record.frame_id,
                    "sim_time_ns": record.sim_time_ns,
                    "source_relative_path": record.relative_path,
                } for record in records]
                for run_id, records in self.records_by_run.items()
            },
            "default": {
                "run_id": default_run,
                "frame_id": default_frame,
                "assignment_mode": "equivalent_area_thickness",
                "minimum_support_ratio": 1.0,
                "distance_ridge_smoothing_sigma_px": 0.0,
                "display_heat_cap_percentile": 99.0,
                "include_clipped_components": True,
                "show_overlap_candidate_flags": True,
            },
            "assignment_modes": [{
                "value": "equivalent_area_thickness",
                "label": "Regional thickness -> equivalent area (recommended)",
            }, {
                "value": "calibrated_thickness_bands",
                "label": "Direct empirical thickness band",
            }, {
                "value": "l2_covering_disk",
                "label": "Legacy L2 radius support",
            }, {
                "value": "square_window",
                "label": "Legacy square-window support",
            }],
            "control_bounds": {
                "minimum_support_ratio": {
                    "minimum": 0.5, "maximum": 2.0, "step": 0.05},
                "distance_ridge_smoothing_sigma_px": {
                    "minimum": 0.0, "maximum": 3.0, "step": 0.25},
                "display_heat_cap_percentile": {
                    "minimum": 90.0, "maximum": 100.0, "step": 0.5},
            },
            "profiles": [{
                "profile_id": profile.profile_id,
                "density_radius_px": profile.density_radius_px,
                "ridge_radius_px": profile.ridge_radius_px,
                "inverse_gamma": profile.inverse_gamma,
                "ridge_gamma": profile.ridge_gamma,
            } for profile in DEFAULT_DENSITY_CONFIGURATION.profiles],
        }

    def record(self, run_id: str, frame_id: int) -> HistoricFrameRecord:
        try:
            return self.record_lookup[(run_id, frame_id)]
        except KeyError as error:
            raise KeyError(f"unknown recorded frame: {run_id}/{frame_id}") \
                from error

    @lru_cache(maxsize=8)
    def _frame(self, run_id: str, frame_id: int):
        record = self.record(run_id, frame_id)
        image = decode_jpeg(record.pipeline_input()["jpeg_bytes"])
        frame = preprocess_frame(
            frame_id=record.frame_id,
            sim_time_ns=record.sim_time_ns,
            image=image,
            lut=self.lut,
        )
        return record, image, frame

    @lru_cache(maxsize=24)
    def render(
        self,
        run_id: str,
        frame_id: int,
        assignment_mode: str,
        minimum_support_ratio: float,
        smoothing_sigma_px: float,
        heat_cap_percentile: float,
        include_clipped_components: bool,
        show_overlap_candidate_flags: bool,
    ) -> tuple[bytes, dict]:
        started = perf_counter()
        record, image, frame = self._frame(run_id, frame_id)
        analysis = analyze_frame(
            frame,
            assignment_mode=assignment_mode,
            minimum_support_ratio=minimum_support_ratio,
            smoothing_sigma_px=smoothing_sigma_px,
            include_clipped_components=include_clipped_components,
        )
        rendered = render_analysis(
            image,
            frame,
            analysis,
            source_label=f"{run_id}/{record.relative_path}",
            heat_cap_percentile=heat_cap_percentile,
            show_overlap_candidate_flags=show_overlap_candidate_flags,
        )
        ok, encoded = cv2.imencode(
            ".png", rendered, (cv2.IMWRITE_PNG_COMPRESSION, 2))
        if not ok:
            raise OSError("unable to encode local-thickness analysis")
        metrics = dict(analysis.metrics)
        metrics.pop("component_thickness_evidence", None)
        overlap_candidates = list(metrics.get("overlap_candidates", ()))
        metrics.update({
            "run_id": run_id,
            "frame_id": frame.frame_id,
            "sim_time_ns": frame.sim_time_ns,
            "preprocessing_version": frame.preprocessing_version,
            "display_heat_cap_percentile": heat_cap_percentile,
            "show_overlap_candidate_flags": show_overlap_candidate_flags,
            "component_evidence_count": len(analysis.component_evidence),
            "overlap_candidate_count": int(metrics.get(
                "overlap_candidate_count", len(overlap_candidates))),
            "overlap_candidates": overlap_candidates,
            "compute_ms": 1000.0 * (perf_counter() - started),
        })
        return encoded.tobytes(), metrics


def _last(query: dict, name: str, default):
    values = query.get(name)
    return default if not values else values[-1]


def _parameters(query: dict, catalog: dict) -> tuple:
    defaults = catalog["default"]
    run_id = str(_last(query, "run_id", defaults["run_id"]))
    frame_id = int(_last(query, "frame_id", defaults["frame_id"]))
    mode = str(_last(
        query, "assignment_mode", defaults["assignment_mode"]))
    if mode not in ASSIGNMENT_MODES:
        raise ValueError(f"unknown assignment mode: {mode}")
    values = {
        "minimum_support_ratio": float(_last(
            query, "minimum_support_ratio",
            defaults["minimum_support_ratio"])),
        "distance_ridge_smoothing_sigma_px": float(_last(
            query, "distance_ridge_smoothing_sigma_px",
            defaults["distance_ridge_smoothing_sigma_px"])),
        "display_heat_cap_percentile": float(_last(
            query, "display_heat_cap_percentile",
            defaults["display_heat_cap_percentile"])),
    }
    for name, value in values.items():
        bounds = catalog["control_bounds"][name]
        if not bounds["minimum"] <= value <= bounds["maximum"]:
            raise ValueError(
                f"{name} must be within "
                f"{bounds['minimum']}..{bounds['maximum']}")
    def boolean(name: str) -> bool:
        text = str(_last(
            query, name, "1" if defaults[name] else "0")).lower()
        if text not in {"0", "1", "false", "true"}:
            raise ValueError(f"{name} must be 0/1 or false/true")
        return text in {"1", "true"}

    include_clipped_components = boolean("include_clipped_components")
    show_overlap_candidate_flags = boolean("show_overlap_candidate_flags")
    return (
        run_id,
        frame_id,
        mode,
        round(values["minimum_support_ratio"], 4),
        round(values["distance_ridge_smoothing_sigma_px"], 4),
        round(values["display_heat_cap_percentile"], 4),
        include_clipped_components,
        show_overlap_candidate_flags,
    )


class ThicknessHandler(BaseHTTPRequestHandler):
    service: ThicknessService

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
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Browser-side AbortController cancels stale frame renders while
            # the review computation is finishing.  The result is disposable.
            return

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
            if request.path == "/api/catalog":
                return self._json(self.service.catalog())
            if request.path == "/api/analysis.png":
                parameters = _parameters(query, self.service.catalog())
                body, metrics = self.service.render(*parameters)
                return self._send(body, "image/png", headers={
                    "X-Analysis-Ms": f"{metrics['compute_ms']:.3f}",
                    "X-Thickness-Metrics": json.dumps(
                        metrics, separators=(",", ":")),
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
        except Exception as error:
            self._json(
                {"error": f"{type(error).__name__}: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8785)
    parser.add_argument(
        "--run", action="append", type=Path,
        help="Explicit run directory; defaults to the latest three")
    args = parser.parse_args()
    run_dirs = (tuple(path.resolve() for path in args.run)
                if args.run else discover_latest_runs(FLIGHT_ROOT / "logs"))
    ThicknessHandler.service = ThicknessService(run_dirs)
    server = ThreadingHTTPServer((args.bind, args.port), ThicknessHandler)
    print(f"local thickness discovery UI: http://{args.bind}:{args.port}/",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
