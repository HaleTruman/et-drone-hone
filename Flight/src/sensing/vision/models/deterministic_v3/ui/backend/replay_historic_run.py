"""Replay historic JPEGs through the implemented production geometry stages.

This is an offline review adapter.  It does not run in, or get imported by,
the production inference path.  It invokes the same shared and specialized
processors as ``src.pipeline``, then losslessly materializes their schema
records for review.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ...src.configurations import (
    DEFAULT_C_SHAPE_CONFIGURATION, DEFAULT_DENSITY_CONFIGURATION,
    DEFAULT_STANDARD_GATE_CONFIGURATION)
from ...src.c_shape.process import process_c_shape
from ...src.density_bank import DensityBank
from ...src.pipeline import GateGeometryPipeline
from ...src.preprocessing import (
    DEFAULT_CONFIG, PreprocessingConfig, decode_jpeg, load_lut,
    preprocess_frame)
from ...src.schema import (
    C_SHAPE_ROUTE, MULTI_GATE_ROUTE, STANDARD_ROUTE,
    CShapeConfiguration, CShapeResult, DensityBankConfiguration,
    DensityEvidence, FrameObservation, GeometryFrameResult,
    StandardGateConfiguration, StandardGateResult, TopologyDecision)
from ...src.standard_gate_processing.process import (
    process_standard_components)
from ...src.topology import assess_frame
from .schema_json import (
    ARRAY_ENCODING, RUNTIME_RECORD_ENCODING, runtime_value, schema_contracts,
    write_json)


REVIEW_FORMAT_VERSION = 8
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FLIGHT_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_SOURCE_RUN = FLIGHT_ROOT / "logs" / "run-20260731T093159Z"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "ui" / "review_runs"
GEOMETRY_OUTPUT_FILENAME = "gate-geometry-pnp-runtime.json"
LEGACY_GEOMETRY_OUTPUT_FILENAME = "standard-gate-pnp-runtime.json"


@dataclass(frozen=True, slots=True)
class HistoricVehicleState:
    """Recorded camera-motion input required by post-PnP regression."""

    sim_time_ns: int
    position_local_ned_m: tuple[float, float, float]
    attitude_quaternion: tuple[float, float, float, float]


def _historic_vehicle_states(
    run_dir: Path,
) -> dict[tuple[int, int], HistoricVehicleState]:
    """Join manifest frame identities to telemetry by an unambiguous cycle."""
    vision_path = run_dir / "lists" / "vision_frames.jsonl"
    telemetry_path = run_dir / "lists" / "telemetry.jsonl"
    if not vision_path.is_file() or not telemetry_path.is_file():
        return {}
    telemetry_by_cycle = {}
    with telemetry_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                cycle = int(record["inner_cycle"])
                value = record["telemetry"]["vehicle_state"]
                position = tuple(map(float, value["position_local_ned_m"]))
                attitude = tuple(map(float, value["attitude_quaternion"]))
                sim_time_ns = int(value["sim_time_ns"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid telemetry record at line {line_number}"
                ) from error
            if (len(position) != 3 or len(attitude) != 4 or
                    not np.all(np.isfinite((*position, *attitude)))):
                raise ValueError(
                    f"invalid vehicle pose at telemetry line {line_number}")
            if cycle in telemetry_by_cycle:
                raise ValueError(f"duplicate telemetry inner_cycle: {cycle}")
            telemetry_by_cycle[cycle] = HistoricVehicleState(
                sim_time_ns, position, attitude)

    vision_cycles: dict[int, list[tuple[int | None, int]]] = {}
    with vision_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)["frame"]
                frame_id = int(record["frame_id"])
                cycle = int(record["inner_cycle"])
                sim_time_value = record.get("sim_time_ns")
                sim_time_ns = (None if sim_time_value is None
                               else int(sim_time_value))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid vision-frame record at line {line_number}"
                ) from error
            vision_cycles.setdefault(frame_id, []).append(
                (sim_time_ns, cycle))

    states = {}
    with (run_dir / "frames.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                frame_id = int(record["frame_id"])
                sim_time_ns = int(record["sim_time_ns"])
                cycle_value = record.get("cycle")
                manifest_cycle = (None if cycle_value is None
                                  else int(cycle_value))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid frame manifest record at line {line_number}"
                ) from error
            candidates = {
                cycle for logged_time, cycle in vision_cycles.get(frame_id, ())
                if logged_time is None or logged_time == sim_time_ns
            }
            if manifest_cycle is not None:
                candidates.intersection_update((manifest_cycle,))
            if len(candidates) != 1:
                continue
            state = telemetry_by_cycle.get(next(iter(candidates)))
            if state is not None:
                states[(frame_id, sim_time_ns)] = state
    return states


@dataclass(frozen=True, slots=True)
class HistoricFrameRecord:
    """Manifest identity plus the recorded JPEG supplied to live ingress."""

    run_id: str
    frame_id: int
    sim_time_ns: int
    jpeg_size: int
    relative_path: str
    source_path: Path
    vehicle_state: HistoricVehicleState | None = None

    def pipeline_input(self) -> dict:
        jpeg_bytes = self.source_path.read_bytes()
        if len(jpeg_bytes) != self.jpeg_size:
            raise ValueError(
                f"JPEG size changed for {self.relative_path}: "
                f"{len(jpeg_bytes)} != {self.jpeg_size}")
        return {
            "frame_id": self.frame_id,
            "sim_time_ns": self.sim_time_ns,
            "jpeg_bytes": jpeg_bytes,
        }


class HistoricRunSource:
    """Read ordered live-ingress records from a run's frame manifest."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.manifest_path = self.run_dir / "frames.jsonl"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"historic frame manifest not found: {self.manifest_path}")
        self.vehicle_states = _historic_vehicle_states(self.run_dir)

    def __iter__(self) -> Iterator[HistoricFrameRecord]:
        seen_identities: set[tuple[int, int]] = set()
        previous_time = -1
        with self.manifest_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    frame_id = int(record["frame_id"])
                    sim_time_ns = int(record["sim_time_ns"])
                    jpeg_size = int(record["jpeg_size"])
                    relative_path = str(record["path"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid frame manifest record at line {line_number}"
                    ) from error
                identity = (frame_id, sim_time_ns)
                if identity in seen_identities:
                    raise ValueError(
                        f"duplicate frame identity in manifest: {identity}")
                if sim_time_ns < previous_time:
                    raise ValueError("frame manifest timing is not monotonic")
                relative = Path(relative_path)
                if (relative.is_absolute() or not relative.parts
                        or relative.parts[0] != "vision_frames"):
                    raise ValueError(
                        f"frame path must be relative to vision_frames: "
                        f"{relative_path}")
                source_path = (self.run_dir / relative).resolve()
                if not source_path.is_relative_to(self.run_dir):
                    raise ValueError(f"frame path escapes run: {relative_path}")
                if not source_path.is_file():
                    raise FileNotFoundError(source_path)
                seen_identities.add(identity)
                previous_time = sim_time_ns
                yield HistoricFrameRecord(
                    self.run_dir.name, frame_id, sim_time_ns, jpeg_size,
                    relative.as_posix(), source_path,
                    self.vehicle_states.get((frame_id, sim_time_ns)))


@dataclass(frozen=True, slots=True)
class SharedReviewFrame:
    frame_observation: FrameObservation
    density_configuration: DensityBankConfiguration
    standard_gate_configuration: StandardGateConfiguration
    c_shape_configuration: CShapeConfiguration
    topology_decisions: tuple[TopologyDecision, ...]
    density_evidence: tuple[DensityEvidence, ...]
    c_shape_results: tuple[CShapeResult, ...]
    geometry_frame_result: GeometryFrameResult

    @property
    def standard_gate_results(self) -> tuple[StandardGateResult, ...]:
        """Expose the canonical runtime tuple without maintaining a copy."""
        return self.geometry_frame_result.standard_gate_results


class SharedReviewAdapter:
    """Use shared production APIs without adding review work to live runtime."""

    def __init__(
        self,
        *,
        lut: np.ndarray | None = None,
        config: PreprocessingConfig = DEFAULT_CONFIG,
    ) -> None:
        self.lut = load_lut() if lut is None else lut
        self.config = config
        self.pipeline = GateGeometryPipeline(self.lut, self.config)

    def process_input(
        self,
        *,
        frame_id: int,
        sim_time_ns: int,
        jpeg_bytes: bytes,
        vehicle_state: HistoricVehicleState | None = None,
    ) -> SharedReviewFrame:
        """Consume the same authoritative identity and JPEG as live ingress."""
        image = decode_jpeg(jpeg_bytes)
        frame = preprocess_frame(
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            image=image,
            lut=self.lut,
            config=self.config,
        )
        bank = DensityBank(frame)
        decisions = assess_frame(frame)
        evidence = tuple(
            item
            for component in frame.components
            for item in bank.precompute(component)
        )
        standard_results = process_standard_components(frame, bank)
        c_shape_results = []
        for component, decision, standard in zip(
                frame.components, decisions, standard_results):
            if (not standard.accepted and decision.accepted and
                    decision.route == C_SHAPE_ROUTE):
                c_shape_results.append(process_c_shape(
                    frame, component, decision, bank))
        geometry = self.pipeline.process_frame(
            frame_id=frame_id,
            sim_time_ns=sim_time_ns,
            jpeg_bytes=jpeg_bytes,
            vehicle_state=vehicle_state,
        )
        return SharedReviewFrame(
            frame,
            bank.configuration(),
            DEFAULT_STANDARD_GATE_CONFIGURATION,
            DEFAULT_C_SHAPE_CONFIGURATION,
            decisions,
            evidence,
            tuple(c_shape_results),
            geometry,
        )

    def process_record(self, record: HistoricFrameRecord) -> SharedReviewFrame:
        return self.process_input(
            **record.pipeline_input(), vehicle_state=record.vehicle_state)


def pipeline_schema_records(result: SharedReviewFrame) -> tuple:
    """Return every schema.py object materialized by the shared review pass."""
    return (
        result.frame_observation,
        result.density_configuration,
        result.standard_gate_configuration,
        result.c_shape_configuration,
        *result.topology_decisions,
        *result.density_evidence,
        *result.standard_gate_results,
        *result.c_shape_results,
        result.geometry_frame_result,
    )


def _frame_payload(
    record: HistoricFrameRecord, result: SharedReviewFrame
) -> dict:
    schema_records = pipeline_schema_records(result)
    return {
        "review_format_version": REVIEW_FORMAT_VERSION,
        "source": {
            "run_id": record.run_id,
            "frame_id": record.frame_id,
            "sim_time_ns": record.sim_time_ns,
            "jpeg_size": record.jpeg_size,
            "relative_path": record.relative_path,
            "timing_source": "frames.jsonl::sim_time_ns",
        },
        "runtime_record_encoding": RUNTIME_RECORD_ENCODING,
        "schema_records": [runtime_value(value) for value in schema_records],
        "schema_record_counts": {
            "FrameObservation": 1,
            "DensityBankConfiguration": 1,
            "StandardGateConfiguration": 1,
            "CShapeConfiguration": 1,
            "TopologyDecision": len(result.topology_decisions),
            "DensityEvidence": len(result.density_evidence),
            "StandardGateResult": len(result.standard_gate_results),
            "CShapeResult": len(result.c_shape_results),
            "GeometryFrameResult": 1,
        },
    }


def materialize_review_run(
    run_dir: str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    limit: int | None = None,
    force: bool = False,
) -> Path:
    """Write a deterministic manifest and complete per-frame schema records."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    source = HistoricRunSource(run_dir)
    records = tuple(source)
    selected = records if limit is None else records[:limit]
    output_dir = Path(output_root) / source.run_dir.name
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"review run already exists; pass --force to replace files: "
            f"{output_dir}")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    adapter = SharedReviewAdapter()
    entries = []
    geometry_entries = []
    topology_counts = Counter()
    quadrilateral_route_counts = Counter()
    accepted_quadrilateral_route_counts = Counter()
    pnp_route_counts = Counter()
    accepted_pnp_route_counts = Counter()
    final_pose_route_counts = Counter()
    accepted_final_pose_route_counts = Counter()
    standard_result_count = 0
    accepted_standard_result_count = 0
    standard_rejection_counts = Counter()
    multi_gate_result_count = 0

    for index, record in enumerate(selected, 1):
        result = adapter.process_record(record)
        filename = f"frame-{record.frame_id:08d}-{record.sim_time_ns}.json"
        relative_output = Path("frames") / filename
        write_json(output_dir / relative_output, _frame_payload(record, result))
        geometry = result.geometry_frame_result
        geometry_entries.append({
            "run_id": record.run_id,
            "runtime_result": asdict(geometry),
        })
        topology_counts.update(
            decision.topology_label for decision in geometry.topology_decisions)
        quadrilateral_route_counts.update(
            quadrilateral.route
            for quadrilateral in geometry.quadrilateral_estimates)
        accepted_quadrilateral_route_counts.update(
            quadrilateral.route
            for quadrilateral in geometry.quadrilateral_estimates
            if quadrilateral.accepted)
        pnp_route_counts.update(
            pose.route for pose in geometry.pnp_relative_pose_estimates)
        accepted_pnp_route_counts.update(
            pose.route for pose in geometry.pnp_relative_pose_estimates
            if pose.accepted)
        final_pose_route_counts.update(
            pose.route for pose in geometry.camera_pose_estimates)
        accepted_final_pose_route_counts.update(
            pose.route for pose in geometry.camera_pose_estimates
            if pose.accepted)
        standard_result_count += len(geometry.standard_gate_results)
        accepted_standard_result_count += sum(
            result.accepted for result in geometry.standard_gate_results)
        standard_rejection_counts.update(
            result.rejection_reason
            for result in geometry.standard_gate_results
            if not result.accepted and result.rejection_reason is not None)
        multi_gate_result_count += len(geometry.multi_gate_results)
        entries.append({
            "frame_id": record.frame_id,
            "sim_time_ns": record.sim_time_ns,
            "source_path": record.relative_path,
            "result_path": relative_output.as_posix(),
            "component_count": len(result.frame_observation.components),
            "standard_gate_result_count": len(result.standard_gate_results),
            "accepted_standard_gate_result_count": sum(
                item.accepted for item in result.standard_gate_results),
            "c_shape_result_count": len(result.c_shape_results),
            "multi_gate_result_count": len(geometry.multi_gate_results),
            "quadrilateral_estimate_count": len(
                geometry.quadrilateral_estimates),
            "pnp_relative_pose_estimate_count": len(
                geometry.pnp_relative_pose_estimates),
            "accepted_pnp_relative_pose_estimate_count": sum(
                pose.accepted for pose in
                geometry.pnp_relative_pose_estimates),
            "secondary_pnp_candidate_count": sum(
                max(0, len(pose.candidates) - 1)
                for pose in geometry.pnp_relative_pose_estimates),
            "camera_pose_estimate_count": len(geometry.camera_pose_estimates),
            "accepted_camera_pose_estimate_count": sum(
                pose.accepted for pose in geometry.camera_pose_estimates),
            "gated": result.frame_observation.gated,
            "rejection_reason": result.frame_observation.rejection_reason,
        })
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"review frames: {index}/{len(selected)}")

    geometry_document = {
        "summary": {
            "format_version": 2,
            "runs": [source.run_dir.name],
            "frames": len(selected),
            "processed_routes": [
                STANDARD_ROUTE, C_SHAPE_ROUTE, MULTI_GATE_ROUTE],
            "topology_labels": dict(topology_counts),
            "standard_gate_results": standard_result_count,
            "accepted_standard_gate_results": accepted_standard_result_count,
            "standard_gate_rejections": dict(standard_rejection_counts),
            "multi_gate_results": multi_gate_result_count,
            "quadrilateral_results_by_route": dict(
                quadrilateral_route_counts),
            "accepted_quadrilaterals_by_route": dict(
                accepted_quadrilateral_route_counts),
            "pnp_relative_pose_results_by_route": dict(pnp_route_counts),
            "accepted_pnp_relative_poses_by_route": dict(
                accepted_pnp_route_counts),
            "camera_pose_results_by_route": dict(final_pose_route_counts),
            "accepted_camera_poses_by_route": dict(
                accepted_final_pose_route_counts),
        },
        "frames": geometry_entries,
    }
    geometry_path = write_json(
        output_dir / GEOMETRY_OUTPUT_FILENAME, geometry_document)
    legacy_geometry_path = output_dir / LEGACY_GEOMETRY_OUTPUT_FILENAME
    if force and legacy_geometry_path.is_file():
        legacy_geometry_path.unlink()

    manifest = {
        "review_format_version": REVIEW_FORMAT_VERSION,
        "run_id": source.run_dir.name,
        "source_manifest": "frames.jsonl",
        "timing_source": "frames.jsonl::sim_time_ns",
        "source_frame_count": len(records),
        "processed_frame_count": len(selected),
        "complete": len(selected) == len(records),
        "pipeline_frontier":
            "shared_density_evidence+standard_c_shape_multi_gate_raw_camera_"
            "pnp+authoritative_gate_pose_regression",
        "geometry_result_path": geometry_path.name,
        "array_encoding": ARRAY_ENCODING,
        "runtime_record_encoding": RUNTIME_RECORD_ENCODING,
        "preprocessing_version": adapter.config.version,
        "maximum_input_components": adapter.config.maximum_input_components,
        "ignore_frame_edge_clipped_for_density":
            DEFAULT_DENSITY_CONFIGURATION.ignore_frame_edge_clipped,
        "density_profile_ids": [profile.profile_id
                                for profile in
                                DEFAULT_DENSITY_CONFIGURATION.profiles],
        "schema_contracts": schema_contracts(),
        "materialized_schema_types": [
            "FrameObservation", "ComponentObservation", "TopologyEvidence",
            "ContourEvidence", "TopologyDecision",
            "DensityBankConfiguration", "DensityEvidence", "DensityProfile",
            "StandardGateConfiguration", "StandardGateResult",
            "StandardGateSideEvidence",
            "CShapeConfiguration", "CShapeDensityProfileRule",
            "CShapeResult", "CShapeRefinedLine",
            "MultiGateIdentificationResult", "MultiGateCandidateAssessment",
            "MultiGateThicknessPopulationEvidence",
            "MultiGateJunctionEvidence", "MultiGateDensityOverlapEvidence",
            "MultiGateResult", "MultiGateApertureFit",
            "QuadrilateralEstimate", "PnPCandidateEstimate",
            "PnPRelativePoseEstimate", "GateRegressionConfiguration",
            "GateRegressionEvidence", "CameraPoseEstimate",
            "GeometryFrameResult",
        ],
        "frames": entries,
    }
    return write_json(output_dir / "manifest.json", manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay historic frames into deterministic-v3 review JSON")
    parser.add_argument("run_dir", nargs="?", type=Path,
                        default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path,
                        default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    manifest = materialize_review_run(
        args.run_dir, args.output_root, limit=args.limit, force=args.force)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
