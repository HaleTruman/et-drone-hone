"""Replay historic JPEGs through the implemented shared production stages.

This is an offline review adapter.  It does not run in, or get imported by,
the production inference path.  It invokes the production shared stages and
the approved C-shape route, then losslessly materializes their schema records.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ...src.configurations import (
    DEFAULT_C_SHAPE_CONFIGURATION, DEFAULT_DENSITY_CONFIGURATION)
from ...src.c_shape.process import process_c_shape
from ...src.density_bank import DensityBank
from ...src.preprocessing import (
    DEFAULT_CONFIG, PreprocessingConfig, decode_jpeg, load_lut,
    preprocess_frame)
from ...src.schema import (
    C_SHAPE_TOPOLOGY, CShapeConfiguration, CShapeResult,
    DensityBankConfiguration, DensityEvidence, FrameObservation,
    TopologyDecision)
from ...src.topology import assess_frame
from .schema_json import (
    ARRAY_ENCODING, RUNTIME_RECORD_ENCODING, runtime_value, schema_contracts,
    write_json)


REVIEW_FORMAT_VERSION = 4
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FLIGHT_ROOT = Path(__file__).resolve().parents[7]
DEFAULT_SOURCE_RUN = FLIGHT_ROOT / "logs" / "run-20260731T093159Z"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "ui" / "review_runs"


@dataclass(frozen=True, slots=True)
class HistoricFrameRecord:
    """Manifest identity plus the recorded JPEG supplied to live ingress."""

    run_id: str
    frame_id: int
    sim_time_ns: int
    jpeg_size: int
    relative_path: str
    source_path: Path

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

    def __iter__(self) -> Iterator[HistoricFrameRecord]:
        seen_ids: set[int] = set()
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
                if frame_id in seen_ids:
                    raise ValueError(f"duplicate frame_id in manifest: {frame_id}")
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
                seen_ids.add(frame_id)
                previous_time = sim_time_ns
                yield HistoricFrameRecord(
                    self.run_dir.name, frame_id, sim_time_ns, jpeg_size,
                    relative.as_posix(), source_path)


@dataclass(frozen=True, slots=True)
class SharedReviewFrame:
    frame_observation: FrameObservation
    density_configuration: DensityBankConfiguration
    c_shape_configuration: CShapeConfiguration
    topology_decisions: tuple[TopologyDecision, ...]
    density_evidence: tuple[DensityEvidence, ...]
    c_shape_results: tuple[CShapeResult, ...]


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

    def process_input(
        self, *, frame_id: int, sim_time_ns: int, jpeg_bytes: bytes
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
        decisions = assess_frame(frame, density_bank=bank)
        evidence = tuple(
            item
            for component in frame.components
            for item in bank.precompute(component)
        )
        components = {
            component.component_id: component for component in frame.components}
        c_shape_results = tuple(
            process_c_shape(
                frame, components[decision.component_id], decision, bank)
            for decision in decisions
            if (decision.accepted and
                decision.topology_label == C_SHAPE_TOPOLOGY)
        )
        return SharedReviewFrame(
            frame, bank.configuration(), DEFAULT_C_SHAPE_CONFIGURATION,
            decisions, evidence, c_shape_results)

    def process_record(self, record: HistoricFrameRecord) -> SharedReviewFrame:
        return self.process_input(**record.pipeline_input())


def pipeline_schema_records(result: SharedReviewFrame) -> tuple:
    """Return every schema.py object materialized by the shared review pass."""
    return (
        result.frame_observation,
        result.density_configuration,
        result.c_shape_configuration,
        *result.topology_decisions,
        *result.density_evidence,
        *result.c_shape_results,
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
            "CShapeConfiguration": 1,
            "TopologyDecision": len(result.topology_decisions),
            "DensityEvidence": len(result.density_evidence),
            "CShapeResult": len(result.c_shape_results),
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

    for index, record in enumerate(selected, 1):
        result = adapter.process_record(record)
        filename = f"frame-{record.frame_id:08d}-{record.sim_time_ns}.json"
        relative_output = Path("frames") / filename
        write_json(output_dir / relative_output, _frame_payload(record, result))
        entries.append({
            "frame_id": record.frame_id,
            "sim_time_ns": record.sim_time_ns,
            "source_path": record.relative_path,
            "result_path": relative_output.as_posix(),
            "component_count": len(result.frame_observation.components),
            "c_shape_result_count": len(result.c_shape_results),
            "gated": result.frame_observation.gated,
            "rejection_reason": result.frame_observation.rejection_reason,
        })
        if index == 1 or index % 25 == 0 or index == len(selected):
            print(f"review frames: {index}/{len(selected)}")

    manifest = {
        "review_format_version": REVIEW_FORMAT_VERSION,
        "run_id": source.run_dir.name,
        "source_manifest": "frames.jsonl",
        "timing_source": "frames.jsonl::sim_time_ns",
        "source_frame_count": len(records),
        "processed_frame_count": len(selected),
        "complete": len(selected) == len(records),
        "pipeline_frontier": "shared_density_evidence+c_shape_refinement",
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
            "CShapeConfiguration", "CShapeDensityProfileRule",
            "CShapeResult", "CShapeRefinedLine",
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
