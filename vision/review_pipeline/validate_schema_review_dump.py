"""Recompute historic frames and prove every schema JSON record is exact."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .replay_historic_run import (
    DEFAULT_OUTPUT_ROOT, DEFAULT_SOURCE_RUN, REVIEW_FORMAT_VERSION,
    HistoricRunSource, SharedReviewAdapter, pipeline_schema_records)
from .schema_json import (
    RUNTIME_RECORD_ENCODING, assert_runtime_equal, runtime_object, write_json)


def load_schema_records(path: str | Path) -> tuple[dict, tuple]:
    """Load one frame document and reconstruct its inference objects."""
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("review_format_version") != REVIEW_FORMAT_VERSION:
        raise ValueError(
            f"unexpected review format: {document.get('review_format_version')}")
    if document.get("runtime_record_encoding") != RUNTIME_RECORD_ENCODING:
        raise ValueError(
            f"unexpected runtime encoding: "
            f"{document.get('runtime_record_encoding')}")
    records = tuple(runtime_object(item)
                    for item in document["schema_records"])
    counted = Counter(type(record).__name__ for record in records)
    expected_counts = document["schema_record_counts"]
    actual_counts = {
        name: counted.get(name, 0) for name in expected_counts}
    unknown_types = set(counted) - set(expected_counts)
    if actual_counts != expected_counts or unknown_types:
        raise ValueError(
            f"schema record counts changed: {actual_counts} != "
            f"{expected_counts}; unknown={sorted(unknown_types)}")
    return document, records


def validate_review_dump(
    run_dir: str | Path,
    review_dir: str | Path,
    *,
    limit: int | None = None,
) -> dict:
    """Prove recorded objects match freshly materialized pipeline objects."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    source_records = tuple(HistoricRunSource(run_dir))
    if limit is not None:
        source_records = source_records[:limit]
    review_root = Path(review_dir)
    manifest = json.loads(
        (review_root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest["frames"][:len(source_records)]
    if len(entries) != len(source_records):
        raise ValueError("review manifest does not cover every source record")

    adapter = SharedReviewAdapter()
    counts: Counter[str] = Counter()
    for index, (source, entry) in enumerate(zip(source_records, entries)):
        if (entry["frame_id"] != source.frame_id
                or entry["sim_time_ns"] != source.sim_time_ns
                or entry["source_path"] != source.relative_path):
            raise AssertionError(f"frame identity changed at manifest index {index}")
        document, recorded = load_schema_records(
            review_root / entry["result_path"])
        source_evidence = document["source"]
        if (source_evidence["frame_id"] != source.frame_id
                or source_evidence["sim_time_ns"] != source.sim_time_ns
                or source_evidence["relative_path"] != source.relative_path):
            raise AssertionError(f"frame source changed at manifest index {index}")
        expected = pipeline_schema_records(adapter.process_record(source))
        if len(expected) != len(recorded):
            raise AssertionError(
                f"record count changed for frame {source.frame_id}: "
                f"{len(expected)} != {len(recorded)}")
        for record_index, (left, right) in enumerate(zip(expected, recorded)):
            assert_runtime_equal(
                left, right,
                f"frame[{source.frame_id}].schema_records[{record_index}]",
            )
            counts[type(left).__name__] += 1

    return {
        "validation": "exact_runtime_round_trip",
        "passed": True,
        "run_id": Path(run_dir).name,
        "validated_frames": len(source_records),
        "schema_record_counts": dict(counts),
        "runtime_record_encoding": RUNTIME_RECORD_ENCODING,
        "review_format_version": REVIEW_FORMAT_VERSION,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path,
                        default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    review_dir = args.review_dir or DEFAULT_OUTPUT_ROOT / args.run_dir.name
    report = validate_review_dump(
        args.run_dir, review_dir, limit=args.limit)
    output = args.output or review_dir / "schema-dump-validation.json"
    write_json(output, report)
    print(json.dumps(report, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
