import hashlib
import json
from dataclasses import fields, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from sensing.vision.models.deterministic_v3.src.schema import (
    ApertureCenterEvidence, ComponentObservation, ContourEvidence,
    ContourNodeEvidence, CShapeConfiguration, DensityBankConfiguration,
    DensityEvidence, DensityProfile, FrameObservation, TopologyDecision,
    TopologyEvidence)
from sensing.vision.models.deterministic_v3.ui.backend.replay_historic_run import (
    REVIEW_FORMAT_VERSION, HistoricRunSource, SharedReviewAdapter,
    _frame_payload, materialize_review_run, pipeline_schema_records)
from sensing.vision.models.deterministic_v3.ui.backend.schema_json import (
    RUNTIME_RECORD_ENCODING, assert_runtime_equal, runtime_object,
    runtime_value)
from sensing.vision.models.deterministic_v3.ui.backend.validate_schema_review_dump import (
    load_schema_records)


def _field_names(contract):
    return {field.name for field in fields(contract)}


def _fixture_run(tmp_path):
    run_dir = tmp_path / "run-fixture"
    frame_dir = run_dir / "vision_frames"
    frame_dir.mkdir(parents=True)
    image = np.zeros((72, 88, 3), np.uint8)
    image[8:64, 12:76] = 255
    image[26:46, 32:56] = 0
    ok, encoded = cv2.imencode(".jpg", image,
                              [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    jpeg_bytes = encoded.tobytes()
    frame_id = 17
    sim_time_ns = 123_456_789
    relative = f"vision_frames/frame-{frame_id:08d}-{sim_time_ns}.jpg"
    path = run_dir / relative
    path.write_bytes(jpeg_bytes)
    record = {
        "frame_id": frame_id,
        "sim_time_ns": sim_time_ns,
        "jpeg_size": len(jpeg_bytes),
        "path": relative,
    }
    (run_dir / "frames.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    lut = np.zeros(1 << 24, np.uint8)
    for blue, green, red in np.unique(decoded.reshape(-1, 3), axis=0):
        if int(blue) + int(green) + int(red) > 600:
            lut[(int(red) << 16) | (int(green) << 8) | int(blue)] = 1
    return run_dir, path, lut


def test_historic_source_preserves_live_ingress_identity_and_source(tmp_path):
    run_dir, source_path, _ = _fixture_run(tmp_path)
    before = hashlib.sha256(source_path.read_bytes()).digest()
    record = next(iter(HistoricRunSource(run_dir)))
    pipeline_input = record.pipeline_input()

    assert set(pipeline_input) == {"frame_id", "sim_time_ns", "jpeg_bytes"}
    assert pipeline_input["frame_id"] == 17
    assert pipeline_input["sim_time_ns"] == 123_456_789
    assert hashlib.sha256(source_path.read_bytes()).digest() == before


def test_review_json_covers_exact_materialized_schema_fields(tmp_path):
    run_dir, source_path, lut = _fixture_run(tmp_path)
    record = next(iter(HistoricRunSource(run_dir)))
    result = SharedReviewAdapter(lut=lut).process_record(record)
    assert result.frame_observation.components

    output_root = tmp_path / "review_runs"
    # Materialization loads the repository LUT, so write the directly processed
    # result through the same payload shape by exercising its serializer below.
    from sensing.vision.models.deterministic_v3.ui.backend import schema_json
    from sensing.vision.models.deterministic_v3.ui.backend.replay_historic_run import (
        _frame_payload)

    destination = output_root / "frame.json"
    schema_json.write_json(destination, _frame_payload(record, result))
    document = json.loads(destination.read_text(encoding="utf-8"))
    encoded_records = document["schema_records"]
    frame = encoded_records[0]["__dataclass__"]
    frame_fields = frame["fields"]
    component = frame_fields["components"]["__tuple__"][0]["__dataclass__"]
    contour_node = frame_fields[
        "closed_contour_nodes"]["__tuple__"][0]["__dataclass__"]
    component_fields = component["fields"]
    topology = component_fields["topology"]["__dataclass__"]["fields"]
    contours = component_fields["contours"]["__dataclass__"]["fields"]
    configuration = next(
        item["__dataclass__"] for item in encoded_records
        if item["__dataclass__"]["name"] == "DensityBankConfiguration")
    c_shape_configuration = next(
        item["__dataclass__"] for item in encoded_records
        if item["__dataclass__"]["name"] == "CShapeConfiguration")
    decision = next(
        item["__dataclass__"] for item in encoded_records
        if item["__dataclass__"]["name"] == "TopologyDecision")
    density = next(
        item["__dataclass__"] for item in encoded_records
        if item["__dataclass__"]["name"] == "DensityEvidence")

    assert document["review_format_version"] == REVIEW_FORMAT_VERSION
    assert document["runtime_record_encoding"] == RUNTIME_RECORD_ENCODING
    assert frame["name"] == "FrameObservation"
    assert list(frame_fields) == [field.name for field in fields(FrameObservation)]
    assert list(component_fields) == [
        field.name for field in fields(ComponentObservation)]
    assert set(contour_node["fields"]) == _field_names(ContourNodeEvidence)
    assert set(topology) == \
        _field_names(TopologyEvidence)
    assert set(contours) == \
        _field_names(ContourEvidence)
    assert set(decision["fields"]) == \
        _field_names(TopologyDecision)
    assert decision["fields"]["topology_label"] in {
        "standard", "multi_void", "c_shape", "unknown", "clipped"}
    center = decision["fields"][
        "aperture_center_evidence"]["__tuple__"][0]["__dataclass__"]
    assert set(center["fields"]) == _field_names(ApertureCenterEvidence)
    assert set(configuration["fields"]) == \
        _field_names(DensityBankConfiguration)
    assert set(c_shape_configuration["fields"]) == \
        _field_names(CShapeConfiguration)
    assert len(c_shape_configuration[
        "fields"]["density_profile_rules"]["__tuple__"]) == 4
    assert len(configuration["fields"]["profiles"]["__tuple__"]) == 10
    assert configuration["fields"]["ignore_frame_edge_clipped"] is True
    assert set(density["fields"]) == _field_names(DensityEvidence)
    assert set(density["fields"]["profile"]["__dataclass__"]["fields"]) == \
        _field_names(DensityProfile)

    restored = tuple(runtime_object(item) for item in encoded_records)
    expected = pipeline_schema_records(result)
    assert len(restored) == len(expected)
    for left, right in zip(expected, restored):
        assert_runtime_equal(left, right)
    assert type(restored[0].components) is tuple
    assert type(restored[0].components[0].topology) is TopologyEvidence
    assert hashlib.sha256(source_path.read_bytes()).digest() == \
        hashlib.sha256(record.pipeline_input()["jpeg_bytes"]).digest()


def test_runtime_encoding_refuses_to_silently_copy_noncontiguous_evidence():
    noncontiguous = np.arange(24, dtype=np.int32).reshape(4, 6)[:, ::2]

    with pytest.raises(TypeError, match="non-contiguous"):
        runtime_value(noncontiguous)


def test_schema_record_counts_retain_explicit_zero_density(tmp_path):
    run_dir, _, lut = _fixture_run(tmp_path)
    record = next(iter(HistoricRunSource(run_dir)))
    result = SharedReviewAdapter(lut=lut).process_record(record)
    result = replace(result, density_evidence=())
    destination = tmp_path / "zero-density.json"
    from sensing.vision.models.deterministic_v3.ui.backend.schema_json import (
        write_json)
    write_json(destination, _frame_payload(record, result))

    document, records = load_schema_records(destination)

    assert document["schema_record_counts"]["DensityEvidence"] == 0
    assert document["schema_record_counts"]["CShapeConfiguration"] == 1
    assert all(type(record).__name__ != "DensityEvidence" for record in records)


def test_run_manifest_uses_source_timing_and_reports_completeness(
    tmp_path, monkeypatch
):
    run_dir, _, lut = _fixture_run(tmp_path)
    from sensing.vision.models.deterministic_v3.ui.backend import (
        replay_historic_run)

    monkeypatch.setattr(
        replay_historic_run, "SharedReviewAdapter",
        lambda: SharedReviewAdapter(lut=lut),
    )
    manifest_path = materialize_review_run(
        run_dir, tmp_path / "outputs")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["source_frame_count"] == 1
    assert manifest["processed_frame_count"] == 1
    assert manifest["complete"] is True
    assert manifest["review_format_version"] == REVIEW_FORMAT_VERSION
    assert manifest["runtime_record_encoding"] == RUNTIME_RECORD_ENCODING
    assert manifest["timing_source"] == "frames.jsonl::sim_time_ns"
    assert manifest["maximum_input_components"] == 0
    assert manifest["ignore_frame_edge_clipped_for_density"] is True
    assert manifest["density_profile_ids"] == [
        f"scale_{index:02d}" for index in range(1, 11)]
    assert manifest["pipeline_frontier"] == \
        "shared_density_evidence+c_shape_refinement"
    assert manifest["frames"][0]["frame_id"] == 17
    assert manifest["frames"][0]["sim_time_ns"] == 123_456_789
    assert not Path(manifest["frames"][0]["source_path"]).is_absolute()
