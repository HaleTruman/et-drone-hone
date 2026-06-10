import json
import struct

from sensing.vision.vision_stream import VISION_HEADER_FORMAT, VisionStreamReceiver


def packet(frame_id: int, chunk_id: int, total_chunks: int, jpeg_size: int, sim_time_ns: int, payload: bytes) -> bytes:
    return struct.pack(VISION_HEADER_FORMAT, frame_id, chunk_id, total_chunks, jpeg_size, len(payload), sim_time_ns) + payload


def test_reassembles_and_persists_chunked_jpeg(tmp_path) -> None:
    receiver = VisionStreamReceiver(output_dir=tmp_path)
    jpeg = b"\xff\xd8mock-jpeg\xff\xd9"

    assert receiver.process_packet(packet(7, 1, 2, len(jpeg), 99, jpeg[5:])) is None
    frame = receiver.process_packet(packet(7, 0, 2, len(jpeg), 99, jpeg[:5]))

    assert frame is not None
    assert frame.frame_id == 7
    assert frame.jpeg_bytes == jpeg
    assert (tmp_path / "frame-00000007-99.jpg").read_bytes() == jpeg
    manifest = json.loads((tmp_path / "frames.jsonl").read_text(encoding="utf-8"))
    assert manifest["frame_id"] == 7
    assert receiver.snapshot()["saved_frame_count"] == 1


def test_persists_one_manifest_entry_per_frame_id(tmp_path) -> None:
    receiver = VisionStreamReceiver(output_dir=tmp_path / "vision_frames")
    receiver.output_dir.mkdir()
    jpeg = b"\xff\xd8mock-jpeg\xff\xd9"

    first = receiver.process_packet(packet(7, 0, 1, len(jpeg), 99, jpeg))
    replay = receiver.process_packet(packet(7, 0, 1, len(jpeg), 100, jpeg))

    assert first is not None
    assert replay is not None
    manifest_lines = (tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 1
    manifest = json.loads(manifest_lines[0])
    assert manifest["frame_id"] == 7
    assert manifest["sim_time_ns"] == 99
    assert manifest["path"] == "vision_frames/frame-00000007-99.jpg"
    assert receiver.snapshot()["saved_frame_count"] == 1


def test_rejects_invalid_packet_payload_size() -> None:
    receiver = VisionStreamReceiver()
    bad_packet = struct.pack(VISION_HEADER_FORMAT, 1, 0, 1, 3, 10, 100) + b"abc"

    assert receiver.process_packet(bad_packet) is None
    assert receiver.invalid_packet_count == 1
