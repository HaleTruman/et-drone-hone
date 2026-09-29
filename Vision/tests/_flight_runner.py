"""Offline Flight integration scenarios; no listeners or initialization run."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import cv2
import numpy as np
import torch

from service import VisionObservation, VisionPerceptionConfig, VisionPerceptionService
import service as vision_service
from core.schema import MavlinkTelemetry, VehicleState, VisionFrame
from core import utils
from sensing.vision import VisionStreamReceiver
from sensing.vision.io.udp_protocol import VisionPacket, chunk_jpeg, pack_packet


START_NS = 1000000000
ELAPSED_NS = 250000000
JPEG = (
    Path(vision_service.__file__).parent / "models/deterministic_v2/assets"
    / "frame-00068527-1784514770942906700.jpg"
).read_bytes()


def ingest(receiver, frame_id, jpeg=JPEG, *, sim_time_ns=9000000000):
    frame = None
    for packet in reversed(chunk_jpeg(frame_id, jpeg, sim_time_ns)):
        frame = receiver.process_packet(pack_packet(packet))
    assert isinstance(frame, VisionFrame)
    assert frame.frame_id == frame_id
    assert frame.sim_time_ns == sim_time_ns
    assert frame.elapsed_time_ns == ELAPSED_NS
    assert frame.jpeg_bytes == jpeg
    assert receiver._socket is None and receiver._thread is None
    return frame


def reassembly(scratch):
    receiver = VisionStreamReceiver(output_dir=scratch / "vision_frames")
    receiver.begin_saving_frames()
    packets = chunk_jpeg(71, JPEG, 9000000000)
    assert len(packets) > 2
    # A duplicate chunk remains incomplete; the remaining chunks arrive backwards.
    assert receiver.process_packet(pack_packet(packets[-1])) is None
    assert receiver.process_packet(pack_packet(packets[-1])) is None
    assert receiver.get_next_frame() is None
    assert receiver.snapshot()["partial_frame_count"] == 1
    for packet in reversed(packets[:-1]):
        frame = receiver.process_packet(pack_packet(packet))
    assert isinstance(frame, VisionFrame)
    assert frame.jpeg_bytes == JPEG
    assert frame.frame_id == 71 and frame.sim_time_ns == 9000000000
    assert frame.elapsed_time_ns == ELAPSED_NS
    assert receiver.get_next_frame() is frame
    assert receiver.get_next_frame() is None
    assert receiver.snapshot()["partial_frame_count"] == 0
    assert Path(frame.saved_path).read_bytes() == JPEG
    manifest = json.loads((scratch / "frames.jsonl").read_text())
    assert manifest["frame_id"] == frame.frame_id
    assert manifest["sim_time_ns"] == frame.sim_time_ns
    assert manifest["elapsed_time_ns"] == ELAPSED_NS
    assert manifest["jpeg_size"] == len(JPEG)
    assert receiver.saved_frame_count == 1
    assert receiver._socket is None and receiver._thread is None


def invalid(_scratch):
    receiver = VisionStreamReceiver()
    assert receiver.process_packet(b"short") is None
    packet = chunk_jpeg(1, b"abcdef", 5, chunk_payload_bytes=3)[0]
    assert receiver.process_packet(pack_packet(replace(packet, payload_size=99))) is None
    assert receiver.process_packet(pack_packet(replace(packet, chunk_id=2))) is None
    # Inconsistent metadata invalidates an in-progress frame.
    assert receiver.process_packet(pack_packet(packet)) is None
    assert receiver.process_packet(pack_packet(replace(packet, jpeg_size=7))) is None
    assert receiver.snapshot()["partial_frame_count"] == 0
    # Completed chunks must also match the declared total JPEG length.
    corrupt = VisionPacket(2, 0, 1, 7, 3, 6, b"abc")
    assert receiver.process_packet(pack_packet(corrupt)) is None
    assert receiver.invalid_packet_count == 5
    assert receiver.get_next_frame() is None
    assert receiver.snapshot()["partial_frame_count"] == 0
    assert ingest(receiver, 3, b"valid").jpeg_bytes == b"valid"


def buffering(_scratch):
    receiver = VisionStreamReceiver(max_buffered_frames=2, max_partial_frames=1)
    for frame_id in (1, 2):
        packet = chunk_jpeg(frame_id, b"abcdef", 5, chunk_payload_bytes=3)[0]
        assert receiver.process_packet(pack_packet(packet)) is None
    assert receiver.dropped_partial_frame_count == 1
    assert receiver.snapshot()["partial_frame_count"] == 1
    receiver.clear_buffer()
    assert receiver.snapshot()["partial_frame_count"] == 0
    for frame_id in (10, 11, 12):
        ingest(receiver, frame_id, b"jpeg")
    assert receiver.snapshot()["buffered_frame_count"] == 2
    assert receiver.get_next_frame().frame_id == 11
    assert receiver.get_next_frame().frame_id == 12
    assert receiver.get_next_frame() is None
    for frame_id in (13, 14):
        ingest(receiver, frame_id, b"jpeg")
    assert receiver.get_latest_frame().frame_id == 14
    assert receiver.get_next_frame() is None
    ingest(receiver, 15, b"jpeg")
    receiver.clear_buffer()
    assert receiver.get_latest_frame() is None


def downstream(_scratch):
    from core.logging.logging import Logger
    from mapping.gates.gate_map import GateMap
    from sensing.telemetry.sync import DataSynchronizer

    receiver = VisionStreamReceiver()
    vision = VisionPerceptionService(
        VisionPerceptionConfig(backend="deterministic_v3_2", device="cpu")
    )
    gate_map = GateMap(required_minimum_observation_count=1)
    synchronizer = DataSynchronizer(max_delta_ns=1000)
    logger = Logger()
    image = cv2.imdecode(np.frombuffer(JPEG, dtype=np.uint8), cv2.IMREAD_COLOR)
    observations = []
    for index in range(5):
        current_image = cv2.warpAffine(
            image, np.float32([[1, 0, -2 * index], [0, 1, 0]]),
            (image.shape[1], image.shape[0]),
        )
        encoded, jpeg = cv2.imencode(".jpg", current_image)
        assert encoded
        frame = ingest(receiver, 200 + index, jpeg.tobytes(), sim_time_ns=2000000000 + index * 50000000)
        assert receiver.get_next_frame() is frame
        state = VehicleState(
            sim_time_ns=frame.sim_time_ns,
            position_local_ned_m=(0.04 * index, 0.02 * index, -1.0),
            velocity_local_ned_mps=(0.8, 0.4, 0.0),
            attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
            body_rates_frd_rps=(0.0, 0.0, 0.0),
            acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
        )
        telemetry = MavlinkTelemetry(frame.sim_time_ns + 100, state)
        assert synchronizer.align_frame_with_telemetry(frame, [telemetry]) == (frame, telemetry)
        assert synchronizer.get_synchronized_data() == (frame, telemetry)
        observation = vision.process_vision_frame(frame, vehicle_state=state)
        assert isinstance(observation, VisionObservation)
        assert observation.trace["pose_available"] is True
        assert synchronizer.align_observation_with_telemetry(observation, [telemetry]) == (observation, telemetry)
        assert synchronizer.get_synchronized_data() == (observation, telemetry)
        assert synchronizer.align_observation_with_telemetry(observation, []) is None
        assert synchronizer.align_observation_with_telemetry(
            observation, [MavlinkTelemetry(frame.sim_time_ns + 10000, state)]
        ) is None
        gate_map.update(observation, observer_position_local_ned_m=state.position_local_ned_m)
        logger.log_vision_observation(observation, frame_id=frame.frame_id)
        assert logger.records["vision_observations"][-1]["observation"] == observation.to_controller_payload()
        observations.append(observation)
    assert any(observation.gates for observation in observations)
    assert gate_map.gates
    assert all(record.source == observations[-1].source for record in gate_map.gates)
    json.dumps([asdict(record) for record in gate_map.gates])
    json.dumps(logger.records["vision_observations"])
    assert len(logger.records["vision_observations"]) == 5
    vision.shutdown()


if __name__ == "__main__":
    torch.set_num_threads(1)
    initialization_package = ModuleType("core.initialization")
    initialization_package.__path__ = []
    initialization_module = ModuleType("core.initialization.initialization")
    initialization_module.DEFINED_START_TIME_NS = START_NS
    # Keep the receiver's own _add_elapsed_time and the real time_since_ns;
    # substitute only their clock inputs and prevent runtime initialization.
    with patch.dict(sys.modules, {
        "core.initialization": initialization_package,
        "core.initialization.initialization": initialization_module,
    }), patch.object(utils.time, "perf_counter_ns", return_value=START_NS + ELAPSED_NS), patch(
        "socket.socket", side_effect=AssertionError("Offline integration must not open sockets")
    ), patch("threading.Thread.start", side_effect=AssertionError("Offline integration must not start threads")):
        {"reassembly": reassembly, "invalid": invalid, "buffering": buffering, "downstream": downstream}[
            sys.argv[1]
        ](Path(sys.argv[2]))
