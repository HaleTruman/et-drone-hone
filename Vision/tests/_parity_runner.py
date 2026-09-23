"""Exercise one implementation in an isolated interpreter for extraction tests."""

from dataclasses import fields
import importlib
import inspect
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


def contract(schema, service):
    def signature(value):
        return str(inspect.signature(value)).replace("core.schema.", "").replace(
            "aigp_vision.schema.", ""
        )

    gate = schema.VisionGateObservation.from_payload(
        {
            "id": "gate-7",
            "position_xyz": [1, 2, 3],
            "position_confidence": 0.8,
            "orientation_quat": [1, 0, 0, 0],
            "orientation_confidence": 0.6,
            "trace": {"source": "fixture"},
        }
    )
    observation = schema.VisionObservation.from_controller_payload(
        {
            "run": {"cycle": 17, "sim_time_ns": 125000000},
            "gates": [gate.to_payload()],
            "trace": {"pose_available": True},
        },
        source="fixture",
    )
    errors = []
    for payload in ({}, {"position_xyz": [1, 2]}, {
        "position_xyz": [1, 2, 3], "orientation_quat": [1, 0, 0]
    }):
        try:
            schema.VisionGateObservation.from_payload(payload)
        except ValueError as error:
            errors.append(str(error))
        else:
            raise AssertionError("Malformed gate payload was accepted")
    names = ("VehicleState", "VisionFrame", "VisionGateObservation", "VisionObservation")
    return {
        "fields": {name: [field.name for field in fields(getattr(schema, name))] for name in names},
        "signatures": {name: signature(getattr(schema, name)) for name in names},
        "methods": {
            name: signature(getattr(service.VisionPerceptionService, name))
            for name in ("process_frame", "process_vision_frame", "detect_gates")
        },
        "observation": observation.to_controller_payload(),
        "source": observation.source,
        "errors": errors,
        "default_backend": service.VisionPerceptionConfig().backend,
    }


def payload(observation, schema):
    assert isinstance(observation, schema.VisionObservation)
    assert all(isinstance(gate, schema.VisionGateObservation) for gate in observation.gates)
    return {
        "controller": observation.to_controller_payload(),
        "source": observation.source,
        "elapsed_time_ns": observation.elapsed_time_ns,
    }


def exercise(target):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    module = "sensing.vision" if target == "flight" else "aigp_vision"
    service = importlib.import_module(f"{module}.service")
    schema = importlib.import_module("core.schema" if target == "flight" else "aigp_vision.schema")
    model_root = Path(service.__file__).parent / "models"
    jpeg = (model_root / "deterministic_v2/assets/frame-00068527-1784514770942906700.jpg").read_bytes()
    result = {"contract": contract(schema, service), "cnn": {}, "deterministic": {}}

    cnn, regressor = None, None
    for mode, options in (
        ("landmarker", {}),
        ("passthrough", {"passthrough_regressor_targets": True}),
        ("regressor", {"run_landmarker": False}),
    ):
        instance = service.VisionPerceptionService(
            service.VisionPerceptionConfig(device="cpu", **options),
            cnn=cnn,
            regressor=regressor,
        )
        outputs = []
        for index in range(2):
            frame = schema.VisionFrame(100 + index, 1000000000 + index * 50000000, jpeg)
            if index == 0:
                observation = instance.process_vision_frame(frame)
            else:
                observation = instance.process_frame(
                    frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns, jpeg_bytes=jpeg
                )
            outputs.append(payload(observation, schema))
        result["cnn"][mode] = outputs
        cnn, regressor = instance.cnn, instance.regressor
        instance.shutdown()

    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    for backend in ("deterministic_v3", "deterministic_v3_2"):
        instance = service.VisionPerceptionService(
            service.VisionPerceptionConfig(backend=backend, device="cpu")
        )
        outputs = []
        for index in range(6):
            # Include absent pose, decoded image input, JPEG input, and a duplicate
            # frame key after several updates of the same tracked gate scene.
            frame_index = min(index, 4)
            current_image = cv2.warpAffine(
                image, np.float32([[1, 0, -frame_index * 2], [0, 1, 0]]),
                (image.shape[1], image.shape[0]),
            )
            encoded, current_jpeg = cv2.imencode(".jpg", current_image)
            assert encoded
            frame = schema.VisionFrame(
                200 + frame_index, 2000000000 + frame_index * 50000000,
                current_jpeg.tobytes(), image=current_image if frame_index % 2 else None,
            )
            state = None if index == 0 else schema.VehicleState(
                sim_time_ns=frame.sim_time_ns,
                position_local_ned_m=(0.04 * frame_index, 0.02 * frame_index, -1.0),
                velocity_local_ned_mps=(0.8, 0.4, 0.0),
                attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
                body_rates_frd_rps=(0.0, 0.0, 0.0),
                acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
            )
            if index == 2:
                observation = instance.process_frame(
                    frame_id=frame.frame_id, sim_time_ns=frame.sim_time_ns,
                    jpeg_bytes=frame.jpeg_bytes, vehicle_state=state,
                )
            else:
                observation = instance.process_vision_frame(frame, vehicle_state=state)
            implementation = getattr(instance, backend)
            outputs.append({
                **payload(observation, schema),
                "processed_frames": implementation._detector.frame_count,
                "detected_ellipses": implementation._detector.ellipse_count,
                "next_track_id": implementation._detector.tracker.next_id,
            })
        result["deterministic"][backend] = outputs
        instance.shutdown()
    return result


if __name__ == "__main__":
    print(json.dumps(exercise(sys.argv[1]), sort_keys=True, allow_nan=False))
