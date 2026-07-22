import json
import sys
import tempfile
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = APP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flight_bridge.vision_observation import VisionObservation  # noqa: E402


def instance_payload(**overrides):
    payload = {
        "observationId": "obs-000123-001",
        "instanceId": "instance-000001",
        "observationQuality": 0.93,
        "pose": {
            "available": True,
            "xyzCameraM": [1.0, 2.0, 3.0],
            "rpyCameraDeg": [-25.0, 3.0, -1.5],
            "depthM": 3.0,
            "fitQuality": {"overall": 0.79},
        },
    }
    payload.update(overrides)
    return payload


def frame_payload(instances=None):
    return {
        "schema": "0721vision-instance-frame.v1",
        "kind": "instance-frame-mapping-v1",
        "runId": "run-test",
        "frameOrdinal": 123,
        "frameId": "source-frame",
        "instances": list(instances if instances is not None else [instance_payload()]),
    }


class FlightBridgeObservationTests(unittest.TestCase):
    def test_controller_payload_round_trip_keeps_legacy_shape(self):
        payload = {
            "run": {"cycle": 7, "sim_time_ns": 42},
            "gates": [
                {
                    "id": "gate-a",
                    "position_xyz": [1, 2, 3],
                    "position_confidence": 0.5,
                    "orientation_xyz": [4, 5, 6],
                    "orientation_confidence": 0.25,
                }
            ],
            "obstacles": [],
        }
        observation = VisionObservation.from_controller_payload(payload)
        output = observation.to_controller_payload(output_dir="memory")
        self.assertEqual(output["run"]["cycle"], 7)
        self.assertEqual(output["run"]["sim_time_ns"], 42)
        self.assertEqual(output["gates"][0]["id"], "gate-a")
        self.assertEqual(output["gates"][0]["position_xyz"], [1.0, 2.0, 3.0])
        self.assertEqual(output["gates"][0]["orientation_xyz"], [4.0, 5.0, 6.0])
        self.assertEqual(output["obstacles"], [])

    def test_valid_instance_frame_produces_gate_observation(self):
        observation = VisionObservation.from_instance_frame(frame_payload())
        self.assertEqual(observation.frame_id, 123)
        self.assertEqual(observation.sim_time_ns, 0)
        self.assertEqual(observation.source, "vision_instance_mapping")
        self.assertEqual(len(observation.gates), 1)
        self.assertEqual(observation.gates[0].gate_id, "instance-000001")

    def test_position_conversion_flips_opencv_y_to_flight_up(self):
        observation = VisionObservation.from_instance_frame(frame_payload())
        self.assertEqual(observation.gates[0].position_camera_m, (1.0, -2.0, 3.0))

    def test_orientation_passthrough_is_preserved_for_v1(self):
        observation = VisionObservation.from_instance_frame(frame_payload())
        self.assertEqual(observation.gates[0].orientation_camera, (-25.0, 3.0, -1.5))
        self.assertEqual(observation.gates[0].orientation_confidence, 0.79)

    def test_invalid_pose_instances_are_skipped(self):
        payload = frame_payload(
            [
                instance_payload(instanceId="valid"),
                instance_payload(instanceId="missing-pose", pose={}),
                instance_payload(instanceId="not-available", pose={"available": False}),
                instance_payload(instanceId="bad-depth", pose={"available": True, "xyzCameraM": [1, 2, 3], "depthM": 0}),
                instance_payload(instanceId="bad-vector", pose={"available": True, "xyzCameraM": [1, 2], "depthM": 3}),
            ]
        )
        observation = VisionObservation.from_instance_frame(payload)
        self.assertEqual([gate.gate_id for gate in observation.gates], ["valid"])

    def test_confidence_values_are_clamped(self):
        high = instance_payload(instanceId="high", observationQuality=3.0)
        high["pose"]["fitQuality"]["overall"] = 2.0
        low = instance_payload(instanceId="low", observationQuality=-1.0)
        low["pose"]["fitQuality"]["overall"] = -5.0
        observation = VisionObservation.from_instance_frame(frame_payload([high, low]))
        self.assertEqual(observation.gates[0].position_confidence, 1.0)
        self.assertEqual(observation.gates[0].orientation_confidence, 1.0)
        self.assertEqual(observation.gates[1].position_confidence, 0.0)
        self.assertEqual(observation.gates[1].orientation_confidence, 0.0)

    def test_sim_time_override_and_camera_origin_defaults(self):
        observation = VisionObservation.from_instance_frame(frame_payload(), sim_time_ns=99)
        self.assertEqual(observation.sim_time_ns, 99)
        self.assertEqual(observation.camera_position_camera_m, (0.0, 0.0, 0.0))
        self.assertEqual(observation.camera_orientation_camera, (0.0, 0.0, 0.0))

    def test_multiple_instances_preserve_valid_input_order(self):
        first = instance_payload(instanceId="instance-a")
        skipped = instance_payload(instanceId="skip", pose={"available": False})
        second = instance_payload(instanceId="instance-b")
        observation = VisionObservation.from_instance_frame(frame_payload([first, skipped, second]))
        self.assertEqual([gate.gate_id for gate in observation.gates], ["instance-a", "instance-b"])

    def test_instance_frame_path_reader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "frame.json"
            path.write_text(json.dumps(frame_payload()), encoding="utf-8")
            observation = VisionObservation.from_instance_frame_path(path)
        self.assertEqual(observation.frame_id, 123)
        self.assertEqual(len(observation.gates), 1)


if __name__ == "__main__":
    unittest.main()
