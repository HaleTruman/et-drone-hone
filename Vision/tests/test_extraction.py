"""Integration checks against the untouched Flight source in this repository."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


VISION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VISION_ROOT.parent


def run_python(args, source, cwd):
    environment = {**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, *args], cwd=cwd, env=environment,
        text=True, capture_output=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def implementations(tmp_path_factory):
    flight_source = REPO_ROOT / "Flight/src"
    if not (flight_source / "sensing/vision/service.py").exists():
        pytest.skip("Original Flight source is required for extraction parity checks")
    scratch = tmp_path_factory.mktemp("vision-parity")
    runner = Path(__file__).with_name("_parity_runner.py")
    return {
        target: json.loads(run_python([str(runner), target], source, scratch))
        for target, source in (("flight", flight_source), ("library", VISION_ROOT / "src"))
    }


def test_imports_are_standalone_and_do_not_change_sys_path(tmp_path):
    run_python(["-c", """
import importlib
import importlib.abc
from pathlib import Path
import sys

class RejectFlightImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition('.')[0] in {'core', 'sensing'}:
            raise AssertionError('Library imports Flight: ' + fullname)

sys.meta_path.insert(0, RejectFlightImports())
original_path = list(sys.path)
import aigp_vision
from aigp_vision import (
    VehicleState, VisionFrame, VisionGateObservation, VisionObservation,
    VisionPerceptionConfig, VisionPerceptionService,
)
for name in (
    'models.cnn.rgb_inference', 'models.cnn.regressor.logit_inference',
    'models.deterministic.service', 'models.deterministic_v2.service',
    'models.deterministic_v3.src.detection_vision',
    'models.deterministic_v3_2.src.detection_vision',
):
    importlib.import_module('aigp_vision.' + name)
assert sys.path == original_path
assert not any(name.partition('.')[0] in {'core', 'sensing'} for name in sys.modules)
assert VisionPerceptionConfig().backend == 'cnn_regressor'
assert VisionPerceptionService().snapshot()['cnn_loaded'] is False
assert Path(VisionPerceptionConfig().checkpoint).is_file()
assert Path(VisionPerceptionConfig().regressor_checkpoint).is_file()
"""], VISION_ROOT / "src", tmp_path)


@pytest.mark.parametrize("first", ["flight", "library"])
def test_flight_objects_and_library_coexist_in_either_import_order(tmp_path, first):
    flight_source = REPO_ROOT / "Flight/src"
    if not (flight_source / "sensing/vision/service.py").exists():
        pytest.skip("Original Flight source is required for coexistence checks")
    source_paths = os.pathsep.join(map(str, (flight_source, VISION_ROOT / "src")))
    run_python(["-c", """
import importlib
from pathlib import Path
import sys

modules = {}
original_modules = {}
for target in (sys.argv[1], 'library' if sys.argv[1] == 'flight' else 'flight'):
    package = 'sensing.vision' if target == 'flight' else 'aigp_vision'
    modules[target] = (
        importlib.import_module(package + '.service'),
        importlib.import_module('core.schema' if target == 'flight' else 'aigp_vision.schema'),
        importlib.import_module(package + '.models.deterministic.service'),
    )
    for name, module in original_modules.items():
        assert sys.modules[name] is module, 'Import replaced module: ' + name
    original_modules.update({
        name: module for name, module in sys.modules.items()
        if name.partition('.')[0] in {'core', 'sensing', 'aigp_vision'}
    })

flight_service, flight_schema, flight_v1 = modules['flight']
service, schema, library_v1 = modules['library']
assert flight_schema is sys.modules['core.schema']
assert flight_schema.VisionFrame is not schema.VisionFrame
assert flight_service.VisionPerceptionService is not service.VisionPerceptionService
assert flight_v1.DeterministicVisionBackend is not library_v1.DeterministicVisionBackend
assert flight_v1.VisionFrame is flight_schema.VisionFrame
assert library_v1.VisionFrame is schema.VisionFrame
assert all(
    module.__name__ == name for name, module in original_modules.items()
    if name.partition('.')[0] in {'core', 'sensing'}
)
jpeg = (Path(service.__file__).parent / 'models/deterministic_v2/assets'
        / 'frame-00068527-1784514770942906700.jpg').read_bytes()
frame = flight_schema.VisionFrame(19, 950000000, jpeg)
state = flight_schema.VehicleState(
    sim_time_ns=frame.sim_time_ns,
    position_local_ned_m=(0.0, 0.0, -1.0),
    velocity_local_ned_mps=(0.0, 0.0, 0.0),
    attitude_quaternion=(1.0, 0.0, 0.0, 0.0),
    body_rates_frd_rps=(0.0, 0.0, 0.0),
    acceleration_local_ned_mps2=(0.0, 0.0, 0.0),
)
vision = service.VisionPerceptionService(
    service.VisionPerceptionConfig(backend='deterministic_v3_2', device='cpu')
)
observation = vision.process_vision_frame(frame, vehicle_state=state)
assert isinstance(observation, schema.VisionObservation)
assert not isinstance(observation, flight_schema.VisionObservation)
assert (observation.frame_id, observation.sim_time_ns) == (frame.frame_id, frame.sim_time_ns)
assert observation.trace['pose_available'] is True
assert observation.trace['detections'] > 0
for name, module in original_modules.items():
    assert sys.modules[name] is module
vision.shutdown()
""", first], source_paths, tmp_path)


def test_public_schema_and_frame_contracts_match_flight(implementations):
    original = implementations["flight"]["contract"]
    assert implementations["library"]["contract"] == original
    assert original["default_backend"] == "cnn_regressor"
    assert original["source"] == "fixture"
    assert original["observation"]["run"]["cycle"] == 17
    assert original["observation"]["gates"][0]["position_local_ned"] == [1.0, 2.0, 3.0]
    assert len(original["errors"]) == 3


@pytest.mark.parametrize("mode", ["landmarker", "passthrough", "regressor"])
def test_cpu_cnn_outputs_and_traces_match_flight(implementations, mode):
    original = implementations["flight"]["cnn"][mode]
    assert implementations["library"]["cnn"][mode] == original
    assert [output["controller"]["run"]["cycle"] for output in original] == [100, 101]
    expected_source = "cnn_regressor" + ("" if mode == "regressor" else "_" + mode)
    assert all(output["source"] == expected_source for output in original)
    assert all(output["controller"]["gates"] for output in original)


@pytest.mark.parametrize("backend", ["deterministic_v3", "deterministic_v3_2"])
def test_stateful_outputs_and_traces_match_flight(implementations, backend):
    original = implementations["flight"]["deterministic"][backend]
    assert implementations["library"]["deterministic"][backend] == original
    assert [output["processed_frames"] for output in original] == list(range(1, 7))
    assert original[0]["controller"]["trace"]["pose_available"] is False
    assert original[1]["controller"]["trace"]["pose_available"] is True
    assert any(output["controller"]["trace"]["detections"] > 0 for output in original)
    assert any(output["controller"]["trace"]["cached_tracks"] > 0 for output in original)
    assert any(output["controller"]["gates"] for output in original)
    assert original[-1]["controller"] == original[-2]["controller"]
