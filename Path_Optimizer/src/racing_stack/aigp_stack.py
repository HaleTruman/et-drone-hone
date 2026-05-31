from typing import Any


class AIGPStack:
    def __init__(
        self,
        *,
        mavlink_bridge: Any,
        vision_stream: Any,
        synchronizer: Any,
        gate_pose_estimator: Any,
        gate_map: Any,
        state_estimator: Any,
        path_manager: Any,
        planner: Any,
        controller: Any,
        command_mapper: Any,
        flight_state: Any,
        logger: Any,
    ):
        self.mavlink_bridge = mavlink_bridge
        self.vision_stream = vision_stream
        self.synchronizer = synchronizer
        self.gate_pose_estimator = gate_pose_estimator
        self.gate_map = gate_map
        self.state_estimator = state_estimator
        self.path_manager = path_manager
        self.planner = planner
        self.controller = controller
        self.command_mapper = command_mapper
        self.flight_state = flight_state
        self.logger = logger

    def run_control_loop(self) -> None:
        raise NotImplementedError("Wire the selected transport, perception model, planner, and controller policies.")

    def shutdown(self) -> None:
        self.vision_stream.shutdown()
        self.mavlink_bridge.shutdown()
