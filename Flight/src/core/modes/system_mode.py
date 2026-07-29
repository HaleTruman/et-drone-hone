from enum import Enum


class SystemMode(str, Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"
    RACING = "RACING"
    FINISHED = "FINISHED"
    FAULT = "FAULT"


class SystemModeManager:
    def __init__(self):
        self.system_mode = SystemMode.IDLE
        self.fault_reason: str | None = None

    def update_mode(self, event: str) -> SystemMode:
        transitions = {
            (SystemMode.IDLE, "arm"): SystemMode.ARMED,
            (SystemMode.ARMED, "start"): SystemMode.RACING,
            (SystemMode.RACING, "finish"): SystemMode.FINISHED,
        }
        self.system_mode = transitions.get((self.system_mode, event), self.system_mode)
        return self.system_mode

    def is_armed(self) -> bool:
        return self.system_mode == SystemMode.ARMED

    def is_racing(self) -> bool:
        return self.system_mode == SystemMode.RACING

    def is_finished(self) -> bool:
        return self.system_mode == SystemMode.FINISHED

    def handle_fault(self, reason: str) -> SystemMode:
        self.fault_reason = reason
        self.system_mode = SystemMode.FAULT
        return self.system_mode
