"""
RunConfig is the config compiler from YAML inputs to one deterministic `SET_CONFIG` payload.

Primary function:
- Load and merge config sources for pose, viewpoint, telemetry, and movement tuning.
- Emit one canonical runtime payload used by UE runtime initialization/apply flows.
- Stamp traceability metadata (`config_id`, `config_hash`, `schema_version`) so runs are auditable.

Pipeline relationships:
- Inputs:
  - `Data_Interface_Config/SensorRigProfileConfig_Pose.yaml`
  - `Data_Interface_Config/SensorRigProfileConfig_Viewpoint.yaml`
  - `Data_Interface_Config/SensorRigProfileConfig_Telemetry.yaml`
  - `Drone_Controller_Config/Config_DroneMovementTuning.yaml`
- Outputs:
  - Shared `SET_CONFIG` JSON consumed by bridge/controller/sampler workflows.
- Must remain the single source of truth for YAML -> JSON mapping decisions.

UE/runtime relationships:
- UE runtime receives only the compiled JSON payload (not YAML) and applies settings in memory.

Current status:
- Documentation scaffold only. Compile/merge implementation is intentionally pending.
"""
