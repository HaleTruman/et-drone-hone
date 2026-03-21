```text
et-drone-hone/
├── README.md                           # What this repo is and how to run it.
├── docs/
│   ├── scope.md                        # v0 scope and explicit non-goals.
│   └── architecture.md                 # Data/control flow across modules.
├── path_optimizer/
│   ├── physics/                        # model physical constrains
│   │   ├── local/                      # Per-prop thrust/torque + mixing model.
│   │   └── global/                     # Vehicle-level limits abstracted from local.
│   ├── course_model/                   # model course constrains 
│   │   ├── targets/                    # Rings/gates/waypoints as goal volumes.
│   │   ├── obstacles/                  # No-go volumes with uncertainty weights.
│   │   └── free_space/                 # Known traversable regions.
│   ├── optimizer/                      # Cost functions, constraints (physics + course_model), gradients, splines.
│   └── planner/                        # Trajectory & path solver using physics + course_model + optimizer
│
├── UE_Tooling/
│   ├── Artifacts/
│   │   ├── build/
│   │   │   └── <run_name>.json                     # Canonical top-level build manifest from run_unreal_build.py (authoritative level + stage statuses/artifacts + IO placement summary)
│   │   ├── build_runs/
│   │   │   └── <run_name>/run_manifest.json        # Compatibility mirror of top-level build manifest (legacy path retained during refactor)
│   │   ├── runs/
│   │   │   └── <course>_<version>_<timestamp>.json # Course build manifest: script metadata, command, log path, generated outputs
│   │   ├── drone_content/
│   │   │   └── <kit>_<version>_<timestamp>.json    # Drone build manifest: ordered generators, per-script metadata, logs, results
│   │   ├── io_build/
│   │   │   └── <kit>_<version>_<timestamp>.json    # IO build manifest: ordered scripts, per-script metadata, target level, placement result
│   │   └── websocket_build/
│   │       └── <run_name>/run_manifest.json        # WebSocket build manifest: step metadata, bootstrap/validation results, chosen build inputs
│   ├── run_unreal_build.py                         # Canonical top-level build orchestrator: WebSocket -> Course -> Drone -> IO (fail-fast)
│   ├── run_unreal_build_gen.py                     # Root compatibility shim; canonical low-level executor lives under UE_Build/
│   ├── run_unreal_runtime_io.py                    # Runtime orchestration placeholder (target rename: run_unreal_io.py)
│   ├── UE_Build/
│   │   ├── run_unreal_build_gen.py                  # Run UE build-generation scripts headless
│   │   ├── Content_Generation/                      # UE Editor content-gen scripts (persisted assets/maps)
│   │   │   ├── Course/                              # Mirrors Drone/IO orchestration + assembly naming (in development)
│   │   │   │   ├── README.md                        # Conventions, version/date policy, and mirrored build-layer naming for Course
│   │   │   │   ├── assemble_course_assets.py        # Shared Course-domain assembly executor; mirrors Drone/IO/WebSocket helper naming
│   │   │   │   ├── run_course_build.py              # Course build orchestration runner; mirrors Drone/IO/WebSocket runner naming
│   │   │   │   ├── Materials/
│   │   │   │   │   └── gen_m_coursetorus_red.py
│   │   │   │   ├── Meshes/
│   │   │   │   │   └── gen_sm_coursetorus_1mopening.py
│   │   │   │   ├── Course_Blueprints/
│   │   │   │   │   └── gen_bp_courselight_main.py
│   │   │   │   └── Maps/
│   │   │   │       └── gen_l_coursetorus.py
│   │   │   ├── Drone/                               # Mirrors Course/IO orchestration + assembly naming (in development)
│   │   │   │   ├── assemble_drone_assets.py         # Shared Drone-domain assembly executor; mirrors Course/IO/WebSocket helper naming
│   │   │   │   ├── run_drone_build.py               # Drone build orchestration runner; mirrors Course/IO/WebSocket runner naming
│   │   │   │   ├── Drone_Blueprints/
│   │   │   │   ├── Interfaces/
│   │   │   │   ├── Structs/                         # Explicit struct schema generators; mirrors the .uasset contracts in /Game/Drone_Content/Structs
│   │   │   │   ├── Data/
│   │   │   │   ├── Materials/
│   │   │   │   └── Meshes/
│   │   │   └── IO/                                  # Mirrors Course/Drone orchestration + assembly naming, with map placement handled in the build runner
│   │   │       ├── assemble_io_assets.py            # Shared IO-domain assembly executor; mirrors Course/Drone/WebSocket helper naming
│   │   │       ├── run_io_build.py                  # IO build orchestration runner; records build provenance and performs deterministic singleton placement for BP_SetDataConfig and BP_SampleManager
│   │   │       ├── Drone_Controller/
│   │   │       │   ├── gen_bp_dronespawner.py       # Generates BP_DroneSpawner (runtime SPAWN_DRONES entrypoint in /Game/io)
│   │   │       │   └── gen_bp_dronecontroller.py    # Generates BP_DroneController (runtime CMD router to drone command receiver)
│   │   │       ├── Data_Interface/
│   │   │       └── Data_Config/
│   │   └── WebSocket/                               # Plugin source-of-truth + bootstrap/validation helpers for UE plugin
│   │       ├── README.md
│   │       ├── AGENTS.md                           # Unreal build guidance for working effectively with plugin bootstrap/validation scripts
│   │       ├── Plugin_Source/
│   │       │   ├── README.md                       # Source-of-truth notes for editing and bootstrapping the Unreal plugin
│   │       │   └── DroneWebSocket/                 # Hand-edited plugin descriptor, config, and C++ source used by bootstrap/build
│   │       ├── assemble_websocket_assets.py         # Shared WebSocket build executor; mirrors Course/Drone/IO helper naming
│   │       ├── bootstrap_plugin.py                  # Canonical project-plugin bootstrap/build entrypoint (keep existing unless overwrite is requested)
│   │       ├── plugin_validate.py                   # Validates source sync, descriptor compatibility, build outputs, and headless startup readiness
│   │       ├── run_websocket_build.py               # WebSocket build orchestration runner; mirrors Course/Drone/IO runner naming
│   ├── Data_Interface/                              # TCP client + dataset writer (and/or file ingester)
│   │   ├── Sampler_Manager.py
│   │   ├── ...
│   ├── Drone_Controller/                             # Tooling-side runtime command emitters via websocket bridge
│   │   ├── Drone_Spawner.py                          # Builds/sends SPAWN_DRONES requests for episode setup
│   │   └── Drone_Controller.py                       # Builds/sends CMD requests keyed by drone_id
│   ├── Config/
│   │   ├── Data_Interface_Config/
│   │   │   ├── SensorRigProfileConfig_Pose.yaml
│   │   │   ├── SensorRigProfileConfig_Viewpoint.yaml
│   │   │   └── SensorRigProfileConfig_Telemetry.yaml
│   │   ├── Drone_Controller_Config/
│   │   │   └── Config_DroneMovementTuning.yaml
│   │   └── RunConfig.py                              # Payload SET_CONFIG JSON 
│   │
│   ├── WebSocket/
│   │   ├── ws_bridge.py                 # One-process bridge: connect ↔ send SET_CONFIG/CMD ↔ recv OBS/STATUS
│   │   ├── protocol.py                  # Message framing + types (SET_CONFIG, CMD, CAPTURE_NOW, OBS, ACK, ERROR)
│   │   ├── schemas.py                   # JSON schema/validation + versioning (optional but nice)
│   │   └── README.md                    # How to run locally + ports/URLs
│   │
│   └── AGENTS.md                                    # md file to help with automation blockers
│
├── UE_Drone_Env/                                  # Unreal Engine project (merged: scaffold + generated structure)
│   ├── UE_Drone_Env.uproject                      # UE project descriptor (generated by UE)
│   ├── UE_Drone_Env (Mac).xcworkspace/            # macOS workspace (generated by UE on Mac)
│   ├── AGENTS.md
│   ├── UE_TermsParams.md                          # small dictionary of Unreal Engine
│   ├── Config/                                    # UE project configuration.
│   │   ├── DefaultEditor.ini                      # UE-generated editor defaults
│   │   ├── DefaultEngine.ini                      # UE-generated engine defaults
│   │   ├── DefaultGame.ini                        # UE-generated game defaults
│   │   └── DefaultInput.ini                       # UE-generated input defaults
│   ├── Plugins/
│   │   └── DroneWebSocket/
│   │      ├── Binaries/
│   │      │   └── Mac/                            # Built editor plugin outputs emitted by websocket bootstrap/build validation flow
│   │      ├── Config/
│   │      │   └── FilterPlugin.ini                 # Plugin packaging/filter config installed by websocket bootstrap/build
│   │      ├── DroneWebSocket.uplugin               # Project-installed plugin descriptor built from tooling-side Plugin_Source
│   │      └── Source/
│   │          └── DroneWebSocket/
│   │              ├── DroneWebSocket.Build.cs           # Adds "WebSockets" dependency
│   │              ├── Public/
│   │              │   ├── WSClientComponent.h           # Blueprint ActorComponent: Connect/Send/Close + events
│   │              │   ├── WSConfigHandshakeActor.h      # Runtime config/action ingress actor used for minimal websocket handshake flow
│   │              │   └── WSProtocolTypes.h             # Shared structs/helpers for websocket envelope and payload parsing
│   │              └── Private/
│   │                  ├── DroneWebSocketModule.cpp      # Minimal module registration for the Unreal plugin
│   │                  ├── WSClientComponent.cpp         # WebSocket client connect/send/receive implementation
│   │                  ├── WSConfigHandshakeActor.cpp    # Minimal runtime handshake/control path used during websocket validation
│   │                  └── WSProtocolTypes.cpp           # JSON envelope build/parse implementation for shared protocol types
│   ├── Source/                                    # UE C++ module targets (generated)
│   │   ├── UE_Drone_Env.Target.cs
│   │   └── UE_Drone_EnvEditor.Target.cs
│   ├── Content/                                   # Maps/assets/blueprints
│   │   ├── Collections/                           # UE-managed content collections (generated)
│   │   ├── Developers/                            # UE per-user/dev content area (generated)
│   │   ├── Course_Content/                        # Course assets
│   │   │   ├── M_Red_Solid.uasset
│   │   │   ├── ...
│   │   │   └── SM_Sphere_1m.uasset
│   │   ├── Drone_Content/                         # “Drone kit” (portable, minimal dependencies)
│   │   │   ├── Meshes/
│   │   │   │   ├── SM_DroneBody.uasset                  # Drone mesh (visual body)
│   │   │   │   └── SM_DroneCollisionProxy.uasset        # (optional) simple collision proxy for stable physics later
│   │   │   ├── Materials/
│   │   │   │   ├── M_DroneBody_Base.uasset              # Base material definition for the drone body.
│   │   │   │   └── MI_DroneBody_Default.uasset          # Default material instance applied to the generated drone body mesh.
│   │   │   ├── Blueprints/
│   │   │   │   ├── BP_DronePawn.uasset                  # The drone container that composes modules
│   │   │   │   ├── BP_DroneSensors.uasset               # Canonical viewpoint/image runtime module
│   │   │   │   ├── BP_DroneMovement_6DOF.uasset         # Movement module (ActorComponent): applies cmd → motion
│   │   │   │   └── BP_DroneTelemetrySampler.uasset      # Canonical telemetry runtime module; pose converges here too
│   │   │   ├── Interfaces/
│   │   │   │   ├── BPI_DroneViewpointProvider.uasset    # Canonical viewpoint query contract
│   │   │   │   ├── BPI_DroneTelemetryProvider.uasset    # Canonical telemetry query contract (pose converges here too)
│   │   │   │   └── BPI_DroneCommandReceiver.uasset      # Contract: accept normalized control commands
│   │   │   ├── Structs/
│   │   │   │   ├── ST_DroneCommandNormalized.uasset      # Typed command payload schema (pitch/roll/yaw/throttle + trace ids).
│   │   │   │   ├── ST_DroneViewpointSnapshot.uasset      # Typed viewpoint/camera settings schema for active mount.
│   │   │   │   └── ST_DroneTelemetrySnapshot.uasset      # Canonical top-level telemetry snapshot schema (pose + telemetry + proximity).
│   │   │   └── Data/
│   │   │       ├── DA_DroneMovementDefault.uasset       # Params: max speed, accel, rate limits, damping
│   │   │       └── DA_SensorRigProfileDefault.uasset    # Params: default FOVs, mount offsets, names
│   │   ├── io/
│   │   │   ├── Drone_Controller/
│   │   │   │   ├── BP_DroneController.uasset            # DroneId → apply command → movement component TCP operator endpoint.
│   │   │   │   └── BP_DroneSpawner.uasset               # TCP operator endpoint.
│   │   │   ├── Data_Interface/                          # Data extraction for training
│   │   │   │   └── BP_SampleManager.uasset              # Atomic snapshot: capture_id: telemetry, vision, pose, config, time
│   │   │   └── Data_Config/
│   │   │       ├── ST_RunConfig.uasset
│   │   │       └── BP_SetDataConfig.uasset              # RuntimeOverrideConfig, ApplyConfigNow(), accept a SET_CONFIG JSON payload TCP operator endpoint.
│   │   └── AGENTS.md                                    # md file to help with automation blockers
│   ├── Build/                                     # UE build metadata and toolchain files (generated)
│   ├── Binaries/                                  # Compiled output binaries (generated)
│   ├── DerivedDataCache/                          # Derived asset cache (generated)
│   ├── Intermediate/                              # Intermediate build artifacts (generated)
│   └── Saved/                                     # Logs/autosaves/runtime files (generated)
│
│   #DRAFTING
│
├── Model_Training/
│   ├── data/                           # MVP: direct on-disk image+label inputs; extensible seam for later upstream imports.
│   │   ├── dummy/
│   │   │   ├── images/                 # MVP: synthetic PNGs for smoke-testing the training loop.
│   │   │   └── labels.jsonl            # MVP: simple labels keyed by sample_id/image_path.
│   │   └── imported/                   # FUTURE: mirrored upstream runtime exports once the data share contract is stable.
│   │       └── run_YYYYMMDD_HHMMSS/
│   │           ├── images/             # FUTURE: PNGs per drone/viewpoint keyed by capture_id or sample_id.
│   │           ├── meta/               # FUTURE: observations.jsonl, labels.jsonl, manifests, and traceability records.
│   │           └── schemas/            # FUTURE: dataset schema versions for backwards-compatible parsing.
│   ├── datasets/                       # MVP: thin dataset/loaders that map disk data -> training samples.
│   │   ├── simple_image_dataset.py     # MVP: PNG + labels.jsonl -> Dataset/DataLoader contract.
│   │   └── ue_artifact_shim.py         # FUTURE: light shim from UE artifacts into the same sample contract.
│   ├── models/                         # MVP: training-time model definitions.
│   │   └── simple_cnn.py               # MVP: small CNN baseline for gate-steering image classification.
│   ├── features/                       # Shared preprocessing and augmentation utilities.
│   │   ├── image_transforms.py         # MVP: resize, normalize, tensor conversion.
│   │   ├── augmentations.py            # FUTURE: controlled augments (crop/blur/noise) with deterministic seeds.
│   │   ├── build_cnn_tensors.py        # FUTURE: optional PNG+labels -> train-ready tensor shards.
│   │   └── build_rl_state_frames.py    # FUTURE: observations.jsonl -> stacked RL state frames.
│   ├── trainers/                       # Offline training loops.
│   │   ├── train_cnn.py                # MVP: supervised vision training on labeled image datasets.
│   │   ├── train_policy_rl.py          # FUTURE: RL training using state frames (+ optional vision features).
│   │   └── train_imitation.py          # FUTURE: imitation learning from expert trajectories.
│   ├── eval/                           # Offline evaluation and regression checks.
│   │   ├── eval_cnn.py                 # MVP: held-out classifier evaluation.
│   │   ├── eval_offline.py             # FUTURE: broader offline eval (accuracy, success rate, drift).
│   │   └── regressions/                # FUTURE: golden runs and expected metrics to prevent silent breakage.
│   └── artifacts/                      # MVP: saved configs/checkpoints/metrics; FUTURE: richer exports and debug outputs.
│       └── run_YYYYMMDD_HHMMSS/
│           ├── config.json             # MVP: saved training config for reproducibility.
│           ├── metrics.json            # MVP: train/validation metrics summary.
│           ├── best_model.pt           # MVP: best checkpoint by validation metric.
│           ├── last_model.pt           # MVP: final checkpoint from the run.
│           └── predictions.jsonl       # FUTURE: sample-level predictions for debugging and regression review.
│
├── operator/
│   ├── http_bridge/                    # Python utilities for in-engine HTTP control for unreal + any endpoint
│   ├── runtime/                        # Live loop: perceive → plan → act → log.
│   ├── models/                         # Inference wrappers + model artifacts.
│   ├── optimizer/                      # Glue to call path_optimizer with beliefs.
│   └── interfaces/                     # UE sim IO, real-drone IO, logging shims.
├── learning/
│   ├── online/                         # Live adaptation and drift handling.
│   └── calibration/                    # Camera pose + kinematics estimation routines.
└── controller/
    ├── fusion/                         # Blend vision + map + priors into beliefs.
    ├── arbitration/                    # Choose: maneuver, replan, slow-recover, abort.
    └── safety/                         # Hard limits, geofences, failsafes, kill-switch.
```
