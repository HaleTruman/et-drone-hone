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
│   ├── run_unreal.py                                # Run UE Scripts headless
│   ├── Artifacts/
│   │   └── runs/
│   │       └── <course>_<version>_<timestamp>.json # Per-run manifest: script metadata, command, log path, generated outputs
│   ├── Content_Generation/                          # UE Editor content-gen scripts (persisted assets/maps)
│   │   ├── Course_Content/
│   │   │   ├── README.md                            # Conventions, version/date policy, and run patterns for course generators
│   │   │   ├── asset_assembly.py                    # Shared create/load/save helpers for materials, meshes, and blueprints
│   │   │   ├── run_versioned_course_generation.py   # Wrapper: builds run folders, executes map generator, writes run manifest
│   │   │   ├── Materials/
│   │   │   │   └── gen_m_coursetorus_red.py         # Generate/reuse `M_CourseTorus_Red`
│   │   │   ├── Meshes/
│   │   │   │   └── gen_sm_coursetorus_1mopening.py  # Generate/reuse torus mesh asset
│   │   │   ├── Course_Blueprints/
│   │   │   │   └── gen_bp_courselight_main.py       # Generate/reuse `BP_CourseLight_Main` asset
│   │   │   └── Maps/
│   │   │       └── gen_l_coursetorus.py             # Assemble level: ensure light actor, place toruses, save map
│   │   ├── Drone_Content/                           # Drone asset generators (currently placeholder TODO scripts)
│   │   │   ├── Drone_Blueprints/
│   │   │   │   ├── gen_bp_dronepawn.py
│   │   │   │   ├── gen_bp_dronesensors.py
│   │   │   │   ├── gen_bp_dronemovement_6dof.py
│   │   │   │   └── gen_bp_dronetelemetrysampler.py
│   │   │   ├── Interfaces/
│   │   │   │   ├── gen_bpi_droneposeprovider.py
│   │   │   │   ├── gen_bpi_droneviewpointprovider.py
│   │   │   │   ├── gen_bpi_dronetelemetryprovider.py
│   │   │   │   └── gen_bpi_dronecommandreceiver.py
│   │   │   ├── Data/
│   │   │   │   ├── gen_da_dronemovementdefault.py
│   │   │   │   └── gen_da_sensorrigprofiledefault.py
│   │   │   ├── Materials/
│   │   │   │   ├── gen_m_dronebody_base.py
│   │   │   │   └── gen_mi_dronebody_default.py
│   │   │   └── Meshes/
│   │   │       ├── gen_sm_dronebody.py
│   │   │       └── gen_sm_dronecollisionproxy.py
│   │   └── io/                                      # IO asset generators (currently placeholder TODO scripts)
│   │       ├── Drone_Controller/
│   │       │   ├── gen_bp_dronespawner.py
│   │       │   └── gen_bp_dronecontroller.py
│   │       ├── Data_Interface/
│   │       │   └── gen_bp_samplemanager.py
│   │       └── Data_Config/
│   │           ├── gen_st_runconfig.py
│   │           └── gen_bp_setdataconfig.py
│   ├── Data_Interface/                              # TCP client + dataset writer (and/or file ingester)
│   │   ├── Sampler_Manager.py
│   │   ├── ...
│   │   └── Sampler.py
│   ├── Drone_Controller/                             # TCP client that sends commands (or later, operator runtime)
│   │   ├── Drone_Spawner.py
│   │   ├── ...
│   │   └── Drone_Controller.py                       # TCP operator endpoint.
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
│   │      ├── DroneWebSocket.uplugin
│   │      └── Source/
│   │          └── DroneWebSocket/
│   │              ├── DroneWebSocket.Build.cs           # Adds "WebSockets" dependency
│   │              ├── Public/
│   │              │   ├── WSClientComponent.h           # Blueprint ActorComponent: Connect/Send/Close + events
│   │              │   └── WSProtocolTypes.h             # Structs/enums for SET_CONFIG/CMD/OBS (optional)
│   │              └── Private/
│   │                  ├── WSClientComponent.cpp
│   │                  └── WSProtocolTypes.cpp
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
│   │   │   │   └── M_DroneBody_Default.uasset           # material for drone.
│   │   │   ├── Blueprints/
│   │   │   │   ├── BP_DronePawn.uasset                  # The drone container that composes modules
│   │   │   │   ├── BP_DroneSensors.uasset               # Sensor rig asset (mount transforms, camera mount(s))
│   │   │   │   ├── BP_DroneMovement_6DOF.uasset         # Movement module (ActorComponent): applies cmd → motion
│   │   │   │   └── BP_DroneTelemetrySampler.uasset      # Telemetry module (ActorComponent): samples pose/physics
│   │   │   ├── Interfaces/
│   │   │   │   ├── BPI_DronePoseProvider.uasset         # Contract: provide pose (+ velocity if available)
│   │   │   │   ├── BPI_DroneViewpointProvider.uasset    # Contract: provide viewpoint transforms (+ camera params)
│   │   │   │   ├── BPI_DroneTelemetryProvider.uasset    # Contract: provide extended telemetry (future-proof)
│   │   │   │   └── BPI_DroneCommandReceiver.uasset      # Contract: accept normalized control commands
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
├── model_training/
│   ├── datasets/                       # Logged episodes, trajectories, sensor streams (source of truth for offline).
│   │   ├── runs/
│   │   │   └── run_YYYYMMDD_HHMMSS/
│   │   │       ├── images/             # PNGs per drone/viewpoint keyed by capture_id.
│   │   │       ├── meta/               # observations.jsonl, captures.csv, episodes.jsonl.
│   │   │       └── manifests/          # manifest.json, schema version, capture profiles used.
│   │   └── schemas/                    # Dataset schema versions for backwards-compatible parsing.
│   ├── features/                       # Feature extraction (CNN preproc, RL state vectors, embeddings).
│   │   ├── build_cnn_tensors.py        # Load PNG+labels → normalized tensors → train-ready shards.
│   │   ├── build_rl_state_frames.py    # observations.jsonl → pandas dataframe → stacked state frames.
│   │   └── augmentations.py            # Controlled augments (crop/blur/noise) with deterministic seeds.
│   ├── trainers/                       # Offline training loops (SL/RL/IL).
│   │   ├── train_cnn.py                # Supervised vision training on disk datasets.
│   │   ├── train_policy_rl.py          # RL training using state frames (+ optional vision features).
│   │   └── train_imitation.py          # Imitation learning from expert trajectories.
│   └── eval/                           # Metrics, benchmarks, regression suites.
│       ├── eval_offline.py             # Offline eval on held-out runs (accuracy, success rate, drift).
│       └── regressions/                # Golden runs and expected metrics to prevent silent breakage.
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
