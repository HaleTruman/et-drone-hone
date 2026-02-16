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
├── ue_automations/
│   ├── extraction/                     # Export meshes, transforms, and metadata from UE.
│   ├── course_build/                   # Generate/modify courses inside UE from specs.
│   └── http_bridge/                    # Python utilities for in-engine HTTP control.
├── unreal_project/                     # Unreal Engine project (sim world + operator).
│   ├── Config/                         # UE project configuration.
│   ├── Content/                        # Maps/assets/course templates (UE-managed).
│   └── Plugins/
│       └── DroneHttpOperator/          # Blueprint/C++ HTTP operator endpoint.
├── model_training/
│   ├── datasets/                       # Logged episodes, trajectories, sensor streams.
│   ├── features/                       # Feature extraction for training/eval.
│   ├── trainers/                       # Offline training loops (SL/RL/IL).
│   └── eval/                           # Metrics, benchmarks, regression suites.
├── operator/
│   ├── runtime/                        # Live loop: perceive → plan → act → log.
│   ├── models/                         # Inference wrappers + model artifacts.
│   ├── optimizer/                      # Glue to call path_optimizer with beliefs.
│   └── interfaces/                     # UE sim IO, real-drone IO, logging shims.
├── learning/
│   ├── online/                         # Live adaptation and drift handling.
│   └── calibration/                    # Camera pose + kinematics estimation routines.
├── maneuvers/
│   ├── library/                        # Macro-actions (commit-to-ring, snap-turn, etc.).
│   └── selectors/                      # When to use maneuvers vs continuous planning.
└── controller/
    ├── fusion/                         # Blend vision + map + priors into beliefs.
    ├── arbitration/                    # Choose: maneuver, replan, slow-recover, abort.
    └── safety/                         # Hard limits, geofences, failsafes, kill-switch.


```mermaid
flowchart TB

  subgraph ROOT[et-drone-hone]
    README[README.md]
    subgraph DOCS[docs]
      SCOPE[scope.md]
      ARCH[architecture.md]
    end

    subgraph PO[path_optimizer]
      subgraph PHYS[physics]
        PHYS_L[local]
        PHYS_G[global]
      end

      subgraph CM[course_model]
        CM_T[targets]
        CM_O[obstacles]
        CM_F[free_space]
      end

      OPT[optimizer]
      PLAN[planner]

      PHYS_L --> PHYS_G
      PHYS_L --> OPT
      PHYS_G --> OPT
      CM_T --> OPT
      CM_O --> OPT
      CM_F --> OPT
      OPT --> PLAN
    end

    subgraph UEA[ue_automations]
      UEA_X[extraction]
      UEA_B[course_build]
      UEA_H[http_bridge]
    end

    subgraph UEP[unreal_project]
      UEP_CFG[Config]
      UEP_CNT[Content]
      subgraph UEP_PLUG[Plugins]
        UEP_HTTP[DroneHttpOperator]
      end
    end

    subgraph MT[model_training]
      MT_D[datasets]
      MT_F[features]
      MT_T[trainers]
      MT_E[eval]
    end

    subgraph OP[operator]
      OP_R[runtime]
      OP_M[models]
      OP_O[optimizer]
      OP_I[interfaces]
    end

    subgraph LE[learning]
      LE_O[online]
      LE_C[calibration]
    end

    subgraph MV[maneuvers]
      MV_L[library]
      MV_S[selectors]
    end

    subgraph CT[controller]
      CT_F[fusion]
      CT_A[arbitration]
      CT_S[safety]
    end
  end

  %% Cross-module flow (high-level)
  UEA_X --> CM
  UEA_B --> UEP_CNT
  UEA_H --> UEP_HTTP

  CM --> PO
  PHYS --> PO
  PLAN --> OP_O
  OP_O --> OP_R

  LE_O --> OP_R
  LE_C --> OP_R

  MV_S --> CT_A
  CT_F --> CT_A
  CT_S --> CT_A
  CT_A --> OP_R

  OP_I --> UEP_HTTP
  OP_R --> MT_D
  MT_F --> MT_T
  MT_T --> OP_M
  MT_E --> MT_T
