Build minimal 

see docs/architecture.md for complete archetecture 

Minimal Repo structure to make drone fly through tauruses (not real archetecture)

Brendan:
├── path_optimizer/      #Brendan
│   ├── physics/         #Brendan               # model physical constrains
│   ├── course_model/    #Truman                # model course constrains 
│   ├── optimizer/       #Brendan               # Cost functions, constraints (physics + course_model), gradients, splines, Manafold Optimization
│   └── planner/         #Brendan               # Trajectory & path solver using physics + course_model + optimizer
├── ue_tooling/      #Truman                    # Accept operator control for tcp
├── unreal_project/      #Truman                # Unreal Engine project (sim world + operator).
│   └── Plugins/         #Truman                # Blueprint/C++ tcp operator endpoint.
├── model_training/      #Truman                # In --> out
├── operator/            #Brendan               # Drone input controler g


Current goals:

Truman: UE scripting --> Data --> CNN --> JSON --> drone builds
Brendan: Optimizer, --> Trailor to JSON-(in-out)

Collab on JSON