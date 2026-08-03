Build minimal 

see docs/architecture.md for complete archetecture 

Minimal Repo structure to make drone fly through tauruses (not real archetecture)

Brendan:
├── Flight/              #Brendan
│   ├── physics/         #Brendan               # model physical constrains
│   ├── optimizer/       #Brendan               # Cost functions, constraints, gradients, splines.
│   └── planner/         #Brendan               # Trajectory & path solver using physics + optimizer
├── ue_automations/      #Truman                # Accept operator control for http
├── unreal_project/      #Truman                # Unreal Engine project (sim world + operator).
│   └── Plugins/         #Truman                # Blueprint/C++ HTTP operator endpoint.
├── model_training/      #Truman                # In --> out
├── operator/            #Brendan               # Drone input controler g
