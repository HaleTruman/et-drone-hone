# Instructions for Codex: Build Barebones MPCC Planner

Create a clean, modular, minimal Python project implementing a barebones MPCC (Model Predictive Contouring Control) planner for the quadrotor racing scenario.

The planner must use the exact 13-state nonlinear quadrotor dynamics from the provided `Path_Optimizer/docs/resources/math/quadrotor_nonlinear_model.md` file.

**Output requirements:**

- Absolutely barebones code: only the minimal lines needed for correct functionality. No type hints, no unnecessary comments, no docstrings, no fluff, no logging beyond required debug output.
- Split into small, single-responsibility files.
- Use only these dependencies: `numpy`, `scipy`, `casadi`, `matplotlib`. Assume they are installed.
- Hardcode all parameters (mass, inertia, kf, km, Jr, l, h, drag coeffs, etc.) at the top of `quadrotor_dynamics.py` using realistic values for the 280mm drone chassis described in the technical spec.
- Discretize dynamics with RK4, horizon N=20, dt=0.05 s.
- MPCC must enforce hard actuator limits 0 ≤ u_i ≤ 1 for all four motors inside the optimization.
- Planner output: quaternion (qw, qx, qy, qz) and normalized collective thrust (0-1) ready for MAVLink SET_ATTITUDE_TARGET.
- Include terminal debug output in the solver: solve time (seconds), number of iterations, solver status, final cost.
- Include a visualization tool that plots: initial guess path, gate positions/orientations (as 1.5m inner square wireframes), optimized trajectory, velocity vectors, orientation arrows, acceleration, and per-motor thrust over time. Use matplotlib with multiple subplots.

## Project Structure

Create exactly these files in `Path_Optimizer/src/drone_mpcc_planner` (no others):

1. `quadrotor_dynamics.py`
2. `reference_path.py`
3. `mpcc_problem.py`
4. `mpcc_solver.py`
5. `visualizer.py`
6. `planner.py`
7. `demo.py`

## File 1: quadrotor_dynamics.py

Implement the full 13-state nonlinear quadrotor model exactly as derived in `Path_Optimizer/docs/resources/math/quadrotor_nonlinear_model.md` (position, inertial velocity, quaternion, body rates).

- Function `def f(x, u):` returns dx/dt (13x1) using the exact equations: position kinematics, velocity dynamics with R(q)\*F^B + gravity, quaternion kinematics with Xi(q), rotational Euler equations with thrust moments, motor torques, gyroscopic moments, quadratic drag.
- Include helper functions: `rotation_matrix(q)`, `quaternion_to_euler(q)`, `thrust_allocation(u)`, `compute_total_thrust(u)`.
- All parameters (m, Ixx/Iyy/Izz, kf, km, Jr, li, hi, cdx/cdy/cdz, crx/cry/crz, g=9.81) hardcoded at top with sensible defaults for 280mm X-config drone.
- No simulation loop — just the continuous dynamics function.

## File 2: reference_path.py

Create a simple local reference path generator from 1-3 visible gate poses.

- Gates provided as list of dicts: {'pos': [x,y,z], 'quat': [qw,qx,qy,qz]} (center + orientation).
- Function `def generate_reference(gates, num_points=50):` returns parametric path (theta, px, py, pz) as cubic spline (use scipy.interpolate) through gate centers, arc-length parameterized.
- Include simple gate visualization points (four corners of 1.5m inner square using gate orientation).

## File 3: mpcc_problem.py

Define the CasADi MPCC optimal control problem.

- Function `def setup_mpcc_problem(dynamics_func, N=20, dt=0.05):` creates CasADi Opti object.
- Variables: states (13 x (N+1)), controls (4 x N), progress variable theta (N+1).
- Dynamics constraints using RK4 discretization of the 13-state model.
- Hard actuator constraints: 0 ≤ u_i ≤ 1 for all motors and steps.
- Cost: contouring error + lag error + strong progress maximization term (-mu \* theta_N) + small control regularization.
- Contouring/lag errors computed w.r.t. reference path spline.
- Return the Opti object, variables, parameters (for reference and initial state).

## File 4: mpcc_solver.py

Solve the MPCC problem.

- Function `def solve_mpcc(opti, x0, reference, warm_start=None):` sets initial guess, parameters, solves with IPOPT (or default solver).
- Print exact debug output to terminal on every solve:
  - "MPCC solve time: XX.XXXX s"
  - "Iterations: YY"
  - "Status: ZZZZ"
  - "Final cost: W.WWWW"
- Return optimal states, controls, and final theta.

## File 5: visualizer.py

Matplotlib visualization tool.

- Function `def visualize( initial_guess, gates, opt_states, opt_controls, reference ):`
  - 3D subplot: plot gates as wireframe squares (1.5m inner), initial guess path (dashed), optimized trajectory (solid thick line).
  - Velocity vectors as quivers along path.
  - Orientation arrows (body X/Y/Z) at several points.
  - Separate 2D subplots: speed over time, body rates, acceleration components, four motor thrusts u1-u4 over horizon.
  - Title with solve stats.
  - Show plot (plt.show()).

## File 6: planner.py

Main planner class (minimal).

- Class `class MPCCPlanner:`
  - `__init__`: load dynamics, setup mpcc_problem once.
  - `def plan(self, current_state, gates):`
    - Generate reference from gates.
    - Solve MPCC.
    - Extract first-step optimal quaternion q_des and collective thrust_cmd = clamp(sum(Ti)/ (4\*Tmax), 0, 1).
    - Call visualizer if enabled.
    - Return q_des, thrust_cmd, and debug info dict.
- Keep class as small as possible.

## File 7: demo.py

Simple runnable demo script.

- Hardcode sample current_state (13-vector, near hover).
- Hardcode 2-3 sample gates with positions and quaternions (aligned along a straight-ish course).
- Instantiate planner.
- Call plan() once.
- Print returned q_des and thrust_cmd.
- Call visualizer.

## Final Instructions to Codex

- Implement files in the exact order above.
- Ensure the entire project runs with `python demo.py` and produces the required visualization + terminal debug output.
- Code must be human-readable, compartmentalized, and contain ONLY the necessary functionality — no extras.
- After all files are created, output a short confirmation message listing the files and a one-line example command to run the demo.

Follow these instructions precisely and output only the Python code for each file when requested.
