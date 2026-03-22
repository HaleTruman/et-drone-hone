# Quadrotor 13-State Nonlinear Dynamic Model Derivation

**For Path Planning, Control, and Simulation**

**Convention:** All quantities in SI units (m, kg, s, rad, N, Nm).  
**Purpose:** Complete, first-principles derivation of a 13-state rigid-body quadrotor model. No small-angle or hover approximations. General geometry (individual arm lengths $l_i$ and Z-offsets $h_i$). Includes rotor inertia gyroscopic effects, quadratic drag (translational + rotational). Algebraic motor model (no rotor-speed states).

**Key Design Choices and Rationale:**

- **13 states** — Position (3), inertial velocity (3), unit quaternion (4), body angular rates (3). This is the standard minimal representation for a rigid-body vehicle with orientation described by quaternions (avoids singularity-prone Euler angles). Rotor speeds are **not** states because the control API provides normalized inputs (0–1) without RPM feedback — we use an algebraic propeller model instead.
- **No small-angle or hover approximations** — The model is valid for large pitch/roll angles (e.g., flips, dives, racing maneuvers) and arbitrary speeds.
- **Quaternion for orientation** — Provides singularity-free, computationally efficient rotation representation.
- **Body-frame forces/torques** — Rotational dynamics are naturally expressed in body frame (Euler's equations); translational dynamics use the rotation matrix to pull forces into inertial frame.
- **General geometry** — Individual arm lengths $l_i$ and Z-offsets $h_i$ per motor allow non-symmetric or tilted configurations.
- **Included effects** — Thrust, motor reaction torques, rotor gyroscopic moments (important for high angular rates), quadratic drag on body and rotation (realistic at high speeds), gravity.
- **Rigid body and rigid rotors** — No flexible modes or blade flapping.

All derivations start from Newton's laws and rigid-body rotational equations.

---

## Nomenclature

| Symbol                                                       | Description                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------------------ |
| $c$                                                          | 45° projection factor $=\sqrt{2}/2$                                      |
| $c_{dx}, c_{dy}, c_{dz}$                                     | Quadratic translational drag coefficients (N·s²/m²)                      |
| $c_{rx}, c_{ry}, c_{rz}$                                     | Quadratic rotational drag coefficients (N·m·s²)                          |
| $d_i \in \{+1, -1\}$                                         | Spin direction of rotor $i$ (+1 = CW viewed from above)                  |
| $\mathbf{F}^B$                                               | Total force vector in body frame (N)                                     |
| $\mathbf{F}_{\text{drag}}^B$                                 | Aerodynamic drag force in body frame (N)                                 |
| $\mathbf{F}_{\text{thrust}}^B$                               | Thrust force contribution in body frame (N)                              |
| $g$                                                          | Gravitational acceleration (9.81 m/s²)                                   |
| $h_i$                                                        | Vertical (z) offset of propeller plane from CG for motor $i$ (m)         |
| $\mathbf{I}_B = \operatorname{diag}(I_{xx}, I_{yy}, I_{zz})$ | Inertia tensor about CG in body frame (kg·m²)                            |
| $J_r$                                                        | Polar moment of inertia of one rotor + propeller about spin axis (kg·m²) |
| $k_f$                                                        | Thrust coefficient (N)                                                   |
| $k_m$                                                        | Rotor torque (reaction) coefficient (N·m)                                |
| $l_i$                                                        | Horizontal distance from CG to motor $i$ along arm (m)                   |
| $\mathbf{M}^B$                                               | Total moment (torque) vector in body frame (N·m)                         |
| $\mathbf{M}_{\text{drag}}^B$                                 | Aerodynamic drag moment in body frame (N·m)                              |
| $\mathbf{M}_{\text{gyro}}^B$                                 | Gyroscopic moment from rotor inertia (N·m)                               |
| $\mathbf{M}_{\text{motor}}^B$                                | Motor reaction torque contribution (N·m)                                 |
| $\mathbf{M}_{\text{thrust}}^B$                               | Thrust moment contribution (N·m)                                         |
| $m$                                                          | Total vehicle mass (kg)                                                  |
| $\boldsymbol{\omega}_B = [p, q, r]^T$                        | Body angular velocity vector (rad/s)                                     |
| $\boldsymbol{\rho}_i$                                        | Position vector from CG to motor/propeller $i$ in body frame (m)         |
| $R(\mathbf{q})$                                              | Direction cosine matrix (body → inertial)                                |
| $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$                        | Unit quaternion (body → inertial rotation, $\|\mathbf{q}\|=1$)           |
| $T_i$                                                        | Thrust produced by motor/propeller $i$ (N)                               |
| $\mathbf{u} = [u_1, u_2, u_3, u_4]^T$                        | Normalized motor commands (0 to 1)                                       |
| $\mathbf{v}_B$                                               | Velocity of CG expressed in body frame (m/s)                             |
| $\mathbf{v}_I = [v_{x_I}, v_{y_I}, v_{z_I}]^T$               | Velocity of CG in inertial frame (m/s)                                   |
| $\Xi(\mathbf{q})$                                            | Quaternion rate transformation matrix (4×3)                              |
| $\mathbf{x}$                                                 | Full state vector (13 states)                                            |
| $\mathbf{p}_I = [x, y, z]^T$                                 | Position of CG in inertial frame (m)                                     |
| $\Omega_i$                                                   | Signed rotational speed of rotor $i$ (rad/s)                             |

---

## 1. Coordinate Frames and Conventions

### Inertial Frame (North-East-Down, NED)

- Fixed to Earth (non-rotating for short flights).
- $X_I$: North (positive forward).
- $Y_I$: East (positive right).
- $Z_I$: Down (positive downward, consistent with gravity direction).
- Gravity acceleration vector in inertial frame:
  $$
  \mathbf{g}_I = \begin{bmatrix} 0 \\ 0 \\ g \end{bmatrix}, \quad g = 9.81\,\text{m/s}^2
  $$
  **Reasoning:** Gravity acts downward in the inertial frame regardless of vehicle orientation. The positive-down convention simplifies signs for altitude control and matches many drone autopilots.

### Body Frame (Standard Drone Racing Convention)

- Origin: Center of gravity (CG).
- $X_B$: Forward (bisects the front two motors/arms).
- $Y_B$: Right.
- $Z_B$: Down (positive thrust force points **upward in world**, so thrust vector is negative in $Z_B$).
- **Why this convention?** Matches Betaflight, iNav, PX4 racing modes, and most FPV/racing firmware. Thrust along $-Z_B$ is intuitive for control allocation (positive throttle = positive $u_i$ → negative $Z_B$ force).

**Motor Layout (X-configuration):**

- Motor 1: front-right, spins clockwise (CW, viewed from above) → direction sign $d_1 = +1$.
- Motor 2: rear-right, counter-clockwise (CCW) → $d_2 = -1$.
- Motor 3: rear-left, CW → $d_3 = +1$.
- Motor 4: front-left, CCW → $d_4 = -1$.

**Reasoning for signs:** CW rotors produce positive torque about $Z_B$ (right-hand rule). The reaction torque on the body is opposite. The signs are chosen so that equal $u_i$ produce zero net yaw torque (torque cancellation).

**Motor positions in body frame (general, non-square):**
Let $c = \frac{\sqrt{2}}{2}$ (45° projection factor).

$$
\boldsymbol{\rho}_1 = \begin{bmatrix} l_1 c \\ l_1 c \\ h_1 \end{bmatrix}, \quad
\boldsymbol{\rho}_2 = \begin{bmatrix} -l_2 c \\ l_2 c \\ h_2 \end{bmatrix}, \quad
\boldsymbol{\rho}_3 = \begin{bmatrix} -l_3 c \\ -l_3 c \\ h_3 \end{bmatrix}, \quad
\boldsymbol{\rho}_4 = \begin{bmatrix} l_4 c \\ -l_4 c \\ h_4 \end{bmatrix}
$$

- $l_i$: Horizontal distance from CG to motor along the arm.
- $h_i$: Vertical offset of propeller plane from CG (positive if props above CG).
- **Why general $l_i, h_i$?** Allows modeling of asymmetric frames, tilted motors, or CG shifts (e.g., camera/gimbal mounting).

---

## 2. State Vector Definition (Exactly 13 States)

$$
\mathbf{x} = \begin{bmatrix}
x & y & z &
v_{x_I} & v_{y_I} & v_{z_I} &
q_w & q_x & q_y & q_z &
p & q & r
\end{bmatrix}^T
$$

- $(x,y,z)$: Inertial position of CG.
- $(v_{x_I}, v_{y_I}, v_{z_I})$: Velocity of CG in inertial frame.
- $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$: Unit quaternion representing rotation from **body** to **inertial** frame ($\mathbf{q} \otimes \mathbf{q}^* = 1$).
- $(p, q, r)$: Body angular rates about $X_B, Y_B, Z_B$.

**Reasoning:**

- Position and velocity are in inertial frame for easy integration of trajectory references.
- Quaternion avoids gimbal lock and is numerically stable for large attitudes.
- Angular rates are in body frame because gyroscopes measure body rates directly.
- 13 states is minimal for full 6-DoF rigid body + orientation.

---

## 3. Input Vector and Configurable Parameters

**Control inputs** (0–1 normalized):

$$
\mathbf{u} = \begin{bmatrix} u_1 & u_2 & u_3 & u_4 \end{bmatrix}^T
$$

**Config parameters (for your file):**  
$m$, $I_{xx}, I_{yy}, I_{zz}$, $k_f$, $k_m$, $J_r$, $l_i$, $h_i$, $c_{dx},c_{dy},c_{dz}$, $c_{rx},c_{ry},c_{rz}$, $g=9.81$.

---

## 4. Algebraic Motor/Propeller Model

Since no RPM feedback is available, we use a **static, algebraic model** based on empirical propeller data:

**Thrust produced by motor $i$** (force in body frame, upward = $-Z_B$):

$$
T_i = k_f \, u_i^2
$$

**Reasoning:** Thrust is approximately quadratic in rotor speed ($\Omega_i^2$), and $\Omega_i \propto u_i$ in steady state for brushed/brushless motors. This is the standard model in most quadrotor literature and gives good accuracy without needing dynamic motor equations.

**Signed rotor spin speed** (rad/s, needed only for gyroscopic effect):

$$
\Omega_i = d_i \sqrt{\frac{T_i}{k_f}} = d_i \sqrt{k_f} \, u_i
$$

**Reasoning:** $\Omega_i^2 = T_i / k_f$ from blade element theory; square root gives magnitude, $d_i$ gives direction.

**Reaction torque on body** (from motor, about $Z_B$):

$$
\tau_{\text{motor},i} = -d_i \, k_m \, u_i^2
$$

**Reasoning:** Motor torque $\tau_i = k_m \Omega_i^2$ opposes spin; reaction on body is opposite. The sign with $d_i$ ensures yaw control works correctly (increasing $u_1$ and $u_3$ while decreasing $u_2$ and $u_4$ produces positive yaw).

---

## 5. Forces and Moments – First-Principles Breakdown

### 5.1 Total Thrust Force

Each propeller produces force purely along its axis (ideal assumption, neglects hub forces):

$$
\mathbf{f}_{\text{thrust},i}^B = \begin{bmatrix} 0 \\ 0 \\ -T_i \end{bmatrix}
$$

Total thrust force:

$$
\mathbf{F}_{\text{thrust}}^B = \sum_{i=1}^4 \mathbf{f}_{\text{thrust},i}^B = \begin{bmatrix} 0 \\ 0 \\ -\sum_{i=1}^4 T_i \end{bmatrix}
$$

**Reasoning:** Thrust is axial; no lateral components in ideal model.

### 5.2 Thrust-Generated Moments

Torque from each thrust:

$$
\boldsymbol{\tau}_{\text{thrust},i}^B = \boldsymbol{\rho}_i \times \mathbf{f}_{\text{thrust},i}^B = \boldsymbol{\rho}_i \times \begin{bmatrix} 0 \\ 0 \\ -T_i \end{bmatrix}
$$

Cross product (step-by-step):

$$
\begin{bmatrix} \rho_{x i} \\ \rho_{y i} \\ \rho_{z i} \end{bmatrix} \times \begin{bmatrix} 0 \\ 0 \\ -T_i \end{bmatrix} = \begin{bmatrix}
\rho_{y i} (-T_i) - \rho_{z i} (0) \\
\rho_{z i} (0) - \rho_{x i} (-T_i) \\
\rho_{x i} (0) - \rho_{y i} (0)
\end{bmatrix} = \begin{bmatrix}
-\rho_{y i} T_i \\
\rho_{x i} T_i \\
0
\end{bmatrix}
$$

Wait — sign correction for consistency with $-T_i$:  
Actually:

$$
\boldsymbol{\rho}_i \times \begin{bmatrix} 0 \\ 0 \\ -T_i \end{bmatrix} = \begin{bmatrix} \rho_{y i} (-T_i) \\ - \rho_{x i} (-T_i) \\ 0 \end{bmatrix} = \begin{bmatrix} -\rho_{y i} T_i \\ \rho_{x i} T_i \\ 0 \end{bmatrix}
$$

Total thrust moment:

$$
\mathbf{M}_{\text{thrust}}^B = \sum_i \begin{bmatrix} -\rho_{y i} T_i \\ \rho_{x i} T_i \\ 0 \end{bmatrix} = \begin{bmatrix}
-\sum \rho_{y i} T_i \\
\sum \rho_{x i} T_i \\
0
\end{bmatrix}
$$

**Reasoning:** Roll/pitch torques come from differential thrust; no yaw torque from thrust (symmetric).

### 5.3 Motor Reaction Torques

$$
\mathbf{M}_{\text{motor}}^B = \begin{bmatrix} 0 \\ 0 \\ \sum_{i=1}^4 \tau_{\text{motor},i} \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ -k_m \sum_{i=1}^4 d_i u_i^2 \end{bmatrix}
$$

**Reasoning:** Only yaw component; cancellation when $u_i$ equal.

### 5.4 Gyroscopic Moments from Rotor Inertia

Each rotor has angular momentum along its spin axis:

$$
\mathbf{h}_i = J_r \Omega_i \mathbf{e}_z^B = J_r \Omega_i \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

Total rotor angular momentum in body frame:

$$
\mathbf{h}_{\text{rot}}^B = J_r \sum_{i=1}^4 \Omega_i \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \\ J_r \sum \Omega_i \end{bmatrix}
$$

In rotating body frame, the rate of change of angular momentum contributes torque on the **body** (opposite sign):

$$
\mathbf{M}_{\text{gyro}}^B = - \boldsymbol{\omega}_B \times \mathbf{h}_{\text{rot}}^B
$$

Expand cross product:

$$
\boldsymbol{\omega}_B = \begin{bmatrix} p \\ q \\ r \end{bmatrix}, \quad \mathbf{h}_{\text{rot}}^B = \begin{bmatrix} 0 \\ 0 \\ h_z \end{bmatrix}, \quad h_z = J_r \sum \Omega_i
$$

$$
- \begin{bmatrix} p \\ q \\ r \end{bmatrix} \times \begin{bmatrix} 0 \\ 0 \\ h_z \end{bmatrix} = - \begin{bmatrix}
q \cdot h_z - r \cdot 0 \\
r \cdot 0 - p \cdot h_z \\
p \cdot 0 - q \cdot 0
\end{bmatrix} = - \begin{bmatrix} q h_z \\ -p h_z \\ 0 \end{bmatrix}
$$

So:

$$
\mathbf{M}_{\text{gyro}}^B = - J_r \left( \sum \Omega_i \right) \begin{bmatrix} q \\ -p \\ 0 \end{bmatrix}
$$

Substitute $\Omega_i$:

$$
\sum \Omega_i = \sqrt{k_f} \sum d_i u_i \quad \Rightarrow \quad \mathbf{M}_{\text{gyro}}^B = -J_r \sqrt{k_f} \left( \sum d_i u_i \right) \begin{bmatrix} q \\ -p \\ 0 \end{bmatrix}
$$

**Reasoning (why important?):** This term couples roll/pitch rates with yaw input and vice versa. Critical for aggressive maneuvers; neglected in many simplified models but physically present.

### 5.5 Aerodynamic Drag

**Translational drag** (quadratic, dominant at high speed):  
First compute body-frame velocity:

$$
\mathbf{v}_B = R(\mathbf{q})^T \mathbf{v}_I
$$

(velocity expressed in body axes, needed for drag direction).

Drag force (opposes velocity):

$$
\mathbf{F}_{\text{drag}}^B = - \begin{bmatrix}
c_{dx} v_{xB} |v_{xB}| \\
c_{dy} v_{yB} |v_{yB}| \\
c_{dz} v_{zB} |v_{zB}|
\end{bmatrix}
$$

**Reasoning:** Quadratic drag $\propto v |v|$ from drag equation $F_d = \frac{1}{2} \rho C_d A v^2 \operatorname{sign}(v)$. Coefficients $c_d$ absorb $\frac{1}{2} \rho C_d A$.

**Rotational drag** (prop/friction torque):

$$
\mathbf{M}_{\text{drag}}^B = - \begin{bmatrix}
c_{rx} p |p| \\
c_{ry} q |q| \\
c_{rz} r |r|
\end{bmatrix}
$$

**Reasoning:** Similar quadratic form for rotational viscous effects.

### 5.6 Total Force and Moment

$$
\mathbf{F}^B = \mathbf{F}_{\text{thrust}}^B + \mathbf{F}_{\text{drag}}^B
$$

$$
\mathbf{M}^B = \mathbf{M}_{\text{thrust}}^B + \mathbf{M}_{\text{motor}}^B + \mathbf{M}_{\text{gyro}}^B + \mathbf{M}_{\text{drag}}^B
$$

---

## 6. Kinematics – From First Principles

### 6.1 Position Kinematics

By definition (velocity is time-derivative of position):

$$
\dot{\mathbf{p}}_I = \mathbf{v}_I \quad \Rightarrow \quad
\begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{z} \end{bmatrix} = \begin{bmatrix} v_{x_I} \\ v_{y_I} \\ v_{z_I} \end{bmatrix}
$$

### 6.2 Quaternion Kinematics

The quaternion $\mathbf{q}$ rotates vectors from body to inertial: $\mathbf{v}_I = \mathbf{q} \otimes \mathbf{v}_B \otimes \mathbf{q}^*$.

The time derivative follows from the angular velocity definition:  
The instantaneous angular velocity $\boldsymbol{\omega}_B$ causes change in orientation. In quaternion terms:

$$
\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\omega}_B \end{bmatrix}
$$

This is equivalent to matrix multiplication (more efficient):

$$
\dot{\mathbf{q}} = \frac{1}{2} \Xi(\mathbf{q}) \boldsymbol{\omega}_B
$$

where

$$
\Xi(\mathbf{q}) = \begin{bmatrix}
-q_x & -q_y & -q_z \\
q_w & -q_z & q_y \\
q_z & q_w & -q_x \\
-q_y & q_x & q_w
\end{bmatrix}
$$

**Reasoning:** This comes from the quaternion exponential map and angular velocity. The factor 1/2 arises from the double-cover property of quaternions (full 360° rotation corresponds to $\mathbf{q} \to -\mathbf{q}$).

**Important:** After numerical integration, always renormalize:

$$
\mathbf{q} \leftarrow \frac{\mathbf{q}}{\|\mathbf{q}\|}
$$

to prevent drift.

### 6.3 Rotation Matrix $R(\mathbf{q})$ (Body → Inertial)

Derived from quaternion conjugation:

$$
R(\mathbf{q}) = \begin{bmatrix}
1-2(q_y^2+q_z^2) & 2(q_x q_y - q_w q_z) & 2(q_x q_z + q_w q_y) \\
2(q_x q_y + q_w q_z) & 1-2(q_x^2+q_z^2) & 2(q_y q_z - q_w q_x) \\
2(q_x q_z - q_w q_y) & 2(q_y q_z + q_w q_x) & 1-2(q_x^2+q_y^2)
\end{bmatrix}
$$

**Reasoning:** This is the standard formula ensuring $R^T R = I$ and $\det R = 1$ when $\|\mathbf{q}\|=1$.

---

## 7. Translational Dynamics – Newton's Second Law

The net force on the vehicle equals mass times acceleration of CG (in inertial frame):

$$
m \dot{\mathbf{v}}_I = \underbrace{R(\mathbf{q}) \mathbf{F}^B}_{\text{body forces rotated to inertial}} + m \mathbf{g}_I
$$

Solving for acceleration:

$$
\dot{\mathbf{v}}_I = \frac{1}{m} R(\mathbf{q}) \mathbf{F}^B + \mathbf{g}_I = \frac{1}{m} R(\mathbf{q}) \mathbf{F}^B + \begin{bmatrix} 0 \\ 0 \\ g \end{bmatrix}
$$

**Reasoning:** All forces are computed in body frame (thrust/drag natural there), then rotated to inertial for integration. Gravity is inertial.

---

## 8. Rotational Dynamics – Euler's Equations

For a rigid body with principal axes, the rotational equation in body frame is:

$$
\mathbf{I}_B \dot{\boldsymbol{\omega}}_B + \boldsymbol{\omega}_B \times (\mathbf{I}_B \boldsymbol{\omega}_B) = \mathbf{M}^B
$$

**Reasoning (derivation sketch):**  
Angular momentum $\mathbf{L}_B = \mathbf{I}_B \boldsymbol{\omega}_B$.  
In body frame, $\frac{d\mathbf{L}}{dt}\big|_I = \frac{d\mathbf{L}}{dt}\big|_B + \boldsymbol{\omega} \times \mathbf{L}$.  
By Newton's law, $\frac{d\mathbf{L}}{dt}\big|_I = \mathbf{M}^B$.  
Thus: $\mathbf{I}_B \dot{\boldsymbol{\omega}}_B + \boldsymbol{\omega}_B \times \mathbf{I}_B \boldsymbol{\omega}_B = \mathbf{M}^B$.

Solving for angular acceleration:

$$
\dot{\boldsymbol{\omega}}_B = \mathbf{I}_B^{-1} \left( \mathbf{M}^B - \boldsymbol{\omega}_B \times \mathbf{I}_B \boldsymbol{\omega}_B \right)
$$

Since $\mathbf{I}_B = \operatorname{diag}(I_{xx}, I_{yy}, I_{zz})$, the cross product expands easily in code.

---

## 9. Full State-Space Equations of Motion

The complete dynamics are $\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u})$:

$$
\begin{aligned}
\dot{x} &= v_{x_I} \\
\dot{y} &= v_{y_I} \\
\dot{z} &= v_{z_I} \\[8pt]
\begin{bmatrix} \dot{v}_{x_I} \\ \dot{v}_{y_I} \\ \dot{v}_{z_I} \end{bmatrix}
&= \frac{1}{m} R(\mathbf{q}) \mathbf{F}^B + \begin{bmatrix} 0 \\ 0 \\ g \end{bmatrix} \\[12pt]
\dot{\mathbf{q}} &= \frac{1}{2} \Xi(\mathbf{q}) \begin{bmatrix} p \\ q \\ r \end{bmatrix} \\[12pt]
\begin{bmatrix} \dot{p} \\ \dot{q} \\ \dot{r} \end{bmatrix}
&= \mathbf{I}_B^{-1} \left( \mathbf{M}^B - \begin{bmatrix} p \\ q \\ r \end{bmatrix} \times \mathbf{I}_B \begin{bmatrix} p \\ q \\ r \end{bmatrix} \right)
\end{aligned}
$$

**Implementation notes:**

- Compute $\mathbf{v}_B = R^T \mathbf{v}_I$ for drag.
- All terms in $\mathbf{F}^B, \mathbf{M}^B$ are functions of $\mathbf{x}, \mathbf{u}$ as defined.
- Use RK4 or similar for integration; renormalize quaternion each step.
