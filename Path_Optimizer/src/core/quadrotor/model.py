import numpy as np


class Quadrotor:
    """Minimal 13-state nonlinear quadrotor model."""

    def __init__(self, params: dict):
        """Initialize model parameters from a dictionary and precompute motor locations."""
        self.m = np.float64(params["m"])
        self.I = np.diag(
            np.array([params["Ixx"], params["Iyy"], params["Izz"]], dtype=float)
        )
        self.kf = np.float64(params["kf"])
        self.km = np.float64(params["km"])
        self.Jr = np.float64(params["Jr"])
        self.l = np.asarray(params["l"], dtype=float)
        self.h = np.asarray(params["h"], dtype=float)
        self.cd = np.asarray(params["cd"], dtype=float)
        self.cr = np.asarray(params["cr"], dtype=float)
        self.d = np.asarray(params["d"], dtype=float)
        self.g = np.float64(params["g"])
        self.c = np.sqrt(2.0) / 2.0
        self.rho = np.array(
            [
                [self.l[0] * self.c, self.l[0] * self.c, self.h[0]],
                [-self.l[1] * self.c, self.l[1] * self.c, self.h[1]],
                [-self.l[2] * self.c, -self.l[2] * self.c, self.h[2]],
                [self.l[3] * self.c, -self.l[3] * self.c, self.h[3]],
            ],
            dtype=np.float64,
        )


    def state_derivative(
        self,
        x: np.ndarray,
        u: np.ndarray,
        external_force_i: np.ndarray | None = None,
        external_moment_b: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the 13-state derivative for state x and motor input u."""
        v_i = np.asarray(x[3:6], dtype=float)
        q = np.asarray(x[6:10], dtype=float)
        omega = np.asarray(x[10:13], dtype=float)

        force_b = self._compute_forces(x, u)
        moment_b = self._compute_moments(x, u)
        rotation = self._rotation_matrix(q)
        xi = self._xi_matrix(q)

        x_dot = np.zeros(13, dtype=float)
        x_dot[0:3] = v_i
        external_force_i = np.zeros(3) if external_force_i is None else np.asarray(external_force_i, dtype=float)
        external_moment_b = np.zeros(3) if external_moment_b is None else np.asarray(external_moment_b, dtype=float)
        x_dot[3:6] = (rotation @ force_b + external_force_i) / self.m + np.array([0.0, 0.0, self.g])
        x_dot[6:10] = 0.5 * (xi @ omega)
        x_dot[10:13] = np.linalg.solve(
            self.I, moment_b + external_moment_b - np.cross(omega, self.I @ omega)
        )
        return x_dot

    def _rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return the body-to-inertial rotation matrix for quaternion q."""
        qw, qx, qy, qz = np.asarray(q, dtype=float)
        return np.array(
            [
                [
                    1.0 - 2.0 * (qy * qy + qz * qz),
                    2.0 * (qx * qy - qw * qz),
                    2.0 * (qx * qz + qw * qy),
                ],
                [
                    2.0 * (qx * qy + qw * qz),
                    1.0 - 2.0 * (qx * qx + qz * qz),
                    2.0 * (qy * qz - qw * qx),
                ],
                [
                    2.0 * (qx * qz - qw * qy),
                    2.0 * (qy * qz + qw * qx),
                    1.0 - 2.0 * (qx * qx + qy * qy),
                ],
            ],
            dtype=float,
        )

    def _xi_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return the quaternion-rate matrix Xi(q) for quaternion q."""
        qw, qx, qy, qz = np.asarray(q, dtype=float)
        return np.array(
            [
                [-qx, -qy, -qz],
                [qw, -qz, qy],
                [qz, qw, -qx],
                [-qy, qx, qw],
            ],
            dtype=float,
        )

    def _get_motor_thrusts_omegas(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return motor thrusts and signed rotor speeds for normalized inputs u."""
        u = np.asarray(u, dtype=float)
        thrusts = self.kf * np.square(u)
        omegas = self.d * np.sqrt(self.kf) * u
        return thrusts, omegas

    def _body_frame_velocity(self, x: np.ndarray) -> np.ndarray:
        """Return inertial velocity from x expressed in the body frame."""
        v_i = np.asarray(x[3:6], dtype=float)
        q = np.asarray(x[6:10], dtype=float)
        return self._rotation_matrix(q).T @ v_i

    def _compute_forces(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Return total body-frame force from thrust and quadratic translational drag."""
        thrusts, _ = self._get_motor_thrusts_omegas(u)
        v_b = self._body_frame_velocity(x)
        thrust_force = np.array([0.0, 0.0, -np.sum(thrusts)], dtype=float)
        drag_force = -self.cd * v_b * np.abs(v_b)
        return thrust_force + drag_force

    def _compute_moments(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Return total body-frame moment from thrust, motor, gyro, and drag effects."""
        thrusts, omegas = self._get_motor_thrusts_omegas(u)
        omega_b = np.asarray(x[10:13], dtype=float)

        thrust_vectors = np.column_stack(
            (
                np.zeros(4, dtype=float),
                np.zeros(4, dtype=float),
                -thrusts,
            )
        )
        thrust_moment = np.sum(np.cross(self.rho, thrust_vectors), axis=0)
        motor_moment = np.array(
            [0.0, 0.0, -self.km * np.sum(self.d * np.square(np.asarray(u, dtype=float)))],
            dtype=float,
        )
        gyro_hz = self.Jr * np.sum(omegas)
        gyro_moment = -np.cross(omega_b, np.array([0.0, 0.0, gyro_hz], dtype=float))
        drag_moment = -self.cr * omega_b * np.abs(omega_b)
        return thrust_moment + motor_moment + gyro_moment + drag_moment

    def _normalize_quaternion(self, q: np.ndarray) -> np.ndarray:
        """Return quaternion q scaled to unit norm."""
        q = np.asarray(q, dtype=float)
        return q / np.linalg.norm(q)


if __name__ == "__main__":
    pass
