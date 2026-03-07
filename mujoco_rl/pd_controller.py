"""PD controller for living room navigation trajectory following."""

import mujoco
import numpy as np


class PDController:
    def __init__(self, kp: float = 4.0, kd: float = 3.0):
        self.kp = kp
        self.kd = kd

    def __call__(self, obs: np.ndarray, target: np.ndarray) -> np.ndarray:
        pos = obs[0:3]
        quat = obs[3:7]
        linvel = obs[7:10]

        # World-frame PD force
        error = target - pos
        error_norm = np.linalg.norm(error)
        if error_norm > 1.0:
            error = error / error_norm * 1.0
        world_force = self.kp * error - self.kd * linvel
        # Add gravity compensation (1.0 kg agent)
        world_force[2] += 9.81  # GRAVITY_COMP

        # Rotate world force to body frame using inverse quaternion
        quat_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
        body_force = np.zeros(3)
        mujoco.mju_rotVecQuat(body_force, world_force, quat_inv)

        # Map to action space (inverse of env scaling)
        thrust_action = body_force[2] / 15.0 * 2.0 - 1.0
        lateral_x_action = body_force[0] / 3.0
        lateral_y_action = body_force[1] / 3.0
        yaw_action = 0.0

        action = np.array([thrust_action, lateral_x_action, lateral_y_action, yaw_action], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)
