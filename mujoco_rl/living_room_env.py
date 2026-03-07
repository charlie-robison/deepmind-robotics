"""Living room navigation environment using MuJoCo."""

import json
import pathlib
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

GRAVITY_COMP = 1.0 * 9.81  # 9.81 N for 1.0 kg agent


def load_trajectories(path: str | pathlib.Path) -> list[np.ndarray]:
    """Load trajectories from JSON or NPZ file."""
    path = pathlib.Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return [data[k] for k in sorted(data.files)]
    elif path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        trajs = []
        for item in raw:
            if isinstance(item, dict):
                trajs.append(np.array(item["waypoints"], dtype=np.float64))
            else:
                trajs.append(np.array(item, dtype=np.float64))
        return trajs
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


_DEFAULT_OBSTACLES = [
    # Room walls (8m x 8m room)
    {"name": "wall_north", "pos": [0, 4, 1.5],  "size": [4, 0.1, 1.5]},
    {"name": "wall_south", "pos": [0, -4, 1.5], "size": [4, 0.1, 1.5]},
    {"name": "wall_east",  "pos": [4, 0, 1.5],  "size": [0.1, 4, 1.5]},
    {"name": "wall_west",  "pos": [-4, 0, 1.5], "size": [0.1, 4, 1.5]},
    # Furniture
    {"name": "couch",        "pos": [-2, 2, 0.4],  "size": [1.0, 0.4, 0.4]},
    {"name": "coffee_table", "pos": [0, 0, 0.25],  "size": [0.6, 0.4, 0.25]},
    {"name": "side_table",   "pos": [-3, 1, 0.3],  "size": [0.3, 0.3, 0.3]},
    {"name": "bookshelf",    "pos": [3, 2, 0.8],   "size": [0.3, 0.5, 0.8]},
]


def _get_asset_dir() -> str:
    """Return absolute path to the assets directory."""
    return str(pathlib.Path(__file__).parent / "assets")


def _build_xml(
    box_obstacles: list[dict] | None = None,
    use_visual_mesh: bool = True,
) -> str:
    obstacles = box_obstacles if box_obstacles is not None else _DEFAULT_OBSTACLES
    asset_dir = _get_asset_dir()
    mesh_path = pathlib.Path(asset_dir) / "living_room.obj"
    has_mesh = use_visual_mesh and mesh_path.exists()

    obs_xml = ""
    for obs in obstacles:
        name = obs["name"]
        pos = " ".join(str(v) for v in obs["pos"])
        size = " ".join(str(v) for v in obs["size"])
        # If we have the visual mesh, make collision boxes invisible
        if has_mesh:
            obs_xml += f'    <geom name="{name}" type="box" pos="{pos}" size="{size}" rgba="0 0 0 0" contype="1" conaffinity="1"/>\n'
        else:
            rgba = " ".join(str(v) for v in obs.get("rgba", [0.6, 0.4, 0.2, 1]))
            obs_xml += f'    <geom name="{name}" type="box" pos="{pos}" size="{size}" rgba="{rgba}" contype="1" conaffinity="1"/>\n'

    asset_xml = ""
    visual_mesh_xml = ""
    if has_mesh:
        asset_xml = f"""
  <compiler meshdir="{asset_dir}"/>
  <asset>
    <mesh name="living_room_mesh" file="living_room.obj" scale="1 1 1"/>
  </asset>"""
        # Visual-only mesh geom (no collision)
        visual_mesh_xml = '    <geom name="room_visual" type="mesh" mesh="living_room_mesh" pos="0 0 0" rgba="0.85 0.75 0.65 1" contype="0" conaffinity="0"/>\n'

    xml = f"""<mujoco model="living_room">
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
{asset_xml}
  <default>
    <geom contype="1" conaffinity="1"/>
  </default>

  <visual>
    <global azimuth="-120" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>

  <worldbody>
    <light pos="0 0 4" dir="0 0.2 -1" diffuse="0.9 0.85 0.75" specular="0.3 0.3 0.3"/>
    <light pos="-2 2 3" dir="0.3 -0.3 -1" diffuse="0.5 0.45 0.35"/>
    <camera name="track" mode="targetbody" target="agent" pos="1.5 -1.5 1.8"/>
    <geom name="ground" type="plane" size="10 10 0.1" rgba="0.65 0.45 0.25 1" contype="1" conaffinity="1"/>

{visual_mesh_xml}{obs_xml}
    <body name="agent" pos="0 0 0.15">
      <freejoint name="agent_joint"/>
      <geom name="agent_body" type="box" size="0.1 0.1 0.08" mass="1.0" rgba="0.2 0.6 0.2 1" contype="1" conaffinity="1"/>
      <geom name="direction" type="cylinder" size="0.03 0.06" pos="0.1 0 0" rgba="1 0 0 0.5" contype="0" conaffinity="0" mass="0"/>
      <site name="thrust_site" pos="0 0 0"/>
    </body>
  </worldbody>

  <sensor>
    <framepos objtype="body" objname="agent" name="agent_pos"/>
    <framequat objtype="body" objname="agent" name="agent_quat"/>
    <framelinvel objtype="body" objname="agent" name="agent_linvel"/>
    <frameangvel objtype="body" objname="agent" name="agent_angvel"/>
  </sensor>
</mujoco>"""
    return xml


class LivingRoomNavEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        trajectories: list[np.ndarray],
        box_obstacles: list[dict] | None = None,
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.trajectories = trajectories
        self.box_obstacles = box_obstacles
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        xml = _build_xml(box_obstacles)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.agent_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent")
        self.agent_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "agent_body")

        # Obstacle geom IDs (for collision detection)
        obstacle_list = box_obstacles if box_obstacles is not None else _DEFAULT_OBSTACLES
        self._obstacle_geom_ids = set()
        for obs in obstacle_list:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obs["name"])
            if gid >= 0:
                self._obstacle_geom_ids.add(gid)

        # 28-d observation: pos(3) + quat(4) + linvel(3) + angvel(3) + relative_wps(5*3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(28,), dtype=np.float32)
        # 4-d action: thrust_z, lateral_x, lateral_y, yaw_torque
        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        self._traj: np.ndarray = trajectories[0]
        self._wp: int = 0
        self._step_count: int = 0
        self._prev_dist: float = 0.0

        self._viewer = None

    def _get_obs(self) -> np.ndarray:
        pos = self.data.sensordata[0:3].copy()
        quat = self.data.sensordata[3:7].copy()
        linvel = self.data.sensordata[7:10].copy()
        angvel = self.data.sensordata[10:13].copy()

        # Relative waypoints (next 5)
        rel_wps = np.zeros(15, dtype=np.float64)
        for i in range(5):
            wp_idx = self._wp + i
            if wp_idx < len(self._traj):
                rel_wps[i * 3 : i * 3 + 3] = self._traj[wp_idx] - pos

        obs = np.concatenate([pos, quat, linvel, angvel, rel_wps])
        return obs.astype(np.float32)

    def _check_collision(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            if g1 == self.agent_geom_id or g2 == self.agent_geom_id:
                other = g2 if g1 == self.agent_geom_id else g1
                if other in self._obstacle_geom_ids:
                    return True
        return False

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)

        # Map actions to forces
        thrust_z = (action[0] + 1.0) / 2.0 * 15.0  # [0, 15] N
        lateral_x = action[1] * 3.0  # [-3, 3] N
        lateral_y = action[2] * 3.0  # [-3, 3] N
        yaw_torque = action[3] * 0.5  # [-0.5, 0.5] Nm

        # Body-frame force
        body_force = np.array([lateral_x, lateral_y, thrust_z])

        # Get agent quaternion (MuJoCo: w,x,y,z)
        quat = self.data.sensordata[3:7].copy()

        # Rotate body-frame force to world frame
        world_force = np.zeros(3)
        mujoco.mju_rotVecQuat(world_force, body_force, quat)

        # Yaw torque in body Z, rotated to world
        body_torque = np.array([0.0, 0.0, yaw_torque])
        world_yaw_torque = np.zeros(3)
        mujoco.mju_rotVecQuat(world_yaw_torque, body_torque, quat)

        # 4 substeps
        for _ in range(4):
            self.data.xfrc_applied[self.agent_body_id, :3] = world_force
            self.data.xfrc_applied[self.agent_body_id, 3:] = world_yaw_torque
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        # Compute reward
        pos = self.data.sensordata[0:3].copy()
        target = self._traj[min(self._wp, len(self._traj) - 1)]
        dist = np.linalg.norm(pos - target)

        # Shaped reward
        reward = (self._prev_dist - dist) * 5.0
        reward += 0.01  # alive bonus

        terminated = False
        info: dict[str, Any] = {
            "height": pos[2],
            "wp_idx": self._wp,
            "wp_total": len(self._traj),
        }

        # Waypoint reached
        if self._wp < len(self._traj) and dist < 0.3:
            reward += 5.0
            self._wp += 1
            info["wp_idx"] = self._wp
            if self._wp >= len(self._traj):
                reward += 10.0
                terminated = True
                info["trajectory_complete"] = True

        # Collision check
        if self._check_collision():
            terminated = True
            info["collision"] = True

        truncated = self._step_count >= self.max_episode_steps

        # Update prev_dist for next step
        if self._wp < len(self._traj):
            self._prev_dist = np.linalg.norm(pos - self._traj[self._wp])
        else:
            self._prev_dist = 0.0

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        # Pick random trajectory
        traj_idx = self.np_random.integers(0, len(self.trajectories))
        self._traj = self.trajectories[traj_idx]
        self._wp = 0
        self._step_count = 0

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)

        # Spawn near first waypoint with small noise
        start_pos = self._traj[0].copy() + self.np_random.uniform(-0.1, 0.1, size=3)
        start_pos[2] = max(start_pos[2], 0.15)  # keep above ground

        # Set position (qpos: [x,y,z, w,x,y,z] for freejoint)
        self.data.qpos[0:3] = start_pos
        self.data.qpos[3:7] = [1, 0, 0, 0]  # identity quaternion
        self.data.qvel[:] = 0

        mujoco.mj_forward(self.model, self.data)

        pos = self.data.sensordata[0:3].copy()
        self._prev_dist = np.linalg.norm(pos - self._traj[0])

        info = {
            "height": pos[2],
            "wp_idx": self._wp,
            "wp_total": len(self._traj),
        }
        return self._get_obs(), info

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
