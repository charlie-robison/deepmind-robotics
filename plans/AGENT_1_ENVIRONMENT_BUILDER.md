# Agent 1: MuJoCo Environment Builder

## Overall Project Goal

We are building an RL training pipeline for a robot navigating a cozy living room. Trajectory data (x,y,z waypoints) is collected from a Three.js frontend. We recreate the environment in MuJoCo and train an RL agent to follow trajectories using Behavioral Cloning + PPO. A teammate already built an identical pipeline for a drone at `/Users/kaushikskaja/Documents/GitHub/mujoco_example` — we mirror that exact pattern for a living room navigation task.

The full pipeline: Trajectory JSON -> MuJoCo Env -> BC warm-start -> PPO fine-tuning -> Trained Policy

There are **two agents** working in parallel:
- **Agent 1 (YOU)**: MuJoCo environment + expert controller
- **Agent 2 (separate)**: Training pipeline, visualization, data conversion, project config

You must NOT create files that Agent 2 owns (train.py, visualize.py, convert_trajectories.py, pyproject.toml).

---

## Your Goal

Create the MuJoCo simulation of the living room and the PD expert controller. You produce the foundation that Agent 2's training code imports.

## Output Directory

`/Users/kaushikskaja/Documents/GitHub/deepmind-robotics/mujoco_rl/`

Create this directory if it doesn't exist.

---

## Files to Create

### 1. `mujoco_rl/__init__.py`

Empty file. Makes it a Python package.

### 2. `mujoco_rl/living_room_env.py`

**Mirror this file line-by-line**: `/Users/kaushikskaja/Documents/GitHub/mujoco_example/src/drone_env.py`

Read that file first, then adapt it with the changes below.

#### `load_trajectories(path: str | pathlib.Path) -> list[np.ndarray]`

Copy from drone_env.py verbatim. Supports JSON format: `[{"target": "...", "waypoints": [[x,y,z], ...]}, ...]` and NPZ format. Returns list of numpy arrays.

#### `_DEFAULT_OBSTACLES`

Living room furniture approximated as box obstacles:

```python
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
```

#### `_build_xml(box_obstacles: list[dict] | None = None) -> str`

Generates MJCF XML dynamically. Follow the drone_env.py `_build_xml` structure exactly, with these changes:

- Physics: `timestep="0.005"`, `gravity="0 0 -9.81"`, `integrator="RK4"` (same)
- Floor: `type="plane" size="10 10 0.1" rgba="0.65 0.45 0.25 1"` (wooden floor color)
- Agent body (replaces "drone"):
  - Name: `"agent"`
  - Position: `pos="0 0 0.15"`
  - Joint: `<freejoint name="agent_joint"/>`
  - Main geom: `type="box" size="0.1 0.1 0.08" mass="1.0" rgba="0.2 0.6 0.2 1"`
  - Direction indicator: `type="cylinder" size="0.03 0.01" pos="0.08 0 0.08" rgba="1 0 0 0.8" contype="0" conaffinity="0" mass="0"`
  - Thrust site: `<site name="thrust_site" pos="0 0 0"/>`
- Sensors (same 4 sensors but objname="agent"):
  - `<framepos objtype="body" objname="agent" name="agent_pos"/>`
  - `<framequat objtype="body" objname="agent" name="agent_quat"/>`
  - `<framelinvel objtype="body" objname="agent" name="agent_linvel"/>`
  - `<frameangvel objtype="body" objname="agent" name="agent_angvel"/>`
- Obstacle geoms: loop over obstacles list, support optional "rgba", "contype", "conaffinity" keys

#### `LivingRoomNavEnv(gym.Env)`

Follow `DroneTrajectoryEnv` structure exactly. Key specs:

**Constructor**: `__init__(self, trajectories, box_obstacles=None, max_episode_steps=1000, render_mode=None)`
- Build model from `_build_xml(box_obstacles or _DEFAULT_OBSTACLES)`
- `self.agent_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "agent")`
- `self.agent_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "agent_body")`
- Observation space: `Box(-inf, inf, shape=(28,), dtype=float32)`
- Action space: `Box(-1.0, 1.0, shape=(4,), dtype=float32)`

**`_get_obs()`**: 28-dim vector:
- pos (0:3), quat (3:7), linvel (7:10), angvel (10:13) from sensordata
- 5 relative waypoints (13:28): `waypoint[wp+i] - drone_pos` for i in 0-4

**`step(action)`**:
- Action scaling: `thrust_z = (action[0]+1)/2 * 15.0`, `lateral_x = action[1]*3.0`, `lateral_y = action[2]*3.0`, `yaw = action[3]*0.5`
- Body-frame force `[lateral_x, lateral_y, thrust_z]` rotated to world frame via `mujoco.mju_rotVecQuat`
- 4 physics substeps per step
- Reward: `(prev_dist - dist)*5.0 + 0.01 + (5.0 if waypoint reached) + (10.0 if trajectory complete)`
- Waypoint threshold: 0.3m
- Terminated on collision or trajectory complete
- Truncated at max_episode_steps

**`_check_collision()`**: Check `data.contact` for agent geom vs obstacle geoms.

**`reset(seed, options)`**:
- Random trajectory selection from self.trajectories
- Start position = first waypoint + noise U(-0.1, 0.1), clamp z >= 0.15
- Identity quaternion, zero velocities

**`close()`**: Close viewer if open.

### 3. `mujoco_rl/pd_controller.py`

**Mirror this file line-by-line**: `/Users/kaushikskaja/Documents/GitHub/mujoco_example/src/pd_controller.py`

Read that file first, then adapt with ONE change:

```python
class PDController:
    def __init__(self, kp: float = 4.0, kd: float = 3.0):
        self.kp = kp
        self.kd = kd

    def __call__(self, obs: np.ndarray, target: np.ndarray) -> np.ndarray:
        # ... same PD law as drone version ...
        # ONLY CHANGE: gravity compensation = 9.81 (for 1.0kg agent)
        # instead of 4.905 (which was for 0.5kg drone)
        world_force[2] += 9.81  # was += 4.905
```

Everything else (error clamping, quaternion inverse, body-frame rotation, action mapping) stays identical.

---

## Integration Contract

Agent 2 will import from your files:
```python
from living_room_env import LivingRoomNavEnv, load_trajectories
from pd_controller import PDController
```

Agent 2 depends on these being available:
- `LivingRoomNavEnv` with `observation_space.shape == (28,)`, `action_space.shape == (4,)`
- Attributes: `model`, `data`, `_traj`, `_wp`, `agent_body_id`, `max_episode_steps`
- `load_trajectories(path)` returning `list[np.ndarray]`
- `PDController().__call__(obs, target)` returning `np.ndarray` shape (4,)

---

## Coordinate System Note

Three.js uses Y-up, MuJoCo uses Z-up. The trajectory data will already be converted to MuJoCo coordinates by Agent 2 before being fed to your environment. Your env just expects waypoints in standard MuJoCo (X-right, Y-forward, Z-up) coordinates.

## Quick Verification

After creating the files, you can verify the environment loads:
```python
import numpy as np
from living_room_env import LivingRoomNavEnv

dummy_traj = [np.array([[0, 0, 0.5], [1, 0, 0.5], [2, 0, 0.5]])]
env = LivingRoomNavEnv(trajectories=dummy_traj)
obs, info = env.reset()
print(f"Obs shape: {obs.shape}")  # Should be (28,)
action = env.action_space.sample()
obs, reward, term, trunc, info = env.step(action)
print(f"Step ok, reward={reward:.3f}")
env.close()
```
