# Agent 2: Training Pipeline Builder

## Overall Project Goal

We are building an RL training pipeline for a robot navigating a cozy living room. Trajectory data (x,y,z waypoints) is collected from a Three.js frontend. We recreate the environment in MuJoCo and train an RL agent to follow trajectories using Behavioral Cloning + PPO. A teammate already built an identical pipeline for a drone at `/Users/kaushikskaja/Documents/GitHub/mujoco_example` — we mirror that exact pattern for a living room navigation task.

The full pipeline: Trajectory JSON -> MuJoCo Env -> BC warm-start -> PPO fine-tuning -> Trained Policy

There are **two agents** working in parallel:
- **Agent 1 (separate)**: MuJoCo environment + expert controller
- **Agent 2 (YOU)**: Training pipeline, visualization, data conversion, project config

You must NOT create files that Agent 1 owns (living_room_env.py, pd_controller.py, __init__.py).

---

## Your Goal

Create the data converter, training pipeline, visualization, and project config. You import from Agent 1's files.

## Output Directory

`/Users/kaushikskaja/Documents/GitHub/deepmind-robotics/mujoco_rl/`

Create this directory and `mujoco_rl/trajectories/` and `mujoco_rl/checkpoints/` if they don't exist.

---

## What Agent 1 Provides (your imports)

Agent 1 creates these files in the same `mujoco_rl/` directory:
- `living_room_env.py` with `LivingRoomNavEnv` class and `load_trajectories()` function
- `pd_controller.py` with `PDController` class

Your imports:
```python
from living_room_env import LivingRoomNavEnv, load_trajectories
from pd_controller import PDController
```

Interface contract:
- `LivingRoomNavEnv(trajectories=list[np.ndarray], max_episode_steps=int)` — Gymnasium env
- `observation_space.shape == (28,)`: pos(3) + quat(4) + linvel(3) + angvel(3) + 5_relative_waypoints(15)
- `action_space.shape == (4,)`: in [-1, 1]
- Attributes: `model`, `data`, `_traj`, `_wp`, `agent_body_id`
- Info dict keys: `wp_idx`, `wp_total`, `trajectory_complete`, `collision`
- `load_trajectories(path)` — reads JSON `[{"target": "...", "waypoints": [[x,y,z],...]}]`, returns `list[np.ndarray]`
- `PDController(kp=4.0, kd=3.0).__call__(obs, target)` — returns action array shape (4,)

---

## Files to Create

### 1. `mujoco_rl/trajectories/sample.json`

The user's sample trajectory data in Three.js format:
```json
[{"x": 0, "y": 2, "z": 10}, {"x": 1, "y": 2, "z": 9}, ... {"x": 9, "y": 2, "z": 1}]
```

Convert to mujoco_example format with Three.js->MuJoCo coordinate swap (X stays, Z -> -Y, Y -> Z):

```json
[
  {
    "target": "living_room_path",
    "waypoints": [
      [0, -10, 2], [1, -9, 2], [2, -8, 2], [3, -7, 2], [4, -6, 2],
      [5, -5, 2], [6, -4, 2], [7, -3, 2], [8, -2, 2], [9, -1, 2]
    ]
  }
]
```

### 2. `mujoco_rl/convert_trajectories.py`

Data conversion utility. **No direct reference file** — this is new, but simple.

```python
"""Convert trajectory data from Three.js format or MongoDB to MuJoCo training format."""

import argparse
import asyncio
import json
import pathlib


def convert_threejs_trajectory(
    poses: list[dict],
    target_name: str = "path_1",
) -> dict:
    """Convert [{x, y, z}, ...] to {"target": name, "waypoints": [[x,-z,y], ...]}

    Applies coordinate swap: Three.js Y-up -> MuJoCo Z-up
    Three.js (X, Y, Z) -> MuJoCo (X, -Z, Y)
    """


def convert_threejs_file(
    input_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
) -> None:
    """Read JSON of [{x,y,z},...] or list-of-lists, write mujoco_example format."""


async def fetch_from_mongodb(
    mongodb_url: str,
    db_name: str = "wapp",
    output_path: str | pathlib.Path = "trajectories/trajectories.json",
) -> None:
    """Query Trajectory + Pose documents from MongoDB, produce output JSON.

    Uses motor.motor_asyncio.AsyncIOMotorClient (same as server/database.py).
    Queries 'trajectory' collection, then for each trajectory queries 'pose'
    collection sorted by iteration_num, extracts [xPos, yPos, zPos].
    """


def main():
    """CLI: --input PATH | --mongodb-url URL, --output PATH, --db-name NAME"""
```

Reference for MongoDB schema: `/Users/kaushikskaja/Documents/GitHub/deepmind-robotics/server/models.py`
- Pose has: `xPos`, `yPos`, `zPos`, `iteration_num`, `trajectory_fk` (Link to Trajectory)
- Trajectory has: `environment_fk`, `poses` list

### 3. `mujoco_rl/train.py`

**Mirror this file line-by-line**: `/Users/kaushikskaja/Documents/GitHub/mujoco_example/src/train.py`

Read that file first, then adapt with import changes only.

#### `collect_bc_data(trajectories, runs_per_traj=5, noise_std=0.02) -> tuple[np.ndarray, np.ndarray]`

Exact mirror of reference. Changes:
- `LivingRoomNavEnv(trajectories=trajectories, max_episode_steps=500)` instead of `DroneTrajectoryEnv`
- `PDController()` import from `pd_controller` instead of `src.pd_controller`
- Target computation: `target = obs[13:16] + obs[0:3]` (relative waypoint + position = absolute)
- 500 steps per episode, 5 runs per trajectory
- Add Gaussian noise std=0.02, clip to [-1,1]

#### `train_bc(obs, actions, save_path, epochs=100) -> nn.Sequential`

Exact mirror. Network architecture:
```python
nn.Sequential(
    nn.Linear(28, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 4),
)
```
- Device: MPS if available, else CPU
- Optimizer: Adam lr=3e-4
- Loss: MSELoss
- DataLoader: batch_size=256, shuffle=True
- 100 epochs, print every 20

#### `load_bc_into_ppo(model: PPO, bc_path: Path) -> None`

Exact mirror. Weight mapping:
```
BC "0.weight"/"0.bias"   -> PPO "mlp_extractor.policy_net.0.weight"/.bias
BC "2.weight"/"2.bias"   -> PPO "mlp_extractor.policy_net.2.weight"/.bias
BC "4.weight"/"4.bias"   -> PPO "action_net.weight"/.bias
```

#### `make_env(trajectories, max_steps=1000)`

Factory closure returning `LivingRoomNavEnv(trajectories=trajectories, max_episode_steps=max_steps)`.

#### `main()`

CLI args:
- `--trajectories` (required): path to JSON
- `--timesteps` (default=200000)
- `--skip-bc` (flag)
- `--n-envs` (default=4)

PPO config (exact match to reference):
```python
PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.005,
    verbose=1,
    device="cpu",
    policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
)
```

EvalCallback: `eval_freq=max(5000 // n_envs, 1)`, `n_eval_episodes=5`, `deterministic=True`, `best_model_save_path="checkpoints/"`

Save final model: `checkpoints/ppo_livingroom.zip`

### 4. `mujoco_rl/visualize.py`

**Mirror this file line-by-line**: `/Users/kaushikskaja/Documents/GitHub/mujoco_example/src/visualize.py`

Read that file first, then adapt with import changes only.

#### `run_interactive(env, model=None, expert=False)`
- `mujoco.viewer.launch_passive(env.model, env.data)`
- Expert: target from `env._traj[min(env._wp, len(env._traj)-1)]`, call `pd(obs, target)`
- Model: `model.predict(obs, deterministic=True)`
- Print episode stats: waypoints, COMPLETE/COLLISION/TIMEOUT

#### `run_headless(env, model=None, expert=False, n_episodes=3)`
- Run episodes without viewer, print stats per episode

#### `_add_visual_markers(scene, env)`
- Green lines for completed path segments, yellow for remaining
- Waypoint spheres: green (reached), red (current), yellow (future)
- Red line from agent to current target
- Use `env.agent_body_id` instead of `env.drone_body_id`

#### `run_record(env, model=None, expert=False, output_path="demo.mp4")`
- `mujoco.Renderer(env.model, width=1280, height=720)`
- Camera: TRACKING mode, `trackbodyid=env.agent_body_id`, distance=3.0, azimuth=-135, elevation=-25
- Add visual markers per frame
- Write with imageio at 50fps, fallback to PIL GIF

#### `main()`
CLI: `--model`, `--trajectories` (required), `--expert`, `--record`, `--headless`

### 5. `mujoco_rl/pyproject.toml`

```toml
[project]
name = "livingroom-rl"
version = "0.1.0"
description = "RL training pipeline for living room navigation using MuJoCo"
requires-python = ">=3.12"
dependencies = [
    "gymnasium>=1.2.3",
    "imageio[ffmpeg]>=2.37.2",
    "mujoco>=3.5.0",
    "numpy>=2.4.2",
    "stable-baselines3>=2.4.0",
    "torch>=2.10.0",
    "motor>=3.3.0",
]
```

---

## Reference Files to Read Before Writing Code

Read these files first to understand the exact patterns:
1. `/Users/kaushikskaja/Documents/GitHub/mujoco_example/src/train.py` — primary template for train.py
2. `/Users/kaushikskaja/Documents/GitHub/mujoco_example/src/visualize.py` — primary template for visualize.py
3. `/Users/kaushikskaja/Documents/GitHub/mujoco_example/trajectories/sample.json` — data format
4. `/Users/kaushikskaja/Documents/GitHub/deepmind-robotics/server/models.py` — MongoDB Pose/Trajectory schema

---

## Quick Verification

After creating files (assumes Agent 1's files also exist):
```bash
cd /Users/kaushikskaja/Documents/GitHub/deepmind-robotics/mujoco_rl
# Convert sample data
python convert_trajectories.py --input trajectories/sample.json --output trajectories/converted.json
# Quick training test
python train.py --trajectories trajectories/sample.json --timesteps 50000
# Expert controller test
python visualize.py --trajectories trajectories/sample.json --expert --headless
# Trained model test
python visualize.py --trajectories trajectories/sample.json --model checkpoints/best_model.zip --headless
```
