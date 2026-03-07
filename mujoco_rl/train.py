"""Train living room navigation with behavioral cloning warm-start + PPO."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from living_room_env import LivingRoomNavEnv, load_trajectories
from pd_controller import PDController


def collect_bc_data(
    trajectories: list[np.ndarray],
    runs_per_traj: int = 5,
    noise_std: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect expert demonstrations using PD controller."""
    env = LivingRoomNavEnv(trajectories=trajectories, max_episode_steps=500)
    pd = PDController()
    all_obs, all_actions = [], []

    for traj_idx in range(len(trajectories)):
        for run in range(runs_per_traj):
            obs, _ = env.reset()
            for step in range(500):
                # Get target waypoint from relative obs: obs[13:16] + obs[0:3]
                target = obs[13:16] + obs[0:3]
                action = pd(obs, target)
                # Add exploration noise
                action = action + np.random.normal(0, noise_std, size=action.shape).astype(np.float32)
                action = np.clip(action, -1.0, 1.0)

                all_obs.append(obs.copy())
                all_actions.append(action.copy())

                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

    env.close()
    return np.array(all_obs), np.array(all_actions)


def train_bc(obs: np.ndarray, actions: np.ndarray, save_path: Path, epochs: int = 100) -> nn.Sequential:
    """Train behavioral cloning network."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    net = nn.Sequential(
        nn.Linear(28, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 4),
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    act_t = torch.tensor(actions, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(obs_t, act_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_obs, batch_act in loader:
            pred = net(batch_obs)
            loss = loss_fn(pred, batch_act)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 20 == 0:
            print(f"  BC epoch {epoch+1}/{epochs}, loss: {total_loss/n_batches:.6f}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_path)
    print(f"  BC weights saved to {save_path}")
    return net


def load_bc_into_ppo(model: PPO, bc_path: Path):
    """Load BC weights into SB3 PPO policy."""
    bc_state = torch.load(bc_path, map_location=model.device, weights_only=True)

    policy_state = model.policy.state_dict()
    mapping = {
        "0.weight": "mlp_extractor.policy_net.0.weight",
        "0.bias": "mlp_extractor.policy_net.0.bias",
        "2.weight": "mlp_extractor.policy_net.2.weight",
        "2.bias": "mlp_extractor.policy_net.2.bias",
        "4.weight": "action_net.weight",
        "4.bias": "action_net.bias",
    }

    for bc_key, ppo_key in mapping.items():
        if bc_key in bc_state and ppo_key in policy_state:
            policy_state[ppo_key] = bc_state[bc_key]

    model.policy.load_state_dict(policy_state)
    print("  BC weights loaded into PPO policy")


def make_env(trajectories: list[np.ndarray], max_steps: int = 1000):
    def _init():
        return LivingRoomNavEnv(trajectories=trajectories, max_episode_steps=max_steps)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train living room navigation with PPO")
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    trajs = load_trajectories(args.trajectories)
    print(f"Loaded {len(trajs)} trajectories")

    bc_path = Path("checkpoints/bc_weights.pt")

    # Phase 1: Behavioral Cloning
    if not args.skip_bc:
        print("Phase 1: Behavioral Cloning")
        print("  Collecting expert data...")
        obs, actions = collect_bc_data(trajs)
        print(f"  Collected {len(obs)} transitions")
        print("  Training BC network...")
        train_bc(obs, actions, bc_path)
    else:
        print("Skipping BC phase")

    # Phase 2: PPO
    print("Phase 2: PPO Training")
    # MLP policies run faster on CPU than MPS
    device = "cpu"

    vec_env = DummyVecEnv([make_env(trajs) for _ in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(trajs)])

    model = PPO(
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
        device=device,
        policy_kwargs=dict(
            net_arch=[64, 64],
            activation_fn=torch.nn.Tanh,
        ),
    )

    # Load BC weights if available
    if bc_path.exists():
        load_bc_into_ppo(model, bc_path)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints/",
        eval_freq=max(5000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save("checkpoints/ppo_livingroom")
    print("Training complete. Model saved to checkpoints/ppo_livingroom.zip")

    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
