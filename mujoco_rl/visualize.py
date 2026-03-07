"""Visualize trained living room policy or expert PD controller.

Interactive mode requires mjpython on macOS:
    .venv/bin/mjpython visualize.py --expert --trajectories trajectories/sample.json
"""

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from living_room_env import LivingRoomNavEnv, load_trajectories
from pd_controller import PDController


def run_interactive(env: LivingRoomNavEnv, model=None, expert: bool = False):
    """Run in interactive MuJoCo viewer using launch_passive."""
    pd = PDController() if expert else None
    obs, _ = env.reset()

    mode = "Expert PD" if expert else "Trained policy"
    print(f"Launching viewer with {mode} controller. Close window to exit.")

    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    ep = 0

    while viewer.is_running():
        if expert:
            target = env._traj[min(env._wp, len(env._traj) - 1)]
            action = pd(obs, target)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        viewer.sync()
        time.sleep(0.02)

        if terminated or truncated:
            ep += 1
            wp = info.get("wp_idx", 0)
            wp_total = info.get("wp_total", 0)
            status = "COMPLETE" if info.get("trajectory_complete") else (
                "COLLISION" if info.get("collision") else "TIMEOUT")
            print(f"  Episode {ep}: wp={wp}/{wp_total} [{status}]")
            obs, _ = env.reset()

    viewer.close()


def run_headless(env: LivingRoomNavEnv, model=None, expert: bool = False, n_episodes: int = 3):
    """Run episodes without viewer and print stats."""
    pd = PDController() if expert else None

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for step in range(env.max_episode_steps):
            if expert:
                target = env._traj[min(env._wp, len(env._traj) - 1)]
                action = pd(obs, target)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_reward += r
            if term or trunc:
                break
        wp = info.get("wp_idx", 0)
        wp_total = info.get("wp_total", 0)
        status = "COMPLETE" if info.get("trajectory_complete") else ("COLLISION" if info.get("collision") else "TIMEOUT")
        print(f"  Episode {ep+1}: {step+1} steps, reward={total_reward:.1f}, wp={wp}/{wp_total} [{status}]")


def _add_visual_markers(scene, env: LivingRoomNavEnv):
    """Draw waypoint markers and a line from agent to current target."""
    traj = env._traj
    wp = env._wp
    agent_pos = env.data.sensordata[0:3]

    # Draw trajectory path lines (green for completed, yellow for remaining)
    for i in range(len(traj) - 1):
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                            np.zeros(3), np.zeros(9), np.float32([0, 0, 0, 0]))
        # Connector: from traj[i] to traj[i+1]
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 0.01,
                             traj[i].astype(np.float32),
                             traj[i + 1].astype(np.float32))
        if i < wp:
            g.rgba[:] = [0.3, 0.8, 0.3, 0.8]  # green = completed
        else:
            g.rgba[:] = [1.0, 0.9, 0.0, 0.8]  # yellow = remaining
        scene.ngeom += 1

    # Draw waypoint spheres
    for i, pt in enumerate(traj):
        if scene.ngeom >= scene.maxgeom:
            break
        g = scene.geoms[scene.ngeom]
        if i < wp:
            rgba = np.float32([0.2, 0.7, 0.2, 0.5])  # green = reached
            size = np.float32([0.06, 0.06, 0.06])
        elif i == wp:
            rgba = np.float32([1.0, 0.2, 0.2, 0.9])  # red = current target
            size = np.float32([0.1, 0.1, 0.1])
        else:
            rgba = np.float32([1.0, 0.9, 0.0, 0.5])  # yellow = future
            size = np.float32([0.07, 0.07, 0.07])
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, size,
                            pt.astype(np.float32), np.eye(3).flatten().astype(np.float32), rgba)
        scene.ngeom += 1

    # Draw line from agent to current target
    if wp < len(traj) and scene.ngeom < scene.maxgeom:
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3),
                            np.zeros(3), np.zeros(9), np.float32([0, 0, 0, 0]))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_LINE, 0.005,
                             agent_pos.astype(np.float32),
                             traj[wp].astype(np.float32))
        g.rgba[:] = [1.0, 0.3, 0.3, 0.6]
        scene.ngeom += 1


def run_record(env: LivingRoomNavEnv, model=None, expert: bool = False, output_path: str = "demo.mp4"):
    """Record an episode to mp4."""
    pd = PDController() if expert else None

    renderer = mujoco.Renderer(env.model, width=1280, height=720)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.agent_body_id
    cam.distance = 3.0
    cam.azimuth = -135
    cam.elevation = -25
    cam.lookat[:] = [0, 0, 1]
    frames = []

    obs, _ = env.reset()
    done = False

    while not done:
        if expert:
            target = env._traj[min(env._wp, len(env._traj) - 1)]
            action = pd(obs, target)
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        renderer.update_scene(env.data, camera=cam)
        _add_visual_markers(renderer.scene, env)
        frames.append(renderer.render().copy())

    renderer.close()

    try:
        import imageio.v3 as iio
        iio.imwrite(output_path, np.stack(frames), fps=50)
    except ImportError:
        from PIL import Image
        gif_path = output_path.replace(".mp4", ".gif")
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=20, loop=0)
        output_path = gif_path

    wp_reached = info.get("wp_idx", 0)
    wp_total = info.get("wp_total", 0)
    print(f"Recorded {len(frames)} frames to {output_path} (wp={wp_reached}/{wp_total})")


def main():
    parser = argparse.ArgumentParser(description="Visualize living room navigation")
    parser.add_argument("--model", type=str, help="Path to PPO checkpoint")
    parser.add_argument("--trajectories", type=str, required=True)
    parser.add_argument("--expert", action="store_true", help="Use PD controller")
    parser.add_argument("--record", type=str, help="Output video path")
    parser.add_argument("--headless", action="store_true", help="Run without viewer, print stats")
    args = parser.parse_args()

    trajs = load_trajectories(args.trajectories)

    model = None
    if not args.expert:
        if args.model is None:
            parser.error("--model is required unless --expert is set")
        model = PPO.load(args.model, device="cpu")

    env = LivingRoomNavEnv(trajectories=trajs, max_episode_steps=1000)

    if args.record:
        run_record(env, model=model, expert=args.expert, output_path=args.record)
    elif args.headless:
        run_headless(env, model=model, expert=args.expert)
    else:
        run_interactive(env, model=model, expert=args.expert)

    env.close()


if __name__ == "__main__":
    main()
