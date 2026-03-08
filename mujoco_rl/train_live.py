"""Train with a live localhost dashboard showing RL progress in real time."""

import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import numpy as np
import torch  # noqa: F401 (used for torch.nn.Tanh in PPO config)
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from living_room_env import LivingRoomNavEnv, load_trajectories

# ---------- shared metrics store ----------
METRICS = {
    "episode_rewards": [],
    "episode_lengths": [],
    "waypoint_fractions": [],
    "collisions": [],
    "timesteps": [],
    "smooth_rewards": [],
    "smooth_lengths": [],
    "smooth_wp": [],
    "smooth_collisions": [],
    "phase": "idle",
}
LOCK = threading.Lock()
WINDOW = 20


def _smooth_append():
    """Append latest smoothed value (incremental — O(WINDOW) per call)."""
    n = len(METRICS["episode_rewards"])
    if n < WINDOW:
        return
    sl = slice(n - WINDOW, n)
    METRICS["smooth_rewards"].append(float(np.mean(METRICS["episode_rewards"][sl])))
    METRICS["smooth_lengths"].append(float(np.mean(METRICS["episode_lengths"][sl])))
    METRICS["smooth_wp"].append(float(np.mean(METRICS["waypoint_fractions"][sl])))
    METRICS["smooth_collisions"].append(float(np.mean(METRICS["collisions"][sl])))


class LiveMetricsCallback(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                with LOCK:
                    METRICS["episode_rewards"].append(float(info["episode"]["r"]))
                    METRICS["episode_lengths"].append(int(info["episode"]["l"]))
                    METRICS["timesteps"].append(int(self.num_timesteps))
                    wp = info.get("wp_idx", 0)
                    wp_total = info.get("wp_total", 1)
                    METRICS["waypoint_fractions"].append(wp / max(wp_total, 1))
                    METRICS["collisions"].append(1.0 if info.get("collision") else 0.0)
                    _smooth_append()
        return True


# ---------- HTTP server ----------
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>RL Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f1117; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; padding: 20px; }
  h1 { text-align: center; margin-bottom: 8px; font-size: 1.6em; color: #7eb6ff; }
  #phase { text-align: center; margin-bottom: 16px; font-size: 1.1em; color: #aaa; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1200px; margin: 0 auto; }
  .card { background: #1a1d27; border-radius: 12px; padding: 16px; }
  .card h2 { font-size: 0.95em; color: #999; margin-bottom: 8px; }
  canvas { width: 100% !important; }
  #stats { display: flex; justify-content: center; gap: 40px; margin-bottom: 16px; }
  .stat { text-align: center; }
  .stat .val { font-size: 1.8em; font-weight: bold; color: #7eb6ff; }
  .stat .lbl { font-size: 0.8em; color: #888; }
</style></head><body>
<h1>RL Training Dashboard</h1>
<div id="phase">Initializing...</div>
<div id="stats">
  <div class="stat"><div class="val" id="s-episodes">0</div><div class="lbl">Episodes</div></div>
  <div class="stat"><div class="val" id="s-timesteps">0</div><div class="lbl">Timesteps</div></div>
  <div class="stat"><div class="val" id="s-reward">-</div><div class="lbl">Avg Reward (last 20)</div></div>
  <div class="stat"><div class="val" id="s-wp">-</div><div class="lbl">Avg Waypoints %</div></div>
</div>
<div class="grid">
  <div class="card"><h2>Episode Reward</h2><canvas id="c-reward"></canvas></div>
  <div class="card"><h2>Episode Length</h2><canvas id="c-length"></canvas></div>
  <div class="card"><h2>Waypoint Completion (%)</h2><canvas id="c-wp"></canvas></div>
  <div class="card"><h2>Collision Rate (%) — rolling</h2><canvas id="c-col"></canvas></div>
</div>
<script>
const chartOpts = (color, yLabel, sugMax) => ({
  responsive: true, animation: { duration: 0 },
  scales: {
    x: { title: { display: true, text: 'Episode', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#2a2d37' } },
    y: { title: { display: true, text: yLabel, color: '#888' }, ticks: { color: '#666' }, grid: { color: '#2a2d37' }, suggestedMax: sugMax }
  },
  plugins: { legend: { display: true, labels: { color: '#aaa', boxWidth: 12 } } },
  elements: { point: { radius: 0 } }
});
function makeChart(id, label, color, yLabel, sugMax) {
  return new Chart(document.getElementById(id), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Raw', data: [], borderColor: color + '44', borderWidth: 1, fill: false },
        { label: 'Smoothed', data: [], borderColor: color, borderWidth: 2.5, fill: false }
      ]
    },
    options: chartOpts(color, yLabel, sugMax)
  });
}
const cReward = makeChart('c-reward', 'Reward', '#5b9aff', 'Reward');
const cLength = makeChart('c-length', 'Steps', '#ff9f43', 'Steps');
const cWp     = makeChart('c-wp', 'Waypoints %', '#2ed573', '% Reached', 100);
const cCol    = makeChart('c-col', 'Collision %', '#ff4757', 'Collision %', 100);

function update(d) {
  document.getElementById('phase').textContent = 'Phase: ' + d.phase;
  const n = d.episode_rewards.length;
  document.getElementById('s-episodes').textContent = n;
  document.getElementById('s-timesteps').textContent = d.timesteps.length ? d.timesteps[d.timesteps.length-1].toLocaleString() : '0';
  if (d.smooth_rewards.length) {
    document.getElementById('s-reward').textContent = d.smooth_rewards[d.smooth_rewards.length-1].toFixed(1);
    document.getElementById('s-wp').textContent = (d.smooth_wp[d.smooth_wp.length-1]*100).toFixed(1) + '%';
  }

  const rawEps = d.episode_rewards.map((_,i) => i+1);
  const smStart = rawEps.length - d.smooth_rewards.length;
  const smEps = d.smooth_rewards.map((_,i) => smStart + i + 1);

  // Hard-clip y-axis from smoothed data so outliers don't squash the chart
  function boundsFrom(arr, padding) {
    if (!arr.length) return {};
    const mn = Math.min(...arr), mx = Math.max(...arr);
    const r = (mx - mn) || 1;
    return { min: mn - r * padding, max: mx + r * padding };
  }

  function setData(chart, rawX, rawY, smX, smY, autoScale) {
    chart.data.datasets[0].data = rawX.map((x,i) => ({x, y: rawY[i]}));
    chart.data.datasets[1].data = smX.map((x,i) => ({x, y: smY[i]}));
    chart.options.scales.x.type = 'linear';
    if (autoScale && smY.length) {
      const b = boundsFrom(smY, 0.5);
      chart.options.scales.y.min = b.min;
      chart.options.scales.y.max = b.max;
    }
    chart.update();
  }
  setData(cReward, rawEps, d.episode_rewards, smEps, d.smooth_rewards, true);
  setData(cLength, rawEps, d.episode_lengths, smEps, d.smooth_lengths, true);
  setData(cWp, rawEps, d.waypoint_fractions.map(v=>v*100), smEps, d.smooth_wp.map(v=>v*100), false);
  setData(cCol, rawEps, d.collisions.map(v=>v*100), smEps, d.smooth_collisions.map(v=>v*100), false);
}

setInterval(() => {
  fetch('/metrics').then(r => r.json()).then(update).catch(() => {});
}, 1000);
</script></body></html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with LOCK:
                self.wfile.write(json.dumps(METRICS).encode())
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass  # suppress request logs


def start_server(port=8050):
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def make_env(trajectories, max_steps=1000):
    def _init():
        return Monitor(LivingRoomNavEnv(trajectories=trajectories, max_episode_steps=max_steps))
    return _init


# ---------- main ----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=str, default="trajectories/sample.json")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--n-envs", type=int, default=4)
    args = parser.parse_args()

    trajs = load_trajectories(args.trajectories)
    print(f"Loaded {len(trajs)} trajectories")

    # Start web dashboard
    server = start_server(args.port)
    print(f"\n  Dashboard: http://localhost:{args.port}\n")

    # PPO from scratch (no behavioral cloning warm-start)
    with LOCK:
        METRICS["phase"] = "PPO Training (from scratch)"
    print("PPO Training from scratch (random policy)")

    vec_env = DummyVecEnv([make_env(trajs) for _ in range(args.n_envs)])

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.005,
        verbose=0, device="cpu",
        policy_kwargs=dict(net_arch=[64, 64], activation_fn=torch.nn.Tanh),
    )

    metrics_cb = LiveMetricsCallback()

    # Train continuously until Ctrl+C
    start_time = time.time()
    chunk = 8192  # one rollout buffer worth
    total_trained = 0
    print("  Training until Ctrl+C — watch the dashboard!\n")

    try:
        while True:
            model.learn(total_timesteps=chunk, callback=metrics_cb, reset_num_timesteps=False)
            total_trained += chunk
            elapsed = time.time() - start_time
            with LOCK:
                n_eps = len(METRICS["episode_rewards"])
                METRICS["phase"] = f"PPO Training — {elapsed:.0f}s  |  {total_trained:,} steps  |  {n_eps} episodes"
    except KeyboardInterrupt:
        print(f"\nStopped. Trained {total_trained:,} timesteps.")

    model.save("checkpoints/ppo_livingroom")
    vec_env.close()
    server.shutdown()
    print("Model saved to checkpoints/ppo_livingroom.zip")


if __name__ == "__main__":
    main()
