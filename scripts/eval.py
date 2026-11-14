"""Scores one snapshot on one skill. Usage: eval.py STEPS TASK --checkpoint_file=... --num_envs 1024 --viewer viser
TASK is a single-skill playback task (G1-Fighter-Walk etc.), so every robot is in the same mode and the averages are plain.
The first 250 steps are a warm-up; after that every command term's metrics, the walking speed error and falls are averaged.
"""
import sys
from collections import defaultdict

import torch

sys.path.insert(0, "scripts")
import play

STEPS = int(sys.argv.pop(1))


class Loop:
  def __init__(self, env, policy): self.env, self.policy = env, policy

  def run(self):
    env, base = self.env, self.env.unwrapped
    obs = env.get_observations()
    tot, n, falls = defaultdict(float), 0, 0.0
    with torch.no_grad():
      for i in range(STEPS):
        out = env.step(self.policy(obs))
        obs = out[0]
        if i < 250: continue
        n += 1
        falls += float(base.termination_manager.terminated.float().mean())
        d = base.scene["robot"].data
        tot["walk_error"] += float(torch.norm(base.command_manager.get_command("twist")[:, :2] - d.root_link_lin_vel_b[:, :2], dim=-1).mean())
        for name in ("twist", "body", "strike", "threat", "assist"):
          for k, v in base.command_manager.get_term(name).metrics.items():
            tot[f"{name}/{k}"] += float(v.mean())
    res = {k: round(v / n, 4) for k, v in tot.items()}
    res["falls_per_robot_minute"] = round(falls / n / base.step_dt * 60, 3)
    print("EVAL", sys.argv[1], res)


play.ViserPlayViewer = play.NativeMujocoViewer = Loop
play.main()
