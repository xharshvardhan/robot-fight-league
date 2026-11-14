"""Runs bouts and prints the scorecard. Usage: bout_run.py STEPS G1-Bout --checkpoint_file=... --num_envs N --viewer viser"""
import json, sys

import torch

sys.path.insert(0, "scripts")
import play

STEPS = int(sys.argv.pop(1))


class Loop:
  def __init__(self, env, policy): self.env, self.policy = env, policy

  def run(self):
    env, base = self.env, self.env.unwrapped
    red, blue = base.command_manager.get_term("strike"), base.command_manager.get_term("threat")
    obs = env.get_observations()
    n = base.num_envs
    was = torch.zeros(2, n, dtype=torch.bool, device=base.device)
    downs, down_steps = torch.zeros(2), torch.zeros(2)
    modes = torch.zeros(2, 4)
    with torch.no_grad():
      for i in range(STEPS):
        obs = env.step(self.policy(obs))[0]
        now = torch.stack([blue.a_down, blue.b_down])
        downs += (now & ~was).sum(-1).cpu(); down_steps += now.sum(-1).cpu(); was = now
        modes[0] += torch.bincount(base.fight_mode, minlength=4).cpu(); modes[1] += torch.bincount(blue.b_mode, minlength=4).cpu()
    minutes = STEPS * base.step_dt * n / 60
    def side(k, forces):
      f = torch.tensor(forces or [0.0])
      return {"punches_per_minute": round(len(forces) / minutes, 1), "force_N_median": round(float(f.median()), 1), "force_N_p90": round(float(f.quantile(0.9)), 1),
              "knocked_down_per_minute": round(float(downs[k]) / minutes, 2), "share_of_time_down": round(float(down_steps[k]) / (STEPS * n), 3),
              "seconds_down_each_time": round(float(down_steps[k]) * base.step_dt / max(float(downs[k]), 1), 2),
              "time_in_walk_strike_defend_getup": [round(x, 2) for x in (modes[k] / modes[k].sum()).tolist()]}
    print("BOUT", json.dumps({"bouts": n, "seconds_each": STEPS * base.step_dt,
      "red_one_network": side(0, [h["force_along_order_N"] for h in red.strike_log]), "blue_specialists": side(1, blue.b_log)}))
    json.dump(red.strike_log[:200], open("strike_log.json", "w"), indent=1)


play.ViserPlayViewer = play.NativeMujocoViewer = Loop
play.main()
