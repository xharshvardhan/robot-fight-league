"""Runs the two-robot test and prints the numbers. Usage: spar_run.py STEPS TASK --checkpoint_file=... --num_envs N --viewer viser"""
import json, sys, time

import torch

sys.path.insert(0, "scripts")
import play

STEPS = int(sys.argv.pop(1))


class Loop:
  def __init__(self, env, policy): self.env, self.policy = env, policy

  def run(self):
    env, base = self.env, self.env.unwrapped
    term, tm = base.command_manager.get_term("strike"), base.termination_manager
    obs = env.get_observations()
    ends = {k: 0 for k in tm.active_terms}
    from src.tasks.spar.spar import dummy_down
    was = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    downs = down_steps = 0
    t0 = time.time()
    with torch.no_grad():
      for i in range(STEPS):
        obs = env.step(self.policy(obs))[0]
        for k in ends:
          ends[k] += int(tm.get_term(k).sum())
        if i < 100: continue  # settle first
        now = dummy_down(base)
        downs += int((now & ~was).sum()); down_steps += int(now.sum()); was = now
    dt = time.time() - t0
    f = torch.tensor([h["force_along_order_N"] for h in term.strike_log] or [0.0])
    v = torch.tensor([h["fist_velocity_before"] for h in term.strike_log] or [[0.0, 0.0, 0.0]]).norm(dim=-1)
    minutes = (STEPS - 100) * base.step_dt * base.num_envs / 60
    print("SPAR", json.dumps({"task": sys.argv[1], "robots_pairs": base.num_envs, "sim_seconds_each": STEPS * base.step_dt,
      "pair_steps_per_second": round(STEPS * base.num_envs / dt), "punches_landed": len(term.strike_log),
      "force_N_median": round(float(f.median()), 1), "force_N_p90": round(float(f.quantile(0.9)), 1), "force_N_max": round(float(f.max()), 1),
      "fist_speed_median": round(float(v.median()), 2), "attacker_episode_ends": ends,
      "dummy_knockdowns_per_minute": round(downs / minutes, 2), "dummy_share_of_time_down": round(down_steps / ((STEPS - 100) * base.num_envs), 3),
      "dummy_seconds_down_per_knockdown": round(down_steps * base.step_dt / max(downs, 1), 2)}))
    json.dump(term.strike_log[:200], open("strike_log.json", "w"), indent=1)


play.ViserPlayViewer = play.NativeMujocoViewer = Loop
play.main()
