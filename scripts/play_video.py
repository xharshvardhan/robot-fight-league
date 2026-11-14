"""Headless playback: runs scripts/play.py but replaces the interactive viewer with a plain stepping loop."""
import sys
import torch
sys.path.insert(0, "scripts")
import play

STEPS = int(sys.argv.pop(1))

class Loop:
  def __init__(self, env, policy): self.env, self.policy = env, policy
  def run(self):
    obs, _ = self.env.reset() if not hasattr(self.env, "get_observations") else (self.env.get_observations(), None)
    resets = 0
    with torch.no_grad():
      for i in range(STEPS):
        out = self.env.step(self.policy(obs))
        obs, dones = out[0], out[2]
        if bool(dones.any()): resets += 1; print("reset at step", i)
    print("RESETS", resets)
    try:  # strike gym: save the label of every landed hit
      import json
      term = self.env.unwrapped.command_manager.get_term("strike")
      json.dump({"targets": float(term.targets.sum()), "hits": term.strike_log}, open("strike_log.json", "w"), indent=1)
    except Exception as e: print("no strike log:", e)

play.ViserPlayViewer = play.NativeMujocoViewer = Loop
play.main()
