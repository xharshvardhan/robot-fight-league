"""Widens a trained body network into a starting point for the shared gym, without changing what it does.

Usage: python grow.py BODY_CHECKPOINT FIGHTER_CHECKPOINT_TEMPLATE OUT
BODY is a G1-Base-V2 snapshot. TEMPLATE is any snapshot of G1-Fighter-Grow (a few iterations is enough): it supplies
the new sizes and the fresh weights for the new neurons. In OUT the old weights sit in the top-left corner of every
layer, the new neurons' outgoing connections are zero and so are the old neurons' connections to the new inputs, so at
the start the network walks exactly as the body network did.
"""
import sys

import torch

body, new = (torch.load(p, map_location="cpu", weights_only=False) for p in sys.argv[1:3])
for net in ("actor_state_dict", "critic_state_dict"):
  old, out = body[net], new[net]
  n_in = old["mlp.0.weight"].shape[1]
  for k in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
    out[k][:, n_in:] = float("mean" not in k)   # new inputs: mean 0, spread 1
    out[k][:, :n_in] = old[k]
  out["obs_normalizer.count"] = old["obs_normalizer.count"]
  for i in (0, 2, 4, 6):
    w, b, ow, ob = out[f"mlp.{i}.weight"], out[f"mlp.{i}.bias"], old[f"mlp.{i}.weight"], old[f"mlp.{i}.bias"]
    r, c = ow.shape
    w[:, c:] = 0.0 if i else w[:, c:]   # nothing downstream listens to the new neurons yet (layer 0's new columns are new inputs)
    w[:r, c:] = 0.0                     # old neurons ignore new inputs and new neurons
    w[:r, :c] = ow
    b[:r] = ob; b[r:] = 0.0
  if "distribution.std_param" in old:
    out["distribution.std_param"] = old["distribution.std_param"].clamp(min=0.4)  # room to explore the new skills
new["optimizer_state_dict"]["state"] = {}  # fresh optimiser memory
new["iter"] = 0
torch.save(new, sys.argv[3])
print("widened", {k: tuple(v.shape) for k, v in new["actor_state_dict"].items() if k.endswith("weight")})
