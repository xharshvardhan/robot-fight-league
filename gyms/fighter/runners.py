"""The two ways of getting one network, as training loops for the shared gym.

GrowRunner (arm 1): ordinary training of a widened network whose old part (the body network's weights) only takes a
fraction of every update, so walking is protected while new skills are learned.
DistilRunner (arm 2): no rewards at first. The student acts, and at every step is pulled toward what the specialist for
that episode's mode would have done (walk, strike, defend or get up). Its value estimate is fitted alongside so ordinary
training can take over afterwards (G1-Fighter-Tune).
"""
from __future__ import annotations

import os
import re
from itertools import chain

import torch
from mjlab.asset_zoo.robots import get_g1_robot_cfg as specialist_robot_cfg

from src.tasks.velocity.rl import VelocityOnPolicyRunner

OLD = {"actor": (105, 512, 256, 128), "critic": (120, 512, 256, 128)}  # input and hidden sizes of the body network
WIDE = (768, 384, 192)
KEEP = 0.2  # share of each update the old weights accept


def old_block(sizes, layer, p):
  """The part of a layer's weight or bias that came from the body network."""
  rows = sizes[layer + 1] if layer + 1 < len(sizes) else p.shape[0]
  return (slice(0, rows), slice(0, sizes[layer])) if p.dim() == 2 else (slice(0, rows),)


STD_CAP = 0.6  # v2 runs: exploration noise may not grow past this (it reached 1.6 in the first experiment)


class CapRunner(VelocityOnPolicyRunner):
  """Ordinary training with the exploration noise capped."""
  std_cap = STD_CAP

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    update = self.alg.update

    def capped_update():
      out = update()
      with torch.no_grad():
        self.alg.actor.distribution.std_param.clamp_(max=self.std_cap)
      return out

    self.alg.update = capped_update


class GrowRunner(VelocityOnPolicyRunner):
  cap = False

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    update = self.alg.update

    def protected_update():
      nets = {n: list(getattr(self.alg, n).mlp.parameters()) for n in OLD}
      before = {n: [p.detach().clone() for p in ps] for n, ps in nets.items()}
      out = update()
      with torch.no_grad():
        for n, ps in nets.items():
          for i, (p, b) in enumerate(zip(ps, before[n])):
            blk = old_block(OLD[n], i // 2, p)
            p[blk] = b[blk] + KEEP * (p[blk] - b[blk])
        if self.cap:
          self.alg.actor.distribution.std_param.clamp_(max=STD_CAP)
      return out

    self.alg.update = protected_update


class GrowCapRunner(GrowRunner):
  cap = True


class Teacher:
  def __init__(self, path, device):
    sd = path if isinstance(path, dict) else torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
    sd = {k: v.detach().clone().to(device) for k, v in sd.items()}   # a private copy: the live network keeps changing
    self.mean, self.std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]
    self.layers = [(sd[f"mlp.{i}.weight"], sd[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

  def __call__(self, x):
    x = (x - self.mean) / (self.std + 1e-2)
    for i, (w, b) in enumerate(self.layers):
      x = x @ w.T + b
      if i < 3:
        x = torch.nn.functional.elu(x)
    return x


class DistilRunner(VelocityOnPolicyRunner):
  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    env, alg, dev = self.env, self.alg, self.device
    base = env.unwrapped
    robot, term = base.scene["robot"], base.action_manager.get_term("joint_pos")
    d = os.path.expanduser(os.environ.get("TEACHERS", "~/teachers"))
    walk, strike, defend, getup = (Teacher(f"{d}/{n}.pt", dev) for n in ("walk", "strike", "defend", "getup"))

    # The strike, defence and get-up specialists were trained around a different resting pose: translate to and from it.
    rest = specialist_robot_cfg().init_state.joint_pos
    pose = lambda names: torch.tensor([next((v for k, v in rest.items() if re.fullmatch(k, n)), 0.0) for n in names], device=dev)
    dq_obs = robot.data.default_joint_pos[0] - pose(robot.joint_names)   # joint order
    scale = term.scale[0] if torch.is_tensor(term.scale) else term.scale
    dq_act = (term.offset[0] - pose(term.target_names)) / scale          # action order, in action units

    om = base.observation_manager
    names, dims = om.active_terms["actor"], om._group_obs_term_dim["actor"]
    at, sl = 0, {}
    for n, dim in zip(names, dims):
      sl[n] = slice(at, at + dim[0]); at += dim[0]
    print("[distil] input layout:", {n: (s.start, s.stop) for n, s in sl.items()})

    def teach(o):
      common = torch.cat([o[:, sl["base_ang_vel"]], o[:, sl["projected_gravity"]], o[:, sl["joint_pos"]] + dq_obs,
                          o[:, sl["joint_vel"]], o[:, sl["actions"]] + dq_act], dim=-1)
      out = torch.stack([walk(o[:, :OLD["actor"][0]]), strike(torch.cat([o[:, sl["strike"]], common], -1)) - dq_act,
                         defend(torch.cat([o[:, sl["threat"]], common], -1)) - dq_act, getup(common) - dq_act], dim=1)
      return out[torch.arange(len(o), device=dev), base.fight_mode]

    opt = torch.optim.Adam(chain(alg.actor.parameters(), alg.critic.parameters()), lr=1e-3)
    alg.train_mode()
    self.logger.init_logging_writer()
    obs = env.get_observations().to(dev)
    steps = self.cfg["num_steps_per_env"]
    for it in range(num_learning_iterations):
      err = torch.zeros(4, device=dev); cnt = torch.zeros(4, device=dev)
      for _ in range(steps):
        with torch.no_grad():
          target = teach(obs["actor"])
        mean = alg.actor(obs)
        value = alg.critic(obs)
        with torch.no_grad():
          act = mean + 0.15 * torch.randn_like(mean)  # a little noise so the student sees states slightly off the teacher's path
          nobs, rew, dones, extras = env.step(act)
          nobs = nobs.to(dev)
          alg.actor.update_normalization(obs); alg.critic.update_normalization(obs)
          td = rew.view(-1, 1) + 0.99 * alg.critic(nobs) * (1 - dones.float().view(-1, 1))
          m = base.fight_mode
        gap = ((mean - target) ** 2).mean(-1)
        loss = gap.mean() + 0.5 * ((value - td) ** 2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(chain(alg.actor.parameters(), alg.critic.parameters()), 1.0)
        opt.step()
        err.index_add_(0, m, gap.detach()); cnt.index_add_(0, m, torch.ones_like(gap))
        obs = nobs
      self.current_learning_iteration = it
      if it % 10 == 0:
        log = {k.split("/")[-1]: round(float(v), 3) for k, v in extras.get("log", {}).items() if "Episode_Metrics" in k or "assist" in k}
        e = (err / cnt.clamp(min=1)).tolist()
        print(f"[distil] iteration {it}  gap to teacher walk {e[0]:.4f} strike {e[1]:.4f} defend {e[2]:.4f} getup {e[3]:.4f}  {log}", flush=True)
      if it % 50 == 0 or it == num_learning_iterations - 1:
        with torch.no_grad():
          alg.actor.distribution.std_param.fill_(0.3)  # starting exploration for the training that follows
        alg.optimizer = torch.optim.Adam(chain(alg.actor.parameters(), alg.critic.parameters()), lr=1e-4)
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    self.logger.stop_logging_writer()
