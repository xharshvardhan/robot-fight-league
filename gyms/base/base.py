"""Athletic base v1: the kit's flat-ground walking task, plus a body state the network can feel and must cope with.

Added on top of Unitree-G1-Flat (no human clips, same gait rewards):
  strength  one number per fighter (0.6-1.0) scaling every motor's torque limit
  weakness  one number per limb group (legs, waist, arms); usually 1.0, sometimes down to 0.3 (a damaged limb)
  energy    a reserve that drains with mechanical work and refills slowly; below 30% every motor fades toward half torque
All seven numbers are inputs to the network, so one network serves strong, weak, hurt and tired fighters.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationTermCfg

from src.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from src.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

GROUPS = ("left_(hip|knee|ankle)", "right_(hip|knee|ankle)", "waist", "left_(shoulder|elbow|wrist)", "right_(shoulder|elbow|wrist)")
FADE_BELOW = 0.3      # reserve fraction under which motors start to fade
MIN_FADE = 0.5        # torque fraction left at an empty reserve


class BodyState(CommandTerm):
  cfg: BodyStateCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene["robot"]
    n, dev = self.num_envs, self.device
    self.ctrl = self.robot.indexing.ctrl_ids
    names = self.robot.actuator_names
    group = [next(g for g, pat in enumerate(GROUPS) if re.search(pat, a)) for a in names]
    self.group = torch.tensor(group, device=dev)
    self.limit = env.sim.get_default_field("actuator_forcerange")[self.ctrl].to(dev)  # [motors, 2]
    self.strength = torch.ones(n, 1, device=dev)
    self.weak = torch.ones(n, len(GROUPS), device=dev)
    self.energy = torch.ones(n, 1, device=dev)
    self.power = torch.zeros(n, device=dev)
    for k in ("power_W", "energy"):
      self.metrics[k] = torch.zeros(n, device=dev)

  @property
  def command(self):
    return torch.cat([self.strength, self.weak, self.energy], dim=-1)

  def _update_metrics(self):
    self.metrics["power_W"][:] = self.power
    self.metrics["energy"][:] = self.energy[:, 0]

  def _resample_command(self, env_ids):  # once per episode (the resampling time is set longer than an episode)
    k, dev = len(env_ids), self.device
    if self.cfg.fixed:
      self.strength[env_ids] = 1.0; self.weak[env_ids] = 1.0; self.energy[env_ids] = 1.0
      return
    c = self.cfg
    self.strength[env_ids] = torch.empty(k, 1, device=dev).uniform_(c.min_strength, 1.0)
    hurt = torch.rand(k, len(GROUPS), device=dev) < c.hurt_chance
    self.weak[env_ids] = torch.where(hurt, torch.empty(k, len(GROUPS), device=dev).uniform_(c.min_weak, 1.0), torch.ones(1, device=dev))
    self.energy[env_ids] = torch.empty(k, 1, device=dev).uniform_(0.1, 1.0)

  def _update_command(self):
    sim = self._env.sim
    self.power = (sim.data.actuator_force[:, self.ctrl] * sim.data.actuator_velocity[:, self.ctrl]).abs().sum(-1)
    self.energy += ((self.cfg.recharge_w - self.power) * self._env.step_dt / self.cfg.capacity_j)[:, None]
    self.energy.clamp_(0.0, 1.0)
    fade = MIN_FADE + (1 - MIN_FADE) * (self.energy / FADE_BELOW).clamp(max=1.0)
    scale = self.strength * fade * self.weak[:, self.group]            # [envs, motors]
    sim.model.actuator_forcerange[:, self.ctrl] = self.limit[None] * scale[..., None]


@dataclass(kw_only=True)
class BodyStateCfg(CommandTermCfg):
  fixed: bool = False  # playback: full strength, no damage, full reserve
  capacity_j: float = 3000.0  # joules of mechanical work in a full reserve
  recharge_w: float = 150.0   # watts refilled all the time: effort below this is sustainable for ever
  min_strength: float = 0.6
  min_weak: float = 0.3
  hurt_chance: float = 0.15   # per limb group, per episode

  def build(self, env):
    return BodyState(self, env)


@requires_model_fields("actuator_forcerange")
def per_fighter_torque_limits(env, env_ids):
  """Does nothing itself: it tells the simulator to keep a separate torque-limit table for every fighter."""


def env_cfg(play: bool = False, v2: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = unitree_g1_flat_env_cfg(play=play)
  cfg.commands["body"] = BodyStateCfg(resampling_time_range=(1e6, 1e6), fixed=play)
  if v2:  # v1 stopped following fast commands (effort settled at the 150 W refill rate): gentler handicaps, bigger reserve
    cfg.commands["body"] = BodyStateCfg(resampling_time_range=(1e6, 1e6), fixed=play, capacity_j=5000.0, recharge_w=250.0,
                                        min_strength=0.7, min_weak=0.5, hurt_chance=0.1)
  for group in ("actor", "critic"):
    cfg.observations[group].terms["body"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "body"})
  cfg.events["torque_tables"] = EventTermCfg(func=per_fighter_torque_limits, mode="startup")
  if not play:  # reach running speeds inside this run (the kit waits until iteration 5,000)
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
      *([{"step": 1500 * 24, "lin_vel_x": (-0.75, 1.5), "lin_vel_y": (-0.75, 0.75)}, {"step": 2500 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)}] if v2 else
        [{"step": 1000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)}, {"step": 2500 * 24, "lin_vel_x": (-1.0, 2.5)}]),
    ]
  return cfg


def rl_cfg(v2: bool = False):
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = "g1_base_v2" if v2 else "g1_base_v1"
  cfg.max_iterations = 3500 if v2 else 4000
  return cfg
