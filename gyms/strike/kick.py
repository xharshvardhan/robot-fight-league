"""Kick gym: the strike gym with the pad low and the feet as the striking limbs.

Same order as the strike gym (where the pad is, which direction to hit it in, which leg, how soon), so the same scoring
applies: the ordered leg must arrive moving and press on the pad's face. Two kinds: a front kick (push forward) and a
round kick (swing inward). Pads start at shin height straight ahead; as the robots keep landing kicks the pads rise to
head height, appear further away and off to the side, and the robot has to walk to them.
Falling ends the episode and costs as much as three perfect kicks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_mul, yaw_quat

from . import strike_v2
from .strike_v2 import StrikeCommand, StrikeCommandCfg

FOOT = (0.08, 0.0, -0.02)   # striking point in the ankle link frame (ball of the foot)
LOW, HIGH = 0.30, 1.25      # pad height above the floor at difficulty 0 and the top height at difficulty 1 (1.25 m is the G1's head; v0 stopped at 0.90, belly)


class KickCommand(StrikeCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    ids = self.robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)[0]
    self.fist_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
    self.glove = torch.tensor(FOOT, device=self.device).expand(self.num_envs, 2, 3)

  def _resample_command(self, env_ids):
    k, dev, c, L = len(env_ids), self.device, self.cfg, self.level
    self.missed[env_ids] = self.active[env_ids].float()
    u = lambda lo, hi: torch.rand(k, device=dev) * (hi - lo) + lo
    limb = torch.randint(0, 2, (k,), device=dev)
    side = 1.0 - 2.0 * limb
    kind = torch.randint(0, 2, (k,), device=dev) * 2          # 0 = front kick (forward), 2 = round kick (inward)
    local = torch.stack([u(0.35, 0.55), side * u(0.05, 0.20), u(LOW, LOW + (HIGH - LOW) * L)], dim=-1)
    local[kind == 2, 1] *= -0.5                               # a round kick's pad sits just across the centre line
    dirs = torch.zeros(k, 3, device=dev)
    dirs[kind == 0, 0] = 1.0
    dirs[kind == 2, 1] = -side[kind == 2]
    far = u(0.0, c.far * L) * (torch.rand(k, device=dev) < 0.7)
    local[:, 0] += far
    bearing = u(-c.bearing * L, c.bearing * L)
    zero = torch.zeros(k, device=dev)
    yaw = quat_mul(yaw_quat(self.robot.data.root_link_quat_w[env_ids]), torch.stack([torch.cos(bearing / 2), zero, zero, torch.sin(bearing / 2)], dim=-1))
    self.pad_yaw[env_ids] = yaw
    self.pad_vel[env_ids] = quat_apply(yaw, torch.stack([u(-1, 1), u(-1, 1), zero], dim=-1)) * c.drift * L
    self.window[env_ids] = 2.0 + far / 0.5 + bearing.abs() / 2.0
    self.time_left[env_ids] = self.window[env_ids]
    root = self.robot.data.root_link_pos_w[env_ids].clone(); root[:, 2] = self._env.scene.env_origins[env_ids, 2]
    self.target_w[env_ids] = root + quat_apply(yaw, local)
    self.dir_w[env_ids] = quat_apply(yaw, dirs)
    self.limb[env_ids], self.kind[env_ids], self.active[env_ids] = limb, kind, True
    self.age[env_ids] = 0.0
    self.targets[env_ids] += 1
    self._place_pad(env_ids)


@dataclass(kw_only=True)
class KickCommandCfg(StrikeCommandCfg):
  min_z: float = 0.2
  far: float = 1.0
  drift: float = 0.3

  def build(self, env):
    return KickCommand(self, env)


def hands_up(env):  # both fists at the guard points while the legs work
  c = strike_v2._cmd(env); d = c.robot.data
  if not hasattr(c, "hand_ids"):
    c.hand_ids = c.robot.find_bodies(("left_wrist_yaw_link", "right_wrist_yaw_link"), preserve_order=True)[0]
  off = strike_v2.quat_apply_inverse(d.root_link_quat_w[:, None].expand(-1, 2, -1), d.body_link_pos_w[:, c.hand_ids] - d.root_link_pos_w[:, None])
  return torch.exp(-torch.norm(off - c.guard, dim=-1) / 0.2).mean(-1)


def env_cfg(play: bool = False):
  cfg = strike_v2.env_cfg(play=play)
  cfg.commands["strike"] = KickCommandCfg(resampling_time_range=(2.0, 2.0), debug_vis=True, log=play, start_level=1.0 if play else 0.0)
  pad = ContactMatch(mode="geom", pattern="pad", entity="pad")
  cfg.scene.sensors = (*(s for s in cfg.scene.sensors if s.name not in ("left_pad", "right_pad")),
    *(ContactSensorCfg(name=f"{s}_pad", primary=ContactMatch(mode="subtree", pattern=f"{s}_knee_link", entity="robot"), secondary=pad,
                       fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4) for s in ("left", "right")))
  cfg.rewards["guard"] = RewardTermCfg(func=hands_up, weight=0.5)
  return cfg


def rl_cfg():
  cfg = strike_v2.rl_cfg()
  cfg.experiment_name = "g1_kick_v0"
  return cfg
