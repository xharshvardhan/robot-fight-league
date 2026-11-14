"""Combo gym (fight gym, stage 1): several targets are up at once and the robot chooses what to hit, with what, in what order.

Why it exists: in the strike gym one pad appeared at a time, placed relative to wherever the robot had ended up, with a pause
before the next. Lunging and then catching itself cost nothing, so that is what it learned. Here:
- K pads (balls) are always up. The moment one is hit, another appears. Time spent recovering is time not scoring.
- Pads sit at fixed places in the ring (at full difficulty), so ending a strike out of position costs travel.
- Any fist, foot or shin may hit any pad. A pad is worth 3 at head height, 2 at body height, 1 at leg height, scaled by
  force up to 800 N; leg strikes get a quarter more. So a braced, hard strike beats a shove, and a kick is a real choice.
- In three episodes out of ten the order is fixed: only the flagged pad scores.
- Stance is paid all the time (feet apart, weight between the feet, guard up) and drifting off balance costs before a fall does.
- The network also sees its last few readings (a short memory), and a block of "opponent limb" inputs that is only noise
  here, so the same network can go on to face an opponent.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import mdp
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat

from . import strike_v2

K = 4
PAD_R = 0.07
HEAD_Z, BODY_Z = 1.05, 0.65          # a pad at or above HEAD_Z is worth 3, at or above BODY_Z 2, below 1
FULL_FORCE, MIN_FORCE, MIN_SPEED, NEAR = 800.0, 25.0, 1.5, 0.20
UP, DOWN = 20.0, 10.0                # hits per robot-minute above which the pads spread out, and below which they come back in
# striking points: body, offset in that body's frame. The first two are fists, the rest legs.
POINTS = (("left_wrist_yaw_link", (0.1, 0.0, 0.0)), ("right_wrist_yaw_link", (0.1, 0.0, 0.0)),
          ("left_ankle_roll_link", (0.08, 0.0, -0.02)), ("right_ankle_roll_link", (0.08, 0.0, -0.02)),
          ("left_knee_link", (0.0, 0.0, -0.15)), ("right_knee_link", (0.0, 0.0, -0.15)))
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def ball_spec():
  s = mujoco.MjSpec()
  s.worldbody.add_body(name="pad", mocap=True).add_geom(name="pad", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[PAD_R, 0, 0], rgba=[0.9, 0.2, 0.15, 1])
  return s


def zone_value(z):
  return 1.0 + (z >= BODY_Z).float() + (z >= HEAD_Z).float()


class Combo(CommandTerm):
  cfg: ComboCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    self.robot = env.scene["robot"]
    self.pads = [env.scene[f"pad{i}"] for i in range(K)]
    self.sensors = [env.scene[f"hit{i}"] for i in range(K)]
    self.point_ids = torch.tensor(self.robot.find_bodies([b for b, _ in POINTS], preserve_order=True)[0], device=dev)
    self.point_off = torch.tensor([o for _, o in POINTS], device=dev).expand(n, len(POINTS), 3)
    self.foot_ids = self.robot.find_bodies(FEET, preserve_order=True)[0]
    self.guard = torch.tensor(strike_v2.GUARD, device=dev).expand(n, 2, 3)
    z = lambda *s: torch.zeros(n, *s, device=dev)
    self.pos, self.age, self.life = z(K, 3), z(K), z(K) + 5.0
    self.ordered = torch.zeros(n, dtype=torch.bool, device=dev)
    self.next = torch.zeros(n, dtype=torch.long, device=dev)
    self.prev_speed = z(len(POINTS))
    self.points, self.foul, self.near = z(), z(), z() + 1.0
    self.level, self.rate = cfg.start_level, 0.0
    self.tot = {k: z() for k in ("hits", "points", "kicks", "head", "body", "legs", "force", "steps")}
    for k in ("rate", "hits_per_min", "points_per_min", "kicks_per_min", "head_per_min", "body_per_min", "legs_per_min", "hit_force", "level"):
      self.metrics[k] = z()
    self.hit_log = []

  # --- geometry ---
  def strikers(self):
    d = self.robot.data
    r = quat_apply(d.body_link_quat_w[:, self.point_ids], self.point_off)
    return d.body_link_pos_w[:, self.point_ids] + r, d.body_link_lin_vel_w[:, self.point_ids] + torch.cross(d.body_link_ang_vel_w[:, self.point_ids], r, dim=-1)

  def scoreable(self):
    return ~self.ordered[:, None] | (torch.arange(K, device=self.device)[None] == self.next[:, None])

  @property
  def command(self):
    d = self.robot.data
    q = d.root_link_quat_w[:, None].expand(-1, K, -1)
    rel = quat_apply_inverse(q, self.pos - d.root_link_pos_w[:, None])
    return torch.cat([rel, zone_value(self.pos[..., 2])[..., None] / 3.0, self.scoreable().float()[..., None]], dim=-1).flatten(1)

  # --- pads ---
  def _spawn(self, mask):
    """Moves the pads selected by mask [envs, K] to new places."""
    e, k = mask.nonzero(as_tuple=True)
    if not len(e):
      return
    L, dev, d = self.level, self.device, self.robot.data
    u = lambda lo, hi: torch.rand(len(e), device=dev) * (hi - lo) + lo
    origin = self._env.scene.env_origins[e, :2]
    me = d.root_link_pos_w[e, :2]
    centre = origin + (1.0 - L) * (me - origin)          # easy: around the robot. Full difficulty: fixed in the ring
    yaw = yaw_quat(d.root_link_quat_w[e])
    fwd = quat_apply(yaw, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(len(e), 3))[:, :2]
    ang = torch.atan2(fwd[:, 1], fwd[:, 0]) + u(-1.0, 1.0) * (0.6 + 2.54 * L)   # easy: in front. Full: anywhere around
    r = u(0.35, 0.55 + 1.1 * L)
    xy = centre + r[:, None] * torch.stack([torch.cos(ang), torch.sin(ang)], -1)
    off = xy - me; dist = off.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    xy = torch.where(dist < 0.3, me + off / dist * 0.35, xy)                      # never inside the robot
    self.pos[e, k, :2] = xy
    self.pos[e, k, 2] = self._env.scene.env_origins[e, 2] + u(0.85 - 0.55 * L, 1.0 + 0.25 * L)
    self.age[e, k] = 0.0
    self.life[e, k] = u(4.0, 7.0)
    for i in range(K):
      ids = e[k == i]
      if len(ids):
        pose = torch.zeros(len(ids), 7, device=dev); pose[:, :3] = self.pos[ids, i]; pose[:, 3] = 1.0
        self.pads[i].write_mocap_pose_to_sim(pose, env_ids=ids)

  def reset(self, env_ids):
    for t in self.tot.values():
      t[env_ids] = 0.0
    out = super().reset(env_ids)
    self.ordered[env_ids] = torch.rand(len(env_ids), device=self.device) < self.cfg.ordered_share
    self.next[env_ids] = torch.randint(0, K, (len(env_ids),), device=self.device)
    mask = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device); mask[env_ids] = True
    self._spawn(mask)
    self.prev_speed[env_ids] = 0.0
    return out

  def _resample_command(self, env_ids):
    pass

  def _update_metrics(self):
    t = self.tot
    mins = (t["steps"] * self._env.step_dt / 60.0).clamp(min=1e-3)
    for k in ("hits", "points", "kicks", "head", "body", "legs"):
      self.metrics[f"{k}_per_min"][:] = t[k] / mins
    self.metrics["hit_force"][:] = t["force"] / t["hits"].clamp(min=1)
    self.metrics["level"][:] = self.level
    self.metrics["rate"][:] = self.rate

  def _update_command(self):
    dt, dev = self._env.step_dt, self.device
    self.age += dt
    force = torch.stack([s.data.force_history[:, 0].norm(dim=-1).amax(dim=-1) for s in self.sensors], dim=1)   # [envs, K]
    p, v = self.strikers()
    dist = (self.pos[:, :, None] - p[:, None]).norm(dim=-1) - PAD_R                                              # [envs, K, points]
    if self.cfg.legs_low:   # leg-height pads only count for feet and shins (nobody learned to kick while a stooping punch scored)
      low = (self.pos[..., 2] < BODY_Z)[..., None] & (torch.arange(len(POINTS), device=dev) < 2)
      dist = torch.where(low, torch.full_like(dist, 9.0), dist)
    dmin, who = dist.min(dim=-1)
    speed = torch.gather(self.prev_speed, 1, who)
    self.prev_speed = v.norm(dim=-1)
    live = self.scoreable()
    touch = force > MIN_FORCE
    hit = touch & (dmin < NEAR) & (speed > MIN_SPEED) & live & (self.age > 0.15)
    leg = who >= 2
    value = zone_value(self.pos[..., 2])
    pts = hit * value * (0.3 + 0.7 * (force / FULL_FORCE).clamp(max=1.0)) * (1.0 + 0.25 * leg)
    self.points = pts.sum(-1)
    self.foul = (touch & ~hit & (dmin >= NEAR) & (dmin < 5.0)).float().sum(-1)   # a pad pressed by something that is not a fist, foot or shin
    self.near = torch.where(live, dmin, torch.full_like(dmin, 9.0)).amin(dim=-1)
    t = self.tot
    t["steps"] += 1; t["hits"] += hit.sum(-1); t["points"] += self.points; t["kicks"] += (hit & leg).sum(-1); t["force"] += (force * hit).sum(-1)
    t["head"] += (hit & (value == 3)).sum(-1); t["body"] += (hit & (value == 2)).sum(-1); t["legs"] += (hit & (value == 1)).sum(-1)
    if self.cfg.log:
      for e, k in hit.nonzero().tolist():
        self.hit_log.append(dict(env=e, t=round(float(t["steps"][e]) * dt, 2), limb=POINTS[int(who[e, k])][0], force_N=round(float(force[e, k]), 1),
                                 height_m=round(float(self.pos[e, k, 2]), 2), points=round(float(pts[e, k]), 2)))
    expired = self.age > self.life
    if not self.cfg.log:   # difficulty follows hits per robot-minute (the share of pads hit misled: robots that fall early miss few pads)
      self.rate += 0.01 * (float(hit.sum()) / self.num_envs / dt * 60.0 - self.rate)
      self.level = min(1.0, max(0.0, self.level + (2e-4 if self.rate > UP else -2e-4 if self.rate < DOWN else 0.0)))
    done_next = hit.any(-1) & self.ordered
    self.next = torch.where(done_next, (self.next + torch.randint(1, K, (self.num_envs,), device=dev)) % K, self.next)
    self._spawn(hit | expired)


@dataclass(kw_only=True)
class ComboCfg(CommandTermCfg):
  ordered_share: float = 0.3
  legs_low: bool = False
  start_level: float = 0.0
  log: bool = False

  def build(self, env):
    return Combo(self, env)


def _c(env) -> Combo:
  return env.command_manager.get_term("combo")

def scored(env): return _c(env).points / env.step_dt
def bumped(env): return _c(env).foul
def reach(env): return torch.exp(-_c(env).near / 0.4)

def _feet(env):
  c = _c(env); d = c.robot.data
  f = d.body_link_pos_w[:, c.foot_ids, :2]
  return f, d.root_link_pos_w[:, :2] - f.mean(dim=1)

def stance(env):  # feet about 30 cm apart, pelvis over the middle of them, fists at the guard
  c = _c(env); d = c.robot.data
  f, lean = _feet(env)
  apart = torch.exp(-((f[:, 0] - f[:, 1]).norm(dim=-1) - 0.30).abs() / 0.10)
  centred = torch.exp(-lean.norm(dim=-1) / 0.08)
  p, _ = c.strikers()
  q = d.root_link_quat_w[:, None].expand(-1, 2, -1)
  hands = torch.exp(-(quat_apply_inverse(q, p[:, :2] - d.root_link_pos_w[:, None]) - c.guard).norm(dim=-1) / 0.2).mean(-1)
  return (apart + centred + hands) / 3

def off_balance(env):  # pelvis drifting beyond the feet, or sinking: the moments before a fall
  d = _c(env).robot.data
  _, lean = _feet(env)
  z = d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return ((lean.norm(dim=-1) - 0.12).clamp(min=0.0) / 0.1 + (0.66 - z).clamp(min=0.0) * 10.0).clamp(max=5.0)

def ghost_limbs(env):  # stands in for an opponent's fists and feet (position, velocity) until there is one
  return torch.rand(env.num_envs, 24, device=env.device) * 2.0 - 1.0


def env_cfg(play: bool = False, stance_w: float = 1.5, history: int = 5, legs_low: bool = False):
  cfg = strike_v2.env_cfg(play=play)
  cfg.scene.entities.pop("pad")
  for i in range(K):
    cfg.scene.entities[f"pad{i}"] = EntityCfg(spec_fn=ball_spec)
  body = ContactMatch(mode="subtree", pattern="pelvis", entity="robot")
  cfg.scene.sensors = (*(s for s in cfg.scene.sensors if s.name == "self_collision"),
    *(ContactSensorCfg(name=f"hit{i}", primary=ContactMatch(mode="geom", pattern="pad", entity=f"pad{i}"), secondary=body,
                       fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4) for i in range(K)))
  cfg.commands = {"combo": ComboCfg(resampling_time_range=(1e6, 1e6), log=play, start_level=1.0 if play else 0.0, legs_low=legs_low)}
  for group in ("actor", "critic"):
    t = cfg.observations[group].terms
    t["command"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "combo"})
    t["opponent"] = ObservationTermCfg(func=ghost_limbs)
    if history > 1:
      for k in ("base_ang_vel", "gravity", "joint_pos", "joint_vel"):
        t[k].history_length = history
  r = cfg.rewards
  for k in ("reach", "hit", "guard", "waiting", "missed", "foul"):
    r.pop(k)
  r["upright"].weight = -2.0
  r.update({
    "scored": RewardTermCfg(func=scored, weight=3.0),       # full-force head pad 9, body 6, leg 3 (a kick a quarter more); a fall costs 30
    "reach": RewardTermCfg(func=reach, weight=0.5),
    "bumped": RewardTermCfg(func=bumped, weight=-1.0),
    "off_balance": RewardTermCfg(func=off_balance, weight=-2.0 if stance_w else 0.0),
    "stance": RewardTermCfg(func=stance, weight=stance_w),
  })
  cfg.sim.nconmax, cfg.sim.njmax = 60, 400
  cfg.episode_length_s = int(1e9) if play else 20.0
  return cfg


def rl_cfg(name: str = "g1_combo", gamma: float = 0.997):
  cfg = strike_v2.rl_cfg()
  cfg.experiment_name = name
  cfg.algorithm.gamma = gamma   # 0.997 at 50 steps a second values roughly the next 7 seconds (0.99 was 2 seconds)
  cfg.max_iterations = 3000
  return cfg
