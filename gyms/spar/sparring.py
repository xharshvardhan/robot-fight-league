"""Sparring gym: the one-network fighter (red) trains against the specialist team (blue), scored like a fight.

Built on the bout. What is new:
- Scorecard. A hit is red's fist, shin or foot arriving at speed on blue. Head 3 points, body 2, legs 1, scaled by force
  (400 N = full), kicks worth a quarter more. Red loses points for blue's punches landing, and for going down.
- Referee. Chest-to-chest for more than a second: both are put back in their corners and red pays. While one fighter is
  down, the other's strike order is switched off (a count).
- Footwork. Red is paid a little for holding punching range and facing blue, and pays for standing in a clinch.
Blue does not learn here.
"""
from __future__ import annotations

import dataclasses

import torch
from mjlab.envs import mdp
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply

from . import bout

ZONES = {"head": (3.0, ("head_collision",)),
         "body": (2.0, ("torso_collision", "pelvis_collision", ".*_hip_collision", ".*shoulder_yaw_collision")),
         "legs": (1.0, (".*_thigh_collision", ".*_shin_collision"))}
# bonus, striking bodies, m/s the limb must arrive at, newtons to count at all (legs brush each other while walking, so kicks need more)
LIMBS = {"punch": (1.0, ("left_wrist_yaw_link", "right_wrist_yaw_link"), 1.5, 30.0), "kick": (1.25, ("left_knee_link", "right_knee_link"), 3.5, 80.0)}
FULL = 400.0   # newtons for full marks
RANGE, CLINCH, CLINCH_T = 0.8, 0.45, 1.0


class SparStrike(bout.BoutStrike):
  """Red's order: straight punches aimed anywhere from blue's belly to its head."""
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.dz = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    self.dz[env_ids] = torch.rand(len(env_ids), device=self.device) * 0.32 - 0.10

  def _aim(self):
    super()._aim()
    self.target_w[:, 2] += self.dz


@dataclasses.dataclass(kw_only=True)
class SparStrikeCfg(bout.BoutStrikeCfg):
  def build(self, env):
    return SparStrike(self, env)


class Referee(bout.BoutThreat):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    z = lambda: torch.zeros(n, device=dev)
    self.zone = {(l, k): env.scene[f"{l}_{k}"] for l in LIMBS for k in ZONES}
    self.limb_ids = {l: self.robot.find_bodies(b, preserve_order=True)[0] for l, (_, b, _, _) in LIMBS.items()}
    self.prev_speed = {l: z() for l in LIMBS}
    self.points, self.conceded, self.cool, self.since_hit, self.clinch_t = z(), z(), z(), z() + 9, z()
    self.a_fell, self.b_fell, self.broke, self.dist, self.facing = z(), z(), z(), z() + bout.GAP, z()
    self.tot = {k: z() for k in ("red_points", "blue_points", "head", "body", "legs", "kicks", "red_down", "blue_down", "knockdowns", "breaks", "steps")}
    for k in self.tot:
      if k != "steps":
        self.metrics[k] = z()

  def reset(self, env_ids):
    for t in (*self.tot.values(), self.clinch_t, self.cool):
      t[env_ids] = 0.0
    return super().reset(env_ids)

  def _update_metrics(self):
    mins = (self.tot["steps"] * self._env.step_dt / 60.0).clamp(min=1e-3)
    for k, v in self.tot.items():
      if k in ("red_down", "blue_down"):
        self.metrics[k][:] = v / self.tot["steps"].clamp(min=1)   # share of the bout spent down
      elif k != "steps":
        self.metrics[k][:] = v / mins                              # per minute

  def _update_command(self):
    env, A, B, dt = self._env, self.robot, self.rival, self._env.step_dt
    a_was, b_was = self.a_down.clone(), self.b_down.clone()
    super()._update_command()
    self.b_log.clear()
    both_up = ~self.a_down & ~self.b_down
    # count: nobody is ordered to strike while a fighter is down
    self.b_pause = torch.where(~both_up, torch.full_like(self.b_pause, 0.3), self.b_pause)
    env.command_manager.get_term("strike").active[~both_up] = False

    # red's scorecard
    pts = torch.zeros_like(self.points); kick = torch.zeros_like(pts); by_zone = {}
    for l, (bonus, _, min_v, min_f) in LIMBS.items():
      d = A.data
      v = (d.body_link_lin_vel_w[:, self.limb_ids[l]] - B.data.root_link_lin_vel_w[:, None]).norm(dim=-1).amax(dim=-1)
      swing = self.prev_speed[l] > min_v
      self.prev_speed[l] = v
      for k, (worth, _) in ZONES.items():
        h = self.zone[(l, k)].data.force_history                      # [envs, shapes, substeps, 3]
        f = h.norm(dim=-1).amax(dim=-1).sum(dim=-1)
        p = worth * bonus * (f / FULL).clamp(max=1.0) * (f > min_f) * swing
        better = p > pts
        pts = torch.where(better, p, pts); kick = torch.where(better, torch.full_like(kick, float(l == "kick")), kick)
        for o in ZONES:  # remember which zone the best hit of this step landed on
          by_zone[o] = torch.where(better, torch.full_like(pts, float(o == k)), by_zone.get(o, torch.zeros_like(pts)))
    self.cool -= dt; self.since_hit += dt
    landed = (pts > 0) & (self.cool <= 0) & ~a_was
    self.points = pts * landed
    self.cool = torch.where(landed, torch.full_like(self.cool, 0.3), self.cool)
    self.since_hit = torch.where(landed, torch.zeros_like(self.since_hit), self.since_hit)
    self.conceded = self.b_hit * 2.0 * (0.25 + (self.b_push / FULL).clamp(max=1.0))

    # knockdowns
    self.a_fell = (self.a_down & ~a_was).float()
    self.b_fell = (self.b_down & ~b_was & (self.since_hit < 1.0)).float()

    # footwork and the referee's break
    off = B.data.root_link_pos_w - A.data.root_link_pos_w; off[:, 2] = 0.0
    self.dist = off.norm(dim=-1)
    fwd = quat_apply(A.data.root_link_quat_w, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.num_envs, 3))
    self.facing = ((fwd * off).sum(-1) / self.dist.clamp(min=1e-3)).clamp(min=0.0)
    self.clinch_t = torch.where((self.dist < CLINCH) & both_up, self.clinch_t + dt, torch.zeros_like(self.clinch_t))
    brk = self.clinch_t > CLINCH_T
    self.broke = brk.float()
    ids = brk.nonzero().flatten()
    if len(ids):
      mdp.reset_root_state_uniform(env, ids, pose_range={"yaw": (-0.3, 0.3)}, velocity_range={})
      mdp.reset_root_state_uniform(env, ids, pose_range={}, velocity_range={}, asset_cfg=SceneEntityCfg("rival"))
      self.clinch_t[ids] = 0.0

    t = self.tot
    t["steps"] += 1; t["red_points"] += self.points; t["blue_points"] += self.conceded; t["kicks"] += landed * kick
    for k in ZONES:
      t[k] += landed * by_zone[k]
    t["red_down"] += self.a_down; t["blue_down"] += self.b_down; t["knockdowns"] += self.b_fell; t["breaks"] += self.broke


@dataclasses.dataclass(kw_only=True)
class RefereeCfg(bout.BoutThreatCfg):
  def build(self, env):
    return Referee(self, env)


def _ref(env) -> Referee:
  return env.command_manager.get_term("threat")

def scored(env): return _ref(env).points / env.step_dt
def conceded(env): return _ref(env).conceded / env.step_dt
def knockdown(env): return _ref(env).b_fell / env.step_dt
def fell(env): return _ref(env).a_fell / env.step_dt
def down(env): return _ref(env).a_down.float()
def broken_up(env): return _ref(env).broke / env.step_dt
def clinching(env): r = _ref(env); return ((r.dist < CLINCH) & ~r.a_down & ~r.b_down).float()
def in_range(env): r = _ref(env); return torch.exp(-(r.dist - RANGE).abs() / 0.3) * r.facing * (~r.a_down).float()


def no_assist(env):  # the copied network's value estimate has an input for the get-up helper; there is none here
  return torch.zeros(env.num_envs, 1, device=env.device)


def env_cfg(play: bool = False):
  cfg = bout.env_cfg(play=play)
  cfg.observations["critic"].terms["assist"] = ObservationTermCfg(func=no_assist)
  touch = dict(fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4, secondary_policy="any")
  cfg.scene.sensors = (*cfg.scene.sensors, *(
    ContactSensorCfg(name=f"{l}_{k}", primary=ContactMatch(mode="geom", pattern=geoms, entity="rival"),
                     secondary=ContactMatch(mode="subtree", pattern=bodies, entity="robot"), **touch)
    for l, (_, bodies, _, _) in LIMBS.items() for k, (_, geoms) in ZONES.items()))
  cfg.commands["strike"] = SparStrikeCfg(resampling_time_range=(1.5, 1.5), debug_vis=True, log=False, start_level=1.0)
  cfg.commands["threat"] = RefereeCfg(resampling_time_range=(1e6, 1e6), start_level=1.0)
  for k in ("hit", "waiting", "missed", "foul", "took_hit", "defended", "ran_off", "alive", "fell_fighting"):
    cfg.rewards.pop(k)
  cfg.rewards["reach"].weight = 0.5
  cfg.rewards.update({
    "scored": RewardTermCfg(func=scored, weight=4.0),         # a full-force head punch is 12, a body punch 8, a leg kick 5
    "conceded": RewardTermCfg(func=conceded, weight=-2.0),    # a full-force punch taken costs 5
    "knockdown": RewardTermCfg(func=knockdown, weight=10.0),
    "fell": RewardTermCfg(func=fell, weight=-15.0),
    "down": RewardTermCfg(func=down, weight=-2.0),
    "broken_up": RewardTermCfg(func=broken_up, weight=-3.0),
    "clinching": RewardTermCfg(func=clinching, weight=-2.0),
    "in_range": RewardTermCfg(func=in_range, weight=1.0),
  })
  cfg.episode_length_s = int(1e9) if play else 20.0
  cfg.sim.nconmax, cfg.sim.njmax = 300, 1800
  return cfg
