"""First bout (no training): the one-network fighter against a team of the three specialists.

Red corner ("robot"): the single copied network. A rule picks its skill every step: get up when down, defend when a
fist is coming, otherwise strike. Its strike target is the rival's chest and its "incoming ball" is the rival's nearest fist.
Blue corner ("rival"): the strike, defence and get-up specialists, switched by the same rule.
Nobody is reset when they fall: they have to get up.
"""
from __future__ import annotations

import dataclasses
import os
import re

import torch
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from src.tasks.defence import defence
from src.tasks.fighter import fighter
from src.tasks.fighter.runners import Teacher
from src.tasks.strike import strike_v2

GAP = 1.2
WALK, STRIKE, DEFEND, GETUP = range(4)
FISTS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def fists_of(entity, ids):
  d = entity.data
  r = quat_apply(d.body_link_quat_w[:, ids], torch.tensor(strike_v2.GLOVE, device=d.root_link_pos_w.device).expand(len(d.root_link_pos_w), 2, 3))
  return d.body_link_pos_w[:, ids] + r, d.body_link_lin_vel_w[:, ids] + torch.cross(d.body_link_ang_vel_w[:, ids], r, dim=-1)


def chest_aim(me, other, chest_id):
  """Point on the other's chest and the horizontal direction toward it."""
  chest = other.data.body_link_pos_w[:, chest_id] + torch.tensor([0.0, 0.0, 0.2], device=me.data.root_link_pos_w.device)
  d = chest - me.data.root_link_pos_w; d[:, 2] = 0.0
  d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
  return chest - 0.05 * d, d


def threat_from(me, other, other_fist_ids):
  """The other's fist that is closing fastest, in my frame: position, velocity, and whether it counts as incoming."""
  pos, vel = fists_of(other, other_fist_ids)
  rel = pos - me.data.root_link_pos_w[:, None]; rel[..., 2] -= 0.25  # measured from my chest height
  dist = rel.norm(dim=-1)
  closing = -(rel * (vel - me.data.root_link_lin_vel_w[:, None])).sum(-1) / dist.clamp(min=1e-3)
  i = closing.argmax(dim=-1); idx = torch.arange(len(i), device=i.device)
  live = (closing[idx, i] > 1.5) & (dist[idx, i] < 0.9)
  q = me.data.root_link_quat_w
  return torch.cat([quat_apply_inverse(q, pos[idx, i] - me.data.root_link_pos_w), quat_apply_inverse(q, vel[idx, i]), live.float()[:, None]], -1), live


def is_down(entity, env, was):
  z = entity.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  g = entity.data.projected_gravity_b[:, 2]
  return torch.where(was, ~((z > 0.66) & (g < -0.93)), (z < 0.45) | (g > -0.5))  # down until fully standing (0.55 m handed over mid-crouch and the fighter toppled again)


class BoutStrike(strike_v2.StrikeCommand):
  """Red's strike order: always the rival's chest, straight punches, alternating fists at random."""
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.rival = env.scene["rival"]
    self.chest = self.rival.find_bodies("torso_link")[0][0]

  def _aim(self):
    self.target_w, self.dir_w = chest_aim(self.robot, self.rival, self.chest)

  def _place_pad(self, env_ids):
    pose = torch.zeros(len(env_ids), 7, device=self.device)
    pose[:, :3] = self._env.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, -1.0], device=self.device); pose[:, 3] = 1.0
    self.pad.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def _resample_command(self, env_ids):
    self.limb[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)
    self.kind[env_ids] = 0; self.pad_vel[env_ids] = 0.0
    self.window[env_ids] = 1.5; self.time_left[env_ids] = 1.5
    self.active[env_ids] = True; self.age[env_ids] = 0.0; self.targets[env_ids] += 1

  def _update_command(self):
    self._aim()
    super()._update_command()


@dataclasses.dataclass(kw_only=True)
class BoutStrikeCfg(strike_v2.StrikeCommandCfg):
  def build(self, env):
    return BoutStrike(self, env)


class BoutThreat(defence.Threat):
  """Red's threat input, the rule that picks red's skill, and the whole of blue (inputs, rule, networks, punches landed)."""
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    self.rival = env.scene["rival"]
    self.r_fists = self.rival.find_bodies(FISTS, preserve_order=True)[0]
    self.a_chest = self.robot.find_bodies("torso_link")[0][0]
    d = os.path.expanduser("~/teachers")
    self.brains = [Teacher(f"{d}/{k}.pt", dev) for k in ("strike", "defend", "getup")]
    self.b_act = torch.zeros(n, 29, device=dev)
    self.b_limb = torch.zeros(n, dtype=torch.long, device=dev)
    self.b_left = torch.full((n,), 1.5, device=dev); self.b_pause = torch.zeros(n, device=dev)
    self.b_prev = torch.zeros(n, 3, device=dev)
    self.hands = [env.scene["rival_left"], env.scene["rival_right"]]
    self.a_down = torch.zeros(n, dtype=torch.bool, device=dev); self.b_down = torch.zeros(n, dtype=torch.bool, device=dev)
    self.b_mode = torch.full((n,), STRIKE, dtype=torch.long, device=dev)
    self.a_threat = torch.zeros(n, 7, device=dev)
    self.b_log = []
    self.b_ids = None

  @property
  def command(self):
    return self.a_threat

  def reset(self, env_ids):
    self.b_act[env_ids] = 0.0; self.a_down[env_ids] = False; self.b_down[env_ids] = False
    return super().reset(env_ids)

  def _resample_command(self, env_ids):  # no ball in a bout
    self._park(env_ids)

  def _update_command(self):
    env, A, B, dt = self._env, self.robot, self.rival, self._env.step_dt
    n, dev = self.num_envs, self.device
    idx = torch.arange(n, device=dev)
    # red: threat input and skill choice
    self.a_threat, a_live = threat_from(A, B, self.r_fists)
    self.a_down = is_down(A, env, self.a_down)
    if os.environ.get("BOUT_NO_DEFEND"):  # diagnosis: red never switches to defending
      a_live = torch.zeros_like(a_live)
    fighter.modes(env)[:] = torch.where(self.a_down, GETUP, torch.where(a_live, DEFEND, STRIKE))
    env.sim.data.xfrc_applied[:, :, :] = 0.0
    # blue: inputs
    b_threat, b_live = threat_from(B, A, self.fist_ids)
    self.b_down = is_down(B, env, self.b_down)
    self.b_mode = torch.where(self.b_down, GETUP, torch.where(b_live, DEFEND, STRIKE))
    target, direction = chest_aim(B, A, self.a_chest)
    self.b_left -= dt; self.b_pause -= dt
    new = self.b_left <= 0
    self.b_limb = torch.where(new, torch.randint(0, 2, (n,), device=dev), self.b_limb)
    self.b_left = torch.where(new, torch.full_like(self.b_left, 1.5), self.b_left)
    active = (self.b_pause <= 0).float()[:, None]
    d, q = B.data, B.data.root_link_quat_w
    order = torch.cat([quat_apply_inverse(q, target - d.root_link_pos_w), quat_apply_inverse(q, direction),
                       torch.nn.functional.one_hot(self.b_limb, 2).float(), active, (self.b_left / 1.5)[:, None]], -1) * active
    common = torch.cat([mdp.builtin_sensor(env, "rival/imu_ang_vel"), d.projected_gravity_b, d.joint_pos - d.default_joint_pos, d.joint_vel, self.b_act], -1)
    acts = torch.stack([self.brains[0](torch.cat([order, common], -1)), self.brains[1](torch.cat([b_threat, common], -1)), self.brains[2](common)], 1)
    self.b_act = acts[idx, self.b_mode - 1]
    if self.b_ids is None:  # blue's joints in the same order as red's action, with the specialists' own scale
      names = env.action_manager.get_term("joint_pos").target_names
      self.b_ids = torch.tensor(B.find_joints(names, preserve_order=True)[0], device=dev)
      self.b_scale = torch.tensor([next(v for k, v in G1_ACTION_SCALE.items() if re.fullmatch(k, j)) for j in names], device=dev)
    B.set_joint_position_target(d.default_joint_pos[:, self.b_ids] + self.b_act * self.b_scale, joint_ids=self.b_ids)
    # blue's punches landed on red
    pos, vel = fists_of(B, self.r_fists)
    h = self.hands[0].data.force_history[:, 0], self.hands[1].data.force_history[:, 0]
    f = torch.where((self.b_limb == 0)[:, None, None], h[0], h[1])
    f = f[idx, f.norm(dim=-1).argmax(-1)]
    push = (f * direction).sum(-1).abs()
    hit = (self.b_mode == STRIKE) & (self.b_pause <= 0) & ((self.b_prev * direction).sum(-1) > 1.0) & (push > 20.0)
    for i in hit.nonzero().flatten().tolist():
      self.b_log.append(round(float(push[i]), 1))
    self.b_hit, self.b_push = hit, push
    self.b_pause = torch.where(hit, torch.rand(n, device=dev) * 0.4 + 0.4, self.b_pause)
    self.b_prev = vel[idx, self.b_limb]


@dataclasses.dataclass(kw_only=True)
class BoutThreatCfg(defence.ThreatCfg):
  def build(self, env):
    return BoutThreat(self, env)


def env_cfg(play: bool = False):
  cfg = fighter.env_cfg(play=True)
  rival = get_g1_robot_cfg()
  rival.init_state = dataclasses.replace(rival.init_state, pos=(GAP, 0.0, rival.init_state.pos[2]), rot=(0.0, 0.0, 0.0, 1.0))
  cfg.scene.entities["rival"] = rival
  a_body = ContactMatch(mode="subtree", pattern="pelvis", entity="robot")
  b_body = ContactMatch(mode="subtree", pattern="pelvis", entity="rival")
  touch = dict(fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4)
  wrist = lambda s, e: ContactMatch(mode="subtree", pattern=f"{s}_wrist_yaw_link", entity=e)
  cfg.scene.sensors = (*(s for s in cfg.scene.sensors if s.name not in ("left_pad", "right_pad", "any_pad")),
    *(ContactSensorCfg(name=f"{s}_pad", primary=wrist(s, "robot"), secondary=b_body, **touch) for s in ("left", "right")),
    ContactSensorCfg(name="any_pad", primary=b_body, secondary=a_body, **touch),
    *(ContactSensorCfg(name=f"rival_{s}", primary=wrist(s, "rival"), secondary=a_body, **touch) for s in ("left", "right")))
  cfg.commands["strike"] = BoutStrikeCfg(resampling_time_range=(1.5, 1.5), debug_vis=True, log=True, start_level=1.0)
  cfg.commands["threat"] = BoutThreatCfg(resampling_time_range=(1e6, 1e6), start_level=1.0)
  cfg.commands.pop("assist")
  cfg.commands["body"].capacity_j = 1e12  # red's stamina reserve never drains: blue has none, and an empty one kept red on the floor
  for group in ("actor", "critic"):
    cfg.observations[group].terms["strike"].func = fighter.only(("strike",), mdp.generated_commands)
  cfg.observations["critic"].terms.pop("assist", None)
  cfg.events["reset_base"] = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", params={"pose_range": {"yaw": (-0.3, 0.3)}, "velocity_range": {}})
  cfg.events["reset_rival"] = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                                           params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("rival")})
  cfg.events["reset_rival_joints"] = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
    params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("rival", joint_names=(".*",))})
  cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
  cfg.episode_length_s = int(1e9)
  cfg.sim.nconmax, cfg.sim.njmax = 300, 1800
  cfg.viewer.distance, cfg.viewer.azimuth = 3.6, 90.0
  return cfg
