"""Two-robot contact test (measurement only, no training).

The strike network from the pad gym faces a second G1 driven by the get-up network, which stands on its own and gets
back up when knocked down (a G1 that only holds its pose falls by itself in about 1.4 s). The "pad" is now a point on the
dummy's chest, and a hit is contact between the ordered fist and the dummy's body. Measures punch force on a real body,
whether the dummy is knocked down, and how fast two robots simulate.
"""
from __future__ import annotations

import dataclasses
import os

import torch
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.fighter.runners import Teacher
from src.tasks.strike import strike_v2

GAP = 0.9  # metres between the two robots at the start


class SparCommand(strike_v2.StrikeCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.dummy = env.scene["dummy"]
    self.chest = self.dummy.find_bodies("torso_link")[0][0]
    self.brain = Teacher(os.path.expanduser("~/teachers/getup.pt"), self.device)
    self.d_act = torch.zeros(self.num_envs, 29, device=self.device)  # the dummy's last action

  def reset(self, env_ids):
    self.d_act[env_ids] = 0.0
    return super().reset(env_ids)

  def _drive_dummy(self):
    env, d = self._env, self.dummy.data
    term = env.action_manager.get_term("joint_pos")  # same robot model, so the attacker's scale and joint order apply
    obs = torch.cat([mdp.builtin_sensor(env, "dummy/imu_ang_vel"), d.projected_gravity_b, d.joint_pos - d.default_joint_pos,
                     d.joint_vel, self.d_act], dim=-1)
    self.d_act = self.brain(obs)
    self.dummy.set_joint_position_target(d.default_joint_pos[:, term.target_ids] + self.d_act * term.scale, joint_ids=term.target_ids)

  def _aim(self, env_ids):
    chest = self.dummy.data.body_link_pos_w[env_ids, self.chest] + torch.tensor([0.0, 0.0, 0.2], device=self.device)
    d = chest - self.robot.data.root_link_pos_w[env_ids]; d[:, 2] = 0.0
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    self.dir_w[env_ids] = d
    self.target_w[env_ids] = chest - 0.05 * d

  def _place_pad(self, env_ids):  # the pad itself stays underground
    pose = torch.zeros(len(env_ids), 7, device=self.device)
    pose[:, :3] = self._env.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, -1.0], device=self.device)
    pose[:, 3] = 1.0
    self.pad.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def _resample_command(self, env_ids):
    self.missed[env_ids] = self.active[env_ids].float()
    self.limb[env_ids] = torch.randint(0, 2, (len(env_ids),), device=self.device)
    self.kind[env_ids] = 0
    self.pad_vel[env_ids] = 0.0
    self.window[env_ids] = 1.5; self.time_left[env_ids] = 1.5
    self._aim(env_ids)
    self.active[env_ids] = True; self.age[env_ids] = 0.0; self.targets[env_ids] += 1

  def _update_command(self):
    self._drive_dummy()
    self._aim(torch.arange(self.num_envs, device=self.device))
    super()._update_command()


@dataclasses.dataclass(kw_only=True)
class SparCommandCfg(strike_v2.StrikeCommandCfg):
  def build(self, env):
    return SparCommand(self, env)


def dummy_down(env):
  d = env.scene["dummy"].data
  return (d.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] < 0.45) | (d.projected_gravity_b[:, 2] > -0.34)


def env_cfg(play: bool = False, punch: bool = True):
  cfg = strike_v2.env_cfg(play=True)
  dummy = get_g1_robot_cfg()
  dummy.init_state = dataclasses.replace(dummy.init_state, pos=(GAP, 0.0, dummy.init_state.pos[2]), rot=(0.0, 0.0, 0.0, 1.0))
  cfg.scene.entities["dummy"] = dummy
  body = ContactMatch(mode="subtree", pattern="pelvis", entity="dummy")
  touch = dict(fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4)
  keep = tuple(s for s in cfg.scene.sensors if s.name == "self_collision")
  cfg.scene.sensors = (*keep,
    *(ContactSensorCfg(name=f"{s}_pad", primary=ContactMatch(mode="subtree", pattern=f"{s}_wrist_yaw_link", entity="robot"), secondary=body, **touch)
      for s in ("left", "right")),
    ContactSensorCfg(name="any_pad", primary=body, secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"), **touch))
  cfg.commands["strike"] = SparCommandCfg(resampling_time_range=(1.5, 1.5), debug_vis=True, log=True, start_level=1.0)
  if not punch:  # baseline: how long the dummy stands when nobody touches it
    cfg.commands["strike"].min_force = 1e9
    cfg.scene.entities["dummy"].init_state = dataclasses.replace(dummy.init_state, pos=(3.0, 0.0, dummy.init_state.pos[2]))
  cfg.events["reset_base"].params["pose_range"] = {"yaw": (-0.3, 0.3)}
  cfg.events["reset_dummy"] = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                                           params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("dummy")})
  cfg.events["reset_dummy_joints"] = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
    params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("dummy", joint_names=(".*",))})
  cfg.sim.nconmax, cfg.sim.njmax = 250, 1500  # both robots overlap for an instant before the first reset
  cfg.viewer.distance, cfg.viewer.azimuth = 3.2, 90.0
  return cfg
