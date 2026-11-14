"""Strike gym: a G1 is shown a target point and an order (which fist, which direction, how soon) and is scored on the hit.

No reference motion. Version 0 uses a virtual target: a hit is the ordered fist passing through the target sphere while
moving in the ordered direction, and its strength is the fist's speed along that direction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, yaw_quat
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.tasks.tracking.mdp.rewards import self_collision_cost

GLOVE = (0.1, 0.0, 0.0)                    # fist centre in the wrist link frame
GUARD = ((0.22, 0.10, 0.20), (0.22, -0.10, 0.20))  # where each fist rests, relative to the pelvis (left, right)
KINDS = ("forward", "up", "inward")       # direction the blow must arrive in: jab/cross, uppercut, hook


class StrikeCommand(CommandTerm):
  cfg: StrikeCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    ids = self.robot.find_bodies(("left_wrist_yaw_link", "right_wrist_yaw_link"), preserve_order=True)[0]
    self.fist_ids = torch.tensor(ids, dtype=torch.long, device=self.device)
    n, dev = self.num_envs, self.device
    self.glove = torch.tensor(GLOVE, device=dev).expand(n, 2, 3)
    self.guard = torch.tensor(GUARD, device=dev).expand(n, 2, 3)
    self.idx = torch.arange(n, device=dev)
    self.target_w = torch.zeros(n, 3, device=dev)
    self.dir_w = torch.zeros(n, 3, device=dev); self.dir_w[:, 0] = 1
    self.limb = torch.zeros(n, dtype=torch.long, device=dev)   # 0 = left fist, 1 = right fist
    self.kind = torch.zeros(n, dtype=torch.long, device=dev)
    self.active = torch.zeros(n, dtype=torch.bool, device=dev)  # target still waiting to be hit
    self.dist = torch.ones(n, device=dev)
    self.hit_bonus = torch.zeros(n, device=dev)
    self.targets, self.hits, self.speed_sum = (torch.zeros(n, device=dev) for _ in range(3))
    self.metrics["hit_fraction"] = torch.zeros(n, device=dev)
    self.metrics["hit_speed"] = torch.zeros(n, device=dev)
    self.strike_log = []  # filled only when cfg.log is on (playback): one dict per landed hit

  def fists(self):
    """World position and velocity of both fist centres, shape [envs, 2, 3]."""
    d = self.robot.data
    r = quat_apply(d.body_link_quat_w[:, self.fist_ids], self.glove)
    vel = d.body_link_lin_vel_w[:, self.fist_ids] + torch.cross(d.body_link_ang_vel_w[:, self.fist_ids], r, dim=-1)
    return d.body_link_pos_w[:, self.fist_ids] + r, vel

  @property
  def command(self):
    d = self.robot.data
    q = d.root_link_quat_w
    return torch.cat([quat_apply_inverse(q, self.target_w - d.root_link_pos_w), quat_apply_inverse(q, self.dir_w),
                      torch.nn.functional.one_hot(self.limb, 2).float(), self.active.float()[:, None],
                      (self.time_left / self.cfg.resampling_time_range[1])[:, None]], dim=-1)

  def reset(self, env_ids):
    for t in (self.targets, self.hits, self.speed_sum, self.hit_bonus):
      t[env_ids] = 0.0
    return super().reset(env_ids)

  def _update_metrics(self):
    self.metrics["hit_fraction"][:] = self.hits / self.targets.clamp(min=1)
    self.metrics["hit_speed"][:] = self.speed_sum / self.hits.clamp(min=1)

  def _resample_command(self, env_ids):
    k, dev, c = len(env_ids), self.device, self.cfg
    u = lambda lo, hi: torch.rand(k, device=dev) * (hi - lo) + lo
    limb = torch.randint(0, 2, (k,), device=dev)
    side = 1.0 - 2.0 * limb  # +1 for the left fist, -1 for the right
    kind = torch.randint(0, len(KINDS), (k,), device=dev)
    local = torch.stack([u(*c.reach_x), side * u(*c.reach_y), u(*c.reach_z)], dim=-1)
    dirs = torch.zeros(k, 3, device=dev)
    dirs[kind == 0, 0] = 1.0
    dirs[kind == 1, 2] = 1.0
    dirs[kind == 2, 1] = -side[kind == 2]
    yaw = yaw_quat(self.robot.data.root_link_quat_w[env_ids])
    self.target_w[env_ids] = self.robot.data.root_link_pos_w[env_ids] + quat_apply(yaw, local)
    self.dir_w[env_ids] = quat_apply(yaw, dirs)
    self.limb[env_ids], self.kind[env_ids], self.active[env_ids] = limb, kind, True
    self.targets[env_ids] += 1

  def _update_command(self):
    pos, vel = self.fists()
    f, v = pos[self.idx, self.limb], vel[self.idx, self.limb]
    self.dist = torch.norm(f - self.target_w, dim=-1)
    speed = (v * self.dir_w).sum(-1)
    hit = self.active & (self.dist < self.cfg.radius) & (speed > self.cfg.min_speed)
    self.hit_bonus = torch.where(hit, 2.0 + 8.0 * (speed / self.cfg.full_speed).clamp(max=1.0), torch.zeros_like(speed))
    self.active &= ~hit
    self.hits += hit; self.speed_sum += speed * hit
    if self.cfg.log:
      for i in hit.nonzero().flatten().tolist():
        self.strike_log.append(dict(env=i, step=int(self._env.common_step_counter), fist=("left", "right")[int(self.limb[i])],
                                    order=KINDS[int(self.kind[i])], speed_along_order=round(float(speed[i]), 2),
                                    fist_velocity=[round(x, 2) for x in v[i].tolist()], target=[round(x, 3) for x in self.target_w[i].tolist()],
                                    reaction_s=round(float(self.cfg.resampling_time_range[1] - self.time_left[i]), 2)))

  def _debug_vis_impl(self, visualizer):
    for b in visualizer.get_env_indices(self.num_envs):
      t = self.target_w[b].cpu().numpy(); live = bool(self.active[b])
      visualizer.add_sphere(t, self.cfg.radius, (1.0, 0.25, 0.2, 0.9) if live else (0.4, 0.4, 0.4, 0.3), label=f"target_{b}")
      if live:
        visualizer.add_arrow(t - 0.25 * self.dir_w[b].cpu().numpy(), t, (1.0, 0.85, 0.2, 0.9), label=f"order_{b}")


@dataclass(kw_only=True)
class StrikeCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  radius: float = 0.08       # metres: the fist centre must come this close
  min_speed: float = 0.5     # m/s along the ordered direction for the hit to count
  full_speed: float = 6.0    # m/s that earns the full bonus
  reach_x: tuple = (0.25, 0.45)   # target box in front of the pelvis (metres)
  reach_y: tuple = (-0.05, 0.25)  # sideways, toward the striking fist's own side
  reach_z: tuple = (0.05, 0.45)   # above the pelvis
  log: bool = False

  def build(self, env):
    return StrikeCommand(self, env)


def _cmd(env) -> StrikeCommand:
  return env.command_manager.get_term("strike")

def reach(env):  # pull the ordered fist toward a waiting target
  c = _cmd(env); return c.active * torch.exp(-c.dist / 0.3)

def hit(env):  # one-off bonus when the hit lands, larger for a faster fist
  return _cmd(env).hit_bonus / env.step_dt

def guard(env):  # fists at the guard points; the ordered fist is excused while its target is waiting
  c = _cmd(env); d = c.robot.data
  pos, _ = c.fists()
  q = d.root_link_quat_w[:, None].expand(-1, 2, -1)
  err = torch.norm(quat_apply_inverse(q, pos - d.root_link_pos_w[:, None]) - c.guard, dim=-1)
  w = torch.ones_like(err); w[c.idx, c.limb] = (~c.active).float()
  return (w * torch.exp(-err / 0.15)).sum(-1) / 2

def stay(env):  # don't wander off the spot (targets are all within reach in this version)
  return torch.sum(torch.square(env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]), dim=-1)


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor = {
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "strike"}),
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=Unoise(n_min=-0.2, n_max=0.2)),
    "gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic = {k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor.items()}
  critic["base_lin_vel"] = ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"})
  robot = SceneEntityCfg("robot", joint_names=(".*",))
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1, entities={"robot": get_g1_robot_cfg()},
                   sensors=(ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                             secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                             fields=("found", "force"), reduce="none", num_slots=1, history_length=4),)),
    observations={"actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=not play),
                  "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False)},
    actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True)},
    commands={"strike": StrikeCommandCfg(resampling_time_range=(2.0, 2.0), debug_vis=True, log=play)},
    events={
      "reset_base": EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                                 params={"pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)}, "velocity_range": {}}),
      "reset_joints": EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
                                   params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0), "asset_cfg": robot}),
      "foot_friction": EventTermCfg(mode="startup", func=dr.geom_friction,
                                    params={"asset_cfg": SceneEntityCfg("robot", geom_names=r"^(left|right)_foot[1-7]_collision$"),
                                            "operation": "abs", "ranges": (0.3, 1.2), "shared_random": True}),
    },
    rewards={
      "reach": RewardTermCfg(func=reach, weight=1.0),
      "hit": RewardTermCfg(func=hit, weight=1.0),
      "guard": RewardTermCfg(func=guard, weight=1.0),
      "stay": RewardTermCfg(func=stay, weight=-2.0),
      "upright": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
      "fell": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
      "joint_limit": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": robot}),
      "self_collisions": RewardTermCfg(func=self_collision_cost, weight=-10.0, params={"sensor_name": "self_collision", "force_threshold": 10.0}),
    },
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
      "too_low": TerminationTermCfg(func=mdp.root_height_below_minimum, params={"minimum_height": 0.45}),
    },
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link",
                        distance=2.6, fovy=55.0, elevation=-8.0, azimuth=120.0),
    sim=SimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
    decimation=4,
    episode_length_s=10.0,
  )
  if not play:
    cfg.events["push_robot"] = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(3.0, 5.0),
                                            params={"velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4), "yaw": (-0.5, 0.5)}})
  else:
    cfg.episode_length_s = int(1e9)
  return cfg


def rl_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
                                   num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                   gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name="g1_strike", save_interval=100, num_steps_per_env=24, max_iterations=3000)
