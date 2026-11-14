"""Get-up gym v0: a G1 starts lying on the floor (face down or face up, any heading) and is scored on standing up and staying up.

No reference motion. One episode in five starts standing, so it also learns that standing is the place to stay.
v0 (reward only) ended sitting upright with legs splayed. v1 adds a helper: an upward force on the torso, starting at 70% of
body weight, that fades as more robots are standing and returns if they stop; the upright reward now only counts with height,
and feet under the hips are rewarded.
"""
from __future__ import annotations

import math

import torch
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from dataclasses import dataclass

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
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.tasks.tracking.mdp.rewards import self_collision_cost

STAND_Z = 0.72  # pelvis height (m) that counts as standing
LIE_Z = 0.25    # pelvis height when lying flat


def reset_fallen(env, env_ids, standing: float = 0.2):
  """Drops most robots from 45 cm lying face down or face up; the rest start standing."""
  r = torch.rand(len(env_ids), device=env.device)
  lying = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.35, -0.30), "roll": (-3.14, 3.14), "yaw": (-3.14, 3.14)}
  for ids, pose in ((env_ids[r < standing], {"yaw": (-3.14, 3.14)}),
                    (env_ids[(r >= standing) & (r < 0.5 + standing / 2)], {**lying, "pitch": (1.27, 1.87)}),
                    (env_ids[r >= 0.5 + standing / 2], {**lying, "pitch": (-1.87, -1.27)})):
    if len(ids):
      mdp.reset_root_state_uniform(env, ids, pose_range=pose, velocity_range={})


def _z(env):
  return env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]

def _upright(env):  # 1 when the pelvis is vertical, 0 when it is upside down
  return (1 - env.scene["robot"].data.projected_gravity_b[:, 2]) / 2

def height(env):  # how far the pelvis has risen from lying to standing
  return ((_z(env) - LIE_Z) / (STAND_Z - LIE_Z)).clamp(0, 1)

def upright(env):
  return _upright(env) ** 2

def standing(env):  # the goal state: up, vertical, and in the normal standing pose
  d = env.scene["robot"].data
  up = (_z(env) > STAND_Z - 0.07) & (_upright(env) > 0.95)
  pose = torch.exp(-torch.sum((d.joint_pos - d.default_joint_pos) ** 2, dim=-1) / 2.0)
  still = torch.exp(-torch.sum(d.root_link_lin_vel_w ** 2, dim=-1))
  return up.float() * (1 + pose + still)

def upright_tall(env):  # v1: being vertical only counts as far as the pelvis has risen
  return upright(env) * height(env)

def feet_under(env, ids=[]):  # v1: both feet near the point under the pelvis
  robot = env.scene["robot"]
  if not ids:
    ids.append(robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)[0])
  d = robot.data
  off = d.body_link_pos_w[:, ids[0], :2] - d.root_link_pos_w[:, None, :2]
  return torch.exp(-torch.norm(off, dim=-1) / 0.3).mean(-1)


class Assist(CommandTerm):
  """Helper lift on the torso, as a fraction of body weight. One shared level, lowered while most robots are standing."""
  cfg: AssistCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    robot = env.scene["robot"]
    gid = robot.indexing.body_ids
    self.torso = int(gid[robot.find_bodies("torso_link")[0][0]])
    self.weight = float(env.sim.mj_model.body_mass[gid.cpu().numpy()].sum()) * 9.81
    self.level = cfg.start
    for k in ("level", "standing"):
      self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    return torch.full((self.num_envs, 1), self.level, device=self.device)

  def _update_metrics(self):
    self.metrics["level"][:] = self.level

  def _resample_command(self, env_ids):
    pass

  def _update_command(self):
    env = self._env
    up = ((_z(env) > STAND_Z - 0.07) & (_upright(env) > 0.95)).float()
    self.metrics["standing"][:] = up
    frac = float(up.mean())
    self.level = min(self.cfg.start, max(0.0, self.level + (-2e-4 if frac > 0.6 else 2e-4 if frac < 0.4 else 0.0)))
    env.sim.data.xfrc_applied[:, self.torso, 2] = self.level * self.weight


@dataclass(kw_only=True)
class AssistCfg(CommandTermCfg):
  start: float = 0.7

  def build(self, env):
    return Assist(self, env)


def joint_speed(env):
  return torch.sum(env.scene["robot"].data.joint_vel ** 2, dim=-1)

def root_height_obs(env):
  return _z(env)[:, None]


def env_cfg(play: bool = False, v1: bool = False) -> ManagerBasedRlEnvCfg:
  actor = {
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=Unoise(n_min=-0.2, n_max=0.2)),
    "gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic = {k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor.items()}
  critic["base_lin_vel"] = ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"})
  critic["root_height"] = ObservationTermCfg(func=root_height_obs)
  robot = SceneEntityCfg("robot", joint_names=(".*",))
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1, entities={"robot": get_g1_robot_cfg()},
                   sensors=(ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                             secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                             fields=("found", "force"), reduce="none", num_slots=1, history_length=4),)),
    observations={"actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=not play),
                  "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False)},
    actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True)},
    events={
      "reset_joints": EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
                                   params={"position_range": (-0.2, 0.2), "velocity_range": (0.0, 0.0), "asset_cfg": robot}),
      "reset_base": EventTermCfg(func=reset_fallen, mode="reset", params={"standing": 0.0 if play else 0.2}),
      "foot_friction": EventTermCfg(mode="startup", func=dr.geom_friction,
                                    params={"asset_cfg": SceneEntityCfg("robot", geom_names=r"^(left|right)_foot[1-7]_collision$"),
                                            "operation": "abs", "ranges": (0.5, 1.2), "shared_random": True}),
    },
    rewards={
      "height": RewardTermCfg(func=height, weight=2.0),
      "upright": RewardTermCfg(func=upright, weight=1.0),
      "standing": RewardTermCfg(func=standing, weight=2.0),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
      "joint_speed": RewardTermCfg(func=joint_speed, weight=-2e-4),
      "joint_limit": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0, params={"asset_cfg": robot}),
      "self_collisions": RewardTermCfg(func=self_collision_cost, weight=-1.0, params={"sensor_name": "self_collision", "force_threshold": 10.0}),
    },
    terminations={"time_out": TerminationTermCfg(func=mdp.time_out, time_out=True)},
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link",
                        distance=3.0, fovy=55.0, elevation=-12.0, azimuth=120.0),
    sim=SimulationCfg(nconmax=80, njmax=600, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
    decimation=4,
    episode_length_s=8.0,
  )
  if v1:
    cfg.commands = {"assist": AssistCfg(resampling_time_range=(1e6, 1e6), start=0.0 if play else 0.7)}
    cfg.observations["critic"].terms["assist"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "assist"})
    cfg.rewards["upright"] = RewardTermCfg(func=upright_tall, weight=1.0)
    cfg.rewards["height"].weight = 3.0
    cfg.rewards["feet_under"] = RewardTermCfg(func=feet_under, weight=1.0)
  return cfg


def rl_cfg(v1: bool = False) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
                                   num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                   gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name="g1_getup_v1" if v1 else "g1_getup_v0", save_interval=100, num_steps_per_env=24, max_iterations=4000 if v1 else 2500)
