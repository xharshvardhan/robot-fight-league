"""Defence gym v0: glove-sized balls are thrown at a standing G1's head and body; it is scored on not being hit there.

The mirror of the strike gym. The network is told where the ball is and how fast it is moving, relative to its own body
(numbers, not pictures). Dodging, ducking, stepping aside and blocking with an arm all count as a clean defence; a ball
that touches the torso, pelvis or hips is a hit taken. Balls get faster and come from wider angles as the robots improve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import torch
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.entity import EntityCfg
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

BALL_R, BALL_KG = 0.06, 0.5
GUARD = ((0.18, 0.10, 0.20), (0.18, -0.10, 0.20))  # where each fist rests, relative to the pelvis (left, right)
GLOVE = (0.1, 0.0, 0.0)


def ball_spec():
  s = mujoco.MjSpec()
  b = s.worldbody.add_body(name="ball", pos=[0, 0, 30], gravcomp=1)  # weightless, so it flies straight
  b.add_freejoint()
  b.add_geom(name="ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0], mass=BALL_KG, rgba=[0.9, 0.2, 0.15, 1])
  return s


class Threat(CommandTerm):
  cfg: ThreatCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot, self.ball = env.scene["robot"], env.scene["ball"]
    self.body_hit, self.arm_hit = env.scene["ball_body"], env.scene["ball_arm"]
    n, dev = self.num_envs, self.device
    self.fist_ids = self.robot.find_bodies(("left_wrist_yaw_link", "right_wrist_yaw_link"), preserve_order=True)[0]
    self.active = torch.zeros(n, dtype=torch.bool, device=dev)  # a ball is in flight
    self.age = torch.zeros(n, device=dev)
    self.flight = torch.ones(n, device=dev)                     # seconds this ball needs to arrive
    self.took = torch.zeros(n, dtype=torch.bool, device=dev)    # this ball has already touched head or body
    self.armed = torch.zeros(n, dtype=torch.bool, device=dev)   # this ball has touched an arm
    self.hit_now = torch.zeros(n, device=dev)                   # 1 on the step a ball first lands on head or body
    self.clean_now = torch.zeros(n, device=dev)                 # 1 on the step a ball is counted as defended
    self.level = cfg.start_level
    self.rate = 0.5
    self.throws, self.clean, self.blocks = (torch.zeros(n, device=dev) for _ in range(3))
    for k in ("clean_fraction", "block_fraction", "level"):
      self.metrics[k] = torch.zeros(n, device=dev)

  @property
  def command(self):
    d, b = self.robot.data, self.ball.data
    q = d.root_link_quat_w
    live = self.active.float()[:, None]
    return torch.cat([quat_apply_inverse(q, b.root_link_pos_w - d.root_link_pos_w) * live,
                      quat_apply_inverse(q, b.root_link_lin_vel_w) * live, live], dim=-1)

  def _update_metrics(self):
    self.metrics["clean_fraction"][:] = self.clean / self.throws.clamp(min=1)
    self.metrics["block_fraction"][:] = self.blocks / self.throws.clamp(min=1)
    self.metrics["level"][:] = self.level

  def reset(self, env_ids):
    for t in (self.throws, self.clean, self.blocks):
      t[env_ids] = 0.0
    out = super().reset(env_ids)
    self.active[env_ids] = False; self.hit_now[env_ids] = 0.0; self.clean_now[env_ids] = 0.0; self.throws[env_ids] = 0.0
    self.time_left[env_ids] = 0.6  # the first ball comes 0.6 s in, aimed from fresh robot data
    self._park(env_ids)
    return out

  def compute(self, dt):
    self.hit_now[:] = 0.0; self.clean_now[:] = 0.0
    super().compute(dt)

  def _park(self, env_ids):
    pose = torch.zeros(len(env_ids), 7, device=self.device)
    pose[:, :3] = self._env.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, 30.0], device=self.device)
    pose[:, 3] = 1.0
    self.ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.ball.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)

  def _finish(self, env_ids):
    """Scores the ball that was in flight for these robots."""
    live = self.active[env_ids]
    ok = live & ~self.took[env_ids]
    self.clean_now[env_ids] = ok.float()
    self.clean[env_ids] += ok; self.blocks[env_ids] += ok & self.armed[env_ids]
    if bool(live.any()):
      self.rate = 0.98 * self.rate + 0.02 * float(ok[live].float().mean())
      self.level = min(1.0, max(0.0, self.level + (0.004 if self.rate > 0.75 else -0.004 if self.rate < 0.5 else 0.0)))
    self.active[env_ids] = False

  def _resample_command(self, env_ids):  # throw the next ball
    self._finish(env_ids)
    k, dev, L = len(env_ids), self.device, self.level
    u = lambda a, b: torch.rand(k, device=dev) * (b - a) + a
    d = self.robot.data
    yaw = yaw_quat(d.root_link_quat_w[env_ids])
    bearing = u(-1, 1) * (0.5 + 1.5 * L)                       # up to +-2 rad off straight ahead
    r = u(1.2, 1.6)
    start = d.root_link_pos_w[env_ids] + quat_apply(yaw, torch.stack([r * torch.cos(bearing), r * torch.sin(bearing), u(0.0, 0.5)], dim=-1))
    aim = d.root_link_pos_w[env_ids] + torch.stack([u(-0.04, 0.04), u(-0.04, 0.04), u(0.05, 0.50)], dim=-1)  # belly to head
    speed = 2.0 + 6.0 * L * torch.rand(k, device=dev)          # 2 m/s at first, up to 8 m/s
    gap = aim - start
    dist = torch.norm(gap, dim=-1)
    pose = torch.zeros(k, 7, device=dev); pose[:, :3] = start; pose[:, 3] = 1.0
    vel = torch.zeros(k, 6, device=dev); vel[:, :3] = gap / dist[:, None] * speed[:, None]
    self.ball.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.ball.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    self.flight[env_ids] = dist / speed
    self.active[env_ids] = True; self.took[env_ids] = False; self.armed[env_ids] = False; self.age[env_ids] = 0.0
    self.throws[env_ids] += 1

  def _touch(self, sensor):
    return torch.norm(sensor.data.force_history, dim=-1).flatten(1).amax(-1) > 5.0  # newtons, any moment in the last control step

  def _update_command(self):
    self.age += self._env.step_dt
    body = self.active & self._touch(self.body_hit) & (self.age > 0.04)
    self.hit_now = (body & ~self.took).float()
    self.took |= body
    self.armed |= self.active & self._touch(self.arm_hit)
    over = (self.active & (self.age > self.flight + 0.3)).nonzero().flatten()  # the ball has gone past
    if len(over):
      self._finish(over)
      self._park(over)
      self.time_left[over] = torch.rand(len(over), device=self.device) * 0.6 + 0.3


@dataclass(kw_only=True)
class ThreatCfg(CommandTermCfg):
  start_level: float = 0.0

  def build(self, env):
    return Threat(self, env)


def _cmd(env) -> Threat:
  return env.command_manager.get_term("threat")

def took_hit(env):   # a ball landed on head or body
  return _cmd(env).hit_now / env.step_dt

def defended(env):   # a ball went by, or was stopped by an arm, without touching head or body
  return _cmd(env).clean_now / env.step_dt

def guard(env):      # fists resting at the guard points
  c = _cmd(env); d = c.robot.data
  glove = torch.tensor(GLOVE, device=env.device).expand(env.num_envs, 2, 3)
  pos = d.body_link_pos_w[:, c.fist_ids] + quat_apply(d.body_link_quat_w[:, c.fist_ids], glove)
  q = d.root_link_quat_w[:, None].expand(-1, 2, -1)
  err = torch.norm(quat_apply_inverse(q, pos - d.root_link_pos_w[:, None]) - torch.tensor(GUARD, device=env.device), dim=-1)
  return torch.exp(-err / 0.15).mean(-1)

def ran_off(env):    # stepping aside is fine; leaving the spot by more than half a metre is not
  d = torch.norm(env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=-1)
  return (d - 0.5).clamp(min=0.0) ** 2


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor = {
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "threat"}),
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=Unoise(n_min=-0.2, n_max=0.2)),
    "gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  critic = {k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor.items()}
  critic["base_lin_vel"] = ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"})
  robot = SceneEntityCfg("robot", joint_names=(".*",))
  ball = ContactMatch(mode="geom", pattern="ball", entity="ball")
  touch = dict(fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4)
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1,
                   entities={"robot": get_g1_robot_cfg(), "ball": EntityCfg(spec_fn=ball_spec)},
                   sensors=(ContactSensorCfg(name="ball_body", secondary=ball, primary=ContactMatch(
                              mode="body", pattern=("pelvis", "torso_link", r"waist.*", r".*hip.*"), entity="robot"), **touch),
                            ContactSensorCfg(name="ball_arm", secondary=ball, primary=ContactMatch(
                              mode="subtree", pattern=("left_shoulder_pitch_link", "right_shoulder_pitch_link"), entity="robot"), **touch),
                            ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                             secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                             fields=("found", "force"), reduce="none", num_slots=1, history_length=4),)),
    observations={"actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=not play),
                  "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False)},
    actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True)},
    commands={"threat": ThreatCfg(resampling_time_range=(1.2, 2.0), start_level=1.0 if play else 0.0)},
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
      "took_hit": RewardTermCfg(func=took_hit, weight=-6.0),
      "defended": RewardTermCfg(func=defended, weight=2.0),
      "guard": RewardTermCfg(func=guard, weight=0.5),
      "ran_off": RewardTermCfg(func=ran_off, weight=-4.0),
      "upright": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-0.5),
      "fell": RewardTermCfg(func=mdp.is_terminated, weight=-1500.0),  # 30 points: five hits taken
      "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
      "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
      "joint_limit": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": robot}),
      "self_collisions": RewardTermCfg(func=self_collision_cost, weight=-10.0, params={"sensor_name": "self_collision", "force_threshold": 10.0}),
    },
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
      "too_low": TerminationTermCfg(func=mdp.root_height_below_minimum, params={"minimum_height": 0.40}),
    },
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="torso_link",
                        distance=3.2, fovy=55.0, elevation=-8.0, azimuth=120.0),
    sim=SimulationCfg(nconmax=45, njmax=300, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
    decimation=4,
    episode_length_s=10.0,
  )
  if play:
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
    experiment_name="g1_defence_v0", save_interval=100, num_steps_per_env=24, max_iterations=3000)
