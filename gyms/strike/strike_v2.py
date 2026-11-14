"""Strike gym: a G1 is shown a target point and an order (which fist, which direction, how soon) and is scored on the hit.

No reference motion. Version 2: as version 1, plus pads that appear out of reach at any bearing, at varied heights, and drift.
How far, how fast and how high grows with a difficulty level (0 to 1) that rises while the robots keep hitting.
Version 1: the target is a physical flat pad that faces the direction it expects to be hit from.
A hit is the ordered fist pressing on the pad's face; its strength is the contact force along the ordered direction.
Slow hits earn less, an unhit pad costs points, and touching the pad with anything else costs points.
"""
from __future__ import annotations

import math

import mujoco
from dataclasses import dataclass

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
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_mul, yaw_quat
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from src.tasks.tracking.mdp.rewards import self_collision_cost

GLOVE = (0.1, 0.0, 0.0)                    # fist centre in the wrist link frame
GUARD = ((0.18, 0.10, 0.20), (0.18, -0.10, 0.20))  # where each fist rests, relative to the pelvis (left, right)
KINDS = ("forward", "up", "inward", "down")  # direction the pad expects the force in: jab/cross, uppercut, hook, chop
# Where each kind of pad may appear, relative to the pelvis: x forward, y toward the striking fist's own side, z up.
BOX = {"forward": ((0.36, 0.50), (0.00, 0.20), (0.15, 0.45)), "up": ((0.33, 0.45), (0.00, 0.20), (0.22, 0.45)),
       "inward": ((0.30, 0.42), (-0.05, 0.10), (0.20, 0.45)), "down": ((0.33, 0.47), (0.05, 0.25), (0.00, 0.18))}
PAD_R, PAD_H = 0.09, 0.02                  # pad radius and half thickness (metres)
R2 = math.sqrt(0.5)
PAD_QUAT = ((R2, 0.0, R2, 0.0), (1.0, 0.0, 0.0, 0.0), (R2, R2, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))  # turns the pad's face axis per kind


def pad_spec():
  s = mujoco.MjSpec()
  s.worldbody.add_body(name="pad", mocap=True).add_geom(name="pad", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[PAD_R, PAD_H, 0],
                                                        rgba=[0.9, 0.2, 0.15, 1])
  return s


class StrikeCommand(CommandTerm):
  cfg: StrikeCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.pad = env.scene["pad"]
    self.sensors = [env.scene["left_pad"], env.scene["right_pad"], env.scene["any_pad"]]
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
    self.missed = torch.zeros(n, device=dev)       # 1 on the step a pad expired unhit
    self.foul = torch.zeros(n, device=dev)         # newtons on the pad from anything but the ordered fist
    self.age = torch.zeros(n, device=dev)          # seconds the current pad has been up
    self.prev_vel = torch.zeros(n, 3, device=dev)  # ordered fist velocity one step ago (the speed just before impact)
    self.pad_quat = torch.tensor(PAD_QUAT, device=dev)
    self.pad_vel = torch.zeros(n, 3, device=dev)   # the pad drifts at this velocity
    self.pad_yaw = torch.zeros(n, 4, device=dev); self.pad_yaw[:, 0] = 1.0  # heading the pad's directions are defined in
    self.window = torch.full((n,), 1.5, device=dev)  # seconds allowed for the current pad
    self.level = cfg.start_level                   # difficulty, 0 to 1
    self.rate = 0.5                                # running hit rate over recent pads
    self.targets, self.hits, self.force_sum, self.react_sum = (torch.zeros(n, device=dev) for _ in range(4))
    for k in ("hit_fraction", "hit_force", "reaction_s", "level"):
      self.metrics[k] = torch.zeros(n, device=dev)
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
                      (self.time_left / self.window)[:, None]], dim=-1)

  def reset(self, env_ids):
    for t in (self.targets, self.hits, self.force_sum, self.react_sum, self.hit_bonus, self.foul):
      t[env_ids] = 0.0
    self.active[env_ids] = False
    out = super().reset(env_ids)
    # The robot has only just been moved to its start pose, so no pad yet: the first one comes 0.6 s in, placed from fresh data.
    self.active[env_ids] = False; self.missed[env_ids] = 0.0; self.targets[env_ids] = 0.0; self.time_left[env_ids] = 0.6
    self._place_pad(env_ids)
    return out

  def compute(self, dt):
    self.missed[:] = 0.0
    super().compute(dt)

  def _update_metrics(self):
    self.metrics["hit_fraction"][:] = self.hits / self.targets.clamp(min=1)
    self.metrics["hit_force"][:] = self.force_sum / self.hits.clamp(min=1)
    self.metrics["reaction_s"][:] = self.react_sum / self.hits.clamp(min=1)
    self.metrics["level"][:] = self.level

  def _force(self, i):
    """Largest contact force (world frame) over the last control step from sensor i, shape [envs, 3]."""
    h = self.sensors[i].data.force_history[:, 0]  # [envs, substeps, 3]
    return h[self.idx, torch.norm(h, dim=-1).argmax(dim=-1)]

  def _place_pad(self, env_ids):
    pose = torch.zeros(len(env_ids), 7, device=self.device)
    live = self.active[env_ids]
    pose[:, :3] = torch.where(live[:, None], self.target_w[env_ids], self._env.scene.env_origins[env_ids] + torch.tensor([0.0, 0.0, -1.0], device=self.device))
    pose[:, 3:] = quat_mul(self.pad_yaw[env_ids], self.pad_quat[self.kind[env_ids]])
    self.pad.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def _resample_command(self, env_ids):
    k, dev, c, L = len(env_ids), self.device, self.cfg, self.level
    self.missed[env_ids] = self.active[env_ids].float()
    u = lambda lo, hi: torch.rand(k, device=dev) * (hi - lo) + lo
    limb = torch.randint(0, 2, (k,), device=dev)
    side = 1.0 - 2.0 * limb  # +1 for the left fist, -1 for the right
    kind = torch.randint(0, len(KINDS), (k,), device=dev)
    box = torch.tensor([BOX[n] for n in KINDS], device=dev)[kind]  # [k, 3, 2]
    local = box[..., 0] + torch.rand(k, 3, device=dev) * (box[..., 1] - box[..., 0])
    local[:, 1] *= side
    dirs = torch.zeros(k, 3, device=dev)
    dirs[kind == 0, 0] = 1.0
    dirs[kind == 1, 2] = 1.0
    dirs[kind == 2, 1] = -side[kind == 2]
    dirs[kind == 3, 2] = -1.0
    far = u(0.0, c.far * L) * (torch.rand(k, device=dev) < 0.7)   # 30% of pads stay within reach
    local[:, 0] += far
    local[:, 2] = (local[:, 2] + u(-c.z_jitter * L, c.z_jitter * L)).clamp(-0.15, 0.75)
    bearing = u(-c.bearing * L, c.bearing * L)
    zero = torch.zeros(k, device=dev)
    turn = torch.stack([torch.cos(bearing / 2), zero, zero, torch.sin(bearing / 2)], dim=-1)
    yaw = quat_mul(yaw_quat(self.robot.data.root_link_quat_w[env_ids]), turn)
    self.pad_yaw[env_ids] = yaw
    self.pad_vel[env_ids] = quat_apply(yaw, torch.stack([u(-1, 1), u(-1, 1), u(-0.3, 0.3)], dim=-1)) * c.drift * L
    self.window[env_ids] = 1.5 + far / 0.5 + bearing.abs() / 2.0   # extra time to walk and turn
    self.time_left[env_ids] = self.window[env_ids]
    root = self.robot.data.root_link_pos_w[env_ids].clone(); root[:, 2] = self._env.scene.env_origins[env_ids, 2] + 0.76
    self.target_w[env_ids] = root + quat_apply(yaw, local)
    self.dir_w[env_ids] = quat_apply(yaw, dirs)
    self.limb[env_ids], self.kind[env_ids], self.active[env_ids] = limb, kind, True
    self.age[env_ids] = 0.0
    self.targets[env_ids] += 1
    self._place_pad(env_ids)

  def _update_command(self):
    c = self.cfg
    live = self.active.nonzero().flatten()
    if len(live):  # drifting pads
      self.target_w[live] += self.pad_vel[live] * self._env.step_dt
      self.target_w[live, 2] = self.target_w[live, 2].clamp(min=c.min_z)
      self._place_pad(live)
    n_hit_or_miss = float(self.missed.sum())  # pads that just expired unhit
    pos, vel = self.fists()
    f, v = pos[self.idx, self.limb], vel[self.idx, self.limb]
    self.dist = torch.norm(f - self.target_w, dim=-1)
    self.age += self._env.step_dt
    fl, fr, fall = self._force(0), self._force(1), self._force(2)
    ordered = torch.where(self.limb[:, None] == 0, fl, fr)
    push = (ordered * self.dir_w).sum(-1).abs()                    # newtons along the ordered direction
    front = ((f - self.target_w) * self.dir_w).sum(-1) < 0         # fist is on the side the pad faces
    self.foul = self.active * (torch.norm(fall, dim=-1) - torch.norm(ordered, dim=-1)).clamp(min=0.0)
    swing = (self.prev_vel * self.dir_w).sum(-1) > c.min_speed    # the fist arrived moving, not resting or shoved into the pad
    hit = self.active & front & swing & (push > c.min_force) & (self.age > 0.1)
    quick = 1.0 - 0.6 * (self.age / self.window).clamp(max=1.0)  # a late hit is worth 40% of an instant one
    self.hit_bonus = torch.where(hit, (2.0 + 8.0 * (push / c.full_force).clamp(max=1.0)) * quick, torch.zeros_like(push))
    self.hits += hit; self.force_sum += push * hit; self.react_sum += self.age * hit
    if c.log:
      for i in hit.nonzero().flatten().tolist():
        self.strike_log.append(dict(env=i, step=int(self._env.common_step_counter), fist=("left", "right")[int(self.limb[i])],
                                    order=KINDS[int(self.kind[i])], force_along_order_N=round(float(push[i]), 1),
                                    force_vector_N=[round(x, 1) for x in ordered[i].tolist()],
                                    fist_velocity_before=[round(x, 2) for x in self.prev_vel[i].tolist()],
                                    pad=[round(x, 3) for x in self.target_w[i].tolist()], reaction_s=round(float(self.age[i]), 2)))
    self.prev_vel = v.clone()
    n_hit = float(hit.sum()); n = n_hit + n_hit_or_miss
    if n and not c.log:  # difficulty follows the recent hit rate
      self.rate += min(1.0, n / 2000.0) * (n_hit / n - self.rate)
      self.level = min(1.0, max(0.0, self.level + (2e-4 if self.rate > 0.8 else -2e-4 if self.rate < 0.6 else 0.0)))
    done = hit.nonzero().flatten()
    if len(done):  # the pad drops away and the next one comes soon, so a fast fighter gets more pads
      self.active[done] = False
      self.time_left[done] = torch.rand(len(done), device=self.device) * 0.4 + 0.4
      self._place_pad(done)

  def _debug_vis_impl(self, visualizer):
    for b in visualizer.get_env_indices(self.num_envs):
      if bool(self.active[b]):
        t, d = self.target_w[b].cpu().numpy(), self.dir_w[b].cpu().numpy()
        visualizer.add_arrow(t - 0.3 * d, t - 0.05 * d, (1.0, 0.85, 0.2, 0.9), label=f"order_{b}")


@dataclass(kw_only=True)
class StrikeCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  min_speed: float = 1.0     # m/s along the ordered direction just before contact
  min_force: float = 20.0    # newtons along the ordered direction for the hit to count
  full_force: float = 400.0  # newtons that earn the full bonus
  far: float = 1.5           # metres beyond reach at full difficulty
  bearing: float = 3.14      # radians either side of straight ahead
  z_jitter: float = 0.25     # metres of extra height spread
  drift: float = 0.6         # m/s pad drift
  min_z: float = 0.55        # a drifting pad never sinks below this
  start_level: float = 0.0
  log: bool = False

  def build(self, env):
    return StrikeCommand(self, env)


def _cmd(env) -> StrikeCommand:
  return env.command_manager.get_term("strike")

def reach(env):  # pull the ordered fist toward a waiting pad, near or far
  c = _cmd(env); return c.active * 0.5 * (torch.exp(-c.dist / 0.3) + torch.exp(-c.dist / 1.2))

def hit(env):  # one-off bonus when the hit lands, larger for a faster fist
  return _cmd(env).hit_bonus / env.step_dt

def guard(env):  # fists at the guard points; the ordered fist is excused while its target is waiting
  c = _cmd(env); d = c.robot.data
  pos, _ = c.fists()
  q = d.root_link_quat_w[:, None].expand(-1, 2, -1)
  err = torch.norm(quat_apply_inverse(q, pos - d.root_link_pos_w[:, None]) - c.guard, dim=-1)
  w = torch.ones_like(err); w[c.idx, c.limb] = (~c.active).float()
  return (w * torch.exp(-err / 0.15)).sum(-1) / 2

def waiting(env):  # every moment a pad is left standing costs a little
  return _cmd(env).active.float()

def missed(env):  # the pad expired unhit
  return _cmd(env).missed / env.step_dt

def foul(env):  # pad touched by the wrong fist, a forearm, the body...
  return (_cmd(env).foul / 100.0).clamp(max=3.0)

def wide_feet(env, ids=[]):  # stopgap for the straddle stance: feet more than 40 cm apart
  d = env.scene["robot"].data
  if not ids: ids.extend(env.scene["robot"].find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)[0])
  p = d.body_link_pos_w[:, ids]
  return (torch.norm(p[:, 0, :2] - p[:, 1, :2], dim=-1) - 0.40).clamp(min=0.0)

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
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1,
                   entities={"robot": get_g1_robot_cfg(), "pad": EntityCfg(spec_fn=pad_spec)},
                   sensors=(*(ContactSensorCfg(name=f"{s}_pad", primary=ContactMatch(mode="subtree", pattern=f"{s}_wrist_yaw_link", entity="robot"),
                                               secondary=ContactMatch(mode="geom", pattern="pad", entity="pad"), fields=("found", "force"),
                                               reduce="netforce", num_slots=1, history_length=4) for s in ("left", "right")),
                            ContactSensorCfg(name="any_pad", primary=ContactMatch(mode="geom", pattern="pad", entity="pad"),
                                             secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"), fields=("found", "force"),
                                             reduce="netforce", num_slots=1, history_length=4),
                            ContactSensorCfg(name="self_collision", primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                             secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                                             fields=("found", "force"), reduce="none", num_slots=1, history_length=4),)),
    observations={"actor": ObservationGroupCfg(terms=actor, concatenate_terms=True, enable_corruption=not play),
                  "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False)},
    actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True)},
    commands={"strike": StrikeCommandCfg(resampling_time_range=(1.5, 1.5), debug_vis=True, log=play, start_level=1.0 if play else 0.0)},
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
      "waiting": RewardTermCfg(func=waiting, weight=-1.0),
      "missed": RewardTermCfg(func=missed, weight=-3.0),
      "foul": RewardTermCfg(func=foul, weight=-2.0),
      "upright": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
      "fell": RewardTermCfg(func=mdp.is_terminated, weight=-1500.0),  # 30 points: three perfect hits, so diving at a pad never pays
      "alive": RewardTermCfg(func=mdp.is_alive, weight=0.5),
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
    sim=SimulationCfg(nconmax=45, njmax=300, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
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
    experiment_name="g1_strike_v2", save_interval=100, num_steps_per_env=24, max_iterations=4000)
