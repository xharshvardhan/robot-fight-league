"""Shared gym: one G1, one network, four kinds of episode (walk, strike, defend, get up).

Every robot gets the same inputs. Each episode is given one mode; the inputs and rewards of the other modes are switched
off for that episode. Built on the athletic-base task (G1-Base-V2), so its first 105 inputs are exactly that task's inputs
and a body network trained there can be carried over. The pad, ball and get-up helper come from the specialist gyms.
"""
from __future__ import annotations

import functools
import math

import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.base import base, getup
from src.tasks.defence import defence
from src.tasks.strike import strike_v2

MODES = ("walk", "strike", "defend", "getup")


def modes(env) -> torch.Tensor:
  if not hasattr(env, "fight_mode"):
    env.fight_mode = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  return env.fight_mode

def in_mode(env, *names) -> torch.Tensor:
  m = modes(env)
  return sum((m == MODES.index(n)) for n in names).bool()

def only(names, func):
  """The same reward or input, but zero outside the named modes."""
  if isinstance(func, type):  # class-style terms keep their own setup
    class Masked(func):
      def __call__(self, env, **kw):
        out = super().__call__(env, **kw)
        return out * in_mode(env, *names).view(-1, *[1] * (out.dim() - 1))
    Masked.__name__ = func.__name__
    return Masked
  @functools.wraps(func)
  def masked(env, **kw):
    out = func(env, **kw)
    return out * in_mode(env, *names).view(-1, *[1] * (out.dim() - 1))
  return masked

def per_mode(env, value, name):
  """Scales a per-robot number so that its average over all robots is the average over robots in that mode."""
  mask = in_mode(env, name).float()
  return value * mask / mask.mean().clamp(min=1e-3)


def reset_by_mode(env, env_ids, share: tuple[float, ...]):
  """Draws the next episode's mode and puts the robot down standing, or lying for a get-up episode."""
  m = torch.multinomial(torch.tensor(share, device=env.device), len(env_ids), replacement=True)
  modes(env)[env_ids] = m
  up, down = env_ids[m != 3], env_ids[m == 3]
  if len(up):
    mdp.reset_root_state_uniform(env, up, pose_range={"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-3.14, 3.14)}, velocity_range={})
  if len(down):
    getup.reset_fallen(env, down, standing=0.0)


class ModeStrike(strike_v2.StrikeCommand):
  def _resample_command(self, env_ids):
    on = in_mode(self._env, "strike")[env_ids]
    if on.any():
      super()._resample_command(env_ids[on])
    off = env_ids[~on]
    if len(off):
      self.active[off] = False; self.missed[off] = 0.0
      self._place_pad(off)

class ModeStrikeCfg(strike_v2.StrikeCommandCfg):
  def build(self, env):
    return ModeStrike(self, env)


class ModeThreat(defence.Threat):
  def _resample_command(self, env_ids):
    on = in_mode(self._env, "defend")[env_ids]
    if on.any():
      super()._resample_command(env_ids[on])
    off = env_ids[~on]
    if len(off):
      self.active[off] = False
      self._park(off)

class ModeThreatCfg(defence.ThreatCfg):
  def build(self, env):
    return ModeThreat(self, env)


class ModeAssist(getup.Assist):
  def _update_command(self):
    env = self._env
    mask = in_mode(env, "getup")
    up = ((getup._z(env) > getup.STAND_Z - 0.07) & (getup._upright(env) > 0.95)).float()
    self.metrics["standing"][:] = per_mode(env, up, "getup")
    if mask.any():
      frac = float(up[mask].mean())
      self.level = min(self.cfg.start, max(0.0, self.level + (-2e-4 if frac > 0.6 else 2e-4 if frac < 0.4 else 0.0)))
    env.sim.data.xfrc_applied[:, self.torso, 2] = self.level * self.weight * mask

class ModeAssistCfg(getup.AssistCfg):
  def build(self, env):
    return ModeAssist(self, env)


def mode_obs(env):
  return torch.nn.functional.one_hot(modes(env), len(MODES)).float()

def standing_up(env, limit_angle: float):  # falls end the episode, except in a get-up episode
  return mdp.bad_orientation(env, limit_angle) & ~in_mode(env, "getup")

def too_low(env, minimum_height: float):
  return mdp.root_height_below_minimum(env, minimum_height) & ~in_mode(env, "getup")

def walk_error(env):  # m/s between the ordered and the actual ground speed, walking episodes only
  d = env.scene["robot"].data
  return per_mode(env, torch.norm(env.command_manager.get_command("twist")[:, :2] - d.root_link_lin_vel_b[:, :2], dim=-1), "walk")

def strike_hits(env):
  c = env.command_manager.get_term("strike")
  return per_mode(env, c.hits / c.targets.clamp(min=1), "strike")

def defend_clean(env):
  c = env.command_manager.get_term("threat")
  return per_mode(env, c.clean / c.throws.clamp(min=1), "defend")


def env_cfg(play: bool = False, mode: str | None = None, share=(0.25, 0.25, 0.25, 0.25), expert: bool = False,
            walk_fix: bool = False) -> ManagerBasedRlEnvCfg:
  """expert: start at full difficulty with no get-up helper (for a network that already has the skills).
  walk_fix: in the first experiment every trained network learned to end walking episodes in 0.2 s, because falling
  cost almost nothing there. With this on, a walking episode pays for staying up and falling costs as much as in a fight."""
  cfg = base.env_cfg(play=play, v2=True)
  full = play or expert
  if mode:
    share = tuple(float(m == mode) for m in MODES)

  # Scene: the pad and the ball join the robot.
  cfg.scene.entities = {**cfg.scene.entities, "pad": EntityCfg(spec_fn=strike_v2.pad_spec), "ball": EntityCfg(spec_fn=defence.ball_spec)}
  touch = dict(fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4)
  pad = ContactMatch(mode="geom", pattern="pad", entity="pad")
  ball = ContactMatch(mode="geom", pattern="ball", entity="ball")
  cfg.scene.sensors = (*cfg.scene.sensors,
    *(ContactSensorCfg(name=f"{s}_pad", primary=ContactMatch(mode="subtree", pattern=f"{s}_wrist_yaw_link", entity="robot"), secondary=pad, **touch)
      for s in ("left", "right")),
    ContactSensorCfg(name="any_pad", primary=pad, secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"), **touch),
    ContactSensorCfg(name="ball_body", secondary=ball, primary=ContactMatch(
      mode="body", pattern=("pelvis", "torso_link", r"waist.*", r".*hip.*"), entity="robot"), **touch),
    ContactSensorCfg(name="ball_arm", secondary=ball, primary=ContactMatch(
      mode="subtree", pattern=("left_shoulder_pitch_link", "right_shoulder_pitch_link"), entity="robot"), **touch))
  cfg.sim.nconmax, cfg.sim.njmax = 80, 600

  # Orders. The speed order and body state come from the base task.
  cfg.commands["strike"] = ModeStrikeCfg(resampling_time_range=(1.5, 1.5), debug_vis=True, log=play, start_level=1.0 if full else 0.0)
  cfg.commands["threat"] = ModeThreatCfg(resampling_time_range=(1.2, 2.0), start_level=1.0 if full else 0.0)
  cfg.commands["assist"] = ModeAssistCfg(resampling_time_range=(1e6, 1e6), start=0.0 if full else 0.7)

  # Inputs: the base task's inputs first and unchanged in layout, then the mode, the strike order and the incoming ball.
  for group in ("actor", "critic"):
    t = cfg.observations[group].terms
    t["command"].func = only(("walk",), t["command"].func)
    t["mode"] = ObservationTermCfg(func=mode_obs)
    t["strike"] = ObservationTermCfg(func=only(("strike",), mdp.generated_commands), params={"command_name": "strike"})
    t["threat"] = ObservationTermCfg(func=only(("defend",), mdp.generated_commands), params={"command_name": "threat"})
  c = cfg.observations["critic"].terms
  c["root_height"] = ObservationTermCfg(func=getup.root_height_obs)
  c["assist"] = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "assist"})

  # Rewards. Walking terms count only in walking episodes; smoothness and safety terms count everywhere.
  everywhere = ("is_terminated", "joint_acc_l2", "joint_pos_limits", "action_rate_l2", "self_collisions")
  for name, term in cfg.rewards.items():
    if name not in everywhere:
      term.func = only(("walk",), term.func)
  fight = ("strike", "defend")
  cfg.rewards.update({
    "reach": RewardTermCfg(func=strike_v2.reach, weight=1.0),
    "hit": RewardTermCfg(func=strike_v2.hit, weight=1.0),
    "waiting": RewardTermCfg(func=strike_v2.waiting, weight=-1.0),
    "missed": RewardTermCfg(func=strike_v2.missed, weight=-3.0),
    "foul": RewardTermCfg(func=strike_v2.foul, weight=-2.0),
    "took_hit": RewardTermCfg(func=defence.took_hit, weight=-6.0),
    "defended": RewardTermCfg(func=defence.defended, weight=2.0),
    "ran_off": RewardTermCfg(func=only(("defend",), defence.ran_off), weight=-4.0),
    "guard": RewardTermCfg(func=only(fight, strike_v2.guard), weight=1.0),
    "upright": RewardTermCfg(func=only(fight, mdp.flat_orientation_l2), weight=-1.0),
    "alive": RewardTermCfg(func=only(fight, mdp.is_alive), weight=0.5),
    "fell_fighting": RewardTermCfg(func=only(fight, mdp.is_terminated), weight=-1300.0),  # with is_terminated: 30 points
    "rise": RewardTermCfg(func=only(("getup",), getup.height), weight=3.0),
    "rise_upright": RewardTermCfg(func=only(("getup",), getup.upright_tall), weight=1.0),
    "rise_standing": RewardTermCfg(func=only(("getup",), getup.standing), weight=2.0),
    "rise_feet_under": RewardTermCfg(func=only(("getup",), getup.feet_under), weight=1.0),
    "rise_joint_speed": RewardTermCfg(func=only(("getup",), getup.joint_speed), weight=-2e-4),
  })

  if walk_fix:
    stand = ("walk", "strike", "defend")
    cfg.rewards["alive"].func = only(stand, mdp.is_alive)
    cfg.rewards["fell_fighting"].func = only(stand, mdp.is_terminated)

  cfg.events["reset_base"] = EventTermCfg(func=reset_by_mode, mode="reset", params={"share": share})
  cfg.terminations["fell_over"] = TerminationTermCfg(func=standing_up, params={"limit_angle": math.radians(70.0)})
  cfg.terminations["too_low"] = TerminationTermCfg(func=too_low, params={"minimum_height": 0.40})
  cfg.metrics.update({"walk_error": MetricsTermCfg(func=walk_error), "strike_hits": MetricsTermCfg(func=strike_hits),
                      "defend_clean": MetricsTermCfg(func=defend_clean)})
  if not play:
    cfg.episode_length_s = 10.0
  return cfg


def rl_cfg(name: str = "g1_fighter_scratch", hidden=(512, 256, 128)):
  cfg = base.rl_cfg(v2=True)
  cfg.experiment_name = name
  cfg.max_iterations = 4000
  cfg.actor.hidden_dims = cfg.critic.hidden_dims = hidden
  return cfg
