"""Fight gym (stages 2 to 4): the combo-gym fighter faces a second robot run by a frozen copy of a fighter network.

Red ("robot") learns. Blue ("rival") is a saved network (FIGHT_RIVAL, default ~/rival.pt) that does not learn in this run;
running again with red's result as the new rival is self-play.
Red's inputs have exactly the combo gym's layout, so the same network carries over:
- the four "pads" are now blue's head, chest, pelvis and nearer knee (worth 3, 2, 2, 1 by height, as in the combo gym);
- the "opponent limbs" block, noise until now, is blue's fists and feet (position and velocity).
Blue gets the mirror image. In some bouts blue is given points in the air instead of red's body, so it moves around
without attacking (the moving dummy); the share of attacking bouts rises from attack_start to 1 during the run.
Scoring is the same both ways: zone value x force (800 N = full), kicks a quarter more. Red is paid its own points and
charged blue's. A fall ends the round. Chest-to-chest for half a second: the referee puts both back in their corners.
"""
from __future__ import annotations

import dataclasses
import os
import random
import re
from dataclasses import dataclass

import torch
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import mdp
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from src.tasks.fighter.runners import Teacher
from src.tasks.strike import combo, strike_v2

GAP, CLINCH, CLINCH_T, H = 1.5, 0.60, 0.5, 5
COOL = 0.5                      # seconds between one fighter's scored hits
WINDOW, J_SOFT, J_FULL = 4, 3.0, 20.0   # control steps (80 ms) a hit's impulse is added up over; N s that earn nothing and full marks (measured 14 Nov: an average hit was 7 N s)
SLOTS = (("torso_link", (0.0, 0.0, 0.42)), ("torso_link", (0.0, 0.0, 0.20)), ("pelvis", (0.0, 0.0, 0.0)))   # head, chest, pelvis; the 4th is a knee
ZONES = {"head": (3.0, ("head_collision",)),
         "body": (2.0, ("torso_collision", "pelvis_collision", ".*_hip_collision", ".*shoulder_yaw_collision")),
         "legs": (1.0, (".*_thigh_collision", ".*_shin_collision"))}
# striking bodies, m/s the limb must arrive at, newtons to count at all, bonus
LIMBS = {"punch": (("left_wrist_yaw_link", "right_wrist_yaw_link"), 1.5, 100.0, 1.0), "kick": (("left_knee_link", "right_knee_link"), 3.0, 100.0, 1.6)}


class Side:
  """What the gym needs to know about one fighter."""
  def __init__(self, env, name):
    self.e = env.scene[name]
    dev, n = env.device, env.num_envs
    self.pt_ids = torch.tensor(self.e.find_bodies([b for b, _ in combo.POINTS], preserve_order=True)[0], device=dev)
    self.pt_off = torch.tensor([o for _, o in combo.POINTS], device=dev).expand(n, len(combo.POINTS), 3)
    self.slot_ids = torch.tensor([self.e.find_bodies(b)[0][0] for b, _ in SLOTS], device=dev)
    self.slot_off = torch.tensor([o for _, o in SLOTS], device=dev).expand(n, len(SLOTS), 3)
    self.knees = torch.tensor(self.e.find_bodies(("left_knee_link", "right_knee_link"), preserve_order=True)[0], device=dev)

  def strikers(self):
    d = self.e.data
    r = quat_apply(d.body_link_quat_w[:, self.pt_ids], self.pt_off)
    return d.body_link_pos_w[:, self.pt_ids] + r, d.body_link_lin_vel_w[:, self.pt_ids] + torch.cross(d.body_link_ang_vel_w[:, self.pt_ids], r, dim=-1)

  def refs(self):
    """Reference points of each scoring zone and how close (m) a striking point must be to count as being at it."""
    d = self.e.data
    top = d.body_link_pos_w[:, self.slot_ids] + quat_apply(d.body_link_quat_w[:, self.slot_ids], self.slot_off)   # head, chest, pelvis
    k = d.body_link_pos_w[:, self.knees]
    return {"head": (top[:, :1], 0.22), "body": (top[:, 1:3], 0.30), "legs": (torch.cat([k, (k + top[:, 2:3]) / 2], dim=1), 0.25)}

  def zones(self, seen_from):
    """Head, chest, pelvis and the knee nearer to seen_from, world positions [envs, 4, 3]."""
    d = self.e.data
    top = d.body_link_pos_w[:, self.slot_ids] + quat_apply(d.body_link_quat_w[:, self.slot_ids], self.slot_off)
    k = d.body_link_pos_w[:, self.knees]
    near = (k - seen_from[:, None]).norm(dim=-1).argmin(dim=-1)
    return torch.cat([top, k[torch.arange(len(near), device=near.device), near][:, None]], dim=1)


def orders(me, targets, live, other_pts, other_vel):
  """The two input blocks of a fighter: its targets (20 numbers) and the other's fists and feet (24)."""
  d = me.data
  q = d.root_link_quat_w[:, None]
  rel = quat_apply_inverse(q.expand(-1, targets.shape[1], -1), targets - d.root_link_pos_w[:, None])
  cmd = torch.cat([rel, combo.zone_value(targets[..., 2] - 0.0)[..., None] / 3.0, live[..., None].float()], dim=-1).flatten(1)
  q4 = q.expand(-1, 4, -1)
  opp = torch.cat([quat_apply_inverse(q4, other_pts[:, :4] - d.root_link_pos_w[:, None]).flatten(1),
                   quat_apply_inverse(q4, other_vel[:, :4] - d.root_link_lin_vel_w[:, None]).flatten(1)], dim=-1)
  return cmd, opp


class Fight(CommandTerm):
  cfg: FightCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    n, dev = self.num_envs, self.device
    self.robot = env.scene["robot"]            # the names the combo gym's reward functions expect
    self.red, self.blue = Side(env, "robot"), Side(env, "rival")
    self.foot_ids = self.robot.find_bodies(combo.FEET, preserve_order=True)[0]
    self.guard = torch.tensor(strike_v2.GUARD, device=dev).expand(n, 2, 3)
    first = Teacher(os.path.expanduser(os.environ.get("FIGHT_RIVAL", "~/rival.pt")), dev)
    self.brains, self.pool = [first, first, first], []          # blue's network: slot 0 the newest copy of red, slots 1 and 2 older snapshots
    self.slot = torch.bucketize(torch.rand(n, device=dev), torch.tensor([0.6, 0.8], device=dev))
    self.home_d, self.face = torch.zeros(n, 2, device=dev), torch.zeros(n, device=dev)
    self.sens = {(s, l, k): env.scene[f"{s}_{l}_{k}"] for s in ("red", "blue") for l in LIMBS for k in ZONES}
    z = lambda *s: torch.zeros(n, *s, device=dev)
    self.cmd, self.opp = z(20), z(24)
    self.b_act, self.b_hist = z(29), z(H, 64)
    self.attack = torch.ones(n, dtype=torch.bool, device=dev)
    self.air, self.air_t = z(4, 3), z()
    self.points, self.conceded, self.foul, self.near = z(), z(), z(), z() + 1.0
    self.cool = {"red": z(), "blue": z()}
    self.d_prev, self.c_prev = {}, {}
    self.b_feet = self.blue.e.find_bodies(combo.FEET, preserve_order=True)[0]
    self.stagger = z()
    self.hit = {s: {"left": torch.zeros(n, dtype=torch.long, device=dev), "which": torch.zeros(n, dtype=torch.long, device=dev), "j": z(), "worth": z()} for s in ("red", "blue")}
    self.impulse = {s: z() for s in ("red", "blue")}
    self.last = {s: z() - 1.0 for s in ("red", "blue")}
    self.was = {(s, l, k): torch.zeros(n, dtype=torch.bool, device=dev) for s in ("red", "blue") for l in LIMBS for k in ZONES}
    self.since_hit, self.clinch_t, self.dist = z() + 9.0, z(), z() + GAP
    self.knockdown, self.broke, self.blue_down = z(), z(), torch.zeros(n, dtype=torch.bool, device=dev)
    self.tot = {k: z() for k in ("red_points", "blue_points", "head", "body", "legs", "kicks", "knockdowns", "breaks", "clinch", "steps", "contacts", "impulse")}
    for k in self.tot:
      if k != "steps":
        self.metrics[k] = z()
    self.metrics["attack_share"] = z()
    self.b_ids = None

  def strikers(self):
    return self.red.strikers()

  def set_rivals(self, state):
    """Self-play: blue becomes the newest copy of red; two smaller groups of bouts face older snapshots."""
    self.pool = (self.pool + [state])[-30:]
    self.brains = [Teacher(state, self.device), Teacher(random.choice(self.pool), self.device), Teacher(random.choice(self.pool), self.device)]

  def _guard_gap(self):
    """How far each of red's fists is from its guard point (m)."""
    d = self.robot.data
    p, _ = self.red.strikers()
    q = d.root_link_quat_w[:, None].expand(-1, 2, -1)
    return (quat_apply_inverse(q, p[:, :2] - d.root_link_pos_w[:, None]) - self.guard).norm(dim=-1)

  @property
  def command(self):
    return self.cmd

  def _resample_command(self, env_ids):
    pass

  def reset(self, env_ids):
    for t in (*self.tot.values(), self.clinch_t, self.cool["red"], self.cool["blue"], self.b_act):
      t[env_ids] = 0.0
    self.since_hit[env_ids] = 9.0; self.air_t[env_ids] = 0.0; self.blue_down[env_ids] = False
    self.last["red"][env_ids] = -1.0; self.last["blue"][env_ids] = -1.0
    c = self.cfg
    p = 1.0 if c.ramp_steps <= 0 else min(1.0, c.attack_start + (1.0 - c.attack_start) * self._env.common_step_counter / c.ramp_steps)
    self.attack[env_ids] = torch.rand(len(env_ids), device=self.device) < p
    self.b_hist[env_ids] = float("nan")   # refilled from the first reading
    self.p_attack = p
    return super().reset(env_ids)

  def _update_metrics(self):
    t = self.tot
    mins = (t["steps"] * self._env.step_dt / 60.0).clamp(min=1e-3)
    for k, v in t.items():
      if k == "clinch":
        self.metrics[k][:] = v / t["steps"].clamp(min=1)   # share of the bout spent chest to chest
      elif k == "impulse":
        self.metrics[k][:] = v / t["contacts"].clamp(min=1)   # N s per strike that made contact, scored or not
      elif k != "steps":
        self.metrics[k][:] = v / mins
    self.metrics["attack_share"][:] = getattr(self, "p_attack", 1.0)

  def _score(self, side, target_root_vel):
    """Points this side landed this step, and for red which zone and whether it was a kick.

    A hit starts when a fist, foot or shin that is closing on the other fighter makes a new contact at one of its zones.
    Its strength is the impulse (newton-seconds: the momentum put into the other fighter) over the first WINDOW steps of
    that contact; the points are paid when the window closes. Peak force, used until 14 Nov, paid for stiff light taps."""
    me, other = (self.red, self.blue) if side == "red" else (self.blue, self.red)
    p, _ = me.strikers()
    refs = other.refs()
    dt, n, dev = self._env.step_dt, self.num_envs, self.device
    st = self.hit[side]
    start = torch.zeros(n, dtype=torch.bool, device=dev)
    imp = []
    for li, (l, (_, min_v, min_f, bonus)) in enumerate(LIMBS.items()):
      idx = slice(0, 2) if l == "punch" else slice(2, 6)
      for zi, (k, (worth, _)) in enumerate(ZONES.items()):
        h = self.sens[(side, l, k)].data.force_history.norm(dim=-1)                # [envs, shapes, substeps] newtons
        imp.append(h.sum(dim=(-1, -2)) * dt / h.shape[-1])                         # impulse this step, N s
        touch = h.amax(dim=-1).sum(dim=-1) > min_f
        key = (side, l, k)
        fresh = touch & ~self.was[key]
        self.was[key] = touch
        d = (p[:, idx, None] - refs[k][0][:, None]).norm(dim=-1).amin(dim=-1)      # each striking point to this zone
        closing = (self.d_prev.get(key, d) - d) / dt
        thrown = torch.maximum(closing, self.c_prev.get(key, closing))             # this step or the one before (contact slows it)
        self.d_prev[key], self.c_prev[key] = d, closing
        dmin, who = d.min(dim=-1)
        ok = fresh & (torch.gather(thrown, 1, who[:, None])[:, 0] > min_v) & (dmin < refs[k][1]) & (st["left"] == 0) & (self.cool[side] <= 0) & ~start
        st["which"] = torch.where(ok, torch.full_like(st["which"], li * 3 + zi), st["which"])
        st["worth"] = torch.where(ok, torch.full_like(st["worth"], worth * bonus), st["worth"])
        start |= ok
    imp = torch.stack(imp, dim=1)                                                   # [envs, 6]
    st["left"] = torch.where(start, torch.full_like(st["left"], WINDOW + 1), st["left"])
    st["j"] = torch.where(start, torch.zeros_like(st["j"]), st["j"])
    open_ = st["left"] > 0
    st["j"] += open_ * torch.gather(imp, 1, st["which"][:, None])[:, 0]
    st["left"] = (st["left"] - 1).clamp(min=0)
    landed = open_ & (st["left"] == 0)
    zone, kick = (st["which"] % 3).float(), (st["which"] >= 3).float()
    self.cool[side] -= dt
    self.cool[side] = torch.where(landed, torch.full_like(self.cool[side], COOL), self.cool[side])
    # Mixing it up pays: the same zone with the same kind of limb as the last hit is worth 0.6, anything else 1.3.
    mult = torch.where(st["which"].float() == self.last[side], 0.6, 1.3) if self.cfg.variety else 1.0
    self.last[side] = torch.where(landed, st["which"].float(), self.last[side])
    self.impulse[side] = st["j"] * landed
    pts = st["worth"] * ((st["j"] - J_SOFT) / (J_FULL - J_SOFT)).clamp(0.0, 1.0) * mult * landed
    return pts, landed & (pts > 0), zone, kick

  def _update_command(self):
    env, dt, dev, n = self._env, self._env.step_dt, self.device, self.num_envs
    A, B = self.robot, self.blue.e
    a_pts, a_vel = self.red.strikers()
    b_pts, b_vel = self.blue.strikers()
    a_root, b_root = A.data.root_link_pos_w, B.data.root_link_pos_w
    ones = torch.ones(n, 4, dtype=torch.bool, device=dev)

    # red's inputs
    b_zones = self.blue.zones(a_root)
    self.cmd, self.opp = orders(A, b_zones, ones, b_pts, b_vel)
    self.near = ((b_zones[:, :, None] - a_pts[:, None]).norm(dim=-1)).flatten(1).amin(dim=-1)

    # blue's inputs: red's body in an attacking bout, points in the air otherwise
    self.air_t -= dt
    new = (self.air_t <= 0).nonzero().flatten()
    if len(new):
      xy = (torch.rand(len(new), 4, 2, device=dev) * 2 - 1) * 1.5
      self.air[new, :, :2] = env.scene.env_origins[new, None, :2] + xy
      self.air[new, :, 2] = env.scene.env_origins[new, None, 2] + torch.rand(len(new), 4, device=dev) * 0.7 + 0.5
      self.air_t[new] = torch.rand(len(new), device=dev) * 2.0 + 2.0
    b_targets = torch.where(self.attack[:, None, None], self.red.zones(b_root), self.air)
    b_cmd, b_opp = orders(B, b_targets, ones, a_pts, a_vel)
    d = B.data
    now = torch.cat([mdp.builtin_sensor(env, "rival/imu_ang_vel"), d.projected_gravity_b, d.joint_pos - d.default_joint_pos, d.joint_vel], dim=-1)
    fresh = torch.isnan(self.b_hist[:, 0, 0])
    self.b_hist = torch.where(fresh[:, None, None], now[:, None].expand(-1, H, -1), torch.cat([self.b_hist[:, 1:], now[:, None]], dim=1))
    h = self.b_hist
    obs = torch.cat([b_cmd, h[..., 0:3].flatten(1), h[..., 3:6].flatten(1), h[..., 6:35].flatten(1), h[..., 35:64].flatten(1), self.b_act, b_opp], dim=-1)
    act = torch.zeros_like(self.b_act)
    for i, brain in enumerate(self.brains):
      m = self.slot == i
      if m.any():
        act[m] = brain(obs[m])
    self.b_act = act
    if self.b_ids is None:
      names = env.action_manager.get_term("joint_pos").target_names
      self.b_ids = torch.tensor(B.find_joints(names, preserve_order=True)[0], device=dev)
      self.b_scale = torch.tensor([next(v for k, v in G1_ACTION_SCALE.items() if re.fullmatch(k, j)) for j in names], device=dev)
    B.set_joint_position_target(d.default_joint_pos[:, self.b_ids] + self.b_act * self.b_scale, joint_ids=self.b_ids)
    self.last_rival_obs = obs

    self.home_d = self._guard_gap()
    to = b_root - a_root; to[:, 2] = 0.0
    fwd = quat_apply(A.data.root_link_quat_w, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
    self.face = ((fwd * to).sum(-1) / to.norm(dim=-1).clamp(min=1e-3)).clamp(min=0.0)

    # scoring, both ways
    self.points, landed, zone, kick = self._score("red", B.data.root_link_lin_vel_w)
    self.conceded, _, _, _ = self._score("blue", A.data.root_link_lin_vel_w)
    self.since_hit = torch.where(landed, torch.zeros_like(self.since_hit), self.since_hit + dt)

    # knockdown of blue, clinch and the referee's break
    bz = b_root[:, 2] - env.scene.env_origins[:, 2]
    down = (bz < 0.45) | (B.data.projected_gravity_b[:, 2] > -0.34)
    self.knockdown = (down & ~self.blue_down & (self.since_hit < 1.0)).float()
    self.blue_down = down
    off = b_root - a_root; off[:, 2] = 0.0
    self.dist = off.norm(dim=-1)
    close = self.dist < CLINCH
    self.clinch_t = torch.where(close, self.clinch_t + dt, torch.zeros_like(self.clinch_t))
    brk = self.clinch_t > CLINCH_T
    self.broke = brk.float()
    ids = brk.nonzero().flatten()
    if len(ids):
      mdp.reset_root_state_uniform(env, ids, pose_range={"yaw": (-0.3, 0.3)}, velocity_range={})
      mdp.reset_root_state_uniform(env, ids, pose_range={}, velocity_range={}, asset_cfg=SceneEntityCfg("rival"))
      self.clinch_t[ids] = 0.0

    # effect of red's hit: blue's weight pushed beyond its feet in the half second after it lands
    lean = (b_root[:, :2] - B.data.body_link_pos_w[:, self.b_feet, :2].mean(dim=1)).norm(dim=-1)
    self.stagger = ((lean - 0.12).clamp(min=0.0) / 0.1).clamp(max=3.0) * (self.since_hit < 0.5)

    t = self.tot
    t["steps"] += 1; t["red_points"] += self.points; t["blue_points"] += self.conceded; t["kicks"] += landed * kick
    for zi, k in enumerate(ZONES):
      t[k] += landed * (zone == zi)
    t["knockdowns"] += self.knockdown; t["breaks"] += self.broke; t["clinch"] += close
    t["contacts"] += (self.impulse["red"] > 0).float(); t["impulse"] += self.impulse["red"]


@dataclass(kw_only=True)
class FightCfg(CommandTermCfg):
  attack_start: float = 0.2     # share of bouts in which blue attacks, at the start of the run
  ramp_steps: int = 800 * 24    # control steps over which that share rises to 1 (0: always attack)
  variety: bool = True

  def build(self, env):
    return Fight(self, env)


def _f(env) -> Fight:
  return env.command_manager.get_term("combo")

def opponent_obs(env): return _f(env).opp
def conceded(env): return _f(env).conceded / env.step_dt
def knockdown(env): return _f(env).knockdown / env.step_dt
def broken_up(env): return _f(env).broke / env.step_dt
def clinching(env): return (_f(env).dist < CLINCH).float()
def rival_down(env): return _f(env).blue_down
def staggered(env): return _f(env).stagger
def facing(env): return _f(env).face
def guard_up(env): return torch.exp(-_f(env).home_d / 0.15).mean(-1)
def idle(env): return (_f(env).since_hit > 6.0).float()   # six seconds without landing anything (waiting can be setting something up)
def stance_in_range(env): return combo.stance(env) * (_f(env).dist < 1.8).float()   # a good stance out of range is not fighting


def env_cfg(play: bool = False, attack_start: float = 0.2, stagger_w: float = 0.0):
  cfg = combo.env_cfg(play=play)
  for i in range(combo.K):
    cfg.scene.entities.pop(f"pad{i}")
  rival = get_g1_robot_cfg()
  rival.init_state = dataclasses.replace(rival.init_state, pos=(GAP, 0.0, rival.init_state.pos[2]), rot=(0.0, 0.0, 0.0, 1.0))
  cfg.scene.entities["rival"] = rival
  touch = dict(fields=("found", "force"), reduce="netforce", num_slots=1, history_length=4, secondary_policy="any")
  pair = {"red": ("robot", "rival"), "blue": ("rival", "robot")}
  cfg.scene.sensors = (*(s for s in cfg.scene.sensors if s.name == "self_collision"), *(
    ContactSensorCfg(name=f"{s}_{l}_{k}", primary=ContactMatch(mode="geom", pattern=geoms, entity=hit),
                     secondary=ContactMatch(mode="subtree", pattern=bodies, entity=hitter), **touch)
    for s, (hitter, hit) in pair.items() for l, (bodies, _, _, _) in LIMBS.items() for k, (_, geoms) in ZONES.items()))
  cfg.commands = {"combo": FightCfg(resampling_time_range=(1e6, 1e6), attack_start=1.0 if play else attack_start, ramp_steps=0 if play else 800 * 24)}
  for group in ("actor", "critic"):
    cfg.observations[group].terms["opponent"].func = opponent_obs
  r = cfg.rewards
  r.pop("bumped")
  r.update({
    "conceded": RewardTermCfg(func=conceded, weight=-2.0),     # red's own points are paid at 3, blue's are charged at 2
    "knockdown": RewardTermCfg(func=knockdown, weight=15.0),
    "broken_up": RewardTermCfg(func=broken_up, weight=-3.0),
    "clinching": RewardTermCfg(func=clinching, weight=-2.0),
    "idle": RewardTermCfg(func=idle, weight=-0.5),   # round 2 with a 150 N floor: three of four fighters stood still in stance and never attacked
  })
  r["stance"].func = stance_in_range
  r["facing"] = RewardTermCfg(func=facing, weight=0.3)      # round 2 circled sideways
  r["guard_up"] = RewardTermCfg(func=guard_up, weight=0.3)  # and held its arms out wide
  r["reach"].weight = 1.0
  r["staggered"] = RewardTermCfg(func=staggered, weight=stagger_w)   # the effect bonus; 0 in the impulse-only version
  r["stance"].weight = 0.5   # posture pay is kept well below what landing hits earns: at 1.5 a fighter stood still in stance and never attacked
  cfg.events["reset_base"] = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                                          params={"pose_range": {"x": (-0.1, 0.1), "y": (-0.3, 0.3), "yaw": (-0.5, 0.5)}, "velocity_range": {}})
  cfg.events["reset_rival"] = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset",
                                           params={"pose_range": {"y": (-0.3, 0.3), "yaw": (-0.5, 0.5)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("rival")})
  cfg.events["reset_rival_joints"] = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset",
    params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("rival", joint_names=(".*",))})
  cfg.terminations["rival_down"] = TerminationTermCfg(func=rival_down, time_out=True)   # a knockdown ends the round (no get-up yet)
  cfg.sim.nconmax, cfg.sim.njmax = 300, 1800
  cfg.viewer.distance, cfg.viewer.azimuth = 3.8, 90.0
  cfg.episode_length_s = 30.0 if play else 20.0
  return cfg


def rl_cfg(name: str = "g1_fight"):
  cfg = combo.rl_cfg(name)
  cfg.max_iterations = 2000
  return cfg
