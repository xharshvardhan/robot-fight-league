from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from src.tasks.fighter.fighter import rl_cfg as fighter_rl_cfg
from src.tasks.strike.strike_v2 import rl_cfg
from . import bout
from .spar import env_cfg

for task, kw in (("G1-Spar-Test", {}), ("G1-Spar-Idle", {"punch": False})):
  register_mjlab_task(task_id=task, env_cfg=env_cfg(**kw), play_env_cfg=env_cfg(play=True, **kw), rl_cfg=rl_cfg(), runner_cls=VelocityOnPolicyRunner)
register_mjlab_task(task_id="G1-Bout", env_cfg=bout.env_cfg(), play_env_cfg=bout.env_cfg(play=True), rl_cfg=fighter_rl_cfg(), runner_cls=VelocityOnPolicyRunner)

from src.tasks.fighter.runners import CapRunner
from . import sparring
_rl = fighter_rl_cfg("g1_sparring"); _rl.max_iterations = 1500


class SparRunner(CapRunner):
  """Sparring starts from the copied network, whose value estimate has never seen fight scores. With ordinary training
  the first updates knocked the fighter over within two seconds and it never recovered. So: a slow fixed learning speed,
  little noise, and for the first WARMUP updates only the value estimate learns."""
  std_cap = 0.2
  WARMUP = 50

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    import torch
    update, self.updates = self.alg.update, 0

    def warm_update():
      actor = [p.detach().clone() for p in self.alg.actor.parameters()]
      out = update()
      self.updates += 1
      if self.updates <= self.WARMUP:
        with torch.no_grad():
          for p, b in zip(self.alg.actor.parameters(), actor):
            p.copy_(b)
      return out

    self.alg.update = warm_update


_rl.algorithm.schedule, _rl.algorithm.learning_rate = "fixed", 1e-4
register_mjlab_task(task_id="G1-Sparring", env_cfg=sparring.env_cfg(), play_env_cfg=sparring.env_cfg(play=True), rl_cfg=_rl, runner_cls=SparRunner)

from . import fight


class FightRunner(SparRunner):
  """Carries a combo-gym network into the fight gym: the value estimate gets 30 updates to itself first."""
  std_cap = 0.6
  WARMUP = 30


register_mjlab_task(task_id="G1-Fight", env_cfg=fight.env_cfg(), play_env_cfg=fight.env_cfg(play=True), rl_cfg=fight.rl_cfg(), runner_cls=FightRunner)
# Self-play rounds: blue attacks from the first step.
register_mjlab_task(task_id="G1-Fight-Full", env_cfg=fight.env_cfg(attack_start=1.0), play_env_cfg=fight.env_cfg(play=True), rl_cfg=fight.rl_cfg("g1_fight_full"), runner_cls=FightRunner)


class SelfPlayRunner(FightRunner):
  """Blue is a copy of red, refreshed every REFRESH updates; some bouts are against older snapshots."""
  REFRESH = 20

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    update, self.rounds = self.alg.update, 0
    term = self.env.unwrapped.command_manager.get_term("combo")

    def selfplay_update():
      out = update()
      self.rounds += 1
      if self.rounds % self.REFRESH == 0:
        term.set_rivals(self.alg.actor.state_dict())
      return out

    self.alg.update = selfplay_update


register_mjlab_task(task_id="G1-SelfPlay", env_cfg=fight.env_cfg(attack_start=1.0), play_env_cfg=fight.env_cfg(play=True), rl_cfg=fight.rl_cfg("g1_selfplay"), runner_cls=SelfPlayRunner)
register_mjlab_task(task_id="G1-SelfPlay-Stagger", env_cfg=fight.env_cfg(attack_start=1.0, stagger_w=4.0), play_env_cfg=fight.env_cfg(play=True, stagger_w=4.0), rl_cfg=fight.rl_cfg("g1_selfplay_stagger"), runner_cls=SelfPlayRunner)
