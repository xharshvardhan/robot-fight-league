from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .fighter import MODES, env_cfg, rl_cfg
from .runners import WIDE, CapRunner, DistilRunner, GrowCapRunner, GrowRunner

reg = lambda task, env, rl, runner=VelocityOnPolicyRunner, **kw: register_mjlab_task(
  task_id=task, env_cfg=env(**kw), play_env_cfg=env(play=True, **kw), rl_cfg=rl, runner_cls=runner)

reg("G1-Fighter", env_cfg, rl_cfg())                                              # control: all four skills from scratch
reg("G1-Fighter-Grow", env_cfg, rl_cfg("g1_fighter_grow", WIDE), GrowRunner)     # arm 1: widened body network, old weights protected
reg("G1-Fighter-Distil", env_cfg, rl_cfg("g1_fighter_distil"), DistilRunner, expert=True)  # arm 2a: copy the four specialists
reg("G1-Fighter-Tune", env_cfg, rl_cfg("g1_fighter_distil"), expert=True)       # arm 2b: then ordinary training
# v2 of both arms: walking episodes reward staying up, and exploration noise is capped
reg("G1-Fighter-Grow2", env_cfg, rl_cfg("g1_fighter_grow", WIDE), GrowCapRunner, walk_fix=True)
reg("G1-Fighter-Tune2", env_cfg, rl_cfg("g1_fighter_distil"), CapRunner, expert=True, walk_fix=True)
for m in MODES:  # playback of one skill at a time, for either network size
  reg(f"G1-Fighter-{m.capitalize()}", env_cfg, rl_cfg(), mode=m)
  reg(f"G1-Fighter-Grow-{m.capitalize()}", env_cfg, rl_cfg("g1_fighter_grow", WIDE), mode=m)
