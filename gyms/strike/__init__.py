from mjlab.tasks.registry import register_mjlab_task

from . import strike_v0, strike_v2
from .strike import env_cfg, rl_cfg

register_mjlab_task(task_id="G1-Strike", env_cfg=env_cfg(), play_env_cfg=env_cfg(play=True), rl_cfg=rl_cfg(), runner_cls=None)

register_mjlab_task(task_id="G1-Strike-V0", env_cfg=strike_v0.env_cfg(), play_env_cfg=strike_v0.env_cfg(play=True), rl_cfg=strike_v0.rl_cfg(), runner_cls=None)

register_mjlab_task(task_id="G1-Strike-V2", env_cfg=strike_v2.env_cfg(), play_env_cfg=strike_v2.env_cfg(play=True), rl_cfg=strike_v2.rl_cfg(), runner_cls=None)

from . import kick
register_mjlab_task(task_id="G1-Kick", env_cfg=kick.env_cfg(), play_env_cfg=kick.env_cfg(play=True), rl_cfg=kick.rl_cfg(), runner_cls=None)

from . import combo
# Combo gym and the variants screened side by side: what each one switches off or changes says what it tests.
for _id, _name, _env, _rl in (
    ("G1-Combo", "g1_combo", {}, {}),
    ("G1-Combo-NoStance", "g1_combo_nostance", {"stance_w": 0.0}, {}),   # does the gym alone make it stable?
    ("G1-Combo-Heavy", "g1_combo_heavy", {"stance_w": 3.0}, {}),          # twice the stance pay
    ("G1-Combo-G99", "g1_combo_g99", {}, {"gamma": 0.99}),                # the old 2-second horizon
    ("G1-Combo-NoHist", "g1_combo_nohist", {"history": 1}, {}),           # no short memory
    ("G1-Combo-Legs", "g1_combo_legs", {"legs_low": True}, {})):          # leg-height pads only count for legs
  register_mjlab_task(task_id=_id, env_cfg=combo.env_cfg(**_env), play_env_cfg=combo.env_cfg(play=True, **_env), rl_cfg=combo.rl_cfg(_name, **_rl), runner_cls=None)
