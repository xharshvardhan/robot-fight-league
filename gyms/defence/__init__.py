from mjlab.tasks.registry import register_mjlab_task

from .defence import env_cfg, rl_cfg

register_mjlab_task(task_id="G1-Defence", env_cfg=env_cfg(), play_env_cfg=env_cfg(play=True), rl_cfg=rl_cfg(), runner_cls=None)
