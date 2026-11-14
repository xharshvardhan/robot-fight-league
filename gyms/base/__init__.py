from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .base import env_cfg, rl_cfg

register_mjlab_task(task_id="G1-Base", env_cfg=env_cfg(), play_env_cfg=env_cfg(play=True), rl_cfg=rl_cfg(), runner_cls=VelocityOnPolicyRunner)

register_mjlab_task(task_id="G1-Base-V2", env_cfg=env_cfg(v2=True), play_env_cfg=env_cfg(play=True, v2=True), rl_cfg=rl_cfg(v2=True), runner_cls=VelocityOnPolicyRunner)

from . import getup

register_mjlab_task(task_id="G1-Getup", env_cfg=getup.env_cfg(), play_env_cfg=getup.env_cfg(play=True), rl_cfg=getup.rl_cfg(), runner_cls=None)

register_mjlab_task(task_id="G1-Getup-V1", env_cfg=getup.env_cfg(v1=True), play_env_cfg=getup.env_cfg(play=True, v1=True), rl_cfg=getup.rl_cfg(v1=True), runner_cls=None)
