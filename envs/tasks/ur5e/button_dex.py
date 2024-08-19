import os
import numpy as np
import json
import collections
from dm_control.utils import rewards

from ... import _SUITE_DIR, _UR5_XML_DIR
from ...robots import UR5WithDex
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_CONTROL_TIMESTEP = .02  # (Seconds)
_DEFAULT_TIME_LIMIT = 10  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'ur5e/button_dex.json'

def ur5_button_dex(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Create a ur5e env, aiming to reach a specified point.
    """
    config_path = os.path.join(_SUITE_DIR, 'configs', _CONFIG_FILE_NAME)
    with open(config_path, mode='r') as f:
        config = json.load(f)
    robot = UR5WithDex.from_file_path(
        xml_path=config['xml'],
        asset_paths=config['assets'],
        actuator_path=os.path.join(_UR5_XML_DIR, 'actuator_dex.xml'),
        mocap_path=os.path.join(_UR5_XML_DIR, 'mocap_dex.xml'),
        config=config
    )
    physics = Physics.from_rand_mjcf(robot)
    task = ButtonDex()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_hover_xy(self):
        data = self.named.data
        end_to_hover = data.site_xpos['hover_site'] - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_hover[:2])
    
    def btn_to_pushdown(self):
        data = self.named.data
        btn_to_pushdown = data.site_xpos['pushdown_site'] - data.site_xpos['btn_site']
        return np.linalg.norm(btn_to_pushdown)

class ButtonDex(BaseTask):
    """A simple reach task for UR5.
    """
    def __init__(
        self,
        target_low=(0.6, -0.15, 0.058),
        target_high=(0.8, 0.15, 0.058),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.target_low = target_low
        self.target_high = target_high

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        """
        physics.set_body_pos('buttonbox', np.random.uniform(
            low=self.target_low, high=self.target_high))
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        end_to_hover_xy = physics.end_to_hover_xy()
        btn_to_pushdown = physics.btn_to_pushdown()

        hover_reward = rewards.tolerance(end_to_hover_xy, bounds=(0, 0.05), margin=0.1)

        if hover_reward > 0.5:
            pushdown_reward = 2 * rewards.tolerance(btn_to_pushdown, bounds=(0, 0.005), margin=0.1)
        else:
            pushdown_reward = 0

        reward = hover_reward + pushdown_reward

        return reward / 3