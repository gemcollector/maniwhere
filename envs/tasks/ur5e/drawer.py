import os
import numpy as np
import collections
import json
from dm_control.utils import rewards
from dm_control.rl import control

from ... import _SUITE_DIR, _UR5_XML_DIR
from ...robots import UR5WithGripper
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_CONTROL_TIMESTEP = .02  # (Seconds)
_DEFAULT_TIME_LIMIT = 10  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'ur5e/drawer.json'

def ur5_drawer(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Create a ur5e env, aiming to push a cube to a specified location.
    """
    config_path = os.path.join(_SUITE_DIR, 'configs', _CONFIG_FILE_NAME)
    with open(config_path, mode='r') as f:
        config = json.load(f)
    robot = UR5WithGripper.from_file_path(
        xml_path=config['xml'],
        asset_paths=config['assets'],
        actuator_path=os.path.join(_UR5_XML_DIR, 'actuator.xml'),
        mocap_path=os.path.join(_UR5_XML_DIR, 'mocap.xml'),
        config=config
    )
    physics = Physics.from_rand_mjcf(robot)
    task = Drawer()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_handle(self):
        data = self.named.data
        end_to_handle = (data.site_xpos['handle_site'] - np.array([0, 0, 0.03])) - data.site_xpos['tcp_site']
        end_to_handle *= np.array([3.0, 3.0, 1.0])
        return np.linalg.norm(end_to_handle)

    def handle_to_target(self):
        data = self.named.data
        end_to_target = data.site_xpos['target_site'] - data.site_xpos['handle_site']
        return np.linalg.norm(end_to_target[:2])
    
    def handle_angle(self):
        data = self.named.data
        handle_angle = float(data.qpos['handle_joint'])
        return handle_angle

class Drawer(BaseTask):
    """A dense reward drawer opening task for UR5.
    """
    def __init__(
        self,
        object_low=(0.7, -0.3, 0),
        object_high=(0.8, -0.2, 0),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = object_low
        self.object_high = object_high

    def initialize_episode(self, physics):
        # physics.set_body_pos('drawer_base', np.random.uniform(
        #     low=self.object_low, high=self.object_high))
        physics.set_body_pos('drawer_base', (0.7, -0.3, 0))
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        end_to_handle = physics.end_to_handle()
        handle_to_target = physics.handle_to_target()
        
        cage_reward = rewards.tolerance(
            end_to_handle, bounds=(0, 0.015), margin=0.3, sigmoid='long_tail')
        if cage_reward > 0.5:
            open_reward = rewards.tolerance(
                handle_to_target, bounds=(0, 0.01), margin=0.1, sigmoid='long_tail')
        else:
            open_reward = 0

        handle_penalty = physics.handle_angle()**2 * 100.0

        reward = (cage_reward + open_reward) / 2 - handle_penalty

        return reward
