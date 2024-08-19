import os
import numpy as np
import collections
import json
from dm_control.utils import rewards
from dm_control.rl import control

from ... import _SUITE_DIR, _XARM_XML_DIR
from ...robots import XArm6WithDex, Robot
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_INIT_POSE = {
    Robot.ControlMode.ACTUATOR: {
        'qpos': [0, 0, -1.29, 0, 1.26, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ctrl': [0, 0, -1.29, 0, 1.26, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    Robot.ControlMode.MOCAP: {
        'qpos': [0, 0, -1.29, 0, 1.26, -1.5708, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ctrl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
}

_CONTROL_TIMESTEP = .02  # (Seconds)
_DEFAULT_TIME_LIMIT = 10  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'xarm/close_dex.json'

def xarm_close_dex(time_limit=_DEFAULT_TIME_LIMIT, environment_kwargs=None):
    """Create a xarm env, aiming to do a closing computer task.
    """
    config_path = os.path.join(_SUITE_DIR, './configs', _CONFIG_FILE_NAME)
    with open(config_path, mode='r') as f:
        config = json.load(f)
    robot = XArm6WithDex.from_file_path(
        xml_path=config['xml'],
        asset_paths=config['assets'],
        actuator_path=os.path.join(_XARM_XML_DIR, 'actuator_dex.xml'),
        mocap_path=None,
        # mocap_path=os.path.join(_XARM_XML_DIR, 'mocap_dex.xml'),
        config=config,
        init_pose=_INIT_POSE
    )
    physics = Physics.from_rand_mjcf(robot)
    task = Close()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_object(self):
        data = self.named.data
        end_to_obj = data.site_xpos['laptop_site'] - data.site_xpos['fingertip_site']
        return np.linalg.norm(end_to_obj)
    
    def hand_to_object(self):
        data = self.named.data
        hand_to_obj = data.site_xpos['laptop_hand_site'] - data.site_xpos['tcp_site']
        return np.linalg.norm(hand_to_obj)

    def laptop_angle(self):
        """1.57 is the angle when the laptop is closed.
        """
        data = self.named.data
        angle = data.qpos['laptop_joint']
        return angle.item()

class Close(BaseTask):
    """A dense reward lifting task for UR5.
    """
    def __init__(
        self,
        object_low=(0.0, -0.125, 0.0),
        object_high=(0.0, -0.075, 0.0),
        angle_low=-0.15,
        angle_high=-0.05,
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = np.array(object_low)
        self.object_high = np.array(object_high)
        self.angle_low = angle_low
        self.angle_high = angle_high

    def initialize_episode(self, physics):
        self.delta_table_height = physics.named.data.xpos['table'][2] - 0.0
        
        object_low = self.object_low.copy()
        object_high = self.object_high.copy()
        object_low[2] += self.delta_table_height
        object_high[2] += self.delta_table_height
        self.init_angle = np.random.uniform(low=self.angle_low, high=self.angle_high)

        physics.set_body_pos('laptop', np.random.uniform(
            low=object_low, high=object_high))
        physics.set_joint_pos('laptop_joint', self.init_angle)

        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        end_to_obj = physics.end_to_object()
        hand_to_obj = physics.hand_to_object()
        angle = physics.laptop_angle()

        reward = np.exp(-10 * np.clip(end_to_obj-0.03, 0, None))
        if angle < 0.349:  # 20 degree
            reward += 2 * np.exp(-10 * np.clip(hand_to_obj-0.03, 0, None))
        else:
            reward += 2
        
        reward += 3 * (physics.laptop_angle() - self.init_angle)

        if abs(1.57 - physics.laptop_angle()) < 0.05:
            reward += 10

        return reward

    # def get_termination(self, physics):
    #     if abs(1.3 - physics.laptop_angle()) < 0.05:
    #         return 1.0 
