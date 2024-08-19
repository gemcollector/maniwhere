import os
import numpy as np
import collections
import json
from dm_control.utils import rewards
from dm_control.rl import control

from ... import _SUITE_DIR, _FRANKA_XML_DIR
from ...robots import FrankaWithDex, FrankaWithGripper, DualRobot, Robot
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_CONTROL_TIMESTEP = .02  # (Seconds)
_DEFAULT_TIME_LIMIT = 10  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'franka/dual_dex.json'

_INIT_POSE = {
    Robot.ControlMode.ACTUATOR: {
        'qpos': [0, -0.606, 0, -2.691, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0.298, 0, -2.256, 0, 2.60, 0.7854, 0.01026, 0.01026],
        'ctrl': [0, -0.606, 0, -2.691, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                    0, 0.298, 0, -2.256, 0, 2.60, 0.7854, 0]
    },
    Robot.ControlMode.MOCAP: {
        'qpos': [0, 0, 0, -2.05, 0, 2.08, 0.7854, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 0.5, 0, -2, -1.57, 1.57, 1.8],
        'ctrl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
}


def franka_dual_dex(time_limit=_DEFAULT_TIME_LIMIT, environment_kwargs=None):
    """Create a franka env, aiming to lift a cube.
    """
    config_path = os.path.join(_SUITE_DIR, './configs', _CONFIG_FILE_NAME)
    with open(config_path, mode='r') as f:
        config = json.load(f)

    robot = DualRobot.from_file_path(
        xml_path=config['xml'],
        asset_paths=config['assets'],
        actuator_path=os.path.join(_FRANKA_XML_DIR, 'actuator_dual.xml'), # TODO check the actuator has aligned dimension.
        mocap_path=None,
        config=config,
        init_pose=_INIT_POSE,
    )

    physics = Physics.from_rand_mjcf(robot)
    task = DualDex()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)


class Physics(RandPhysics):
    def hand_to_object(self):
        data = self.named.data
        end_to_obj = data.site_xpos['grasp_site'] - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj)


class DualDex(BaseTask):
    """A dense reward lifting task for UR5.
    """
    def __init__(
        self,
        object_low=(0.7, 0.05, 0.141),
        object_high=(0.7, 0.05, 0.141),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = np.array(object_low)
        self.object_high = np.array(object_high)
        self.delta_table_height = 0

    def initialize_episode(self, physics):
        self.delta_table_height = physics.named.data.xpos['small_table_body'][2] - 0.08

        object_low = self.object_low.copy()
        object_high = self.object_high.copy()
        object_low[2] += self.delta_table_height
        object_high[2] += self.delta_table_height

        physics.set_freejoint_pos('object_anchor', np.random.uniform(
            low=object_low, high=object_high))
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        target_height = 0.12 + self.delta_table_height + 0.15
        hand_height = physics.named.data.site_xpos['tcp_site'][2]
        gripper_height = physics.named.data.site_xpos['tcp_site_dual'][2]
        # print(hand_height, gripper_height)
        hand_to_obj = physics.hand_to_object()

        # reach
        reward = 3 * np.exp(-10 * np.clip(hand_to_obj - 0.03, 0, None))
        if hand_to_obj < 0.03:
            reward += np.exp(-10 * np.clip(abs(hand_height - target_height) - 0.03, 0, None))
            reward += np.exp(-10 * np.clip(abs(gripper_height - target_height) - 0.03, 0, None))
            reward += np.exp(-10 * np.clip(abs(hand_height - gripper_height) - 0.03, 0, None))
        
        # grasp
        joints = ['ffj1', 'mfj1', 'rfj1']
        finger_qs = []
        for j in joints:
            finger_qs.append(physics.named.data.qpos[j])
        finger_qs = np.array(finger_qs)
        if hand_to_obj < 0.03:
            # print("fingers", finger_qs)
            reward += 4 - np.linalg.norm(finger_qs - 1.0)
        else:
            reward -= 0.05 * np.sum(finger_qs)
        
        return reward
    
    # def get_termination(self, physics):
    #     if physics.hand_to_object() < 0.03:
    #         return 1.0