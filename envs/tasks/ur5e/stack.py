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

_CONFIG_FILE_NAME = 'ur5e/stack.json'

def ur5_stack(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
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
    task = Stack()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_top(self):
        data = self.named.data
        end_to_top = data.site_xpos['top_site'] - np.array([0, 0, 0.01]) - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_top)

    def bottom_to_top_xy(self):
        data = self.named.data
        bottom_to_top = data.site_xpos['top_site'] - data.site_xpos['bottom_site']
        return np.linalg.norm(bottom_to_top[:2])
    
    def bottom_to_top_hover(self):
        data = self.named.data
        target_pos = data.site_xpos['bottom_site']
        target_pos[2] = 0.1
        bottom_to_top = data.site_xpos['top_site'] - target_pos
        return np.linalg.norm(bottom_to_top)

    def top_to_lift_z(self):
        data = self.named.data
        z = data.site_xpos['top_site'][2] - 0.1
        return np.linalg.norm(z)
    
    def site_height(self, site_name):
        data = self.named.data
        pos = data.site_xpos[site_name]
        return pos[2]

    def check_grasp(self, geom):
        pad_boxes = ['left_pad_box1', 'left_pad_box2', 'right_pad_box1', 'right_pad_box2']
        for pad in pad_boxes:
            if not self.check_contact(pad, geom):
                return False
        return True

    def check_lift(self, site, margin=0.04):
        """Successful when cube is above the table top by a margin.
            Table top is at z=0.
        """
        data = self.named.data
        height = data.site_xpos[site][2]
        return height >= margin
    
    def check_close_xy(self, radius):
        data = self.named.data
        top_to_bottom = data.site_xpos['top_site'] - data.site_xpos['bottom_site']
        dist_xy = np.linalg.norm(top_to_bottom[:2])
        return dist_xy < radius

class Stack(BaseTask):
    """A dense reward stacking task for UR5.
    """
    def __init__(
        self,
        bottom_low=(0.8, -0.08, 0.026),
        bottom_high=(0.8, -0.08, 0.026),
        top_low=(0.8, 0.08, 0.026),
        top_high=(0.8, 0.08, 0.026),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.bottom_low = bottom_low
        self.bottom_high = bottom_high
        self.top_low = top_low
        self.top_high = top_high
        self._success_cnt = 0

    def initialize_episode(self, physics):
        # sample position
        bottom_pos = np.random.uniform(low=self.bottom_low, high=self.bottom_high)
        top_pos = np.random.uniform(low=self.top_low, high=self.top_high)
        while np.linalg.norm(top_pos - bottom_pos) < 0.1:
            top_pos = np.random.uniform(low=self.top_low, high=self.top_high)
        # sample rotation
        bottom_angle = np.random.uniform(low=0, high=2*np.pi)
        bottom_quat = np.array([np.cos(bottom_angle / 2), 0, 0, np.sin(bottom_angle / 2)])
        top_angle = np.random.uniform(low=0, high=2*np.pi)
        top_quat = np.array([np.cos(top_angle / 2), 0, 0, np.sin(top_angle / 2)])
        
        physics.set_freejoint_pos('bottom_anchor', bottom_pos)
        physics.set_freejoint_pos('top_anchor', top_pos)
        super(Stack, self).initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        rewards = self._stage_reward(physics)
        num_stage = len(rewards)
        reward = 0.0
        for i, r in enumerate(reversed(rewards)):
            if r > 0.01:
                reward = num_stage - (i + 1) + r
                break
        return reward / num_stage  # rescale max reward to 1.0
    
    # def get_termination(self, physics):
    #     rewards = self._stage_reward(physics)
    #     if rewards[-1] > 0.1:
    #         self._success_cnt += 1
    #     else:
    #         self._success_cnt = 0
    #     # success if keep stacking for 10 frames
    #     if self._success_cnt >= 10:
    #         return 0.0

    def _hamacher_product(self, a, b):
        denominator = a + b - (a * b)
        h_prod = ((a * b) / denominator) if denominator > 0 else 0
        assert 0.0 <= h_prod <= 1.0
        return h_prod

    def _stage_reward(self, physics: Physics):
        # reach reward
        end_to_top = physics.end_to_top()
        reward_reach = 0.5 * (1 - np.tanh(6.0 * end_to_top))
        # grasp reward
        top_grasped = physics.check_grasp('top_box')
        if top_grasped and reward_reach > 0.4:
            reward_reach += 0.5

        # lift reward
        top_z = physics.site_height('top_site')
        reward_lift = self._lift_reward(top_z)
        
        # align reward
        hover_z = physics.top_to_lift_z()
        if reward_lift > 0.9:
            align_dist = physics.bottom_to_top_hover()
            reward_align = 1 - np.tanh(8.0 * align_dist)
            # align_dist = physics.bottom_to_top_xy()
            # print(f"{align_dist:.2f} {hover_z:.2f} {1 - np.tanh(8.0 * align_dist):.2f} {1 - np.tanh(8.0 * hover_z):.2f}")
            # reward_align = self._hamacher_product(
            #     1 - np.tanh(8.0 * align_dist), 1 - np.tanh(8.0 * hover_z))
        else:
            reward_align = 0.0

        # stack reward
        cube_contact = physics.check_contact('bottom_box', 'top_box')
        obj_close = physics.check_close_xy(0.02)
        leave_table = reward_lift > 0
        reward_stack = 1.0 if obj_close and leave_table and cube_contact else 0.0

        # leave reward
        reward_leave = reward_stack * (1 - np.tanh(hover_z))

        return reward_reach, reward_lift, reward_align, reward_stack, reward_leave
    
    def _lift_reward(self, object_z, target_z=0.1, min_z=0.03):
        if object_z >= target_z:
            return 1.0
        elif object_z <= min_z:
            return 0.0
        else:
            return (object_z - min_z) / (target_z - min_z)
    