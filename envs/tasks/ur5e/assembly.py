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
_DEFAULT_TIME_LIMIT = 20  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'ur5e/assembly.json'

def ur5_assembly(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Create a ur5e env, aiming to reach a specified point.
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
    task = Assembly()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def reset(self, keyframe_id=None):
        super().reset(keyframe_id)
        data = self.named.data
        self._init_finger_pos = (data.site_xpos['right_fingertip'] + data.site_xpos['left_fingertip']) / 2
        self._init_nut_pos = data.site_xpos['nut_site']

    def end_to_handle(self):
        data = self.named.data
        handle_pos = data.site_xpos['nut_handle_site'] - np.array([0, 0, 0.01])
        finger_pos = (data.site_xpos['right_fingertip'] + data.site_xpos['left_fingertip']) / 2
        end_to_handle = handle_pos - finger_pos
        return np.linalg.norm(end_to_handle), np.linalg.norm(end_to_handle[:2])

    def nut_to_target_xy_z(self):
        data = self.named.data
        nut_to_target = data.site_xpos['nut_site'] - data.site_xpos['target_site']
        return np.linalg.norm(nut_to_target[:2]), np.linalg.norm(nut_to_target[2])

    def finger_to_init_z(self):
        data = self.named.data
        finger_pos = (data.site_xpos['right_fingertip'] + data.site_xpos['left_fingertip']) / 2
        return np.linalg.norm(finger_pos[2] - self._init_finger_pos[2])

    def nut_to_init_z(self):
        data = self.named.data
        nut_pos = data.site_xpos['nut_site']
        return np.linalg.norm(nut_pos[2] - self._init_nut_pos[2])

    def max_placing_dist(self):
        data = self.named.data
        return np.linalg.norm(self._init_nut_pos - data.site_xpos['target_site']) + 0.15

    def nut_center_pos(self):
        data = self.named.data
        return data.site_xpos['nut_site']

    def check_contact(self, geom1, geom2):
        """Successful when all pad box on gripper contact target object
        """
        contacts = self.data.contact
        for i in range(self.data.ncon):
            contact1 = self.model.id2name(contacts[i].geom1, 'geom')
            contact2 = self.model.id2name(contacts[i].geom2, 'geom')
            if (contact1 == geom1 and contact2 == geom2) or \
                (contact1 == geom2 and contact2 == geom1):
                return True
        return False

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

class Assembly(BaseTask):
    """An assembly task for UR5.
    """
    def __init__(
        self,
        nut_low=(0.7, -0.1, 0.021),
        nut_high=(0.75, -0.08, 0.021),
        target_low=(0.7, 0.08, 0.0),
        target_high=(0.75, 0.1, 0.0),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.target_low = target_low
        self.target_high = target_high
        self.nut_low = nut_low
        self.nut_high = nut_high

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.
        """
        nut_angle = np.random.uniform(low=-0.05*np.pi, high=0.05*np.pi)
        nut_quat = np.array([np.cos(nut_angle / 2), 0, 0, np.sin(nut_angle / 2)])
        physics.set_freejoint_pos('nut_anchor', np.random.uniform(
            low=self.nut_low, high=self.nut_high), nut_quat)
        physics.set_body_pos('peg', np.random.uniform(
            low=self.target_low, high=self.target_high))
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs

    def get_reward(self, physics):
        reward_reach, reward_pick, reward_place = self._stage_reward(physics)
        reward = reward_reach + reward_pick + reward_place
        return reward
    
    def get_termination(self, physics):
        nut_to_target_xy, nut_to_target_z = physics.nut_to_target_xy_z()
        if nut_to_target_xy < 0.03 and nut_to_target_z < 0.03:
            return 0.0

    def _stage_reward(self, physics: Physics):
        nut_pos = physics.nut_center_pos()

        # reach reward
        reach_dist, reach_dist_xy = physics.end_to_handle()
        if reach_dist_xy < 0.04:
            reward_reach = -reach_dist
        else:
            reward_reach = -reach_dist_xy - physics.finger_to_init_z()
        # encourage finger close when dist is small
        if reach_dist < 0.04:
            reward_reach = -reach_dist + 0.02 * max(self.current_action[-1], 0) / 255

        # pick reward
        lift_margin = 0.15
        grasped = physics.check_grasp('nut_8')
        lifted = physics.check_lift('nut_site', lift_margin)
        nut_to_target_xy, nut_to_target_z = physics.nut_to_target_xy_z()
        if nut_to_target_xy < 0.03 or grasped and lifted:
            reward_pick = 100 * lift_margin
        elif reach_dist < 0.04 and physics.nut_to_init_z() > 0.005:
            reward_pick = 100 * min(nut_pos[2], lift_margin)
        else:
            reward_pick = 0
        
        # place reward
        if nut_to_target_xy < 0.03 or grasped and lifted and reach_dist < 0.04:
            c1, c2, c3 = 1000, 0.01, 0.001
            reward_place = 1000 * (physics.max_placing_dist() - nut_to_target_xy) + c1 * (
                np.exp(-(nut_to_target_xy**2) / c2) + np.exp(-(nut_to_target_xy**2) / c3)
            )
            if nut_to_target_xy < 0.03:
                c4, c5, c6 = 2000, 0.003, 0.0003
                reward_place += 2000 * (lift_margin - nut_to_target_z) + c4 * (
                    np.exp(-(nut_to_target_z**2) / c5) + np.exp(-(nut_to_target_z**2) / c6)
                )
            reward_place = max(reward_place, 0)
        else:
            reward_place = 0

        return reward_reach, reward_pick, reward_place