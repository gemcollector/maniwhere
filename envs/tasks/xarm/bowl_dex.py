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


_CONTROL_TIMESTEP = .02  # (Seconds)
_DEFAULT_TIME_LIMIT = 10  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'xarm/bowl_dex.json'

def xarm_bowl_dex(time_limit=_DEFAULT_TIME_LIMIT, environment_kwargs=None):
    """Create a xarm env, aiming to do a pick-place task.
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
        config=config
    )
    physics = Physics.from_rand_mjcf(robot)
    task = Bowl()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_object(self):
        data = self.named.data
        end_to_obj = (data.site_xpos['object_site'] + np.array([0, 0, 0.08])) - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj)

    def end_to_object_xy(self):
        data = self.named.data
        end_to_obj = (data.site_xpos['object_site'] + np.array([0, 0, 0.08])) - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj[:2])

    def check_lift(self, site, margin=0.04):
        """Successful when cube is above the table top by a margin.
            Table top is at z=0.
        """
        data = self.named.data
        height = data.site_xpos[site][2]
        return height >= margin
    
    def object_to_bowl_xy(self):
        data = self.named.data
        bottom_to_top = data.site_xpos['object_site'] - data.site_xpos['bowl_site']
        return np.linalg.norm(bottom_to_top[:2])
    
    def object_to_bowl(self):
        data = self.named.data
        top_to_bottom = data.site_xpos['object_site'] - data.site_xpos['bowl_site']
        return np.linalg.norm(top_to_bottom[:3])

    def get_tip_dists(self):
        tip_names = ['rf_tip_site', 'mf_tip_site', 'ff_tip_site', 'th_tip_site']
        tip_dists = []
        obj_pos = self.named.data.site_xpos['object_site']
        for tip in tip_names:
            pos = self.named.data.site_xpos[tip]
            tip_dists.append(np.linalg.norm(pos - obj_pos))
        return tip_dists

class Bowl(BaseTask):
    """A dense reward lifting task for UR5.
    """
    def __init__(
        self,
        object_low=(0.15, 0, 0.051),
        object_high=(0.15, 0, 0.051),
        bowl_low=(-0.15, 0, 0.061),
        bowl_high=(-0.15, 0, 0.061),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = np.array(object_low)
        self.object_high = np.array(object_high)
        self.bowl_low = np.array(bowl_low)
        self.bowl_high = np.array(bowl_high)

    def initialize_episode(self, physics):
        self.delta_table_height = physics.named.data.xpos['table'][2] - 0.0
        
        object_low = self.object_low.copy()
        object_high = self.object_high.copy()
        bowl_low = self.bowl_low.copy()
        bowl_high = self.bowl_high.copy()
        object_low[2] += self.delta_table_height
        object_high[2] += self.delta_table_height
        bowl_low[2] += self.delta_table_height
        bowl_high[2] += self.delta_table_height

        physics.set_freejoint_pos('object_anchor', np.random.uniform(
            low=object_low, high=object_high))
        physics.set_freejoint_pos('bowl_anchor', np.random.uniform(
            low=bowl_low, high=bowl_high))
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        # print("named_qpos", physics.named.data.qpos)
        # print("named_xpos", physics.named.data.xpos)
        # print("quat", physics.named.data.xquat['tcp_center'])
        return obs

    def get_reward(self, physics):
        # reach reward
        end_to_obj = physics.end_to_object()
        # end_to_obj_xy = physics.end_to_object_xy()
        
        reward = np.exp(-10 * np.clip(end_to_obj-0.03, 0, None))

        # lift reward_lift
        obj_height = physics.get_site_xpos('object_site')[2]
        min_z = 0.05 + self.delta_table_height + 0.02
        target_z = 0.05 + self.delta_table_height + 0.08
        # print(obj_height)
        # ifend_to_obj < 0.03: 
        #     tip_dists = physics.get_tip_dists()
        #     tip_dists = np.mean([d for d in tip_dists])
        #     reward_lift += np.exp(-20 * np.clip(tip_dists-0.03, 0, None))
        if end_to_obj < 0.1: 
            reward += 100 * self._lift_reward(obj_height, target_z=target_z, min_z=min_z)
        
        reward -= 3e-4 * np.linalg.norm(physics.data.qvel.ravel())
        # joints = ['ffj1', 'mfj1', 'rfj1']
        # print(physics.named.data.qpos)
        # finger_qs = []
        # for j in joints:
        #     finger_qs.append(np.linalg.norm(physics.named.data.qpos[j]))
        # finger_qs = np.array(finger_qs)
        # reward_lift -= 5e-1 * np.sum(finger_qs)  # open reward_lift
        # reward_lift -= 5e-1 * np.linalg.norm((finger_qs - np.mean(finger_qs)))  # align reward_lift

        if physics.check_lift('object_site', margin=target_z) and end_to_obj < 0.1:
            reward += 5

        align_dist = physics.object_to_bowl_xy()
        # align reward
        if physics.check_lift('object_site', margin=target_z) and end_to_obj < 0.1:
            reward += 100 * np.exp(align_dist * -10)
        else:
            # NOTE: maybe not necessary here
            if align_dist < 0.03 and end_to_obj < 0.1:
                reward += 100

        # stack reward
        # cube_contact = physics.check_contact('object_box', 'box_bot')
        if align_dist < 0.03 and end_to_obj < 0.1:
            dist_object_to_bowl = physics.object_to_bowl()
            reward += 500 * np.exp(np.clip(dist_object_to_bowl, 0, None) * -5) + 100

        # reward /= 20

        return reward
    
    def _lift_reward(self, object_z, target_z=0.3, min_z=0.25):
        if object_z >= target_z:
            return 1.0
        elif object_z <= min_z:
            return 0.0
        else:
            return (object_z - min_z) / (target_z - min_z)

    # def get_termination(self, physics):
    #     end_to_obj = physics.end_to_object()
    #     obj_pose = physics.get_site_xpos('object_site')
    #     init_obj_pose = np.array([0.77, 0.22, 0.15])
    #     if end_to_obj > 0.1 and np.linalg.norm(obj_pose - init_obj_pose) > 0.05:
    #         return 0.0