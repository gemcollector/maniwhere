import os
import numpy as np
import collections
import json
from dm_control.utils import rewards
from dm_control.rl import control

from ... import _SUITE_DIR, _FRANKA_XML_DIR
from ...robots import FrankaWithDex, Robot
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_CONTROL_TIMESTEP = .02  # (Seconds)
_DEFAULT_TIME_LIMIT = 10  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'franka/bowl_dex.json'

def franka_bowl_dex(time_limit=_DEFAULT_TIME_LIMIT, environment_kwargs=None):
    """Create a franka env, aiming to lift a dragon.
    """
    config_path = os.path.join(_SUITE_DIR, './configs', _CONFIG_FILE_NAME)
    with open(config_path, mode='r') as f:
        config = json.load(f)
    robot = FrankaWithDex.from_file_path(
        xml_path=config['xml'],
        asset_paths=config['assets'],
        actuator_path=os.path.join(_FRANKA_XML_DIR, 'actuator_dex.xml'),
        mocap_path=os.path.join(_FRANKA_XML_DIR, 'mocap_dex.xml'),
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
        # end_to_obj = (data.site_xpos['object_site'] + np.array([0, 0, 0.12])) - data.site_xpos['tcp_site']
        end_to_obj = (data.site_xpos['object_site'] + np.array([0, 0, 0.04])) - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj)

    def end_to_object_xy(self):
        data = self.named.data
        # end_to_obj = (data.site_xpos['object_site'] + np.array([0, 0, 0.12])) - data.site_xpos['tcp_site']
        end_to_obj = (data.site_xpos['object_site'] + np.array([0, 0, 0.04])) - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj[:2])

    def check_lift(self, site, margin=0.04):
        """Successful when cube is above the table top by a margin.
            Table top is at z=0.
        """
        data = self.named.data
        height = data.site_xpos[site][2]
        return height >= margin
    
    def bottom_to_top_hover(self, hover_z):
        data = self.named.data
        target_pos = data.site_xpos['bowl_site'].copy()
        target_pos[2] = hover_z
        bottom_to_top = data.site_xpos['object_site'] - target_pos
        return np.linalg.norm(bottom_to_top[:2])
    
    def bottom_to_top_hover_xy(self):
        data = self.named.data
        target_pos = data.site_xpos['bowl_site'].copy()
        bottom_to_top = data.site_xpos['object_site'] - target_pos
        return np.linalg.norm(bottom_to_top[:2])

    def check_close_xy(self, radius):
        data = self.named.data
        top_to_bottom = data.site_xpos['object_site'] - data.site_xpos['bowl_site']
        dist_xy = np.linalg.norm(top_to_bottom[:2])
        return dist_xy < radius
    
    def object_to_bowl_xy(self):
        data = self.named.data
        top_to_bottom = data.site_xpos['object_site'] - data.site_xpos['bowl_site']
        return np.linalg.norm(top_to_bottom[:2])
    
    def object_to_bowl(self):
        data = self.named.data
        top_to_bottom = data.site_xpos['object_site'] - data.site_xpos['bowl_site']
        return np.linalg.norm(top_to_bottom)
    
    def end_to_lift_z(self, hover_z):
        data = self.named.data
        z = data.site_xpos['tcp_site'][2] - hover_z
        return np.linalg.norm(z)
    
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
        object_low=(0.75, 0.1, 0.18),
        object_high=(0.75, 0.1, 0.18),
        bowl_low=(0.75, -0.2, 0.16),
        bowl_high=(0.75, -0.2, 0.16),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = np.array(object_low)
        self.object_high = np.array(object_high)
        self.bowl_low = np.array(bowl_low)
        self.bowl_high = np.array(bowl_high)

    def initialize_episode(self, physics):
        table_pos = np.array([0.15, 0, 0.1])
        self.delta_table_height = 0
        table_pos[2] += self.delta_table_height
        
        object_low = self.object_low.copy()
        object_high = self.object_high.copy()
        bowl_low = self.bowl_low.copy()
        bowl_high = self.bowl_high.copy()
        object_low[2] += self.delta_table_height
        object_high[2] += self.delta_table_height
        bowl_low[2] += self.delta_table_height
        bowl_high[2] += self.delta_table_height

        physics.set_body_pos('small_table_body', table_pos)
        physics.set_freejoint_pos('object_anchor', np.random.uniform(
            low=object_low, high=object_high))
        physics.set_freejoint_pos('bowl_anchor', np.random.uniform(
            low=bowl_low, high=bowl_high))
        super().initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['tcp_pos'] = physics.named.data.site_xpos['tcp_site'].copy()
        obs['obj_pos'] = physics.named.data.site_xpos['object_site'].copy()
        obs['bowl_pos'] = physics.named.data.site_xpos['bowl_site'].copy()
        obs['arm_joint_qpos'] = physics.named.data.qpos[physics.robot.arm_joint_names].copy()
        obs['hand_joint_qpos'] = physics.named.data.qpos[physics.robot.gripper_joint_names].copy()
        # obs['position'] = physics.data.qpos[:].copy()
        # obs['velocity'] = physics.data.qvel[:].copy()
        # obs['end_to_object'] = physics.end_to_object()[None].copy()
        # obs['lift'] = np.clip(physics.end_to_object().copy()-0.03, 0, None)[None].copy()
        # obs['obj_height'] = physics.get_site_xpos('object_site')[2][None].copy()
        # obs['obj_to_bowl'] = physics.object_to_bowl()[None].copy()
        # joints = ['ffj1', 'mfj1', 'rfj1']
        # finger_qs = []
        # for j in joints:
        #     finger_qs.append(np.linalg.norm(physics.named.data.qpos[j]))
        # obs['finger_qs'] = np.array(finger_qs)
        # obs['align_dist'] = physics.bottom_to_top_hover_xy()[None].copy()
        # obs['finger_qs'] = np.array(finger_qs)
        # obs['bottom_to_top'] = physics.data.site_xpos['object_site']
        # print("named_qpos", physics.named.data.qpos)
        # print("named_xpos", physics.named.data.xpos)
        # print("quat", physics.named.data.xquat['tcp_center'])
        return obs

    # def get_reward(self, physics):
    #     # reach reward
    #     end_to_obj = physics.end_to_object()
    #     # reward_lift = 1 - np.tanh(10.0 * end_to_obj)
    #     reward_lift = rewards.tolerance(end_to_obj, bounds=(0, 0.03), margin=0.12)

    #     # lift reward_lift
    #     obj_height = physics.get_site_xpos('object_site')[2]
    #     min_z = 0.19 + self.delta_table_height + 0.03
    #     target_z = 0.19 + self.delta_table_height + 0.08
    #     reward_lift += 6 * self._lift_reward(obj_height, target_z=target_z, min_z=min_z)
        
    #     # print("height", obj_height, physics.get_site_xpos('bowl_site')[2])
    #     # tip_dists = physics.get_tip_dists()
    #     # reward_lift += np.mean([rewards.tolerance(d, bounds=(0, 0.05), margin=0.1) for d in tip_dists])
        
    #     reward_lift -= 1e-3 * np.linalg.norm(physics.data.qvel.ravel())
    #     joints = ['ffj1', 'mfj1', 'rfj1']
    #     # print(physics.named.data.qpos)
    #     finger_qs = []
    #     for j in joints:
    #         finger_qs.append(np.linalg.norm(physics.named.data.qpos[j]))
    #     finger_qs = np.array(finger_qs)
    #     reward_lift -= 5e-2 * np.sum(finger_qs)  # open reward_lift
    #     reward_lift -= 5e-2 * np.linalg.norm((finger_qs - np.mean(finger_qs)))  # align reward_lift

    #     if physics.check_lift('object_site', margin=target_z):
    #         reward_lift = 8
        
    #     # align reward
    #     if physics.check_lift('object_site', margin=target_z):
    #         align_dist = physics.bottom_to_top_hover(target_z + 0.02)
    #         reward_align = 2 * (1 - np.tanh(6.0 * align_dist))
    #     else:
    #         reward_align = 0.0

    #     # stack reward
    #     # cube_contact = physics.check_contact_group(
    #     #     ['apple_collision0', 'apple_collision1', 'apple_collision2',
    #     #      'apple_collision3', 'apple_collision4', 'apple_collision5',
    #     #      'apple_collision6', 'apple_collision7', 'apple_collision8'],
    #     #     ['bowl_contact0', 'bowl_contact1', 'bowl_contact2', 'bowl_contact3'
    #     #      'bowl_contact4', 'bowl_contact5', 'bowl_contact6', 'bowl_contact7',
    #     #      'bowl_contact8', 'bowl_contact9', 'bowl_contact10', 'bowl_contact11']
    #     # )
    #     cube_contact = physics.check_contact_group(['object_box'], ['box_bot'])
    #     # obj_close = physics.check_close_xy(0.06)
    #     # leave_table = not physics.check_contact('apple_contact1', 'small_table')
    #     leave_table = not physics.check_contact('object_box', 'small_table')
    #     if leave_table and cube_contact:
    #         align_xy = physics.object_to_bowl_xy()
    #         reward_stack = 10 * rewards.tolerance(align_xy, bounds=(0, 0.04), margin=0.1)
    #     else:
    #         reward_stack = 0
    #     # reward_stack = 10.0 if obj_close and leave_table and cube_contact else 0.0

    #     # leave reward
    #     if reward_stack > 1:
    #         hover_z = physics.end_to_lift_z(target_z)
    #         reward_stack += 1 - np.tanh(hover_z)
    #     # else:
    #     #     reward_leave = 0.0

    #     reward = 0
    #     # if reward_leave > 0.01:
    #     #     reward = reward_leave + 18
    #     if reward_stack > 0.01:
    #         reward = reward_stack + 10
    #     elif reward_align > 0.01:
    #         reward = reward_align + 8
    #     else:
    #         reward = reward_lift
    #     # for i, stage_reward in enumerate((reversed[reward_lift, reward_align, reward_stack, reward_leave])):
    #     #     if stage_reward > 0.01:
    #     #         reward = stage_reward
    #     #         break
    #     # reward scale
    #     reward /= 20

    #     # print("rewards", [f'{a:.3f}' for a in [reward_lift, reward_align, reward_stack, reward]])

    #     # reward = (reward_lift + reward_align + reward_stack + reward_leave) / 18

    #     return reward

    def get_reward(self, physics):
        # reach reward
        end_to_obj = physics.end_to_object()
        end_to_obj_xy = physics.end_to_object_xy()
        
        # reward_lift = 1 - np.tanh(10.0 * end_to_obj)
        reward_lift = np.exp(-10 * np.clip(end_to_obj-0.03, 0, None))

        # lift reward_lift
        obj_height = physics.get_site_xpos('object_site')[2]
        min_z = 0.15 + self.delta_table_height + 0.02
        target_z = 0.15 + self.delta_table_height + 0.08
        # print(obj_height)
        # ifend_to_obj < 0.03: 
        #     tip_dists = physics.get_tip_dists()
        #     tip_dists = np.mean([d for d in tip_dists])
        #     reward_lift += np.exp(-20 * np.clip(tip_dists-0.03, 0, None))
        if end_to_obj < 0.1: 
            reward_lift += 100 * self._lift_reward(obj_height, target_z=target_z, min_z=min_z)
        
        # print("height", obj_height, physics.get_site_xpos('bowl_site')[2])
        # tip_dists = physics.get_tip_dists()
        # reward_lift += np.mean([rewards.tolerance(d, bounds=(0, 0.05), margin=0.1) for d in tip_dists])
        
        # reward_lift -= 3e-4 * np.linalg.norm(physics.data.qvel.ravel())
        # print(physics.named.data.qpos)
        if end_to_obj > 0.15:
            joints = ['ffj1', 'mfj1', 'rfj1']
            finger_qs = []
            for j in joints:
                finger_qs.append(np.linalg.norm(physics.named.data.qpos[j]))
            finger_qs = np.array(finger_qs)
            reward_lift -= 10e-1 * np.sum(finger_qs)  # open reward_lift
        # reward_lift -= 5e-1 * np.linalg.norm((finger_qs - np.mean(finger_qs)))  # align reward_lift

        if physics.check_lift('object_site', margin=target_z) and end_to_obj < 0.1:
            reward_lift += 5

        align_dist = physics.bottom_to_top_hover(target_z)
        # align reward
        if physics.check_lift('object_site', margin=target_z) and end_to_obj < 0.1:
            reward_lift += 100 * np.exp(align_dist * -10)
        else:
            # reward_lift + 0.
            # NOTE: maybe not necessary here
            if align_dist < 0.03 and end_to_obj < 0.1:
                reward_lift += 100

        # stack reward
        cube_contact = physics.check_contact_group(['object_box'], ['box_bot'])
        # # obj_close = physics.check_close_xy(0.06)
        # # leave_table = not physics.check_contact('apple_contact1', 'small_table')
        # leave_table = not physics.check_contact('object_box', 'small_table')
        # if leave_table and cube_contact:
        #     align_xy = physics.object_to_bowl_xy()
        #     reward_lift += 500 * rewards.tolerance(align_xy, bounds=(0, 0.04), margin=0.1)
        # else:
        #     reward_lift += 0
        # reward_stack = 10.0 if obj_close and leave_table and cube_contact else 0.0

        # leave reward
        if align_dist < 0.05 and end_to_obj < 0.15:
            dist_object_to_bowl = physics.object_to_bowl()
            reward_lift += 1000 * np.exp(np.clip(dist_object_to_bowl, 0, None) * -10) + 100
        # else:
        #     reward_leave = 0.0

        # if reward_leave > 0.01:
        #     reward = reward_leave + 18
        reward = reward_lift
        # for i, stage_reward in enumerate((reversed[reward_lift, reward_align, reward_stack, reward_leave])):
        #     if stage_reward > 0.01:
        #         reward = stage_reward
        #         break
        # reward scale
        # reward /= 806

        # print("rewards", [f'{a:.3f}' for a in [reward_lift, reward_align, reward_stack, reward]])

        # reward = (reward_lift + reward_align + reward_stack + reward_leave) / 18

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
    
    # def get_termination(self, physics):
    #     dist_object_to_bowl = physics.object_to_bowl()
    #     if dist_object_to_bowl < 0.04:
    #         return 1.0