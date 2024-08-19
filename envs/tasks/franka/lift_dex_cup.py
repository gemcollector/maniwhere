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

_CONFIG_FILE_NAME = 'franka/lift_dex_cup.json'

_INIT_POSE = {
    Robot.ControlMode.ACTUATOR: {
        'qpos': [0.6, 0.5, 0, -2, -1.57, 1.57, 1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ctrl': [0.6, 0.5, 0, -2, -1.57, 1.57, 1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    Robot.ControlMode.MOCAP: {
        'qpos': [0.6, 0.5, 0, -2, -1.57, 1.57, 1.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ctrl': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
}

def franka_lift_dex_cup(time_limit=_DEFAULT_TIME_LIMIT, environment_kwargs=None):
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
        config=config,
        init_pose=_INIT_POSE
    )
    physics = Physics.from_rand_mjcf(robot)
    task = Lift()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def end_to_object(self):
        data = self.named.data
        end_to_obj = (data.site_xpos['object_site'] + np.array([0, 0, 0.06])) - data.site_xpos['tcp_site']
        return np.linalg.norm(end_to_obj)

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

    def check_grasp(self):
        hand_col = ['rf_col', 'mf_col', 'ff_col', 'th_col']
        obj_col = ['cup_contact0', 'cup_contact1', 'cup_contact2', 'cup_contact3',
                   'cup_contact4', 'cup_contact5', 'cup_contact6', 'cup_contact7',
                   'cup_contact8', 'cup_contact9', 'cup_contact10', 'cup_contact11']
        return np.all([self.check_contact_group([f'{col}{i}' for i in range(5)], obj_col) for col in hand_col])

    def check_lift(self, site, margin=0.04):
        """Successful when cube is above the table top by a margin.
            Table top is at z=0.
        """
        data = self.named.data
        height = data.site_xpos[site][2]
        return height >= margin
    
    def get_tip_dists(self):
        tip_names = ['rf_tip_site', 'mf_tip_site', 'ff_tip_site']
        tip_dists = []
        obj_pos = self.named.data.site_xpos['object_site']
        for tip in tip_names:
            pos = self.named.data.site_xpos[tip]
            if pos[2] > obj_pos[2] + 0.05:  # ignore tip above object top
                tip_dists.append(np.inf)
            else:
                tip_dists.append(np.linalg.norm(pos - obj_pos))
        return tip_dists

    def check_table_collision(self):
        table_col = ['small_table']
        hand_col = ['rf_col', 'mf_col', 'ff_col', 'th_col']
        hand_col = [f'{col}{i}' for i in range(5) for col in hand_col]
        return self.check_contact_group(hand_col, table_col)

class Lift(BaseTask):
    """A dense reward lifting task for UR5.
    """
    def __init__(
        self,
        object_low=(0.75, -0.1, 0.181),
        object_high=(0.95, 0.1, 0.181),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.object_low = np.array(object_low)
        self.object_high = np.array(object_high)
        self.delta_table_height = 0

    def initialize_episode(self, physics):
        table_pos = np.array([0.15, 0, 0.1])
        self.delta_table_height = np.random.uniform(-0.01, 0.01)
        table_pos[2] += self.delta_table_height
        
        object_low = self.object_low.copy()
        object_high = self.object_high.copy()
        object_low[2] += self.delta_table_height
        object_high[2] += self.delta_table_height
        
        physics.set_body_pos('small_table_body', table_pos)
        physics.set_freejoint_pos('object_anchor', np.random.uniform(
            low=object_low, high=object_high))
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
        # reward = 1 - np.tanh(10.0 * end_to_obj)
        reward_reach = rewards.tolerance(end_to_obj, bounds=(0, 0.03), margin=0.12)
        tip_dists = physics.get_tip_dists()
        reward_reach += np.mean([rewards.tolerance(d, bounds=(0, 0.06), margin=0.1) for d in tip_dists])
        
        # lift reward
        obj_height = physics.get_site_xpos('object_site')[2]
        min_z = 0.18 + self.delta_table_height
        target_z = 0.26 + self.delta_table_height
        reward_lift = 8 * self._lift_reward(
            obj_height, target_z=target_z, min_z=min_z)

        if physics.check_grasp():
            reward_grasp = 3
        else:
            reward_grasp = 0

        reward_penalty = 1e-3 * np.linalg.norm(physics.data.qvel.ravel())
        if physics.check_table_collision():
            reward_penalty += 3

        # reward scale
        reward = reward_reach + reward_lift + reward_grasp - reward_penalty
        reward /= 13

        # print(np.mean([rewards.tolerance(d, bounds=(0, 0.06), margin=0.1) for d in tip_dists]), reward_grasp, reward_penalty, reward)

        return reward
    
    def _lift_reward(self, object_z, target_z=0.3, min_z=0.25):
        if object_z >= target_z:
            return 1.0
        elif object_z <= min_z:
            return 0.0
        else:
            return (object_z - min_z) / (target_z - min_z)

    # def get_termination(self, physics):
    #     if physics.check_lift('object_site', margin=0.04):
    #         return 0.0
