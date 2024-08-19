import os
import numpy as np
import collections
import json
from dm_control.utils import rewards
from dm_control.rl import control
import transforms3d

from ... import _SUITE_DIR, _FRANKA_XML_DIR
from ...robots import FrankaWithDex, Robot
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_CONTROL_TIMESTEP = .02  # (Seconds)
_DEFAULT_TIME_LIMIT = 10  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'franka/pour_dex.json'

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

def franka_pour_dex(time_limit=_DEFAULT_TIME_LIMIT, environment_kwargs=None):
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
    task = Pour()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    # CUP_SIZE = (0.11, 0.11, 0.15)
    CUP_SIZE = (0.092, 0.092, 0.15)
    BOWL_SIZE = (0.14, 0.14, 0.07)
    NUM_PARTICLES = 16
    PARTICLE_SHAPE = (2, 2, 4)

    def check_grasp(self):
        hand_col = ['rf_col', 'mf_col', 'ff_col', 'th_col']
        obj_col = ['cup_contact0', 'cup_contact1', 'cup_contact2', 'cup_contact3',
                   'cup_contact4', 'cup_contact5', 'cup_contact6', 'cup_contact7',
                   'cup_contact8', 'cup_contact9', 'cup_contact10', 'cup_contact11']
        return np.all([self.check_contact_group([f'{col}{i}' for i in range(5)], obj_col) for col in hand_col])
    
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

    def get_particle_pos(self):
        prefix = 'particle'
        names = [f'{prefix}B{i}_{j}_{k}' for i in range(self.PARTICLE_SHAPE[0]) 
                 for j in range(self.PARTICLE_SHAPE[1]) for k in range(self.PARTICLE_SHAPE[2])]
        particle_pos = self.named.data.xpos[names].copy()
        return particle_pos
    
    def set_particle_pos(self, pos):
        prefix = 'particle'
        spacing = 0.022
        corner = pos - (np.array(self.PARTICLE_SHAPE) / 2 - 0.5) * spacing
        for i in range(self.PARTICLE_SHAPE[0]):
            for j in range(self.PARTICLE_SHAPE[1]):
                for k in range(self.PARTICLE_SHAPE[2]):
                    name = f'{prefix}B{i}_{j}_{k}'
                    self.set_body_pos(name, corner + np.array([i, j, k]) * spacing)
    
    def check_grasp(self):
        hand_col = ['rf_col', 'mf_col', 'ff_col', 'th_col']
        obj_col = ['cup_contact0', 'cup_contact1', 'cup_contact2', 'cup_contact3',
                   'cup_contact4', 'cup_contact5', 'cup_contact6', 'cup_contact7',
                   'cup_contact8', 'cup_contact9', 'cup_contact10', 'cup_contact11']
        return np.all([self.check_contact_group([f'{col}{i}' for i in range(5)], obj_col) for col in hand_col])

    def check_table_collision(self):
        table_col = ['small_table']
        arm_col = ['link6_col', 'link7_col']
        # hand_col = ['rf_col', 'mf_col', 'ff_col', 'th_col']
        # hand_col = [f'{col}{i}' for i in range(5) for col in hand_col]
        return self.check_contact_group(arm_col, table_col)

    def check_in_mug_particles(self):
        def inverse_pose(pose: np.ndarray):
            inv_pose = np.eye(4, dtype=pose.dtype)
            inv_pose[:3, :3] = pose[:3, :3].T
            inv_pose[:3, 3] = -pose[:3, :3].T @ pose[:3, 3]
            return inv_pose
        particle_pos = self.get_particle_pos()
        pose_mug = np.eye(4)
        pose_mug[:3, 3] = self.named.data.site_xpos['object_site'].ravel()
        pose_mug[:3, :3] = transforms3d.quaternions.quat2mat(self.named.data.xquat['object'].ravel())
        pose_mug_inv = inverse_pose(pose_mug)
        particle_pos_mug = particle_pos @ pose_mug_inv[:3, :3].T + pose_mug_inv[:3, 3]
        size = np.array(Physics.CUP_SIZE) / 2
        # Margin z size for 0.05 since particles may out of mug
        within_mug = np.logical_and.reduce([particle_pos_mug[:, 0] < size[0], particle_pos_mug[:, 0] > -size[0],
                                            particle_pos_mug[:, 1] < size[1], particle_pos_mug[:, 1] > -size[1],
                                            particle_pos_mug[:, 2] < size[2] + 0.05, particle_pos_mug[:, 2] > -size[2]])
        return within_mug

    def check_above_particle(self, margin=0.01):
        particle_pos = self.get_particle_pos()
        bowl_pos_xy = self.named.data.site_xpos['bowl_site'][:2]
        upper_limit = (np.array(self.BOWL_SIZE)[:2] / 2 + bowl_pos_xy)
        lower_limit = (-np.array(self.BOWL_SIZE)[:2] / 2 + bowl_pos_xy)
        x_within = np.logical_and(particle_pos[:, 0] < upper_limit[0] + margin,
                                  particle_pos[:, 0] > lower_limit[0] - margin)
        y_within = np.logical_and(particle_pos[:, 1] < upper_limit[1] + margin,
                                  particle_pos[:, 1] > lower_limit[1] - margin)
        z_within = particle_pos[:, 2] > 0
        xy_within = np.logical_and(x_within, y_within)
        tank_above = np.logical_and(z_within, xy_within)
        return tank_above
    
    def check_success_particles(self):
        # Assume that the water particles are the last body in MuJoCo model
        particle_pos = self.get_particle_pos()
        bowl_pos = self.named.data.site_xpos['bowl_site']
        upper_limit = np.array(self.BOWL_SIZE) / 2 + bowl_pos
        lower_limit = -np.array(self.BOWL_SIZE) / 2 + bowl_pos
        x_within = np.logical_and(particle_pos[:, 0] < upper_limit[0], particle_pos[:, 0] > lower_limit[0])
        y_within = np.logical_and(particle_pos[:, 1] < upper_limit[1], particle_pos[:, 1] > lower_limit[1])
        z_within = np.logical_and(particle_pos[:, 2] < upper_limit[2], particle_pos[:, 2] > lower_limit[2])
        xy_within = np.logical_and(x_within, y_within)
        tank_within = np.logical_and(z_within, xy_within)
        return tank_within

class Pour(BaseTask):
    """A dense reward lifting task for UR5.
    """
    def __init__(
        self,
        bowl_low=(0.85, -0.18, 0.171),
        bowl_high=(0.9, -0.13, 0.171),
        cup_low=(0.85, 0.1, 0.201),
        cup_high=(0.95, 0.15, 0.201),
        random=None,
        action_delay=0
    ):
        super().__init__(random, action_delay)
        self.bowl_low = np.array(bowl_low)
        self.bowl_high = np.array(bowl_high)
        self.cup_low = np.array(cup_low)
        self.cup_high = np.array(cup_high)
        self.delta_table_height = 0

    def initialize_episode(self, physics):
        table_pos = np.array([0.15, 0, 0.1])
        self.delta_table_height = np.random.uniform(-0.01, 0.01)
        table_pos[2] += self.delta_table_height
        
        bowl_low = self.bowl_low.copy()
        bowl_high = self.bowl_high.copy()
        bowl_low[2] += self.delta_table_height
        bowl_high[2] += self.delta_table_height
        
        physics.set_body_pos('small_table_body', table_pos)
        physics.set_freejoint_pos('bowl_anchor', np.random.uniform(
            low=bowl_low, high=bowl_high))
        cup_pos = np.random.uniform(low=self.cup_low, high=self.cup_high)
        physics.set_freejoint_pos('object_anchor', cup_pos)
        physics.set_particle_pos(cup_pos)
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
        obj_pos = physics.named.data.site_xpos['object_site'].ravel()
        palm_pos = physics.named.data.site_xpos['tcp_site'].ravel()
        target_pos = physics.named.data.site_xpos['target_site'].ravel()
        end_to_obj = np.linalg.norm(palm_pos - obj_pos)
        is_contact = end_to_obj < 0.065
        all_contact = physics.check_grasp()

        # Punish particles out of mug
        out_of_mug_bool = ~physics.check_in_mug_particles()
        out_of_container_bool = ~physics.check_above_particle()
        dropping_bool = np.logical_and(out_of_mug_bool, out_of_container_bool)
        dropping_num = np.sum(dropping_bool)
        dropping_ratio = dropping_num / Physics.NUM_PARTICLES

        # Relocation Reward
        reward = -0.1 * end_to_obj  # take hand to object
        if is_contact:
            reward += 0.1
            if all_contact:
                reward += 0.2
            lift = max(min(obj_pos[2], target_pos[2]) - physics.CUP_SIZE[2] / 2.0 - (0.1 + self.delta_table_height), 0)
            reward += 50 * lift
            condition = lift > 0.06
            if condition:  # if object off the table
                obj_target_distance = np.linalg.norm(obj_pos - target_pos)
                reward += 2.0  # bonus for lifting the object
                reward += -0.5 * np.linalg.norm(palm_pos - target_pos)  # make hand go to target
                reward += -1.5 * obj_target_distance  # make object go to target

                if obj_target_distance < 0.05:
                    reward += 1 / (max(obj_target_distance, 0.03))
                    if obj_target_distance < 0.05:
                        obj_quat = physics.named.data.xquat['object'].ravel()
                        z_axis = transforms3d.quaternions.quat2mat(obj_quat) @ np.array([0, 0, 1])
                        reward += -(z_axis[0] + z_axis[1]) * 20 - abs(z_axis[0] - z_axis[1]) * 10
                        if z_axis[0] < 0 and z_axis[1] < 0:
                            reward += np.arccos(z_axis[2]) * 100
                    
                    reward += np.sum(physics.check_success_particles()) * 100 / physics.NUM_PARTICLES

        # punish water dropping
        reward -= 0.1 * dropping_ratio

        # collision penalty
        if physics.check_table_collision():
            reward -= 0.05 * abs(reward)

        # rescale
        max_reward = 0.3 + 50 * (target_pos[2] - physics.CUP_SIZE[2] / 2.0 - (0.1 + self.delta_table_height)) + 2.0 + 1 / 0.03 + 20 * 1.4142 + 100 + 100
        reward /= max_reward
        
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
