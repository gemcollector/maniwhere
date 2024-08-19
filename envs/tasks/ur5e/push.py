import os
import numpy as np
import json
import collections
from dm_control.utils import rewards

from ... import _SUITE_DIR, _UR5_XML_DIR
from ...robots import UR5WithGripper
from ..base import BaseTask
from ...randomize.wrapper import RandPhysics, RandEnvironment


_CONTROL_TIMESTEP = .01  # (Seconds)
_DEFAULT_TIME_LIMIT = 5  # Default duration of an episode, in seconds.

_CONFIG_FILE_NAME = 'ur5e/push.json'

def ur5_push(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
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
    task = Push()
    environment_kwargs = environment_kwargs or {}
    return RandEnvironment(
        physics, task, config, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **environment_kwargs)

class Physics(RandPhysics):
    def reset(self, keyframe_id=None):
        super().reset(keyframe_id)
        data = self.named.data
        self._init_left_finger = data.site_xpos['left_fingertip'].copy()
        self._init_right_finger = data.site_xpos['right_fingertip'].copy()
        self._init_finger_pos = (self._init_left_finger + self._init_right_finger) / 2
        self._init_obj_pos = data.site_xpos['object_site'].copy()
    
    def object_to_target(self):
        data = self.named.data
        obj_to_target = data.site_xpos['target_site'] - data.site_xpos['object_site']
        return np.linalg.norm(obj_to_target)
    
    def init_object_to_target(self):
        data = self.named.data
        obj_to_target = data.site_xpos['target_site'] - self._init_obj_pos
        return np.linalg.norm(obj_to_target)
    
    def end_to_obj_xz(self):
        data = self.named.data
        obj_pos = data.site_xpos['object_site']
        end_pos = (data.site_xpos['left_fingertip'] + data.site_xpos['right_fingertip']) / 2
        obj_xz = np.array([obj_pos[0], 0, obj_pos[2] - 0.005])  # let finger touch more
        end_xz = np.array([end_pos[0], 0, end_pos[2]])
        return np.linalg.norm(obj_xz - end_xz)

    def init_end_to_obj_xz(self):
        init_obj_xz = np.array([self._init_obj_pos[0], 0, self._init_obj_pos[2]])
        init_finger_xz = np.array([self._init_finger_pos[0], 0, self._init_finger_pos[2]])
        return np.linalg.norm(init_obj_xz - init_finger_xz)

    def end_to_obj_y(self):
        data = self.named.data
        obj_pos = data.site_xpos['object_site']
        end_left_pos = data.site_xpos['left_fingertip']
        end_right_pos = data.site_xpos['right_fingertip']
        return np.linalg.norm(obj_pos[1] - end_left_pos[1]), \
            np.linalg.norm(obj_pos[1] - end_right_pos[1])

    def init_end_to_obj_y(self):
        data = self.named.data
        obj_pos = data.site_xpos['object_site']
        return np.linalg.norm(obj_pos[1] - self._init_left_finger[1]), \
            np.linalg.norm(obj_pos[1] - self._init_right_finger[1])

    def object_pos(self):
        data = self.named.data
        return data.site_xpos['object_site']
    
    # def gripper_caging(self, pad_success_thresh, obj_radius, xz_thread):
    #     def hamacher_product(a, b):
    #         denominator = a + b - (a * b)
    #         h_prod = ((a * b) / denominator) if denominator > 0 else 0
    #         assert 0.0 <= h_prod <= 1.0
    #         return h_prod
        
    #     data = self.named.data
    #     obj_pos = data.site_xpos['object_site']
    #     left_finger = data.site_xpos['left_fingertip']
    #     right_finger = data.site_xpos['right_fingertip']
    #     finger_pos = (left_finger + right_finger) / 2
        
    #     pad_y_lr = np.hstack(left_finger[1], right_finger[1])
    #     pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
    #     pad_to_init_obj_lr = np.abs(pad_y_lr - self._init_obj_pos[1])

    #     caging_lr_margin = np.abs(pad_to_init_obj_lr - pad_success_thresh)
    #     caging_lr = [
    #         rewards.tolerance(
    #             pad_to_obj_lr[i],
    #             bounds=(obj_radius, pad_success_thresh),
    #             margin=caging_lr_margin[i],
    #             sigmoid='long_tail'
    #         ) for i in range(2)
    #     ]
    #     caging_y = hamacher_product(*caging_lr)

    #     xz = [0, 2]
    #     caging_xz_margin = np.abs(
    #         np.linalg.norm(self._init_obj_pos[xz] - self._init_finger_pos[xz]) - xz_thread)
    #     caging_xz = rewards.tolerance(
    #         np.linalg.norm(finger_pos[xz] - obj_pos[xz]),
    #         bounds=(0, xz_thread),
    #         margin=caging_xz_margin,
    #         sigmoid='longtail'
    #     )

    #     caging = hamacher_product(caging_y, caging_xz)
    #     gripping = self.check_grasp('object_box') if caging > 0.97 else 0.0
    #     caging_and_gripping = hamacher_product(caging, gripping)

    #     caging_and_gripping = (caging_and_gripping + caging) / 2

    #     return caging_and_gripping


class Push(BaseTask):
    """A simple push task for UR5e.
    """
    def __init__(
        self,
        random=None,
        target_low=(0.6, 0.05, 0.025),
        target_high=(0.8, 0.2, 0.025),
        object_low=(0.6, -0.2, 0.026),
        object_high=(0.8, -0.05, 0.026)
    ):
        super().__init__(random)
        self.target_low = target_low
        self.target_high = target_high
        self.object_low = object_low
        self.object_high = object_high

    def initialize_episode(self, physics):
        physics.named.model.body_pos['target'] = np.random.uniform(
            low=self.target_low, high=self.target_high)
        physics.set_freejoint_pos('object_anchor', np.random.uniform(
            low=self.object_low, high=self.object_high), np.zeros(4))
        super(Push, self).initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.data.qpos[:].copy()
        obs['velocity'] = physics.data.qvel[:].copy()
        return obs
    
    def get_reward(self, physics: Physics):
        obj_pos = physics.object_pos()
        obj_to_target = physics.object_to_target()
        init_obj_to_target = physics.init_object_to_target()
        # print(init_obj_to_target)

        in_place = rewards.tolerance(
            obj_to_target,
            bounds=(0, 0.025),
            margin=init_obj_to_target,
            sigmoid='long_tail'
        )

        object_grasped = self._gripper_caging_reward(
            physics=physics,
            obj_radius=0.025
        )
        
        in_place_and_object_grasped = self._hamacher_product(
            object_grasped, in_place
        )

        # print('object_grasped: ', object_grasped, 'inplace: ', in_place, 'in_place_and_object_grasped: ', in_place_and_object_grasped)
        reward = 2 * object_grasped + 6 * in_place_and_object_grasped
        if obj_to_target < 0.025:
            reward = 10.0
        
        return reward

    def _hamacher_product(self, a, b):
        denominator = a + b - (a * b)
        h_prod = ((a * b) / denominator) if denominator > 0 else 0
        assert 0.0 <= h_prod <= 1.0
        return h_prod

    def _gripper_caging_reward(self, physics: Physics, obj_radius):
        pad_success_margin = 0.05
        grip_success_margin = obj_radius + 0.005
        x_z_success_margin = 0.01

        delta_object_y_left_pad, delta_object_y_right_pad = physics.end_to_obj_y()
        init_delta_y_left_pad, init_delta_y_right_pad = physics.init_end_to_obj_y()
        right_caging_margin = np.abs(init_delta_y_right_pad - pad_success_margin)
        left_caging_margin = np.abs(init_delta_y_left_pad - pad_success_margin)

        right_caging = rewards.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_caging = rewards.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        right_gripping = rewards.tolerance(
            delta_object_y_right_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=right_caging_margin,
            sigmoid="long_tail",
        )
        left_gripping = rewards.tolerance(
            delta_object_y_left_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=left_caging_margin,
            sigmoid="long_tail",
        )

        # print("lr caging: ", left_caging, right_caging, "lr gripping: ", left_gripping, right_gripping)

        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = self._hamacher_product(right_caging, left_caging)
        y_gripping = self._hamacher_product(right_gripping, left_gripping)

        # print("y caging: ", y_caging, "y gripping: ", y_gripping)

        assert y_caging >= 0 and y_caging <= 1

        # tcp_xz = np.array([tcp[0], 0.0, tcp[2]])
        # obj_position_x_z = np.array([obj_position[0], 0.0, obj_position[2]])
        # # print(tcp_xz, obj_position_x_z)
        # tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
        tcp_obj_norm_x_z = physics.end_to_obj_xz()
        tcp_obj_x_z_margin = np.abs(
            physics.init_end_to_obj_xz() - x_z_success_margin
        )
        x_z_caging = rewards.tolerance(
            tcp_obj_norm_x_z,
            bounds=(0, x_z_success_margin),
            margin=tcp_obj_x_z_margin,
            sigmoid="long_tail",
        )

        caging = self._hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        # print("xz caging: ", x_z_caging, "caging: ", caging)

        if caging > 0.9:
            gripping = y_gripping
        else:
            gripping = 0.0
        assert gripping >= 0 and gripping <= 1

        caging_and_gripping = (caging + gripping) / 2
        assert caging_and_gripping >= 0 and caging_and_gripping <= 1

        return caging_and_gripping