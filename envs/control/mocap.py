import numpy as np
from dm_control.rl.control import specs

from ..randomize.wrapper import RandEnvironment
from .utils import euler2quat, quat2euler

class MocapCtrlWrapper(RandEnvironment):
    """A wrapper for controlling robot arm in operational space,
        actuated by mocap body in mujoco.
    """
    def __init__(
        self,
        env: RandEnvironment,
        action_min=-np.ones(6),
        action_max=np.ones(6),
        tcp_min=np.array([0, -0.6, 0]),
        tcp_max=np.array([1.2, 0.6, 1.0]),
        action_scale=0.002,
        mocap_name="mocap"
    ):
        
        action_min = np.asarray(action_min, dtype=np.float32)
        action_max = np.asarray(action_max, dtype=np.float32)
        assert action_min.shape == (6,) and action_max.shape == (6,)

        self._env = env
        self._env.physics.enable_mocap()

        wrapped_action_spec = self._env.action_spec()
        action_min = np.append(action_min, wrapped_action_spec.minimum)
        action_max = np.append(action_max, wrapped_action_spec.maximum)
        self._action_spec = specs.BoundedArray(
            shape=action_min.shape,
            dtype=np.float32,
            minimum=action_min,
            maximum=action_max,
            name='action'
        )
        self._tcp_min = tcp_min
        self._tcp_max = tcp_max
        self._action_scale = action_scale
        self._mocap_name = mocap_name

    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        arm_action = self._rescale_action(action)[:6]
        gripper_action = action[6:]
        
        cur_tcp_pos = self._env.physics.named.data.mocap_pos[self._mocap_name]
        cur_tcp_quat = self._env.physics.named.data.mocap_quat[self._mocap_name]
        cur_tcp_euler = quat2euler(cur_tcp_quat)
        
        target_pos = cur_tcp_pos + arm_action[:3]
        target_euler = cur_tcp_euler + arm_action[3:6]
        target_quat = euler2quat(target_euler)
        
        clipped_target_pos = np.clip(target_pos, self._tcp_min, self._tcp_max)
        self._env.physics.named.data.mocap_pos[self._mocap_name] = clipped_target_pos
        self._env.physics.named.data.mocap_quat[self._mocap_name] = target_quat
        
        return self._env.step(gripper_action)

    def _rescale_action(self, action):
        """Rescale action to [-action_scale, action_scale]
        """
        minimum = self._action_spec.minimum
        maximum = self._action_spec.maximum
        scale = 2.0 * self._action_scale * np.ones_like(action) / (maximum - minimum)
        return -self._action_scale + (action - minimum) * scale

    def randomize(self):
        self._env.randomize()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)