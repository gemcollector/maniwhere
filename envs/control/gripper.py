import numpy as np
from dm_control.rl.control import specs

from ..randomize.wrapper import RandEnvironment

class BinaryGripperWrapper(RandEnvironment):
    """A wrapper that converts a continuous control gripper into a binary gripper.
    """
    def __init__(
        self,
        env: RandEnvironment
    ):
        self._env = env
        self._gripper_min = self._env.action_spec().minimum[6:]
        self._gripper_max = self._env.action_spec().maximum[6:]

    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        arm_action = action[:6]
        gripper_action = action[6:]
        
        thre = (self._gripper_max + self._gripper_min) / 2
        gripper_action[gripper_action < thre] = self._gripper_min[gripper_action < thre]
        gripper_action[gripper_action >= thre] = self._gripper_max[gripper_action >= thre]

        target_action = np.concatenate([arm_action, gripper_action])
        
        return self._env.step(target_action)

    def randomize(self):
        self._env.randomize()

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)