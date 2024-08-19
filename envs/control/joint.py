import numpy as np
from dm_control.rl.control import specs
from numpy.core.numeric import ones as ones

from ..randomize.wrapper import RandEnvironment
from .utils import euler2quat, quat2euler

class JointCtrlWrapper(RandEnvironment):
    """A wrapper for controlling robot arm directly in joint space,
        actuated by actuators.
    """
    def __init__(
        self,
        env: RandEnvironment,
        action_min=-np.ones(6),
        action_max=np.ones(6),
        joint_clip_min=None,
        joint_clip_max=None,
        action_scale=0.04,
        moving_average=1,
    ):
        action_min = np.asarray(action_min, dtype=np.float32)
        action_max = np.asarray(action_max, dtype=np.float32)
        assert action_min.shape == (env.arm_joint_num,) and action_max.shape == (env.arm_joint_num,)

        self._env = env
        self._env.physics.enable_actuator()

        wrapped_action_spec = self._env.action_spec()
        action_min = np.append(action_min, wrapped_action_spec.minimum[env.arm_joint_num:])
        action_max = np.append(action_max, wrapped_action_spec.maximum[env.arm_joint_num:])

        if joint_clip_min is None:
            joint_clip_min = [None] * wrapped_action_spec.shape[0]
        if joint_clip_max is None:
            joint_clip_max = [None] * wrapped_action_spec.shape[0]
        self._joint_clip_min = np.array([new_min or orig_min for new_min, orig_min in zip(joint_clip_min, wrapped_action_spec.minimum)])
        self._joint_clip_max = np.array([new_max or orig_max for new_max, orig_max in zip(joint_clip_max, wrapped_action_spec.maximum)])

        self._action_spec = specs.BoundedArray(
            shape=action_min.shape,
            dtype=np.float32,
            minimum=action_min,
            maximum=action_max,
            name='action'
        )
        self._action_scale = action_scale
        self._moving_average = moving_average
        self._prev_gripper_action = np.zeros(self._env.gripper_joint_num)

    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        gripper_joint_num = self._env.physics.robot.gripper_joint_num
        arm_joint_num = self._env.physics.robot.arm_joint_num
        
        arm_action = self._rescale_action(action)[:arm_joint_num]
        # gripper_action = action[arm_joint_num:]
        gripper_action = action[arm_joint_num:] * self._moving_average +  self._prev_gripper_action * (1 - self._moving_average)
        
        joint_names = self._env.physics.robot.arm_joint_names
        cur_qpos = self._env.physics.named.data.qpos[joint_names]
        target_qpos = cur_qpos + arm_action

        target_action = np.concatenate([target_qpos, gripper_action])
        
        self._prev_gripper_action = gripper_action.copy()

        # import pdb; pdb.set_trace()

        target_action = np.clip(target_action, self._joint_clip_min, self._joint_clip_max)
        return self._env.step(target_action)

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
    
    def get_ctrl_qpos(self):
        return np.float32(self._env.physics.data.qpos[-self._env.gripper_joint_num-self._env.arm_joint_num:])
    

# class DualJointCtrlWrapper(JointCtrlWrapper):
#     def __init__(self, env: RandEnvironment, action_min=-np.ones(6), action_max=np.ones(6), action_scale=0.04, moving_average=1, dual_action_min=-np.ones(6), dual_action_max=np.ones(6)):
#         # super().__init__(env, action_min, action_max, action_scale, moving_average)
#         action_min = np.asarray(action_min, dtype=np.float32)
#         action_max = np.asarray(action_max, dtype=np.float32)
#         assert action_min.shape == (env.arm_joint_num,) and action_max.shape == (env.arm_joint_num,)

#         self._env = env
#         self._env.physics.enable_actuator()

#         wrapped_action_spec = self._env.action_spec()
#         # TODO deal with the dual gripper num
#         action_min = np.append(np.append(action_min, wrapped_action_spec.minimum[env.arm_joint_num:(env.arm_joint_num+env.gripper_joint_num)]), dual_action_min)
#         action_max = np.append(np.append(action_max, wrapped_action_spec.maximum[env.arm_joint_num:(env.arm_joint_num+env.gripper_joint_num)]), dual_action_max)
#         action_min = np.append(action_min, wrapped_action_spec.minimum[env.arm_joint_num+env.gripper_joint_num+env.dual_arm_joint_num:])
#         action_max = np.append(action_max, wrapped_action_spec.maximum[env.arm_joint_num+env.gripper_joint_num+env.dual_arm_joint_num:])
#         self._action_spec = specs.BoundedArray(
#             shape=action_min.shape,
#             dtype=np.float32,
#             minimum=action_min,
#             maximum=action_max,
#             name='action'
#         )
#         self._action_scale = action_scale
#         self._moving_average = moving_average
#         self._prev_gripper_action = np.zeros(self._env.gripper_joint_num)

    
#     def step(self, action):
#         gripper_joint_num = self._env.gripper_joint_num
#         arm_joint_num = self._env.arm_joint_num
#         dual_arm_joint_num = self._env.dual_arm_joint_num
#         dual_gripper_joint_num = self._env.dual_gripper_joint_num

#         arm_action = self._rescale_action(action)[:arm_joint_num]
#         another_arm_action = self._rescale_action(action)[arm_joint_num+gripper_joint_num:arm_joint_num+gripper_joint_num+dual_arm_joint_num]
#         # another_arm_action = self._rescale_action(action)[-dual_arm_joint_num:-dual_gripper_joint_num]
#         gripper_action = action[arm_joint_num:arm_joint_num+gripper_joint_num]
        
#         cur_qpos = self._env.physics.data.qpos[-gripper_joint_num-arm_joint_num-dual_arm_joint_num-dual_gripper_joint_num:-gripper_joint_num-dual_arm_joint_num-dual_gripper_joint_num]
#         target_qpos = cur_qpos + arm_action
#         dual_cur_qpos = self._env.physics.data.qpos[-dual_arm_joint_num-dual_gripper_joint_num:-dual_gripper_joint_num]
#         target_dual_qpos = dual_cur_qpos + another_arm_action
#         dual_gripper_action = action[(arm_joint_num + gripper_joint_num + dual_arm_joint_num):]
#         target_action = np.concatenate([target_qpos, gripper_action, target_dual_qpos, dual_gripper_action])
#         self._prev_gripper_action = gripper_action.copy()

#         return self._env.step(target_action)


class DualJointCtrlWrapper(JointCtrlWrapper):
    def __init__(self, env: RandEnvironment, action_min=-np.ones(6), action_max=np.ones(6), joint_clip_min=None, joint_clip_max=None, action_scale=0.04, moving_average=1, dual_action_min=-np.ones(6), dual_action_max=np.ones(6)):
        super().__init__(env, action_min, action_max, joint_clip_min, joint_clip_max, action_scale, moving_average)
        
    def get_ctrl_qpos(self):
        joint_names = self._env.physics.robot.arm_joint_names
        gripper_names = self._env.physics.robot.gripper_joint_names
        joint_qpos = self._env.physics.named.data.qpos[joint_names]
        gripper_qpos = self._env.physics.named.data.qpos[gripper_names][:16]
        return np.float32(np.concatenate([joint_qpos, gripper_qpos], axis=0))



class OriginalJointCtrlWrapper(JointCtrlWrapper):
    def __init__(self, env: RandEnvironment, action_min=-np.ones(6), action_max=np.ones(6), joint_clip_min=None, joint_clip_max=None, action_scale=0.025, moving_average=1):
        super().__init__(env, action_min, action_max, joint_clip_min, joint_clip_max, action_scale, moving_average)
        
    def step(self, action):
        gripper_joint_num = self._env.gripper_joint_num
        arm_joint_num = self._env.arm_joint_num

        arm_action = self._rescale_action(action)[:arm_joint_num]
        gripper_action = action[arm_joint_num:]

        cur_qpos = self._env.physics.data.qpos[-gripper_joint_num-arm_joint_num:-gripper_joint_num]
        target_qpos = cur_qpos + arm_action

        target_action = np.concatenate([target_qpos, gripper_action])

        return self._env.step(target_action)


class ArmJointCtrlWrapper(JointCtrlWrapper):
    
    def __init__(self, env: RandEnvironment, action_min=-np.ones(6), action_max=np.ones(6), joint_clip_min=None, joint_clip_max=None, action_scale=0.04, moving_average=1):
        super().__init__(env, action_min, action_max, joint_clip_min, joint_clip_max, action_scale, moving_average)
        
    def get_ctrl_qpos(self):
        joint_names = self._env.physics.robot.arm_joint_names
        joint_qpos = self._env.physics.named.data.qpos[joint_names]
        return np.float32(joint_qpos)
    
    
    
class DualOpenPickJointCtrlWrapper(RandEnvironment):
    """A wrapper for controlling robot arm directly in joint space,
        actuated by actuators.
    """
    def __init__(
        self,
        env: RandEnvironment,
        action_min=-np.ones(6),
        action_max=np.ones(6),
        joint_clip_min=None,
        joint_clip_max=None,
        action_scale=0.025,
        moving_average=1,
    ):
        action_min = np.asarray(action_min, dtype=np.float32)
        action_max = np.asarray(action_max, dtype=np.float32)
        assert action_min.shape == (env.arm_joint_num,) and action_max.shape == (env.arm_joint_num,)

        self._env = env
        self._env.physics.enable_actuator()

        wrapped_action_spec = self._env.action_spec()
        action_min = np.append(action_min, wrapped_action_spec.minimum[env.arm_joint_num:])
        action_max = np.append(action_max, wrapped_action_spec.maximum[env.arm_joint_num:])

        if joint_clip_min is None:
            joint_clip_min = [None] * wrapped_action_spec.shape[0]
        if joint_clip_max is None:
            joint_clip_max = [None] * wrapped_action_spec.shape[0]
        self._joint_clip_min = np.array([new_min or orig_min for new_min, orig_min in zip(joint_clip_min, wrapped_action_spec.minimum)])
        self._joint_clip_max = np.array([new_max or orig_max for new_max, orig_max in zip(joint_clip_max, wrapped_action_spec.maximum)])

        self._action_spec = specs.BoundedArray(
            shape=action_min.shape,
            dtype=np.float32,
            minimum=action_min,
            maximum=action_max,
            name='action'
        )
        self._action_scale = action_scale
        self._moving_average = moving_average
        self._prev_gripper_action = np.zeros(self._env.gripper_joint_num)

    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        gripper_joint_num = self._env.physics.robot.gripper_joint_num
        arm_joint_num = self._env.physics.robot.arm_joint_num
        
        arm_action = self._rescale_action(action)[:arm_joint_num]
        # gripper_action = action[arm_joint_num:]
        gripper_action = action[arm_joint_num:] * self._moving_average +  self._prev_gripper_action * (1 - self._moving_average)
        
        joint_names = self._env.physics.robot.arm_joint_names
        cur_qpos = self._env.physics.named.data.qpos[joint_names]
        target_qpos = cur_qpos + arm_action

        target_action = np.concatenate([target_qpos, gripper_action])
        
        self._prev_gripper_action = gripper_action.copy()

        target_action = np.clip(target_action, self._joint_clip_min, self._joint_clip_max)
        return self._env.step(target_action)

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
    
    def get_ctrl_qpos(self):
        return np.float32(self._env.physics.data.qpos[-self._env.gripper_joint_num-self._env.arm_joint_num:])
    
    
    
    

class OldJointCtrlWrapper(RandEnvironment):
    """A wrapper for controlling robot arm directly in joint space,
        actuated by actuators.
    """
    def __init__(
        self,
        env: RandEnvironment,
        action_min=-np.ones(6),
        action_max=np.ones(6),
        action_scale=0.025
    ):
        action_min = np.asarray(action_min, dtype=np.float32)
        action_max = np.asarray(action_max, dtype=np.float32)
        assert action_min.shape == (env.arm_joint_num,) and action_max.shape == (env.arm_joint_num,)

        self._env = env
        self._env.physics.enable_actuator()

        wrapped_action_spec = self._env.action_spec()
        action_min = np.append(action_min, wrapped_action_spec.minimum[env.arm_joint_num:])
        action_max = np.append(action_max, wrapped_action_spec.maximum[env.arm_joint_num:])
        self._action_spec = specs.BoundedArray(
            shape=action_min.shape,
            dtype=np.float32,
            minimum=action_min,
            maximum=action_max,
            name='action'
        )
        self._action_scale = action_scale

    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        gripper_joint_num = self._env.gripper_joint_num
        arm_joint_num = self._env.arm_joint_num
        
        arm_action = self._rescale_action(action)[:arm_joint_num]
        gripper_action = action[arm_joint_num:]
        
        cur_qpos = self._env.physics.data.qpos[-gripper_joint_num-arm_joint_num:-gripper_joint_num]
        target_qpos = cur_qpos + arm_action

        target_action = np.concatenate([target_qpos, gripper_action])
        
        return self._env.step(target_action)

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