# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
from dm_control.utils import rewards
import re
from envs.tasks import *
from envs.control import MocapCtrlWrapper, JointCtrlWrapper, JointIKCtrlWrapper, DofWrapper, BinaryGripperWrapper, DualJointCtrlWrapper, OriginalJointCtrlWrapper, ArmJointCtrlWrapper, DualOpenPickJointCtrlWrapper, OldJointCtrlWrapper
import torch
import copy
import utils
import cv2
import torch.nn.functional as F


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)
        

class StateExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    state: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)
        


class AugExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    aug_observation: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)



class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)



class StateFrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        state_num = 0
        for key in env.observation_spec().keys():
            state_num += env.observation_spec()[key].shape[0]
        print('state_num:', state_num)
        self._obs_spec = specs.BoundedArray(shape=np.array([num_frames * state_num]),
                                            dtype=np.float32,
                                            minimum=-np.inf,
                                            maximum=np.inf,
                                            name='observation')

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)


    def _extract_state(self, time_step):
        
        for i, key in enumerate(time_step.observation.keys()):
            if i == 0:
                state = time_step.observation[key]
            else:
                state = np.concatenate([state, time_step.observation[key]], axis=0)
        return np.float32(state.copy())


    def reset(self):
        time_step = self._env.reset()
        state = self._extract_state(time_step)
        for _ in range(self._num_frames):
            self._frames.append(state)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        state = self._extract_state(time_step)
        self._frames.append(state)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)

class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env, randomize):
        self._env = env
        self._randomize = randomize
        
    def transform(self, action):
        scale =  np.array([6.28319,  6.28319, 3.1415 , 6.28319, 6.28319, 6.28319, 127.5])
        orig_minimum = np.array([-6.28319, -6.28319, -3.1415 , -6.28319, -6.28319, -6.28319,0.])
        minimum = np.array([-1.])
        new_action = orig_minimum + scale * (action - minimum)
        return new_action.astype(self._env.action_spec.dtype, copy=False)

    def reset(self):
        if self._randomize:
            self._env.randomize()
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class StateExtendedTimeStepWrapper(ExtendedTimeStepWrapper):
    
    def __init__(self, env, randomize):
        super().__init__(env, randomize)
    
    def _augment_time_step(self, time_step, action=None, state=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return StateExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0,
                                state=state)
    
    def reset(self):
        if self._randomize:
            self._env.randomize()
        time_step = self._env.reset()
        state = np.concatenate([self._env._state_frames], axis=0)
        # import ipdb;ipdb.set_trace()
        return self._augment_time_step(time_step, state=state)
        
    def step(self, action):
        time_step = self._env.step(action)
        state = np.concatenate([self._env._state_frames], axis=0)
        return self._augment_time_step(time_step, action, state)


class CameraViewWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, cam_id, height=84, width=84, depth=False):
        self._env = env

        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)

        self._height = height
        self._width = width
        self._cam_id = cam_id
        self._depth = depth
        self.channel = 3
        
        extra_channel = 1 if self._depth else 0

        self._observation_spec = specs.BoundedArray(
            shape=np.array([self.channel * num_frames + extra_channel, height, width]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        
    def _get_pixels(self, **render_kwargs):
        pixels = self._env.physics.render(**render_kwargs)
        return pixels.transpose(2, 0, 1).copy()
    
    def _add_depth_noise(self, depth, depth_dependent_noise=True, gaussion_noise_scale=0.01, depth_noise_scale=0.05):
        gaussion_noise = np.random.normal(0, gaussion_noise_scale, depth.shape)
        
        if depth_dependent_noise:
            depth_scale = depth_noise_scale * np.abs(depth)
            depth_noise = np.random.normal(0, depth_scale, depth.shape)
            noisy_depth = depth + gaussion_noise + depth_noise
        else:
            noisy_depth = depth + gaussion_noise
        
        noisy_depth = cv2.GaussianBlur(noisy_depth, (7, 7), 1)

        return noisy_depth

    def _get_depth(self, **render_kwargs):
        depth = self._env.physics.render(**render_kwargs)
        depth = self._add_depth_noise(depth)
        depth_max = 2
        depth[depth >= depth_max] = depth_max
        # depth[depth >= depth_max] = depth.max()
        depth = 255 * (depth - depth.min()) / (depth.max() - depth.min())
        depth = np.clip(depth, 0, 255).astype(np.uint8)
        return depth[None].copy()
    
    def _get_obs(self):
        obs = self._get_pixels(height=self._height, width=self._width, camera_id=self._cam_id)
        return obs

    def _transform_obs(self, time_step):
        if isinstance(time_step.observation, np.ndarray):
            obs = np.concatenate([time_step.observation, *self._frames], axis=0)
        else:
            obs = np.concatenate(list(self._frames), axis=0)
        if self._depth:
            depth_obs = self._get_depth(height=self._height, width=self._width, camera_id=self._cam_id, depth=True)
            obs = np.concatenate([obs, depth_obs], axis=0)
        return time_step._replace(observation=obs)
    
    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs()
        for _ in range(self._num_frames):
            self._frames.append(obs)
        return self._transform_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._get_obs()
        self._frames.append(obs)
        return self._transform_obs(time_step)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
    
    
class StateCameraViewWrapper(CameraViewWrapper):
    def __init__(self, env, num_frames, cam_id, height=84, width=84, depth=False):
        super().__init__(env, num_frames, cam_id, height, width, depth)
        self._state_frames = deque([], maxlen=num_frames)
        self._state_spec = specs.BoundedArray(
            shape=np.array([num_frames, env.state_num[0]]),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name='state'
        )
    
        
    def _get_state(self):
        state = self._env.get_ctrl_qpos()
        return state


    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs()
        state = self._get_state()
        for _ in range(self._num_frames):
            self._frames.append(obs)
            self._state_frames.append(state)
        return self._transform_obs(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        obs = self._get_obs()
        state = self._get_state()
        self._frames.append(obs)
        self._state_frames.append(state)
        return self._transform_obs(time_step)

    def state_spec(self):
        return self._state_spec

class MultiDepthCameraViewWrapper(CameraViewWrapper):
    
    def __init__(self, env, num_frames, cam_id, height=84, width=84, depth=False):
        super().__init__(env, num_frames, cam_id, height, width, depth)
        self._observation_spec = specs.BoundedArray(
            shape=np.array([3 * num_frames + 3, height, width]),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation'
        )
        
    def _transform_obs(self, time_step):
        if isinstance(time_step.observation, np.ndarray):
            obs = np.concatenate([time_step.observation, *self._frames], axis=0)
        else:
            obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)
    
    def _get_obs(self):
        obs = self._get_pixels(height=self._height, width=self._width, camera_id=self._cam_id)
        if self._depth:
            depth_obs = self._get_depth(height=self._height, width=self._width, camera_id=self._cam_id, depth=True)
            obs = np.concatenate([obs, depth_obs], axis=0)
        return obs
    
class ActionClipWrapper(dm_env.Environment):
    def __init__(self, env, minimum, maximum):
        self._env = env
        
        wrapped_action_spec = self._env.action_spec()
        self._action_min = np.array([new_min or orig_min for new_min, orig_min in zip(minimum, wrapped_action_spec.minimum)])
        self._action_max = np.array([new_max or orig_max for new_max, orig_max in zip(maximum, wrapped_action_spec.maximum)])
        


    def step(self, action):
        # TODO: Clip action or Set action space ?
        action = np.clip(action, self._action_min, self._action_max)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, img_size=84, randomize=False, use_embedding=False, use_aug=True, two_cam=False, use_depth=False, control='joint', use_state=False):
    if re.match('^anymal', name):
        name_list = name.split('_')
        domain = name_list[0] + '_' + name_list[1]
        task = name_list[2]
    else:
        domain, task = name.split('_', 1)
        # overwrite cup to ball_in_cup
        domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    elif domain in ['ur5', 'franka', 'xarm']:
        env = globals()[name]()
        pixels_key = 'pixels'

        JOINT_NUM = {
            'franka': 7,
            'ur5': 6,
            'xarm': 6,
        }
        DOF_SETTING = {
            'franka': [None, None, 0., None, 0., None, 0.],
            'ur5': [None, None, None, 0., 0., 0.],
            'xarm': [None, None, None, 0., 0., 0.],
        }
        if control == 'mocap':
            arm_dof = 6
            arm_dof_setting = [None, None, None, 0., 0., 0.]
            if domain == 'franka':
                env = MocapCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), tcp_min=np.array([0, -0.45, 0]), tcp_max=np.array([1.0, 0.45, 0.6]))
            else:
                env = MocapCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
        elif control == 'joint':
            arm_dof = JOINT_NUM[domain]
            arm_dof_setting = DOF_SETTING[domain]
            if 'dual' in task:
                env = DualOpenPickJointCtrlWrapper(env, action_min=-np.ones(14), action_max=np.ones(14))
                # env = DualJointCtrlWrapper(env, action_min=-np.ones(14), action_max=np.ones(14), moving_average=1, action_scale=0.025)
            elif task == 'bowl_dex':
                env = JointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), moving_average=0.4, action_scale=0.04)
            elif task == 'close_dex':
                env = ArmJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
            # elif task == 'dual_dex':
            #     arm_dof = 7
            #     arm_dof_setting = [None, None, 0., None, 0., None, 0.]
            #     dual_arm_dof = 7
            #     env = DualJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), moving_average=0.4, dual_action_min=-np.ones(dual_arm_dof), dual_action_max=np.ones(dual_arm_dof))
            else:
                # env = OriginalJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
                env = OldJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
        
        if domain == 'xarm' and task == 'bowl_dex':
            env = DofWrapper(env, arm_dof_setting + [0, None, 0.4, 0.6] * 3 + [1.8, 0.2, -0.4, 0.5])
        elif domain == 'xarm' and task == 'close_dex':
            env = DofWrapper(env, [0., None, None, 0., None, 0.] + [0, 0.1, 0.3, 0.7] * 3 + [0.2, 0, 0, 0.2])
        elif task == 'lift_dex_cube' or task == 'bowl_dex':
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
        elif task == 'lift_dex_cup':
            arm_dof_setting[5] = 0.
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
        elif task == 'pour_dex':
            env = DofWrapper(env, [None, None, 0., None, None, None, None] + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, -0.1, 0.6])  
            env = ActionClipWrapper(env, [None] * 9, [None] * 6 + [0.9, 1.0, 1.0])
        elif task == 'bowl_stack_dex':
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
            env = ActionClipWrapper(env, [None] * 9, [None] * 6 + [0.5, 0.6, 0.5])
        elif task == 'button_dex':
            env = DofWrapper(env, arm_dof_setting + [0., 0.3, 0.3, 0.3] * 3 + [0.5, 0.3, 0.3, 0.3])
        elif 'lift_dex' in task:
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
            # env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.2, 0.2, None, 0.6])
        elif task == 'dual_open_pick':
            env = DofWrapper(env, [0., None, 0., None, 0., None, 0.] + [0., None, 0., None, 0., 0., 0.] + [0., 0.1, 0.8, 1.0] * 3 + [0., 0.6, 0, 0.2] + [None])
        elif 'dual' in task:
            # env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6] + arm_dof_setting + [None])
            env = DofWrapper(env, [0., None, 0., None, 0., 0., 0.] + [0., None, 0., None, 0., None, 0.] + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6] + [0.])
        elif 'dex' in task:
           env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.3] *  3 + [0.3, 0.2, None, 0.3])
        elif task == 'drawer':
            env = DofWrapper(env, arm_dof_setting + [127])
        else:
            env = DofWrapper(env, arm_dof_setting + [None])
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks

    
    if (domain, task) in suite.ALL_TASKS or domain in ['ur5', 'franka', 'xarm']:
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)

        if use_state:
            env = StateCameraViewWrapper(
                env, num_frames=frame_stack, cam_id=camera_id, height=img_size, width=img_size, depth=use_depth)
        else:
            env = CameraViewWrapper(
                env, num_frames=frame_stack, cam_id=camera_id, height=img_size, width=img_size, depth=use_depth)
    # stack several frames
    # env = FrameStackWrapper(env, frame_stack, pixels_key)

    if two_cam:
        if use_state:
            env = StateCameraViewWrapper(
                env, num_frames=frame_stack, cam_id='track_cam', height=img_size, width=img_size, depth=use_depth)
        else:
            env = CameraViewWrapper(
                env, num_frames=frame_stack, cam_id='track_cam', height=img_size, width=img_size, depth=use_depth)
    if use_state:
        env = StateExtendedTimeStepWrapper(env, randomize)
    else:
        env = ExtendedTimeStepWrapper(env, randomize)
    
    return env





def state_make(name, frame_stack, action_repeat, seed, randomize=False, control='joint'):
    if re.match('^anymal', name):
        name_list = name.split('_')
        domain = name_list[0] + '_' + name_list[1]
        task = name_list[2]
    else:
        domain, task = name.split('_', 1)
        # overwrite cup to ball_in_cup
        domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs={'random': seed},
                         visualize_reward=False)
        pixels_key = 'pixels'
    elif domain in ['ur5', 'franka', 'xarm']:
        env = globals()[name]()
        pixels_key = 'pixels'

        JOINT_NUM = {
            'franka': 7,
            'ur5': 6,
            'xarm': 6,
        }
        DOF_SETTING = {
            'franka': [None, None, 0., None, 0., None, 0.],
            'ur5': [None, None, None, 0., 0., 0.],
            'xarm': [None, None, None, 0., 0., 0.],
        }
        if control == 'mocap':
            arm_dof = 6
            arm_dof_setting = [None, None, None, 0., 0., 0.]
            if domain == 'franka':
                env = MocapCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), tcp_min=np.array([0, -0.45, 0]), tcp_max=np.array([1.0, 0.45, 0.6]))
            else:
                env = MocapCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
        elif control == 'joint':
            arm_dof = JOINT_NUM[domain]
            arm_dof_setting = DOF_SETTING[domain]
            if 'dual' in task:
                env = DualJointCtrlWrapper(env, action_min=-np.ones(14), action_max=np.ones(14), moving_average=0.4)
            elif task == 'bowl_dex':
                env = JointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), moving_average=0.4)
            elif task == 'close_dex':
                env = ArmJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
            # elif task == 'dual_dex':
            #     arm_dof = 7
            #     arm_dof_setting = [None, None, 0., None, 0., None, 0.]
            #     dual_arm_dof = 7
            #     env = DualJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof), moving_average=0.4, dual_action_min=-np.ones(dual_arm_dof), dual_action_max=np.ones(dual_arm_dof))
            else:
                env = OriginalJointCtrlWrapper(env, action_min=-np.ones(arm_dof), action_max=np.ones(arm_dof))
        
        if domain == 'xarm' and task == 'bowl_dex':
            env = DofWrapper(env, arm_dof_setting + [0, None, 0.4, 0.6] * 3 + [1.8, 0.2, -0.4, 0.5])
        elif domain == 'xarm' and task == 'close_dex':
            env = DofWrapper(env, [0., None, None, 0., None, 0.] + [0, 0.1, 0.3, 0.7] * 3 + [0.2, 0, 0, 0.2])
        elif task == 'lift_dex_cube' or task == 'bowl_dex':
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
        elif task == 'lift_dex_cup':
            arm_dof_setting[5] = 0.
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
        elif task == 'pour_dex':
            env = DofWrapper(env, [None, None, 0., None, None, None, None] + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, -0.1, 0.6])  
            env = ActionClipWrapper(env, [None] * 9, [None] * 6 + [0.9, 1.0, 1.0])
        elif task == 'bowl_stack_dex':
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
            env = ActionClipWrapper(env, [None] * 9, [None] * 6 + [0.5, 0.6, 0.5])
        elif task == 'button_dex':
            env = DofWrapper(env, arm_dof_setting + [0., 0.3, 0.3, 0.3] * 3 + [0.5, 0.3, 0.3, 0.3])
        elif 'lift_dex' in task:
            env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6])
            # env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.2, 0.2, None, 0.6])
        elif task == 'dual_open_pick':
            env = DofWrapper(env, [0., None, 0., None, 0., None, 0.] + [0., None, 0., None, 0., 0., 0.] + [0., 0.1, 0.8, 1.0] * 3 + [0., 0.6, 0, 0.2] + [None])
        elif 'dual' in task:
            # env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6] + arm_dof_setting + [None])
            env = DofWrapper(env, [0., None, 0., None, 0., 0., 0.] + [0., None, 0., None, 0., None, 0.] + [0., None, 0.6, 0.6] * 3 + [1.4, 0.2, 0, 0.6] + [0.])
        elif 'dex' in task:
           env = DofWrapper(env, arm_dof_setting + [0., None, 0.6, 0.3] *  3 + [0.3, 0.2, None, 0.3])
        elif task == 'drawer':
            env = DofWrapper(env, arm_dof_setting + [127])
        else:
            env = DofWrapper(env, arm_dof_setting + [None])
    else:
        name = f'{domain}_{task}_vision'
        env = manipulation.load(name, seed=seed)
        pixels_key = 'front_close'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # stack several frames
    env = StateFrameStackWrapper(env, frame_stack)
    env = ExtendedTimeStepWrapper(env, randomize)
    
    return env