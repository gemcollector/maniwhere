# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import wandb

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        
        self.agent_name = cfg.wandb_group
        print(f'agent name: {self.agent_name}')
        
        model_path = f'/cephfs/yzc/project/domain_randomization/exp_local/model_log/{self.cfg.task_name}/{self.agent_name}/{self.cfg.seed}/best_snapshot.pt'
        
        agent = torch.load(model_path, map_location='cuda:0')
        self.agent = agent['agent']
        self.timer = utils.Timer()
        self._global_step = agent['_global_step']
        self._global_episode = 0
        self._obs_channel = self.eval_env.observation_spec().shape[0]

    def setup(self):
        if self.cfg.use_wandb:
            exp_name = '_'.join([
                self.cfg.task_name,
                str(self.cfg.seed)
            ])
            wandb.init(project="gvrlb_algo", group=self.cfg.wandb_group, name=exp_name)
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed, randomize=True, two_cam=True)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, randomize=True, two_cam=True, img_size=self.cfg.img_size, use_depth=self.cfg.use_depth)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        episode = 0
        eval_until_episode = utils.Until(50)
        success = 0
        success_task_lists = ['xarm_close_dex', 'franka_dual_dex', 'franka_bowl_dex']

        while eval_until_episode(episode):
            step, total_reward = 0, 0
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env)
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation[self._obs_channel:],
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'eval_{episode}.mp4')
            if step < 500 and (self.cfg.task_name in success_task_lists):
                success += 1
            else:
                if total_reward > 250 and self.cfg.task_name == 'ur5_lift':
                    success += 1
                elif total_reward > 250 and self.cfg.task_name == 'ur5_reach':
                    success += 1
                elif total_reward > 250 and self.cfg.task_name == 'ur5_stack':
                    success += 1
                elif total_reward > 200 and self.cfg.task_name == 'franka_lift_robotiq':
                    success += 1
                elif total_reward > 130 and self.cfg.task_name == 'franka_reach_dex':
                    success += 1
                elif total_reward > 230 and self.cfg.task_name == 'ur5_bowl':
                    success += 1
                elif total_reward > 50 and self.cfg.task_name == 'franka_lift_dex_cube':
                    success += 1
                elif total_reward > 250 and self.cfg.task_name == 'franka_button_dex':
                    success += 1
                elif total_reward > 230 and self.cfg.task_name == 'ur5_drawer':
                    success += 1

            with self.logger.log_and_dump_ctx(episode, ty='eval') as log:
                log('episode_reward', total_reward)
                log('episode_length', step * self.cfg.action_repeat)
                log('episode', episode)
                log('step', episode)
        print(f'seed {self.cfg.seed}, success rate: {success / episode}')

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.cfg.save_snapshot and (self.global_step % int(5e4) == 0):
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='camera_aug_config')
def main(cfg):
    from mani_eval import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    
    workspace.eval()


if __name__ == '__main__':
    main()