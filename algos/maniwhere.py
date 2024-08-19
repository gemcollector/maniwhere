# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet
import utils
import random
from collections import deque
from algos.stn import TransformNet_STN_PerImage, TransformNet_STN1, PerspectiveSTNPerImage, PerspectiveSTN
from utils import random_overlay, random_mask_freq_v2

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        self.repr_dim = 1024

        self.model = resnet.resnet18()
        self.model.conv1 = nn.Conv2d(obs_shape[0], self.model.conv1.out_channels, kernel_size=7, stride=2, padding=1, bias=False)
        self.model = self.model.cuda()

        # Construct STN
        with torch.no_grad():
            x = torch.randn((1, *obs_shape)).cuda()
            conv1_out_shape = self.model.conv1(x).shape[1:]

        self.stn = [
            PerspectiveSTNPerImage(obs_shape).cuda(),
            PerspectiveSTN(conv1_out_shape).cuda()
        ]

        # Construct Linear
        with torch.no_grad():
            x = torch.randn((1, *obs_shape)).cuda()
            out_shape = self._forward_conv(x).shape
        self.out_dim = out_shape[1]

        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)


    def _forward_conv(self, x, layer_feat=False, return_theta=False):
        layers = []
        
        x = x / 255. - 0.5
        # obs = torch.stack([self.transforms(img) for i in stack], dim=0)

        x, theta1 = self.stn[0](x, True)
        layers.append(x)
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        
        x = self.stn[1](x)
        layers.append(x)
        
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        layers.append(x)
        x = self.model.layer2(x)
        layers.append(x)

        feat = x.view(x.shape[0], -1)

        if return_theta:
            if layer_feat:
                return feat, layers, theta1
            else:
                return feat, theta1
        else:
            if layer_feat:
                return feat, layers
            else:
                return feat

    def forward(self, obs, layer_feat=False, return_theta=False):
        if return_theta:
            if layer_feat:
                feat, layer, theta = self._forward_conv(obs, layer_feat, return_theta)
            else:
                feat, theta = self._forward_conv(obs, layer_feat)
        else:
            if layer_feat:
                feat, layer = self._forward_conv(obs, layer_feat, return_theta)
            else:
                feat = self._forward_conv(obs, layer_feat)
        feat = self.fc(feat)
        feat = self.ln(feat)

        if return_theta:
            if layer_feat:
                return feat, layer, theta
            else:
                return feat, theta
        else:
            if layer_feat:
                return feat, layer
            else:
                return feat

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist
    

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class Auxiliary(nn.Module):
    def __init__(self, repr_dim, feature_dim, temp=0.1):
        super().__init__()

        self.temp = temp
        
        self.projector = nn.Linear(repr_dim, feature_dim)

        self.apply(utils.weight_init)

    def contrastive_loss(self, q, k):
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        labels = torch.arange(logits.shape[0], dtype=torch.long).cuda()
        contrastive_loss = nn.CrossEntropyLoss()(logits, labels)
        return contrastive_loss

    def forward(self, q, k):
        q = self.projector(q)
        k = self.projector(k)
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        loss = (self.contrastive_loss(q, k) + self.contrastive_loss(k, q)) / 2
        return loss

class SigmoidLR:
    def __init__(self, optimizer, lr_max=1, lr_min=0, sigmoid_slope=0.015, sigmoid_center=500):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.sigmoid_slope = sigmoid_slope
        self.sigmoid_center = sigmoid_center
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch):
        lr = (self.lr_max - self.lr_min) / (1 + np.exp(-self.sigmoid_slope * (epoch - self.sigmoid_center))) + self.lr_min
        return lr

    def step(self):
        self.scheduler.step()

class ManiAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, use_wandb,
                 temp, aux_coef, aux_l2_coef, aux_tcc_coef, aux_latency, lr_stn):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb or use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.aux_coef = aux_coef
        self.aux_l2_coef = aux_l2_coef
        self.aux_tcc_coef = aux_tcc_coef
        self.aux_latency = aux_latency

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.auxiliary = Auxiliary(self.encoder.repr_dim, feature_dim=256, temp=temp).to(device)

        # optimizers
        self.encoder_no_stn_opt = torch.optim.Adam(self.encoder.model.parameters(), lr=lr)
        self.stn_opt = torch.optim.Adam([
            {'params': self.encoder.stn[0].parameters(), 'lr': lr_stn},
            {'params': self.encoder.stn[1].parameters(), 'lr': lr_stn}
        ], lr=lr_stn)
        self.encoder_opt = torch.optim.Adam(self.encoder.model.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.encoder_no_stn_aux_opt = torch.optim.Adam(self.encoder.model.parameters(), lr=lr)
        # self.aux_opt_scheduler = SigmoidLR(self.encoder_no_stn_aux_opt)
        # self.stn_opt_scheduler = SigmoidLR(self.stn_opt)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def read_q(self, obs, step):
        stddev = utils.schedule(self.stddev_schedule, step)
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        dist = self.actor(obs, stddev)
        action = dist.mean
        q = self.critic(obs, action)
        
        return q


    def update_critic(self, obs, action, reward, discount, next_obs, step, aug_obs, aug_move_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)
        
        
        # critic_loss = 0.5 * (critic_loss + aug_loss) + 0.3 * aug_move_loss
        if step > self.aux_latency:
            aug_move_Q1, aug_move_Q2 = self.critic(aug_move_obs, action)
            aug_move_loss = F.mse_loss(aug_move_Q1, target_Q) + F.mse_loss(aug_move_Q2, target_Q)
            critic_loss = 0.5 * critic_loss + 0.25 * (aug_loss + aug_move_loss)
            # critic_loss = 0.5 * (critic_loss + aug_move_loss)
            # critic_loss = 0.5 * (critic_loss + aug_loss) + 0.3 * aug_move_loss
        else:
            critic_loss = 0.5 * (critic_loss + aug_loss)
        # critic_loss = 0.5 * (critic_loss + aug_move_loss)

        # l2_loss_aug = F.mse_loss(obs, aug_obs) * 0.1

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_no_stn_opt.zero_grad(set_to_none=True)
        self.stn_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        # (critic_loss + l2_loss_aug).backward()
        critic_loss.backward()
        self.critic_opt.step()
        self.stn_opt.step()
        self.encoder_no_stn_opt.step()
        
        if self.use_tb:
            grad = self.encoder.model.conv1.weight.grad
            metrics['grad_critic_mean'] = grad.mean().item() if grad is not None else 0
            metrics['grad_critic_max'] = grad.max().item() if grad is not None else 0
            metrics['grad_critic_min'] = grad.min().item() if grad is not None else 0
            # metrics['aux_l2_loss_aug'] = l2_loss_aug.item()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_auxiliary(self, step, fix_obs, move_obs, trajs):
        metrics = dict()
        
        # update auxiliary task
        def calc_aux():
            fix_view_feat, fix_layers, theta1 = self.encoder(fix_obs, layer_feat=True, return_theta=True)
            move_view_feat, move_layers = self.encoder(move_obs, layer_feat=True)
            
            contrastive_loss = self.auxiliary(fix_view_feat, move_view_feat)
            
            l2_loss = F.mse_loss(fix_view_feat, move_view_feat)
            
            # only use layer1 & layer2 feat
            fix_layers = fix_layers[-2:]
            move_layers = move_layers[-2:]
            l2_loss_layers = 0
            for fix_layer, move_layer in zip(fix_layers, move_layers):
                l2_loss_layers += F.mse_loss(fix_layer, move_layer)
            l2_loss_layers /= len(fix_layers)

            aux_loss = contrastive_loss * self.aux_coef + \
                    l2_loss * self.aux_l2_coef + l2_loss_layers * self.aux_l2_coef

                
            if self.use_tb:
                metrics['aux_contrastive_loss'] = contrastive_loss.item()
                metrics['aux_l2_loss'] = l2_loss.item()
                metrics['aux_l2_loss_layers'] = l2_loss_layers.item()
                metrics['theta1_02'] = theta1.mean(dim=0)[0][0].item()
                metrics['aux_lr'] = self.encoder_no_stn_aux_opt.param_groups[0]['lr']
                    
            return aux_loss
        
        self.encoder_no_stn_aux_opt.zero_grad(set_to_none=True)
        self.stn_opt.zero_grad(set_to_none=True)
        
        if step > self.aux_latency:
            aux_loss = calc_aux()

            aux_loss.backward()
            # nn.utils.clip_grad_norm_(self.encoder.parameters(), 25, error_if_nonfinite=False)
            self.stn_opt.step()
            self.encoder_no_stn_aux_opt.step()
        else:
            # with torch.no_grad(), utils.eval_mode(self.encoder):
            #     aux_loss = calc_aux()
            metrics['aux_contrastive_loss'] = 0
            metrics['aux_l2_loss'] = 0
            metrics['aux_l2_loss_layers'] = 0
            metrics['theta1_02'] = 0
            metrics['aux_lr'] = self.encoder_no_stn_aux_opt.param_groups[0]['lr']
            
        
        if self.use_tb:
            grad = self.encoder.model.conv1.weight.grad
            metrics['grad_aux_mean'] = grad.mean().item() if grad is not None else 0
            metrics['grad_aux_max'] = grad.max().item() if grad is not None else 0
            metrics['grad_aux_min'] = grad.min().item() if grad is not None else 0

        return metrics

    def update(self, replay_iter, trajs, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # auxiliary
        l = obs.shape[1] // 2
        fix_obs=obs.float()[:, :l]
        move_obs=obs.float()[:, l:]
        fix_next_obs = next_obs.float()[:, :l]
        
        # augment
        obs = self.aug(fix_obs)
        original_obs = obs.clone()
        next_obs = self.aug(fix_next_obs)
        # import ipdb;ipdb.set_trace()
        original_move_obs = move_obs.clone()

        # TODO: not elegant
        # strong augmentation + SRM
        if l % 3 == 0:
            aug_obs = random_mask_freq_v2(random_overlay(original_obs))
            if step > self.aux_latency:
                aug_move_obs = random_mask_freq_v2(random_overlay(original_move_obs))
                aug_move_obs = self.encoder(aug_move_obs)  
            else:
                aug_move_obs = None
        else:
            aug_obs = random_mask_freq_v2(random_overlay(original_obs[:, :l-1]))
            # print("", aug_obs.shape, )
            aug_obs = torch.cat([aug_obs, original_obs[:, l-1:l]], dim=1)
            if step > self.aux_latency:
                aug_move_obs = random_mask_freq_v2(random_overlay(original_move_obs[:, :l-1]))
                aug_move_obs = torch.cat([aug_move_obs, original_move_obs[:, l-1:l]], dim=1)
                aug_move_obs = self.encoder(aug_move_obs)       
            else:
                aug_move_obs = None
            
        aug_obs = self.encoder(aug_obs)
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, aug_obs, aug_move_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update auxiliary task
        metrics.update(self.update_auxiliary(step, fix_obs, move_obs, trajs))
        
        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    
