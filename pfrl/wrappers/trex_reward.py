import collections
import os
import random

import scipy
import torch
from torch import nn
import torch.nn.functional as F
import gym
import numpy as np

from pfrl.envs import MultiprocessVectorEnv
from pfrl.utils.batch_states import batch_states
from pfrl.wrappers import VectorFrameStack


def subseq(seq, subseq_len, start):
    return seq[start: start + subseq_len]


def threshold_l1_loss(values, threshold, device):
    threshold = torch.full(values.shape, threshold, device=device)
    abs_values = torch.abs(values)
    diff = abs_values - threshold
    # penalize values not in range [-threshold, threshold]
    return torch.mean(torch.max(diff,
                                torch.full(diff.shape, 0.0, device=device)))


class TREXArch(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, traj):
        x = traj
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        return r


class TREXReward():
    """Implements Trajectory-ranked Reward EXtrapolation (TREX):
    https://arxiv.org/abs/1904.06387.
    Args:
        ranked_demos (RankedDemoDataset): A list of ranked demonstrations
        steps: number of gradient steps
        num_sub_trajs: Number of subtrajectories
        sub_traj_len: a tuple containing (min, max) traj length to sample
        traj_batch_size: num trajectory pairs to use per update
        opt: optimizer
        sample_live (bool): whether to create examples as you sample
        network: A reward network to train
        train_network (bool): whether to train the TREX network
        gpu: the device
        outdir: directory to output network and information,
        phi: a preprocessing function
        save_network: whether to save the T-REX network
    """

    def __init__(self,
                 ranked_demos,
                 optimizer,
                 steps=30000,
                 num_sub_trajs=6000,
                 sub_traj_len=(50,100),
                 traj_batch_size=16,
                 sample_live=True,
                 network=TREXArch(),
                 train_network=True,
                 gpu=None,
                 outdir=None,
                 phi=lambda x: x,
                 l1_lambda=0.0,
                 l1_threshold=5.0,
                 save_network=False,
                 alignment_type='post'):
        self.ranked_demos = ranked_demos
        self.steps = steps
        self.trex_network = network
        self.train_network = train_network
        self.training_observations = []
        self.training_labels = []
        self.prev_reward = None
        self.traj_batch_size = traj_batch_size
        self.min_sub_traj_len = sub_traj_len[0]
        self.max_sub_traj_len = sub_traj_len[1]
        self.num_sub_trajs = num_sub_trajs
        self.sample_live = sample_live
        self.outdir = outdir
        self.examples = []      
        self.phi = phi
        self.l1_lambda = l1_lambda
        self.l1_threshold = l1_threshold
        self.running_losses = collections.deque([], maxlen=10)
        assert alignment_type in ('post', 'same', 'random')
        self.alignment_type = alignment_type
        if gpu is not None and gpu >= 0:
            assert torch.cuda.is_available()
            self.device = torch.device("cuda:{}".format(gpu))
            self.trex_network.to(self.device)
        else:
            self.device = torch.device("cpu")
        if self.train_network:
            if optimizer is None:
                self.optimizer = torch.optim.Adam(self.trex_network.parameters(),
                                                  lr=5e-5)
            else:
                self.optimizer = optimizer
            self.save_network = save_network
            if self.save_network:
                assert self.outdir
            self._train()


    def create_example(self):
        '''Creates a training example.'''
        ranked_trajs = self.ranked_demos.episodes
        indices = np.arange(len(ranked_trajs)).tolist()
        traj_indices = np.random.choice(indices, size=2, replace=False)
        i = traj_indices[0]
        j = traj_indices[1]
        traj_i = ranked_trajs[i]
        traj_j = ranked_trajs[j]
        min_ep_len = min(len(traj_1), len(traj_2))
        sub_traj_len = np.random.randint(self.min_sub_traj_len,
                                         self.max_sub_traj_len)
        if self.alignment_type == 'post'
            if i < j:
                i_start = np.random.randint(min_ep_len - sub_traj_len + 1)
                j_start = np.random.randint(i_start, len(traj_j) - sub_traj_len + 1)
            else:
                j_start = np.random.randint(min_ep_len - sub_traj_len + 1)
                i_start = np.random.randint(j_start, len(traj_i) - sub_traj_len + 1)
        elif self.alignment_type == 'same':
            if i < j:
                i_start = np.random.randint(min_ep_len - sub_traj_len + 1)
                j_start = i_start
            else:
                j_start = np.random.randint(min_ep_len - sub_traj_len + 1)
                i_start = j_start
        elif self.alignment_type == 'random':
                i_start = np.random.randint(len(traj_i) - sub_traj_len + 1)
                j_start = np.random.randint(len(traj_j) - sub_traj_len + 1)
        sub_traj_i = subseq(traj_i, sub_traj_len, start=i_start)
        sub_traj_j = subseq(traj_j, sub_traj_len, start=j_start)
        # if trajectory i is better than trajectory j
        if i > j:
            label = 0
        else:
            label = 1
        return sub_traj_i, sub_traj_j, label

    def create_training_dataset(self):
        self.examples = []
        self.index = 0
        for _ in range(self.num_sub_trajs):
            self.examples.append(self.create_example())


    def get_training_batch(self):
        if not self.examples:
            self.create_training_dataset()
        if self.index + self.traj_batch_size > len(self.examples):
            self.index = 0
            if not self.sample_live:
                random.shuffle(self.examples)
            else:
                self.create_training_dataset()
        batch = self.examples[self.index:self.index + self.traj_batch_size]
        return batch

    def _compute_loss(self, batch):
        device = self.device
        preprocessed = {
            'i' : [batch_states([transition["obs"] for transition in example[0]], device, self.phi)
                               for example in batch],
            'j' : [batch_states([transition["obs"] for transition in example[1]], device, self.phi)
                                           for example in batch],
            'label' : torch.tensor([example[2] for example in batch], device=device)
        }
        # Sum up rewards of each trajectory in the batch
        rewards_i = [torch.sum(self.trex_network(preprocessed['i'][i])) for i in range(len(preprocessed['i']))]
        all_state_rewards = [self.trex_network(preprocessed['i'][i]) for i in range(len(preprocessed['i']))] + \
                            [self.trex_network(preprocessed['j'][i]) for i in range(len(preprocessed['j']))]
        individual_rewards = torch.cat(all_state_rewards, dim=0)
        rewards_j = [torch.sum(self.trex_network(preprocessed['j'][i])) for i in range(len(preprocessed['j']))]
        rewards_i = torch.unsqueeze(torch.stack(rewards_i), 1)
        rewards_j = torch.unsqueeze(torch.stack(rewards_j), 1)
        predictions = torch.cat((rewards_i, rewards_j), dim=1)
        if self.l1_lambda != 0.0:
            output_l1_loss = self.l1_lambda * threshold_l1_loss(individual_rewards, self.l1_threshold, device)
        else:
            output_l1_loss = 0.0
        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = cross_entropy_loss(predictions, preprocessed['label']) + output_l1_loss
        return loss

    def _train(self):
        for step in range(1, self.steps + 1):
            # get batch of traj pairs
            batch = self.get_training_batch()
            # do updates
            self.optimizer.zero_grad()
            loss = self._compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            self.running_losses.append(loss.detach().cpu().numpy())
            if len(self.running_losses) == 10:
                with open(os.path.join(self.outdir, 'trex_loss_info.txt'), 'a') as f:
                    print(sum(self.running_losses)/10.0, file=f)
            if step % int(self.steps / min(self.steps, 100)) == 0:
                print("Performed update " + str(step) + "/" + str(self.steps))
        print("Finished training TREX network.")
        if self.save_network:
            torch.save(self.trex_network.state_dict(), os.path.join(self.outdir, "network.pt"))

    def __call__(self, x):
        with torch.no_grad():
            return self.trex_network(x)

class TREXRewardEnv(gym.Wrapper):
    """Environment Wrapper for neural network reward:
    Args:
        env: an Env
        trex_reward: A TREXReward
    Attributes:
        trex_reward: A TREXReward
    """

    def __init__(self, env,
                 trex_reward):
        super().__init__(env)
        self.trex_reward = trex_reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        info["true_reward"] = reward
        obs = batch_states([observation], self.trex_reward.device,
                          self.trex_reward.phi)
        # Outputs a reward of a single state, so shape is (1,1)
        inverse_reward = self.trex_reward(obs).cpu().numpy()[0][0]
        info['pre_sigmoid_reward'] = inverse_reward
        inverse_reward = scipy.special.expit(inverse_reward)
        info['inverse_reward'] = inverse_reward
        return observation, inverse_reward, done, info

class TREXMultiprocessRewardEnv(MultiprocessVectorEnv):
    """Environment Wrapper for neural network reward:
    Args:
        env_fns: an Env
        trex_reward: A TREXReward
    Attributes:
        trex_reward: A TREXReward
    """

    def __init__(self, env_fns,
                 trex_reward):
        super().__init__(env_fns)
        self.trex_reward = trex_reward


    def step(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)
        obs = batch_states(self.last_obs, self.trex_reward.device,
                          self.trex_reward.phi)
        trex_rewards = torch.sigmoid(self.trex_reward(obs))
        trex_rewards = tuple(trex_rewards.array[:,0].tolist())
        for i in range(len(rews)):
            infos[i]["true_reward"] = rews[i]
        return self.last_obs, trex_rewards, dones, infos


class TREXVectorEnv(VectorFrameStack):
    """Environment Wrapper for vector of environments
    to replace with a neural network reward.
    Args:
        env: a MultiProcessVectorEnv
        k: Num frames to stack
        stack_axis: axis to stack frames
        trex_reward: A TREXReward
    Attributes:
        trex_reward: A TREXReward
    """

    def __init__(self, env, k, stack_axis,
                 trex_reward):
        super().__init__(env, k, stack_axis)
        self.trex_reward = trex_reward

    def step(self, actions):
        batch_ob, rewards, dones, infos = self.env.step(actions)
        for frames, ob in zip(self.frames, batch_ob):
            frames.append(ob)
        obs = self._get_ob()
        processed_obs = batch_states(obs, self.trex_reward.device,
                                     self.trex_reward.phi)
        # Convert tensor to numpy array of shape (num_envs, 1)
        inverse_rewards = self.trex_reward(processed_obs).cpu().numpy()
        # Apply sigmoid
        inverse_rewards = scipy.special.expit(inverse_rewards)
        # Convert (num_envs, 1) array of rewards to tuple of len num_envs
        inverse_rewards = tuple(inverse_rewards[:,0].tolist())
        for i in range(len(rewards)):
            infos[i]["inverse_reward"] = inverse_rewards[i]
            infos[i]["true_reward"] = rewards[i]
        return obs, inverse_rewards, dones, infos

class TREXShapedRewardEnv(gym.Wrapper):
    """Environment Wrapper for neural network reward:
    Args:
        env: an Env
        trex_reward: A TREXReward
    Attributes:
        trex_reward: A TREXReward
    """

    def __init__(self, env,
                 trex_reward, gamma):
        super().__init__(env)
        self.trex_reward = trex_reward
        self.gamma = gamma

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        obs = batch_states([observation], self.trex_reward.device,
                          self.trex_reward.phi)
        # Outputs a reward of a single state, so shape is (1,1)
        inverse_reward = self.trex_reward(obs).cpu().numpy()[0][0]
        info['pre_sigmoid_reward'] = inverse_reward
        inverse_reward = scipy.special.expit(inverse_reward)
        shaped_reward = self.gamma * inverse_reward - self.prev_trex_reward
        self.prev_trex_reward = inverse_reward
        info["true_reward"] = reward
        info['inverse_reward'] = inverse_reward
        info['shaped_reward'] = shaped_reward

        return observation, shaped_reward, done, info

    def reset(self):
        observation = self.env.reset()
        obs = batch_states([observation], self.trex_reward.device,
                          self.trex_reward.phi)
        self.prev_trex_reward = self.trex_reward(obs).cpu().numpy()[0][0]
        self.prev_trex_reward = scipy.special.expit(self.prev_trex_reward)
        return observation


class TREXShapedVectorEnv(VectorFrameStack):
    """Environment Wrapper for vector of environments
    to replace with a neural network reward.
    Args:
        env: a MultiProcessVectorEnv
        k: Num frames to stack
        stack_axis: axis to stack frames
        trex_reward: A TREXReward
    Attributes:
        trex_reward: A TREXReward
    """

    def __init__(self, env, k, stack_axis, gamma,
                 trex_reward):
        super().__init__(env, k, stack_axis)
        self.gamma = gamma
        self.trex_reward = trex_reward

    def step(self, actions):
        batch_ob, rewards, dones, infos = self.env.step(actions)
        for frames, ob in zip(self.frames, batch_ob):
            frames.append(ob)
        obs = self._get_ob()
        processed_obs = batch_states(obs, self.trex_reward.device,
                                     self.trex_reward.phi)
        # Convert tensor to numpy array of shape (num_envs, 1)
        inverse_rewards = self.trex_reward(processed_obs).cpu().numpy()
        # Apply sigmoid
        inverse_rewards = scipy.special.expit(inverse_rewards)
        # Convert (num_envs, 1) array of rewards to tuple of len num_envs
        inverse_rewards = tuple(inverse_rewards[:,0].tolist())

        shaped_rewards = []
        for env_id in range(len(rewards)):
            shaped_rewards.append(self.gamma * inverse_rewards[env_id] - self.prev_trex_rewards[env_id])
        shaped_rewards = tuple(shaped_rewards)
        self.prev_trex_rewards = inverse_rewards
        for i in range(len(rewards)):
            infos[i]["true_reward"] = rewards[i]
            infos[i]["inverse_reward"] = inverse_rewards[i]
            infos[i]["shaped_reward"] = shaped_rewards[i]
        return obs, shaped_rewards, dones, infos   

    def reset(self, mask=None):
        obs = VectorFrameStack.reset(self, mask)
        processed_obs = batch_states(obs, self.trex_reward.device,
                                     self.trex_reward.phi)
        inverse_rewards = self.trex_reward(processed_obs).cpu().numpy()
        inverse_rewards = scipy.special.expit(inverse_rewards)
        self.prev_trex_rewards = tuple(inverse_rewards[:,0].tolist())
        return obs
