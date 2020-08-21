import argparse
import json
import operator
from operator import xor
import os


import torch
from torch import nn
import chainer
from chainer import functions as F
from chainer import optimizers
import gym
import gym.wrappers
import numpy as np


import pfrl
from pfrl.q_functions import DiscreteActionValueHead
from pfrl import agents
from pfrl import demonstration
from pfrl import experiments
from pfrl import explorers
from pfrl import nn as pnn
from pfrl import utils
from pfrl.q_functions import DuelingDQN
from pfrl import replay_buffers
from pfrl.wrappers import atari_wrappers
from pfrl.initializers import init_chainer_default
from pfrl.wrappers import score_mask_atari
from pfrl.wrappers.trex_reward import TREXArch
from pfrl.wrappers.trex_reward import TREXReward
from pfrl.wrappers.trex_reward import TREXRewardEnv


import demo_parser


class SingleSharedBias(nn.Module):
    """Single shared bias used in the Double DQN paper.
    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.
    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros([1], dtype=torch.float32))

    def __call__(self, x):
        return x + self.bias.expand_as(x)


def parse_arch(arch, n_actions):
    if arch == "nature":
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "doubledqn":
        return nn.Sequential(
            pnn.LargeAtariCNN(),
            init_chainer_default(nn.Linear(512, n_actions, bias=False)),
            SingleSharedBias(),
            DiscreteActionValueHead(),
        )
    elif arch == "nips":
        return nn.Sequential(
            pnn.SmallAtariCNN(),
            init_chainer_default(nn.Linear(256, n_actions)),
            DiscreteActionValueHead(),
        )
    elif arch == "dueling":
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError("Not supported architecture: {}".format(arch))


def parse_agent(agent):
    return {'DQN': agents.DQN,
            'DoubleDQN': agents.DoubleDQN,
            'PAL': agents.PAL}[agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        default='SpaceInvadersNoFrameskip-v4',
                        help='OpenAI Atari domain to perform algorithm on.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6,
                        help='Timesteps after which we stop ' +
                        'annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.01,
                        help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001,
                        help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--arch', type=str, default='doubledqn',
                        choices=['nature', 'nips', 'dueling', 'doubledqn'],
                        help='Network architecture to use.')
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--target-update-interval',
                        type=int, default=3 * 10 ** 4,
                        help='Frequency (in timesteps) at which ' +
                        'the target network is updated.')
    parser.add_argument('--update-interval', type=int, default=4,
                        help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--eval-n-steps', type=int, default=125000)
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--n-best-episodes', type=int, default=200)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.add_argument('--num-step-return', type=int, default=1)
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--agent', type=str, default='DoubleDQN',
                        choices=['DQN', 'DoubleDQN', 'PAL'])
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate.')
    parser.add_argument('--prioritized', action='store_true', default=False,
                        help='Use prioritized experience replay.')
    # TREX arguments
    parser.add_argument('--load-trex', type=str, default=None)
    parser.add_argument('--mask-render', action='store_true', default=False,
                        help='Mask when you render.')
    parser.add_argument('--trex-steps', type=int, default=30000,
                        help='Number of TREX updates.')
    parser.add_argument('--demo-type', type=str,
                        choices=['agc', 'synth'], required=True)
    parser.add_argument('--load-demos', type=str,
                        help='Atari Grand Challenge Data location or demo pickle file location.')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    if args.demo_type == 'agc':
        assert args.env in ['SpaceInvadersNoFrameskip-v4',
                            'MontezumaRevengeNoFrameskip-v4',
                            'MsPacmanNoFrameskip-v4',
                            'QbertNoFrameskip-v4',
                            'VideoPinballNoFrameskip-v4'
                            ]
    if args.demo_type == 'synth':
        assert args.env in ['SpaceInvadersNoFrameskip-v4',
                            'PongNoFrameskip-v4',
                            'BreakoutNoFrameskip-v4',
                            'EnduroNoFrameskip-v4',
                            'SeaquestNoFrameskip-v4',
                            'QbertNoFrameskip-v4',
                            'BeamRiderNoFrameskip-v4',
                            'HeroNoFrameskip-v4',
                            'MontezumaRevengeNoFrameskip-v4',
                            'MsPacmanNoFrameskip-v4',
                            'VideoPinballNoFrameskip-v4',
                            'FreewayNoFrameskip-v4',
                            'AssaultNoFrameskip-v4',
                            'BoxingNoFrameskip-v4',
                            'StarGunnerNoFrameskip-v4',
                            'ZaxxonNoFrameskip-v4',
                            'SkiingNoFrameskip-v4',
                            'DemonAttackNoFrameskip-v4',
                            'AsteroidsNoFrameskip-v4',
                            'TennisNoFrameskip-v4',
                            'IceHockeyNoFrameskip-v4']


    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            score_mask_atari.make_atari(args.env, max_frames=args.max_frames,
                                        mask_render=args.mask_render),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        else:
            train_network=(False if args.load_trex else True)
            if args.demo_type == 'agc':
                demo_extractor = demo_parser.AtariGrandChallengeParser(
                    args.load_demos, env, args.outdir, not train_network)
            else:
                demo_extractor = demo_parser.PFRLAtariDemoParser(
                    args.load_demos, env, 12, args.outdir, not train_network)
            demo_dataset = None
            if train_network:
                episodes = demo_extractor.episodes
                # Sort episodes by ground truth ranking
                # episodes contain transitions of (obs, a, r, new_obs, done, info)
                # redundance for sanity - demoparser should return sorted
                ranked_episodes = sorted(episodes,
                                         key=lambda ep:sum([ep[i]['reward'] for i in range(len(ep))]))
                episode_rewards = [sum([episode[i]['reward']  \
                                   for i in range(len(episode))]) \
                                   for episode in ranked_episodes]
                demo_dataset = demonstration.RankedDemoDataset(ranked_episodes)
                assert sorted(episode_rewards) == episode_rewards
            network = TREXArch()
            if args.load_trex:
                if args.gpu is not None and args.gpu >= 0:
                    assert torch.cuda.is_available()
                    device = torch.device("cuda:{}".format(args.gpu))
                else:
                    device = torch.device('cpu')
                network.load_state_dict(torch.load(args.load_trex, map_location=device))


            trex_reward = TREXReward(ranked_demos=demo_dataset,
                             optimizer=None,
                             steps=args.trex_steps,
                             network=network,
                             train_network=train_network,
                             gpu=args.gpu,
                             outdir=args.outdir,
                             phi=phi,
                             save_network=True)
            env = TREXRewardEnv(env=env, trex_reward=trex_reward)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    q_func = parse_arch(args.arch, n_actions)

    if args.noisy_net_sigma is not None:
        pnn.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()
    else:
        explorer = explorers.LinearDecayEpsilonGreedy(
            1.0, args.final_epsilon,
            args.final_exploration_frames,
            lambda: np.random.randint(n_actions))

    opt = pfrl.optimizers.RMSpropEpsInsideSqrt(
        q_func.parameters(),
        lr=2.5e-4,
        alpha=0.95,
        momentum=0.0,
        eps=1e-2,
        centered=True,
    )
    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.6,
            beta0=0.4, betasteps=betasteps,
            num_steps=args.num_step_return)
    else:
        rbuf = replay_buffers.ReplayBuffer(10 ** 6, args.num_step_return)

    Agent = parse_agent(args.agent)
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  clip_delta=args.clip_delta,
                  update_interval=args.update_interval,
                  batch_accumulator='sum',
                  phi=phi)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=args.max_frames/4,
            logger=None)
        with open(os.path.join(args.outdir, 'bestscores.json'), 'w') as f:
            json_stats = json.dumps(stats)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))


if __name__ == '__main__':
    main()