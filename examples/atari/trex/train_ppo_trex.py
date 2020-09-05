"""An example of training PPO on Atari Envs with learned rewards.

This script is an example of training a PPO agent on Atari envs.

To train PPO for 10M timesteps on Breakout, run:
    python train_ppo_ale.py

"""
import argparse

import numpy as np
import torch
from torch import nn

import pfrl
from pfrl.agents import PPO
from pfrl import demonstration
from pfrl import experiments
from pfrl import utils
from pfrl.wrappers import atari_wrappers
from pfrl.policies import SoftmaxCategoricalHead
from pfrl.wrappers import score_mask_atari
from pfrl.wrappers.trex_reward import TREXArch
from pfrl.wrappers.trex_reward import TREXReward
from pfrl.wrappers.trex_reward import TREXVectorEnv
from pfrl.wrappers.trex_reward import TREXShapedVectorEnv

import demo_parser


def ground_truth_trajectory_comparison(trex_reward, trajectories):
    # Compute True episode scores
    episode_scores = [sum([episode[i]['reward']  \
                      for i in range(len(episode))]) \
                      for episode in trajectories]
    print(episode_scores)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, default="BreakoutNoFrameskip-v4", help="Gym Env ID."
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU device ID. Set to -1 to use CPUs only."
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of env instances run in parallel.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--steps", type=int, default=10 ** 7, help="Total time steps for training."
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100000,
        help="Interval (in timesteps) between evaluation phases.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=10,
        help="Number of episodes ran in an evaluation phase.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=False,
        help="Run demo episodes, not training.",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help=(
            "Directory path to load a saved agent data from"
            " if it is a non-empty string."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=128 * 8,
        help="Interval (in timesteps) between PPO iterations.",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=32 * 8,
        help="Size of minibatch (in timesteps).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs used for each PPO iteration.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10000,
        help="Interval (in timesteps) of printing logs.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=None,
        help="Frequency at which agents are stored.",
    )
    # TREX arguments
    parser.add_argument('--load-trex', type=str, default=None)
    parser.add_argument('--mask-render', action='store_true', default=False,
                        help='Mask when you render.')
    parser.add_argument('--trex-steps', type=int, default=30000,
                        help='Number of TREX updates.')
    parser.add_argument('--traj-batch-size', type=int, default=16,
                        help='Trajectory batch size')
    parser.add_argument('--sample-live', type=bool, default=True,
                        help='Whether or not to sample new trajectories during training.')
    parser.add_argument('--l1-lambda', type=float, default=0.001,
                        help='L1 Lambda')
    parser.add_argument('--l1-threshold', type=float, default=3.5,
                        help='Value below which we do not penalize the reward output.')
    parser.add_argument('--demo-type', type=str,
                        choices=['agc', 'synth'], required=True)
    parser.add_argument('--load-demos', type=str,
                        help='Atari Grand Challenge Data location or demo pickle file location.')
    # TREX extension argument
    parser.add_argument('--shaped-reward', type=bool, default=False)
    parser.add_argument('--pretrain-steps', type=int, default=0)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            score_mask_atari.make_atari(args.env, max_frames=args.max_frames,
                                        mask_render=args.mask_render),
            episode_life=not test,
            clip_rewards=not test,
            frame_stack=False)
        env.seed(env_seed)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test, sample=False):
        vec_env = pfrl.envs.MultiprocessVectorEnv(
                [
                    (lambda: make_env(idx, test))
                    for idx, env in enumerate(range(args.num_envs))
                ])
        if test:
            vec_env = pfrl.wrappers.VectorFrameStack(vec_env, 4)
        else:
            train_network=(False if args.load_trex else True)
            if args.demo_type == 'agc':
                demo_extractor = demo_parser.AtariGrandChallengeParser(
                    args.load_demos, vec_env, args.outdir, not train_network)
            else:
                demo_extractor = demo_parser.PFRLAtariDemoParser(
                    args.load_demos, vec_env, 12, args.outdir, not train_network)
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
                print(episode_rewards)
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
            if sample:
                train_network = False
            trex_reward = TREXReward(ranked_demos=demo_dataset,
                             optimizer=None,
                             steps=args.trex_steps,
                             network=network,
                             train_network=train_network,
                             gpu=args.gpu,
                             outdir=args.outdir,
                             phi=phi,
                             traj_batch_size=args.traj_batch_size,
                             sample_live=args.sample_live,
                             l1_lambda=args.l1_lambda,
                             l1_threshold=args.l1_threshold,
                             save_network=True)
            if train_network:
                ground_truth_trajectory_comparison(trex_reward, ranked_episodes)

            if not args.shaped_reward:
                vec_env = TREXVectorEnv(env=vec_env, k=4, stack_axis=0,
                                    trex_reward=trex_reward)
            else:
                vec_env = TREXShapedVectorEnv(env=vec_env, k=4, stack_axis=0,
                                              gamma=0.99, trex_reward=trex_reward)
        return vec_env

    sample_env = make_batch_env(test=False, sample=True)
    print("Observation space", sample_env.observation_space)
    print("Action space", sample_env.action_space)
    n_actions = sample_env.action_space.n
    obs_n_channels = sample_env.observation_space.low.shape[0]
    del sample_env

    def lecun_init(layer, gain=1):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            pfrl.initializers.init_lecun_normal(layer.weight, gain)
            nn.init.zeros_(layer.bias)
        else:
            pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
            pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
            nn.init.zeros_(layer.bias_ih_l0)
            nn.init.zeros_(layer.bias_hh_l0)
        return layer

    model = nn.Sequential(
        lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
        nn.ReLU(),
        lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        lecun_init(nn.Linear(3136, 512)),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                lecun_init(nn.Linear(512, n_actions), 1e-2),
                SoftmaxCategoricalHead(),
            ),
            lecun_init(nn.Linear(512, 1)),
        ),
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=phi,
        update_interval=args.update_interval,
        minibatch_size=args.batchsize,
        epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
        recurrent=False,
        max_grad_norm=0.5,
    )
    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_runs: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_runs,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = value

        step_hooks.append(
            experiments.LinearInterpolationHook(args.steps, args.lr, 0, lr_setter)
        )

        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            checkpoint_freq=args.checkpoint_frequency,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
        )


if __name__ == "__main__":
    main()
