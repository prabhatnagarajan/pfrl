import argparse
from collections import deque

import gym
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

import pfrl
from pfrl import agents, experiments, replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead


def reward_fn(dg, ag):
    return -1.0 if (ag != dg).any() else 0.0


class BitFlip(gym.GoalEnv):
    """BitFlip environment from https://arxiv.org/pdf/1707.01495.pdf

    Args:
        n: State space is {0,1}^n
    """

    observation: dict

    def __init__(self, n):
        self.n = n
        self.steps = 0
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.MultiBinary(n),
                achieved_goal=spaces.MultiBinary(n),
                observation=spaces.MultiBinary(n),
            )
        )

    def step(self, action):
        desired_goal = self.observation["desired_goal"].copy()
        before = self.observation["observation"][action]
        new_bit = int(not before)
        new_state = self.observation["observation"].copy()
        new_state[action] = new_bit
        self.observation = {
            "desired_goal": desired_goal,
            "achieved_goal": new_state,
            "observation": new_state,
        }
        reward = reward_fn(new_state, desired_goal)
        done = (new_state == desired_goal).all()
        if done:
            assert reward == 0.0
        # Run out of moves
        if self.steps == (self.n + 1):
            done = True
        self.steps += 1
        return self.observation, reward, done, {}

    def reset(self):
        sample_obs = self.observation_space.sample()
        state, goal = sample_obs["observation"], sample_obs["desired_goal"]
        # Generate state/goal pairs until they're distinct
        while (state == goal).all():
            sample_obs = self.observation_space.sample()
            state, goal = sample_obs["observation"], sample_obs["desired_goal"]
        self.observation = {
            "desired_goal": goal,
            "achieved_goal": state,
            "observation": state,
        }
        self.steps = 0
        return self.observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU to use, set to -1 if no GPU."
    )
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument(
        "--log-level",
        type=int,
        default=30,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--num-bits",
        type=int,
        default=16,
        help="Total number of bits in the env.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 7,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=5 * 100,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument("--use-hindsight", type=bool, default=True)
    parser.add_argument("--eval-n-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--n-best-episodes", type=int, default=100)
    args = parser.parse_args()

    import logging

    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print("Output files are saved in {}".format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = BitFlip(args.num_bits)
        env.seed(int(env_seed))
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    q_func = nn.Sequential(
        init_chainer_default(nn.Linear(args.num_bits * 2, 256)),
        nn.ReLU(),
        init_chainer_default(nn.Linear(256, n_actions)),
        DiscreteActionValueHead(),
    )

    opt = torch.optim.Adam(q_func.parameters(), eps=1e-3)

    if args.use_hindsight:
        rbuf = replay_buffers.hindsight.HindsightReplayBuffer(
            reward_fn=reward_fn,
            replay_strategy=replay_buffers.hindsight.ReplayFutureGoal(),
            capacity=10 ** 6,
        )
    else:
        rbuf = replay_buffers.ReplayBuffer(10 ** 6)

    # Decaying exploration is important in order to converge at 1.0 success rate
    explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=0.3,
        end_epsilon=0.0,
        decay_steps=5000,
        random_action_func=env.action_space.sample,
    )

    def phi(observation):
        # Feature extractor
        obs = np.asarray(observation["observation"], dtype=np.float32)
        dg = np.asarray(observation["desired_goal"], dtype=np.float32)
        return np.concatenate((obs, dg))

    Agent = agents.DoubleDQN
    agent = Agent(
        q_func,
        opt,
        rbuf,
        gpu=args.gpu,
        gamma=0.99,
        explorer=explorer,
        replay_start_size=args.replay_start_size,
        target_update_interval=10 ** 3,
        clip_delta=True,
        update_interval=4,
        batch_accumulator="sum",
        phi=phi,
    )

    if args.load:
        agent.load(args.load)

    wins_window = deque(maxlen=100)
    total_rewards_window = deque(maxlen=100)

    def episode_summary(env, agent, episode_idx, total_reward, last_reward, statistics):
        nonlocal wins_window, total_rewards_window
        wins_window.append(last_reward)
        total_rewards_window.append(total_reward)
        total_rewards_window.append(np.mean(total_rewards_window))
        success_rate = (len(wins_window) - abs(np.sum(wins_window))) / len(wins_window)
        msg = "\rEpisode {}\tAverage Score: {:.2f} \tSuccess Rate: {:.2f}".format(
            episode_idx, np.mean(total_rewards_window), success_rate
        )
        if episode_idx % 100 == 0:
            print(msg)
        else:
            print(msg, end="")

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env, agent=agent, n_steps=args.eval_n_steps, n_episodes=None
        )
        print(
            "n_episodes: {} mean: {} median: {} stdev {}".format(
                eval_stats["episodes"],
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_with_evaluation(
            agent=agent,
            env=env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_episodes,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
            episode_hooks=(episode_summary,),
        )


if __name__ == "__main__":
    main()
