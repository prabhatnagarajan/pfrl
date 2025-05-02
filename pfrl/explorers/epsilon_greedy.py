from logging import getLogger

import numpy as np

from pfrl import explorer


def select_action_epsilon_greedily(epsilon, random_action_func, greedy_action_func):
    if np.random.rand() < epsilon:
        return random_action_func(), False
    else:
        return greedy_action_func(), True


class ConstantEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with constant epsilon.

    Args:
      epsilon: epsilon used
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(self, epsilon, random_action_func, logger=getLogger(__name__)):
        assert epsilon >= 0 and epsilon <= 1
        self.epsilon = epsilon
        self.random_action_func = random_action_func
        self.logger = logger

    def select_action(self, t, greedy_action_func, action_value=None):
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        return a

    def __repr__(self):
        return "ConstantEpsilonGreedy(epsilon={})".format(self.epsilon)


class LinearDecayEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with linearly decayed epsilon

    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay_steps: how many steps it takes for epsilon to decay
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(
        self,
        start_epsilon,
        end_epsilon,
        decay_steps,
        random_action_func,
        logger=getLogger(__name__),
    ):
        assert start_epsilon >= 0 and start_epsilon <= 1
        assert end_epsilon >= 0 and end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        self.logger = logger
        self.epsilon = start_epsilon

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (t / self.decay_steps)

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        return a

    def __repr__(self):
        return "LinearDecayEpsilonGreedy(epsilon={})".format(self.epsilon)


class ExponentialDecayEpsilonGreedy(explorer.Explorer):
    """Epsilon-greedy with exponentially decayed epsilon

    Args:
      start_epsilon: max value of epsilon
      end_epsilon: min value of epsilon
      decay: epsilon decay factor
      random_action_func: function with no argument that returns action
      logger: logger used
    """

    def __init__(
        self,
        start_epsilon,
        end_epsilon,
        decay,
        random_action_func,
        logger=getLogger(__name__),
    ):
        assert 0 <= start_epsilon <= 1
        assert 0 <= end_epsilon <= 1
        assert 0 < decay < 1
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay = decay
        self.random_action_func = random_action_func
        self.logger = logger
        self.epsilon = start_epsilon

    def compute_epsilon(self, t):
        epsilon = self.start_epsilon * (self.decay**t)
        return max(epsilon, self.end_epsilon)

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        return a

    def __repr__(self):
        return "ExponentialDecayEpsilonGreedy(epsilon={})".format(self.epsilon)

def power_law_duration_sampler(mu=2.0, max_duration=10000):
    """
    Sample duration n ~ z(n) ∝ n^{-μ} using inverse transform sampling.

    Args:
        mu (float): Power-law exponent.
        max_duration (int): Maximum cap for duration.

    Returns:
        int: Sampled duration (≥1).
    """
    # Sample from discrete power law using inverse CDF
    u = np.random.uniform()
    n = int((1 / u) ** (1 / (mu - 1)))
    return min(max(1, n), max_duration)


class EZGreedy(explorer.Explorer):
    """
    εz-Greedy exploration policy with linearly decayed epsilon and temporally extended exploration.

    Args:
        start_epsilon (float): Initial value of epsilon.
        end_epsilon (float): Final value of epsilon.
        decay_steps (int): Number of steps over which epsilon decays.
        random_action_func (callable): Function that returns a random action.
        greedy_action_func (callable): Function that returns the greedy action.
        duration_sampler (callable): Function that samples the exploratory duration n ~ z.
        logger (logging.Logger): Logger instance for debug output.
    """

    def __init__(
        self,
        start_epsilon,
        end_epsilon,
        decay_steps,
        random_action_func,
        # greedy_action_func,
        # duration_sampler,
        logger=getLogger(__name__),
    ):
        assert 0 <= start_epsilon <= 1
        assert 0 <= end_epsilon <= 1
        assert decay_steps >= 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_action_func = random_action_func
        # self.greedy_action_func = greedy_action_func
        self.duration_sampler = power_law_duration_sampler()
        self.logger = logger

        self.epsilon = start_epsilon
        self.n = 0            # Remaining duration of exploratory action
        self.omega = None     # Current exploratory action

    def compute_epsilon(self, t):
        if t >= self.decay_steps:
            return self.end_epsilon
        return self.start_epsilon + (self.end_epsilon - self.start_epsilon) * (t / self.decay_steps)

    def select_action(self, t, greedy_action_func=None, action_value=None):
        self.epsilon = self.compute_epsilon(t)

        if self.n == 0:
            if np.random.rand() <= self.epsilon:
                # Exploration phase
                self.n = self.duration_sampler()
                self.omega = self.random_action_func()
                a = self.omega
                greedy = False
            else:
                # Greedy phase
                a = greedy_action_func()
                greedy = True
        else:
            # Continue exploratory action
            a = self.omega
            self.n -= 1
            greedy = False

        # greedy_str = "greedy" if greedy else "non-greedy"
        # self.logger.debug("t:%s a:%s %s n:%s", t, a, greedy_str, self.n)
        return a

    def __repr__(self):
        return f"EZGreedy(epsilon={self.epsilon})"
