from logging import getLogger

import numpy as np

from pfrl import explorer

class LinearDecayEasyGreedy(explorer.Explorer):
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
        self.duration = 0
        self.previous_action = 

    def select_action_easy_greedily(self, random_action_func, greedy_action_func):
        if np.random.rand() < self.epsilon:
            return self.random_action_func(), False
        else:
            if self.duration == 0:
                self.duration = 1 # TODO: Change.
                return greedy_action_func(), True
            else:
                return self.previous_action

    def compute_epsilon(self, t):
        if t > self.decay_steps:
            return self.end_epsilon
        else:
            epsilon_diff = self.end_epsilon - self.start_epsilon
            return self.start_epsilon + epsilon_diff * (t / self.decay_steps)

    def select_action(self, t, greedy_action_func, action_value=None):
        self.epsilon = self.compute_epsilon(t)
        a, easy_greedy = select_action_epsilon_greedily(
            self.epsilon, self.random_action_func, greedy_action_func
        )
        if easy_greedy:
            self.duration -= 1
        self.previous_action = a
        greedy_str = "easy-greedy" if easy_greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)
        return a

    def __repr__(self):
        return "LinearDecayEpsilonGreedy(epsilon={})".format(self.epsilon)

