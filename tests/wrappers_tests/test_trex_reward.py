import os
import shutil
import tempfile

import gym
import pytest
from gym.wrappers import TimeLimit

import pfrl


@pytest.mark.parametrize("n_episodes", [1, 2, 3, 4])
def test_monitor(n_episodes):
	