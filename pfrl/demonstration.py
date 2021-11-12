import numpy as np


def extract_episodes(dataset):
    """ Splits a sequential dataset of transitions into episodes.
    Args:
        dataset (chainer Dataset): a dataset consisting of sequential transitions.
    Returns:
        list of episodes, each of which is a list of transitions
    """

    episodes = []
    current_episode = []
    for i in range(len(dataset)):
        obs, a, r, new_obs, done, info = dataset[i]
        current_episode.append(
            {"obs" : obs,
            "action" : a,
            "reward" : r,
            "new_obs" : new_obs,
            "done" : done,
            "info" : info})
        if done:
            episodes.append(current_episode)
            current_episode = []
    return episodes


class RankedDemoDataset():
    """A dataset of episodes ranked by performance quality
    Args:
        episodes: a list of lists of transition_dicts.
    """

    def __init__(self, ranked_episodes):
        self.episodes = ranked_episodes
        self.length = sum([len(episode) for episode in self.episodes])
        self.weights = [float(len(self.episodes[i])) / float(len(self))
                        for i in range(len(self.episodes))]
        np.testing.assert_almost_equal(np.sum(self.weights), 1.0) 

    def __len__(self):
        return self.length 