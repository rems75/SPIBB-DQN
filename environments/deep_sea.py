"""
Chain environments from Osband et al. [2018, http://bit.ly/rpf_nips] with small adaptations for compatibility.
"""

import collections
import numpy as np
from typing import Tuple

TimeStep = collections.namedtuple('TimeStep', ['observation', 'reward', 'term', "info"])


class DeepSeaEnv(object):

    def __init__(self,
                 size: int,
                 seed: int = None,
                 randomize: bool = True):

        self._size = size
        self._move_cost = 0.01 / size
        self._goal_reward = 1.

        self._column = 0
        self._row = 0

        if randomize:
            rng = np.random.RandomState(seed)
            self._action_mapping = rng.binomial(1, 0.5, size)
        else:
            self._action_mapping = np.ones(size)

        self._reset_next_step = False

    def step(self, action: int) -> TimeStep:
        if self._reset_next_step:
            return self.reset()
        # Remap actions according to column (action_right = go right)
        action_right = action == self._action_mapping[self._column]

        # Compute the reward
        reward = 0.
        if self._column == self._size - 1 and action_right:
            reward += self._goal_reward

        # State dynamics
        if action_right:  # right
            self._column = np.clip(self._column + 1, 0, self._size - 1)
            reward -= self._move_cost
        else:  # left
            self._column = np.clip(self._column - 1, 0, self._size - 1)

        # Compute the observation
        self._row += 1
        if self._row == self._size:
            observation = self._get_observation(self._row - 1, self._column)
            self._reset_next_step = True
            return TimeStep(observation=observation, reward=reward, term=True, info=None)
        else:
            observation = self._get_observation(self._row, self._column)
            return TimeStep(observation=observation, reward=reward, term=False, info=None)

    def reset(self) -> TimeStep:
        self._reset_next_step = False
        self._column = 0
        self._row = 0
        observation = self._get_observation(self._row, self._column)
        return observation
        # return TimeStep(observation=observation, reward=0, term=False, info="")

    def _get_observation(self, row, column) -> np.ndarray:
        observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        observation[row, column] = 1
        return observation.reshape(self._size * self._size)

    @property
    def obs_shape(self) -> Tuple[int]:
        return self.reset().observation.shape

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def optimal_return(self) -> float:
        return self._goal_reward - self._move_cost

