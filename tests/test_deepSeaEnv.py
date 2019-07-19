from unittest import TestCase
from environments.deep_sea import DeepSeaEnv
import numpy as np


class TestDeepSeaEnv(TestCase):
    def test_rewards(self):
        env = DeepSeaEnv(size=2, randomize=False)
        initial_observation = env.reset()
        self.assertEqual(initial_observation[0], 1)

        observation, reward, term, _ = env.step(0)
        self.assertEqual(reward, 0)
        self.assertFalse(term)
        self.assertEqual(observation[2], 1)

        _, reward, term, _ = env.step(0)
        self.assertEqual(reward, 0)
        self.assertTrue(term)

    def test_end_episode(self):
        size = np.random.randint(1, 20)
        env = DeepSeaEnv(size=size, randomize=True)
        term = False
        for i in range(size):
            self.assertFalse(term)
            _, _, term, _ = env.step(np.random.randint(2))
        self.assertTrue(term)

    def test_big_reward(self):
        size = np.random.randint(1, 20)
        env = DeepSeaEnv(size=size, randomize=False)

        for i in range(size-1):
            observation, reward, term, _ = env.step(1)
            self.assertEqual(observation[(size+1) * (i+1)], 1)
            self.assertAlmostEqual(reward, -0.01/size)
            self.assertFalse(term)

        _, reward, term, _ = env.step(1)
        self.assertEqual(reward, 1-0.01/size)
        self.assertTrue(term)
