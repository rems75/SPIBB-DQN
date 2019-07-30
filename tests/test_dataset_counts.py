from unittest import TestCase
from dataset import Dataset_Counts
from collections import namedtuple

Transition = namedtuple('Transition', ('s', 'a', 'r', 't', 'p'))
transitions = [
    Transition([2, 2], 1, 0, False, [.2, .8, 0]),
    Transition([2, 2], 2, 0, False, [.1, .9, 0]),
    Transition([10, 10], 0, 0, False, [.3, .0, .7]),
    Transition([2.03, 2.04], 1, 0, True, [.25, .7, .05]),
    Transition([20, 20], 2, 3, False, [.2, .1, .7]),
]


class TestDatasetCountsAdd(TestCase):
    def setUp(self):
        self.dataset = Dataset_Counts(state_shape=[2], nb_actions=3, count_param=0.2)

    def test_populate(self):
        self.assertEqual(self.dataset.size, 0)
        self.dataset.add(*transitions[0])
        self.assertEqual(self.dataset.size, 1)
        self.assertAlmostEqual(self.dataset.c[0][1], 1)

        self.dataset.add(s=[2, 2], a=2, r=0, t=False, p=[.2, .8, 0])
        self.assertEqual(self.dataset.size, 2)
        self.assertAlmostEqual(self.dataset.c[0][1], 1)
        self.assertAlmostEqual(self.dataset.c[0][2], 1)
        self.assertAlmostEqual(self.dataset.c[1][1], 1)
        self.assertAlmostEqual(self.dataset.c[1][2], 1)

        self.dataset.add(s=[10, 10], a=0, r=0, t=False, p=[.2, .8, 0])
        self.assertAlmostEqual(self.dataset.c[0][0], 0)
        self.assertAlmostEqual(self.dataset.c[0][1], 1)
        self.assertAlmostEqual(self.dataset.c[0][2], 1)
        self.assertAlmostEqual(self.dataset.c[1][1], 1)
        self.assertAlmostEqual(self.dataset.c[1][2], 1)
        self.assertAlmostEqual(self.dataset.c[2][0], 1)

        self.dataset.add(s=[2.03, 2.04], a=1, r=0, t=True, p=[.2, .8, 0])
        self.assertAlmostEqual(self.dataset.c[0][0], 0)
        self.assertAlmostEqual(self.dataset.c[0][1], 1.75)
        self.assertAlmostEqual(self.dataset.c[0][2], 1)
        self.assertAlmostEqual(self.dataset.c[1][0], 0)
        self.assertAlmostEqual(self.dataset.c[1][1], 1.75)
        self.assertAlmostEqual(self.dataset.c[1][2], 1)
        self.assertAlmostEqual(self.dataset.c[3][0], 0)
        self.assertAlmostEqual(self.dataset.c[3][1], 1.75)
        self.assertAlmostEqual(self.dataset.c[3][2], 0.75)


class TestDatasetCounts(TestCase):
    def setUp(self):
        self.dataset = Dataset_Counts(state_shape=[2], nb_actions=3, count_param=0.2)
        for t in transitions:
            self.dataset.add(*t)

    def test_transition_to_same_state(self):
        s, a, _, r, s2, t, c, p, c1 = self.dataset._get_transition(0)
        self.assertSequenceAlmostEqual(s, transitions[0].s)
        self.assertEqual(a, transitions[0].a)
        self.assertAlmostEqual(r, transitions[0].r)
        self.assertSequenceAlmostEqual(s2, transitions[1].s)
        self.assertEqual(t, transitions[0].t)
        self.assertSequenceAlmostEqual(c, self.dataset.c[1])
        self.assertSequenceAlmostEqual(p, transitions[1].p)
        self.assertAlmostEqual(c1, 1.75)

    def test_transition_to_different_state(self):
        s, a, _, r, s2, t, c, p, c1 = self.dataset._get_transition(1)
        self.assertSequenceAlmostEqual(s, transitions[1].s)
        self.assertEqual(a, transitions[1].a)
        self.assertAlmostEqual(r, transitions[1].r)
        self.assertSequenceAlmostEqual(s2, transitions[2].s)
        self.assertEqual(t, transitions[1].t)
        self.assertSequenceAlmostEqual(c, self.dataset.c[2])
        self.assertSequenceAlmostEqual(c, [1, 0, 0])
        self.assertSequenceAlmostEqual(p, transitions[2].p)
        self.assertAlmostEqual(c1, 1)

    def test_terminal_transition(self):
        s, a, _, r, s2, t, c, p, c1 = self.dataset._get_transition(3)
        self.assertSequenceAlmostEqual(s, transitions[3].s)
        self.assertEqual(a, transitions[3].a)
        self.assertAlmostEqual(r, transitions[3].r)
        self.assertSequenceAlmostEqual(s2, [0, 0])
        self.assertEqual(t, True)
        self.assertSequenceAlmostEqual(c, [0, 0, 0])
        self.assertSequenceAlmostEqual(p, [0, 0, 0])
        self.assertAlmostEqual(c1, 1.75)

    def test_save_and_load(self):
        self.dataset.save_dataset('tmp.pickle')
        new_dataset = Dataset_Counts.load_dataset('tmp.pickle')
        self.assertSequenceAlmostEqual(self.dataset.a[0:self.dataset.size], new_dataset.a[0:self.dataset.size])
        self.assertSequenceAlmostEqual(self.dataset.t[0:self.dataset.size], new_dataset.t[0:self.dataset.size])
        self.assertSequenceAlmostEqual(self.dataset.r[0:self.dataset.size], new_dataset.r[0:self.dataset.size])
        for i in range(self.dataset.size):
            self.assertSequenceAlmostEqual(self.dataset.s[i], new_dataset.s[i])
            self.assertSequenceAlmostEqual(self.dataset.c[i], new_dataset.c[i])
        new_dataset.add(*transitions[0])
        import os
        os.remove('tmp.pickle')

    def assertSequenceAlmostEqual(self, it1, it2):
        for i, j in zip(it1, it2):
            self.assertAlmostEqual(i, j)
