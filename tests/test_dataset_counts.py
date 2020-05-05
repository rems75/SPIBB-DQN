from unittest import TestCase
from dataset import Dataset_Counts
from collections import namedtuple
import numpy as np

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
        assert_sequence_almost_equal(self, it1, it2)


class TestSplittingDataset(TestCase):
    def setUp(self):
        self.dataset = Dataset_Counts(state_shape=[2], nb_actions=3, count_param=0.2)
        self.dataset_size = 100
        for i in np.random.randint(0, len(transitions), self.dataset_size):
            self.dataset.add(*transitions[i])

    def test_empty_test(self):
        dataset_train, dataset_test = self.dataset.train_validation_split(test_size=0)
        self.assertEqual(dataset_test.size, 0)
        self.assertEqual(dataset_train.size, self.dataset_size)

    def test_default(self):
        dataset_train, dataset_test = self.dataset.train_validation_split()
        self.assertEqual(dataset_test.size, 20)
        self.assertEqual(dataset_train.size, 80)

    def test_empty_train(self):
        dataset_train, dataset_test = self.dataset.train_validation_split(1)
        self.assertEqual(dataset_test.size, self.dataset_size)
        self.assertEqual(dataset_train.size, 0)

    def test_original_data_set_does_not_change(self):
        random_ind = np.random.randint(0, self.dataset_size)
        s, a, p1, r, s2, t, c, p2, c1 = self.dataset._get_transition(random_ind)
        _, _ = self.dataset.train_validation_split(np.random.rand())
        new_s, new_a, new_p1, new_r, new_s2, new_t, new_c, new_p2, new_c1 = self.dataset._get_transition(random_ind)

        assert_sequence_almost_equal(self, new_s, s)
        self.assertEqual(new_a, a)
        assert_sequence_almost_equal(self, new_p1, p1)
        self.assertAlmostEqual(new_r, r)
        assert_sequence_almost_equal(self, new_s2, s2)
        self.assertEqual(new_t, t)
        assert_sequence_almost_equal(self, new_c, c)
        assert_sequence_almost_equal(self, new_p2, p2)
        self.assertAlmostEqual(new_c1, c1)


class TestSmallSplittingDataset(TestCase):
    def test_empty_test(self):
        dataset = Dataset_Counts(state_shape=[2], nb_actions=3, count_param=0.2)
        dataset.add(*transitions[0])
        dataset.add(*transitions[1])

        dataset_train, dataset_test = dataset.train_validation_split(test_size=0.5)
        s_train, a_train, p_train, r_train, s2_train, t_train, _, _, _ = dataset_train._get_transition(0)
        s_test, a_test, p_test, r_test, s2_test, t_test, _, _, _ = dataset_test._get_transition(0)

        if a_train == transitions[0].a:
            trans_train = transitions[0]
            trans_test = transitions[1]
        else:
            trans_train = transitions[1]
            trans_test = transitions[0]

        assert_sequence_almost_equal(self, s_train, trans_train.s)
        self.assertEqual(a_train, trans_train.a)
        assert_sequence_almost_equal(self, p_train, trans_train.p)
        self.assertAlmostEqual(r_train, trans_train.r)
        self.assertEqual(t_train, trans_train.t)

        assert_sequence_almost_equal(self, s_test, trans_test.s)
        self.assertEqual(a_test, trans_test.a)
        assert_sequence_almost_equal(self, p_test, trans_test.p)
        self.assertAlmostEqual(r_test, trans_test.r)
        self.assertEqual(t_test, trans_test.t)


def assert_sequence_almost_equal(test_case, it1, it2):
    for i, j in zip(it1, it2):
        test_case.assertAlmostEqual(i, j)
