import numpy as np
import pickle
import os


class DataSet(object):

    def __init__(self, path, dataset_size=10000, state_shape=[4], state_dtype=np.float32, nb_actions=9):
        self.size = 0
        self.head = 0
        self.tail = 0
        self.frame_shape = (42, 42)
        self.dataset_size = dataset_size
        self.nb_actions = nb_actions
        self.state_shape = state_shape
        self.state_dtype = state_dtype
        self._minibatch_size = None
        self.states = np.zeros([self.dataset_size] +
                               list(self.state_shape), dtype=self.state_dtype)
        self.actions = np.zeros(self.dataset_size, dtype='int32')
        self.rewards = np.zeros(self.dataset_size, dtype='float32')
        self.terms = np.zeros(self.dataset_size, dtype='bool')
        self.policy = np.zeros([self.dataset_size, self.nb_actions], dtype=self.state_dtype)
        self.qfunction = np.zeros([self.dataset_size, self.nb_actions], dtype=self.state_dtype)

        # Path to the dataset
        self.path = path
        if self.path != None and not os.path.exists(self.path):
            os.makedirs(self.path)

    def add(self, s, a, r, t, p, q=[]):
        self.states[self.tail] = s
        self.actions[self.tail] = a
        self.rewards[self.tail] = r
        self.terms[self.tail] = t
        self.policy[self.tail] = p
        if len(q) > 0:
            self.qfunction[self.tail] = q
        self.tail = (self.tail + 1) % self.dataset_size
        if self.size == self.dataset_size:
            self.head = (self.head + 1) % self.dataset_size
        else:
            self.size += 1

    def reset(self):
        self.size = 0
        self.head = 0
        self.tail = 0
        self._minibatch_size = None
        self.states = np.zeros([self.dataset_size] + list(self.state_shape), dtype=self.state_dtype)
        self.terms = np.zeros(self.dataset_size, dtype='bool')
        self.actions = np.zeros(self.dataset_size, dtype='int32')
        self.rewards = np.zeros(self.dataset_size, dtype='float32')
        self.policy = np.zeros([self.dataset_size, self.nb_actions], dtype='float32')
        self.qfunction = np.zeros([self.dataset_size, self.nb_actions], dtype='float32')

    def save_dataset(self, filename):
        full_path = os.path.join(self.path, filename)
        print("Saving dataset to {}".format(full_path))
        if not os.path.exists(os.path.dirname(full_path)):
            print('Creating dataset directory {}'.format(os.path.dirname(full_path)))
            os.makedirs(os.path.dirname(full_path))
        with open(full_path, "wb") as f:
            pickle.dump([self.states, self.actions, self.rewards, self.terms, self.policy, self.qfunction], f)
        print("Dataset saved")

    '''
    Loads the batch of transitions and creates transitions objects from them.
    (s, a, r, s')
    '''
    def load_dataset(self, filename):
        '''
        It is used to compute the density of state-action pairs.
        rho(s, a) = rho_a(s) * marginal(a)
        Where rho(s, a) is the density of the pair (s, a)
        And rho_action is the density of state s estimated from all the states where action a was taken.
        See Bellemare's Unifying count-based Exploration and Intrinsic Motivation for more details
        '''
        full_path = os.path.join(self.path, filename)

        if not os.path.exists(full_path):
            raise ValueError("We could not find the dataset file: {}".format(full_path))
        print("\nLoading dataset from file {}".format(full_path))
        with open(full_path, "rb") as f:
            self.states, self.actions, self.rewards, self.terms, self.policy, self.qfunction = pickle.load(f)
        self.dataset_size = self.states.shape[0]
        self.size = self.dataset_size

    def counts_weights(self):
        return np.mean(self.states, axis=0), np.std(self.states, axis=0)


class Dataset_Counts(object):
    def __init__(self, s=None, a=None, r=None, t=None, p=None, c=None,
                 count_param=1, dtype=np.float32, state_shape=None, nb_actions=2,
                 initial_size=512, replay_max_size=100):
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.dtype = dtype
        self.count_param = count_param
        self.replay_max_size = replay_max_size
        if s is not None:
            self.s, self.a, self.r, self.t, self.p = s, a, r, t, p
            self.size = len(self.s)
            if c is not None:
                self.c = c
            else:
                self.compute_count_from_scratch()

            self.expand_vectors()
        else:
            self.s = np.zeros(shape=[initial_size] + list(self.state_shape), dtype=self.dtype)
            self.t = np.zeros(shape=initial_size, dtype='bool')
            self.a = np.zeros(shape=initial_size, dtype='int32')
            self.r = np.zeros(shape=initial_size, dtype='float32')
            self.p = np.zeros(shape=[initial_size, self.nb_actions], dtype='float32')
            self.c = np.zeros(shape=[initial_size, self.nb_actions], dtype='float32')
            self.size = 0

    @staticmethod
    def distance(x1, x2):
        return np.linalg.norm(x1-x2)

    @staticmethod
    def similarity(x1, x2, count_param):
        return max(0, 1 - Dataset_Counts.distance(x1, x2) / count_param)

    def sample(self, batch_size=1, full_batch=False):
        """
        return a list containing past experiences

        :param batch_size: number of samples to be drawn from the dataset
        :param full_batch:
            if True: samples from the full dataset
            if False: samples only from the last replay_max_size transitions

        :return: s: states
        :return: a: actions
        :return: r: rewards
        :return: s2: next states
        :return: t: terminal signals (indicate end of trajectory)
        :return: c: state-action visits for the next state s2
        :return: pi_b: baseline policy pi_b(a|s2) for the next state
        :return: c1: state-action counter (related to s,a)
        """

        # TODO avoid instantiating a new arrays for each call
        s = np.zeros([batch_size] + [1] + list(self.state_shape), dtype=self.dtype)
        s2 = np.zeros([batch_size] + [1] + list(self.state_shape), dtype=self.dtype)
        t = np.zeros(batch_size, dtype='bool')
        a = np.zeros(batch_size, dtype='int32')
        r = np.zeros(batch_size, dtype='float32')
        c1 = np.zeros(batch_size, dtype='float32')
        c = np.zeros([batch_size, self.nb_actions], dtype='float32')
        p = np.zeros([batch_size, self.nb_actions], dtype='float32')

        for i, randint in enumerate(self.sample_indexes(full_batch, batch_size)):
            s[i], a[i], r[i], s2[i], t[i], c[i], p[i], c1[i] = self._get_transition(randint)
        return s, a, r, s2, t, c, p, c1

    def sample_indexes(self, full_batch, size):
        if full_batch:
            # sample from the full dataset ignoring last transition that might be incomplete
            return np.random.randint(self.size - 1, size=size)
        else:
            # sample from the last replay_max_size transitions, ignoring last transition that might be incomplete
            return np.random.randint(max(self.size - self.replay_max_size, 0), self.size - 1, size=size)

    def _get_transition(self, ind):
        s = self.s[ind]
        a = self.a[ind]
        r = self.r[ind]
        t = self.t[ind]
        if not t:
            s2 = self.s[ind + 1]
            c = self.c[ind + 1]
            p = self.p[ind + 1]
        else:
            s2 = np.zeros(self.state_shape)
            c = np.zeros(self.nb_actions)
            p = np.zeros(self.nb_actions)
        c1 = self.c[ind][self.a[ind]]
        return s, a, r, s2, t, c, p, c1

    def compute_counts(self, state):
        x, y = np.broadcast_arrays(state, self.s[0:self.size, :])
        distance = np.linalg.norm(x - y, axis=1)
        sim_vec = np.maximum(1 - distance / self.count_param, 0)
        z = np.zeros((self.size, self.nb_actions))
        z[np.arange(self.size), self.a[0:self.size]] = sim_vec
        return z.sum(axis=0)

        # counts = np.zeros(self.nb_actions)
        # for j in range(self.size):
        #     s = Dataset_Counts.similarity(state, self.s[j], self.count_param)
        #     counts[self.a[j]] += s
        # return counts

    def add(self, s, a, r, t, p):
        self.s[self.size] = s
        self.a[self.size] = a
        self.r[self.size] = r
        self.t[self.size] = t
        self.p[self.size] = p

        self.increment_counters(s, a)

        self.size += 1
        if self.size == len(self.s):
            self.expand_vectors()

    def expand_vectors(self):
            self.s = self.expand(self.s)
            self.a = self.expand(self.a)
            self.r = self.expand(self.r)
            self.t = self.expand(self.t)
            self.p = self.expand(self.p)
            self.c = self.expand(self.c)

    def increment_counters(self, s, a):
        """
        increase counters of previous transitions according to the similarity between s and a
        :param s:
        :param a:
        """
        # increment counters with array operations
        self.c[self.size][a] = 1
        if self.size > 0:
            x, y = np.broadcast_arrays(s, self.s[0:self.size, :])
            distance = np.linalg.norm(x - y, axis=1)
            sim_vec = np.maximum(1 - distance/self.count_param, 0)
            self.c[np.arange(self.size), a] += sim_vec

            z = np.zeros((self.size, self.nb_actions))
            z[np.arange(self.size), self.a[0:self.size]] = sim_vec
            self.c[self.size] += z.sum(axis=0)

        # # increment counter iteratively
        # self.c[self.size][a] = 1
        # for ind in range(self.size):
        #     sim = Dataset_Counts.similarity(self.s[ind], s, self.count_param)
        #     # increase counter of s', a in transitions of the dataset with a state s' similar to s
        #     self.c[ind][a] += sim
        #     # increase counter of s, a' if the dataset has a transition s', a', such that s' is similar to s
        #     self.c[self.size][self.a[ind]] += sim

    def expand(self, a):
        # allocate more memory for the array a
        new_shape = list(a.shape)
        new_shape[0] *= 4
        new_a = np.zeros(new_shape, dtype=a.dtype)
        new_a[:self.size] = a
        return new_a

    @staticmethod
    def from_dataset(dataset: DataSet, count_param):
        """
        construct a dataset_count from a regular dataset

        :param dataset: Dataset
        :param count_param:
        :return: new dataset_count
        """
        return Dataset_Counts(dataset.states, dataset.actions, dataset.rewards, dataset.terms, dataset.policy,
                           state_shape=dataset.state_shape, nb_actions=dataset.nb_actions, count_param=count_param)

    def save_dataset(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump([self.s[0:self.size], self.a[0:self.size], self.r[0:self.size], self.t[0:self.size],
                         self.p[0:self.size], self.c[0:self.size],
                         self.count_param, self.dtype, self.state_shape, self.nb_actions], f)

    @staticmethod
    def load_dataset(file_path):
        with open(file_path, "rb") as f:
            return Dataset_Counts(*(pickle.load(f)))

    def compute_count_from_scratch(self):
        """
        computes the counters of each state action pair,
        assuming self.s and self.a are already populated and self.size is well defined
        """
        self.c = np.zeros(shape=[self.size, self.nb_actions], dtype='float32')
        for i in range(self.size):
            self.c[i] = self.compute_counts(self.s[i])
