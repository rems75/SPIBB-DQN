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
    def __init__(self, data=None, count_param=1, dtype=np.float32, state_shape=None, nb_actions=2):

        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.dtype = dtype
        self.count_param = count_param
        if data is not None:
            self.s, self.a, self.r = data['s'], data['a'], data['r']
            self.s2, self.t, self.c, self.p = data['s2'], data['t'], data['c'], data['p']
            if 'c1' in data:
                self.c1 = data['c1']
            else:
                self.c1 = None
            # TODO remove this parameters
            self.mean = np.mean(self.s, axis=0)
            self.std = np.std(self.s, axis=0)
        else:
            self.s = np.empty(shape=[0] + list(self.state_shape), dtype=self.dtype)
            self.s2 = np.empty(shape=[0] + list(self.state_shape), dtype=self.dtype)
            self.t = np.empty(shape=0, dtype='bool')
            self.a = np.empty(shape=0, dtype='int32')
            self.r = np.empty(shape=0, dtype='float32')
            self.p = np.empty(shape=[0, self.nb_actions], dtype='float32')
            self.c1 = np.empty(shape=0, dtype='float32')
            self.c = np.empty(shape=[0, self.nb_actions], dtype='float32')
        self.dataset_size = len(self.s)


    @staticmethod
    def from_data(data, count_param, dtype=np.float32):
        """
        returns a new DatasetCount initialized with the given data
        :param data: dictionary containing the keys s, a, r, s2, t, c, p[, c1]
        :param count_param:
        :param dtype:
        :return: Dataset_Counts
        """
        assert 's' in data
        assert 'p' in data
        return Dataset_Counts(data, count_param, dtype,
                              state_shape=data['s'][0].shape,
                              nb_actions=data['p'][0].shape[0])

    @staticmethod
    def distance(x1, x2):
        return np.linalg.norm(x1-x2, ord=2)

    @staticmethod
    def similarity(x1, x2, count_param, mean, std):
        return max(0, 1 - Dataset_Counts.distance(x1, x2) / count_param)

    def sample(self, batch_size=1):
        s = np.zeros([batch_size] + list(self.state_shape), dtype=self.dtype)
        s2 = np.zeros([batch_size] + list(self.state_shape), dtype=self.dtype)
        t = np.zeros(batch_size, dtype='bool')
        a = np.zeros(batch_size, dtype='int32')
        r = np.zeros(batch_size, dtype='float32')
        c1 = np.zeros(batch_size, dtype='float32')
        c = np.zeros([batch_size, self.nb_actions], dtype='float32')
        p = np.zeros([batch_size, self.nb_actions], dtype='float32')
        # capacity = 100
        # batch_size = 20
        # size = 20
        # states = np.arange(capacity) * 10
        # locations = np.random.randint(low=size - batch_size, high=size, size=batch_size)
        # samples = states[locations]
        for i in range(batch_size):
            # https://scipy-cookbook.readthedocs.io/items/Indexing.html#List-of-locations-indexing
            # locations = np.random(low =self.size-batch_size, high=self.size, size=batch_size)
            j = np.random.randint(self.dataset_size)
            s[i], a[i], r[i] = self.s[j], self.a[j], self.r[j]
            s2[i], t[i], c[i], p[i] = self.s2[j], self.t[j], self.c[j], self.p[j]
            if self.c1 is not None:
                c1[i] = self.c1[j]
        return s, a, r, s2, t, c, p, c1

    def compute_counts(self, state):
        counts = np.zeros(self.nb_actions)
        for j in range(self.dataset_size):
            s = Dataset_Counts.similarity(state, self.s[j], self.count_param, self.mean, self.std)
            counts[self.a[j]] += s
        return counts

    def add(self, s, a, r, t, p):
        # appending to numpy arrays is very expensive
        # TODO: change how Dataset_Count adds more samples
        # idea: instantiate a large array and increase size when it gets full

        if self.dataset_size > 0 and not self.t[-1]:
            self.s2[-1] = s

        self.s = np.append(self.s, [s], axis=0)
        self.a = np.append(self.a, [a], axis=0)
        self.r = np.append(self.r, [r], axis=0)
        self.t = np.append(self.t, [t], axis=0)
        self.p = np.append(self.p, [p], axis=0)

        empty_state = np.full(shape=list(self.state_shape), fill_value=np.nan)
        self.s2 = np.append(self.s2, [empty_state], axis=0)
