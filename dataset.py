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
    def __init__(self, data, param, dtype=np.float32):
        self.data = data
        self.dataset_size = len(self.data['s'])
        self.state_shape = self.data['s'][0].shape
        self.nb_actions = self.data['p'][0].shape[0]
        self.dtype = dtype
        self.param = param
        self.mean = np.mean(self.data['s'], axis=0)
        self.std = np.std(self.data['s'], axis=0)

    @staticmethod
    def distance(x1, x2):
        return np.linalg.norm(x1-x2)

    @staticmethod
    def similarity(x1, x2, param, mean, std):
        return max(0, 1 - Dataset_Counts.distance(x1, x2) / param)

    def sample(self, batch_size=1):
        s = np.zeros([batch_size] + list(self.state_shape),
                     dtype=self.dtype)
        s2 = np.zeros([batch_size] + list(self.state_shape),
                      dtype=self.dtype)
        t = np.zeros(batch_size, dtype='bool')
        a = np.zeros(batch_size, dtype='int32')
        r = np.zeros(batch_size, dtype='float32')
        c1 = np.zeros(batch_size, dtype='float32')
        c = np.zeros([batch_size, self.nb_actions], dtype='float32')
        p = np.zeros([batch_size, self.nb_actions], dtype='float32')
        for i in range(batch_size):
            j = np.random.randint(self.dataset_size)
            s[i], a[i], r[i] = self.data['s'][j], self.data['a'][j], self.data['r'][j]
            s2[i], t[i], c[i], p[i] = self.data['s2'][j], self.data['t'][j], self.data['c'][j], self.data['p'][j]
            if 'c1' in self.data:
                c1[i] = self.data['c1'][j]
        return s, a, r, s2, t, c, p, c1

    def compute_counts(self, state):
        counts = np.zeros(self.nb_actions)
        for j in range(self.dataset_size):
            s = Dataset_Counts.similarity(state, self.data['s'][j], self.param, self.mean, self.std)
            counts[self.data['a'][j]] += s
        return counts
