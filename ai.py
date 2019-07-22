import numpy as np
from utils import ExperienceReplay
from model import SmallDenseNetwork, DenseNetwork, Network, LargeNetwork, NatureNetwork
import torch
import torch.optim as optim

from baseline import Baseline


# Upper bound on q-values. Just used as an artefact
MAX_Q = 100000


class AI(object):
    def __init__(self, baseline, state_shape=[4], nb_actions=9, action_dim=1, reward_dim=1, history_len=1, gamma=.99,
                 learning_rate=0.00025, epsilon=0.05, final_epsilon=0.05, test_epsilon=0.0, annealing_steps=1000,
                 minibatch_size=32, replay_max_size=100, update_freq=50, learning_frequency=1, ddqn=False, learning_type='pi_b',
                 network_size='nature', normalize=1., device=None, kappa=0.003, minimum_count=0, epsilon_soft=0,
                 baseline_update_freq=0, baseline_temp=0):

        self.history_len = history_len
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.start_learning_rate = learning_rate
        self.epsilon = epsilon
        self.start_epsilon = epsilon
        self.test_epsilon = test_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = annealing_steps
        self.minibatch_size = minibatch_size
        self.network_size = network_size
        self.update_freq = update_freq
        self.update_counter = 0
        self.normalize = normalize
        self.learning_frequency = learning_frequency  # frequency that the target network is updated
        self.replay_max_size = replay_max_size
        # TODO instantiate DatasetCounts instead
        # TODO get an initial dataset for batch experiments
        self.transitions = ExperienceReplay(max_size=self.replay_max_size, history_len=history_len,
                                            state_shape=state_shape, action_dim=action_dim, reward_dim=reward_dim)
        self.ddqn = ddqn
        self.device = device
        self.network = self._build_network()
        self.target_network = self._build_network()
        self.weight_transfer(from_model=self.network, to_model=self.target_network)
        self.network.to(self.device)
        self.target_network.to(self.device)
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, alpha=0.95, eps=1e-07)

        # SPIBB parameters
        if baseline is not None:
            self.baseline = baseline
        else:
            self.baseline = Baseline(network_path=None, network_size=self.network_size,
                                     network_state=self.network.state_dict(), state_shape=state_shape,
                                     nb_actions=nb_actions, temperature=baseline_temp, normalize=255., device=self.device)

        self.learning_type = learning_type
        self.kappa = kappa
        self.minimum_count = minimum_count
        self.epsilon_soft = epsilon_soft
        self.baseline_update_freq = baseline_update_freq
        self.training_step = 0
        self.interaction_step = 0  # counts interactions with the environment (during training and evaluation)
        self.logger = None

    def _build_network(self):
        if self.network_size == 'small':
            return Network()
        elif self.network_size == 'large':
            return LargeNetwork(state_shape=self.state_shape, nb_channels=4, nb_actions=self.nb_actions, device=self.device)
        elif self.network_size == 'nature':
            return NatureNetwork(state_shape=self.state_shape, nb_channels=4, nb_actions=self.nb_actions, device=self.device)
        elif self.network_size == 'dense':
            return DenseNetwork(state_shape=self.state_shape[0], nb_actions=self.nb_actions, device=self.device)
        elif self.network_size == 'small_dense':
            return SmallDenseNetwork(state_shape=self.state_shape[0], nb_actions=self.nb_actions, device=self.device)
        else:
            raise ValueError('Invalid network_size.')

    def train_on_batch(self, s, a, r, s2, term, c=None, pi_b=None, c1=None):
        """
        each parameter is a list containing past experiences

        :param s: current states
        :param a: actions
        :param r: rewards
        :param s2: next states
        :param term: terminal signals (indicate end of trajectory)
        :param c: state-action visits for the next state s2
        :param pi_b: baseline policy pi_b(a|s2) for the next state
        :param c1: state-action counter related to (s,a)
        :return: loss
        """
        s = torch.FloatTensor(s).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        t = torch.FloatTensor(np.float32(term)).to(self.device)

        # Squeeze dimensions for history_len = 1
        s = torch.squeeze(s)
        s2 = torch.squeeze(s2)
        q = self.network(s / self.normalize)
        q2 = self.target_network(s2 / self.normalize).detach()
        q_pred = q.gather(1, a.unsqueeze(1)).squeeze(1)

        def _get_q2max(mask=None):
            if mask is None:
                mask = torch.FloatTensor(np.ones(list(a.shape) + [self.nb_actions])).to(self.device)
                # mask = torch.FloatTensor(np.ones(c.shape)).to(self.device)
            if self.ddqn:
                q2_net = self.network(s2 / self.normalize).detach()
                a_max = torch.max(q2_net - (1-mask)*MAX_Q, 1)[1].unsqueeze(1)
                return q2.gather(1, a_max).squeeze(1), a_max
            else:
                return torch.max(q2 - (1-mask)*MAX_Q, 1)

        def _get_bellman_target_dqn():
            q2_max, _ = _get_q2max()
            return r + (1 - t) * self.gamma * q2_max.detach()

        def _get_bellman_target_ramdp(c1):
            # State/action counts for state s1 (used for RaMDP)
            q2_max, _ = _get_q2max()
            c1 = torch.FloatTensor(c1).to(self.device)
            return r - self.kappa / torch.sqrt(c1) + (1 - t) * self.gamma * q2_max

        def _get_bellman_target_pi_b(c, pi_b_):
            # All state/action counts for state s2
            c = torch.FloatTensor(c).to(self.device)
            # Policy on state s2 (estimated using softmax on the q-values)
            pi_b_ = torch.FloatTensor(pi_b_).to(self.device)
            # Mask for "bootstrapped actions"
            mask_non_bootstrapped = (c >= self.minimum_count).float()
            # print("mask_non_bootstrapped %d, bootstrapped: %d" % ((c >= self.minimum_count).float().sum(), (c < self.minimum_count).float().sum()))
            # r + (1 - t) * gamma * max_{a s.t. (s',a) not in B}(Q'(s',a)) * proba(actions not in B)
            #   + (1 - t) * gamma * sum(proba(a') Q'(s',a'))
            q2_max, _ = _get_q2max(mask_non_bootstrapped)
            # (1 - t): if terminal state does not add expected future reward

            self.logger.add_scalar('bootstrapped', torch.sum(1 - mask_non_bootstrapped).item(), self.training_step)
            return r + (1 - t) * self.gamma * (
                    q2_max * torch.sum(pi_b_ * mask_non_bootstrapped, 1)  # prob mass of non_bootstrapped (s,a) pairs
                    + torch.sum(q2 * pi_b_ * (1 - mask_non_bootstrapped), 1))  # prob mass of bootstrapped (s,a) pairs

        def _get_bellman_target_online_pi_b(c, pi_b_):
            # All state/action counts for state s2
            c = torch.FloatTensor(c).to(self.device)
            # Policy on state s2 (estimated using softmax on the q-values)
            pi_b_ = torch.FloatTensor(pi_b_).to(self.device)
            # Mask for "bootstrapped actions"
            mask_non_bootstrapped = (c >= self.minimum_count).float()
            # print("mask_non_bootstrapped %d, bootstrapped: %d" % ((c >= self.minimum_count).float().sum(), (c < self.minimum_count).float().sum()))
            # r + (1 - t) * gamma * max_{a s.t. (s',a) not in B}(Q'(s',a)) * proba(actions not in B)
            #   + (1 - t) * gamma * sum(proba(a') Q'(s',a'))
            q2_max, _ = _get_q2max(mask_non_bootstrapped)

            self.logger.add_scalar('bootstrapped', torch.sum(1 - mask_non_bootstrapped).item(), self.training_step)
            # (1 - t): if terminal state does not add expected future reward
            return r + (1 - t) * self.gamma * (
                    q2_max * torch.sum(pi_b_ * mask_non_bootstrapped, 1)  # prob mass of non_bootstrapped (s,a) pairs
                    + torch.sum(q2 * pi_b_ * (1 - mask_non_bootstrapped), 1))  # prob mass of bootstrapped (s,a) pairs

        def _get_bellman_target_soft_sort(c, pi_b):
            # All state/action counts for state s2
            c = torch.FloatTensor(c).to(self.device)
            # e est le vecteur d'erreur
            e = torch.sqrt(1 / (c + 1e-9))
            # Policy on state s2 (estimated using softmax on the q-values)
            pi_b = torch.FloatTensor(pi_b).to(self.device)
            _pi_b = torch.FloatTensor(pi_b).to(self.device)
            allowed_error = self.epsilon_soft * torch.ones((self.minibatch_size))
            if self.ddqn:
                _q2_net = self.network(s2 / self.normalize).detach()
            else:
                _q2_net = q2
            sorted_qs, arg_sorted_qs = torch.sort(_q2_net, dim=1)
            # Sort errors and baseline worst -> best actions
            dp = torch.arange(self.minibatch_size)
            pi_b = pi_b[dp[:, None], arg_sorted_qs]
            sorted_e = e[dp[:, None], arg_sorted_qs]
            for a_bot in range(self.nb_actions):
                mass_bot = torch.min(pi_b[:, a_bot], allowed_error / (2 * sorted_e[:, a_bot]))
                _, A_top = torch.max(
                    (_q2_net - sorted_qs[:, a_bot][:, None]) / e, dim=1)
                mass_top = torch.min(
                    mass_bot, allowed_error / (2 * e[dp, A_top]))
                mass_bot -= mass_top
                _pi_b[dp, arg_sorted_qs[:, a_bot]] -= mass_top
                _pi_b[dp, A_top] += mass_top
                allowed_error -= mass_top * (sorted_e[:, a_bot] + e[dp, A_top])
            return r + (1 - t) * self.gamma * torch.sum(q2 * _pi_b, 1)

        if self.learning_type == 'ramdp':
            bellman_target = _get_bellman_target_ramdp(c1)
        elif self.learning_type == 'regular' or self.minimum_count == 0: # shouldn't minimum_count be used in the
        # elif self.learning_type == 'regular':
            bellman_target = _get_bellman_target_dqn()
        elif self.learning_type == 'pi_b':
            bellman_target = _get_bellman_target_pi_b(c, pi_b)
        elif self.learning_type == 'pi_b_hat':
            total_states_visits = c.sum(axis=1)
            # avoid division problem
            # this does not change the result since the future value of terminal transitions is ignored
            total_states_visits[term] = 1
            pi_b_hat = c / total_states_visits[:, np.newaxis]
            bellman_target = _get_bellman_target_pi_b(c, pi_b_hat)
        elif self.learning_type == 'online_pi_b':
            bellman_target = _get_bellman_target_online_pi_b(c, pi_b)

        elif self.learning_type == 'soft_sort':
            bellman_target = _get_bellman_target_soft_sort(c, pi_b)
        else:
            raise ValueError('We did not recognize that learning type')

        # Huber loss
        errs = (bellman_target - q_pred).unsqueeze(1)
        quad = torch.min(torch.abs(errs), 1)[0]
        lin = torch.abs(errs) - quad
        loss = torch.sum(0.5 * quad.pow(2) + lin)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.add_scalar('loss', loss, self.training_step)
        self.training_step += 1
        return loss

    def get_q(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        return self.network(state / self.normalize).detach().cpu().numpy()

    def get_policy_distribution(self, states, counts):
        """

        :param states:
        :param counts: number of times each action was taken in the given state
        :return: action
        """
        states = np.expand_dims(states, 0)
        q_values = self.get_q(states)[0][0]
        if self.learning_type in ['pi_b', 'pi_b_hat'] and self.minimum_count > 0.0:
            mask = (counts < self.minimum_count)

            self.logger.add_scalar('bootstrapped_interaction', torch.sum(mask).item(), self.interaction_step)
            if self.learning_type == 'pi_b':
                _, _, policy, _ = self.baseline.inference(states[0])
            elif self.learning_type == 'pi_b_hat':
                # estimate policy according to visits counter
                total_state_visits = counts.sum()
                if total_state_visits > 0:
                    policy = counts/total_state_visits
                    self.logger.add_scalar('randompolicy', 0, self.interaction_step)
                else:
                    policy = np.ones(self.nb_actions) / self.nb_actions
                    self.logger.add_scalar('randompolicy', 1, self.interaction_step)
            pi_b = np.multiply(mask, policy)
            pi_b[np.argmax(q_values - mask*MAX_Q)] += np.maximum(0, 1 - np.sum(pi_b))
            pi_b /= np.sum(pi_b)
            return pi_b
        elif self.learning_type in ['online_pi_b', 'online_pi_b_hat'] and self.minimum_count > 0.0:
            mask = (counts < self.minimum_count)
            self.logger.add_scalar('bootstrapped_interaction', torch.sum(mask).item(), self.interaction_step)
            if self.learning_type == 'online_pi_b':
                _, _, policy, _ = self.baseline.inference(states[0])
            else:
                # estimate policy according to visits counter
                total_state_visits = counts.sum()
                if total_state_visits > 0:
                    policy = counts/total_state_visits
                    self.logger.add_scalar('randompolicy', 0, self.interaction_step)
                else:
                    policy = np.ones(self.nb_actions) / self.nb_actions
                    self.logger.add_scalar('randompolicy', 1, self.interaction_step)
            pi_b = np.multiply(mask, policy)
            pi_b[np.argmax(q_values - mask*MAX_Q)] += np.maximum(0, 1 - np.sum(pi_b))
            pi_b /= np.sum(pi_b)
            return pi_b
        elif self.learning_type == 'soft_sort' and self.epsilon_soft > 0.0:
            e = np.sqrt(1 / (np.array(counts) + 1e-9))
            _, _, policy, _ = self.baseline.inference(states[0])
            pi_b = np.array(policy)
            allowed_error = self.epsilon_soft
            A_bot = np.argsort(q_values)
            # Sort errors and baseline worst -> best actions
            policy = policy[A_bot]
            sorted_e = e[A_bot]
            for a_bot in range(self.nb_actions):
                mass_bot = min(policy[a_bot], allowed_error / (2 * sorted_e[a_bot]))
                A_top = np.argmax((q_values - q_values[A_bot[a_bot]]) / e)
                mass_top = min(mass_bot, allowed_error / (2 * e[A_top]))
                mass_bot -= mass_top
                pi_b[A_bot[a_bot]] -= mass_top
                pi_b[A_top] += mass_top
                allowed_error -= mass_top * (sorted_e[a_bot] + e[A_top])
            pi_b[pi_b < 0] = 0
            pi_b /= np.sum(pi_b)
            return pi_b
        elif self.learning_type == 'soft_sort' and self.epsilon_soft == 0.0:
            _, _, policy, _ = self.baseline.inference(states[0])
            return policy
        else:
            greedy_action = np.argmax(q_values)
            dist = np.zeros(shape=[self.nb_actions])
            dist[greedy_action] = 1
            return dist

    def get_action_and_policy(self, states, evaluate, counts=None):
        # get action WITH exploration
        eps = self.epsilon if not evaluate else self.test_epsilon

        policy = np.array(self.get_policy_distribution(states, counts=counts))

        policy_with_exploration = policy * (1.-eps) + np.ones(self.nb_actions) * eps
        policy_with_exploration /= np.sum(policy_with_exploration)
        action = np.random.choice(self.nb_actions, size=1, replace=True, p=policy_with_exploration)[0]
        self.interaction_step += 1
        return action, policy_with_exploration

    def learn_on_batch(self, batch):
        objective = self.train_on_batch(*batch)
        # updating target network
        if self.update_counter == self.update_freq:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)
            self.update_counter = 0
        else:
            self.update_counter += 1
        return objective

    def anneal_eps(self, step):
        """
        reduce the probability of taking random actions
        :param step:
        :return:
        """
        if self.epsilon > self.final_epsilon:
            decay = (self.start_epsilon - self.final_epsilon) * step / self.decay_steps
            self.epsilon = self.start_epsilon - decay
        if step >= self.decay_steps:
            self.epsilon = self.final_epsilon

    def update_lr(self, epoch):
        self.learning_rate = self.start_learning_rate / (epoch + 2)
        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate

    def dump_network(self, weights_file_path):
        torch.save(self.network.state_dict(), weights_file_path)

    def load_weights(self, weights_file_path, target=False):
        self.network.load_state_dict(torch.load(weights_file_path))
        if target:
            self.weight_transfer(from_model=self.network, to_model=self.target_network)

    @staticmethod
    def weight_transfer(from_model, to_model):
        to_model.load_state_dict(from_model.state_dict())

    def __getstate__(self):
        _dict = {k: v for k, v in self.__dict__.items()}
        del _dict['device']  # is not picklable
        del _dict['transitions']  # huge object (if you need the replay buffer, save its contnts with np.save)
        return _dict

    def needs_state_action_counter(self):
        return self.learning_type in ["pi_b", "pi_b_hat", "online_pi_b", "online_pi_b_hat"]

    def try_update_baseline(self, epoch):
        # print(">>>> Trying to update baseline")
        if self.baseline_update_freq and epoch % self.baseline_update_freq == 0:
            self.update_baseline()

    def update_baseline(self):
        print(">>>>> Updating baseline policy")
        self.weight_transfer(self.network, self.baseline.network)
        # self.baseline._copy_weight_from(self.network.state_dict())
