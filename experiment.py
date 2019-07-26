import csv
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime

from utils import write_to_csv, plot


class DQNExperiment(object):
    def __init__(self, env, ai, episode_max_len, annealing=False, history_len=1, max_start_nullops=1, test_epsilon=0.0,
                 replay_min_size=0, score_window_size=100, folder_name='expt',
                 saving_period=1, network_path='weights.pt', extra_stochasticity=0.0):
        self.fps = 0
        self.episode_num = 0
        self.last_episode_steps = 0
        self.total_training_steps = 0
        self.score_computer = 0
        self.score_agent = 0
        self.eval_scores = []
        self.eval_steps = []
        self.env = env
        self.ai = ai
        self.history_len = history_len
        # self.annealing = annealing  # QUESTION: can we remove this parameter? it does not seem to be used
        self.test_epsilon = test_epsilon
        self.max_start_nullops = max_start_nullops
        self.saving_period = saving_period  # after each `saving_period` epochs, the results so far will be saved.
        self.episode_max_len = episode_max_len
        self.score_agent_window = np.zeros(score_window_size)
        self.steps_agent_window = np.zeros(score_window_size)
        self.replay_min_size = replay_min_size
        if self.history_len > 1:
            self.last_state = np.empty(tuple([self.history_len] + self.env.state_shape), dtype=np.uint8)
        else:
            self.last_state = np.empty(tuple(self.env.state_shape), dtype=np.uint8)
        self.folder_name = folder_name
        self.network_path = network_path
        self.curr_epoch = 0
        self.all_rewards = []
        self.extra_stochasticity = extra_stochasticity

        self.dataset_counter = None
        self.steps = 0

        self.logger = get_logger(self.folder_name)

    def do_epochs(self, number_of_epochs=1, steps_per_epoch=10000, is_learning=True, is_testing=True, steps_per_test=10000, **kwargs):
        best_perf = -10000
        rewards_over_all_episodes = []
        self.ai.logger = self.logger
        for epoch in range(self.curr_epoch, number_of_epochs):

            if is_testing:
                eval_steps = 0
                eval_episodes = 0
                eval_scores = 0
                print('Evaluation ...', flush=True)
                with tqdm(total=steps_per_test, desc=">>>>> Evaluation Step ", file=sys.stdout) as progress_bar:
                    while eval_steps < steps_per_test:
                        eval_scores += self.evaluate(number_of_epochs=1)
                        eval_steps += self.last_episode_steps
                        eval_episodes += 1
                        progress_bar.update(self.last_episode_steps)

                self.eval_scores.append(eval_scores / eval_episodes)
                self.eval_steps.append(eval_steps / eval_episodes)
                print('Average performance on {} episodes: {}'.format(eval_episodes, eval_scores / eval_episodes), flush=True)
                self._plot_and_write(plot_dict={'eval_scores': self.eval_scores}, loc=self.folder_name + "/scores",
                                     x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                     moving_average=True)
                self._plot_and_write(plot_dict={'eval_steps': self.eval_steps}, loc=self.folder_name + "/steps",
                                     x_label="Epochs", y_label="Mean Steps", title="", kind='line', legend=True)

                if eval_scores / eval_episodes > best_perf:
                    best_perf = eval_scores / eval_episodes
                    self.ai.dump_network(weights_file_path=os.path.join(self.folder_name, self.network_path))
                    print('Saving best network', flush=True)
                self.all_rewards.append(eval_scores / eval_episodes)

            print('\n')
            print('=' * 60, flush=True)
            print('>>>>>  Epoch {} / {}  >>>>> Current eps  {:.2f} '.format(epoch + 1, number_of_epochs, self.ai.epsilon), flush=True)
            print('Training ...', flush=True)
            b = time.time()
            self.steps = 0
            epoch_rewards = []
            training_episodes = 0
            with tqdm(total=steps_per_epoch, desc=">>>>> Training Step ", file=sys.stdout) as progress_bar:
                while self.steps < steps_per_epoch:
                    new_rewards = self.do_episodes(number_of_epochs=1, is_learning=is_learning)
                    epoch_rewards.extend(new_rewards)
                    self.logger.add_scalar('episode_reward', new_rewards[0], self.total_training_steps)
                    training_episodes += 1
                    progress_bar.update(self.last_episode_steps)

            print("Epoch ran in {:.1f} seconds".format(time.time() - b), flush=True)
            print("Epoch ran {} episodes".format(training_episodes), flush=True)
            print("Average performance during training: {:.1f} ".format(np.array(epoch_rewards).mean()))
            print("Dataset has: {:.1f} transitions".format(self.dataset_counter.size))
            rewards_over_all_episodes.extend(epoch_rewards)
            self._plot_and_write(plot_dict={'training_scores': rewards_over_all_episodes}, loc=self.folder_name + "/training_scores",
                                 x_label="Epochs", y_label="Mean Score", title="", kind='line', legend=True,
                                 moving_average=True)

            self.ai.anneal_eps(epoch * steps_per_epoch)
            self.ai.update_lr(epoch)
            self.ai.try_update_baseline(epoch)

        print("Best performance: {}".format(best_perf), flush=True)

    def do_episodes(self, number_of_epochs=1, is_learning=True):
        all_rewards = []
        for _ in range(number_of_epochs):
            episode_rewards = self._do_episode(is_learning=is_learning)
            all_rewards.append(sum(episode_rewards))
            self.score_agent_window = self._update_window(self.score_agent_window, self.score_agent)
            self.steps_agent_window = self._update_window(self.steps_agent_window, self.last_episode_steps)
            if self.episode_num % 1000 == -1:
                print_string = ("\nSteps: {0} | Fps: {1} | Eps: {2} | Score: {3} | Agent Moving Avg: {4} | "
                                "Agent Moving Steps: {5} | Total Steps: {6} ")
                print('=' * 30, flush=True)
                print('::Episode::  ' + str(self.episode_num), flush=True)
                print(print_string.format(self.last_episode_steps, self.fps, round(self.ai.epsilon, 2),
                                          round(self.score_agent, 2), round(np.mean(self.score_agent_window), 2),
                                          np.mean(self.steps_agent_window), self.total_training_steps), flush=True)
            self.episode_num += 1
        return all_rewards

    def evaluate(self, number_of_epochs=1):
        for num in range(number_of_epochs):
            self._do_episode(is_learning=False, evaluate=True)
        # TODO this only returns the score of the last episode
        return self.score_agent

    # QUESTION: is_learning seems to be the opposite of evaluate, can we use only one?
    def _do_episode(self, is_learning=True, evaluate=False):
        assert (is_learning and not evaluate) or (not is_learning and evaluate)
        rewards = []
        self._episode_reset()
        term = False
        self.fps = 0
        while not term:
            reward, term = self._step(evaluate=evaluate)
            rewards.append(reward)
            self.score_agent += reward

            if self.dataset_counter.size >= max(self.replay_min_size, self.ai.minibatch_size) and is_learning \
                    and (self.steps % self.ai.learning_frequency) == 0:
                mini_batch = self.dataset_counter.sample(self.ai.minibatch_size)
                loss = self.ai.learn_on_batch(mini_batch)
                self.ai.update_boltzmann_parameter(self.episode_num)

            if not term and self.last_episode_steps >= self.episode_max_len:
                print('Reaching maximum number of steps in the current episode.', flush=True)
                term = True
        return rewards

    def _step(self, evaluate=False):
        self.last_episode_steps += 1
        if self.ai.needs_state_action_counter():
            # TODO try to reuse this counts since they are computed again when added to the dataset during training
            counts = self.dataset_counter.compute_counts(self.last_state)
        else:
            counts = None
        action, policy = self.ai.get_action_and_policy(self.last_state, evaluate, counts)
        new_obs, reward, game_over, _ = self.env.step(action)
        if new_obs.ndim == 1 and len(self.env.state_shape) == 2:
            new_obs = new_obs.reshape(self.env.state_shape)
        if not evaluate:
            self.steps += 1
            if self.history_len > 1:
                state = self.last_state[-1].astype('float32')
            else:
                state = self.last_state.astype('float32')
            # self.ai.transitions.add(s=state, a=action, r=reward, t=game_over)
            self.dataset_counter.add(s=state, a=action, r=reward, t=game_over, p=policy)
            self.total_training_steps += 1
        self._update_state(new_obs)
        return reward, game_over

    def _episode_reset(self):
        obs = self.env.reset()
        self.last_episode_steps = 0
        self.score_agent = 0
        self.score_computer = 0
        assert self.max_start_nullops >= self.history_len or self.max_start_nullops == 0
        if self.max_start_nullops != 0:
            num_nullops = np.random.randint(self.history_len, self.max_start_nullops)
            for i in range(num_nullops - self.history_len):
                self.env.step(0)
        if self.history_len > 1:
            for i in range(self.history_len):
                if i > 0:
                    self.env.step(0)
                obs = self.env.get_state()
                if obs.ndim == 1 and len(self.env.state_shape) == 2:
                    obs = obs.reshape(self.env.state_shape)
                self.last_state[i] = obs
        else:
            self.last_state = obs

    def _update_state(self, new_obs):
        if self.history_len > 1:
            temp_buffer = np.empty(self.last_state.shape, dtype=np.uint8)
            temp_buffer[:-1] = self.last_state[-self.history_len + 1:]
            temp_buffer[-1] = new_obs
            self.last_state = temp_buffer
        else:
            self.last_state = new_obs

    @staticmethod
    def _plot_and_write(plot_dict, loc, x_label="", y_label="", title="", kind='line', legend=True,
                        moving_average=False):
        for key in plot_dict:
            plot(data={key: plot_dict[key]}, loc=loc + str(key) + ".pdf", x_label=x_label, y_label=y_label, title=title,
                 kind=kind, legend=legend, index_col=None, moving_average=moving_average)
            write_to_csv(data={key: plot_dict[key]}, loc=loc + ".csv")

    @staticmethod
    def _update_window(window, new_value):
        window[:-1] = window[1:]
        window[-1] = new_value
        return window


class BatchExperiment(object):
    def __init__(self, dataset=None, env=None, ai=None, episode_max_len=1000, folder_name='/experiments',
                 minimum_count=0, max_start_nullops=0, history_len=1, extra_stochasticity=0.0):

        self.dataset = dataset
        self.last_episode_steps = 0
        self.score_agent = 0
        self.env = env
        self.ai = ai
        self.folder_name = folder_name
        self.max_start_nullops = max_start_nullops
        self.history_len = history_len
        self.extra_stochasticity = extra_stochasticity
        self.episode_max_len = episode_max_len
        self.logger = get_logger(self.folder_name)


    def do_epochs(self, number_of_epochs=1, steps_per_test=10000, exp_id=0, passes_on_dataset=1, **kwargs):
        if self.ai.learning_type == 'soft_sort':
            filename = os.path.join(self.folder_name, "soft_{}_{}.csv".format(exp_id, self.ai.epsilon_soft))
        elif self.ai.learning_type == 'ramdp':
            filename = os.path.join(self.folder_name, "ramdp_{}_{:.2f}.csv".format(exp_id, self.ai.kappa))
        elif self.ai.learning_type == 'pi_b':
            filename = os.path.join(self.folder_name, "spibb_{}_{}.csv".format(exp_id, self.ai.minimum_count))
        elif self.ai.learning_type == 'pi_b_hat':
            filename = os.path.join(self.folder_name, "spibb_hat_{}_{}.csv".format(exp_id, self.ai.minimum_count))
        elif self.ai.learning_type == 'regular':
            filename = os.path.join(self.folder_name, "dqn_{}.csv".format(exp_id))
        else:
            raise ValueError()
        try:
            os.remove(filename)
        except OSError:
            pass
        self.ai.logger = self.logger
        total_steps, updates = 0, 0
        for epoch in range(number_of_epochs):
            begin = time.time()
            print('=' * 30, flush=True)
            print('>>>>> Epoch  ' + str(epoch) + '/' + str(number_of_epochs - 1) + '  >>>>>', flush=True)
            for _ in tqdm(range(passes_on_dataset), desc=">>>>> Pass ", file=sys.stdout, disable=True):
                steps = 0
                while steps < self.dataset.size:
                    mini_batch = self.dataset.sample(self.ai.minibatch_size, full_batch=True)
                    self.ai.learn_on_batch(mini_batch)
                    steps += self.ai.minibatch_size
                    total_steps += self.ai.minibatch_size
                    # Update learning rate every pass on the dataset or every 20000 steps whichever is larger
                    if 0 <= total_steps % max(20000, self.dataset.size) < self.ai.minibatch_size:
                        self.ai.update_lr(updates)
                        updates += 1

            print('>>>>> Training ran in {} seconds.'.format(time.time() - begin), flush=True)
            print('>>>>> Start testing.', flush=True)
            begin_testing = time.time()
            if steps_per_test > 0:
                eval_steps = 0
                eval_episodes = 0
                eval_scores = 0
                while eval_steps < steps_per_test:
                    eval_scores += self.evaluate(print_score=False)
                    eval_steps += self.last_episode_steps
                    eval_episodes += 1
            # flush(self.ai.logger)
            print('>>>>> Testing ran in {} seconds.'.format(time.time() - begin_testing), flush=True)
            print('>>>>> Average performance {}.'.format(eval_scores / eval_episodes), flush=True)

            with open(filename, 'a') as f:
                csv_file_writer = csv.writer(f)
                csv_file_writer.writerow(
                    [epoch, eval_scores / eval_episodes, eval_steps / eval_episodes, eval_episodes])
            print('>>>>> Results written in {}.'.format(filename), flush=True)

    def evaluate(self, print_score=False):
        self._do_episode(print_score=print_score)
        return self.score_agent

    def _do_episode(self, print_score=False):
        self._reset()
        term = False
        while not term:
            reward, term = self._step()
            if print_score:
                print(reward, flush=True)
            self.score_agent += reward
            if not term and self.last_episode_steps >= self.episode_max_len:
                print('Reaching maximum number of steps in the current episode.')
                term = True

    def _step(self):
        self.last_episode_steps += 1
        if self.ai.minimum_count > 0 or self.ai.epsilon_soft > 0:
            counts = self.dataset.compute_counts(self.last_state)
        else:
            counts = []
        action, policy = self.ai.get_action_and_policy(self.last_state, evaluate=True, counts=counts)
        new_obs, reward, game_over, _ = self.env.step(action)
        self._update_state(new_obs)
        return reward, game_over

    def _reset(self):
        obs = self.env.reset()
        self.last_episode_steps = 0
        self.score_agent = 0
        if self.max_start_nullops != 0:
            num_nullops = np.random.randint(self.history_len, self.max_start_nullops)
            for i in range(num_nullops):
                self.env.step(0)
        if self.history_len > 1:
            for i in range(self.history_len):
                if i > 0:
                    self.env.step(0)
                obs = self.env.get_state()
                if obs.ndim == 1 and len(self.env.state_shape) == 2:
                    obs = obs.reshape(self.env.state_shape)
                self.last_state[i] = obs
        else:
            self.last_state = obs

    def _update_state(self, new_obs):
        self.last_state = new_obs


def get_logger(folder_name):
    now = datetime.now()
    log_path = os.path.join(folder_name, 'logs', now.strftime("%Y%m%d-%H%M%S"))
    print(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    return SummaryWriter(log_path)
