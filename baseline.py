import argparse
import numpy as np
import os
import pickle
import time
import torch
import yaml

from dataset import DataSet, Dataset_Counts
from utils import softmax
from model import SmallDenseNetwork, DenseNetwork, Network, LargeNetwork, NatureNetwork
from environments import environment


SUFFIX = 'dataset.pkl'
COUNTS_SUFFIX = 'counts_dataset.pkl'


class Baseline(object):

    def __init__(self, network_path, network_size, state_shape=[4], nb_actions=9,
                 device='cuda', seed=123, temperature=1.0, normalize=255.):

        self.seed = seed
        self.state_shape = state_shape
        self.nb_actions = nb_actions
        self.network_size = network_size
        self.device = device
        self.temperature = temperature
        self.normalize = normalize
        self.network = self._build_network()
        self._load_model(network_path)
        print("Using soft q-values with a temperature of {}".format(temperature), flush=True)

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

    def _load_model(self, network_path):
        if not os.path.exists(network_path):
            raise ValueError('Missing model at location {}'.format(network_path))
        print('Loading model from {}'.format(network_path), flush=True)
        self.network.load_state_dict(torch.load(network_path))

    def dump_network(self, weights_file_path):
        torch.save(self.network.state_dict(), weights_file_path)

    def set_temp(self, temp):
        self.temperature = temp

    def get_q_values(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        return self.network(state / self.normalize).detach().cpu().numpy()

    def inference(self, state):
        q_values = self.get_q_values(state)
        # Use soft q-values
        p = softmax(q_values[0], temperature=self.temperature, axis=0)
        choice = np.random.choice(self.nb_actions, 1, p=p)[0]
        return choice, q_values, p, choice == np.argmax(p)

    # Returns an np array containing the policy generated by the average of the models used for the baseline
    def compute_policy(self, state):
        q_values = self.get_q_values(state)
        return softmax(q_values, temperature=self.temperature, axis=1)

    def _reset(self, env):
        # Choose a new policy to govern the trajectory
        return env.reset()

    def _update_state(self, new_obs, last_state):
        return new_obs.flatten()

    def generate_dataset(self, env, path, params, dataset_size=1e6, overwrite=False, noise_factor=1):
        """ Generates a dataset using the loaded baseline

        Args:
          path: path to the folder where the dataset will be written (minus the subfolder related to different generation parameters)
          params: the configuration file loaded in run
          dataset_size: the size of the dataset to generate
          overwrite: whether to overwrite an existing dataset of the same size, generated with the same seed and same noise_factor.
          noise_factor: the noise factor additionally applied to the environment. 1 in our experiments.
        Returns:
          The dataset object containing dataset_size transitions generated from the baseline, the dataset is also saved in
            path/dataset/{dataset_size}/{seed}/{noise_factor}/dataset.pkl.
        """

        dataset = DataSet(path, dataset_size=dataset_size, state_shape=self.state_shape, state_dtype=np.float32, nb_actions=self.nb_actions)

        dataset.dataset_folder = os.path.join(str(dataset_size), str(self.seed), str(noise_factor).replace('.', '_'))
        file_path = os.path.join(dataset.dataset_folder, SUFFIX)
        if os.path.exists(os.path.join(dataset.path, file_path)):
            if overwrite:
                print('Found existing dataset. Removing it.', flush=True)
                os.remove(os.path.join(dataset.path, file_path))
            else:
                print('Found existing dataset.', flush=True)
                dataset.load_dataset(file_path)
                return dataset

        last_state = np.empty(tuple(env.state_shape), dtype=np.uint8)
        last_state = self._reset(env)
        term, start_time = False, time.time()
        rewards, all_nb_steps, current_reward, nb_steps = [], [], 0, 0
        while dataset.size < dataset_size:
            if dataset.size % 10000 == 0 and dataset.size > 0:
                print("Generated: {} samples in {} seconds".format(dataset.size, time.time() - start_time), flush=True)
            if not term:
                action, qfunction, policy, _ = self.inference(last_state)
                new_obs, new_reward, term, _ = env.step(action)
                dataset.add(s=last_state.astype('float32'), a=action, r=new_reward, t=term, p=policy, q=qfunction)
                last_state = self._update_state(new_obs, last_state)
                current_reward += new_reward
                nb_steps += 1
            else:
                last_state = self._reset(env)
                rewards.append(current_reward)
                all_nb_steps.append(nb_steps)
                current_reward, nb_steps, term = 0, 0, False

        print("Generated: {} samples in {} seconds".format(dataset.size, time.time() - start_time), flush=True)
        print("Average reward: {}. Average steps: {}.".format(np.mean(rewards), np.mean(all_nb_steps)), flush=True)
        dataset.save_dataset(file_path)
        return dataset

    def evaluate_baseline(self, env, params, number_of_steps, number_of_epochs, noise_factor=1.0):
        """ Evaluate the baseline number_of_epochs times for number_of_steps steps.

        Args:
          number_of_steps: number of steps to simulate during each epoch
          number_of_epochs: number of epochs to simulate
          noise_factor: the noise factor additionally applied to the environment. 1 in our experiments.
        Returns:
          Prints the mean performance on each epoch. And the mean, 10% and 1% CVAR of the performance on those epochs.
        """

        all_rewards = []
        for epoch in range(number_of_epochs):
            print("Starting epoch {}".format(epoch), flush=True)
            last_state = np.empty(tuple(env.state_shape), dtype=np.uint8)
            last_state = self._reset(env)
            term, start_time = False, time.time()
            rewards, all_nb_steps, current_reward, nb_steps, total_nb_steps = [], [], 0, 0, 0

            while total_nb_steps < number_of_steps:
                if not term:
                    action, _, _, _ = self.inference(last_state)
                    new_obs, new_reward, term, _ = env.step(action)
                    last_state = self._update_state(new_obs, last_state)
                    current_reward += new_reward
                    nb_steps += 1
                else:
                    last_state = self._reset(env)
                    rewards.append(current_reward)
                    all_nb_steps.append(nb_steps)
                    total_nb_steps += nb_steps
                    current_reward, nb_steps = 0, 0
                    term = False

            all_rewards.append(np.mean(rewards))
            print("Average reward: {}. Average steps: {}".format(
                np.mean(rewards), np.mean(all_nb_steps)), flush=True)
            print("Epoch finished in {:.2f} seconds.\n".format(time.time() - start_time), flush=True)

        all_rewards.sort()
        print("Mean Average: {}.".format(np.mean(all_rewards)), flush=True)
        if number_of_epochs > 10:
            print("Average decile: {}.".format(np.mean(all_rewards[:int(number_of_epochs/10)])), flush=True)
        if number_of_epochs > 100:
            print("Average centile: {}".format(
                np.mean(all_rewards[:int(number_of_epochs/100)])), flush=True)


def compute_counts(dataset, overwrite=False, count_param=0.2):
    """ Compute the pseudo-counts for each state-action pair present in the dataset following the methodology described in the paper.

    Args:
      dataset: the dataset instance for which to computed counts
      overwrite: whether to overwrite an existing counts file of the same size, generated with the same seed and same noise_factor.
    Returns:
      Saves the dataset augmented with the counts in /dataset/{dataset_size}/{seed}/{noise_factor}/counts_dataset.pkl.
    """

    full_path = os.path.join(dataset.path, dataset.dataset_folder, COUNTS_SUFFIX)
    if os.path.isfile(full_path):
        if overwrite:
            print("Found existing counts file. Overwriting.", flush=True)
            os.remove(full_path)
        else:
            print("Found existing counts file. Aborting.", flush=True)
            return

    t = time.time()
    print("Computing counts. The dataset contains {} transitions.".format(len(dataset.states)), flush=True)
    d = Dataset_Counts.from_dataset(dataset, count_param)
    print("Saving data with counts to {}".format(full_path), flush=True)
    d.save_dataset(full_path)
    print("Data with counts saved, {} samples".format(d.size), flush=True)
    print("Counts computed in " + str(time.time() - t) + " seconds", flush=True)


def run(args):
    """ Either generates a dataset from a baseline and computes its associated counts, or evaluates a baseline.
    """
    for fff in os.listdir(args.baseline_dir):
        if fff.endswith(".yaml"):
            yaml_file = os.path.join(args.baseline_dir, fff)
            params = yaml.safe_load(open(yaml_file, 'r'))
            print('Loading config from {}'.format(yaml_file))
            break
    else:
        # no yaml file found
        raise ValueError('We could not find the configuration file for the baseline, it should be a yaml file.')

    if args.seed is not None:
        # if seed is given in the command line ignores seed from yaml file
        np.random.seed(args.seed)
        params['seed'] = args.seed
    else:
        if 'seed' in params:
            args.seed = params['seed']
        else:
            print("no seed found, using 123")
            args.seed = 123
            params['seed'] = 123

    if args.extra_stochasticity > 0.0:
        params['extra_stochasticity'] = args.extra_stochasticity

    env = environment.Environment(params['domain'], params)

    baseline = Baseline(os.path.join(args.baseline_dir, args.baseline_name), params['network_size'], state_shape=params['state_shape'],
                        nb_actions=params['nb_actions'], seed=args.seed, temperature=args.temperature,
                        device=args.device, normalize=params['normalize'])

    if args.evaluate_baseline:
        baseline.evaluate_baseline(env, params, args.eval_steps, args.eval_epochs, args.noise_factor)

    if args.generate_dataset:
        print("Generating dataset with actual size {}...".format(args.dataset_size), flush=True)
        dataset = baseline.generate_dataset(
            env, os.path.join(args.baseline_dir, args.dataset_dir), params, dataset_size=args.dataset_size,
            overwrite=args.overwrite, noise_factor=args.noise_factor)

        compute_counts(dataset, overwrite=args.overwrite, count_param=args.count_param)


if __name__ == '__main__':
    # TODO switch command line options to click like in train.py
    parser = argparse.ArgumentParser(description="Options")
    parser.add_argument('baseline_dir', type=str, default='baseline',
                        help='path of the baseline')
    parser.add_argument('baseline_name', type=str, default='weights.pt',
                        help='file containing the weights of thep baseline policy')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature used for the soft q-values')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu')

    # Arguments for baseline evaluation
    parser.add_argument('--evaluate_baseline',
                        action="store_true", help='evaluate the baseline')
    parser.add_argument('--eval_steps', type=int, default=10000)
    parser.add_argument('--eval_epochs', type=int, default=300)

    # Arguments for the dataset generation
    parser.add_argument('--generate_dataset',
                        action="store_true", help='generate a dataset')
    parser.add_argument('--dataset_size', type=int, default=100,
                        help='number of transitions in the dataset')
    parser.add_argument('--noise_factor', type=float, default=1.0,
                        help='amount of noise in the environment')
    parser.add_argument('--extra_stochasticity', type=float, default=0.0,
                        help='additional noise in the actions')
    parser.add_argument('--count_param', type=float, default=0.2,
                        help='param for similarity')
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='path where to save the dataset')
    parser.add_argument('--overwrite', action="store_true",
                        help='overwrite existing dataset')

    run(parser.parse_args())
