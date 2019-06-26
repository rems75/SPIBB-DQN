import click
import numpy as np
import os
import pickle
import random
import sys
import torch
import yaml

from ai import AI
from experiment import DQNExperiment, BatchExperiment
from dataset import Dataset_Counts
from environments import environment

@click.command()
@click.option('--domain', '-d', default='helicopter', help="'helicopter' or 'catch' or 'atari'")
@click.option('--config', '-c', default=None, help="config file name, if not specified, config_\{domain\}")
@click.option('--options', '-o', multiple=True, nargs=2, type=click.Tuple([str, str]))
def run(domain, config, options):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    if not config:
        config = 'config_' + domain
    cfg_file = os.path.join(dir_path, config + '.yaml')
    params = yaml.safe_load(open(cfg_file, 'r'))

    # replacing params with command line options
    for opt in options:
        assert opt[0] in params
        dtype = type(params[opt[0]])
        if dtype == bool:
            new_opt = False if opt[1] != 'True' else True
        else:
            new_opt = dtype(opt[1])
        params[opt[0]] = new_opt

    print('\n')
    print('Parameters ')
    for key in params:
        print(key, params[key])
    print('\n')

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    random_state = np.random.RandomState(params['seed'])
    device = torch.device(params["device"])

    DATA_DIR = os.path.join(params['folder_location'], params['folder_name'])

    env = environment.Environment(domain, params, random_state)

    if params['batch']:
        from baseline import Baseline
        baseline_path = os.path.join(DATA_DIR, params['baseline_path'])
        baseline = Baseline(baseline_path, params['network_size'], state_shape=params['state_shape'],
                            nb_actions=params['nb_actions'], seed=params['seed'], temperature=params['baseline_temp'],
                            device=params['device'], normalize=params['normalize'])

        dataset_path = os.path.join(DATA_DIR, params['dataset_path'])
        print("\nLoading dataset from file {}".format(dataset_path), flush=True)
        if not os.path.exists(dataset_path):
            raise ValueError("The dataset file does not exist")
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)
        dataset = Dataset_Counts(data, params['count_param'])
        print("Data with counts loaded: {} samples".format(len(data['s'])), flush=True)
        folder_name = os.path.dirname(dataset_path)
        expt = BatchExperiment(dataset=dataset, env=env, folder_name= folder_name, episode_max_len=params['episode_max_len'],
                               minimum_count=params['minimum_count'], extra_stochasticity=params['extra_stochasticity'],
                               history_len=params['history_len'], max_start_nullops=params['max_start_nullops'])

    else:
        # Create experiment folder
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        baseline = None
        expt = DQNExperiment(env=env, ai=None, episode_max_len=params['episode_max_len'], annealing=params['annealing'],
                             history_len=params['history_len'], max_start_nullops=params['max_start_nullops'],
                             replay_min_size=params['replay_min_size'], test_epsilon=params['test_epsilon'],
                             folder_name=DATA_DIR, network_path=params['network_path'], extra_stochasticity=params['extra_stochasticity'],
                             score_window_size=100)

    for ex in range(params['num_experiments']):
        print('\n')
        print('>>>>> Experiment ', ex, ' >>>>> ',
              params['learning_type'], ' >>>>> Epsilon >>>>> ',
              params['epsilon_soft'], ' >>>>> Minimum Count >>>>> ',
              params['minimum_count'], ' >>>>> Kappa >>>>> ',
              params['kappa'], ' >>>>> ', flush=True)
        print('\n')
        ai = AI(baseline, state_shape=env.state_shape, nb_actions=env.nb_actions, action_dim=params['action_dim'],
                reward_dim=params['reward_dim'], history_len=params['history_len'], gamma=params['gamma'],
                learning_rate=params['learning_rate'], epsilon=params['epsilon'], final_epsilon=params['final_epsilon'],
                test_epsilon=params['test_epsilon'], annealing_steps=params['annealing_steps'], minibatch_size=params['minibatch_size'],
                replay_max_size=params['replay_max_size'], update_freq=params['update_freq'],
                learning_frequency=params['learning_frequency'], ddqn=params['ddqn'], learning_type=params['learning_type'],
                network_size=params['network_size'], normalize=params['normalize'], device=device,
                kappa=params['kappa'], minimum_count=params['minimum_count'], epsilon_soft=params['epsilon_soft'])
        expt.ai = ai

        env.reset()
        with open(expt.folder_name + '/config.yaml', 'w') as y:
            yaml.safe_dump(params, y)  # saving params for reference
        expt.do_epochs(number_of_epochs=params['num_epochs'], is_learning=params['is_learning'],
                       steps_per_epoch=params['steps_per_epoch'], is_testing=params['is_testing'],
                       steps_per_test=params['steps_per_test'],
                       passes_on_dataset=params['passes_on_dataset'], exp_id=ex)


if __name__ == '__main__':
    run()
