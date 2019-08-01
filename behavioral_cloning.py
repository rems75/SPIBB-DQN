import click
import torch
import os
import yaml

import numpy as np

from baseline import Baseline
from environments import environment
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
from torch import distributions
from dataset import Dataset_Counts
from tensorboardX import SummaryWriter

from utils import flush


@click.command()
@click.option('--number_of_epochs',
              default=100)
@click.option('--testing_steps',
              default=10000)
@click.option('--training_steps',
              default=2000)
@click.option('--mini_batch_size',
              default=32)
@click.option('--learning_rate',
              default=0.01,
              help="learning rate for optimizer")
@click.option('--network_size',
              default='dense')
@click.option('--network_path',
              default='./cloned_network_weights.pt')
@click.option('--folder_location',
              default='./baseline/helicopter_env/')
@click.option('--dataset_file',
              default='dataset/10000/123/1_0/counts_dataset.pkl')
@click.option('--state_shape',
              default=[4])
@click.option('--nb_actions',
              default=9)
@click.option('--device',
              default='cuda')
@click.option('--seed',
              default=123)
@click.option('--learning_rate',
              default=0.1)
@click.option('--entropy_coefficient',
              default=0.0)
@click.option('--experiment_name',
              default="")
@click.option('--config_file',
              default='config.yaml',
              help="config file to instantiate env and baseline to train sampling from environment")
@click.option('--sample_from_env', is_flag=True)
def train_behavior_cloning(training_steps, testing_steps, mini_batch_size, learning_rate, number_of_epochs, network_size,
                           folder_location, dataset_file, network_path,
                           state_shape, nb_actions, sample_from_env,
                           entropy_coefficient,
                           device, seed, experiment_name, config_file):
    # initialize seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    out_folder = os.path.join(os.getenv("PT_OUTPUT_DIR", './'), 'logs/' + experiment_name)
    data_dir = os.getenv("PT_DATA_DIR", os.path.join(folder_location))
    dataset_path = os.path.join(data_dir, dataset_file)
    network_path = os.path.join(os.path.dirname(dataset_path), network_path)

    logger = SummaryWriter(out_folder)

    # import data
    full_dataset = Dataset_Counts.load_dataset(dataset_path)
    dataset_train, dataset_test = full_dataset.train_test_split(test_size=0.2)

    # create model
    network = _build_network(network_size, state_shape, nb_actions, device)

    # define loss and optimizer

    nll_loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate, alpha=0.95, eps=1e-07)

    smaller_testing_loss = float('inf')

    # instantiate environment for policy evaluation
    try:
        params = yaml.safe_load(open(config_file, 'r'))
    except FileNotFoundError as e:
        print("Configuration file not found; Define a config_file to be able to sample from environment")
        raise e
    env = environment.Environment(params['domain'], params)

    if sample_from_env:
        print("sampling from environment")
        baseline_network_path = os.path.join(data_dir, params["network_path"])
        baseline = Baseline(baseline_network_path,
                            params['network_size'],
                            state_shape=params['state_shape'], nb_actions=params['nb_actions'],
                            seed=seed, temperature=params.get("baseline_temp", 0.1),
                            device=device, normalize=params['normalize'])
    else:
        baseline = None

    def train(data, data_test, current_epoch=0, log_frequency=200):
        for step in range(training_steps):
            # clear gradients
            optimizer.zero_grad()

            # sample mini_batch
            if not sample_from_env:
                s, a, behavior_policy, _, _, _, _, _, _ = data.sample(mini_batch_size=mini_batch_size, full_batch=True)
            else:
                # sanity check: train on new samples instead of fixed dataset
                mini_batch = Dataset_Counts(state_shape=params['state_shape'], nb_actions=params['nb_actions'],
                                            count_param=0.2)
                while mini_batch.size < mini_batch_size:
                    state = env.reset()
                    action, _, policy, _ = baseline.inference(state)
                    _, new_reward, term, _ = env.step(action)
                    mini_batch.add(s=state.astype('float32'), a=action, r=new_reward, t=term, p=policy)
                s, a, behavior_policy, _, _, _, _, _, _ = mini_batch.get_all_data()

            # prepare tensors
            batch_states = torch.FloatTensor(s).to(device)
            batch_states = torch.squeeze(batch_states)
            target = torch.LongTensor(a).to(device)  # NLLLoss gets the indexes of the correct class as input

            # get predictions
            estimated_policy = network.forward(batch_states)

            # computing losses
            # negative loglikelihood
            nll_loss = nll_loss_function(torch.log(estimated_policy), target)

            # policy entropy
            estimated_policy_entropy = torch.mean(distributions.Categorical(estimated_policy).entropy())
            # regularize entropy
            entropy_bonus = entropy_coefficient * estimated_policy_entropy

            total_loss = nll_loss - entropy_bonus

            if step % log_frequency == 0:
                # compute stats
                with torch.no_grad():

                    # makes an one-hot vector for the action
                    one_hot_behavior_policy = np.zeros(list(a.shape) + [nb_actions])
                    one_hot_behavior_policy[np.arange(len(a)), a] = 1
                    one_hot_behavior_policy = torch.FloatTensor(one_hot_behavior_policy).to(device)

                    # compute MSE loss of estimated probability with one_hot policy
                    mse_loss = F.mse_loss(estimated_policy, one_hot_behavior_policy)

                    # compute MSE with true policy
                    behavior_policy = torch.FloatTensor(behavior_policy).to(device)
                    mse_loss_true_policy = F.mse_loss(estimated_policy, behavior_policy)

                    # compute entropy of the behavior policy
                    behavior_policy_entropy = torch.mean(distributions.Categorical(behavior_policy).entropy())
                performance = evaluate_policy(policy=network, env=env, number_of_episodes=10, device=device)

                # logging stats
                s = 'step {:7d}, training: '
                s += 'nll{:7.3f}, '
                s += 'entropy_bonus {:7.6f}, '
                s += 'total_loss {:7.6f} '
                s += 'nll_loss_minus_pi_b_entropy {:7.6f} '
                total_steps = current_epoch * training_steps + step
                nll_loss_minus_pi_b_entropy = total_loss.item() - behavior_policy_entropy.item()
                print(s.format(total_steps,
                               nll_loss.item(),
                               entropy_bonus.item(),
                               total_loss.item(),
                               nll_loss_minus_pi_b_entropy))
                logger.add_scalar("training/nll_loss", nll_loss.item(), total_steps)
                logger.add_scalar("training/entropy_bonus", entropy_bonus.item(), total_steps)
                logger.add_scalar("training/total_loss", total_loss.item(), total_steps)
                logger.add_scalar("training/nll_loss_minus_entropy", nll_loss_minus_pi_b_entropy, total_steps)

                logger.add_scalar("estimated_policy/performance", performance, total_steps)
                logger.add_scalar("training/mse_a", mse_loss.item(), total_steps)
                logger.add_scalar("estimated_policy/entropy", estimated_policy_entropy.item(), total_steps)
                logger.add_scalar("estimated_policy/mse_pi_b", mse_loss_true_policy.item(), total_steps)

                # run test on test_dataset
                test(data_test, total_steps)
            # update weights
            total_loss.backward()
            optimizer.step()

        # log loss
        # logger.add_scalar('loss', loss, i)

    def test(data, total_steps):
        total_loss = 0
        for step in range(testing_steps):

            # sample mini_batch
            # indexes = np.random.choice(observations.shape[0], batch_size, replace=False)
            s, a, pi, r, s2, t, c, pi2, cl = data.sample(mini_batch_size=mini_batch_size, full_batch=True)
            batch_states = torch.FloatTensor(s).to(device)
            batch_states = torch.squeeze(batch_states)

            target = torch.LongTensor(a).to(device)  # NLLLoss gets the indexes of the correct class as input

            with torch.no_grad():
                # get predictions
                estimated_policy = network.forward(batch_states)
                # compute loss
                loss = nll_loss_function(torch.log(estimated_policy), target)
            total_loss += loss.item()
        average_loss = total_loss/testing_steps
        if average_loss < smaller_testing_loss:
            dump_network(network, network_path)

        s = 'testing accuracy: '
        s += 'negative log likelihood{:7.3f}, '
        print(s.format(average_loss))
        logger.add_scalar("testing/neg_log_likelihood", average_loss, total_steps)

        dump_network(network, network_path)

    for epoch in range(number_of_epochs):
        print("\nPROGRESS: {0:02.2f}%\n".format(epoch/number_of_epochs*100))
        train(dataset_train, dataset_test, epoch)

        flush(logger)
        update_lr(optimizer, epoch, start_learning_rate=learning_rate)


def evaluate_policy(policy, env, number_of_episodes, device):
    all_rewards = []
    for i in range(number_of_episodes):
        last_state = env.reset()
        term = False
        episode_reward = 0
        while not term:
            with torch.no_grad():
                state_tensor = torch.FloatTensor([last_state]).to(device)
                dist = distributions.Categorical(policy.forward(state_tensor))
                action = dist.sample().item()
            new_obs, new_reward, term, _ = env.step(action)
            last_state = new_obs
            episode_reward += new_reward
        all_rewards.append(episode_reward)
    return np.mean(all_rewards)


def dump_network(network, file_path):
    torch.save(network.state_dict(), file_path)


def update_lr(optimizer, epoch, start_learning_rate):
    new_learning_rate = start_learning_rate / (epoch + 2)
    for g in optimizer.param_groups:
        g['lr'] = new_learning_rate


def init_weights(m):
    """
    initializes the weights of the given module using a uniform distribution
    sets all the bias parameters to 0
    """
    if type(m) in [nn.Linear, nn.Conv2d]:
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)


class SmallDensePolicyNetwork(nn.Module):
    def __init__(self, state_shape, nb_actions, device):
        super(SmallDensePolicyNetwork, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(state_shape, 8)),
            ('relu_1', nn.ReLU()),
            ('linear_3', nn.Linear(8, 4)),
            ('relu_1', nn.ReLU()),
            ('linear_3', nn.Linear(4, nb_actions)),
            ('softmax_1', nn.Softmax(dim=1))
        ]))
        self.fc.apply(init_weights)
        super(SmallDensePolicyNetwork, self).to(device)

    def forward(self, x):
        x = self.fc(x)
        return x


class DensePolicyNetwork(nn.Module):
    def __init__(self, state_shape, nb_actions, device):
        super(DensePolicyNetwork, self).__init__()

        self.fc = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(state_shape, 32)),
            ('relu_1', nn.ReLU()),
            ('linear_2', nn.Linear(32, 128)),
            ('relu_2', nn.ReLU()),
            ('linear_3', nn.Linear(128, 32)),
            ('relu_3', nn.ReLU()),
            ('linear_4', nn.Linear(32, nb_actions)),
            ('softmax_1', nn.Softmax(dim=1))
        ]))
        self.fc.apply(init_weights)
        super(DensePolicyNetwork, self).to(device)

    def forward(self, x):
        x = self.fc(x)
        return x


def _build_network(network_size, state_shape, nb_actions, device):
    if network_size == 'dense':
        return DensePolicyNetwork(state_shape=state_shape[0], nb_actions=nb_actions, device=device)
    if network_size == 'small_dense':
        return SmallDensePolicyNetwork(state_shape=state_shape[0], nb_actions=nb_actions, device=device)
    raise ValueError('Invalid network_size.')


if __name__ == "__main__":
    train_behavior_cloning()

