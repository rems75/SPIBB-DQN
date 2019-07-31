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
@click.option('--experiment_name',
              default="")
@click.option('--config_file',
              default='config.yaml',
              help="config file to instantiate env and baseline to train sampling from environment")
@click.option('--sample_from_env', is_flag=True)
def train_behavior_cloning(training_steps, testing_steps, mini_batch_size, learning_rate, number_of_epochs, network_size,
                           folder_location, dataset_file, network_path,
                           state_shape, nb_actions, sample_from_env,
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

    if sample_from_env:
        print("sampling from environment")
        try:
            params = yaml.safe_load(open(config_file, 'r'))
        except FileNotFoundError as e:
            print("Configuration file not found; Define a config_file to be able to sample from environment")
            raise e
        baseline_network_path = os.path.join(data_dir, params["network_path"])
        baseline = Baseline(baseline_network_path,
                            params['network_size'],
                            state_shape=params['state_shape'], nb_actions=params['nb_actions'],
                            seed=seed, temperature=params.get("baseline_temp", 0.1),
                            device=device, normalize=params['normalize'])

        env = environment.Environment(params['domain'], params)
    else:
        baseline, env = None, None

    def train(data, current_epoch=0, log_frequency=200):
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
            estimated_probabilities, log_probabilities = network.forward(batch_states)

            # compute losses
            loss = nll_loss_function(log_probabilities, target)

            if step % log_frequency == 0:
                with torch.no_grad():


                    # makes an one-hot vector for the action
                    one_hot_behavior_policy = np.zeros(list(a.shape) + [nb_actions])
                    one_hot_behavior_policy[np.arange(len(a)), a] = 1
                    one_hot_behavior_policy = torch.FloatTensor(one_hot_behavior_policy).to(device)

                    # compute MSE loss of estimated probability with one_hot policy
                    mse_loss = F.mse_loss(estimated_probabilities, one_hot_behavior_policy)

                    # compute MSE with true policy
                    behavior_policy = torch.FloatTensor(behavior_policy).to(device)
                    mse_loss_true_policy = F.mse_loss(estimated_probabilities, behavior_policy)

                    # compute entropy of the behavior policy
                    behavior_policy_entropy = torch.mean(distributions.Categorical(behavior_policy).entropy())

                s = 'step {:7d}, training accuracy: '
                s += 'negative log likelihood{:7.3f}, '
                s += 'mse a {:7.6f}, '
                s += 'mse pi {:7.6f} '
                s += 'normalized loss {:7.6f} '
                total_steps = current_epoch * training_steps + step
                print(s.format(total_steps, loss.item(), mse_loss.item(), mse_loss_true_policy.item(), loss.item() - behavior_policy_entropy.item()))
                logger.add_scalar("training/mse_a", mse_loss.item(), total_steps)
                logger.add_scalar("training/mse_pi_b", mse_loss_true_policy.item(), total_steps)
                logger.add_scalar("training/total_loss", loss.item(), total_steps)
                logger.add_scalar("training/training_loss_minus_entropy", loss.item() - behavior_policy_entropy.item(), total_steps)


            # update weights
            loss.backward()
            optimizer.step()

        # log loss
        # logger.add_scalar('loss', loss, i)

    def test(data, current_epoch):
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
                log_probabilities = network.forward(batch_states)
                # compute loss
                loss = nll_loss_function(log_probabilities, target)
            total_loss += loss.item()
        average_loss = total_loss/training_steps
        if average_loss < smaller_testing_loss:
            dump_network(network, network_path)

        s = 'epoch {:7d}, testing accuracy: '
        s += 'negative log likelihood{:7.3f}, '
        print(s.format(current_epoch, average_loss))
        logger.add_scalar("testing/neg_log_likelihood", loss.item(), current_epoch)

        dump_network(network, network_path)

    for epoch in range(number_of_epochs):
        print("\nPROGRESS: {0:02.2f}%\n".format(epoch/number_of_epochs*100))
        train(dataset_train, epoch)
        test(dataset_test, epoch)
        flush(logger)
        update_lr(optimizer, epoch, start_learning_rate=learning_rate)


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
        return x, torch.log(x)


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
        return x, torch.log(x)


def _build_network(network_size, state_shape, nb_actions, device):
    if network_size == 'dense':
        return DensePolicyNetwork(state_shape=state_shape[0], nb_actions=nb_actions, device=device)
    if network_size == 'small_dense':
        return SmallDensePolicyNetwork(state_shape=state_shape[0], nb_actions=nb_actions, device=device)
    raise ValueError('Invalid network_size.')


if __name__ == "__main__":
    train_behavior_cloning()

