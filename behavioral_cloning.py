import click
import torch
import os
import yaml

import numpy as np

from torch import nn
from torch.nn import functional as F
from torch import distributions
from dataset import Dataset_Counts
from tensorboardX import SummaryWriter

from baseline import Baseline
from environments import environment
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
def train_behavior_cloning(training_steps, testing_steps,
                           mini_batch_size, learning_rate,  number_of_epochs, network_size,
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
    estimated_baseline_policy = EstimatedBaseline(network_size=network_size, network_path=None, state_shape=state_shape,
                                         nb_actions=nb_actions, device=device, seed=seed, temperature=0)

    # define loss and optimizer
    nll_loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(estimated_baseline_policy.network.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate, alpha=0.95, eps=1e-07)

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
        baseline = Baseline(params['network_size'], network_path=baseline_network_path,
                            state_shape=params['state_shape'],
                            nb_actions=params['nb_actions'], device=device, seed=seed,
                            temperature=params.get("baseline_temp", 0.1), normalize=params['normalize'])
    else:
        baseline = None

    smaller_testing_loss = float('inf')

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
            estimated_policy = estimated_baseline_policy.policy(batch_states)

            # computing losses
            # negative loglikelihood
            nll_loss = nll_loss_function(torch.log(estimated_policy), target)

            # policy entropy
            estimated_policy_entropy = torch.mean(distributions.Categorical(estimated_policy).entropy())
            # regularize entropy
            entropy_bonus = entropy_coefficient * estimated_policy_entropy

            total_loss = nll_loss - entropy_bonus

            if step % log_frequency == 0:
                total_steps = current_epoch * training_steps + step
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
                    performance = estimated_baseline_policy.evaluate_baseline(env, number_of_steps=20,
                                                                              number_of_epochs=100,
                                                                              verbose=False)

                # run test on test_dataset
                testing_loss = test(data_test, total_steps)


                # logging stats
                s = 'step {:7d}, training: '
                s += 'nll{:7.3f}, '
                s += 'entropy_bonus {:7.3f}, '
                s += 'total_loss {:7.3f} '
                s += 'nll_loss_minus_pi_b_entropy {:7.3f} '
                s += 'testing accuracy: {:7.3f} '
                s += 'estimated policy performance {:7.3f} '
                nll_loss_minus_pi_b_entropy = total_loss.item() - behavior_policy_entropy.item()
                print(s.format(total_steps,
                               nll_loss.item(),
                               entropy_bonus.item(),
                               total_loss.item(),
                               nll_loss_minus_pi_b_entropy,
                               testing_loss,
                               performance))

                logger.add_scalar("training/nll_loss", nll_loss.item(), total_steps)
                logger.add_scalar("training/entropy_bonus", entropy_bonus.item(), total_steps)
                logger.add_scalar("training/total_loss", total_loss.item(), total_steps)
                logger.add_scalar("training/nll_loss_minus_entropy", nll_loss_minus_pi_b_entropy, total_steps)

                logger.add_scalar("estimated_policy/performance", performance, total_steps)
                logger.add_scalar("estimated_policy/entropy", estimated_policy_entropy.item(), total_steps)
                logger.add_scalar("estimated_policy/mse_a", mse_loss.item(), total_steps)
                logger.add_scalar("estimated_policy/mse_pi_b", mse_loss_true_policy.item(), total_steps)
                logger.add_scalar("testing/neg_log_likelihood", testing_loss, total_steps)

            # update weights
            total_loss.backward()
            optimizer.step()

    def test(data, total_steps):
        total_loss = 0
        for step in range(testing_steps):

            # sample mini_batch
            s, a, pi, r, s2, t, c, pi2, cl = data.sample(mini_batch_size=mini_batch_size, full_batch=True)
            batch_states = torch.FloatTensor(s).to(device)
            batch_states = torch.squeeze(batch_states)

            target = torch.LongTensor(a).to(device)  # NLLLoss gets the indexes of the correct class as input

            with torch.no_grad():
                # get predictions
                estimated_policy = estimated_baseline_policy.policy(batch_states)
                # compute loss
                loss = nll_loss_function(torch.log(estimated_policy), target)
            total_loss += loss.item()
        average_loss = total_loss/testing_steps
        if average_loss < smaller_testing_loss:
            estimated_baseline_policy.dump_network(network_path)
        return average_loss

    for epoch in range(number_of_epochs):
        print("\nPROGRESS: {0:02.2f}%\n".format(epoch/number_of_epochs*100), flush=True)
        train(dataset_train, dataset_test, epoch)

        flush(logger)
        update_lr(optimizer, epoch, start_learning_rate=learning_rate)
    estimated_baseline_policy.evaluate_baseline(env, number_of_steps=100, number_of_epochs=1000)


class EstimatedBaseline(Baseline):
    def inference(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        policy = self.policy(state_tensor)
        choice = distributions.Categorical(policy).sample()
        return choice.item(), None, policy, None

    def policy(self, state):
        x = self.network.forward(state)
        return torch.softmax(x, dim=1)


def update_lr(optimizer, epoch, start_learning_rate):
    new_learning_rate = start_learning_rate / (epoch + 2)
    for g in optimizer.param_groups:
        g['lr'] = new_learning_rate


if __name__ == "__main__":
    train_behavior_cloning()

