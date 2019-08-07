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
@click.option('--estimated_network_path',
              default='./cloned_network_weights.pt')
@click.option('--folder_location',
              default='./baseline/helicopter_env/')
@click.option('--dataset_file',
              default='dataset/10000/123/1_0/counts_dataset.pkl')
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
@click.option('--update_learning_rate', is_flag=True)
def train_behavior_cloning(**kwargs):
    x = BehaviorCloning(**kwargs)
    x.run_training()


class BehaviorCloning:
    def __init__(self, training_steps, testing_steps,
                 mini_batch_size, learning_rate, number_of_epochs, network_size,
                 folder_location, dataset_file, estimated_network_path, sample_from_env,
                 entropy_coefficient,
                 device, seed, experiment_name, config_file, update_learning_rate):
        self.training_steps = training_steps
        self.testing_steps = testing_steps
        self.mini_batch_size = mini_batch_size
        self.number_of_epochs = number_of_epochs
        self.network_size = network_size
        self.entropy_coefficient = entropy_coefficient
        self.device = device
        self.sample_from_env = sample_from_env
        self.smaller_testing_loss = None
        self.seed = seed
        self.learning_rate = learning_rate
        self.update_learning_rate = update_learning_rate
        self.log_frequency = 200
        try:
            self.params = yaml.safe_load(open(config_file, 'r'))
        except FileNotFoundError as e:
            print("Configuration file not found; Define a config_file to be able to sample from environment")
            raise e

        # initialize seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)

        # set paths for data and output path
        log_path = os.path.join(os.getenv("PT_OUTPUT_DIR", './'), 'logs/' + experiment_name)
        data_dir = os.getenv("PT_DATA_DIR", os.path.join(folder_location))
        dataset_path = os.path.join(data_dir, dataset_file)
        self.estimated_network_path = os.path.join(os.path.dirname(dataset_path), estimated_network_path)

        # start
        self.logger = SummaryWriter(log_path)

        # import data
        full_dataset = Dataset_Counts.load_dataset(dataset_path)
        self.dataset_train, self.dataset_test = full_dataset.train_test_split(test_size=0.2)

        # create model
        self.estimated_baseline_policy = EstimatedBaseline(network_size=network_size, network_path=None,
                                                           state_shape=self.params['state_shape'],
                                                           nb_actions=self.params['nb_actions'], device=device,
                                                           seed=seed, temperature=0)

        # define loss and optimizer
        self.nll_loss_function = nn.NLLLoss()
        self.optimizer = torch.optim.SGD(self.estimated_baseline_policy.network.parameters(), lr=learning_rate)
        # optimizer = torch.optim.RMSprop(network.parameters(), lr=learning_rate, alpha=0.95, eps=1e-07)

        # instantiate environment for policy evaluation
        self.env = environment.Environment(self.params['domain'], self.params)

        if sample_from_env:
            print("sampling from environment")
            baseline_network_path = os.path.join(data_dir, self.params["network_path"])
            self.baseline = Baseline(self.params['network_size'], network_path=baseline_network_path,
                                     state_shape=self.params['state_shape'],
                                     nb_actions=self.params['nb_actions'], device=device, seed=seed,
                                     temperature=self.params.get("baseline_temp", 0.1),
                                     normalize=self.params['normalize'])
        else:
            self.baseline = None

    def train(self, current_epoch=0):
        for step in range(self.training_steps):
            # clear gradients
            self.optimizer.zero_grad()

            # sample mini_batch
            if not self.sample_from_env:
                s, a, behavior_policy, _, _, _, _, _, _ = self.dataset_train.sample(
                    mini_batch_size=self.mini_batch_size, full_batch=True)
            else:
                # sanity check: train on new samples instead of fixed dataset
                mini_batch = Dataset_Counts(state_shape=self.params['state_shape'],
                                            nb_actions=self.params['nb_actions'],
                                            count_param=0.2)
                while mini_batch.size < self.mini_batch_size:
                    state = self.env.reset()
                    action, _, policy, _ = self.baseline.inference(state)
                    _, new_reward, term, _ = self.env.step(action)
                    mini_batch.add(s=state.astype('float32'), a=action, r=new_reward, t=term, p=policy)
                s, a, behavior_policy, _, _, _, _, _, _ = mini_batch.get_all_data()

            # prepare tensors
            batch_states = torch.FloatTensor(s).to(self.device)
            batch_states = torch.squeeze(batch_states)
            target = torch.LongTensor(a).to(self.device)  # NLLLoss gets the indexes of the correct class as input

            # get predictions
            estimated_policy = self.estimated_baseline_policy.policy(batch_states)

            # computing losses
            # negative loglikelihood
            nll_loss = self.nll_loss_function(torch.log(estimated_policy), target)

            # policy entropy
            estimated_policy_entropy = torch.mean(distributions.Categorical(estimated_policy).entropy())
            # regularize entropy
            entropy_bonus = self.entropy_coefficient * estimated_policy_entropy

            total_loss = nll_loss - entropy_bonus

            if step % self.log_frequency == 0:
                total_steps = current_epoch * self.training_steps + step
                self.test_and_log_stats(a, behavior_policy, entropy_bonus, estimated_policy,
                                        estimated_policy_entropy, nll_loss, total_loss, total_steps)

            # update weights
            total_loss.backward()
            self.optimizer.step()

    def test_and_log_stats(self, a, behavior_policy, entropy_bonus, estimated_policy,
                           estimated_policy_entropy, nll_loss, total_loss, total_steps):

        # run test on test_dataset
        testing_loss = self.test()

        # compute stats
        with torch.no_grad():
            # makes an one-hot vector for the action
            one_hot_behavior_policy = np.zeros(list(a.shape) + [self.params['nb_actions']])
            one_hot_behavior_policy[np.arange(len(a)), a] = 1
            one_hot_behavior_policy = torch.FloatTensor(one_hot_behavior_policy).to(self.device)

            # compute MSE loss of estimated probability with one_hot policy
            mse_loss = F.mse_loss(estimated_policy, one_hot_behavior_policy)

            # compute MSE with true policy
            behavior_policy = torch.FloatTensor(behavior_policy).to(self.device)
            mse_loss_true_policy = F.mse_loss(estimated_policy, behavior_policy)

            # compute entropy of the behavior policy
            behavior_policy_entropy = torch.mean(distributions.Categorical(behavior_policy).entropy())
            mean_perfomance, _, _ = self.estimated_baseline_policy.evaluate_baseline(self.env,
                                                                                     number_of_steps=300,
                                                                                     number_of_epochs=20,
                                                                                     verbose=False)

        self.log_stats(behavior_policy_entropy, entropy_bonus, estimated_policy_entropy, mean_perfomance, mse_loss,
                       mse_loss_true_policy, nll_loss, testing_loss, total_loss, total_steps)

    def log_stats(self, behavior_policy_entropy, entropy_bonus, estimated_policy_entropy, mean_perfomance, mse_loss,
                  mse_loss_true_policy, nll_loss, testing_loss, total_loss, total_steps):
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
                       mean_perfomance))
        self.logger.add_scalar("training/nll_loss", nll_loss.item(), total_steps)
        self.logger.add_scalar("training/entropy_bonus", entropy_bonus.item(), total_steps)
        self.logger.add_scalar("training/total_loss", total_loss.item(), total_steps)
        self.logger.add_scalar("training/nll_loss_minus_entropy", nll_loss_minus_pi_b_entropy, total_steps)
        self.logger.add_scalar("estimated_policy/performance", mean_perfomance, total_steps)
        self.logger.add_scalar("estimated_policy/entropy", estimated_policy_entropy.item(), total_steps)
        self.logger.add_scalar("estimated_policy/mse_a", mse_loss.item(), total_steps)
        self.logger.add_scalar("estimated_policy/mse_pi_b", mse_loss_true_policy.item(), total_steps)
        self.logger.add_scalar("testing/neg_log_likelihood", testing_loss, total_steps)

    def test(self):
        losses = []
        for step in range(self.testing_steps):
            # sample mini_batch
            s, a, pi, r, s2, t, c, pi2, cl = self.dataset_test.sample(mini_batch_size=self.mini_batch_size,
                                                                      full_batch=True)
            batch_states = torch.FloatTensor(s).to(self.device)
            batch_states = torch.squeeze(batch_states)

            target = torch.LongTensor(a).to(self.device)  # NLLLoss gets the indexes of the correct class as input

            with torch.no_grad():
                # get predictions
                estimated_policy = self.estimated_baseline_policy.policy(batch_states)
                # compute loss
                loss = self.nll_loss_function(torch.log(estimated_policy), target)
            losses.append(loss.item())
        average_loss = np.mean(losses)
        if average_loss < self.smaller_testing_loss:
            self.estimated_baseline_policy.dump_network(self.estimated_network_path)
            self.smaller_testing_loss = average_loss
            print("\n>>> best testing accuracy so far: {:7.3f}\n>>> new policy saved to {}\n".format(average_loss,
                                                                                         self.estimated_network_path))
        return average_loss

    def run_training(self):
        self.smaller_testing_loss = float('inf')
        for epoch in range(self.number_of_epochs):
            print("\nPROGRESS: {0:02.2f}%\n".format(epoch / self.number_of_epochs * 100), flush=True)
            self.train(epoch)

            flush(self.logger)
            if self.update_learning_rate:
                self.update_lr(epoch)

        print("training complete")
        mean, decile, centile = self.estimated_baseline_policy.evaluate_baseline(self.env, number_of_steps=10000,
                                                                                 number_of_epochs=300, verbose=False)
        print("current policy performance, mean:{}, centile {}, centile {}".format(mean, decile, centile))

        self.evaluate_learned_policy()

    def evaluate_learned_policy(self):
        print("evaluating policy cloned")
        # load best policy, evaluate and save data
        final_policy = EstimatedBaseline(network_size=self.network_size,
                                         network_path=self.estimated_network_path,
                                         state_shape=self.params['state_shape'], nb_actions=self.params['nb_actions'],
                                         device=self.device, seed=self.seed, temperature=0)
        mean, decile, centile = final_policy.evaluate_baseline(self.env, number_of_steps=10000, number_of_epochs=300,
                                                               verbose=False)
        with open(self.estimated_network_path.replace('.pt', '_performance.csv'), 'w') as f:
            f.write('mean, decile, centile\n')
            f.write(','.join([str(mean), str(decile), str(centile)]))
            f.write('\n')
        print("saved policy performance, mean:{}, centile {}, centile {}".format(mean, decile, centile))

    def update_lr(self, epoch):
        new_learning_rate = self.learning_rate / (epoch + 2)
        print(">>>> new learning rate: {}".format(new_learning_rate))
        for g in self.optimizer.param_groups:
            g['lr'] = new_learning_rate


class EstimatedBaseline(Baseline):
    def inference(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        policy = self.policy(state_tensor)
        choice = distributions.Categorical(policy).sample()
        choice = choice.item()
        return choice, None, policy, None

    def policy(self, state):
        x = self.network.forward(state)
        return torch.softmax(x, dim=1)


if __name__ == "__main__":
    train_behavior_cloning()
