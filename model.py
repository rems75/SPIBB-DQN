import torch
import torch.nn as nn

floatX = 'float32'


def init_weights(m):
    """
    initializes the weights of the given module using a uniform distribution
    sets all the bias parameters to 0
    """
    if type(m) in [nn.Linear, nn.Conv2d]:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.)


class Network(nn.Module):
    def __init__(self, device='cpu'):
        super(Network, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=1),
            nn.ReLU(),
        )
        self.features.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )
        self.fc.apply(init_weights)
        super(Network, self).to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _feature_size(self):
        return self.features(torch.zeros(1, 1, 24, 24)).view(-1).size(0)


class DenseNetwork(nn.Module):
    def __init__(self, state_shape, nb_actions, device='cpu'):
        super(DenseNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, nb_actions)
        )
        self.fc.apply(init_weights)
        super(DenseNetwork, self).to(device)

    def forward(self, x):
        x = self.fc(x)
        return x


class SmallDenseNetwork(nn.Module):
    def __init__(self, state_shape, nb_actions, device='cpu'):
        super(SmallDenseNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(state_shape, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, nb_actions)
        )
        self.fc.apply(init_weights)
        super(SmallDenseNetwork, self).to(device)

    def forward(self, x):
        x = self.fc(x)
        return x


class LargeNetwork(nn.Module):
    def __init__(self, state_shape=[84, 84], nb_channels=4, nb_actions=None, device='cpu'):
        super(LargeNetwork, self).__init__()

        self.state_shape = state_shape
        self.nb_channels = nb_channels
        self.nb_actions = nb_actions

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_channels, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.features.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.nb_actions)
        )
        self.fc.apply(init_weights)
        super(LargeNetwork, self).to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _feature_size(self):
        return self.features(torch.zeros(1, 4, 84, 84)).view(-1).size(0)


class NatureNetwork(nn.Module):
    def __init__(self, state_shape=[84, 84], nb_channels=4, nb_actions=None, device='cpu'):
        super(NatureNetwork, self).__init__()

        self.state_shape = state_shape
        self.nb_channels = nb_channels
        self.nb_actions = nb_actions

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.nb_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.features.apply(init_weights)

        self.fc = nn.Sequential(
            nn.Linear(self._feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.nb_actions)
        )
        self.fc.apply(init_weights)
        super(NatureNetwork, self).to(device)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _feature_size(self):
        return self.features(torch.zeros(1, 4, 84, 84)).view(-1).size(0)


def build_network(state_shape, nb_actions, device, network_size):
    if network_size == 'small':
        return Network()
    elif network_size == 'large':
        return LargeNetwork(state_shape=state_shape, nb_channels=4, nb_actions=nb_actions, device=device)
    elif network_size == 'nature':
        return NatureNetwork(state_shape=state_shape, nb_channels=4, nb_actions=nb_actions, device=device)
    elif network_size == 'dense':
        return DenseNetwork(state_shape=state_shape[0], nb_actions=nb_actions, device=device)
    elif network_size == 'small_dense':
        return SmallDenseNetwork(state_shape=state_shape[0], nb_actions=nb_actions, device=device)
    else:
        raise ValueError('Invalid network_size.')