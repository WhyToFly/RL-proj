import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class ConvNet(nn.Module):
    class Block(nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=1):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            #self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            #return F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x))))) + self.skip(x))
            return F.relu(self.c2(F.relu(self.c1(x))) + self.skip(x))

    def __init__(self, layers, input_channels, action_nums, output_nums, kernel_size=3):
        super().__init__()

        L = []
        c = input_channels
        for l in layers:
            L.append(self.Block(c, l, kernel_size))
            c = l

        # apply upconvolution to widen width to at least number of actions
        # ConvTranspose output: Wout​=(Win​−1)×stride[1]−2×padding[1]+dilation[1]×(kernel_size[1]−1)+output_padding[1]+1
        # goal: 4 * width - 2
        L.append(nn.ConvTranspose2d(c, 1, kernel_size=[1,4], stride=[1,4], padding=[0,1]))

        self.network = torch.nn.Sequential(*L)

        #self.classifier = torch.nn.Linear(c, action_nums)
        self.classifier = torch.nn.Linear(action_nums, output_nums)

    def forward(self, x):
        #z = self.network(x)
        z = F.relu(self.network(x))

        #z = self.classifier(z.mean(dim=[2, 3]))
        z = self.classifier(z.mean(dim=[1, 2]))

        return z

# straight up compied from spinup.algos.pytorch.ppo.core to not depend on the library in this file
# since there are version conflicts...
# Rendering the environment does not work in a conda env with spinup since spinup depends on TensorFlow 1.x which is only available in Python <= 3.6
# however there is a bug that happens when trying to render on newer versions of Mac OS which was only fixed in Python > 3.8...
class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

# CNN Critic; adapted from MLPCritic (spinup.algos.pytorch.ppo.core)
class CNNCritic(nn.Module):
    def __init__(self, layers, input_channels, action_nums):
        super().__init__()
        self.v_net = ConvNet(layers, input_channels, action_nums, 1)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

# CNN Actor; adapted from MLPCategoricalActor (spinup.algos.pytorch.ppo.core)
class CNNCategoricalActor(Actor):
    def __init__(self, layers, input_channels, action_nums):
        super().__init__()
        self.logits_net = ConvNet(layers, input_channels, action_nums, action_nums)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

# CNN Actor Critic; adapted from MLPActorCritic (spinup.algos.pytorch.ppo.core)
class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, layers):
        super().__init__()

        input_channels = observation_space.shape[0]

        # build policy
        self.pi = CNNCategoricalActor(layers, input_channels, action_space.n)

        # build value function
        self.v  = CNNCritic(layers, input_channels, action_space.n)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.item(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


# below also adapted from core file to avoid dependency on the spinup import... (reasons in line 50)
# also to adapt to our other network formats

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], nn.ReLU)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], nn.ReLU)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, layers):
        super().__init__()

        obs_dim = observation_space.shape[0]

        self.pi = MLPCategoricalActor(obs_dim, action_space.n, layers)

        # build value function
        self.v  = MLPCritic(obs_dim, layers)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
