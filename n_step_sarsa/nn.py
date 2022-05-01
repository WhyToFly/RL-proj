import numpy as np
from n_step import ValueFunctionWithApproximation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=1):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.c3(F.relu(self.c2(F.relu(self.c1(x))))) + self.skip(x))

    def __init__(self, layers=[8, 16, 32], input_channels=3, action_nums=10, kernel_size=3):
        super().__init__()

        L = []
        c = input_channels
        for l in layers:
            L.append(self.Block(c, l, kernel_size))
            c = l
        self.network = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(c, action_nums)

    def forward(self, x):
        z = self.network(x)
        return self.classifier(z.mean(dim=[2, 3]))


class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 action_nums,
                 consider_future,
                 alpha):

        # number of input channels is 3 if not considering future puyos; 7 otherwise
        input_channels = 3
        if consider_future:
            input_channels = 7

        self.model = ConvNet(layers=[4, 8], input_channels=input_channels, action_nums=action_nums, kernel_size=3)

        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

        TODO: add scheduler?

    def eval_actions(self,s):
        self.model.eval()
        s = torch.Tensor(s).unsqueeze(0)
        return self.model(s)[0]

    def __call__(self,s,a):
        self.model.eval()
        s = torch.Tensor(s).unsqueeze(0)
        # print(s)
        return self.model(s)[0][a].item()

    def update(self,G,s_tau,a_tau):
        self.model.train()
        self.optimizer.zero_grad()
        s_tau = torch.Tensor(s_tau).unsqueeze(0)
        pred = self.model(s_tau)
        loss = 1/2 * (pred[0][a_tau] - G) * (pred[0][a_tau] - G)
        loss.backward()
        self.optimizer.step()
        return None
