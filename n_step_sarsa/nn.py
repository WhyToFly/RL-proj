import numpy as np
from n_step import ValueFunctionWithApproximation

import torch
import torch.nn as nn
import torch.optim as optim



class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims, 
                 action_nums,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_nums: num of actions
        """
        # TODO: implement this method
        self.model = nn.Linear(state_dims, action_nums)
        # self.model = nn.Sequential(
        #     nn.Linear(state_dims, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, action_nums))
        # self.model = nn.Sequential(
        #     nn.Linear(state_dims, 32),
        #     nn.ReLU(),
        #     # nn.Linear(32, 32),
        #     # nn.ReLU(),
        #     nn.Linear(32, action_nums))
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, betas=(0.9, 0.999))

    def __call__(self,s,a):
        # TODO: implement this method
        s = np.concatenate((s[0].reshape(-1), s[1].reshape(-1)))
        self.model.eval()
        s = torch.Tensor(s)
        # print(s)
        return float(self.model(s)[a])

    def update(self,G,s_tau,a_tau):
        # TODO: implement this method
        self.model.train()
        self.optimizer.zero_grad()
        # s_tau = torch.Tensor(s_tau)
        s_tau = torch.Tensor(np.concatenate((s_tau[0].reshape(-1), s_tau[1].reshape(-1))))
        loss = 1/2 * (self.model(s_tau)[a_tau] - G) * (self.model(s_tau)[a_tau] - G)
        loss.backward()
        self.optimizer.step()
        return None

