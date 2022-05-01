import numpy as np
class EpsGreedyPolicy(object):
    def __init__(self, V, action_nums, eps=0.01):
        self.V = V
        self.action_nums = action_nums
        self.eps = eps
    def action(self,state):
        if np.random.rand() < self.eps:
            return np.random.randint(self.action_nums)
        else:
            Q = self.V.eval_actions(state).detach()
            return np.argmax(Q)
