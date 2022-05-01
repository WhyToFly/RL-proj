import numpy as np
import gym
from gym_puyopuyo import register
from n_step import semi_gradient_n_step_td
from policy import EpsGreedyPolicy
from nn import ValueFunctionWithNN

import torch.utils.tensorboard as tb
from os import path
from datetime import datetime

register()

import torch

def test_nn(n, plot):
    env = gym.make("PuyoPuyoEndlessSmall-v2")

    """
    from https://github.com/frostburn/gym_puyopuyo:
    The pieces to be played are encoded as a numpy arrays with (n_colors, 3, 2) shape.
    The playing field is encoded as a numpy array with (n_colors, heights, width) shape.
    An observations is a tuple of (pieces, field).
    """
    print(env.observation_space, env.action_space.n)

    # V = ValueFunctionWithNN(env.observation_space.shape[0], env.action_space.n)
    logger = tb.SummaryWriter(path.join(".logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S)")), flush_secs=1)

    V = ValueFunctionWithNN(env.action_space.n, alpha=0.001, consider_future=False)

    policy = EpsGreedyPolicy(V=V, action_nums=env.action_space.n, eps=0.01)

    semi_gradient_n_step_td(env, 0.95, policy, n, V, 10000, 50, logger, plot)

    # Vs = [V(s) for s in testing_states]
    # print(Vs)
    # assert np.allclose(Vs,correct_values,0.20,5.), f'{correct_values} != {Vs}, but it might due to stochasticity'

if __name__ == "__main__":
    test_nn(1, "n=1.png")
    # test_nn(2, "n=2.png")
    # test_nn(4, "n=4.png")
    # test_nn(6, "n=6.png")
    # test_nn(8, "n=8.png")
