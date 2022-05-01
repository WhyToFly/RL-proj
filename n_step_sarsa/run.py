import numpy as np
import gym
from gym_puyopuyo import register
from n_step import semi_gradient_n_step_td
from policy import EpsGreedyPolicy
from nn import ValueFunctionWithNN

import torch.utils.tensorboard as tb
from os import path
from datetime import datetime

import argparse

register()

import torch

def test_nn(n, consider_future, small=True):
    if small:
        env = gym.make("PuyoPuyoEndlessSmall-v2")
        run_name = "n-" + str(n) + "_consider_future-" + str(consider_future) + "_small"
    else:
        env = gym.make("PuyoPuyoEndlessWide-v2")
        run_name = "n-" + str(n) + "_consider_future-" + str(consider_future) + "_wide"

    plot = run_name + ".png"

    """
    from https://github.com/frostburn/gym_puyopuyo:
    The pieces to be played are encoded as a numpy arrays with (n_colors, 3, 2) shape.
    The playing field is encoded as a numpy array with (n_colors, heights, width) shape.
    An observations is a tuple of (pieces, field).
    """
    print(env.observation_space, env.action_space.n)

    logger = tb.SummaryWriter(path.join(".logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S)") + "_" + run_name), flush_secs=1)

    V = ValueFunctionWithNN(env.action_space.n, consider_future, alpha=1e-3)

    policy = EpsGreedyPolicy(V=V, action_nums=env.action_space.n, eps=0.01)

    semi_gradient_n_step_td(env, 0.95, policy, n, V, 100000, 100, consider_future, logger, plot)

    # Vs = [V(s) for s in testing_states]
    # print(Vs)
    # assert np.allclose(Vs,correct_values,0.20,5.), f'{correct_values} != {Vs}, but it might due to stochasticity'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--n', type=int, required=True)
    parser.add_argument('-c', '--consider_future', required=True, choices=['True', 'False'])
    parser.add_argument('-s', '--size', default='small', choices=['small', 'wide'])

    args = parser.parse_args()

    consider_future = args.consider_future == "True"

    small = True
    if args.size == 'wide':
        small = False

    test_nn(args.n, consider_future, small)
