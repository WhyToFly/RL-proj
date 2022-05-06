import argparse
from os import path
from datetime import datetime

import numpy as np
import torch

from spinup import ppo_pytorch as ppo
from spinup.utils.run_utils import ExperimentGrid
import gym
from gym_puyopuyo import register

from nn import CNNActorCritic, MLPActorCritic
from process_state import ProcessStateCNN, ProcessStateMLP

def test_nn(consider_future, small, cnn, gamma):
    register()

    if cnn:
        run_name = "ppo_cnn_consider_future_" + str(consider_future)
    else:
        run_name = "ppo_mlp"


    if small:
        env = gym.make("PuyoPuyoEndlessSmall-v2")
        run_name += "_small"
    else:
        env = gym.make("PuyoPuyoEndlessWide-v2")
        run_name += "_wide"

    run_name += "_gamma-" + str(gamma)

    logger_kwargs = dict(output_dir=path.join(".logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S)") + "_" + run_name), exp_name=run_name)

    if cnn:
        # wrap environment to get observations usable in nn
        # TODO: try linear processing (concatenate observation using ProcessStateMLP), network
        env = ProcessStateCNN(env, consider_future)

        env_fn = lambda : env

        ac_kwargs = dict(layers=[8,16,16])
        ppo(env_fn=env_fn,actor_critic=CNNActorCritic, ac_kwargs=ac_kwargs, steps_per_epoch=4000, epochs=200, max_ep_len=250, gamma=gamma, logger_kwargs=logger_kwargs)
    else:
        raise NotImplementedError

def create_env_wide_cnn():
    from gym_puyopuyo import register

    register()

    env = gym.make("PuyoPuyoEndlessWide-v2")
    env = ProcessStateCNN(env, False)
    return env

def create_env_wide_future_cnn():
    from gym_puyopuyo import register

    register()

    env = gym.make("PuyoPuyoEndlessWide-v2")
    env = ProcessStateCNN(env, True)
    return env

def create_env_small_cnn():
    from gym_puyopuyo import register

    register()

    env = gym.make("PuyoPuyoEndlessSmall-v2")
    env = ProcessStateCNN(env, False)
    return env

def create_env_small_future_cnn():
    from gym_puyopuyo import register

    register()

    env = gym.make("PuyoPuyoEndlessSmall-v2")
    env = ProcessStateCNN(env, True)
    return env

def create_env_wide_mlp():
    from gym_puyopuyo import register

    register()

    env = gym.make("PuyoPuyoEndlessWide-v2")
    env = ProcessStateMLP(env)
    return env

def create_env_small_mlp():
    from gym_puyopuyo import register

    register()

    env = gym.make("PuyoPuyoEndlessSmall-v2")
    env = ProcessStateMLP(env)
    return env

if __name__ == "__main__":
    '''
    replaced cmd arguments with gridsearch for wide env after proving that small env can be solved by ppo
    only works when adding

    import sys
    sys.path.append("../ppo/")

    to spinningup/spinup/utils/run_entrypoint.py
    Comment this/uncomment below for command line argument/single experiment version
    '''

    eg = ExperimentGrid(name='ppo-pyt-bench')
    eg.add('env_fn', [create_env_small_mlp, create_env_wide_mlp], 'env_fn')
    eg.add('actor_critic', MLPActorCritic)
    eg.add('gamma', [0.9,0.95,0.98,0.99,0.999], 'gamma')
    eg.add('epochs', 150)
    eg.add('steps_per_epoch', 4000)
    eg.add('max_ep_len', 100)
    eg.add('ac_kwargs:layers', [[], [16], [64], [32, 16], [128, 64]], 'layers')
    eg.run(ppo, num_cpu=6)


    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--consider_future', default='False', choices=['True', 'False'])
    parser.add_argument('-n', '--network', default='cnn', choices=['cnn', 'mlp'])
    parser.add_argument('-s', '--size', default='small', choices=['small', 'wide'])
    parser.add_argument('-g', '--gamma', type=float, default=0.99)

    args = parser.parse_args()

    consider_future = args.consider_future == "True"
    small = args.size == "small"
    cnn = args.network == "cnn"

    test_nn(consider_future, small, cnn, args.gamma)
    '''
